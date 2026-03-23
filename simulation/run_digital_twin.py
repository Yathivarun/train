"""
run_digital_twin.py  —  Fixed simulation runner
Key fixes:
  1. DQN trigger lowered from ~50% to 30% congestion
  2. Train type classification uses route ID not vehicle name (fixes priority bug)
  3. Stress-test mode available via --stress flag
  4. File lock for HITL JSON communication
  5. Per-vehicle type properly read from SUMO route assignment
"""

import os, sys, random, json, threading, argparse, csv, statistics, uuid
import pandas as pd
import torch
import torch.nn as nn
import traci

parser = argparse.ArgumentParser()
parser.add_argument("--routing",     default="dqn_hitl",
                    choices=["hardcoded", "dqn_simple", "dqn_full", "dqn_hitl"])
parser.add_argument("--stress",      action="store_true",
                    help="Stress-test: flood UP_FAST/DOWN_FAST to trigger DQN")
args = parser.parse_args()

SUMO_BINARY  = "sumo-gui"
SUMO_CONFIG  = "sumo.sumocfg"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
LIVE_JSON    = os.path.join(OUTPUT_DIR, "live_trains.json")
RESULT_CSV   = os.path.join(OUTPUT_DIR, "simulation_results.csv")
PENDING_FILE = os.path.join(OUTPUT_DIR, "pending_actions.json")
APPROVED_FILE= os.path.join(OUTPUT_DIR, "approved_actions.json")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models",
                             "saved_models", "dqn_railway_model.pth")
LIVE_FILE    = os.path.join(OUTPUT_DIR, "live_status.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

_lock = threading.Lock()

def _read_json(path, default):
    with _lock:
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return default

def _write_json(path, data):
    with _lock:
        with open(path, "w") as f:
            json.dump(data, f)

_write_json(PENDING_FILE,  [])
_write_json(APPROVED_FILE, [])

MAX_SIM_TIME    = 3500
SPAWN_INTERVAL  = 3
SAFE_HEADWAY    = 40

# ── Fixed: DQN trigger at 30% not 50% ───────────────────────────────────────
DQN_TRIGGER     = 0.30

PLATFORM_OVERTAKE_EDGES  = ["521712768", "385081047"]
CRITICAL_JUNCTION_EDGES  = ["44629483", "44629484", "44629491"]

# ── Train type → route mapping (source of truth: ROUTE ID) ──────────────────
ROUTE_TYPE_MAP = {
    "UP_FAST":    "SUPERFAST",
    "DOWN_FAST":  "EXPRESS",
    "UP_SLOW":    "PASSENGER",
    "DOWN_SLOW":  "FREIGHT",
}

PRIORITY   = {"SUPERFAST": 5, "EXPRESS": 4, "PASSENGER": 3, "EMU": 2, "FREIGHT": 1}
BASE_SPEED = {"SUPERFAST": 44, "EXPRESS": 36, "PASSENGER": 30, "EMU": 28, "FREIGHT": 22}

spawn_queue    = []
junction_locks = {}
metrics_log    = []


# ── DQN model ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)   # Fixed: 4 inputs matching dqn_env.py
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 2)
        self.relu= nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


dqn_model = None
if args.routing in ["dqn_simple", "dqn_full", "dqn_hitl"]:
    if os.path.exists(MODEL_PATH):
        try:
            dqn_model = QNetwork()
            dqn_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
            dqn_model.eval()
            print(f"✅ DQN loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ DQN load failed: {e} — falling back to hardcoded")
            args.routing = "hardcoded"
    else:
        print(f"⚠️ DQN model not found at {MODEL_PATH}")
        print("   Run: python backend/dqn_env.py  to train first.")
        args.routing = "hardcoded"


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_from_id(train_id: str) -> str:
    """Classify train type from numeric ID prefix (fallback only)."""
    try:
        num = int(str(train_id).split("_")[0])
        if 37000 <= num < 38000: return "EMU"
        if 15000 <= num < 16000: return "PASSENGER"
        if 13000 <= num < 14000: return "EXPRESS"
        if 22000 <= num < 23000: return "SUPERFAST"
        if 63000 <= num < 64000: return "FREIGHT"
    except Exception:
        pass
    return "EXPRESS"


def get_type(vehicle_id: str) -> str:
    """
    FIX: Classify by ROUTE ID not vehicle name string.
    Route IDs are deterministic; vehicle name strings are not.
    """
    try:
        route_id = traci.vehicle.getRouteID(vehicle_id)
        for prefix, ttype in ROUTE_TYPE_MAP.items():
            if route_id.startswith(prefix):
                return ttype
    except Exception:
        pass
    return classify_from_id(vehicle_id)


def detect_direction(lat: float) -> str:
    return "UP" if lat > 22.8 else "DOWN"


def assign_route(ttype: str, direction: str) -> str:
    if args.stress:
        # Stress mode: force 80% of trains to FAST tracks to trigger DQN
        if random.random() < 0.8:
            return f"{direction}_FAST"
    if direction == "UP":
        return "UP_SLOW"   if ttype in ["EMU", "PASSENGER", "FREIGHT"] else "UP_FAST"
    return "DOWN_SLOW" if ttype in ["EMU", "PASSENGER", "FREIGHT"] else "DOWN_FAST"


def enforce_headway():
    for v in traci.vehicle.getIDList():
        try:
            base   = BASE_SPEED[get_type(v)]
            leader = traci.vehicle.getLeader(v, SAFE_HEADWAY)
            if leader and leader[1] < SAFE_HEADWAY:
                traci.vehicle.setSpeed(v, max(5, leader[1] * 0.8))
            else:
                traci.vehicle.setSpeed(v, base)
        except Exception:
            pass


def enforce_junctions() -> int:
    conflicts = 0
    vehicles  = traci.vehicle.getIDList()
    for v in vehicles:
        try:
            edge = traci.vehicle.getRoadID(v)
            if edge not in CRITICAL_JUNCTION_EDGES:
                continue
            prio = PRIORITY[get_type(v)]
            if edge not in junction_locks:
                junction_locks[edge] = v
            else:
                locked = junction_locks[edge]
                if locked not in vehicles:
                    junction_locks[edge] = v
                    continue
                if prio > PRIORITY[get_type(locked)]:
                    traci.vehicle.setSpeed(locked, 5)
                    junction_locks[edge] = v
                else:
                    traci.vehicle.setSpeed(v, 5)
                    conflicts += 1
        except Exception:
            pass

    for edge in list(junction_locks):
        if junction_locks[edge] not in vehicles:
            del junction_locks[edge]
    return conflicts


def process_approved() -> int:
    approved = _read_json(APPROVED_FILE, [])
    if not approved:
        return 0
    executed = 0
    for action in approved:
        tid       = action.get("train_id")
        new_route = action.get("new_route")
        if tid in traci.vehicle.getIDList():
            try:
                traci.vehicle.setRouteID(tid, new_route)
                print(f"🚦 HUMAN APPROVED: {tid} → {new_route}")
                executed += 1
            except Exception as e:
                print(f"⚠️ Route set failed for {tid}: {e}")
    _write_json(APPROVED_FILE, [])
    return executed


def predictive_congestion(congestion: float, vehicles: list) -> int:
    switches = 0
    pending  = _read_json(PENDING_FILE, [])
    pending_trains = {p["train_id"] for p in pending}

    # ── FIX: trigger at DQN_TRIGGER (0.30) not 0.50 ─────────────────────────
    if congestion < DQN_TRIGGER:
        return 0

    for v in vehicles:
        try:
            route_id = traci.vehicle.getRouteID(v)
            if "FAST" not in route_id:
                continue

            ttype    = get_type(v)
            priority = PRIORITY[ttype]
            speed    = traci.vehicle.getSpeed(v)

            switch = False

            if args.routing == "hardcoded":
                switch = congestion > 0.40 and ttype in ["EXPRESS", "PASSENGER"]

            elif args.routing in ["dqn_full", "dqn_hitl"] and dqn_model:
                import time
                state = torch.tensor([
                    min(congestion, 1.0),
                    min(len(vehicles) / 50.0, 1.0),
                    min(priority / 5.0, 1.0),
                    min(time.localtime().tm_hour / 23.0, 1.0),
                ], dtype=torch.float32)

                with torch.no_grad():
                    action = int(dqn_model(state.unsqueeze(0)).argmax().item())
                switch = (action == 1)

            if switch:
                new_route = route_id.replace("FAST", "SLOW")
                if args.routing == "dqn_hitl":
                    if v not in pending_trains:
                        alert = {
                            "id":        str(uuid.uuid4())[:8],
                            "train_id":  v,
                            "train_type": ttype,
                            "new_route": new_route,
                            "congestion": round(congestion, 3),
                            "message": (
                                f"DQN [{congestion*100:.0f}% congestion]: "
                                f"Reroute {ttype} train {v} to SLOW track "
                                f"to prevent gridlock."
                            ),
                        }
                        pending.append(alert)
                        _write_json(PENDING_FILE, pending)
                        pending_trains.add(v)
                else:
                    try:
                        edges = traci.route.getEdges(new_route)
                        if traci.vehicle.getRoadID(v) in edges:
                            traci.vehicle.setRouteID(v, new_route)
                            switches += 1
                    except Exception:
                        pass
        except Exception:
            pass

    return switches


# ─── Main loop ────────────────────────────────────────────────────────────────

print(f"🚆 Railway Digital Twin | Mode: {args.routing.upper()} "
      f"{'| STRESS TEST 🔴' if args.stress else ''}")

try:
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--no-warnings", "true"])
except Exception as e:
    print(f"SUMO failed to start: {e}")
    sys.exit(1)

step, last_modified, next_spawn = 0, 0, 0

while step < MAX_SIM_TIME:
    traci.simulationStep()

    # Load live CSV
    if os.path.exists(LIVE_FILE):
        mod = os.path.getmtime(LIVE_FILE)
        if mod != last_modified:
            last_modified = mod
            try:
                df = pd.read_csv(LIVE_FILE)
                spawn_queue.clear()
                for _, row in df.iterrows():
                    raw_id = str(int(float(row["train_id"])))
                    ttype  = classify_from_id(raw_id)
                    lat    = float(row.get("latitude", 22.7))
                    route  = assign_route(ttype, detect_direction(lat))
                    uid    = f"{raw_id}_{step}_{random.randint(100, 999)}"
                    spawn_queue.append((uid, route, ttype))
            except Exception as e:
                print(f"CSV read error: {e}")

    # Spawn
    if spawn_queue and step >= next_spawn:
        tid, route, ttype = spawn_queue.pop(0)
        try:
            first_edge = traci.route.getEdges(route)[0]
            if traci.edge.getLastStepVehicleNumber(first_edge) < 2:
                traci.vehicle.add(tid, route, typeID="LIVE_RAIL")
                traci.vehicle.setMaxSpeed(tid, BASE_SPEED[ttype])
        except Exception:
            pass
        next_spawn = step + SPAWN_INTERVAL

    enforce_headway()
    jconflicts = enforce_junctions()
    vehicles   = list(traci.vehicle.getIDList())
    avg_speed  = statistics.mean([traci.vehicle.getSpeed(v) for v in vehicles]) if vehicles else 0
    congestion = 1 - (avg_speed / 44) if vehicles else 0

    switches   = predictive_congestion(congestion, vehicles)
    switches  += process_approved()

    # Write live positions
    live_out = []
    for v in vehicles:
        try:
            x, y   = traci.vehicle.getPosition(v)
            lon, lat = traci.simulation.convertGeo(x, y)
            live_out.append({"id": v, "lat": lat, "lon": lon,
                             "type": get_type(v), "speed": round(traci.vehicle.getSpeed(v), 2)})
        except Exception:
            pass
    _write_json(LIVE_JSON, live_out)

    metrics_log.append([step, len(vehicles), avg_speed, congestion, 0, jconflicts, switches])

    if step % 50 == 0:
        mode_flag = "🔴 DQN ACTIVE" if congestion >= DQN_TRIGGER else "🟢 Nominal"
        print(f"[{step}] Trains:{len(vehicles)} | Speed:{avg_speed:.1f} | "
              f"Cong:{congestion:.3f} | {mode_flag}")

    if step % 5 == 0:
        try:
            with open(RESULT_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "active_trains", "avg_speed", "congestion_index",
                             "overtake_events", "junction_conflicts", "dynamic_switches"])
                w.writerows(metrics_log)
        except Exception:
            pass

    step += 1

traci.close()
print("✅ Simulation complete.")

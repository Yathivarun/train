import os
import random
import sys
import traci
import pandas as pd
import statistics
import csv
import json
import argparse
import torch
import torch.nn as nn
import uuid

parser = argparse.ArgumentParser(description="Run the Railway Digital Twin")
parser.add_argument("--routing", type=str, default="dqn_hitl", 
                    choices=["hardcoded", "dqn_simple", "dqn_full", "dqn_hitl"], 
                    help="Choose the routing logic")
args = parser.parse_args()

SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "sumo.sumocfg"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
LIVE_FILE = os.path.join(OUTPUT_DIR, "live_status.csv")
LIVE_JSON = os.path.join(OUTPUT_DIR, "live_trains.json")
RESULT_CSV = os.path.join(OUTPUT_DIR, "simulation_results.csv")
PENDING_FILE = os.path.join(OUTPUT_DIR, "pending_actions.json")
APPROVED_FILE = os.path.join(OUTPUT_DIR, "approved_actions.json")

with open(PENDING_FILE, "w") as f: json.dump([], f)
with open(APPROVED_FILE, "w") as f: json.dump([], f)

MODEL_PATH = os.path.join(OUTPUT_DIR, "dqn_routing_model_full.pth")

MAX_SIM_TIME = 3500
SPAWN_INTERVAL = 3  # <--- REDUCED FROM 25 TO 3 TO ALLOW MASSIVE TRAFFIC
SAFE_HEADWAY = 40
OVERTAKE_DISTANCE = 200

PLATFORM_OVERTAKE_EDGES = ["521712768", "385081047"]
CRITICAL_JUNCTION_EDGES = ["44629483", "44629484", "44629491"]

PRIORITY = {"SUPERFAST": 5, "EXPRESS": 4, "PASSENGER": 3, "EMU": 2, "FREIGHT": 1}
BASE_SPEED = {"SUPERFAST": 44, "EXPRESS": 36, "PASSENGER": 30, "EMU": 28, "FREIGHT": 22}

spawn_queue = []
junction_locks = {}
metrics_log = []
overtake_events = []
dynamic_switches = []

class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(3, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

dqn_model = None
if args.routing in ["dqn_simple", "dqn_full", "dqn_hitl"]:
    try:
        dqn_model = DQNAgent()
        dqn_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        dqn_model.eval()
        print(f"✅ Loaded {args.routing} model successfully.")
    except Exception as e:
        print(f"⚠️ Could not load DQN model. Falling back to hardcoded. Error: {e}")
        args.routing = "hardcoded"

def classify(train_id):
    # Split by underscore to handle our new unique IDs (e.g., "15000_12")
    num = int(str(train_id).split('_')[0])
    if 37000 <= num < 38000: return "EMU"
    elif 15000 <= num < 16000: return "PASSENGER"
    elif 13000 <= num < 14000: return "EXPRESS"
    elif 22000 <= num < 23000: return "SUPERFAST"
    elif 63000 <= num < 64000: return "FREIGHT"
    return "EXPRESS"

def detect_direction(lat):
    return "UP" if lat > 22.8 else "DOWN"

def assign_route(train_type, direction):
    if direction == "UP":
        return "UP_SLOW" if train_type in ["EMU","PASSENGER","FREIGHT"] else "UP_FAST"
    else:
        return "DOWN_SLOW" if train_type in ["EMU","PASSENGER","FREIGHT"] else "DOWN_FAST"

def enforce_headway():
    for v in traci.vehicle.getIDList():
        try:
            base = BASE_SPEED[classify(v)]
            leader = traci.vehicle.getLeader(v, SAFE_HEADWAY)
            if leader and leader[1] < SAFE_HEADWAY:
                traci.vehicle.setSpeed(v, max(5, leader[1] * 0.8))
            else:
                traci.vehicle.setSpeed(v, base)
        except: pass

def enforce_junctions():
    conflicts = 0
    vehicles = traci.vehicle.getIDList()
    for v in vehicles:
        try:
            edge = traci.vehicle.getRoadID(v)
            if edge not in CRITICAL_JUNCTION_EDGES: continue
            priority = PRIORITY[classify(v)]
            if edge not in junction_locks: junction_locks[edge] = v
            else:
                locked = junction_locks[edge]
                if locked not in vehicles:
                    junction_locks[edge] = v
                    continue
                if priority > PRIORITY[classify(locked)]:
                    traci.vehicle.setSpeed(locked, 5)
                    junction_locks[edge] = v
                else:
                    traci.vehicle.setSpeed(v, 5)
                    conflicts += 1
        except: pass
    
    for edge in list(junction_locks.keys()):
        if junction_locks[edge] not in vehicles: junction_locks.pop(edge)
    return conflicts

def process_human_approvals():
    try:
        with open(APPROVED_FILE, "r") as f:
            approved = json.load(f)
        if not approved: return 0
        executed = 0
        for action in approved:
            tid = action["train_id"]
            new_route = action["new_route"]
            if tid in traci.vehicle.getIDList():
                traci.vehicle.setRouteID(tid, new_route)
                print(f"🚦 HUMAN OVERRIDE EXECUTED: Switched Train {tid} to {new_route}")
                executed += 1
        with open(APPROVED_FILE, "w") as f: json.dump([], f)
        return executed
    except Exception: return 0

def predictive_congestion():
    vehicles = traci.vehicle.getIDList()
    if not vehicles: return 0

    speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
    avg_speed = statistics.mean(speeds)
    congestion_index = 1 - (avg_speed / 44)
    switches = 0

    try:
        with open(PENDING_FILE, "r") as f: pending = json.load(f)
        pending_trains = [p["train_id"] for p in pending]
    except:
        pending = []
        pending_trains = []

    for v in vehicles:
        try:
            current_edge = traci.vehicle.getRoadID(v)
            route_id = traci.vehicle.getRouteID(v)
            if "FAST" not in route_id: continue

            ttype = classify(v)
            priority = PRIORITY[ttype]
            speed = traci.vehicle.getSpeed(v)
            switch_track = False

            if args.routing == "hardcoded":
                if congestion_index > 0.4 and ttype == "EXPRESS": switch_track = True
            elif args.routing in ["dqn_full", "dqn_hitl"] and dqn_model:
                state = torch.tensor([priority, speed, congestion_index], dtype=torch.float32)
                with torch.no_grad(): action = torch.argmax(dqn_model(state)).item()
                if action == 1: switch_track = True

            if switch_track:
                new_route_id = route_id.replace("FAST", "SLOW")
                if args.routing == "dqn_hitl":
                    if v not in pending_trains:
                        alert = {
                            "id": str(uuid.uuid4())[:8],
                            "train_id": v,
                            "train_type": ttype,
                            "new_route": new_route_id,
                            "message": f"DQN Advisor: Congestion is {round(congestion_index*100)}%. Reroute {ttype} train ({v}) to SLOW track to clear bottleneck."
                        }
                        pending.append(alert)
                        with open(PENDING_FILE, "w") as f: json.dump(pending, f)
                else:
                    new_edges = traci.route.getEdges(new_route_id)
                    if current_edge in new_edges:
                        traci.vehicle.setRouteID(v, new_route_id)
                        switches += 1
        except: pass
    return switches

print(f"🚆 Starting Railway Digital Twin | Routing Mode: {args.routing.upper()}")
try:
    # Added --no-warnings to silence the annoying block length errors
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--no-warnings", "true"])
except Exception as e:
    print(f"SUMO failed to start: {e}")
    sys.exit()

step, last_modified, next_spawn = 0, 0, 0

while step < MAX_SIM_TIME:
    traci.simulationStep()

    if os.path.exists(LIVE_FILE):
        mod = os.path.getmtime(LIVE_FILE)
        if mod != last_modified:
            last_modified = mod
            try:
                df = pd.read_csv(LIVE_FILE)
                spawn_queue.clear() # Clear old to prioritize new traffic
                for _, row in df.iterrows():
                    tid = str(int(float(row["train_id"])))
                    ttype = classify(tid)
                    route = assign_route(ttype, detect_direction(float(row["latitude"])))
                    # Append a unique step identifier so SUMO never complains about duplicates
                    unique_tid = f"{tid}_{step}_{random.randint(100,999)}"
                    spawn_queue.append((unique_tid, route, ttype))
            except: pass

    if spawn_queue and step >= next_spawn:
        tid, route, ttype = spawn_queue.pop(0)
        try:
            if traci.edge.getLastStepVehicleNumber(traci.route.getEdges(route)[0]) < 2:
                traci.vehicle.add(tid, route, typeID="LIVE_RAIL")
                traci.vehicle.setMaxSpeed(tid, BASE_SPEED[ttype])
        except: pass
        next_spawn = step + SPAWN_INTERVAL

    enforce_headway()
    junction_conflicts = enforce_junctions()
    auto_switches = predictive_congestion()
    manual_switches = process_human_approvals() 
    total_switches = auto_switches + manual_switches
    vehicles = traci.vehicle.getIDList()

    live_trains = []
    for v in vehicles:
        try:
            x, y = traci.vehicle.getPosition(v)
            lon, lat = traci.simulation.convertGeo(x, y)
            live_trains.append({"id": v, "lat": lat, "lon": lon})
        except: pass

    with open(LIVE_JSON, "w") as f: json.dump(live_trains, f)

    avg_speed = statistics.mean([traci.vehicle.getSpeed(v) for v in vehicles]) if vehicles else 0
    congestion = 1 - (avg_speed/44) if vehicles else 0
    metrics_log.append([step, len(vehicles), avg_speed, congestion, len(overtake_events), junction_conflicts, total_switches])

    if step % 50 == 0:
        print(f"📊 Active:{len(vehicles)} | AvgSpeed:{round(avg_speed,2)} | Cong:{round(congestion,3)} | Switches:{total_switches}")

    # <--- NEW: Save metrics to CSV continuously so the UI updates live! --->
    if step % 5 == 0:
        try:
            with open(RESULT_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "active_trains", "avg_speed", "congestion_index", "overtake_events", "junction_conflicts", "dynamic_switches"])
                # Write the last 20 rows to keep file operations lightning fast
                writer.writerows(metrics_log[-20:]) 
        except: pass

    step += 1

traci.close()
print("🚆 Simulation Complete")
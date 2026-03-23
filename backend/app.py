"""
app.py  —  Fixed & extended Flask backend
New endpoints:
  GET  /train_details        per-train enriched data (XGBoost prediction per train)
  GET  /dqn_status           live DQN recommendation from current sim state
  GET  /model_benchmarks     benchmark metrics from saved files
  POST /predict_train        single-train delay prediction (from ai_dashboard logic)
  POST /predict_route        chained route propagation prediction
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, json, joblib, threading, time
import xgboost as xgb

app = Flask(__name__)
CORS(app)

import logging
log = logging.getLogger(__name__)

# Edge lookup cached at startup
_edge_cache = {}

def load_edges_for_route() -> dict:
    """Returns {(from_station, to_station): {distance_km, block_count, min_run_time_min}}"""
    global _edge_cache
    if _edge_cache:
        return _edge_cache
    path = os.path.join(os.path.dirname(__file__), "..", "data", "csv", "section_edges.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            _edge_cache[(row["from_station"], row["to_station"])] = {
                "distance_km":      float(row.get("distance_km", 8)),
                "block_count":      int(row.get("block_count", 4)),
                "min_run_time_min": float(row.get("min_run_time_min", 8)),
            }
    return _edge_cache

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models", "saved_models")

LIVE_FILE        = os.path.join(OUTPUT_DIR, "live_trains.json")
LIVE_BLOCK_FILE  = os.path.join(OUTPUT_DIR, "live_blocks.json")
LIVE_EDGE_FILE   = os.path.join(OUTPUT_DIR, "live_edges.json")
RESULT_FILE      = os.path.join(OUTPUT_DIR, "simulation_results.csv")
TRACKS_FILE      = os.path.join(OUTPUT_DIR, "tracks.geojson")
PENDING_FILE     = os.path.join(OUTPUT_DIR, "pending_actions.json")
APPROVED_FILE    = os.path.join(OUTPUT_DIR, "approved_actions.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

for fp in [PENDING_FILE, APPROVED_FILE]:
    if not os.path.exists(fp):
        with open(fp, "w") as f:
            json.dump([], f)

# ─── File lock for concurrent JSON access ─────────────────────────────────────
_file_lock = threading.Lock()


def _read_json_safe(path: str, default):
    with _file_lock:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return default


def _write_json_safe(path: str, data):
    with _file_lock:
        with open(path, "w") as f:
            json.dump(data, f)


# ─── Model loading ────────────────────────────────────────────────────────────

xgb_model  = None
le_train    = None
le_station  = None
dqn_agent   = None

def _load_models():
    global xgb_model, le_train, le_station, dqn_agent

    xgb_path = os.path.join(MODELS_DIR, "xgboost_delay_model.json")
    pkl_path  = os.path.join(MODELS_DIR, "xgboost_delay_model.pkl")

    if os.path.exists(xgb_path):
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(xgb_path)
            print("✅ XGBoost loaded from .json")
        except Exception as e:
            print(f"⚠️ XGBoost .json failed: {e}")
    elif os.path.exists(pkl_path):
        try:
            xgb_model = joblib.load(pkl_path)
            print("✅ XGBoost loaded from .pkl")
        except Exception as e:
            print(f"⚠️ XGBoost .pkl failed: {e}")
    else:
        print("⚠️ XGBoost model not found — run train_xgboost_real.py first")

    for name, var_name in [("le_train_type", "le_train"), ("le_station_code", "le_station")]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                if var_name == "le_train":
                    le_train = obj
                else:
                    le_station = obj
            except Exception as e:
                print(f"⚠️ Encoder {name} failed: {e}")

    try:
        from dqn_env import load_trained_agent
        dqn_agent = load_trained_agent()
        if dqn_agent:
            print("✅ DQN model loaded")
        else:
            print("⚠️ DQN model not found — run dqn_env.py first")
    except Exception as e:
        print(f"⚠️ DQN load failed: {e}")


_load_models()


# ─── Helpers ──────────────────────────────────────────────────────────────────

# SUMO vehicle priority (numeric ID ranges → SUMO internal type)
SUMO_PRIORITY_MAP = {"SUPERFAST": 5, "EXPRESS": 4, "PASSENGER": 3, "EMU": 2, "FREIGHT": 1}

# Real CSV data priority (section_train_master_expanded.dispatch_priority)
# Local=2, MEMU=3, Passenger=4, Express=5
PRIORITY_MAP = {"Express": 5, "Passenger": 4, "MEMU": 3, "Local": 2}

# Howrah–Bandel corridor (section_master_final.csv)
STATION_ORDER    = ["HWH", "BLY", "SRP", "CGR", "CNS", "BDC"]
REAL_TRAIN_TYPES = ["Local", "MEMU", "Passenger", "Express"]

def _classify_train(train_id: str) -> str:
    """
    Classify SUMO-generated numeric train IDs into SUMO route types.
    Real CSV trains (T-prefix) fall through to 'Local' default.
    SUMO numeric ranges are defined in generate_dummy_traffic.py.
    """
    try:
        num = int(str(train_id).split("_")[0])
        if 37000 <= num < 38000: return "EMU"
        if 15000 <= num < 16000: return "PASSENGER"
        if 13000 <= num < 14000: return "EXPRESS"
        if 22000 <= num < 23000: return "SUPERFAST"
        if 63000 <= num < 64000: return "FREIGHT"
    except Exception:
        pass
    return "Local"   # default for real T-prefix trains


def _sumo_type_to_priority(ttype: str) -> int:
    """Map SUMO type → dispatch_priority integer used by XGBoost."""
    return {"SUPERFAST": 5, "EXPRESS": 4, "PASSENGER": 4,
            "EMU": 3, "FREIGHT": 2, "Local": 2,
            "MEMU": 3, "Passenger": 4, "Express": 5}.get(ttype, 2)


def _encode_safe(encoder, value: str) -> int:
    if encoder is None:
        return 0
    classes = list(encoder.classes_)
    return int(encoder.transform([value])[0]) if value in classes else 0


def _predict_delay(
    train_type: str,
    from_station: str,
    to_station: str = "BDC",
    hour: int = 12,
    excess_travel_time: float = 0.0,
    priority: int = 2,
    congestion: float = 0.3,
    distance_km: float = 8.0,
    block_count: int = 4,
    num_tracks: int = 4,
    station_avg_delay: float = 0.81,
) -> float:
    """
    Predict section delay using the 10-feature section-traffic model.
    Feature order must match build_xgboost_section_dataset() exactly:
      excess_travel_time, hour_of_day, train_type_enc, from_station_enc,
      to_station_enc, dispatch_priority, station_avg_delay, num_tracks,
      distance_km, block_count
    """
    if xgb_model is None:
        return 0.0
    try:
        t_enc  = _encode_safe(le_train,   train_type)
        fs_enc = _encode_safe(le_station, from_station)
        ts_enc = _encode_safe(le_station, to_station)

        feats = pd.DataFrame([{
            "excess_travel_time": float(excess_travel_time),
            "hour_of_day":        int(hour),
            "train_type_enc":     t_enc,
            "from_station_enc":   fs_enc,
            "to_station_enc":     ts_enc,
            "dispatch_priority":  int(priority),
            "station_avg_delay":  float(station_avg_delay),
            "num_tracks":         int(num_tracks),
            "distance_km":        float(distance_km),
            "block_count":        int(block_count),
        }])
        pred = xgb_model.predict(feats)[0]
        return max(0.0, round(float(pred), 2))
    except Exception as e:
        log.debug(f"_predict_delay error: {e}")
        return 0.0


def _get_latest_metrics():
    if not os.path.exists(RESULT_FILE):
        return {}
    try:
        df = pd.read_csv(RESULT_FILE)
        if df.empty:
            return {}
        return df.iloc[-1].to_dict()
    except Exception:
        return {}


# ─── Existing endpoints (fixed) ───────────────────────────────────────────────

@app.route("/metrics")
def metrics():
    m = _get_latest_metrics()
    if not m:
        return jsonify({})
    return jsonify({
        "active_trains": int(m.get("active_trains", 0)),
        "avg_speed":     round(float(m.get("avg_speed", 0)), 2),
        "congestion":    round(float(m.get("congestion_index", 0)), 3),
        "conflicts":     int(m.get("junction_conflicts", 0)),
        "step":          int(m.get("step", 0)),
    })


@app.route("/trains")
def trains():
    return jsonify(_read_json_safe(LIVE_FILE, []))


@app.route("/edges")
def edges():
    return jsonify(_read_json_safe(LIVE_EDGE_FILE, []))


@app.route("/blocks")
def blocks():
    return jsonify(_read_json_safe(LIVE_BLOCK_FILE, []))


@app.route("/tracks")
def tracks():
    if not os.path.exists(TRACKS_FILE):
        return jsonify({})
    try:
        with open(TRACKS_FILE, "r") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({})


# ─── Original predict_delay (network-level, kept for backward compat) ─────────

@app.route("/predict_delay")
def predict_delay():
    """Network-level delay estimate from live simulation state."""
    m = _get_latest_metrics()
    if not m or xgb_model is None:
        return jsonify({"expected_delay_minutes": 0})
    try:
        active    = int(m.get("active_trains", 5))
        cong      = float(m.get("congestion_index", 0.3))
        hour      = int(pd.Timestamp.now().hour)
        pred      = _predict_delay("Local", "HWH", "BDC", hour, 0.0, 2)
        return jsonify({"expected_delay_minutes": pred})
    except Exception:
        return jsonify({"expected_delay_minutes": 0})


@app.route("/custom_predict", methods=["POST"])
def custom_predict():
    """Legacy what-if endpoint — kept for backward compatibility."""
    data = request.json or {}
    try:
        cong  = float(data.get("congestion_index", 0.3))
        hour  = int(data.get("hour_of_day", 12))
        prev  = float(data.get("prev_delay", 0))
        pred  = _predict_delay("Local", "HWH", "BDC", hour, 0.0, 2)
        return jsonify({"expected_delay_minutes": pred})
    except Exception:
        return jsonify({"expected_delay_minutes": 0})


# ─── NEW: Per-train enriched details ─────────────────────────────────────────

@app.route("/train_details")
def train_details():
    """
    Returns live trains enriched with:
      - train_type (classified from ID)
      - priority
      - xgboost_predicted_delay
      - dqn_recommendation (if congestion > trigger)
    """
    live_trains = _read_json_safe(LIVE_FILE, [])
    m           = _get_latest_metrics()
    congestion  = float(m.get("congestion_index", 0.3))
    hour        = int(pd.Timestamp.now().hour)

    enriched = []
    for t in live_trains:
        tid        = str(t.get("id", ""))
        ttype      = _classify_train(tid)
        priority   = _sumo_type_to_priority(ttype)

        # XGBoost per-train delay prediction using section-traffic model
        pred_delay = _predict_delay(
            ttype, "HWH", "BDC", hour,
            excess_travel_time=0.0,
            priority=priority,
        )

        enriched.append({
            "id":              tid,
            "lat":             t.get("lat", 0),
            "lon":             t.get("lon", 0),
            "train_type":      ttype,
            "priority":        priority,
            "predicted_delay": pred_delay,
            "congestion":      round(congestion, 3),
        })

    return jsonify(enriched)


# ─── NEW: Single-train prediction (from ai_dashboard.py logic) ───────────────

@app.route("/predict_train", methods=["POST"])
def predict_train():
    """
    Per-train section delay prediction using 10-feature model.
    Body: {train_type, from_station, to_station, hour_of_day,
           excess_travel_time, dispatch_priority,
           distance_km, block_count, num_tracks, station_avg_delay}
    """
    if xgb_model is None:
        return jsonify({"error": "Model not loaded", "predicted_delay": 0})

    data    = request.json or {}
    ttype   = data.get("train_type",         "Local")
    from_s  = data.get("from_station",       "HWH")
    to_s    = data.get("to_station",         "BDC")
    hour    = int(data.get("hour_of_day",    12))
    excess  = float(data.get("excess_travel_time", 0.0))
    priority= int(data.get("dispatch_priority", 2))
    dist    = float(data.get("distance_km",  8.0))
    blocks  = int(data.get("block_count",    4))
    tracks  = int(data.get("num_tracks",     4))
    avg_d   = float(data.get("station_avg_delay", 0.81))

    pred = _predict_delay(ttype, from_s, to_s, hour, excess, priority,
                          0.3, dist, blocks, tracks, avg_d)
    return jsonify({
        "predicted_delay": pred,
        "train_type":      ttype,
        "from_station":    from_s,
        "to_station":      to_s,
        "unit":            "minutes",
    })


# ─── NEW: Chained route propagation ──────────────────────────────────────────

@app.route("/predict_route", methods=["POST"])
def predict_route():
    """
    Predict delay cascade across a route using section-by-section predictions.
    Body: {train_type, stations: [list], departure_hour, dispatch_priority}
    Each hop uses section_edges defaults for distance/block_count.
    """
    if xgb_model is None:
        return jsonify({"error": "Model not loaded", "route_predictions": []})

    # Load edge lookup for realistic features
    edges_df  = load_edges_for_route()
    data      = request.json or {}
    ttype     = data.get("train_type",     "Local")
    stations  = data.get("stations",       STATION_ORDER)
    dep_hour  = int(data.get("departure_hour",  7))
    priority  = int(data.get("dispatch_priority", 2))

    results       = []
    running_delay = float(data.get("initial_delay", 0))

    for i in range(len(stations) - 1):
        from_s = stations[i]
        to_s   = stations[i + 1]
        hour   = (dep_hour + i) % 24

        # Look up edge properties
        edge  = edges_df.get((from_s, to_s), {})
        dist  = float(edge.get("distance_km",     8.0))
        blk   = int(edge.get("block_count",        4))
        min_t = float(edge.get("min_run_time_min", 8.0))

        # excess_travel_time: running delay adds to travel time
        excess = max(0.0, running_delay * 0.3)  # partial cascade

        pred          = _predict_delay(ttype, from_s, to_s, hour, excess, priority,
                                       0.3, dist, blk)
        running_delay = running_delay + pred
        results.append({
            "from_station":     from_s,
            "to_station":       to_s,
            "predicted_delay":  round(pred, 2),
            "cumulative_delay": round(running_delay, 2),
        })

    return jsonify({
        "train_type":        ttype,
        "initial_delay":     float(data.get("initial_delay", 0)),
        "route_predictions": results,
        "total_delay":       round(running_delay, 2),
    })


# ─── NEW: DQN live recommendation ────────────────────────────────────────────

@app.route("/dqn_status")
def dqn_status():
    """
    Returns DQN recommendation based on current simulation state.
    Always fires when congestion > CONGESTION_TRIGGER (0.30).
    """
    m          = _get_latest_metrics()
    congestion = float(m.get("congestion_index", 0.0))
    active     = float(m.get("active_trains", 0))
    hour       = float(pd.Timestamp.now().hour)

    from dqn_env import CONGESTION_TRIGGER, get_dqn_recommendation

    if dqn_agent is None:
        return jsonify({
            "available": False,
            "message":   "DQN model not loaded. Run dqn_env.py to train.",
        })

    if congestion < CONGESTION_TRIGGER:
        return jsonify({
            "available":    True,
            "triggered":    False,
            "congestion":   round(congestion, 3),
            "trigger_threshold": CONGESTION_TRIGGER,
            "message":      "Network nominal. DQN idle.",
        })

    rec = get_dqn_recommendation(
        dqn_agent,
        congestion  = congestion,
        active_trains = active,
        priority_w  = 0.4,
        time_of_day = hour,
    )
    rec["available"] = True
    rec["triggered"] = True
    return jsonify(rec)


# ─── NEW: Model benchmarks ────────────────────────────────────────────────────

@app.route("/model_benchmarks")
def model_benchmarks():
    result = {}

    xgb_bench = os.path.join(MODELS_DIR, "xgboost_benchmark.txt")
    if os.path.exists(xgb_bench):
        with open(xgb_bench) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    try:
                        result[f"xgb_{k}"] = float(v)
                    except ValueError:
                        result[f"xgb_{k}"] = v

    dqn_bench = os.path.join(MODELS_DIR, "dqn_benchmark.json")
    if os.path.exists(dqn_bench):
        with open(dqn_bench) as f:
            dqn_data = json.load(f)
        result.update({f"dqn_{k}": v for k, v in dqn_data.items()})

    result["xgb_loaded"] = xgb_model is not None
    result["dqn_loaded"] = dqn_agent is not None
    return jsonify(result)


# ─── HITL endpoints (fixed with file lock) ────────────────────────────────────

@app.route("/alerts")
def get_alerts():
    return jsonify(_read_json_safe(PENDING_FILE, []))


@app.route("/resolve_alert", methods=["POST"])
def resolve_alert():
    data     = request.json or {}
    alert_id = data.get("alert_id")
    action   = data.get("action")  # "approve" or "dismiss"

    with _file_lock:
        try:
            with open(PENDING_FILE, "r") as f:
                pending = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pending = []

        target  = next((a for a in pending if a.get("id") == alert_id), None)
        pending = [a for a in pending if a.get("id") != alert_id]

        with open(PENDING_FILE, "w") as f:
            json.dump(pending, f)

        if target and action == "approve":
            try:
                with open(APPROVED_FILE, "r") as f:
                    approved = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                approved = []

            approved.append({
                "train_id":  target.get("train_id"),
                "new_route": target.get("new_route"),
            })
            with open(APPROVED_FILE, "w") as f:
                json.dump(approved, f)

    return jsonify({"status": "success"})


# ─── Available stations & types (for frontend dropdowns) ─────────────────────

@app.route("/metadata")
def metadata():
    le_t = le_train
    le_s = le_station
    train_types   = list(le_t.classes_) if le_t else REAL_TRAIN_TYPES
    station_codes = list(le_s.classes_) if le_s else STATION_ORDER
    return jsonify({
        "train_types":   train_types,
        "station_codes": station_codes,
        "priorities":    [2, 3, 4, 5],
        "priority_labels": {"2": "Local", "3": "MEMU", "4": "Passenger", "5": "Express"},
    })


if __name__ == "__main__":
    app.run(port=5000, debug=False)

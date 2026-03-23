from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import joblib

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

LIVE_FILE = os.path.join(OUTPUT_DIR, "live_trains.json")
LIVE_BLOCK_FILE = os.path.join(OUTPUT_DIR, "live_blocks.json")
LIVE_EDGE_FILE = os.path.join(OUTPUT_DIR, "live_edges.json")
RESULT_FILE = os.path.join(OUTPUT_DIR, "simulation_results.csv")
TRACKS_FILE = os.path.join(OUTPUT_DIR, "tracks.geojson")
MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_delay_model.pkl")

# NEW: Human-in-the-loop communication files
PENDING_FILE = os.path.join(OUTPUT_DIR, "pending_actions.json")
APPROVED_FILE = os.path.join(OUTPUT_DIR, "approved_actions.json")

# Ensure communication files exist
for file_path in [PENDING_FILE, APPROVED_FILE]:
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)

# ---------------------------------------------------------
# LOAD AI MODEL (XGBOOST)
# ---------------------------------------------------------
xgb_model = None
if os.path.exists(MODEL_PATH):
    try:
        xgb_model = joblib.load(MODEL_PATH)
        print("✅ XGBoost Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")

# ---------------------------------------------------------
# STANDARD API ENDPOINTS (Metrics, Map Data, Predictions)
# ---------------------------------------------------------

@app.route("/metrics")
def metrics():
    if not os.path.exists(RESULT_FILE): return jsonify({})
    try:
        df = pd.read_csv(RESULT_FILE)
        if df.empty: return jsonify({})
        latest = df.iloc[-1]
        return jsonify({
            "active_trains": int(latest["active_trains"]),
            "avg_speed": round(latest["avg_speed"], 2),
            "congestion": round(latest["congestion_index"], 2),
            "conflicts": int(latest["junction_conflicts"])
        })
    except Exception: return jsonify({})

@app.route("/edges")
def edges():
    try:
        with open(LIVE_EDGE_FILE, "r") as f: return jsonify(json.load(f))
    except: return jsonify([])

@app.route("/blocks")
def blocks():
    try:
        with open(LIVE_BLOCK_FILE, "r") as f: return jsonify(json.load(f))
    except: return jsonify([])

@app.route("/trains")
def trains():
    try:
        with open(LIVE_FILE, "r") as f: return jsonify(json.load(f))
    except: return jsonify([])

@app.route("/tracks")
def tracks():
    try:
        with open(TRACKS_FILE, "r") as f: return jsonify(json.load(f))
    except: return jsonify({})

@app.route("/predict_delay")
def predict_delay():
    if xgb_model is None or not os.path.exists(RESULT_FILE):
        return jsonify({"expected_delay_minutes": 0})
    try:
        df = pd.read_csv(RESULT_FILE)
        if df.empty: return jsonify({"expected_delay_minutes": 0})
        latest = df.iloc[-1]
        features = pd.DataFrame([{
            'active_trains': latest['active_trains'],
            'congestion_index': latest['congestion_index'],
            'junction_conflicts': latest['junction_conflicts']
        }])
        prediction = xgb_model.predict(features)[0]
        return jsonify({"expected_delay_minutes": max(0, round(float(prediction), 1))})
    except Exception: return jsonify({"expected_delay_minutes": 0})

@app.route("/custom_predict", methods=["POST"])
def custom_predict():
    if xgb_model is None: return jsonify({"expected_delay_minutes": 0})
    try:
        data = request.json
        features = pd.DataFrame([{
            'active_trains': float(data.get('active_trains', 0)),
            'congestion_index': float(data.get('congestion_index', 0)),
            'junction_conflicts': float(data.get('junction_conflicts', 0))
        }])
        prediction = xgb_model.predict(features)[0]
        return jsonify({"expected_delay_minutes": max(0, round(float(prediction), 1))})
    except Exception: return jsonify({"expected_delay_minutes": 0})

# ---------------------------------------------------------
# NEW: HITL DISPATCHER ENDPOINTS
# ---------------------------------------------------------

@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Fetch pending AI recommendations for the UI."""
    try:
        with open(PENDING_FILE, "r") as f:
            return jsonify(json.load(f))
    except json.JSONDecodeError:
        return jsonify([])

@app.route("/resolve_alert", methods=["POST"])
def resolve_alert():
    """Process human approval or dismissal."""
    try:
        data = request.json
        alert_id = data.get("alert_id")
        action = data.get("action") # "approve" or "dismiss"

        # 1. Read pending actions
        with open(PENDING_FILE, "r") as f:
            pending = json.load(f)

        # 2. Find the alert and remove it from pending
        target_alert = next((a for a in pending if a["id"] == alert_id), None)
        pending = [a for a in pending if a["id"] != alert_id]

        with open(PENDING_FILE, "w") as f:
            json.dump(pending, f)

        # 3. If approved, add it to the approved actions file for SUMO to execute
        if target_alert and action == "approve":
            with open(APPROVED_FILE, "r") as f:
                approved = json.load(f)
            
            approved.append({
                "train_id": target_alert["train_id"],
                "new_route": target_alert["new_route"]
            })

            with open(APPROVED_FILE, "w") as f:
                json.dump(approved, f)

        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error resolving alert: {e}")
        return jsonify({"status": "error"}), 500

if __name__ == "__main__":
    app.run(port=5000)
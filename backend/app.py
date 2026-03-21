from flask import Flask, jsonify
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
# API ENDPOINTS
# ---------------------------------------------------------

@app.route("/metrics")
def metrics():
    if not os.path.exists(RESULT_FILE):
        return jsonify({})

    try:
        df = pd.read_csv(RESULT_FILE)
        if df.empty:
            return jsonify({})

        latest = df.iloc[-1]
        return jsonify({
            "active_trains": int(latest["active_trains"]),
            "avg_speed": round(latest["avg_speed"], 2),
            "congestion": round(latest["congestion_index"], 2),
            "conflicts": int(latest["junction_conflicts"])
        })
    except Exception:
        return jsonify({})

@app.route("/edges")
def edges():
    if not os.path.exists(LIVE_EDGE_FILE):
        return jsonify([])
    try:
        with open(LIVE_EDGE_FILE, "r") as f:
            return jsonify(json.load(f))
    except json.JSONDecodeError:
        return jsonify([])

@app.route("/blocks")
def blocks():
    if not os.path.exists(LIVE_BLOCK_FILE):
        return jsonify([])
    try:
        with open(LIVE_BLOCK_FILE, "r") as f:
            return jsonify(json.load(f))
    except json.JSONDecodeError:
        return jsonify([])

@app.route("/trains")
def trains():
    if not os.path.exists(LIVE_FILE):
        return jsonify([])
    try:
        with open(LIVE_FILE, "r") as f:
            return jsonify(json.load(f))
    except json.JSONDecodeError:
        # Prevents crashing when SUMO is mid-write
        return jsonify([])

@app.route("/tracks")
def tracks():
    # Added safety check so missing tracks.geojson doesn't crash the app
    if not os.path.exists(TRACKS_FILE):
        return jsonify({})
    try:
        with open(TRACKS_FILE, "r") as f:
            return jsonify(json.load(f))
    except json.JSONDecodeError:
        return jsonify({})

@app.route("/predict_delay")
def predict_delay():
    if xgb_model is None:
        return jsonify({"error": "Model not loaded", "expected_delay_minutes": 0})

    if not os.path.exists(RESULT_FILE):
        return jsonify({"expected_delay_minutes": 0})

    try:
        df = pd.read_csv(RESULT_FILE)
        if df.empty:
            return jsonify({"expected_delay_minutes": 0})

        # Grab the very latest state of the digital twin
        latest = df.iloc[-1]
        
        # Format the data exactly how the XGBoost model expects it
        features = pd.DataFrame([{
            'active_trains': latest['active_trains'],
            'congestion_index': latest['congestion_index'],
            'junction_conflicts': latest['junction_conflicts']
        }])

        # Ask the AI to predict the delay
        prediction = xgb_model.predict(features)[0]
        
        # Ensure it doesn't predict negative time, and round to 1 decimal
        final_delay = max(0, round(float(prediction), 1))

        return jsonify({"expected_delay_minutes": final_delay})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"expected_delay_minutes": 0})

if __name__ == "__main__":
    app.run(port=5000)
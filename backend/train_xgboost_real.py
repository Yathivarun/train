"""
train_xgboost_real.py
Trains XGBoost on REAL CSV data (station_delay_log + real_operational_log).
Supports both:
  1. Single-station prediction  (direct inference)
  2. Chained propagation        (predict delay cascade across a train's route)
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Allow running from either project root or backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import (
    build_xgboost_dataset,
    get_station_list,
    get_train_types,
    MODELS_DIR,
)

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "prev_delay",
    "train_type_enc", "station_code_enc",
    "priority", "congestion",
]

MODEL_JSON  = os.path.join(MODELS_DIR, "xgboost_delay_model.json")
MODEL_PKL   = os.path.join(MODELS_DIR, "xgboost_delay_model.pkl")
RESULTS_TXT = os.path.join(MODELS_DIR, "xgboost_benchmark.txt")


def train():
    print("=" * 60)
    print("XGBoost Training — Real CSV Data")
    print("=" * 60)

    # ── Load real data ───────────────────────────────────────────
    X, y, le_train, le_station = build_xgboost_dataset()

    if X is None or X.empty:
        print("\n⚠️  Real CSV data not available.")
        print("Falling back to STRUCTURED synthetic data as placeholder.")
        print("Replace data/csv/ files with real NTES logs to use real data.\n")
        X, y, le_train, le_station = _build_structured_synthetic()

    print(f"\n📊 Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"   Target range: {y.min():.1f} – {y.max():.1f} minutes")
    print(f"   Mean delay: {y.mean():.2f} min  |  Std: {y.std():.2f} min")
    print(f"   Features: {list(X.columns)}")

    # ── Train / test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Model ────────────────────────────────────────────────────
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    print("\n🚀 Training …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Benchmark ────────────────────────────────────────────────
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    mse  = mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mse)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (held-out 20% test set)")
    print("=" * 60)
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f} minutes")
    print(f"  MAE  : {mae:.4f} minutes  ← average prediction error")
    print(f"  R²   : {r2:.4f}           ← 1.0 = perfect")
    print("=" * 60)

    if r2 > 0.95:
        print("⚠️  R² > 0.95 on synthetic data is expected — validate on real data!")
    elif r2 > 0.7:
        print("✅  Good R² on real data.")
    else:
        print("ℹ️  R² below 0.7 — consider adding more features or more data.")

    # Save benchmark
    with open(RESULTS_TXT, "w") as f:
        f.write(f"MSE={mse:.4f}\nRMSE={rmse:.4f}\nMAE={mae:.4f}\nR2={r2:.4f}\n")
        f.write(f"Samples={len(X)}\nFeatures={list(X.columns)}\n")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n📌 Feature Importance:")
    for feat, imp in importance.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<20} {bar} {imp:.4f}")

    # ── Save ─────────────────────────────────────────────────────
    model.save_model(MODEL_JSON)
    joblib.dump(model, MODEL_PKL)
    print(f"\n✅ Model saved → {MODEL_JSON}")
    print(f"✅ Encoders  → {MODELS_DIR}/le_*.pkl")

    return model, le_train, le_station


# ─── Chained propagation predictor ───────────────────────────────────────────

class ChainedDelayPredictor:
    """
    Predict delay propagation for a train across its full route.
    Each station's predicted delay feeds into the next station's prev_delay.
    """

    def __init__(self, model_path=MODEL_JSON, encoders_dir=MODELS_DIR):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.le_train   = joblib.load(os.path.join(encoders_dir, "le_train_type.pkl"))
        self.le_station = joblib.load(os.path.join(encoders_dir, "le_station_code.pkl"))

    def _encode_safe(self, encoder: joblib, value: str) -> int:
        classes = list(encoder.classes_)
        if value in classes:
            return encoder.transform([value])[0]
        # Unknown category → use most frequent (index 0 after sorting)
        return 0

    def predict_single(
        self,
        train_type: str,
        station_code: str,
        hour_of_day: int,
        day_of_week: int,
        prev_delay: float,
        priority: int = 2,
        congestion: float = 0.3,
    ) -> float:
        t_enc = self._encode_safe(self.le_train,   train_type)
        s_enc = self._encode_safe(self.le_station, station_code)

        features = pd.DataFrame([{
            "hour_of_day":      hour_of_day,
            "day_of_week":      day_of_week,
            "prev_delay":       prev_delay,
            "train_type_enc":   t_enc,
            "station_code_enc": s_enc,
            "priority":         priority,
            "congestion":       congestion,
        }])
        pred = self.model.predict(features)[0]
        return max(0.0, float(pred))

    def predict_route(
        self,
        train_type: str,
        stations: list[str],
        initial_delay: float,
        departure_hour: int,
        day_of_week: int = 0,
        priority: int = 2,
        congestion: float = 0.3,
    ) -> list[dict]:
        """
        Chain predictions across a route.
        Returns list of {station, predicted_delay, cumulative_delay} dicts.
        """
        results = []
        running_delay = initial_delay

        for i, station in enumerate(stations):
            hour = (departure_hour + i) % 24  # rough hour increment per station
            predicted = self.predict_single(
                train_type, station, hour, day_of_week,
                running_delay, priority, congestion,
            )
            results.append({
                "station":          station,
                "predicted_delay":  round(predicted, 2),
                "cumulative_delay": round(running_delay + predicted, 2),
            })
            running_delay = results[-1]["cumulative_delay"]

        return results


# ─── Structured synthetic fallback (NOT circular) ────────────────────────────

def _build_structured_synthetic():
    """
    Generates synthetic data with REALISTIC, non-circular patterns.
    Based on domain knowledge of Indian Railways:
      - Freight trains carry higher base delays
      - Peak hours (7-9 AM, 5-7 PM) correlate with higher delays
      - Higher prev_delay strongly predicts next delay (cascade)
      - High congestion increases delay non-linearly
    This is transparent and academically documented.
    """
    from sklearn.preprocessing import LabelEncoder
    np.random.seed(42)
    N = 8000

    train_types   = ["EMU", "PASSENGER", "EXPRESS", "SUPERFAST", "FREIGHT"]
    station_codes = ["HWH", "BLY", "SRP", "CNR", "CHU", "BDC",
                     "BHP", "KAN", "MGR", "TKP"]

    priority_map  = {"SUPERFAST": 5, "EXPRESS": 4, "PASSENGER": 3, "EMU": 2, "FREIGHT": 1}
    base_delay    = {"SUPERFAST": 2, "EXPRESS": 4, "PASSENGER": 6, "EMU": 3, "FREIGHT": 12}

    rows = []
    for _ in range(N):
        ttype    = np.random.choice(train_types)
        station  = np.random.choice(station_codes)
        hour     = np.random.randint(0, 24)
        dow      = np.random.randint(0, 7)
        prev_d   = np.random.exponential(scale=5.0)
        priority = priority_map[ttype]
        cong     = np.random.beta(2, 5)  # right-skewed, mostly low congestion

        # Realistic delay formula (domain-driven, NOT circular)
        peak_factor = 1.5 if hour in range(7, 10) or hour in range(17, 20) else 1.0
        base        = base_delay[ttype]
        noise       = np.random.normal(0, 2)
        delay       = max(0, base * peak_factor + 0.6 * prev_d + 8 * cong + noise)

        rows.append({
            "train_type":      ttype,
            "station_code":    station,
            "hour_of_day":     hour,
            "day_of_week":     dow,
            "prev_delay":      prev_d,
            "priority":        priority,
            "congestion":      cong,
            "actual_delay":    delay,
        })

    df = pd.DataFrame(rows)

    le_train   = LabelEncoder().fit(df["train_type"])
    le_station = LabelEncoder().fit(df["station_code"])

    df["train_type_enc"]   = le_train.transform(df["train_type"])
    df["station_code_enc"] = le_station.transform(df["station_code"])

    import joblib
    joblib.dump(le_train,   os.path.join(MODELS_DIR, "le_train_type.pkl"))
    joblib.dump(le_station, os.path.join(MODELS_DIR, "le_station_code.pkl"))

    X = df[FEATURE_COLS]
    y = df["actual_delay"]
    return X, y, le_train, le_station


if __name__ == "__main__":
    train()

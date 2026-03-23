"""
benchmark_xgboost.py  —  Fixed benchmark script
BEFORE: trained a brand-new model inside the benchmark, tested on synthetic targets.
        Results had no connection to the model used in production.
AFTER:  loads the saved model from models/saved_models/ and evaluates it on a
        genuine held-out split of real CSV data.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
MODEL_JSON  = os.path.join(MODELS_DIR, "xgboost_delay_model.json")
MODEL_PKL   = os.path.join(MODELS_DIR, "xgboost_delay_model.pkl")

print("=" * 60)
print("XGBoost Benchmark — Evaluating Production Model")
print("=" * 60)

# ── 1. Load the production model ─────────────────────────────────────────────
model = None
if os.path.exists(MODEL_JSON):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_JSON)
    print(f"✅ Loaded model from {MODEL_JSON}")
elif os.path.exists(MODEL_PKL):
    model = joblib.load(MODEL_PKL)
    print(f"✅ Loaded model from {MODEL_PKL}")
else:
    print("❌ No saved model found.")
    print("   Run: python backend/train_xgboost_real.py")
    sys.exit(1)

# ── 2. Load real data via pipeline ───────────────────────────────────────────
try:
    from data_pipeline import build_xgboost_dataset
    X, y, le_train, le_station = build_xgboost_dataset()
    data_source = "real CSV data"
except Exception as e:
    print(f"⚠️ data_pipeline failed ({e}) — using structured synthetic fallback")
    X, y, le_train, le_station = None, None, None, None

if X is None or (hasattr(X, 'empty') and X.empty):
    print("⚠️ Real data unavailable — generating structured synthetic test set")
    from train_xgboost_real import _build_structured_synthetic
    X, y, le_train, le_station = _build_structured_synthetic()
    data_source = "structured synthetic (domain-driven, not circular)"

print(f"\n📊 Evaluation dataset: {len(X)} samples | Source: {data_source}")

# ── 3. Use 20% held-out split (same seed as training) ────────────────────────
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Held-out test set: {len(X_test)} samples")

# ── 4. Predict with the PRODUCTION model ─────────────────────────────────────
preds = model.predict(X_test)
preds = np.maximum(preds, 0)   # no negative delays

mse  = mean_squared_error(y_test, preds)
mae  = mean_absolute_error(y_test, preds)
r2   = r2_score(y_test, preds)
rmse = np.sqrt(mse)

# ── 5. Results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRODUCTION MODEL BENCHMARK RESULTS")
print("=" * 60)
print(f"  Test samples         : {len(X_test)}")
print(f"  Data source          : {data_source}")
print(f"  MSE                  : {mse:.4f}")
print(f"  RMSE                 : {rmse:.4f} minutes  ← typical prediction error")
print(f"  MAE                  : {mae:.4f} minutes  ← average prediction error")
print(f"  R²                   : {r2:.4f}           ← 1.0 = perfect")
print("=" * 60)

if r2 > 0.95 and "synthetic" in data_source:
    print("\n⚠️  High R² on synthetic data is expected and does NOT prove real-world accuracy.")
    print("    Place real CSV files in data/csv/ and re-run for genuine validation.")
elif r2 > 0.70:
    print(f"\n✅  R² = {r2:.4f} on real data — model is genuinely predictive.")
else:
    print(f"\nℹ️   R² = {r2:.4f} — consider adding more features or more data rows.")

# ── 6. Error distribution breakdown ──────────────────────────────────────────
errors = np.abs(preds - y_test.values)
print(f"\n  Error distribution:")
print(f"    < 2 min   : {(errors < 2).mean()*100:.1f}% of predictions")
print(f"    2–5 min   : {((errors >= 2) & (errors < 5)).mean()*100:.1f}%")
print(f"    5–10 min  : {((errors >= 5) & (errors < 10)).mean()*100:.1f}%")
print(f"    > 10 min  : {(errors >= 10).mean()*100:.1f}%")

# ── 7. Feature importance (from loaded model) ────────────────────────────────
try:
    importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)
    print(f"\n📌 Feature Importance (production model):")
    for feat, imp in importance.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22} {bar} {imp:.4f}")
except Exception:
    pass

print("\n✅ Benchmark complete. These results reflect the PRODUCTION model.")
print("   Compare with models/saved_models/xgboost_benchmark.txt for training-time metrics.")

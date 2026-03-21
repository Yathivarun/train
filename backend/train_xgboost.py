import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# 1. Define where to save the trained model
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_delay_model.pkl")

print("🧠 Generating synthetic historical railway data...")
np.random.seed(42)
n_samples = 10000

# 2. Generate synthetic historical features based on your simulation metrics
active_trains = np.random.randint(1, 15, n_samples)
congestion = np.random.uniform(0.1, 0.9, n_samples)
conflicts = np.random.randint(0, 5, n_samples)

# 3. Create a realistic formula for Delay Minutes (Target Variable)
# More trains + high congestion + conflicts = high delay
delay = (active_trains * 1.2) + (congestion * 25) + (conflicts * 8) + np.random.normal(0, 3, n_samples)
delay = np.maximum(0, delay) # No negative delays

df = pd.DataFrame({
    'active_trains': active_trains,
    'congestion_index': congestion,
    'junction_conflicts': conflicts,
    'delay_minutes': delay
})

# 4. Define Features (X) and Target (y)
X = df[['active_trains', 'congestion_index', 'junction_conflicts']]
y = df['delay_minutes']

print("🚀 Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=150, 
    max_depth=5,
    learning_rate=0.1
)
model.fit(X, y)

# 5. Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model trained successfully! Saved to: {MODEL_PATH}")
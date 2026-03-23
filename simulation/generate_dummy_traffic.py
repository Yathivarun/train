"""
generate_dummy_traffic.py  —  Fixed with stress-test mode
Stress mode deliberately floods FAST routes to trigger DQN recommendations.
"""

import pandas as pd
import random
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--traffic", default="low",
                    choices=["low", "medium", "high", "very-high"])
parser.add_argument("--stress", action="store_true",
                    help="Stress mode: flood fast tracks, forces DQN above 30% congestion")
args = parser.parse_args()

TRAFFIC_LEVELS = {
    "low":       (1,  3,  5.0),
    "medium":    (5,  10, 3.0),
    "high":      (15, 25, 1.0),
    "very-high": (30, 50, 0.5),
}

# Stress mode: many trains, fast injection
STRESS_CONFIG = (40, 60, 0.3)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LIVE_FILE = os.path.join(OUTPUT_DIR, "live_status.csv")

# Train ID ranges per type — kept consistent with classify_from_id()
TRAIN_RANGES = {
    "EMU":       (37000, 37999),
    "PASSENGER": (15000, 15999),
    "EXPRESS":   (13000, 13999),
    "SUPERFAST": (22000, 22999),
    "FREIGHT":   (63000, 63999),
}

# In stress mode: bias toward higher-priority trains on fast routes
STRESS_WEIGHTS = {
    "EMU":       0.05,
    "PASSENGER": 0.10,
    "EXPRESS":   0.35,   # heavy express traffic
    "SUPERFAST": 0.40,   # heavy superfast traffic
    "FREIGHT":   0.10,
}

NORMAL_WEIGHTS = {
    "EMU":       0.30,
    "PASSENGER": 0.25,
    "EXPRESS":   0.20,
    "SUPERFAST": 0.15,
    "FREIGHT":   0.10,
}


def generate_train(stress: bool = False) -> dict:
    weights = STRESS_WEIGHTS if stress else NORMAL_WEIGHTS
    types   = list(weights.keys())
    probs   = list(weights.values())
    ttype   = random.choices(types, weights=probs, k=1)[0]

    lo, hi     = TRAIN_RANGES[ttype]
    train_id   = random.randint(lo, hi)

    # Stress: concentrate around a single corridor latitude to cause conflict
    if stress:
        latitude = round(random.uniform(22.55, 22.95), 4)  # tight UP corridor
    else:
        latitude = round(random.uniform(22.0, 23.5), 4)

    return {"train_id": train_id, "latitude": latitude, "train_type": ttype}


if args.stress:
    min_t, max_t, sleep_dur = STRESS_CONFIG
    print(f"🔴 STRESS TEST MODE — Flooding UP_FAST/DOWN_FAST routes")
    print(f"   Injection: {min_t}–{max_t} trains every {sleep_dur}s")
    print(f"   Expected: congestion will exceed 30% → DQN triggers")
else:
    min_t, max_t, sleep_dur = TRAFFIC_LEVELS[args.traffic]
    print(f"🚂 Traffic Generator | Level: {args.traffic.upper()}")
    print(f"   Injection: {min_t}–{max_t} trains every {sleep_dur}s")

print(f"   Output: {LIVE_FILE}")

try:
    while True:
        n      = random.randint(min_t, max_t)
        trains = [generate_train(stress=args.stress) for _ in range(n)]
        df     = pd.DataFrame(trains)
        df.to_csv(LIVE_FILE, index=False)

        type_counts = df["train_type"].value_counts().to_dict()
        print(f"[{time.strftime('%X')}] Injected {n} trains: {type_counts}")

        time.sleep(sleep_dur)

except KeyboardInterrupt:
    print("\n🛑 Generator stopped.")

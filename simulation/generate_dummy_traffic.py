import pandas as pd
import random
import time
import os
import argparse

# =====================================================
# CLI ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser(description="Synthetic Traffic Generator")
parser.add_argument("--traffic", type=str, default="low",
                    choices=["low", "medium", "high", "very-high"],
                    help="Set the traffic volume level")
args = parser.parse_args()

# =====================================================
# TRAFFIC CONFIGURATION
# =====================================================
# Define (min_trains, max_trains, sleep_time) for each level
TRAFFIC_LEVELS = {
    "low": (1, 3, 5.0),
    "medium": (5, 10, 3.0),
    "high": (15, 25, 1.0),
    "very-high": (30, 50, 0.5)
}

min_t, max_t, sleep_duration = TRAFFIC_LEVELS[args.traffic]

# =====================================================
# FILE PATHS
# =====================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LIVE_FILE = os.path.join(OUTPUT_DIR, "live_status.csv")

def generate_random_train():
    """Generate a train ID based on the classification logic"""
    train_types = [
        random.randint(37000, 37999), # EMU
        random.randint(15000, 15999), # PASSENGER
        random.randint(13000, 13999), # EXPRESS
        random.randint(22000, 22999), # SUPERFAST
        random.randint(63000, 63999)  # FREIGHT
    ]
    train_id = random.choice(train_types)
    
    # Generate a latitude around 22.8 to trigger both UP (>22.8) and DOWN (<=22.8) directions
    latitude = round(random.uniform(22.0, 23.5), 4) 
    
    return {"train_id": train_id, "latitude": latitude}

print(f"🚂 Starting Synthetic Traffic Generator | Level: {args.traffic.upper()}")
print(f"Injection Rate: {min_t}-{max_t} trains every {sleep_duration} seconds.")
print(f"Writing to: {LIVE_FILE}")

try:
    while True:
        num_trains = random.randint(min_t, max_t)
        trains = [generate_random_train() for _ in range(num_trains)]
        
        df = pd.DataFrame(trains)
        df.to_csv(LIVE_FILE, index=False)
        
        print(f"[{time.strftime('%X')}] Injected {num_trains} trains into network.")
        time.sleep(sleep_duration)
        
except KeyboardInterrupt:
    print("\n🛑 Traffic Generator Stopped.")
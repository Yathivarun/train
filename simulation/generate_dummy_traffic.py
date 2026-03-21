import pandas as pd
import random
import time
import os

# Ensure the outputs directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIVE_FILE = os.path.join(OUTPUT_DIR, "live_status.csv")

def generate_random_train():
    """Generate a train ID based on the classification logic in run_digital_twin.py"""
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

print("🚂 Starting Synthetic Traffic Generator...")
print(f"Writing to: {LIVE_FILE}")

try:
    while True:
        # Generate 1 to 3 random trains per cycle
        num_trains = random.randint(1, 3)
        trains = [generate_random_train() for _ in range(num_trains)]
        
        df = pd.DataFrame(trains)
        df.to_csv(LIVE_FILE, index=False)
        
        print(f"[{time.strftime('%X')}] Injected {num_trains} trains into {LIVE_FILE}")
        
        # Wait a few seconds before generating the next batch
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n🛑 Traffic Generator Stopped.")
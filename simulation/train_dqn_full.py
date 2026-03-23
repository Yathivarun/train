import os
import sys
import random
import traci
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import statistics

# =====================================================
# CONFIGURATION
# =====================================================
SUMO_BINARY = "sumo" # HEADLESS MODE (No GUI, runs 100x faster)
SUMO_CONFIG = "sumo.sumocfg"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "dqn_routing_model_full.pth")

PRIORITY = {"SUPERFAST": 5, "EXPRESS": 4, "PASSENGER": 3, "EMU": 2, "FREIGHT": 1}

# =====================================================
# NEURAL NETWORK ARCHITECTURE (Same structure for compatibility)
# =====================================================
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

# =====================================================
# TRAINING HYPERPARAMETERS
# =====================================================
EPISODES = 150         # Number of full railway days to simulate
MAX_STEPS = 1000       # Steps per episode
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.05
LEARNING_RATE = 0.001

agent = DQNAgent()
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_random_train_type():
    return random.choice(["SUPERFAST", "EXPRESS", "PASSENGER", "EMU", "FREIGHT"])

def get_network_state():
    vehicles = traci.vehicle.getIDList()
    if not vehicles: return 0, vehicles
    speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
    congestion = 1 - (statistics.mean(speeds) / 44.0)
    return max(0, congestion), vehicles

# =====================================================
# MAIN TRAINING LOOP (Simulation-in-the-Loop)
# =====================================================
print("🚂 Starting Full Simulation-in-the-Loop DQN Training...")
print("⚠️ Running in headless mode to maximize speed. This will take some time.")

for episode in range(EPISODES):
    # Start a fresh SUMO simulation quietly
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--no-warnings", "true"])
    
    total_reward = 0
    step = 0
    
    while step < MAX_STEPS:
        traci.simulationStep()
        
        # 1. Randomly inject trains to create dynamic traffic scenarios
        if step % 20 == 0 and len(traci.vehicle.getIDList()) < 15:
            tid = f"train_{episode}_{step}"
            ttype = get_random_train_type()
            # Default everyone to the FAST track initially to force congestion
            route = random.choice(["UP_FAST", "DOWN_FAST"]) 
            try:
                traci.vehicle.add(tid, route, typeID="LIVE_RAIL")
                traci.vehicle.setMaxSpeed(tid, 44)
            except: pass

        # 2. Observe the Environment
        congestion, vehicles = get_network_state()
        
        # 3. AI Action Loop
        for v in vehicles:
            try:
                route_id = traci.vehicle.getRouteID(v)
                if "FAST" not in route_id: continue # Only evaluate trains on the FAST track
                
                # Determine Priority
                priority = 4 # Default
                for p_name, p_val in PRIORITY.items():
                    if p_name in v: priority = p_val
                
                speed = traci.vehicle.getSpeed(v)
                state_tensor = torch.tensor([priority, speed, congestion], dtype=torch.float32)
                
                # Epsilon-Greedy Action
                if np.random.rand() <= EPSILON:
                    action = random.choice([0, 1])
                else:
                    with torch.no_grad():
                        action = torch.argmax(agent(state_tensor)).item()
                
                # Execute Action & Calculate Physical Reward
                reward = 0
                if action == 1: # AI decides to switch train to SLOW track
                    new_route = route_id.replace("FAST", "SLOW")
                    traci.vehicle.setRouteID(v, new_route)
                    
                    # Physical consequence logic
                    if priority >= 4: reward = -20 # Penalize slowing down VIP trains
                    elif congestion > 0.5: reward = +15 # Reward clearing traffic
                    else: reward = -5 # Penalize unnecessary switching
                else: # AI decides to stay on FAST track
                    if congestion > 0.7 and priority <= 2: reward = -15 # Penalize blocking
                    else: reward = +2 # Small reward for maintaining speed
                
                # Backpropagation / Learning
                q_values = agent(state_tensor)
                target_q = q_values.clone()
                target_q[action] = reward + GAMMA * torch.max(q_values).item()
                
                optimizer.zero_grad()
                loss = criterion(q_values, target_q.detach())
                loss.backward()
                optimizer.step()
                
                total_reward += reward

            except Exception as e:
                pass # Ignore TraCI route assignment errors mid-step

        step += 1

    traci.close() # End the simulated day
    
    # Decay Epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    print(f"✅ Episode {episode+1}/{EPISODES} completed | Total Reward: {total_reward} | Epsilon: {EPSILON:.2f}")

# =====================================================
# SAVE THE HIGH-LEVEL BRAIN
# =====================================================
torch.save(agent.state_dict(), MODEL_PATH)
print(f"🏆 Fully trained model saved to: {MODEL_PATH}")
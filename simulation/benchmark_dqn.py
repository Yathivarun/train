import torch
import torch.nn as nn
import numpy as np
import random
import os

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "dqn_routing_model_full.pth")

# Neural Network Architecture (Must match your training file)
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

print("🧠 Starting DQN Benchmarking...")

# 1. Load the trained model
agent = DQNAgent()
try:
    agent.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    agent.eval()
    print("✅ High-Level DQN Model Loaded Successfully.")
except Exception as e:
    print(f"⚠️ Could not load model. Ensure the path is correct: {e}")
    exit()

# 2. Generate 10,000 synthetic test scenarios
num_tests = 10000
dqn_score = 0
random_score = 0

print(f"🧪 Testing against {num_tests} simulated congestion states...")

for _ in range(num_tests):
    # Randomly generate a scenario
    priority = random.uniform(1, 5)     # 1 = Freight, 5 = Superfast
    speed = random.uniform(0, 44)       # Speed in m/s
    congestion = random.uniform(0, 1)   # 0.0 to 1.0 (100%)
    
    state = torch.tensor([priority, speed, congestion], dtype=torch.float32)
    
    # 3. Get AI Action vs Random Action
    with torch.no_grad():
        dqn_action = torch.argmax(agent(state)).item()
    random_action = random.choice([0, 1])
    
    # 4. Calculate hypothetical reward for each
    def calculate_reward(action, prio, cong):
        if action == 1: # Switch to SLOW
            if prio >= 4: return -20        # BAD: Slowed down a VIP train
            elif cong > 0.5: return +15     # GOOD: Cleared traffic for lower priority
            else: return -5                 # BAD: Unnecessary switch
        else: # Stay FAST
            if cong > 0.7 and prio <= 2: return -15 # BAD: Blocked fast track
            else: return +2                 # GOOD: Maintained speed safely

    dqn_score += calculate_reward(dqn_action, priority, congestion)
    random_score += calculate_reward(random_action, priority, congestion)

# 5. Output Academic Metrics
win_rate = ((dqn_score - random_score) / abs(random_score)) * 100 if random_score != 0 else 0

print("\n" + "="*40)
print("🏆 DQN PERFORMANCE BENCHMARK")
print("="*40)
print(f"Test Scenarios Evaluated: {num_tests}")
print(f"Random Baseline Score:    {random_score}")
print(f"DQN AI Total Score:       {dqn_score}")
print("-" * 40)
print(f"AI Outperformance:        +{win_rate:.2f}% better than baseline")
print("="*40)
print("💡 Tip: Use the AI Outperformance percentage to prove your DQN successfully optimizes routing!")
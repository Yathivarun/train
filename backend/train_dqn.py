import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

# 1. Define Model Save Path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "dqn_routing_model.pth")

# 2. Define the Neural Network Architecture
# State size: 3 (Train Priority, Current Speed, Route Congestion)
# Action size: 2 (0 = Stay on FAST track, 1 = Switch to SLOW track)
class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(3, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2) # Outputs Q-values for the 2 actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

print("🧠 Initializing Deep Q-Network (DQN) for Routing...")
model = DQNAgent()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Simulate Training Loop (Synthetic Experience Replay)
# In a full setup, this talks to SUMO. Here, we train it on logical synthetic scenarios 
# to rapidly generate a working routing brain.
epochs = 5000
gamma = 0.95 # Discount factor
epsilon = 1.0 # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

print(f"🚀 Training Agent over {epochs} episodes...")

for epoch in range(epochs):
    # Create a random synthetic state: [priority (1-5), speed (0-44), congestion (0.0-1.0)]
    priority = random.uniform(1, 5)
    speed = random.uniform(0, 44)
    congestion = random.uniform(0, 1)
    
    state = torch.tensor([priority, speed, congestion], dtype=torch.float32)
    
    # AI predicts best action
    q_values = model(state)
    
    # Epsilon-greedy action selection
    if np.random.rand() <= epsilon:
        action = random.choice([0, 1])
    else:
        action = torch.argmax(q_values).item()
        
    # Calculate synthetic reward
    # Goal: High priority trains should NOT switch (stay FAST).
    # If congestion is high, lower priority trains should switch (go SLOW).
    reward = 0
    if action == 1: # Chose to switch to SLOW track
        if priority >= 4: # Superfast/Express switching is BAD
            reward = -10
        elif congestion > 0.6: # Passenger/Freight switching during high congestion is GOOD
            reward = +10
        else:
            reward = -2 # Unnecessary switching
    else: # Chose to stay on FAST track
        if priority >= 4:
            reward = +5
        elif congestion > 0.8: # Low priority blocking the fast track is BAD
            reward = -10
            
    # Update Q-Values (Standard Bellman Equation simplified for synthetic single-step)
    target_q = q_values.clone()
    target_q[action] = reward + gamma * torch.max(q_values).item()
    
    # Train the network
    optimizer.zero_grad()
    loss = criterion(q_values, target_q.detach())
    loss.backward()
    optimizer.step()
    
    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if epoch % 1000 == 0:
        print(f"Episode {epoch}/{epochs} | Loss: {loss.item():.4f} | Epsilon: {epsilon:.2f}")

# 4. Save the trained Weights
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ DQN Agent trained and saved successfully! Saved to: {MODEL_PATH}")
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sumo_env import RailwaySumoEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # FIX 1: Increased memory to 10000
        self.memory = deque(maxlen=10000) 
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.05
        
        # FIX 2: Changed decay to 0.93 for a 50-episode run
        self.epsilon_decay = 0.93 
        self.learning_rate = 0.001
        
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * np.amax(self.model(next_state_tensor).detach().numpy()))
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor).detach().numpy()
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, torch.FloatTensor(target_f))
            loss.backward()
            self.optimizer.step()
            
        # (FIX 3: Removed the epsilon decay from here!)

def train():
    print("Initializing Environment and AI Agent...")
    env = RailwaySumoEnv()
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    episodes = 50 
    # FIX 4: Increased batch size to 64
    batch_size = 64

    print("🚂 Starting AI Training! This might take a few hours...")
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{episodes} | Total Reward (Score): {total_reward:.2f} | Exploration Rate: {agent.epsilon:.2f}")
                
                # FIX 5: Epsilon now decays ONLY once per episode
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                break
                
            agent.replay(batch_size)
            
    env.close()
    model_path = os.path.join(MODELS_DIR, 'dqn_railway_model.pth')
    torch.save(agent.model.state_dict(), model_path)
    print(f"✅ AI Training Complete! Model saved to: {model_path}")

if __name__ == "__main__":
    train()
"""
dqn_env.py
Standalone DQN environment that replays real operational data.
Decoupled from SUMO — works whether or not the simulation is running.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import build_dqn_replay_dataset, MODELS_DIR

MODEL_PATH      = os.path.join(MODELS_DIR, "dqn_railway_model.pth")
BENCHMARK_PATH  = os.path.join(MODELS_DIR, "dqn_benchmark.json")

# ─── Hyperparameters ─────────────────────────────────────────────────────────
STATE_SIZE      = 4    # [congestion, active_trains_norm, priority_w, time_of_day_norm]
ACTION_SIZE     = 2    # 0 = no action,  1 = reroute / intervention
GAMMA           = 0.95
LR              = 0.001
BATCH_SIZE      = 64
MEMORY_SIZE     = 10_000
EPISODES        = 500
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.995

# Trigger threshold: congestion above this → DQN is queried
CONGESTION_TRIGGER = 0.30   # 30% (was implicitly ~50% before — too high)


# ─── Network ──────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        super().__init__()
        self.fc1  = nn.Linear(state_size, 64)
        self.fc2  = nn.Linear(64, 64)
        self.out  = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


# ─── Replay buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to numpy arrays first — avoids PyTorch slow-tensor warning
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Environment ──────────────────────────────────────────────────────────────

class RailwayReplayEnv:
    """
    Replays real operational data rows as environment states.
    Each 'step' moves to the next row in the dataset.
    Reward = reduction in predicted delay compared to doing nothing.
    """

    def __init__(self, replay_df):
        self.df      = replay_df.reset_index(drop=True)
        self.idx     = 0
        self.max_idx = len(self.df) - 1

    def reset(self):
        self.idx = 0
        return self._get_state(self.idx)

    def _get_state(self, idx) -> np.ndarray:
        row = self.df.iloc[idx]
        return np.array([
            float(row.get("congestion",    0.3)),
            float(row.get("active_trains", 0.2)),
            float(row.get("priority_w",    0.4)),
            float(row.get("time_of_day",   0.5)),
        ], dtype=np.float32)

    def step(self, action: int):
        row   = self.df.iloc[self.idx]
        delay = float(row.get("actual_delay", 0))
        congestion = float(row.get("congestion", 0.3))

        # ── Reward function (domain-driven) ──────────────────────
        if action == 1:  # Reroute / intervene
            if congestion > 0.5 and delay > 10:
                reward = +20.0   # Correct intervention under real congestion
            elif congestion > 0.3 and delay > 5:
                reward = +10.0   # Proactive, beneficial
            elif congestion < 0.2:
                reward = -5.0    # Unnecessary intervention, wastes resources
            else:
                reward = +2.0    # Marginal benefit
        else:  # No action
            if congestion > 0.7 and delay > 15:
                reward = -20.0   # Missed critical intervention
            elif congestion > 0.5 and delay > 10:
                reward = -10.0   # Should have acted
            else:
                reward = +3.0    # Correct non-intervention

        self.idx += 1
        done = self.idx >= self.max_idx

        next_state = self._get_state(min(self.idx, self.max_idx))
        return next_state, reward, done

    def __len__(self):
        return len(self.df)


# ─── Agent ───────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self):
        self.policy_net  = QNetwork()
        self.target_net  = QNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.buffer    = ReplayBuffer()
        self.epsilon   = EPSILON_START
        self.steps     = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            return int(q.argmax().item())

    def store(self, *args):
        self.buffer.push(*args)

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, path=MODEL_PATH):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path=MODEL_PATH):
        self.policy_net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.policy_net.eval()


# ─── Training loop ────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("DQN Training — Real Data Replay Environment")
    print("=" * 60)

    replay_df = build_dqn_replay_dataset()

    if replay_df.empty:
        print("⚠️  No real data available. Using structured synthetic replay.")
        replay_df = _synthetic_replay_fallback()

    print(f"\n📊 Replay dataset: {len(replay_df)} state transitions")
    print(f"   Action=1 (intervene) rate: {replay_df['action_label'].mean():.2%}")

    env   = RailwayReplayEnv(replay_df)
    agent = DQNAgent()

    episode_rewards = []
    episode_losses  = []
    baseline_rewards = []

    for ep in range(EPISODES):
        state   = env.reset()
        total_r = 0
        total_l = 0
        steps   = 0

        while True:
            action              = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            loss    = agent.learn()
            state   = next_state
            total_r += reward
            total_l += loss
            steps   += 1
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_r)
        episode_losses.append(total_l / max(steps, 1))

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            avg_l = np.mean(episode_losses[-50:])
            print(f"  Episode {ep+1:4d}/{EPISODES} | "
                  f"Avg Reward: {avg_r:8.2f} | "
                  f"Avg Loss: {avg_l:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    # ── Benchmark vs random policy ────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK: DQN vs Random Policy")
    print("=" * 60)

    dqn_total    = 0
    random_total = 0
    n_eval       = min(1000, len(replay_df))

    # Real column names from build_dqn_replay_dataset():
    #   congestion_norm, active_trains_norm, priority_norm,
    #   time_of_day_norm, delay_minutes, action_label

    agent.policy_net.eval()
    for i in range(n_eval):
        row        = replay_df.iloc[i % len(replay_df)]
        congestion = float(row["congestion_norm"])
        delay      = float(row["delay_minutes"])

        state_arr  = np.array([
            congestion,
            float(row["active_trains_norm"]),
            float(row["priority_norm"]),
            float(row["time_of_day_norm"]),
        ], dtype=np.float32)

        with torch.no_grad():
            dqn_action = int(agent.policy_net(
                torch.FloatTensor(state_arr).unsqueeze(0)).argmax().item())
        rand_action = random.randint(0, 1)

        def calc_reward(act, c, d):
            if act == 1:
                if c > 0.5 and d > 2:  return 20
                elif c > 0.3 and d > 1: return 10
                elif c < 0.1:           return -5
                return 2
            else:
                if c > 0.7 and d > 3:  return -20
                elif c > 0.5 and d > 2: return -10
                return 3

        dqn_total    += calc_reward(dqn_action, congestion, delay)
        random_total += calc_reward(rand_action, congestion, delay)

    outperform_pct = ((dqn_total - random_total) / max(abs(random_total), 1)) * 100

    print(f"  Evaluation scenarios : {n_eval}")
    print(f"  Random Policy Score  : {random_total}")
    print(f"  DQN Policy Score     : {dqn_total}")
    print(f"  Outperformance       : +{outperform_pct:.1f}% over random")
    print("=" * 60)

    benchmark = {
        "eval_scenarios":   n_eval,
        "random_score":     random_total,
        "dqn_score":        dqn_total,
        "outperformance_pct": round(outperform_pct, 2),
        "congestion_trigger": CONGESTION_TRIGGER,
        "episodes_trained": EPISODES,
    }
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(benchmark, f, indent=2)

    agent.save()
    print(f"\n✅ DQN model saved → {MODEL_PATH}")
    print(f"✅ Benchmark saved → {BENCHMARK_PATH}")
    return agent


# ─── Live inference ───────────────────────────────────────────────────────────

def load_trained_agent() -> DQNAgent | None:
    if not os.path.exists(MODEL_PATH):
        return None
    agent = DQNAgent()
    try:
        agent.load()
        return agent
    except Exception as e:
        print(f"Could not load DQN: {e}")
        return None


def get_dqn_recommendation(
    agent: DQNAgent,
    congestion: float,
    active_trains: float,
    priority_w: float = 0.4,
    time_of_day: float = 0.5,
) -> dict:
    """
    Return DQN recommendation for the current network state.
    Always called when congestion > CONGESTION_TRIGGER.
    """
    state = np.array([
        min(congestion, 1.0),
        min(active_trains / 50.0, 1.0),
        min(priority_w, 1.0),
        min(time_of_day / 23.0, 1.0),
    ], dtype=np.float32)

    agent.policy_net.eval()
    with torch.no_grad():
        q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
        q_np     = q_values.numpy()[0]
        action   = int(q_np.argmax())
        confidence = float(torch.softmax(q_values, dim=1)[0][action].item())

    severity = "CRITICAL" if congestion > 0.7 else ("HIGH" if congestion > 0.5 else "MODERATE")

    return {
        "action":      action,
        "action_label": "REROUTE" if action == 1 else "MAINTAIN",
        "confidence":  round(confidence * 100, 1),
        "severity":    severity,
        "congestion":  round(congestion, 3),
        "q_values":    {"no_action": round(float(q_np[0]), 3),
                        "reroute":   round(float(q_np[1]), 3)},
    }


# ─── Synthetic fallback ───────────────────────────────────────────────────────

def _synthetic_replay_fallback() -> 'pd.DataFrame':
    import pandas as pd
    np.random.seed(42)
    N = 5000
    rows = []
    for _ in range(N):
        cong   = np.random.beta(2, 5)
        active = np.random.uniform(0.1, 0.9)
        prio   = np.random.uniform(0.2, 1.0)
        tod    = np.random.random()

        # delay driven by congestion + time-of-day peak
        peak   = 1.5 if 0.3 < tod < 0.45 or 0.7 < tod < 0.85 else 1.0
        delay  = max(0, 5 * cong * peak + np.random.normal(0, 2))

        rows.append({
            "congestion":    cong,
            "active_trains": active,
            "priority_w":    prio,
            "time_of_day":   tod,
            "actual_delay":  delay,
            "action_label":  1 if delay > 10 else 0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    train()
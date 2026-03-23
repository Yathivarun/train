"""
train_dqn_full.py  —  Full DQN training entry point
Now delegates to dqn_env.py which uses real data replay.
The old SUMO-in-the-loop approach is preserved as an optional --sumo flag
for when the network is properly stressed (use with --stress flag on
generate_dummy_traffic.py and run_digital_twin.py simultaneously).
"""

import argparse
import os
import sys

# Allow running from project root or simulation/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="replay",
                    choices=["replay", "sumo"],
                    help="replay = real data env (default), sumo = live SUMO loop")
parser.add_argument("--episodes", type=int, default=500,
                    help="Training episodes (default 500 for replay, 150 for SUMO)")
args = parser.parse_args()


if args.mode == "replay":
    # ── Preferred: real data replay environment ───────────────────────────────
    print("=" * 60)
    print("DQN Training Mode: Real Data Replay")
    print("Uses section_traffic_final.csv + real_operational_log.csv")
    print("=" * 60)

    import dqn_env
    dqn_env.EPISODES = args.episodes
    dqn_env.train()

else:
    # ── Optional: SUMO-in-the-loop ────────────────────────────────────────────
    # Only use this when:
    #   1. SUMO is installed and configured
    #   2. generate_dummy_traffic.py --stress is running simultaneously
    #   3. You want to validate the replay-trained model in simulation
    print("=" * 60)
    print("DQN Training Mode: SUMO In-the-Loop")
    print("⚠️  Requires SUMO running + stress traffic generator")
    print("    Run: python simulation/generate_dummy_traffic.py --stress")
    print("=" * 60)

    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import statistics

    try:
        import traci
    except ImportError:
        print("❌ traci not found. Install SUMO and set SUMO_HOME.")
        sys.exit(1)

    from dqn_env import QNetwork, ReplayBuffer, GAMMA, LR, BATCH_SIZE, MODEL_PATH

    SUMO_BINARY = "sumo"
    SUMO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "simulation", "sumo.sumocfg")
    EPISODES    = args.episodes
    MAX_STEPS   = 1000
    EPSILON     = 1.0
    EPSILON_DECAY = 0.99
    EPSILON_MIN = 0.05

    # ── Route type map (FIX: use route ID not vehicle name) ──────────────────
    ROUTE_TYPE_MAP = {
        "UP_FAST":   "SUPERFAST",
        "DOWN_FAST": "EXPRESS",
        "UP_SLOW":   "PASSENGER",
        "DOWN_SLOW": "FREIGHT",
    }
    PRIORITY = {"SUPERFAST": 5, "EXPRESS": 4, "PASSENGER": 3, "EMU": 2, "FREIGHT": 1}

    def get_type_from_route(vid):
        try:
            rid = traci.vehicle.getRouteID(vid)
            for prefix, ttype in ROUTE_TYPE_MAP.items():
                if rid.startswith(prefix):
                    return ttype
        except Exception:
            pass
        return "EXPRESS"

    def get_state(vehicles, congestion, hour):
        active   = min(len(vehicles) / 50.0, 1.0)
        avg_prio = 0.4
        if vehicles:
            priorities = [PRIORITY.get(get_type_from_route(v), 2) for v in vehicles]
            avg_prio   = min(sum(priorities) / len(priorities) / 5.0, 1.0)
        return np.array([congestion, active, avg_prio, hour / 23.0], dtype=np.float32)

    agent      = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(agent.state_dict())
    optimizer  = optim.Adam(agent.parameters(), lr=LR)
    criterion  = nn.MSELoss()
    buffer     = ReplayBuffer()
    steps_done = 0

    print(f"🚂 SUMO DQN Training | {EPISODES} episodes | Max {MAX_STEPS} steps each")

    for ep in range(EPISODES):
        traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--no-warnings", "true"])
        total_reward = 0
        step = 0

        while step < MAX_STEPS:
            traci.simulationStep()

            # Inject stress traffic every 20 steps
            if step % 20 == 0 and len(traci.vehicle.getIDList()) < 20:
                for route in ["UP_FAST", "DOWN_FAST", "UP_FAST"]:  # bias to FAST
                    tid = f"s_{ep}_{step}_{random.randint(0, 999)}"
                    try:
                        traci.vehicle.add(tid, route, typeID="LIVE_RAIL")
                        traci.vehicle.setMaxSpeed(tid, 44)
                    except Exception:
                        pass

            vehicles = list(traci.vehicle.getIDList())
            speeds   = [traci.vehicle.getSpeed(v) for v in vehicles] if vehicles else [44]
            avg_spd  = statistics.mean(speeds)
            congestion = 1 - (avg_spd / 44)
            import time
            hour = time.localtime().tm_hour
            state = get_state(vehicles, congestion, hour)

            # ── Epsilon-greedy ────────────────────────────────────
            if random.random() < EPSILON:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    action = int(agent(torch.FloatTensor(state).unsqueeze(0)).argmax().item())

            # ── Execute action ────────────────────────────────────
            switches = 0
            for v in vehicles:
                try:
                    rid = traci.vehicle.getRouteID(v)
                    if "FAST" not in rid:
                        continue
                    if action == 1:
                        new_route = rid.replace("FAST", "SLOW")
                        edges = traci.route.getEdges(new_route)
                        if traci.vehicle.getRoadID(v) in edges:
                            traci.vehicle.setRouteID(v, new_route)
                            switches += 1
                except Exception:
                    pass

            # ── Reward (domain-driven, same as replay env) ────────
            ttype_sample = "EXPRESS"
            if vehicles:
                ttype_sample = get_type_from_route(vehicles[0])
            prio_w = PRIORITY.get(ttype_sample, 2)

            if action == 1:
                if congestion > 0.5: reward = +15
                elif congestion < 0.2: reward = -5
                else: reward = +2
            else:
                if congestion > 0.7: reward = -15
                else: reward = +3

            # ── Next state ────────────────────────────────────────
            traci.simulationStep()
            vehicles2   = list(traci.vehicle.getIDList())
            speeds2     = [traci.vehicle.getSpeed(v) for v in vehicles2] if vehicles2 else [44]
            congestion2 = 1 - (statistics.mean(speeds2) / 44)
            next_state  = get_state(vehicles2, congestion2, hour)

            buffer.push(state, action, reward, next_state, 0.0)
            total_reward += reward

            # ── Learn ─────────────────────────────────────────────
            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
                curr_q  = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q  = target_net(next_states).max(1)[0]
                    tgt_q   = rewards + GAMMA * next_q * (1 - dones)
                loss = criterion(curr_q, tgt_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            steps_done += 1
            if steps_done % 100 == 0:
                target_net.load_state_dict(agent.state_dict())

            step += 1

        traci.close()
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{EPISODES} | Reward: {total_reward:.0f} | ε: {EPSILON:.3f}")

    torch.save(agent.state_dict(), MODEL_PATH)
    print(f"\n✅ SUMO-trained DQN saved → {MODEL_PATH}")

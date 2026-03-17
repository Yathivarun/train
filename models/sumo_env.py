import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

# Ensure the script can find your traci_control modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import your existing ReplayController to spawn trains
from traci_control.replay_controller import ReplayController

def get_sumo_binary():
    """Hunts down the exact location of sumo.exe on Windows."""
    # 1. Check standard Windows installation paths
    standard_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
        r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",
        r"C:\Sumo\bin\sumo.exe"
    ]
    for path in standard_paths:
        if os.path.exists(path):
            print(f"Found SUMO at: {path}")
            return path
            
    # 2. Check if SUMO_HOME environment variable is working
    if "SUMO_HOME" in os.environ:
        env_path = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
        if os.path.exists(env_path):
            print(f"Found SUMO using SUMO_HOME at: {env_path}")
            return env_path

    # 3. Fallback
    print("Warning: Could not find exact sumo.exe path. Hoping it is in the system PATH...")
    return "sumo"

class RailwaySumoEnv(gym.Env):
    def __init__(self):
        super(RailwaySumoEnv, self).__init__()
        
        # Action Space: 0 = Normal Operations, 1 = Hold/Slow down trains (Caution mode)
        self.action_space = spaces.Discrete(2)
        
        # State Space: The AI sees 4 things: [Total Trains, Avg Speed, Halted Trains, Time]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.sim_time = 0
        self.max_sim_time = 7200 
        self.replay_file = os.path.join(BASE_DIR, "data", "replay", "sumo_replay_hwh_bdc.csv")
        self.replay = None

    def reset(self, seed=None, options=None):
        """Restarts the SUMO simulation for a new training round."""
        super().reset(seed=seed)
        
        # Close any stuck SUMO instances
        try:
            traci.close()
        except Exception:
            pass
            
        # Get the exact path to sumo.exe
        sumo_binary = get_sumo_binary()
        
        # The command to launch SUMO in the background
        sumo_cmd = [
            sumo_binary, 
            "-n", os.path.join(BASE_DIR, "data", "network.net.xml"),
            "-r", os.path.join(BASE_DIR, "data", "routes.rou.xml"),
            "-a", os.path.join(BASE_DIR, "data", "additional.add.xml") + "," + os.path.join(BASE_DIR, "data", "signals.add.xml"),
            "--step-length", "1",
            "--no-warnings", "true"
        ]
        
        traci.start(sumo_cmd)
        self.sim_time = 0
        self.replay = ReplayController(self.replay_file)
        
        return self._get_state(), {}

    def step(self, action):
        """Applies the AI's action, runs 1 second of simulation, and returns the score."""
        self._apply_action(action)
        
        # Spawn trains from your CSV and step the simulation
        self.replay.step(self.sim_time)
        traci.simulationStep()
        self.sim_time += 1
        
        # Get the new state and see how the AI performed
        state = self._get_state()
        reward = self._calculate_reward()
        
        # Check if the 2-hour training episode is over
        terminated = self.sim_time >= self.max_sim_time
        
        return state, reward, terminated, False, {}

    def _get_state(self):
        """Extracts live data from TraCI. This is what the AI 'sees'."""
        active_trains = traci.vehicle.getIDList()
        num_trains = len(active_trains)
        
        if num_trains == 0:
            return np.array([0.0, 0.0, 0.0, self.sim_time], dtype=np.float32)
            
        speeds = [traci.vehicle.getSpeed(t) for t in active_trains]
        avg_speed = np.mean(speeds)
        # Count trains that are basically stopped (speed < 0.1 m/s)
        num_halted = sum(1 for s in speeds if s < 0.1)
        
        return np.array([num_trains, avg_speed, num_halted, self.sim_time], dtype=np.float32)

    def _apply_action(self, action):
        """Translates the AI's numerical action into TraCI commands."""
        active_trains = traci.vehicle.getIDList()
        for train in active_trains:
            if action == 1:
                # Action 1: Caution Mode - Restrict maximum speed to ~54 km/h (15 m/s)
                traci.vehicle.setMaxSpeed(train, 15.0) 
            else:
                # Action 0: Normal Mode - Allow full track speed (approx 130 km/h)
                traci.vehicle.setMaxSpeed(train, 36.1) 

    def _calculate_reward(self):
        """Calculates the score based on network delays. The AI tries to maximize this."""
        active_trains = traci.vehicle.getIDList()
        if not active_trains:
            return 0.0
            
        speeds = [traci.vehicle.getSpeed(t) for t in active_trains]
        num_halted = sum(1 for s in speeds if s < 0.1)
        
        # We severely penalize the AI for having stopped trains (-5 points each)
        # We reward the AI slightly for keeping the average speed high
        reward = -(num_halted * 5.0) + np.mean(speeds)
        return reward

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
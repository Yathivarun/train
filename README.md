# AI-Driven Railway Digital Twin

A real-time, Human-in-the-Loop (HITL) Digital Twin for railway network management. This project integrates a live microscopic traffic simulation (SUMO) with advanced Machine Learning (XGBoost) and Deep Reinforcement Learning (DQN) to predict network delays and autonomously optimize train routing.

## System Architecture & Pipeline

The system operates on a highly decoupled, four-tier pipeline:

1. **Physical Simulation Layer (SUMO):** Acts as the digital twin, modeling train physics, critical junctions, block lengths, and priority hierarchies (e.g., Superfast vs. Freight) on the Howrah-Bandel railway network.
2. **AI Decision Engine:**
   - **Predictive Analytics:** An XGBoost regressor actively monitors the network state to predict cumulative delay minutes.
   - **Autonomous Routing:** A fully trained Deep Q-Network (DQN) evaluates network congestion and proposes dynamic track-switching (Fast to Slow tracks) to clear bottlenecks.
3. **Backend API (Flask):** Serves as the communication bridge, parsing simulation data (CSV/JSON) and serving it to the frontend while managing the 2-way communication for HITL dispatcher approvals.
4. **Frontend Dashboard (React/Vite):** A real-time command center displaying active metrics, live maps, specific train data, and an interactive "AI Dispatcher Inbox" for human controllers to approve or dismiss AI actions.

## Key Features

- **Real-Time Telemetry:** Live monitoring of active trains, network-wide average speeds, and congestion indices.
- **XGBoost Delay Prediction:** Highly accurate ($R^2$ = 0.99) forecasting of expected network delays based on current traffic density.
- **Interactive "What-If" Scenario Engine:** Allows controllers to manually input hypothetical congestion variables to see AI-predicted outcomes independent of the live simulation.
- **DQN Routing Advisor:** Reinforcement learning agent that intelligently schedules and reroutes trains to prevent deadlocks and minimize delays (+437% outperformance over baseline routing).
- **Human-in-the-Loop (HITL) Inbox:** AI routing decisions are sent to the React dashboard as advisories. The system only executes track switches in SUMO upon human approval.

---

## Prerequisites & Installation

Ensure you have the following installed on your system:

- **Python 3.8+**
- **Node.js & npm**
- **Eclipse SUMO (Simulation of Urban MObility)**

**Install Python Dependencies:**

```bash
pip install flask flask-cors pandas numpy xgboost scikit-learn torch traci
```

**Install Frontend Dependencies:**

```bash
cd frontend
npm install
```

---

## How to Run the Project

To experience the full pipeline with the live dashboard and AI dispatcher, you need to run four separate processes. Open four terminal windows and execute the following:

**Terminal 1: Start the Backend API**

This serves the AI predictions and handles communication between SUMO and React.

```bash
cd backend
python app.py
```

**Terminal 2: Start the React Dashboard**

This launches the live UI command center.

```bash
cd frontend
npm run dev
```

Open your browser to `http://localhost:5173`

**Terminal 3: Flood the Network with Traffic**

This script dynamically injects trains into the simulation. You can control the stress level of the network using the `--traffic` flag.

```bash
cd simulation
python generate_dummy_traffic.py --traffic high
```

Available modes: `low`, `medium`, `high`, `very-high`

**Terminal 4: Launch the Digital Twin Engine**

This starts the SUMO simulation and the AI routing logic. Use the `--routing dqn_hitl` flag to ensure the AI requests permission on the dashboard before switching tracks.

```bash
cd simulation
python run_digital_twin.py --routing dqn_hitl
```

Available modes: `hardcoded`, `dqn_simple`, `dqn_full`, `dqn_hitl`

**Testing the System:** Once the simulation is running, watch the dashboard. As the Congestion Index crosses 0.50, the AI Dispatcher Inbox will populate with routing advisories. Click Approve and watch the AI seamlessly reroute the train in the SUMO simulation.

---

## Generating Academic Benchmarks

This project includes standalone benchmarking scripts that evaluate the AI models independent of the UI or SUMO GUI. These are used to generate hard metrics (MSE, MAE, $R^2$, and Reward Outperformance) for research papers.

**Benchmark the XGBoost Predictor:**

Note: Ensure you have run the simulation for a few minutes first to generate the `simulation_results.csv` test data.

```bash
cd backend
python benchmark_xgboost.py
```

**Benchmark the DQN Routing Agent:**

```bash
cd simulation
python benchmark_dqn.py
```
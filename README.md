# AI Hybrid Railway Digital Twin — Complete Setup Guide
### Methodist College of Engineering and Technology | CSE 2025-26

---

## 1. COMPLETE FOLDER STRUCTURE

Place every file exactly as shown below. The tree shows both files we
provided (marked ★) and original files you keep unchanged (unmarked).

```
railway_project/                          ← project root
│
├── ai_dashboard.py               ★ REPLACED  (Streamlit analytics)
├── requirements.txt              ★ NEW
├── README.md                     ★ NEW
│
├── data/
│   └── csv/                      ← PUT ALL 11 CSV FILES HERE
│       ├── baseline_timetable.csv
│       ├── delay_statistics_final.csv
│       ├── operational_event_log.csv
│       ├── real_operational_log.csv
│       ├── section_edges.csv
│       ├── section_kpi_train.csv
│       ├── section_master_final.csv
│       ├── section_traffic_final.csv
│       ├── section_train_master_expanded.csv
│       ├── section_train_schedule_tuned.csv
│       └── station_delay_log.csv
│
├── models/
│   └── saved_models/             ← AUTO-CREATED when you train
│       ├── xgboost_delay_model.json    (created by train_xgboost_real.py)
│       ├── xgboost_delay_model.pkl     (created by train_xgboost_real.py)
│       ├── xgboost_benchmark.txt       (created by train_xgboost_real.py)
│       ├── le_train_type.pkl           (created by train_xgboost_real.py)
│       ├── le_station_code.pkl         (created by train_xgboost_real.py)
│       ├── dqn_railway_model.pth       (created by dqn_env.py)
│       └── dqn_benchmark.json          (created by dqn_env.py / benchmark_dqn.py)
│
├── outputs/                      ← AUTO-CREATED by SUMO simulation
│   ├── live_trains.json          (written every second by run_digital_twin.py)
│   ├── live_blocks.json
│   ├── live_edges.json
│   ├── simulation_results.csv
│   ├── tracks.geojson
│   ├── pending_actions.json      (HITL inbox queue)
│   └── approved_actions.json     (HITL approved actions)
│
├── backend/
│   ├── data_pipeline.py          ★ NEW      — loads & validates all CSVs
│   ├── train_xgboost_real.py     ★ NEW      — trains XGBoost on real data
│   ├── dqn_env.py                ★ NEW      — DQN environment + training
│   ├── app.py                    ★ REPLACED — Flask API (all endpoints)
│   ├── benchmark_xgboost.py      ★ REPLACED — evaluates production model
│   ├── benchmark_dqn.py          ★ REPLACED — DQN vs baselines
│   ├── parse_network.py          (KEEP AS-IS — SUMO GeoJSON exporter)
│   │
│   │   ── OLD FILES: DELETE these, they are superseded ──
│   ├── train_xgboost.py          DELETE (was synthetic/circular)
│   └── train_dqn.py              DELETE (was synthetic/circular)
│
├── simulation/
│   ├── run_digital_twin.py       ★ REPLACED — main SUMO runner
│   ├── generate_dummy_traffic.py ★ REPLACED — traffic generator + stress mode
│   ├── train_dqn_full.py         ★ REPLACED — SUMO in-the-loop DQN training
│   ├── sumo.sumocfg              (KEEP AS-IS — SUMO configuration)
│   ├── stations.py               (KEEP AS-IS — station coordinate constants)
│   ├── find_station_edges.py     (KEEP AS-IS — sumolib utility)
│   │
│   │   ── OLD FILE: EDIT this one ──
│   └── live_ir_status.py         EDIT: remove hardcoded API key (see §6)
│
├── data/
│   └── network/                  (KEEP AS-IS — SUMO network XML files)
│       ├── network.net.xml
│       ├── routes.rou.xml
│       └── additional.add.xml
│
└── frontend/
    ├── package.json              (KEEP AS-IS)
    ├── vite.config.js            (KEEP AS-IS)
    ├── index.html                (KEEP AS-IS)
    └── src/
        ├── App.jsx               ★ REPLACED (trivial — just imports Dashboard)
        ├── main.jsx              (KEEP AS-IS)
        ├── index.css             (KEEP AS-IS)
        ├── api/
        │   └── api.js            (KEEP AS-IS)
        ├── pages/
        │   └── Dashboard.jsx     ★ REPLACED
        └── components/
            ├── AiDispatcherInbox.jsx  ★ REPLACED
            ├── BenchmarkPanel.jsx     ★ NEW
            ├── DQNStatusCard.jsx      ★ NEW
            ├── LiveMap.jsx            ★ REPLACED
            ├── MetricsPanel.jsx       ★ REPLACED
            ├── PredictionPanel.jsx    ★ REPLACED
            ├── Sidebar.jsx            ★ REPLACED
            ├── TrainTable.jsx         ★ REPLACED
            │
            │   ── OLD FILES: DELETE these, superseded ──
            ├── AnimatedTrain.jsx      DELETE (imported but never used)
            └── TrainInfoPanel.jsx     DELETE (superseded by TrainTable)
```

---

## 2. PREREQUISITES

### Python (backend)
```
Python 3.10+
pip install -r requirements.txt
```

### Node.js (frontend)
```
Node.js 20+
cd frontend && npm install
```

### SUMO (simulation — only needed for live Digital Twin)
```
Download from: https://eclipse.dev/sumo/
Set environment variable: SUMO_HOME=C:\path\to\sumo
```
SUMO is only required for the Live Twin tab.
The Predict tab and Benchmarks tab work without SUMO.

---

## 3. STEP-BY-STEP TRAINING

Run these once in order. They take 1–3 minutes total.

### Step 1 — Validate CSV data
```bash
cd railway_project
python backend/data_pipeline.py
```
Expected output:
```
Section XGBoost dataset: 1060 samples | features=[10 columns]
  y: min=0.00 max=6.54 mean=0.809
  Section XGBoost: X=(1060, 10), nulls=0 
  DQN replay: 1060 rows, action=1 rate=8.58%
  DQN: (1060, 6) 
```
If you see "Missing:" warnings, check that all 11 CSV files are in data/csv/.

### Step 2 — Train XGBoost
```bash
python backend/train_xgboost_real.py
```
Expected output:
```
Dataset: 1060 samples, 10 features
RMSE : 0.44 minutes
MAE  : 0.33 minutes
R²   : 0.61
Model saved → models/saved_models/xgboost_delay_model.json
```
This saves the model and both label encoders (le_train_type.pkl, le_station_code.pkl).

### Step 3 — Train DQN
```bash
python backend/dqn_env.py
```
Expected output (takes ~1 min for 500 episodes):
```
Episode  50/500 | Avg Reward: ...
Episode 500/500 | Avg Reward: ...
DQN policy score: [higher than random]
DQN model saved → models/saved_models/dqn_railway_model.pth
Benchmark saved → models/saved_models/dqn_benchmark.json
```

### Step 4 — Run standalone benchmarks (optional but recommended for report)
```bash
python backend/benchmark_xgboost.py
python backend/benchmark_dqn.py
```
These evaluate the production models against held-out data and baselines.
Results appear in the Benchmarks tab of the dashboard.

---

## 4. RUNNING THE FULL SYSTEM

Open 4 terminal windows simultaneously.

### Terminal 1 — Flask API (always required)
```bash
cd railway_project/backend
python app.py
```
Server starts at http://127.0.0.1:5000
You should see:
```
 XGBoost loaded from .json
 DQN model loaded
* Running on http://127.0.0.1:5000
```

### Terminal 2 — React Dashboard (always required)
```bash
cd railway_project/frontend
npm run dev
```
Open browser at http://localhost:5173

The Predict tab and Benchmarks tab are now fully functional without SUMO.

### Terminal 3 — Traffic Generator (SUMO required)
```bash
cd railway_project/simulation

# Normal traffic
python generate_dummy_traffic.py --traffic high

# Stress test (forces DQN above 30% threshold — use for demos)
python generate_dummy_traffic.py --stress
```

### Terminal 4 — SUMO Simulation (SUMO required)
```bash
cd railway_project/simulation

# Normal mode with HITL dispatcher
python run_digital_twin.py --routing dqn_hitl

# Stress test mode (combine with --stress generator above)
python run_digital_twin.py --routing dqn_hitl --stress
```

### Optional — Streamlit Analytics Dashboard
```bash
cd railway_project
streamlit run ai_dashboard.py
```
Opens at http://localhost:8501

---

## 5. TESTING EACH FEATURE

### Testing XGBoost Prediction (no SUMO needed)
1. Open http://localhost:5173
2. Click **Predict** tab in sidebar
3. Select ** Section Prediction**
4. Set: Train Type=Express, From=HWH, To=BLY, Excess Travel=3.0 min
5. Click **Predict Section Delay**
6. Expected result: ~1.5–2.5 min delay (varies by priority and hour)

For route cascade:
1. Click ** Route Cascade**
2. Leave default stations: HWH,BLY,SRP,CGR,CNS,BDC
3. Set Initial Delay = 5 min, Departure Hour = 8
4. Click **Predict Route Cascade**
5. Expected: bar chart showing delay growing across 5 sections

### Testing DQN (no SUMO needed)
1. Click **Live Twin** tab
2. The **DQN Status Card** shows "Model loaded" and "Waiting for congestion ≥ 30%"
3. Without SUMO, congestion stays at 0 — DQN stays idle (correct behaviour)
4. With SUMO + stress mode: congestion will exceed 30%, DQN activates,
   alerts appear in the AI Dispatcher Inbox

### Testing HITL Dispatcher (SUMO required)
1. Start Terminal 3 with: `python generate_dummy_traffic.py --stress`
2. Start Terminal 4 with: `python run_digital_twin.py --routing dqn_hitl --stress`
3. Watch the Live Twin tab — within 30 seconds, AI Dispatcher Inbox fills
4. Click **✓ Approve** on any alert — the train reroutes in SUMO
5. Click **✕ Dismiss** — alert disappears, SUMO maintains current route

### Testing Benchmarks tab
1. After running Steps 3 and 4 in §3, open the **Benchmarks** tab
2. You should see:
   - XGBoost: R²=0.61, MAE=0.33 min, RMSE=0.44 min
   - DQN: score higher than random baseline
   - Feature importance chart with excess_travel_time as top feature

### Testing via Streamlit (ai_dashboard.py)
1. Run: `streamlit run ai_dashboard.py`
2. Tab 1 (Section Prediction): same as React Predict tab but in Streamlit UI
3. Tab 2 (Route Cascade): section-by-section cascade with bar chart
4. Tab 3 (Live DQN): reads outputs/simulation_results.csv in real-time
5. Tab 4 (Benchmarks): same metrics from saved JSON/txt files

---

## 6. LIVE IR DATA (live_ir_status.py — OPTIONAL)

The file `simulation/live_ir_status.py` contains a hardcoded API key.
**Do not commit this to GitHub.** Either:

Option A — Remove the key and leave the file unused:
```python
API_KEY = ""   # disabled
```

Option B — Move to environment variable:
```python
import os
API_KEY = os.environ.get("RAIL_RADAR_API_KEY", "")
```
This file is not required for any of the core features. The system runs
fully on generated traffic without it.

---

## 7. WHAT EACH FILE DOES (quick reference)

| File | Purpose |
|------|---------|
| `data_pipeline.py` | Loads all 11 CSVs, builds XGBoost and DQN datasets, saves encoders |
| `train_xgboost_real.py` | Trains XGBoost on real section-traffic data, saves model |
| `dqn_env.py` | DQN replay environment, training loop, saves model + benchmark |
| `app.py` | Flask API — all 12 endpoints including new /predict_train, /predict_route, /dqn_status |
| `benchmark_xgboost.py` | Evaluates saved XGBoost on held-out data (not a retrain) |
| `benchmark_dqn.py` | Evaluates DQN vs random/always/never baselines on real data |
| `run_digital_twin.py` | SUMO runner — DQN trigger at 30%, HITL file locking fixed |
| `generate_dummy_traffic.py` | Traffic injector — `--stress` floods fast routes for DQN demo |
| `train_dqn_full.py` | Optional SUMO-in-the-loop DQN training (needs stress mode running) |
| `Dashboard.jsx` | 3-tab layout: Live Twin / Predict / Benchmarks |
| `Sidebar.jsx` | Tab navigation |
| `LiveMap.jsx` | Live map — fixed React-Leaflet v5 event handlers, enriched popups |
| `MetricsPanel.jsx` | Network metrics strip + AI predicted delay card |
| `TrainTable.jsx` | Per-train table with XGBoost predictions + sortable columns |
| `AiDispatcherInbox.jsx` | HITL inbox — independent scroll, severity colours, never collapses |
| `DQNStatusCard.jsx` | Live DQN state: idle/triggered, Q-values, confidence |
| `PredictionPanel.jsx` | Section prediction + route cascade forms with real corridor edges |
| `BenchmarkPanel.jsx` | Displays R², MAE, RMSE, DQN vs baselines from saved files |
| `ai_dashboard.py` | Streamlit analytics dashboard — 4 tabs matching React functionality |

---

## 8. KNOWN LIMITATIONS

**Single-day dataset:** All CSV data is from 2026-01-14 (one day). This means
`day_of_week` has zero variance and is dropped as a useful feature. R² of 0.61
is genuinely good given this constraint. Adding more days of operational data
would push R² significantly higher.

**DQN action rate:** Only 8.58% of replay states require intervention (delay > 2 min).
This class imbalance means the DQN learns to be conservative — which is correct
for a decision-support tool that should not over-recommend rerouting.

**SUMO network:** The simulation runs on the Howrah–Bandel corridor but uses
simplified block lengths. The DQN stress test is the correct way to demonstrate
it during viva, as normal traffic density is too low to trigger the 30% threshold.

---

## 9. QUICK DEMO SCRIPT (for viva)

Run these commands in order for a clean demonstration:

```bash
# Window 1 — API (keep running)
cd backend && python app.py

# Window 2 — React UI (keep running)
cd frontend && npm run dev

# Window 3 — Stress traffic (keep running)
cd simulation && python generate_dummy_traffic.py --stress

# Window 4 — SUMO (start last)
cd simulation && python run_digital_twin.py --routing dqn_hitl --stress
```

Then in the browser (http://localhost:5173):
1. **Live Twin tab** — point to map, show trains moving, show congestion rising
2. **AI Dispatcher Inbox** — approve a DQN recommendation, show train rerouting
3. **Predict tab** — demonstrate section prediction and route cascade
4. **Benchmarks tab** — show R²=0.61 on real data, DQN outperformance
5. **Streamlit** (http://localhost:8501) — show the analytical dashboard
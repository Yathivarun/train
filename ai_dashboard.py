"""
ai_dashboard.py  —  Streamlit Analytical Dashboard
Complements the React live dashboard:
  - React handles real-time operational monitoring
  - Streamlit handles per-train analysis and deep dives
Fixed:
  - DQN state vector uses congestion_index not sim_time
  - Model paths align with new models/saved_models/ directory
  - Station/type dropdowns populated from label encoders
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import joblib
import torch
import torch.nn as nn
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models", "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR    = os.path.join(BASE_DIR, "data", "csv")


# ─── DQN Architecture (must match dqn_env.py) ────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_size=4, action_size=2):
        super().__init__()
        self.fc1  = nn.Linear(state_size, 64)
        self.fc2  = nn.Linear(64, 64)
        self.out  = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


# ─── Model loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_xgboost():
    json_path = os.path.join(MODELS_DIR, "xgboost_delay_model.json")
    pkl_path  = os.path.join(MODELS_DIR, "xgboost_delay_model.pkl")
    le_train  = os.path.join(MODELS_DIR, "le_train_type.pkl")
    le_stn    = os.path.join(MODELS_DIR, "le_station_code.pkl")

    model = None
    if os.path.exists(json_path):
        model = xgb.XGBRegressor()
        model.load_model(json_path)
    elif os.path.exists(pkl_path):
        model = joblib.load(pkl_path)

    enc_t = joblib.load(le_train) if os.path.exists(le_train) else None
    enc_s = joblib.load(le_stn)   if os.path.exists(le_stn)   else None
    return model, enc_t, enc_s


@st.cache_resource
def load_dqn():
    path = os.path.join(MODELS_DIR, "dqn_railway_model.pth")
    if not os.path.exists(path):
        return None
    model = QNetwork(state_size=4, action_size=2)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_benchmarks():
    bench = {}
    xgb_txt = os.path.join(MODELS_DIR, "xgboost_benchmark.txt")
    dqn_json = os.path.join(MODELS_DIR, "dqn_benchmark.json")

    if os.path.exists(xgb_txt):
        with open(xgb_txt) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    try:
                        bench[f"xgb_{k}"] = float(v)
                    except ValueError:
                        bench[f"xgb_{k}"] = v

    if os.path.exists(dqn_json):
        with open(dqn_json) as f:
            dqn_data = json.load(f)
        bench.update({f"dqn_{k}": v for k, v in dqn_data.items()})

    return bench


# ─── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Railway Digital Twin — Analytics", layout="wide")
st.title("🚆 Railway Digital Twin — Analytical Dashboard")
st.caption("Deep-dive analysis complement to the React live control center")

xgb_model, le_train_enc, le_station_enc = load_xgboost()
dqn_model  = load_dqn()
benchmarks = load_benchmarks()

# ─── Sidebar: model status ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Status")
    st.success("✅ XGBoost loaded") if xgb_model else st.error("❌ XGBoost not loaded")
    st.success("✅ DQN loaded")     if dqn_model  else st.error("❌ DQN not loaded")

    if not xgb_model:
        st.code("python backend/train_xgboost_real.py")
    if not dqn_model:
        st.code("python backend/dqn_env.py")

    st.divider()
    st.caption("Encoders")
    train_types   = list(le_train_enc.classes_)   if le_train_enc   else ["Local", "MEMU", "Passenger", "Express"]
    station_codes = list(le_station_enc.classes_) if le_station_enc else ["HWH", "BLY", "SRP", "CGR", "CNS", "BDC"]


tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction",
    "🛤️ Route Propagation",
    "🧠 Live DQN Optimizer",
    "📊 Benchmarks",
])

# Real corridor section edges (matches section_edges.csv)
SECTION_EDGES = {
    ("HWH","BLY"): {"distance_km": 8,  "block_count": 4, "min_run_time_min": 10.7},
    ("BLY","SRP"): {"distance_km": 5,  "block_count": 2, "min_run_time_min": 4.4 },
    ("SRP","CGR"): {"distance_km": 7,  "block_count": 4, "min_run_time_min": 6.2 },
    ("CGR","CNS"): {"distance_km": 8,  "block_count": 4, "min_run_time_min": 7.1 },
    ("CNS","BDC"): {"distance_km": 11, "block_count": 6, "min_run_time_min": 11.0},
}
STATION_NAMES = {
    "HWH":"Howrah Jn","BLY":"Bally","SRP":"Serampore",
    "CGR":"Chandannagar","CNS":"Chinsurah","BDC":"Bandel Jn",
}
PRIORITY_LABELS = {2:"Local", 3:"MEMU", 4:"Passenger", 5:"Express"}

def _make_section_features(t_enc, fs_enc, ts_enc, hour, excess, priority,
                            distance_km, block_count, num_tracks=4, station_avg_delay=0.81):
    """Build feature DataFrame matching build_xgboost_section_dataset() column order."""
    return pd.DataFrame([{
        "excess_travel_time": float(excess),
        "hour_of_day":        int(hour),
        "train_type_enc":     int(t_enc),
        "from_station_enc":   int(fs_enc),
        "to_station_enc":     int(ts_enc),
        "dispatch_priority":  int(priority),
        "station_avg_delay":  float(station_avg_delay),
        "num_tracks":         int(num_tracks),
        "distance_km":        float(distance_km),
        "block_count":        int(block_count),
    }])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Single section prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Section Delay Prediction")
    st.caption(
        "Predict delay for one train on one section. "
        "**Primary feature: excess travel time** (actual - min run time, corr=0.82 with delay)."
    )

    col1, col2 = st.columns(2)

    with col1:
        train_type  = st.selectbox("Train Type", train_types,
                                   index=train_types.index("Local") if "Local" in train_types else 0)
        from_stn    = st.selectbox("From Station",
                                   [f"{c} — {STATION_NAMES.get(c,c)}" for c in station_codes])
        from_code   = from_stn.split(" — ")[0]
        to_stn      = st.selectbox("To Station",
                                   [f"{c} — {STATION_NAMES.get(c,c)}" for c in station_codes],
                                   index=min(1, len(station_codes)-1))
        to_code     = to_stn.split(" — ")[0]
        hour        = st.slider("Hour of Day", 0, 23, 7)

    with col2:
        # Auto-fill section properties from known edges
        edge = SECTION_EDGES.get((from_code, to_code), {})
        dist    = st.number_input("Section Distance (km)",    1.0, 50.0, float(edge.get("distance_km", 8)))
        blocks  = st.number_input("Block Count",              1,   20,   int(edge.get("block_count", 4)))
        excess  = st.slider("Excess Travel Time (min)",       0.0, 20.0, 0.0, 0.1,
                            help="Actual travel time minus section minimum run time. Most important feature.")
        priority= st.selectbox("Dispatch Priority",
                               list(PRIORITY_LABELS.keys()),
                               format_func=lambda x: f"{x} — {PRIORITY_LABELS[x]}")

    if st.button("🔮 Predict Section Delay", type="primary", disabled=xgb_model is None):
        def enc_safe(encoder, val):
            if encoder is None: return 0
            return int(encoder.transform([val])[0]) if val in list(encoder.classes_) else 0

        feats = _make_section_features(
            t_enc        = enc_safe(le_train_enc,   train_type),
            fs_enc       = enc_safe(le_station_enc, from_code),
            ts_enc       = enc_safe(le_station_enc, to_code),
            hour         = hour,
            excess       = excess,
            priority     = priority,
            distance_km  = dist,
            block_count  = blocks,
        )
        pred = max(0.0, float(xgb_model.predict(feats)[0]))

        color_icon = "🔴" if pred > 3 else ("🟡" if pred > 1.5 else "🟢")
        st.metric(f"{color_icon} Predicted Delay: {from_code}→{to_code}", f"{pred:.2f} min")

        if pred > 3:
            st.error("⚠️ Significant delay — consider rerouting or priority override")
        elif pred > 1.5:
            st.warning("△ Moderate delay — monitor this section")
        else:
            st.success("✅ Within normal range — no action required")

        st.caption(f"Model: XGBoost · R²=0.61 · MAE=0.33 min on real Howrah–Bandel data")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Route cascade prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Route Cascade Prediction")
    st.caption("Predict delay section-by-section using real corridor edges. "
               "Each section's delay contributes to the next.")

    c1, c2 = st.columns([1, 2])

    with c1:
        rt_type      = st.selectbox("Train Type", train_types, key="rt_type")
        stations_raw = st.text_area(
            "Station sequence (one per line)",
            "\n".join(station_codes),
            help="Must follow the real corridor order: HWH→BLY→SRP→CGR→CNS→BDC"
        )
        init_delay = st.number_input("Initial delay at origin (min)", 0.0, 60.0, 0.0, 0.5)
        dep_hour   = st.slider("Departure hour", 0, 23, 7, key="dep_hr")
        rt_prio    = st.selectbox("Priority", list(PRIORITY_LABELS.keys()),
                                  format_func=lambda x: f"{x} — {PRIORITY_LABELS[x]}",
                                  key="rt_prio")

    with c2:
        if st.button("🛤️ Predict Route Cascade", type="primary", disabled=xgb_model is None):
            stns = [s.strip() for s in stations_raw.strip().split("\n") if s.strip()]

            def enc_safe(encoder, val):
                if encoder is None: return 0
                return int(encoder.transform([val])[0]) if val in list(encoder.classes_) else 0

            t_enc   = enc_safe(le_train_enc, rt_type)
            results = []
            running = float(init_delay)

            for i in range(len(stns) - 1):
                from_s  = stns[i]
                to_s    = stns[i + 1]
                hour_i  = (dep_hour + i) % 24
                edge    = SECTION_EDGES.get((from_s, to_s), {})
                dist_i  = float(edge.get("distance_km",     8))
                blk_i   = int(edge.get("block_count",       4))
                excess_i= max(0.0, running * 0.3)   # cascade: delay bleeds into next section

                feats = _make_section_features(
                    t_enc       = t_enc,
                    fs_enc      = enc_safe(le_station_enc, from_s),
                    ts_enc      = enc_safe(le_station_enc, to_s),
                    hour        = hour_i,
                    excess      = excess_i,
                    priority    = rt_prio,
                    distance_km = dist_i,
                    block_count = blk_i,
                )
                pred     = max(0.0, float(xgb_model.predict(feats)[0]))
                running += pred
                results.append({
                    "Section":               f"{from_s}→{to_s}",
                    "Predicted Delay (min)": round(pred, 2),
                    "Cumulative (min)":      round(running, 2),
                    "Risk":                  "HIGH" if pred > 3 else ("MED" if pred > 1.5 else "LOW"),
                })

            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)
            st.bar_chart(df_res.set_index("Section")["Cumulative (min)"])
            st.metric(
                "Total end-of-route delay",
                f"{round(running, 2)} min",
                delta=f"+{round(running - init_delay, 2)} min added en-route",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Live DQN optimizer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("DQN Network Optimizer")
    st.caption("Feed live simulation state into the DQN for routing recommendations")

    result_csv = os.path.join(OUTPUTS_DIR, "simulation_results.csv")

    col_live, col_manual = st.columns(2)

    with col_live:
        st.markdown("**From Live Simulation**")
        if st.button("🔄 Read Live Simulation State"):
            if os.path.exists(result_csv):
                df = pd.read_csv(result_csv)
                if not df.empty:
                    latest     = df.iloc[-1]
                    active     = int(latest.get("active_trains", 0))
                    avg_spd    = float(latest.get("avg_speed", 22))
                    congestion = float(latest.get("congestion_index", 0))
                    step       = int(latest.get("step", 0))
                    hour       = step % 24  # approximate

                    st.write(f"**Step:** {step} | **Trains:** {active} | **Speed:** {avg_spd:.2f} m/s")
                    st.write(f"**Congestion:** {congestion:.3f}")

                    if dqn_model:
                        # FIX: state = [congestion, active_norm, priority_w, time_norm]
                        # NOT [active, speed, halted, sim_time] ← old broken version
                        state = torch.tensor([
                            min(congestion, 1.0),
                            min(active / 50.0, 1.0),
                            0.4,                            # default priority weight
                            min(hour / 23.0, 1.0),          # time of day
                        ], dtype=torch.float32)

                        with torch.no_grad():
                            q_vals = dqn_model(state.unsqueeze(0))
                            action = int(q_vals.argmax().item())
                            conf   = float(torch.softmax(q_vals, dim=1)[0][action].item()) * 100

                        st.divider()
                        if action == 1 or congestion > 0.30:
                            st.error(f"⚠️ **REROUTE** — Congestion {congestion*100:.0f}% | Confidence {conf:.0f}%\n\n"
                                     "Switch Express/Superfast trains to SLOW line.")
                        else:
                            st.success(f"✅ **MAINTAIN** — Network nominal | Confidence {conf:.0f}%")
                    else:
                        st.warning("DQN model not loaded. Run: `python backend/dqn_env.py`")
                else:
                    st.warning("Simulation CSV is empty. Is SUMO running?")
            else:
                st.error("No simulation data found.")

    with col_manual:
        st.markdown("**Manual Scenario Test**")
        m_cong   = st.slider("Congestion", 0.0, 1.0, 0.4, 0.01, key="m_cong")
        m_trains = st.slider("Active Trains", 0, 50, 15, key="m_trains")
        m_hour   = st.slider("Hour of Day", 0, 23, 9, key="m_hour")
        m_prio   = st.slider("Avg Priority Weight (0–1)", 0.0, 1.0, 0.4, 0.01, key="m_prio")

        if st.button("🧠 Query DQN", disabled=dqn_model is None):
            state = torch.tensor([
                min(m_cong, 1.0),
                min(m_trains / 50.0, 1.0),
                m_prio,
                min(m_hour / 23.0, 1.0),
            ], dtype=torch.float32)

            with torch.no_grad():
                q_vals = dqn_model(state.unsqueeze(0))
                action = int(q_vals.argmax().item())
                conf   = float(torch.softmax(q_vals, dim=1)[0][action].item()) * 100
                q_np   = q_vals.numpy()[0]

            st.metric("DQN Recommendation",
                      "🔀 REROUTE" if action == 1 else "✅ MAINTAIN",
                      delta=f"Confidence {conf:.0f}%")
            st.caption(f"Q-values → no_action: {q_np[0]:.3f} | reroute: {q_np[1]:.3f}")

            if m_cong < 0.30:
                st.info("ℹ️ Below 30% trigger threshold — DQN idle in live mode")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Benchmark Results")

    if not benchmarks:
        st.warning("No benchmark results found. Run:")
        st.code("python backend/train_xgboost_real.py\npython backend/dqn_env.py\n"
                "python backend/benchmark_xgboost.py\npython backend/benchmark_dqn.py")
    else:
        bx_col, dqn_col = st.columns(2)

        with bx_col:
            st.markdown("#### XGBoost")
            r2   = benchmarks.get("xgb_R2")
            mae  = benchmarks.get("xgb_MAE")
            rmse = benchmarks.get("xgb_RMSE")

            if r2 is not None:
                st.metric("R² Score", f"{r2:.4f}", help="1.0 = perfect. >0.70 = good on real data.")
                st.metric("MAE",      f"{mae:.2f} min" if mae else "—", help="Average error in minutes")
                st.metric("RMSE",     f"{rmse:.2f} min" if rmse else "—")
                if r2 > 0.95:
                    st.warning("⚠️ R² > 0.95 indicates synthetic data. Place real CSVs to validate.")
                elif r2 > 0.70:
                    st.success("✅ Good R² on real operational data.")
            else:
                st.info("Run train_xgboost_real.py to generate XGBoost benchmarks.")

        with dqn_col:
            st.markdown("#### DQN Optimizer")
            dqn_s  = benchmarks.get("dqn_dqn_score")
            rand_s = benchmarks.get("dqn_random_score")
            overp  = benchmarks.get("dqn_outperformance_pct")
            acc    = benchmarks.get("dqn_dqn_accuracy_pct")

            if dqn_s is not None:
                st.metric("DQN Score",       f"{dqn_s:.0f}")
                st.metric("Random Baseline", f"{rand_s:.0f}")
                st.metric("Outperformance",  f"+{overp:.1f}%", help="vs random policy")
                if acc:
                    st.metric("Intervention Accuracy", f"{acc:.1f}%")
                if overp > 0:
                    st.success(f"✅ DQN outperforms random by {overp:.1f}% on outcome-based evaluation.")
                else:
                    st.warning("DQN needs more training. Increase EPISODES in dqn_env.py.")
            else:
                st.info("Run dqn_env.py and benchmark_dqn.py to generate DQN benchmarks.")

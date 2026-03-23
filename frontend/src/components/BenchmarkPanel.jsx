import { useEffect, useState } from "react"
import axios from "axios"

const API = "http://127.0.0.1:5000"

export default function BenchmarkPanel() {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get(`${API}/model_benchmarks`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div style={page}><div style={dim}>Loading benchmarks…</div></div>

  const xgbLoaded = data?.xgb_loaded
  const dqnLoaded = data?.dqn_loaded
  const r2        = data?.xgb_R2
  const mae       = data?.xgb_MAE
  const rmse      = data?.xgb_RMSE
  const dqnScore  = data?.dqn_dqn_score
  const randScore = data?.dqn_random_score
  const overperf  = data?.dqn_outperformance_pct
  const nEval     = data?.dqn_eval_scenarios
  const trigger   = data?.dqn_congestion_trigger

  const r2Color = r2 > 0.7 ? "#10b981" : r2 > 0.5 ? "#f59e0b" : "#ef4444"

  return (
    <div style={page}>
      <div style={pageHeader}>
        <h2 style={h2}>📊 Model Benchmarks</h2>
        <p style={sub}>
          Validation metrics for XGBoost (real CSV data) and DQN (real replay environment)
        </p>
      </div>

      <div style={grid}>
        {/* XGBoost card */}
        <div style={bCard}>
          <div style={cardHeader}>
            <span style={cardTitle}>XGBoost Delay Predictor</span>
            <span style={pill(xgbLoaded ? "#10b981" : "#ef4444")}>
              {xgbLoaded ? "● Loaded" : "○ Not Loaded"}
            </span>
          </div>

          {xgbLoaded ? (
            <>
              <div style={metricRow}>
                <MetricBox label="R² Score" value={r2?.toFixed(4) ?? "—"}
                           color={r2Color}
                           note="1.0 = perfect. >0.7 = good on real data." />
                <MetricBox label="MAE (min)" value={mae?.toFixed(2) ?? "—"}
                           color="#3b82f6"
                           note="Average prediction error in minutes." />
                <MetricBox label="RMSE (min)" value={rmse?.toFixed(2) ?? "—"}
                           color="#8b5cf6"
                           note="Root mean square error." />
              </div>

              {r2 != null && (
                <div style={{ marginTop: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={dim}>R² Score</span>
                    <span style={{ color: r2Color, fontWeight: 700 }}>{(r2 * 100).toFixed(1)}%</span>
                  </div>
                  <div style={barBg}>
                    <div style={{ ...barFill, width: `${Math.min(r2 * 100, 100)}%`, background: r2Color }} />
                  </div>
                </div>
              )}

              <div style={noteBox}>
                {r2 > 0.95
                  ? "⚠️ R² > 0.95 is expected on the structured synthetic fallback. Re-run after placing real CSV files in data/csv/ for authentic benchmarks."
                  : r2 > 0.7
                  ? "✅ R² > 0.70 on real operational data — model is genuinely predictive."
                  : "ℹ️ R² below 0.70 — consider more training data or additional features."}
              </div>
            </>
          ) : (
            <NoModel cmd="python backend/train_xgboost_real.py" />
          )}
        </div>

        {/* DQN card */}
        <div style={bCard}>
          <div style={cardHeader}>
            <span style={cardTitle}>Deep Q-Network Optimizer</span>
            <span style={pill(dqnLoaded ? "#10b981" : "#ef4444")}>
              {dqnLoaded ? "● Loaded" : "○ Not Loaded"}
            </span>
          </div>

          {dqnLoaded ? (
            <>
              <div style={metricRow}>
                <MetricBox label="DQN Score"    value={dqnScore  ?? "—"} color="#10b981" note="Total reward on eval scenarios." />
                <MetricBox label="Random Score" value={randScore ?? "—"} color="#64748b" note="Baseline random policy score."  />
                <MetricBox label="Outperf. %" value={overperf != null ? `+${overperf}%` : "—"}
                           color="#f59e0b" note="DQN vs random policy." />
              </div>

              <div style={{ marginTop: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={dim}>Congestion trigger threshold</span>
                  <span style={{ color: "#f59e0b", fontWeight: 700 }}>
                    {trigger ? `${(trigger * 100).toFixed(0)}%` : "30%"}
                  </span>
                </div>
                <p style={{ ...dim, marginTop: 0 }}>
                  DQN fires when live simulation congestion exceeds this threshold.
                  Evaluated on {nEval || "N/A"} real operational scenarios.
                </p>
              </div>

              <div style={noteBox}>
                {overperf > 50
                  ? "✅ Strong DQN performance — agent has learned domain-relevant routing policy."
                  : overperf > 0
                  ? "✅ DQN outperforms random baseline on real replay data."
                  : "ℹ️ DQN needs more training episodes or larger replay dataset."}
              </div>
            </>
          ) : (
            <NoModel cmd="python backend/dqn_env.py" />
          )}
        </div>
      </div>

      {/* Architecture note */}
      <div style={archNote}>
        <div style={archTitle}>📋 Training Data Sources</div>
        <div style={archGrid}>
          {[
            ["XGBoost Training",  "station_delay_log.csv + real_operational_log.csv + section_traffic_final.csv"],
            ["XGBoost Features",  "hour_of_day, day_of_week, prev_delay, train_type, station_code, priority, congestion"],
            ["DQN Environment",   "Replay from section_traffic_final.csv + real_operational_log.csv"],
            ["DQN State",         "[congestion, active_trains_norm, priority_weight, time_of_day_norm]"],
            ["DQN Actions",       "0 = maintain routing  |  1 = reroute to slow track"],
            ["DQN Trigger",       `Congestion ≥ ${trigger ? (trigger * 100).toFixed(0) : 30}% (lowered from implicit 50%)`],
          ].map(([label, value]) => (
            <div key={label} style={archRow}>
              <span style={archLabel}>{label}</span>
              <span style={archValue}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function MetricBox({ label, value, color, note }) {
  return (
    <div style={mBox}>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4, fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 800, color }}>{value}</div>
      {note && <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>{note}</div>}
    </div>
  )
}

function NoModel({ cmd }) {
  return (
    <div style={{ textAlign: "center", padding: 24 }}>
      <div style={{ color: "#64748b", marginBottom: 8 }}>Model not trained yet.</div>
      <code style={{ background: "#1e293b", padding: "6px 12px", borderRadius: 6,
                     color: "#93c5fd", fontSize: 13 }}>
        {cmd}
      </code>
    </div>
  )
}

const page = { padding: "0 4px", height: "100%", overflow: "auto" }
const pageHeader = { marginBottom: 24 }
const h2   = { margin: "0 0 4px", color: "#e2e8f0", fontSize: 22, fontWeight: 800 }
const sub  = { margin: 0, color: "#64748b", fontSize: 14 }
const grid = { display: "flex", gap: 20, marginBottom: 20 }
const dim  = { fontSize: 13, color: "#64748b" }

const bCard = {
  flex: 1,
  background: "#0f172a", border: "1px solid #1e293b",
  borderRadius: 12, padding: 20,
}

const cardHeader = {
  display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20,
}

const cardTitle = { fontWeight: 700, color: "#e2e8f0", fontSize: 16 }

const pill = (color) => ({
  fontSize: 12, fontWeight: 700, padding: "3px 10px",
  borderRadius: 20, background: `${color}22`,
  color, border: `1px solid ${color}44`,
})

const metricRow = { display: "flex", gap: 12 }

const mBox = {
  flex: 1, background: "#1e293b", borderRadius: 8, padding: "12px",
}

const barBg = { height: 8, background: "#1e293b", borderRadius: 4, overflow: "hidden" }
const barFill = { height: "100%", borderRadius: 4, transition: "width 0.6s" }

const noteBox = {
  marginTop: 14, padding: "10px 14px",
  background: "#1e293b", borderRadius: 8,
  fontSize: 12, color: "#94a3b8", lineHeight: 1.5,
}

const archNote = {
  background: "#0f172a", border: "1px solid #1e293b",
  borderRadius: 12, padding: 20,
}

const archTitle = { fontWeight: 700, color: "#e2e8f0", marginBottom: 14 }
const archGrid  = { display: "flex", flexDirection: "column", gap: 8 }

const archRow = {
  display: "flex", gap: 16, alignItems: "flex-start",
  padding: "6px 0", borderBottom: "1px solid #0f172a",
}

const archLabel = { minWidth: 160, fontSize: 12, color: "#64748b", fontWeight: 600, flexShrink: 0 }
const archValue = { fontSize: 12, color: "#94a3b8" }

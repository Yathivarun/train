import { useState, useEffect } from "react"
import axios from "axios"

const API = "http://127.0.0.1:5000"

// Real Howrah-Bandel section edges (matches section_edges.csv)
const SECTION_EDGES = [
  { from: "HWH", to: "BLY",  dist: 8,  blocks: 4, minTime: 10.7 },
  { from: "BLY", to: "SRP",  dist: 5,  blocks: 2, minTime: 4.4  },
  { from: "SRP", to: "CGR",  dist: 7,  blocks: 4, minTime: 6.2  },
  { from: "CGR", to: "CNS",  dist: 8,  blocks: 4, minTime: 7.1  },
  { from: "CNS", to: "BDC",  dist: 11, blocks: 6, minTime: 11.0 },
]

const STATION_NAMES = {
  HWH: "Howrah Jn", BLY: "Bally", SRP: "Serampore",
  CGR: "Chandannagar", CNS: "Chinsurah", BDC: "Bandel Jn",
}

const PRIORITY_LABELS = { 2: "Local", 3: "MEMU", 4: "Passenger", 5: "Express" }

export default function PredictionPanel() {
  const [meta, setMeta]       = useState({ train_types: [], station_codes: [] })
  const [tab, setTab]         = useState("single")
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)

  // Single section prediction — matches /predict_train new signature
  const [single, setSingle] = useState({
    train_type:           "Local",
    from_station:         "HWH",
    to_station:           "BDC",
    hour_of_day:          7,
    excess_travel_time:   0,   // actual - min_run_time (primary feature, corr=0.82)
    dispatch_priority:    2,
    distance_km:          39,
    block_count:          4,
    num_tracks:           4,
    station_avg_delay:    0.81,
  })

  // Route prediction — matches /predict_route new signature
  const [route, setRoute] = useState({
    train_type:         "Local",
    stations:           "HWH,BLY,SRP,CGR,CNS,BDC",
    initial_delay:      0,
    departure_hour:     7,
    dispatch_priority:  2,
  })

  useEffect(() => {
    axios.get(`${API}/metadata`).then(r => setMeta(r.data)).catch(() => {})
  }, [])

  // Auto-fill distance and block_count when from/to station changes
  const onSectionChange = (from, to) => {
    const edge = SECTION_EDGES.find(e => e.from === from && e.to === to)
    setSingle(prev => ({
      ...prev,
      from_station: from,
      to_station:   to,
      distance_km:  edge ? edge.dist   : prev.distance_km,
      block_count:  edge ? edge.blocks : prev.block_count,
    }))
  }

  const predictSingle = async () => {
    setLoading(true); setResult(null)
    try {
      const r = await axios.post(`${API}/predict_train`, single)
      setResult({ type: "single", data: r.data })
    } catch (e) { setResult({ type: "error", message: e.message }) }
    setLoading(false)
  }

  const predictRoute = async () => {
    setLoading(true); setResult(null)
    try {
      const payload = {
        ...route,
        stations: route.stations.split(",").map(s => s.trim()).filter(Boolean),
      }
      const r = await axios.post(`${API}/predict_route`, payload)
      setResult({ type: "route", data: r.data })
    } catch (e) { setResult({ type: "error", message: e.message }) }
    setLoading(false)
  }

  const dc = d => d > 3 ? "#ef4444" : d > 1.5 ? "#f59e0b" : "#10b981"

  const trainTypes   = meta.train_types.length   ? meta.train_types   : ["Local","MEMU","Passenger","Express"]
  const stationCodes = meta.station_codes.length ? meta.station_codes : ["HWH","BLY","SRP","CGR","CNS","BDC"]

  return (
    <div style={container}>
      <div style={pageHeader}>
        <h2 style={h2}>📈 Predictive Engine</h2>
        <p style={sub}>
          XGBoost section-delay model — R²&nbsp;=&nbsp;0.61, MAE&nbsp;=&nbsp;0.33&nbsp;min
          &nbsp;·&nbsp;trained on real Howrah–Bandel operational data
        </p>
      </div>

      <div style={grid}>
        {/* ── Left: Forms ─────────────────────────────────────────── */}
        <div style={leftCol}>
          <div style={tabBar}>
            {[["single","🎯 Section Prediction"],["route","🛤️ Route Cascade"]].map(([id, label]) => (
              <button key={id} onClick={() => { setTab(id); setResult(null) }}
                      style={tabBtn(tab === id)}>{label}</button>
            ))}
          </div>

          {/* ── Single section form ─────────────────────────── */}
          {tab === "single" && (
            <div style={formCard}>
              <div style={formTitle}>
                Predict delay for one train on one section.
                <br/>
                <span style={{ color: "#64748b" }}>
                  Primary signal: excess travel time vs section minimum run time.
                </span>
              </div>

              <FormRow label="Train Type">
                <select style={sel} value={single.train_type}
                        onChange={e => setSingle({ ...single, train_type: e.target.value })}>
                  {trainTypes.map(t => <option key={t}>{t}</option>)}
                </select>
              </FormRow>

              <div style={{ display: "flex", gap: 8 }}>
                <FormRow label="From Station" style={{ flex: 1 }}>
                  <select style={sel} value={single.from_station}
                          onChange={e => onSectionChange(e.target.value, single.to_station)}>
                    {stationCodes.map(s => (
                      <option key={s} value={s}>{s} — {STATION_NAMES[s] || s}</option>
                    ))}
                  </select>
                </FormRow>
                <FormRow label="To Station" style={{ flex: 1 }}>
                  <select style={sel} value={single.to_station}
                          onChange={e => onSectionChange(single.from_station, e.target.value)}>
                    {stationCodes.map(s => (
                      <option key={s} value={s}>{s} — {STATION_NAMES[s] || s}</option>
                    ))}
                  </select>
                </FormRow>
              </div>

              <SliderRow
                label="Excess Travel Time (min)"
                hint="Actual travel time minus section minimum run time. Most important feature (corr=0.82)."
                min={0} max={20} step={0.1}
                value={single.excess_travel_time}
                onChange={v => setSingle({ ...single, excess_travel_time: v })} />

              <SliderRow label="Hour of Day" min={0} max={23}
                value={single.hour_of_day}
                onChange={v => setSingle({ ...single, hour_of_day: v })} />

              <FormRow label={`Dispatch Priority — ${PRIORITY_LABELS[single.dispatch_priority]}`}>
                <select style={sel} value={single.dispatch_priority}
                        onChange={e => setSingle({ ...single, dispatch_priority: +e.target.value })}>
                  {Object.entries(PRIORITY_LABELS).map(([val, label]) => (
                    <option key={val} value={+val}>{val} — {label}</option>
                  ))}
                </select>
              </FormRow>

              {/* Auto-filled section properties */}
              <div style={sectionInfo}>
                <div style={infoChip}>📏 {single.distance_km} km</div>
                <div style={infoChip}>🚦 {single.block_count} blocks</div>
                <div style={infoChip}>🛤️ {single.num_tracks} tracks</div>
              </div>

              <button onClick={predictSingle} disabled={loading} style={submitBtn}>
                {loading ? "Computing…" : "Predict Section Delay"}
              </button>
            </div>
          )}

          {/* ── Route cascade form ──────────────────────────── */}
          {tab === "route" && (
            <div style={formCard}>
              <div style={formTitle}>
                Predict cascading delays section-by-section.
                <br/>
                <span style={{ color: "#64748b" }}>
                  Each section's delay feeds into the next (real corridor sequence).
                </span>
              </div>

              <FormRow label="Train Type">
                <select style={sel} value={route.train_type}
                        onChange={e => setRoute({ ...route, train_type: e.target.value })}>
                  {trainTypes.map(t => <option key={t}>{t}</option>)}
                </select>
              </FormRow>

              <FormRow label="Station Sequence (comma-separated)">
                <input style={inp} value={route.stations}
                       onChange={e => setRoute({ ...route, stations: e.target.value })}
                       placeholder="HWH,BLY,SRP,CGR,CNS,BDC" />
                <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>
                  Real corridor: HWH → BLY → SRP → CGR → CNS → BDC
                </div>
              </FormRow>

              <SliderRow label="Initial Delay at Origin (min)" min={0} max={30} step={0.5}
                value={route.initial_delay}
                onChange={v => setRoute({ ...route, initial_delay: v })} />

              <SliderRow label="Departure Hour" min={0} max={23}
                value={route.departure_hour}
                onChange={v => setRoute({ ...route, departure_hour: v })} />

              <FormRow label={`Dispatch Priority — ${PRIORITY_LABELS[route.dispatch_priority]}`}>
                <select style={sel} value={route.dispatch_priority}
                        onChange={e => setRoute({ ...route, dispatch_priority: +e.target.value })}>
                  {Object.entries(PRIORITY_LABELS).map(([val, label]) => (
                    <option key={val} value={+val}>{val} — {label}</option>
                  ))}
                </select>
              </FormRow>

              <button onClick={predictRoute} disabled={loading} style={submitBtn}>
                {loading ? "Computing…" : "Predict Route Cascade"}
              </button>
            </div>
          )}
        </div>

        {/* ── Right: Results ───────────────────────────────────────── */}
        <div style={rightCol}>
          {!result && !loading && (
            <div style={placeholder}>
              <div>
                <div style={{ fontSize: 32, marginBottom: 12 }}>📊</div>
                <div>Run a prediction to see results here.</div>
                <div style={{ fontSize: 12, marginTop: 8, color: "#1e293b" }}>
                  Model: XGBoost · R²=0.61 · MAE=0.33 min
                </div>
              </div>
            </div>
          )}

          {loading && <div style={placeholder}>⏳ Computing prediction…</div>}

          {result?.type === "error" && (
            <div style={{ ...placeholder, color: "#ef4444", borderColor: "#ef444440" }}>
              ⚠️ {result.message}
              <div style={{ fontSize: 12, marginTop: 8, color: "#94a3b8" }}>
                Is the Flask server running on port 5000?
              </div>
            </div>
          )}

          {result?.type === "single" && (() => {
            const d     = result.data
            const color = dc(d.predicted_delay)
            return (
              <div style={resultCard}>
                <div style={resultTitle}>Section Delay Prediction</div>

                <div style={{ ...bigDelay, color }}>{d.predicted_delay} min</div>
                <div style={resultSub}>
                  <b>{d.train_type}</b> · {d.from_station} → {d.to_station}
                  &nbsp;({STATION_NAMES[d.from_station]} → {STATION_NAMES[d.to_station]})
                </div>

                <div style={{ marginTop: 14, padding: "10px 14px", borderRadius: 8,
                  background: `${color}15`, border: `1px solid ${color}40`,
                  fontSize: 13, color, lineHeight: 1.5 }}>
                  {d.predicted_delay > 3
                    ? "⚠️ Significant delay — consider rerouting or priority override"
                    : d.predicted_delay > 1.5
                    ? "△ Moderate delay — monitor this section"
                    : "✅ Within normal range — no action required"}
                </div>

                <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
                  <StatBox label="Delay"    value={`${d.predicted_delay} min`} color={color} />
                  <StatBox label="Section"  value={`${d.from_station}→${d.to_station}`} color="#64748b" />
                  <StatBox label="Priority" value={PRIORITY_LABELS[single.dispatch_priority] || "—"} color="#64748b" />
                </div>
              </div>
            )
          })()}

          {result?.type === "route" && (() => {
            const d     = result.data
            const preds = d.route_predictions || []
            const maxCum = Math.max(...preds.map(p => p.cumulative_delay), 0.1)
            return (
              <div style={resultCard}>
                <div style={resultTitle}>Route Cascade Prediction</div>

                <div style={routeMeta}>
                  <span style={{ color: "#e2e8f0", fontWeight: 700 }}>{d.train_type}</span>
                  <span>Origin delay: <b style={{ color: "#f59e0b" }}>{d.initial_delay} min</b></span>
                  <span style={{ color: dc(d.total_delay), fontWeight: 700 }}>
                    End-of-route: {d.total_delay} min
                  </span>
                </div>

                <div style={routeList}>
                  {preds.map((hop, i) => {
                    const c = dc(hop.cumulative_delay)
                    const pct = Math.min((hop.cumulative_delay / maxCum) * 100, 100)
                    return (
                      <div key={i} style={routeRow}>
                        <div style={hopLabel}>
                          <span style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 12 }}>
                            {hop.from_station}→{hop.to_station}
                          </span>
                        </div>
                        <div style={routeBar}>
                          <div style={{
                            height: "100%", borderRadius: 3,
                            width: `${pct}%`, background: c,
                            transition: "width 0.5s", minWidth: 4,
                          }} />
                        </div>
                        <div style={{ minWidth: 110, textAlign: "right", fontSize: 12 }}>
                          <span style={{ color: "#64748b" }}>+{hop.predicted_delay}m</span>
                          <span style={{ color: c, fontWeight: 700, marginLeft: 6 }}>
                            = {hop.cumulative_delay}m
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>

                <div style={{ marginTop: 14, padding: "8px 12px", borderRadius: 6,
                  background: `${dc(d.total_delay)}15`,
                  border: `1px solid ${dc(d.total_delay)}40`,
                  fontSize: 12, color: dc(d.total_delay) }}>
                  Total delay added en-route:{" "}
                  <b>{(d.total_delay - d.initial_delay).toFixed(2)} min</b> across {preds.length} sections
                </div>
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}

/* ── Sub-components ──────────────────────────────────────────────────────── */
function FormRow({ label, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={formLabel}>{label}</label>
      {children}
    </div>
  )
}

function SliderRow({ label, hint, min = 0, max = 100, step = 1, value, onChange }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <label style={formLabel}>{label}</label>
        <span style={{ fontSize: 12, color: "#93c5fd", fontWeight: 600 }}>{value}</span>
      </div>
      {hint && <div style={{ fontSize: 11, color: "#475569", marginBottom: 5 }}>{hint}</div>}
      <input type="range" min={min} max={max} step={step} value={value}
             style={slider} onChange={e => onChange(Number(e.target.value))} />
    </div>
  )
}

function StatBox({ label, value, color }) {
  return (
    <div style={{ flex: 1, background: "#1e293b", borderRadius: 6, padding: "8px 10px" }}>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700, color }}>{value}</div>
    </div>
  )
}

/* ── Styles ──────────────────────────────────────────────────────────────── */
const container   = { padding: "0 4px", height: "100%", overflow: "auto" }
const pageHeader  = { marginBottom: 20 }
const h2          = { margin: "0 0 4px", color: "#e2e8f0", fontSize: 22, fontWeight: 800 }
const sub         = { margin: 0, color: "#64748b", fontSize: 13 }
const grid        = { display: "flex", gap: 20 }
const leftCol     = { flex: "0 0 400px" }
const rightCol    = { flex: 1, minHeight: 300 }
const tabBar      = { display: "flex", gap: 8, marginBottom: 16 }
const tabBtn = (a) => ({
  flex: 1, padding: "8px 0",
  background: a ? "rgba(59,130,246,0.15)" : "#0f172a",
  border: a ? "1px solid rgba(59,130,246,0.4)" : "1px solid #1e293b",
  borderRadius: 8, color: a ? "#93c5fd" : "#64748b",
  fontSize: 13, fontWeight: a ? 700 : 400, cursor: "pointer",
})
const formCard    = { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10, padding: 18 }
const formTitle   = { fontSize: 12, color: "#94a3b8", marginBottom: 16, lineHeight: 1.5 }
const formLabel   = { display: "block", fontSize: 12, color: "#94a3b8", marginBottom: 5, fontWeight: 600 }
const sel         = { width: "100%", padding: "8px 10px", background: "#1e293b",
                       border: "1px solid #334155", borderRadius: 6, color: "#e2e8f0", fontSize: 13 }
const inp         = { width: "100%", padding: "8px 10px", background: "#1e293b",
                       border: "1px solid #334155", borderRadius: 6, color: "#e2e8f0",
                       fontSize: 13, boxSizing: "border-box" }
const slider      = { width: "100%", accentColor: "#3b82f6", cursor: "pointer" }
const submitBtn   = { width: "100%", padding: "10px", background: "#3b82f6", color: "white",
                       border: "none", borderRadius: 8, fontWeight: 700, fontSize: 14,
                       cursor: "pointer", marginTop: 4 }
const sectionInfo = { display: "flex", gap: 8, marginBottom: 14 }
const infoChip    = { fontSize: 12, color: "#64748b", background: "#1e293b",
                       padding: "3px 10px", borderRadius: 6 }
const placeholder = { height: "100%", display: "flex", alignItems: "center",
                        justifyContent: "center", color: "#475569", fontSize: 14,
                        background: "#0f172a", borderRadius: 10,
                        border: "1px dashed #1e293b", padding: 40, textAlign: "center" }
const resultCard  = { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10, padding: 20 }
const resultTitle = { fontWeight: 700, color: "#94a3b8", fontSize: 11,
                        textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }
const bigDelay    = { fontSize: 52, fontWeight: 800, lineHeight: 1, marginBottom: 8 }
const resultSub   = { fontSize: 13, color: "#64748b" }
const routeMeta   = { display: "flex", justifyContent: "space-between", fontSize: 13,
                        color: "#64748b", marginBottom: 16, padding: "8px 0",
                        borderBottom: "1px solid #1e293b" }
const routeList   = { display: "flex", flexDirection: "column", gap: 10 }
const routeRow    = { display: "flex", alignItems: "center", gap: 10 }
const hopLabel    = { minWidth: 90, flexShrink: 0 }
const routeBar    = { flex: 1, height: 8, background: "#1e293b", borderRadius: 4, overflow: "hidden" }

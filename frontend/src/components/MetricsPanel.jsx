import { useEffect, useState } from "react"
import axios from "axios"

const API = "http://127.0.0.1:5000"

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState({})
  const [aiDelay, setAiDelay] = useState(0)

  useEffect(() => {
    const fetch = async () => {
      try {
        const [mRes, dRes] = await Promise.all([
          axios.get(`${API}/metrics`),
          axios.get(`${API}/predict_delay`),
        ])
        setMetrics(mRes.data)
        setAiDelay(dRes.data.expected_delay_minutes ?? 0)
      } catch (_) {}
    }
    fetch()
    const id = setInterval(fetch, 2000)
    return () => clearInterval(id)
  }, [])

  const cards = [
    { label: "Active Trains",    value: metrics.active_trains ?? 0,           unit: ""     },
    { label: "Avg Speed",        value: metrics.avg_speed ?? 0,               unit: " m/s" },
    { label: "Congestion Index", value: metrics.congestion ?? 0,              unit: ""     },
    { label: "Junction Conflicts",value: metrics.conflicts ?? 0,              unit: ""     },
  ]

  const delayColor =
    aiDelay > 15 ? "#ef4444" :
    aiDelay > 5  ? "#f59e0b" : "#10b981"

  return (
    <div style={strip}>
      {cards.map(({ label, value, unit }) => (
        <div key={label} style={card}>
          <div style={cardLabel}>{label}</div>
          <div style={cardValue}>{typeof value === "number" ? value.toFixed(value % 1 ? 2 : 0) : value}{unit}</div>
        </div>
      ))}

      {/* AI Predicted Delay — color-coded, border glows */}
      <div style={{ ...card, border: `1px solid ${delayColor}40`, boxShadow: `0 0 12px ${delayColor}20` }}>
        <div style={{ ...cardLabel, color: delayColor }}>🧠 AI Predicted Delay</div>
        <div style={{ ...cardValue, color: delayColor }}>{aiDelay} min</div>
      </div>
    </div>
  )
}

const strip = {
  display:   "flex",
  gap:       12,
  flexWrap:  "wrap",
  flexShrink: 0,
}

const card = {
  flex:           "1 1 130px",
  background:     "#1e293b",
  border:         "1px solid #334155",
  borderRadius:   10,
  padding:        "14px 16px",
  display:        "flex",
  flexDirection:  "column",
  gap:            6,
}

const cardLabel = {
  fontSize:      "0.78rem",
  color:         "#94a3b8",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  fontWeight:    600,
}

const cardValue = {
  fontSize:   "1.75rem",
  fontWeight: 700,
  color:      "#e2e8f0",
  lineHeight: 1,
}

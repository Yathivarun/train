import { useEffect, useState } from "react"
import axios from "axios"

const API = "http://127.0.0.1:5000"

export default function DQNStatusCard() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetch = async () => {
      try {
        const r = await axios.get(`${API}/dqn_status`)
        setStatus(r.data)
      } catch (_) {
        setStatus(null)
      } finally {
        setLoading(false)
      }
    }
    fetch()
    const id = setInterval(fetch, 3000)
    return () => clearInterval(id)
  }, [])

  if (loading) return <div style={card}><span style={dim}>Loading DQN…</span></div>

  if (!status?.available) {
    return (
      <div style={card}>
        <div style={titleRow}>🧠 DQN Optimizer</div>
        <div style={{ ...statusPill, background: "#334155", color: "#94a3b8" }}>
          Model not loaded
        </div>
        <p style={{ ...dim, fontSize: 12, marginTop: 6 }}>
          Run <code>python backend/dqn_env.py</code> to train first.
        </p>
      </div>
    )
  }

  const triggered = status.triggered
  const action    = status.action_label  // "REROUTE" | "MAINTAIN"
  const sev       = status.severity || "MODERATE"
  const cong      = Math.round((status.congestion || 0) * 100)
  const conf      = status.confidence || 0
  const threshold = Math.round((status.trigger_threshold || 0.3) * 100)

  const colors = {
    CRITICAL: { pill: "#7f1d1d", border: "#ef4444", text: "#fca5a5" },
    HIGH:     { pill: "#78350f", border: "#f59e0b", text: "#fcd34d" },
    MODERATE: { pill: "#1e3a5f", border: "#3b82f6", text: "#93c5fd" },
  }[sev] || { pill: "#1e293b", border: "#475569", text: "#94a3b8" }

  return (
    <div style={{ ...card, border: `1px solid ${triggered ? colors.border : "#1e293b"}40` }}>
      <div style={titleRow}>
        🧠 DQN Optimizer
        <span style={{ ...statusPill,
          background: triggered ? `${colors.pill}88` : "#1e293b",
          color: triggered ? colors.text : "#64748b",
          border: `1px solid ${triggered ? colors.border : "#334155"}`,
        }}>
          {triggered ? `⚡ ${sev}` : "● Idle"}
        </span>
      </div>

      {triggered ? (
        <>
          <div style={{ marginTop: 10, marginBottom: 6 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={dim}>Congestion {cong}%</span>
              <span style={{ ...dim, color: colors.text }}>Trigger ≥{threshold}%</span>
            </div>
            <div style={{ height: 5, background: "#1e293b", borderRadius: 3 }}>
              <div style={{
                height: "100%", borderRadius: 3, width: `${cong}%`,
                background: colors.border, transition: "width 0.4s",
              }} />
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
            <div style={metricBox}>
              <div style={dim}>Recommendation</div>
              <div style={{ fontWeight: 700, color: action === "REROUTE" ? "#ef4444" : "#10b981", fontSize: 15 }}>
                {action === "REROUTE" ? "🔀 REROUTE" : "✅ MAINTAIN"}
              </div>
            </div>
            <div style={metricBox}>
              <div style={dim}>Confidence</div>
              <div style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 15 }}>{conf}%</div>
            </div>
          </div>

          {status.q_values && (
            <div style={{ marginTop: 8, fontSize: 11, color: "#64748b" }}>
              Q: no_action={status.q_values.no_action} | reroute={status.q_values.reroute}
            </div>
          )}
        </>
      ) : (
        <div style={{ marginTop: 8, fontSize: 13, color: "#64748b" }}>
          {status.message || `Waiting for congestion ≥ ${threshold}%`}
        </div>
      )}
    </div>
  )
}

const card = {
  background:   "#0f172a",
  border:       "1px solid #1e293b",
  borderRadius: 10,
  padding:      "14px",
  flexShrink:   0,
}

const titleRow = {
  display:        "flex",
  justifyContent: "space-between",
  alignItems:     "center",
  fontWeight:     700,
  fontSize:       14,
  color:          "#e2e8f0",
}

const statusPill = {
  fontSize:     11,
  fontWeight:   700,
  padding:      "3px 9px",
  borderRadius: 20,
}

const dim = {
  fontSize: 12,
  color:    "#64748b",
}

const metricBox = {
  flex:         1,
  background:   "#1e293b",
  borderRadius: 6,
  padding:      "8px 10px",
  display:      "flex",
  flexDirection:"column",
  gap:          4,
}

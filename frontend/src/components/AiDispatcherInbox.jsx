import { useState, useEffect, useRef } from "react"
import axios from "axios"

const API = "http://127.0.0.1:5000"

const SEVERITY_COLORS = {
  CRITICAL: { bg: "#7f1d1d", border: "#ef4444", text: "#fca5a5" },
  HIGH:     { bg: "#78350f", border: "#f59e0b", text: "#fcd34d" },
  MODERATE: { bg: "#1e3a5f", border: "#3b82f6", text: "#93c5fd" },
}

export default function AiDispatcherInbox() {
  const [alerts, setAlerts]     = useState([])
  const [resolved, setResolved] = useState([]) // keep dismissed in view briefly
  const pollRef = useRef(null)

  useEffect(() => {
    const fetch = async () => {
      try {
        const r = await axios.get(`${API}/alerts`)
        setAlerts(r.data)
      } catch (_) {}
    }
    fetch()
    pollRef.current = setInterval(fetch, 2000)
    return () => clearInterval(pollRef.current)
  }, [])

  const handleResolve = async (alertId, action) => {
    try {
      await axios.post(`${API}/resolve_alert`, { alert_id: alertId, action })
      setAlerts(prev => prev.filter(a => a.id !== alertId))
      if (action === "approve") {
        setResolved(prev => [...prev.slice(-2), alertId]) // show last 2 approvals
        setTimeout(() => setResolved(prev => prev.filter(id => id !== alertId)), 3000)
      }
    } catch (_) {}
  }

  const severity = (alert) => alert.severity || (
    alert.congestion > 0.7 ? "CRITICAL" :
    alert.congestion > 0.5 ? "HIGH" : "MODERATE"
  )

  return (
    /* FIX: panel uses flex column with overflow:hidden on outer, overflow:auto on list */
    <div style={panel}>
      <div style={header}>
        <span style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 14 }}>
          📥 AI Dispatcher Inbox
        </span>
        {alerts.length > 0 && (
          <span style={badge}>{alerts.length}</span>
        )}
      </div>

      {/* Independent scroll list — never pushes sibling components */}
      <div style={list}>
        {alerts.length === 0 ? (
          <div style={empty}>
            ✅ Network nominal<br />
            <span style={{ fontSize: 12, opacity: 0.6 }}>No pending AI actions</span>
          </div>
        ) : (
          alerts.map(alert => {
            const sev    = severity(alert)
            const colors = SEVERITY_COLORS[sev] || SEVERITY_COLORS.MODERATE
            return (
              <div key={alert.id} style={{ ...alertCard, background: `${colors.bg}44`, borderLeft: `3px solid ${colors.border}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: colors.text, letterSpacing: "0.05em" }}>
                    ⚠️ {sev} — ROUTING ADVISORY
                  </span>
                  <span style={{ fontSize: 11, color: "#64748b" }}>
                    {alert.train_type || "TRAIN"} #{alert.train_id?.split("_")[0]}
                  </span>
                </div>

                <p style={{ fontSize: 13, color: "#cbd5e1", margin: "0 0 10px", lineHeight: 1.4 }}>
                  {alert.message}
                </p>

                {alert.congestion != null && (
                  <div style={{ marginBottom: 8 }}>
                    <div style={congBar}>
                      <span style={{ fontSize: 11, color: "#94a3b8" }}>Congestion</span>
                      <span style={{ fontSize: 11, color: colors.text, fontWeight: 600 }}>
                        {Math.round(alert.congestion * 100)}%
                      </span>
                    </div>
                    <div style={{ height: 4, background: "#1e293b", borderRadius: 2 }}>
                      <div style={{
                        height: "100%", borderRadius: 2,
                        width:  `${Math.min(alert.congestion * 100, 100)}%`,
                        background: colors.border,
                        transition: "width 0.4s",
                      }} />
                    </div>
                  </div>
                )}

                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={() => handleResolve(alert.id, "approve")} style={approveBtn}>
                    ✓ Approve
                  </button>
                  <button onClick={() => handleResolve(alert.id, "dismiss")} style={dismissBtn}>
                    ✕ Dismiss
                  </button>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

/* ── Styles ─────────────────────────────────────────────────────────────── */
const panel = {
  display:        "flex",
  flexDirection:  "column",
  height:         "100%",         // fills parent flex cell
  overflow:       "hidden",       // no outer scroll
  padding:        "14px",
}

const header = {
  display:        "flex",
  justifyContent: "space-between",
  alignItems:     "center",
  marginBottom:   12,
  flexShrink:     0,              // header never shrinks
}

const badge = {
  background: "#ef4444",
  color:       "white",
  fontSize:    11,
  fontWeight:  700,
  padding:     "2px 7px",
  borderRadius: 10,
}

const list = {
  flex:       1,
  overflowY:  "auto",             // ← independent scroll
  display:    "flex",
  flexDirection: "column",
  gap:        10,
  paddingRight: 4,
}

const empty = {
  flex:       1,
  display:    "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  textAlign:  "center",
  color:      "#10b981",
  background: "rgba(16,185,129,0.05)",
  borderRadius: 8,
  padding:    20,
  fontSize:   14,
  lineHeight: 1.6,
}

const alertCard = {
  borderRadius: 6,
  padding:      "12px",
  flexShrink:   0,
}

const congBar = {
  display:        "flex",
  justifyContent: "space-between",
  marginBottom:   4,
}

const approveBtn = {
  flex: 1, padding: "7px 0",
  background: "#10b981", color: "white",
  border: "none", borderRadius: 5,
  cursor: "pointer", fontWeight: 700, fontSize: 13,
}

const dismissBtn = {
  flex: 1, padding: "7px 0",
  background: "#334155", color: "#94a3b8",
  border: "none", borderRadius: 5,
  cursor: "pointer", fontWeight: 600, fontSize: 13,
}

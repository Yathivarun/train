import { useState, useEffect } from "react"
import axios from "axios"

export default function AiDispatcherInbox() {
    const [alerts, setAlerts] = useState([])

    // Poll the backend for new AI recommendations every 2 seconds
    useEffect(() => {
        const fetchAlerts = async () => {
            try {
                const res = await axios.get("http://127.0.0.1:5000/alerts")
                setAlerts(res.data)
            } catch (err) {
                console.error("Failed to fetch alerts:", err)
            }
        }
        fetchAlerts()
        const interval = setInterval(fetchAlerts, 2000)
        return () => clearInterval(interval)
    }, [])

    // Handle the user clicking Approve or Dismiss
    const handleResolve = async (alertId, action) => {
        try {
            await axios.post("http://127.0.0.1:5000/resolve_alert", {
                alert_id: alertId,
                action: action
            })
            // Instantly remove it from the UI for a snappy feel
            setAlerts(alerts.filter(a => a.id !== alertId))
        } catch (err) {
            console.error("Failed to resolve alert:", err)
        }
    }

    return (
        <div style={panelStyle}>
            <h3 style={{ margin: "0 0 15px 0", color: "#e2e8f0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>📥 AI Dispatcher Inbox</span>
                {alerts.length > 0 && (
                    <span style={badgeStyle}>{alerts.length}</span>
                )}
            </h3>

            <div style={listContainerStyle}>
                {alerts.length === 0 ? (
                    <div style={emptyStyle}>
                        ✅ Network operating nominally.<br/>No pending AI actions.
                    </div>
                ) : (
                    alerts.map(alert => (
                        <div key={alert.id} style={alertCardStyle}>
                            <div style={warningHeaderStyle}>
                                ⚠️ ROUTING ADVISORY
                            </div>
                            <div style={{ fontSize: "0.95rem", marginBottom: "15px", lineHeight: "1.4", color: "#cbd5e1" }}>
                                {alert.message}
                            </div>
                            <div style={{ display: "flex", gap: "10px" }}>
                                <button onClick={() => handleResolve(alert.id, "approve")} style={approveBtnStyle}>
                                    ✓ Approve
                                </button>
                                <button onClick={() => handleResolve(alert.id, "dismiss")} style={dismissBtnStyle}>
                                    ✕ Dismiss
                                </button>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}

// --- STYLING ---
const panelStyle = {
    flex: 1,
    background: "#1e293b",
    padding: "20px",
    borderRadius: "10px",
    border: "1px solid #334155",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden" // Prevent the whole panel from growing too large
}

const listContainerStyle = {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    paddingRight: "5px"
}

const badgeStyle = {
    background: "#ef4444",
    color: "white",
    fontSize: "0.8rem",
    padding: "3px 8px",
    borderRadius: "12px",
    fontWeight: "bold"
}

const emptyStyle = {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    textAlign: "center",
    color: "#10b981",
    fontStyle: "italic",
    background: "rgba(16, 185, 129, 0.1)",
    borderRadius: "8px",
    padding: "20px"
}

const alertCardStyle = {
    background: "rgba(15, 23, 42, 0.7)",
    borderLeft: "4px solid #f59e0b",
    padding: "15px",
    borderRadius: "6px"
}

const warningHeaderStyle = {
    fontSize: "0.80rem",
    color: "#f59e0b",
    fontWeight: "bold",
    letterSpacing: "1px",
    marginBottom: "8px"
}

const approveBtnStyle = {
    flex: 1, padding: "8px", background: "#10b981", color: "white", border: "none", borderRadius: "4px", cursor: "pointer", fontWeight: "bold", transition: "0.2s"
}

const dismissBtnStyle = {
    flex: 1, padding: "8px", background: "#475569", color: "white", border: "none", borderRadius: "4px", cursor: "pointer", fontWeight: "bold", transition: "0.2s"
}
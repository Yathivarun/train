import { useEffect, useState } from "react"
import axios from "axios"

export default function MetricsPanel(){

    const [metrics, setMetrics] = useState({})
    const [aiDelay, setAiDelay] = useState(0)

    useEffect(() => {
        const fetchData = async () => {
            try {
                // 1. Fetch standard simulation metrics
                const res = await axios.get("http://127.0.0.1:5000/metrics")
                setMetrics(res.data)

                // 2. Fetch the new AI XGBoost Prediction
                const aiRes = await axios.get("http://127.0.0.1:5000/predict_delay")
                setAiDelay(aiRes.data.expected_delay_minutes)

            } catch(err) {
                console.log(err)
            }
        }

        fetchData()
        const interval = setInterval(fetchData, 2000)

        return () => clearInterval(interval)
    }, [])

    // Dynamic styling for the AI card
    const delayColor = aiDelay > 15 ? "#ef4444" : (aiDelay > 5 ? "#f59e0b" : "#10b981")

    return(
        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>

            <div className="metric" style={metricStyle}>
                <h4 style={headerStyle}>Active Trains</h4>
                <h2 style={valueStyle}>{metrics.active_trains || 0}</h2>
            </div>

            <div className="metric" style={metricStyle}>
                <h4 style={headerStyle}>Average Speed</h4>
                <h2 style={valueStyle}>{metrics.avg_speed || 0} m/s</h2>
            </div>

            <div className="metric" style={metricStyle}>
                <h4 style={headerStyle}>Congestion Index</h4>
                <h2 style={valueStyle}>{metrics.congestion || 0}</h2>
            </div>

            <div className="metric" style={metricStyle}>
                <h4 style={headerStyle}>Junction Conflicts</h4>
                <h2 style={valueStyle}>{metrics.conflicts || 0}</h2>
            </div>

            {/* 🤖 NEW AI PREDICTION CARD */}
            <div className="metric" style={{
                ...metricStyle, 
                border: `2px solid ${delayColor}`,
                background: "rgba(15, 23, 42, 0.8)",
                boxShadow: `0 0 10px ${delayColor}40`
            }}>
                <h4 style={{...headerStyle, color: delayColor}}>🧠 AI Predicted Delay</h4>
                <h2 style={{...valueStyle, color: delayColor}}>{aiDelay} min</h2>
            </div>

        </div>
    )
}

// Basic inline styling for the cards to make them look sharp
const metricStyle = {
    flex: 1,
    minWidth: "150px",
    background: "#1e293b",
    padding: "15px",
    borderRadius: "10px",
    border: "1px solid #334155",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center"
}

const headerStyle = {
    margin: 0,
    fontSize: "0.9rem",
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "1px"
}

const valueStyle = {
    margin: "10px 0 0 0",
    fontSize: "2rem",
    fontWeight: "bold"
}
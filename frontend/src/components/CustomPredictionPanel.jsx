import { useState } from "react"
import axios from "axios"

export default function CustomPredictionPanel() {
    // Default values for the sliders
    const [inputs, setInputs] = useState({
        active_trains: 5,
        congestion_index: 0.3,
        junction_conflicts: 1
    })
    
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(false)

    // Handle slider changes
    const handleChange = (e) => {
        setInputs({
            ...inputs,
            [e.target.name]: parseFloat(e.target.value)
        })
    }

    // Send data to Flask backend
    const handlePredict = async () => {
        setLoading(true)
        try {
            const res = await axios.post("http://127.0.0.1:5000/custom_predict", inputs)
            setPrediction(res.data.expected_delay_minutes)
        } catch (err) {
            console.error(err)
        }
        setLoading(false)
    }

    return (
        <div style={panelStyle}>
            <h3 style={{ margin: "0 0 15px 0", color: "#e2e8f0" }}>🛠️ Custom "What-If" Scenario</h3>
            
            <div style={inputGroup}>
                <label>Active Trains: <strong>{inputs.active_trains}</strong></label>
                <input type="range" name="active_trains" min="1" max="25" step="1" 
                    value={inputs.active_trains} onChange={handleChange} style={sliderStyle} />
            </div>

            <div style={inputGroup}>
                <label>Congestion Index: <strong>{inputs.congestion_index.toFixed(2)}</strong></label>
                <input type="range" name="congestion_index" min="0" max="1" step="0.01" 
                    value={inputs.congestion_index} onChange={handleChange} style={sliderStyle} />
            </div>

            <div style={inputGroup}>
                <label>Junction Conflicts: <strong>{inputs.junction_conflicts}</strong></label>
                <input type="range" name="junction_conflicts" min="0" max="10" step="1" 
                    value={inputs.junction_conflicts} onChange={handleChange} style={sliderStyle} />
            </div>

            <button onClick={handlePredict} style={buttonStyle}>
                {loading ? "Calculating..." : "Predict Scenario Delay"}
            </button>

            {prediction !== null && (
                <div style={resultStyle(prediction)}>
                    AI Estimated Delay: <strong>{prediction} mins</strong>
                </div>
            )}
        </div>
    )
}

// --- STYLING ---
const panelStyle = {
    background: "#1e293b",
    padding: "20px",
    borderRadius: "10px",
    border: "1px solid #334155",
    color: "#94a3b8",
    display: "flex",
    flexDirection: "column",
    gap: "10px"
}

const inputGroup = {
    display: "flex",
    flexDirection: "column",
    gap: "5px",
    fontSize: "0.9rem"
}

const sliderStyle = {
    cursor: "pointer",
    accentColor: "#3b82f6" // Blue slider thumb
}

const buttonStyle = {
    marginTop: "10px",
    padding: "10px",
    background: "#3b82f6",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontWeight: "bold",
    transition: "background 0.2s"
}

const resultStyle = (delay) => ({
    marginTop: "15px",
    padding: "15px",
    textAlign: "center",
    fontSize: "1.2rem",
    background: "rgba(15, 23, 42, 0.5)",
    borderRadius: "5px",
    border: `1px solid ${delay > 15 ? "#ef4444" : (delay > 5 ? "#f59e0b" : "#10b981")}`,
    color: delay > 15 ? "#ef4444" : (delay > 5 ? "#f59e0b" : "#10b981")
})
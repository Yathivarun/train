import { useState } from "react"
import Sidebar from "../components/Sidebar"
import LiveMap from "../components/LiveMap"
import MetricsPanel from "../components/MetricsPanel"
import TrainTable from "../components/TrainTable"
import AiDispatcherInbox from "../components/AiDispatcherInbox"
import PredictionPanel from "../components/PredictionPanel"
import BenchmarkPanel from "../components/BenchmarkPanel"
import DQNStatusCard from "../components/DQNStatusCard"

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("live")

  return (
    <div style={shell}>
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      <div style={main}>
        {activeTab === "live"     && <LiveView />}
        {activeTab === "predict"  && <PredictionView />}
        {activeTab === "bench"    && <BenchmarkPanel />}
      </div>
    </div>
  )
}

/* ── Live simulation view ───────────────────────────────────────────────── */
function LiveView() {
  return (
    <div style={col}>
      {/* Row 1: metrics strip */}
      <MetricsPanel />

      {/* Row 2: map + DQN status side-by-side */}
      <div style={{ display: "flex", gap: 16, flex: "0 0 420px" }}>
        <div style={{ ...card, flex: 2, padding: 8, overflow: "hidden" }}>
          <LiveMap />
        </div>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
          <DQNStatusCard />
          {/* AI Dispatcher Inbox — fixed height, independent scroll */}
          <div style={{ ...card, flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <AiDispatcherInbox />
          </div>
        </div>
      </div>

      {/* Row 3: per-train table — full width */}
      <div style={{ ...card, flex: "0 0 280px", overflow: "hidden" }}>
        <TrainTable />
      </div>
    </div>
  )
}

/* ── Prediction / analysis view ─────────────────────────────────────────── */
function PredictionView() {
  return (
    <div style={col}>
      <PredictionPanel />
    </div>
  )
}

/* ── Shared styles ───────────────────────────────────────────────────────── */
const shell = {
  display:    "flex",
  height:     "100vh",
  background: "#020617",
  color:      "white",
  overflow:   "hidden",
}

const main = {
  flex:       1,
  display:    "flex",
  flexDirection: "column",
  padding:    "16px",
  gap:        "16px",
  overflow:   "hidden",
}

const col = {
  display:        "flex",
  flexDirection:  "column",
  gap:            "16px",
  height:         "100%",
  overflow:       "hidden",
}

const card = {
  background:   "#0f172a",
  borderRadius: "10px",
  border:       "1px solid #1e293b",
}

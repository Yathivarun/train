import Sidebar from "../components/Sidebar"
import LiveMap from "../components/LiveMap"
import MetricsPanel from "../components/MetricsPanel"
import TrainTable from "../components/TrainTable"
import CustomPredictionPanel from "../components/CustomPredictionPanel"
import AiDispatcherInbox from "../components/AiDispatcherInbox"

export default function Dashboard() {

    return (
        <div style={{
            display: "flex",
            height: "100vh",
            background: "#020617",
            color: "white"
        }}>

            <Sidebar />

            <div style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                padding: "20px",
                gap: "20px"
            }}>

                {/* --- Top Row: Live Metrics & AI Predictions --- */}
                <MetricsPanel />

                {/* --- Middle Row: Live Simulation Map --- */}
                <div style={{
                    flex: 1,
                    background: "#0f172a",
                    borderRadius: "10px",
                    padding: "10px"
                }}>
                    <LiveMap />
                </div>

                {/* --- Bottom Row: Train Table, AI Inbox & Custom Scenarios --- */}
                                        <div style={{
                                            display: "flex",
                                            gap: "20px",
                                            height: "450px" // Slightly taller
                                        }}>
                                            
                                            {/* Left side: Train Table */}
                                            <div style={{
                                                flex: 1.2, // Reduced from 1.5 to give the right side more room
                                                background: "#0f172a",
                                                borderRadius: "10px",
                                                padding: "10px",
                                                overflow: "auto"
                                            }}>
                                                <TrainTable />
                                            </div>

                                            {/* Right side: Stacked AI Inbox and Custom Predictor */}
                                            <div style={{
                                                flex: 1, 
                                                minWidth: "400px", // Forces the panel to stay wide
                                                display: "flex",
                                                flexDirection: "column",
                                                gap: "20px"
                                            }}>
                                                <AiDispatcherInbox />
                                                <CustomPredictionPanel />
                                            </div>

                                        </div>

            </div>

        </div>
    )
}
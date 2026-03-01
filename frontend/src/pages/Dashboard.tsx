import Header from "../components/Header"
import Sidebar from "../components/Sidebar"
import MapView from "../components/MapView"
import MetricsPanel from "../components/MetricsPanel"
import BottomLogs from "../components/BottomLogs"

export default function Dashboard() {
  return (
    <div style={{display:"flex",flexDirection:"column",height:"100vh"}}>
      <Header />

      <div style={{display:"flex",flex:1}}>
        <Sidebar />
        <MapView />
        <MetricsPanel />
      </div>

      <BottomLogs />
    </div>
  )
}
import { Train, Activity, BarChart2 } from "lucide-react"

const tabs = [
  { id: "live",    icon: Train,    label: "Live Twin"   },
  { id: "predict", icon: Activity, label: "Predict"     },
  { id: "bench",   icon: BarChart2,label: "Benchmarks"  },
]

export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <div style={sidebar}>
      <div style={logo}>
        <span style={logoIcon}>🚆</span>
        <div>
          <div style={logoTitle}>Rail Twin</div>
          <div style={logoSub}>AI Control Center</div>
        </div>
      </div>

      <nav style={{ marginTop: 32 }}>
        {tabs.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            style={tabBtn(activeTab === id)}
          >
            <Icon size={16} style={{ flexShrink: 0 }} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div style={footer}>
        <div style={footerText}>Methodist College</div>
        <div style={{ ...footerText, opacity: 0.4, fontSize: 11 }}>CSE Dept · 2025-26</div>
      </div>
    </div>
  )
}

const sidebar = {
  width:          220,
  background:     "#0f172a",
  borderRight:    "1px solid #1e293b",
  padding:        "20px 12px",
  display:        "flex",
  flexDirection:  "column",
  flexShrink:     0,
}

const logo = {
  display:    "flex",
  alignItems: "center",
  gap:        12,
  padding:    "0 8px 20px",
  borderBottom: "1px solid #1e293b",
}

const logoIcon   = { fontSize: 28 }
const logoTitle  = { fontWeight: 700, fontSize: 16, color: "#e2e8f0" }
const logoSub    = { fontSize: 11, color: "#64748b", marginTop: 2 }

const tabBtn = (active) => ({
  display:        "flex",
  alignItems:     "center",
  gap:            10,
  width:          "100%",
  padding:        "10px 12px",
  marginBottom:   4,
  background:     active ? "rgba(59,130,246,0.15)" : "transparent",
  border:         active ? "1px solid rgba(59,130,246,0.3)" : "1px solid transparent",
  borderRadius:   8,
  color:          active ? "#93c5fd" : "#64748b",
  fontSize:       14,
  fontWeight:     active ? 600 : 400,
  cursor:         "pointer",
  transition:     "all 0.15s",
  textAlign:      "left",
})

const footer = {
  marginTop: "auto",
  padding:   "16px 8px 0",
  borderTop: "1px solid #1e293b",
}

const footerText = {
  color:    "#475569",
  fontSize: 12,
  marginBottom: 2,
}

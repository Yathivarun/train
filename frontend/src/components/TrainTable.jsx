import { useEffect, useState } from "react"
import axios from "axios"

const API = "http://127.0.0.1:5000"

// Real CSV train types (Local/MEMU/Passenger/Express from section_train_master_expanded)
// SUMO numeric-ID types kept as fallback (SUPERFAST/EXPRESS/PASSENGER/EMU/FREIGHT)
const TYPE_BADGE = {
  Local:     { bg: "#064e3b", color: "#6ee7b7" },
  MEMU:      { bg: "#1e3a5f", color: "#93c5fd" },
  Passenger: { bg: "#3b0764", color: "#d8b4fe" },
  Express:   { bg: "#78350f", color: "#fcd34d" },
  SUPERFAST: { bg: "#7f1d1d", color: "#fca5a5" },
  EXPRESS:   { bg: "#78350f", color: "#fcd34d" },
  PASSENGER: { bg: "#3b0764", color: "#d8b4fe" },
  EMU:       { bg: "#064e3b", color: "#6ee7b7" },
  FREIGHT:   { bg: "#1e293b", color: "#94a3b8" },
}

export default function TrainTable() {
  const [details, setDetails] = useState([])
  const [sortKey, setSortKey] = useState("predicted_delay")
  const [sortDir, setSortDir] = useState("desc")

  useEffect(() => {
    const fetch = async () => {
      try {
        const r = await axios.get(`${API}/train_details`)
        setDetails(r.data)
      } catch (_) {}
    }
    fetch()
    const id = setInterval(fetch, 3000)
    return () => clearInterval(id)
  }, [])

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc")
    } else {
      setSortKey(key)
      setSortDir("desc")
    }
  }

  const sorted = [...details].sort((a, b) => {
    const va = a[sortKey] ?? 0
    const vb = b[sortKey] ?? 0
    return sortDir === "asc" ? va - vb : vb - va
  })

  const delayColor = (d) =>
    d > 15 ? "#ef4444" : d > 5 ? "#f59e0b" : "#10b981"

  return (
    <div style={wrapper}>
      <div style={titleBar}>
        <span style={title}>🚆 Active Train Details</span>
        <span style={count}>{details.length} trains</span>
      </div>

      <div style={tableWrapper}>
        <table style={table}>
          <thead>
            <tr>
              {COLUMNS.map(({ key, label }) => (
                <th key={key} style={th} onClick={() => handleSort(key)}>
                  {label} {sortKey === key ? (sortDir === "asc" ? "↑" : "↓") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={COLUMNS.length} style={empty}>
                  No trains active. Start the SUMO simulation.
                </td>
              </tr>
            ) : (
              sorted.map(t => {
                const badge  = TYPE_BADGE[t.train_type] || TYPE_BADGE.EXPRESS
                const delay  = t.predicted_delay ?? 0
                const dColor = delayColor(delay)
                return (
                  <tr key={t.id} style={row}>
                    <td style={td}>
                      <code style={{ fontSize: 12, color: "#94a3b8" }}>
                        {t.id?.split("_")[0]}
                      </code>
                    </td>
                    <td style={td}>
                      <span style={{
                        ...typeBadge,
                        background: `${badge.bg}88`,
                        color: badge.color,
                        border: `1px solid ${badge.bg}`,
                      }}>
                        {t.train_type}
                      </span>
                    </td>
                    <td style={td}>
                      <span style={{ ...priorityDot, background: PRIORITY_COLORS[t.priority] || "#64748b" }} />
                      {t.priority}
                    </td>
                    <td style={td}>{Number(t.lat).toFixed(4)}</td>
                    <td style={td}>{Number(t.lon).toFixed(4)}</td>
                    <td style={td}>
                      <span style={{ color: dColor, fontWeight: 600 }}>
                        {delay} min
                      </span>
                    </td>
                    <td style={td}>
                      <div style={{
                        display: "inline-block",
                        padding: "2px 8px",
                        borderRadius: 4,
                        fontSize: 11,
                        background: `${dColor}22`,
                        color: dColor,
                        fontWeight: 700,
                      }}>
                        {delay > 15 ? "⚠ HIGH" : delay > 5 ? "△ MED" : "✓ LOW"}
                      </div>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const COLUMNS = [
  { key: "id",               label: "Train ID"    },
  { key: "train_type",       label: "Type"        },
  { key: "priority",         label: "Priority"    },
  { key: "lat",              label: "Lat"         },
  { key: "lon",              label: "Lon"         },
  { key: "predicted_delay",  label: "Pred. Delay" },
  { key: "predicted_delay",  label: "Risk"        },
]

const PRIORITY_COLORS = { 5: "#ef4444", 4: "#f59e0b", 3: "#3b82f6", 2: "#10b981", 1: "#8b5cf6" }

const wrapper = {
  height:         "100%",
  display:        "flex",
  flexDirection:  "column",
  padding:        "14px",
  overflow:       "hidden",
}

const titleBar = {
  display:        "flex",
  justifyContent: "space-between",
  alignItems:     "center",
  marginBottom:   10,
  flexShrink:     0,
}

const title = { fontWeight: 700, color: "#e2e8f0", fontSize: 14 }

const count = {
  fontSize:   12,
  color:      "#64748b",
  background: "#1e293b",
  padding:    "2px 8px",
  borderRadius: 10,
}

const tableWrapper = {
  flex:       1,
  overflowY:  "auto",
}

const table = {
  width:           "100%",
  borderCollapse:  "collapse",
  fontSize:        13,
  color:           "#e2e8f0",
}

const th = {
  padding:       "8px 10px",
  textAlign:     "left",
  color:         "#64748b",
  fontSize:      11,
  fontWeight:    600,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  cursor:        "pointer",
  userSelect:    "none",
  borderBottom:  "1px solid #1e293b",
  position:      "sticky",
  top:           0,
  background:    "#0f172a",
}

const td = {
  padding:      "8px 10px",
  borderBottom: "1px solid #0f172a",
  verticalAlign: "middle",
}

const row = {
  transition: "background 0.1s",
}

const empty = {
  textAlign: "center",
  color:     "#475569",
  padding:   "24px",
}

const typeBadge = {
  fontSize:     11,
  fontWeight:   700,
  padding:      "2px 7px",
  borderRadius: 4,
}

const priorityDot = {
  display:      "inline-block",
  width:        7,
  height:       7,
  borderRadius: "50%",
  marginRight:  5,
  verticalAlign: "middle",
}

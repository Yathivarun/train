import { MapContainer, TileLayer, Marker, Popup, GeoJSON, CircleMarker } from "react-leaflet"
import { useEffect, useState } from "react"
import axios from "axios"
import L from "leaflet"
import "leaflet/dist/leaflet.css"

const API = "http://127.0.0.1:5000"

// Fix default icon paths
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
  iconUrl:       "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl:     "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
})

const TRAIN_ICON = new L.Icon({
  iconUrl:  "https://cdn-icons-png.flaticon.com/512/565/565410.png",
  iconSize: [28, 28],
})

// Color per train type
const TYPE_COLORS = {
  SUPERFAST: "#ef4444",
  EXPRESS:   "#f59e0b",
  PASSENGER: "#3b82f6",
  EMU:       "#10b981",
  FREIGHT:   "#8b5cf6",
}

export default function LiveMap() {
  const [trains, setTrains]   = useState([])
  const [details, setDetails] = useState([]) // enriched with XGBoost predictions
  const [tracks, setTracks]   = useState(null)
  const [blocks, setBlocks]   = useState([])
  const [selected, setSelected] = useState(null)

  // Load static track geometry once
  useEffect(() => {
    axios.get(`${API}/tracks`).then(r => setTracks(r.data)).catch(() => {})
  }, [])

  // Poll live positions every second
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await axios.get(`${API}/trains`)
        setTrains(r.data)
      } catch (_) {}
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // Poll enriched details (XGBoost per-train) every 3s
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await axios.get(`${API}/train_details`)
        setDetails(r.data)
      } catch (_) {}
    }, 3000)
    return () => clearInterval(id)
  }, [])

  // Poll block occupancy every second
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await axios.get(`${API}/blocks`)
        setBlocks(r.data)
      } catch (_) {}
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // Build a quick lookup: train_id → detail
  const detailMap = {}
  details.forEach(d => { detailMap[d.id] = d })

  const delayColor = (delay) =>
    delay > 15 ? "#ef4444" : delay > 5 ? "#f59e0b" : "#10b981"

  return (
    <MapContainer center={[22.65, 88.34]} zoom={11} style={{ height: "100%", width: "100%" }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      {tracks && (
        <GeoJSON data={tracks} style={{ color: "#3b82f6", weight: 2, opacity: 0.6 }} />
      )}

      {/* Train markers — FIX: use eventHandlers not onClick */}
      {trains.map(train => {
        const d     = detailMap[train.id] || {}
        const ttype = d.train_type || "EXPRESS"
        const delay = d.predicted_delay ?? 0
        const color = delayColor(delay)

        return (
          <Marker
            key={train.id}
            position={[train.lat, train.lon]}
            icon={TRAIN_ICON}
            eventHandlers={{                    // ← FIX for react-leaflet v4/v5
              click: () => setSelected(train.id),
            }}
          >
            <Popup>
              <div style={{ minWidth: 200, fontFamily: "monospace", fontSize: 13 }}>
                <div style={{ fontWeight: 700, marginBottom: 6, borderBottom: "1px solid #e2e8f0", paddingBottom: 4 }}>
                  🚆 Train {train.id}
                </div>
                <div><b>Type:</b> {ttype}</div>
                <div><b>Priority:</b> {d.priority ?? "—"}</div>
                <div><b>Speed:</b> {train.speed ?? d.speed ?? "—"} m/s</div>
                <div><b>Lat:</b> {Number(train.lat).toFixed(4)}</div>
                <div><b>Lon:</b> {Number(train.lon).toFixed(4)}</div>
                <div style={{ marginTop: 6, padding: "4px 8px", borderRadius: 4,
                              background: `${color}22`, border: `1px solid ${color}` }}>
                  <b style={{ color }}>Predicted Delay: {delay} min</b>
                </div>
              </div>
            </Popup>
          </Marker>
        )
      })}

      {/* Block occupancy markers */}
      {blocks.map(block => (
        <CircleMarker
          key={block.train_id}
          center={[block.lat, block.lon]}
          radius={5}
          pathOptions={{ color: "#ef4444", fillColor: "#ef4444", fillOpacity: 0.6 }}
        >
          <Popup>Block: {block.train_id} | Edge: {block.edge}</Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  )
}

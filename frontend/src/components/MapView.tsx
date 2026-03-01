import { MapContainer, TileLayer, Polyline, Marker } from "react-leaflet"
import "leaflet/dist/leaflet.css"
import { useEffect, useState, useContext } from "react"
import { ScenarioContext } from "../context/ScenarioContext"
import L from "leaflet"

const trainIcon = new L.Icon({
  iconUrl: "https://maps.google.com/mapfiles/ms/icons/red-dot.png",
  iconSize: [32, 32]
})

export default function MapView() {

  const [edges, setEdges] = useState<any[]>([])
  const [position, setPosition] = useState<[number, number] | null>(null)
  const { scenario } = useContext(ScenarioContext)

  useEffect(() => {
    fetch("../outputs/network_edges.json")
      .then(res => res.json())
      .then(data => {
        console.log("Edges loaded:", data.length)
        setEdges(data)
      })
      .catch(err => console.error("Edge load error:", err))
  }, [])

  useEffect(() => {

    if (edges.length === 0) return

    let step = 0

    const speed =
      scenario === "Peak" ? 300 :
      scenario === "Emergency" ? 150 :
      800

    const interval = setInterval(() => {

      const coords = edges[0]?.coords

      if (!coords || coords.length === 0) return

      step = (step + 1) % coords.length

      setPosition([coords[step][0], coords[step][1]])

    }, speed)

    return () => clearInterval(interval)

  }, [edges, scenario])

  return (
    <div style={{ width: "60%" }}>
      <MapContainer
        center={[22.57, 88.36]}
        zoom={11}
        style={{ height: "100%" }}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        {edges.map((edge, index) => (
          <Polyline
            key={index}
            positions={edge.coords}
            pathOptions={{ color: "#00ffff", weight: 2 }}
          />
        ))}

        {position && (
          <Marker position={position} icon={trainIcon} />
        )}

      </MapContainer>
    </div>
  )
}
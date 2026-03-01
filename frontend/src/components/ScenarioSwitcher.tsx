import { useContext } from "react"
import { ScenarioContext } from "../context/ScenarioContext"

export default function ScenarioSwitcher() {

  const { scenario, setScenario } = useContext(ScenarioContext)

  return (
    <div style={{ marginTop: 20 }}>
      <h3>📡 Scenario Mode</h3>
      <select
        value={scenario}
        onChange={(e) => setScenario(e.target.value)}
      >
        <option value="Peak">Peak</option>
        <option value="Emergency">Emergency</option>
        <option value="Maintenance">Maintenance</option>
      </select>
    </div>
  )
}
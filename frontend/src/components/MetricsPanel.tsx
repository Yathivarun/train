import { motion } from "framer-motion"
import ScenarioSwitcher from "./ScenarioSwitcher"

export default function MetricsPanel() {

  return (
    <motion.div
      className="panel"
      initial={{ x: 300 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.8 }}
      style={{ width: "20%" }}
    >
      <h3>📊 KPIs</h3>
      <p>On-Time: 92%</p>
      <p>Avg Delay: 3 min</p>
      <p className="alert">Conflicts: 1</p>

      <ScenarioSwitcher />
    </motion.div>
  )
}
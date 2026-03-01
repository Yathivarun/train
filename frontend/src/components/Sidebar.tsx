import { motion } from "framer-motion"

export default function Sidebar() {

  return (
    <motion.div
      className="panel"
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.8 }}
      style={{ width: "20%" }}
    >
      <h3>🚄 Active Trains</h3>
      <ul>
        <li>Train 37215 - Running</li>
        <li className="alert">Train 53001 - Delay</li>
      </ul>
    </motion.div>
  )
}
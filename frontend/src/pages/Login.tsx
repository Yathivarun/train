import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"

export default function Login() {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const navigate = useNavigate()

  const handleLogin = () => {
    if (username === "admin" && password === "railway") {
      localStorage.setItem("token", "demo-jwt-token")
      navigate("/dashboard")
    } else {
      alert("Unauthorized Access")
    }
  }

  return (
    <div style={{height:"100vh",display:"flex",justifyContent:"center",alignItems:"center"}}>
      <motion.div 
        className="panel"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <h2>🔐 OCC Secure Login</h2>
        <input placeholder="Username" onChange={(e)=>setUsername(e.target.value)} /><br/>
        <input type="password" placeholder="Password" onChange={(e)=>setPassword(e.target.value)} /><br/>
        <button onClick={handleLogin}>Authorize</button>
      </motion.div>
    </div>
  )
}
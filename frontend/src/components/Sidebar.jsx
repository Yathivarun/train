import { Train, Activity, Map } from "lucide-react"

export default function Sidebar(){

return(

<div style={{
width:"240px",
background:"#0f172a",
color:"white",
padding:"20px",
height:"100vh"
}}>

<h2>Rail Twin</h2>

<div style={{marginTop:"40px"}}>

<p><Map size={18}/> Network Map</p>
<p><Train size={18}/> Trains</p>
<p><Activity size={18}/> Metrics</p>

</div>

</div>

)

}
import Sidebar from "../components/Sidebar"
import LiveMap from "../components/LiveMap"
import MetricsPanel from "../components/MetricsPanel"
import TrainTable from "../components/TrainTable"

export default function Dashboard(){

return(

<div style={{
display:"flex",
height:"100vh",
background:"#020617",
color:"white"
}}>

<Sidebar/>

<div style={{
flex:1,
display:"flex",
flexDirection:"column",
padding:"20px",
gap:"20px"
}}>

<MetricsPanel/>

<div style={{
flex:1,
background:"#0f172a",
borderRadius:"10px",
padding:"10px"
}}>
<LiveMap/>
</div>

<div style={{
height:"200px",
background:"#0f172a",
borderRadius:"10px",
padding:"10px",
overflow:"auto"
}}>
<TrainTable/>
</div>

</div>

</div>

)

}
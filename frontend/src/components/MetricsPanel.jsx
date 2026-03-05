import { useEffect, useState } from "react"
import axios from "axios"

export default function MetricsPanel(){

const [metrics,setMetrics] = useState({})

useEffect(()=>{

const fetchMetrics = async ()=>{

try{

const res = await axios.get("http://127.0.0.1:5000/metrics")
setMetrics(res.data)

}catch(err){
console.log(err)
}

}

fetchMetrics()

const interval = setInterval(fetchMetrics,2000)

return ()=>clearInterval(interval)

},[])

return(

<div style={{
display:"flex",
gap:"20px"
}}>

<div className="metric">
<h4>Active Trains</h4>
<h2>{metrics.active_trains}</h2>
</div>

<div className="metric">
<h4>Average Speed</h4>
<h2>{metrics.avg_speed} m/s</h2>
</div>

<div className="metric">
<h4>Congestion Index</h4>
<h2>{metrics.congestion}</h2>
</div>

<div className="metric">
<h4>Junction Conflicts</h4>
<h2>{metrics.conflicts}</h2>
</div>

</div>

)

}
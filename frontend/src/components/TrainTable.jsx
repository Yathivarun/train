import { useEffect, useState } from "react"
import axios from "axios"

export default function TrainTable(){

const [trains,setTrains] = useState([])

useEffect(()=>{

const fetchTrains = async ()=>{
const res = await axios.get("http://127.0.0.1:5000/trains")
setTrains(res.data)
}

fetchTrains()

const interval = setInterval(fetchTrains,2000)

return ()=>clearInterval(interval)

},[])

return(

<table style={{width:"100%",color:"white"}}>

<thead>
<tr>
<th>ID</th>
<th>Latitude</th>
<th>Longitude</th>
</tr>
</thead>

<tbody>

{trains.map(t=>(
<tr key={t.id}>
<td>{t.id}</td>
<td>{t.lat.toFixed(3)}</td>
<td>{t.lon.toFixed(3)}</td>
</tr>
))}

</tbody>

</table>

)

}
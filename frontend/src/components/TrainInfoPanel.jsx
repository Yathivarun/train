export default function TrainInfoPanel({train}){

if(!train) return null

return(

<div style={{
position:"absolute",
right:"20px",
top:"100px",
background:"#0f172a",
padding:"20px",
borderRadius:"10px",
width:"220px"
}}>

<h3>Train Details</h3>

<p><b>ID:</b> {train.id}</p>
<p><b>Latitude:</b> {train.lat.toFixed(4)}</p>
<p><b>Longitude:</b> {train.lon.toFixed(4)}</p>

</div>

)

}
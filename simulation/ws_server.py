import asyncio
import websockets
import json
import csv

PORT = 8765

async def send_live_data(websocket):
    while True:
        try:
            with open("../outputs/live_status.csv") as f:
                reader = csv.DictReader(f)
                trains = list(reader)

            payload = {
                "trains": trains,
                "timestamp": asyncio.get_event_loop().time()
            }

            await websocket.send(json.dumps(payload))
            await asyncio.sleep(1)

        except Exception as e:
            print("Error:", e)
            await asyncio.sleep(2)

async def main():
    async with websockets.serve(send_live_data, "localhost", PORT):
        print(f"WebSocket running on ws://localhost:{PORT}")
        await asyncio.Future()

asyncio.run(main())
import websockets

uri = "ws://localhost:8765"


async def send_mallet_position(x: float, y: float) -> None:
    async with websockets.connect(uri) as websocket:
        # Just an example
        await websocket.send(f'{x}')
        await websocket.send(f'{y}')


async def get_opponent_mallet_pos() -> tuple[float, float]:
    return (0, 0)

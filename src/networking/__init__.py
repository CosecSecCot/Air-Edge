import websockets

from utils.logger import server_logger


async def get_mallet_position(websocket: websockets.ServerConnection) -> tuple[int, int]:
    # Just an example (please implement it more elegantly)
    x = await websocket.recv()
    y = await websocket.recv()

    try:
        res = (int(x.strip()), int(y.strip()))
        return res
    except:
        server_logger.error("Couldn't parse mallet position!")
        return (0, 0)


async def server(websocket: websockets.ServerConnection) -> None:
    # Just an example (please implement it more elegantly)
    (x, y) = await get_mallet_position(websocket)
    server_logger.debug(f'Mallet postion: ({x}, {y})')

    # Do some physics with puck and mallet
    # Send the resulting position of the puck to the clients connected to the game
    # Send the opponent's mallet position to each client accordingly

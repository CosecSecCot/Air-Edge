import websockets
from utils.logger import server_logger

clients: dict[int, websockets.ServerConnection] = {}


async def get_mallet_position(client: int, websocket: websockets.ServerConnection):
    """Receives mallet position from a client, validates input, and broadcasts it."""
    try:
        async for message in websocket:
            if validate_position(message):
                server_logger.info(f"Received from Client {client}: {message}")
                # Send this valid position to the opponent(s)
                await broadcast_message(client, str(message))
            else:
                server_logger.warning(
                    f"Invalid input from Client {client}: {message}")
    except websockets.exceptions.ConnectionClosedError:
        server_logger.error(f"Client {client} disconnected unexpectedly")
    finally:
        server_logger.info(f"Client {client} disconnected")
        # Remove client if present
        if client in clients:
            del clients[client]


def validate_position(message: websockets.Data) -> bool:
    """Validates if the received message is a tuple of two numeric values."""
    try:
        # Expects message format like "(x,y)"
        _ = map(float, str(message).strip("()").split(","))
        return True
    except (ValueError, AttributeError):
        return False


async def broadcast_message(sender: int, message: str):
    """Sends the mallet position from one client to all other connected clients."""
    for client, ws in clients.items():
        if client != sender:
            try:
                server_logger.info(f"Sent to Client {client}: {message}")
                await ws.send(message)
            except Exception as e:
                server_logger.error(f"Failed to send to client {client}: {e}")


async def server(websocket: websockets.ServerConnection) -> None:
    """Handles client connections with a maximum of two clients allowed."""
    if len(clients) >= 2:
        server_logger.warning(
            "Maximum clients connected. Rejecting new connection.")
        await websocket.send("Server full. Only two players allowed.")
        await websocket.close()
        return

    client_id = len(clients) + 1  # Assign a unique ID
    clients[client_id] = websocket
    server_logger.info(f"Client {client_id} connected.")

    await get_mallet_position(client_id, websocket)

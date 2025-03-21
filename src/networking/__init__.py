import asyncio
import websockets
from utils.logger import server_logger 

clients = {}  

async def get_mallet_position(client, websocket: websockets.ClientConnection):
    """Receives mallet position from a client and validates input."""
    try:
        async for message in websocket:
            if validate_position(message):
                server_logger.info(f"Received from Client {client}: {message}")
            else:
                server_logger.warning(f"Invalid input from Client {client}: {message}")
    except websockets.exceptions.ConnectionClosedError:
        server_logger.error(f"Client {client} disconnected unexpectedly")
    finally:
        server_logger.info(f"Client {client} disconnected")
        del clients[client]

def validate_position(message):
    """Validates if the received message is a tuple of two numeric values."""
    try:
        x, y = map(float, message.strip("()").split(","))
        return True 
    except (ValueError, AttributeError):
        return False 

async def send_mallet_position():
    """Placeholder function for sending updated mallet positions."""
    pass  # To be implemented later

async def server(websocket: websockets.ServerConnection) -> None:
    """Handles client connections."""
    client_id = len(clients) + 1  # Assign a unique ID
    clients[client_id] = websocket
    server_logger.info(f"Client {client_id} connected.")
    
    await get_mallet_position(client_id, websocket)

    # Do some physics with puck and mallet
    # Send the resulting position of the puck to the clients connected to the game
    # Send the opponent's mallet position to each client accordingly

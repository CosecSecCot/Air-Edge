import asyncio
import threading
import queue
import websockets
from websockets.exceptions import ConnectionClosed

from utils.logger import client_logger


class NetworkClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.client_id: int | None = None

        # For sending messages from game loop to the network thread.
        self.send_queue = queue.Queue()

        # For receiving messages from the network thread to the game loop.
        self.recv_queue = queue.Queue()

        self.loop = asyncio.new_event_loop()
        self.websocket: websockets.ClientConnection | None = None
        self.running = True

        # Start the background thread that runs the event loop.
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())
        self.loop.close()
        client_logger.info("Event loop closed.")

    async def _connect(self):
        try:
            self.websocket = await websockets.connect(self.uri)
            client_logger.info("Connected to the server.")
            send_task = asyncio.create_task(self._sender())
            recv_task = asyncio.create_task(self._receiver())
            await asyncio.gather(send_task, recv_task)
        except Exception as e:
            client_logger.error(f"Connection error: {e}")
            self.running = False

    async def _sender(self):
        """Continuously send messages from the send_queue to the server."""
        while self.running:
            message = await self.loop.run_in_executor(None, self.send_queue.get)
            if message is None:  # Shutdown signal.
                break
            try:
                if self.websocket:
                    await self.websocket.send(message)
            except Exception as e:
                client_logger.error(f"Send error: {e}")
                self.running = False

    async def _receiver(self):
        """Continuously listen for messages from the server."""
        while self.running:
            try:
                # if self.websocket:
                #     message = await self.websocket.recv()
                #     self.recv_queue.put(message)

                if self.websocket:
                    message = await self.websocket.recv()
                    # handle assign_id messages immediately
                    try:
                        data = __import__('json').loads(message)
                        if data.get("type") == "assign_id":
                            self.client_id = data["id"]
                            client_logger.info(
                                f"Assigned client_id = {self.client_id}")
                            continue
                    except Exception:
                        pass
                    self.recv_queue.put(message)
            except ConnectionClosed as e:
                # This exception is expected when the connection is closed normally.
                client_logger.info(
                    f"WebSocket closed (code: {e.code}, reason: {e.reason}).")
                break
            except Exception as e:
                client_logger.error(f"Receive error: {e}")
                self.running = False

    def send_mallet_position(self, x: float, y: float) -> None:
        """Sends the mallet's position to the server in the format '(x,y)'."""
        # message = f"({x},{y})"
        message = __import__('json').dumps({"type": "mallet", "x": x, "y": y})
        self.send_queue.put(message)

    # def get_opponent_mallet_pos(self) -> tuple[float, float] | None:
    #     """
    #     Retrieves the latest opponent mallet position, if available.
    #     Expects the message in the format '(x,y)'.
    #     """
    #     try:
    #         msg = self.recv_queue.get_nowait()  # Non-blocking.
    #         client_logger.info(f"Recieved From Client: {msg}")
    #         if msg.startswith("(") and msg.endswith(")"):
    #             coords = msg[1:-1]
    #             x_str, y_str = coords.split(",")
    #             return float(x_str), float(y_str)
    #     except queue.Empty:
    #         return None
    #     except Exception as e:
    #         client_logger.error(f"Parsing error: {e}")
    #         return None

    def send_ready(self) -> None:
        """Notify server that this client is ready."""
        msg = __import__('json').dumps({"type": "ready"})
        self.send_queue.put(msg)

    def get_server_state(self) -> dict | None:
        """Non-blocking: return the next 'state_update' payload or None."""
        import json
        import queue
        try:
            raw = self.recv_queue.get_nowait()
            data = json.loads(raw)
            client_logger.debug(data)
            if data.get("type") == "state_update":
                return data
        except queue.Empty:
            return None
        except Exception as e:
            client_logger.error(f"State parse error: {e}")
        return None

    def close(self):
        """Properly shuts down the WebSocket connection and stops the event loop."""
        self.running = False
        self.send_queue.put(None)  # Unblock sender if waiting.
        if self.websocket:
            fut = asyncio.run_coroutine_threadsafe(
                self.websocket.close(), self.loop)
            try:
                fut.result(timeout=3)
            except Exception as e:
                client_logger.error(f"Error closing websocket: {e}")
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)
        except Exception as e:
            client_logger.error(f"Error terminating thread: {e}")
        client_logger.info("Network client closed.")

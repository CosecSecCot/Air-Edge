import asyncio
import websockets

import networking
from utils.logger import server_logger


PORT = 8765


async def main():
    server_logger.debug("WELCOME TO AIR EDGE!")
    async with websockets.serve(networking.server, "localhost", PORT):
        server_logger.info(f"Listening on localhost:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == '__main__':
    asyncio.run(main())

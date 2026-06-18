import asyncio
import logging
import json
from typing import List
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("websocket")


class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Active connections: {len(self.active_connections)}")

    async def disconnect_all(self):
        """Used for graceful shutdown."""
        connections = list(self.active_connections)
        for ws in connections:
            try:
                await ws.close(code=1001, reason="Server shutting down")
            except Exception:
                pass
        self.active_connections.clear()

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        json_message = json.dumps(message, default=str)
        dead_connections = []

        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)


# Global singleton instance
ws_manager = WebSocketManager()

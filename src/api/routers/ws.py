from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..websocket import ws_manager
import logging

logger = logging.getLogger("api")
router = APIRouter(tags=["websocket"])


@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Simple ping/pong heartbeat
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

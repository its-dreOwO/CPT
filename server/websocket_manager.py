"""
server/websocket_manager.py

Manages active WebSocket client connections.
Broadcasts PredictionResult and live price updates to all connected clients.
"""

from __future__ import annotations

import json
import structlog
from fastapi import WebSocket

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Thread-safe WebSocket connection pool with graceful dead-socket cleanup."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("ws_connected", channel=self.name, total=len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info("ws_disconnected", channel=self.name, total=len(self.active))

    async def broadcast(self, data: dict) -> None:
        """Send JSON payload to all connected clients; prune dead sockets."""
        if not self.active:
            return
        msg = json.dumps(data, default=str)
        dead: list[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, data: dict) -> None:
        """Send JSON payload to a single client."""
        try:
            await ws.send_text(json.dumps(data, default=str))
        except Exception:
            self.disconnect(ws)


# Module-level singletons consumed by app.py and pipeline
prediction_manager = ConnectionManager("predictions")
price_manager = ConnectionManager("prices")

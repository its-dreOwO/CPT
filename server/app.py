"""
server/app.py

FastAPI application entry point.

Entry point:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Routes:
    GET  /                          -> dashboard (index.html)
    GET  /api/health                -> liveness
    GET  /api/status                -> engine status
    GET  /api/predictions/{coin}    -> latest prediction
    GET  /api/predictions/{coin}/history
    GET  /api/prices/latest
    WS   /ws/predictions            -> live prediction stream
    WS   /ws/prices                 -> live price tick stream
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.routes import health, predictions
from server.websocket_manager import prediction_manager, price_manager
from storage.database import Base, engine

logger = structlog.get_logger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Ensure all DB tables exist on startup
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("cpt_server_starting", static_dir=_STATIC_DIR)
    yield
    logger.info("cpt_server_stopped")


app = FastAPI(
    title="CPT — Crypto Price Terminal",
    description="Real-time SOL/DOGE price predictions via 4-engine ensemble.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static assets (JS, CSS, icons, etc.)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# API routes
app.include_router(health.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")


# ---------------------------------------------------------------------------
# Dashboard entry point
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------


@app.websocket("/ws/predictions")
async def ws_predictions(websocket: WebSocket) -> None:
    """Stream live PredictionResult updates to connected clients."""
    await prediction_manager.connect(websocket)
    try:
        while True:
            # Keep socket alive; pipeline broadcasts via prediction_manager.broadcast()
            await websocket.receive_text()
    except WebSocketDisconnect:
        prediction_manager.disconnect(websocket)


@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket) -> None:
    """Stream live price ticks (SOL/USDT, DOGE/USDT) to connected clients."""
    await price_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        price_manager.disconnect(websocket)

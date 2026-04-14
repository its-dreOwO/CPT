"""
server/routes/health.py

GET /api/health  — liveness probe
GET /api/status  — full engine status (last fetch times, model load state)
"""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter(tags=["health"])

_start_time: datetime = datetime.now(timezone.utc)
_engine_status: dict[str, dict] = {}


def update_engine_status(engine: str, ok: bool, detail: str = "") -> None:
    """Called by pipeline code after each data fetch to update engine health."""
    _engine_status[engine] = {
        "status": "ok" if ok else "error",
        "detail": detail,
        "last_fetch": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health")
async def health() -> dict:
    """Liveness probe — returns 200 while server is running."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/status")
async def status() -> dict:
    """Full engine status including last fetch times and uptime."""
    uptime = int((datetime.now(timezone.utc) - _start_time).total_seconds())

    # Default idle status when pipeline hasn't run yet
    engines = _engine_status or {
        "macro": {"status": "idle", "detail": "Not started", "last_fetch": None},
        "onchain": {"status": "idle", "detail": "Not started", "last_fetch": None},
        "sentiment": {"status": "idle", "detail": "Not started", "last_fetch": None},
        "ml": {"status": "idle", "detail": "Not started", "last_fetch": None},
    }

    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "started_at": _start_time.isoformat(),
        "engines": engines,
    }

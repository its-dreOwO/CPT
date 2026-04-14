"""
server/routes/predictions.py

GET /api/predictions/{coin}          — latest prediction for SOL or DOGE
GET /api/predictions/{coin}/history  — last N predictions
GET /api/prices/latest               — most recent OHLCV candle per coin
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from storage.database import SessionLocal
from storage.models import Prediction, PriceData
from storage import prediction_repository

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["predictions"])


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------


def _get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred_to_dict(pred: Prediction) -> dict:
    return {
        "coin": pred.coin,
        "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
        "direction": pred.direction,
        "magnitude_pct": pred.magnitude_pct,
        "confidence": pred.confidence,
        "target_24h": pred.target_24h,
        "target_72h": pred.target_72h,
        "target_7d": pred.target_7d,
        "metadata": pred.metadata_json,
    }


def _demo_prediction(coin: str) -> dict:
    """Placeholder returned when no predictions exist in the DB yet."""
    return {
        "coin": coin,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": "neutral",
        "magnitude_pct": 0.0,
        "confidence": 0.0,
        "target_24h": None,
        "target_72h": None,
        "target_7d": None,
        "metadata": None,
        "_demo": True,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/predictions/{coin}")
async def get_prediction(coin: str, db: Session = Depends(_get_db)) -> dict:
    """Return the most recent prediction for the given coin."""
    pred = prediction_repository.get_latest(db, coin.upper())
    if pred is None:
        return _demo_prediction(coin.upper())
    return _pred_to_dict(pred)


@router.get("/predictions/{coin}/history")
async def get_history(coin: str, limit: int = 20, db: Session = Depends(_get_db)) -> list[dict]:
    """Return the last N predictions for the given coin."""
    preds = prediction_repository.get_history(db, coin.upper(), limit=limit)
    if not preds:
        return [_demo_prediction(coin.upper())]
    return [_pred_to_dict(p) for p in preds]


@router.get("/prices/latest")
async def get_latest_prices(db: Session = Depends(_get_db)) -> dict:
    """Return the most recent OHLCV candle for SOL and DOGE."""
    result: dict[str, dict] = {}
    for coin in ("SOL", "DOGE"):
        row = db.execute(
            select(PriceData)
            .where(PriceData.coin == coin)
            .order_by(desc(PriceData.timestamp))
            .limit(1)
        ).scalar_one_or_none()
        if row:
            result[coin] = {
                "price": row.close,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "volume": row.volume,
                "timestamp": row.timestamp.isoformat(),
            }
    return result

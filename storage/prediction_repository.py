import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from typing import List, Optional
from storage.models import Prediction
from utils.validators import PredictionResult

def save(session: Session, result: PredictionResult) -> Prediction:
    pred = Prediction(
        coin=result.coin,
        timestamp=result.timestamp,
        direction=result.direction,
        magnitude_pct=result.magnitude_pct,
        confidence=result.confidence,
        target_24h=result.target_24h,
        target_72h=result.target_72h,
        target_7d=result.target_7d,
        metadata_json={"weights": result.ensemble_weights_used} if result.ensemble_weights_used else None
    )
    session.add(pred)
    return pred

def get_latest(session: Session, coin: str) -> Optional[Prediction]:
    return session.execute(
        select(Prediction).where(Prediction.coin == coin).order_by(desc(Prediction.timestamp)).limit(1)
    ).scalar_one_or_none()

def get_history(session: Session, coin: str, limit: int = 50) -> List[Prediction]:
    return list(session.execute(
        select(Prediction).where(Prediction.coin == coin).order_by(desc(Prediction.timestamp)).limit(limit)
    ).scalars().all())

import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import List
from storage.models import PriceData


def upsert_candle(
    session: Session,
    coin: str,
    timestamp: datetime.datetime,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> PriceData:
    existing = session.execute(
        select(PriceData).where(PriceData.coin == coin, PriceData.timestamp == timestamp)
    ).scalar_one_or_none()

    if existing:
        existing.open = open
        existing.high = high
        existing.low = low
        existing.close = close
        existing.volume = volume
        return existing
    else:
        new_candle = PriceData(
            coin=coin,
            timestamp=timestamp,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        session.add(new_candle)
        return new_candle


def get_range(
    session: Session, coin: str, start_time: datetime.datetime, end_time: datetime.datetime
) -> List[PriceData]:
    result = session.execute(
        select(PriceData)
        .where(
            PriceData.coin == coin,
            PriceData.timestamp >= start_time,
            PriceData.timestamp <= end_time,
        )
        .order_by(PriceData.timestamp)
    )
    return list(result.scalars().all())

"""
scripts/backfill_prices.py

Backfills historical hourly OHLCV data for SOL and DOGE using yfinance.
Saves to the local SQLite database via price_repository.

Usage:
    python scripts/backfill_prices.py --coins SOL DOGE --days 730
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import structlog
import pandas as pd
import yfinance as yf

from config.constants import YFINANCE_TICKERS
from config.logging_config import setup_logging
from storage.database import get_session, Base, engine
from storage.price_repository import upsert_candle

logger = structlog.get_logger(__name__)

# yfinance caps hourly data at 730 days
_MAX_HOURLY_DAYS = 730


def _fetch_ohlcv(coin: str, days: int) -> pd.DataFrame:
    """Fetch hourly OHLCV from yfinance for the given coin and day count.

    Args:
        coin: 'SOL' or 'DOGE'.
        days: Number of days of history to fetch (capped at 730 for hourly).

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume and UTC index.
    """
    symbol = YFINANCE_TICKERS.get(coin.upper())
    if not symbol:
        raise ValueError(f"Unknown coin '{coin}'. Supported: {list(YFINANCE_TICKERS)}")

    actual_days = min(days, _MAX_HOURLY_DAYS)
    if days > _MAX_HOURLY_DAYS:
        logger.warning(
            "backfill_days_capped",
            coin=coin,
            requested=days,
            capped=actual_days,
            reason="yfinance hourly limit is 730 days",
        )

    logger.info("backfill_fetching", coin=coin, symbol=symbol, days=actual_days)
    hist = yf.Ticker(symbol).history(period=f"{actual_days}d", interval="1h")

    if hist.empty:
        logger.warning("backfill_empty_response", coin=coin, symbol=symbol)
        return pd.DataFrame()

    hist.index = pd.to_datetime(hist.index, utc=True)
    return hist


def backfill_coin(coin: str, days: int) -> int:
    """Fetch and persist OHLCV history for a single coin.

    Args:
        coin: 'SOL' or 'DOGE'.
        days: Days of history to backfill.

    Returns:
        Number of candles inserted/updated.
    """
    hist = _fetch_ohlcv(coin, days)
    if hist.empty:
        return 0

    count = 0
    with get_session() as session:
        for ts, row in hist.iterrows():
            upsert_candle(
                session=session,
                coin=coin.upper(),
                timestamp=ts.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            count += 1

    logger.info("backfill_coin_done", coin=coin, candles=count)
    return count


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV price data")
    parser.add_argument(
        "--coins", nargs="+", default=["SOL", "DOGE"], help="Coins to backfill (default: SOL DOGE)"
    )
    parser.add_argument(
        "--days", type=int, default=730, help="Days of history to backfill (default: 730, max: 730)"
    )
    args = parser.parse_args()

    # Ensure DB tables exist
    Base.metadata.create_all(bind=engine)

    total = 0
    for coin in args.coins:
        try:
            n = backfill_coin(coin, args.days)
            print(f"  {coin}: {n:,} candles inserted/updated")
            total += n
        except Exception as exc:
            logger.error("backfill_coin_failed", coin=coin, error=str(exc))
            print(f"  {coin}: FAILED -- {exc}")

    print(f"\nTotal: {total:,} candles")


if __name__ == "__main__":
    main()

"""
engines/macro/dxy_tracker.py

Fetches the US Dollar Index (DXY) from yfinance.
Fetch only -- no feature engineering here.
"""

import structlog
import pandas as pd
import yfinance as yf

from config.constants import DXY_TICKER

logger = structlog.get_logger(__name__)


def fetch_dxy(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch DXY (US Dollar Index) from yfinance.

    A rising DXY generally puts pressure on risk assets including crypto.
    Used as a macro headwind/tailwind signal.

    Args:
        period: yfinance period string (e.g., '2y', '1y', '6mo').
        interval: yfinance interval string (e.g., '1d', '1h').

    Returns:
        DataFrame with columns:
            dxy_close       -- closing price of DXY
            dxy_pct_change  -- period-over-period % change
        UTC DatetimeIndex. Returns empty DataFrame on failure.
    """
    logger.info("fetching_dxy", symbol=DXY_TICKER, period=period, interval=interval)

    try:
        hist = yf.Ticker(DXY_TICKER).history(period=period, interval=interval)
    except Exception as exc:
        logger.warning("dxy_fetch_failed", error=str(exc))
        return pd.DataFrame()

    if hist.empty:
        logger.warning("dxy_empty_response", symbol=DXY_TICKER)
        return pd.DataFrame()

    df = pd.DataFrame(index=hist.index)
    df.index = pd.to_datetime(df.index, utc=True)
    df["dxy_close"] = hist["Close"].values
    df["dxy_pct_change"] = df["dxy_close"].pct_change() * 100

    logger.info("dxy_fetched", rows=len(df), latest=round(float(df["dxy_close"].iloc[-1]), 2))
    return df

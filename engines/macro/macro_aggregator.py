"""
engines/macro/macro_aggregator.py

Merges all macro DataFrames onto a single UTC hourly DatetimeIndex.
If any engine fails, logs a warning and continues with partial data.
"""

import structlog
import pandas as pd

from engines.macro.fred_client import fetch_all as fetch_fred
from engines.macro.dxy_tracker import fetch_dxy
from utils.time_utils import resample_hourly

logger = structlog.get_logger(__name__)


def aggregate(observation_start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch all macro signals and merge onto a UTC hourly index.

    Sources:
        - FRED: fed_funds_rate, treasury_10y, m2_supply
        - yfinance: dxy_close, dxy_pct_change

    All DataFrames are resampled to 1h UTC and forward-filled before merging
    so that the output always has a clean hourly DatetimeIndex.

    Args:
        observation_start: Start date string in 'YYYY-MM-DD' format passed to FRED.

    Returns:
        DataFrame with UTC hourly DatetimeIndex and all macro feature columns.
        Returns empty DataFrame if every source fails.
    """
    frames: list[pd.DataFrame] = []

    # FRED: interest rates + M2
    try:
        fred_df = fetch_fred(observation_start=observation_start)
        if not fred_df.empty:
            frames.append(resample_hourly(fred_df))
            logger.info("macro_fred_merged", cols=list(fred_df.columns))
    except Exception as exc:
        logger.warning("macro_fred_failed", error=str(exc))

    # DXY: dollar strength
    try:
        dxy_df = fetch_dxy(period="2y", interval="1d")
        if not dxy_df.empty:
            frames.append(resample_hourly(dxy_df))
            logger.info("macro_dxy_merged")
    except Exception as exc:
        logger.warning("macro_dxy_failed", error=str(exc))

    if not frames:
        logger.warning("macro_aggregator_no_data")
        return pd.DataFrame()

    # Outer join all frames on the hourly UTC index, then forward-fill gaps
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, how="outer")

    result = result.ffill()

    logger.info(
        "macro_aggregated",
        rows=len(result),
        cols=list(result.columns),
        start=str(result.index.min()),
        end=str(result.index.max()),
    )
    return result


if __name__ == "__main__":
    import sys
    from config.logging_config import setup_logging

    setup_logging()
    df = aggregate()
    if df.empty:
        print("No macro data returned.")
        sys.exit(1)
    print(df.tail(5).to_string())
    print(f"\nShape: {df.shape}")

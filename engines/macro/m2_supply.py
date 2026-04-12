"""
engines/macro/m2_supply.py

M2 money supply trend analysis.
Wraps fred_client.fetch_series("M2SL") and computes expansion/contraction signal.
Fetch + light signal extraction only -- deep feature engineering lives in macro_features.py.
"""

import structlog
import pandas as pd

from engines.macro.fred_client import fetch_series

logger = structlog.get_logger(__name__)


def fetch_m2_trend(observation_start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch M2 supply and compute month-over-month and year-over-year growth.

    Args:
        observation_start: Start date string in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns:
            m2_supply    -- raw M2SL value (billions USD)
            m2_mom_pct   -- month-over-month % change
            m2_yoy_pct   -- year-over-year % change (12 periods back)
            m2_signal    -- +1 expanding, -1 contracting, 0 flat
        UTC DatetimeIndex (monthly cadence).
        Returns empty DataFrame on fetch failure.
    """
    try:
        m2 = fetch_series("M2SL", observation_start=observation_start)
    except Exception as exc:
        logger.warning("m2_fetch_failed", error=str(exc))
        return pd.DataFrame()

    df = m2.to_frame("m2_supply")
    df["m2_mom_pct"] = df["m2_supply"].pct_change(1) * 100
    df["m2_yoy_pct"] = df["m2_supply"].pct_change(12) * 100

    def _signal(yoy: float) -> int:
        if pd.isna(yoy):
            return 0
        if yoy > 0.5:
            return 1
        if yoy < -0.5:
            return -1
        return 0

    df["m2_signal"] = df["m2_yoy_pct"].apply(_signal)
    logger.info("m2_trend_ready", rows=len(df))
    return df

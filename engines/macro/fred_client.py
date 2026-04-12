"""
engines/macro/fred_client.py

Fetches macroeconomic series from FRED (Federal Reserve Economic Data).
Fetch only -- no feature engineering here.
"""

import structlog
import pandas as pd
from fredapi import Fred

from config.settings import settings
from config.constants import FRED_SERIES
from utils.retry import retry

logger = structlog.get_logger(__name__)


def _get_client() -> Fred:
    return Fred(api_key=settings.FRED_API_KEY)


@retry(max_attempts=3, min_wait_sec=2.0, max_wait_sec=10.0)
def fetch_series(series_id: str, observation_start: str = "2020-01-01") -> pd.Series:
    """Fetch a single FRED series as a UTC-indexed pandas Series.

    Args:
        series_id: FRED series identifier (e.g., 'FEDFUNDS').
        observation_start: Start date string in 'YYYY-MM-DD' format.

    Returns:
        pd.Series with UTC DatetimeIndex. Name is the human-readable column
        name from constants.FRED_SERIES, or the lowercased series_id.
    """
    col_name = next((k for k, v in FRED_SERIES.items() if v == series_id), series_id.lower())
    logger.info("fetching_fred_series", series_id=series_id, col_name=col_name)

    raw = _get_client().get_series(series_id, observation_start=observation_start)
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.name = col_name
    raw = raw.dropna()

    logger.info("fred_series_fetched", series_id=series_id, rows=len(raw))
    return raw


def fetch_all(observation_start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch FEDFUNDS, DGS10, and M2SL into a single DataFrame.

    Failed individual series are skipped with a warning so partial data
    still flows downstream -- consistent with data_pipeline fault-tolerance.

    Args:
        observation_start: Start date string in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns: fed_funds_rate, treasury_10y, m2_supply.
        UTC DatetimeIndex at monthly/weekly cadence (forward-filled in aggregator).
        Returns empty DataFrame if all fetches fail.
    """
    collected: dict[str, pd.Series] = {}

    for col_name, series_id in FRED_SERIES.items():
        try:
            collected[col_name] = fetch_series(series_id, observation_start)
        except Exception as exc:
            logger.warning("fred_series_skipped", series_id=series_id, error=str(exc))

    if not collected:
        logger.warning("fred_fetch_all_empty")
        return pd.DataFrame()

    df = pd.DataFrame(collected)
    logger.info("fred_fetch_all_done", cols=list(df.columns), rows=len(df))
    return df

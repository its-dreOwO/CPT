"""
engines/macro/macro_features.py

Feature engineering on top of macro_aggregator output.
No fetching or API calls here -- input is always a DataFrame from aggregate().
"""

import structlog
import numpy as np
import pandas as pd

logger = structlog.get_logger(__name__)

# Rolling window sizes (in hours)
_ZSCORE_WINDOW = 2160  # 90 days
_ZSCORE_MIN_PERIODS = 24
_LAG_7D = 168  # 7 days in hours
_LAG_30D = 720  # 30 days in hours


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate engineered feature columns from raw macro data.

    Adds for each numeric column:
        {col}_zscore    -- rolling 90-day z-score (normalised signal strength)
        {col}_lag7d     -- value 7 days ago (short-term change context)
        {col}_lag30d    -- value 30 days ago (medium-term change context)
        {col}_mom7d     -- 7-day momentum (% change)

    Also adds a composite macro_sentiment score in [-1, +1]:
        +1 = all macro signals bullish for risk assets (low rates, weak DXY)
        -1 = all macro signals bearish (high rates, strong DXY)

    Args:
        df: DataFrame from macro_aggregator.aggregate() with UTC hourly index.

    Returns:
        DataFrame with all original columns plus engineered features.
        Returns df unchanged if it is empty.
    """
    if df.empty:
        logger.warning("macro_features_empty_input")
        return df

    out = df.copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        s = df[col]
        rolling = s.rolling(window=_ZSCORE_WINDOW, min_periods=_ZSCORE_MIN_PERIODS)

        out[f"{col}_zscore"] = (s - rolling.mean()) / rolling.std().replace(0, np.nan)
        out[f"{col}_lag7d"] = s.shift(_LAG_7D)
        out[f"{col}_lag30d"] = s.shift(_LAG_30D)
        out[f"{col}_mom7d"] = s.pct_change(_LAG_7D) * 100

    # Composite macro sentiment: combines rate, DXY, and M2 signals
    # High rates -> bearish for crypto -> negative signal
    # High DXY   -> bearish for crypto -> negative signal
    # High M2    -> bullish (liquidity expanding) -> positive signal
    sentiment_parts: list[pd.Series] = []

    if "fed_funds_rate_zscore" in out:
        sentiment_parts.append(-out["fed_funds_rate_zscore"])  # invert: high rates = bad
    if "treasury_10y_zscore" in out:
        sentiment_parts.append(-out["treasury_10y_zscore"])  # invert: high yield = bad
    if "dxy_close_zscore" in out:
        sentiment_parts.append(-out["dxy_close_zscore"])  # invert: strong USD = bad
    if "m2_supply_zscore" in out:
        sentiment_parts.append(out["m2_supply_zscore"])  # keep: more money = good

    if sentiment_parts:
        raw = pd.concat(sentiment_parts, axis=1).mean(axis=1)
        # Clip to [-1, +1] using tanh-like scaling; fill early NaNs (pre-window) with 0
        out["macro_sentiment"] = (raw.clip(-3, 3) / 3).fillna(0.0)
    else:
        out["macro_sentiment"] = 0.0

    logger.info(
        "macro_features_built",
        original_cols=len(numeric_cols),
        total_cols=len(out.columns),
    )
    return out

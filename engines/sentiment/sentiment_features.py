"""
engines/sentiment/sentiment_features.py

Feature engineering on top of sentiment_aggregator output.
No fetching or scoring here — input is always a list of SentimentResult objects
(or a DataFrame of historical sentiment scores).

Adds rolling windows, momentum, and divergence signals that the ML models consume.
"""

import structlog
import pandas as pd
from typing import Sequence

from engines.sentiment.sentiment_aggregator import SentimentResult

logger = structlog.get_logger(__name__)

# Rolling window sizes (number of hourly observations)
_WINDOW_4H = 4
_WINDOW_24H = 24
_WINDOW_72H = 72

# Minimum observations required before emitting a rolling stat
_MIN_PERIODS = 3


def results_to_frame(results: Sequence[SentimentResult]) -> pd.DataFrame:
    """Convert a sequence of SentimentResult objects into a UTC-indexed DataFrame.

    Useful for building a historical sentiment time series that can be merged
    with price data in the feature builder.

    Args:
        results: Ordered sequence of SentimentResult (oldest → newest).

    Returns:
        DataFrame with columns: composite, cryptobert_score, finbert_score,
        vader_score, post_count.  Index is a UTC DatetimeIndex with freq='1h'
        if the caller spaced results hourly; otherwise unsampled.
    """
    if not results:
        return pd.DataFrame()

    records = [
        {
            "composite": r.composite,
            "cryptobert_score": r.cryptobert_score,
            "finbert_score": r.finbert_score,
            "vader_score": r.vader_score,
            "post_count": r.post_count,
        }
        for r in results
    ]
    df = pd.DataFrame(records)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate engineered sentiment feature columns from a historical score DataFrame.

    Adds for each score column:
        {col}_ma4h    -- 4-hour moving average (short-term mood)
        {col}_ma24h   -- 24-hour moving average (daily mood)
        {col}_ma72h   -- 72-hour moving average (3-day trend)
        {col}_mom24h  -- 24-hour momentum (% change in score)
        {col}_std24h  -- 24-hour rolling std (uncertainty / disagreement)

    Also adds:
        sentiment_divergence -- cryptobert_score minus finbert_score.
            Large divergence = social media and news disagree → signal ambiguity.

    Args:
        df: DataFrame with UTC hourly index containing at minimum a 'composite'
            column. Typically produced by results_to_frame() or loaded from DB.

    Returns:
        DataFrame with all original columns plus engineered features.
        Returns df unchanged if it is empty.
    """
    if df.empty:
        logger.warning("sentiment_features_empty_input")
        return df

    out = df.copy()
    score_cols = [
        c
        for c in df.columns
        if c in ("composite", "cryptobert_score", "finbert_score", "vader_score")
    ]

    for col in score_cols:
        s = df[col].fillna(0.0)  # treat missing scores as neutral

        out[f"{col}_ma4h"] = s.rolling(_WINDOW_4H, min_periods=_MIN_PERIODS).mean()
        out[f"{col}_ma24h"] = s.rolling(_WINDOW_24H, min_periods=_MIN_PERIODS).mean()
        out[f"{col}_ma72h"] = s.rolling(_WINDOW_72H, min_periods=_MIN_PERIODS).mean()
        out[f"{col}_mom24h"] = s.diff(_WINDOW_24H)
        out[f"{col}_std24h"] = s.rolling(_WINDOW_24H, min_periods=_MIN_PERIODS).std()

    # Divergence: social (CryptoBERT) vs news (FinBERT)
    if "cryptobert_score" in df.columns and "finbert_score" in df.columns:
        cb = df["cryptobert_score"].fillna(0.0)
        fb = df["finbert_score"].fillna(0.0)
        out["sentiment_divergence"] = cb - fb

    logger.info(
        "sentiment_features_built",
        original_cols=len(df.columns),
        total_cols=len(out.columns),
    )
    return out


def latest_features(result: SentimentResult, history: pd.DataFrame) -> dict[str, float]:
    """Compute sentiment features for a single new result given recent history.

    Appends the new result to the history DataFrame, runs build_features(),
    and returns the feature values for the last row as a flat dict.
    Designed for real-time use in the feature_builder pipeline.

    Args:
        result: The most recent SentimentResult from sentiment_aggregator.
        history: DataFrame of recent historical scores (at least 72 rows ideal).

    Returns:
        Dict of feature name → float value. Missing values are 0.0.
    """
    new_row = pd.DataFrame(
        [
            {
                "composite": result.composite,
                "cryptobert_score": result.cryptobert_score,
                "finbert_score": result.finbert_score,
                "vader_score": result.vader_score,
                "post_count": result.post_count,
            }
        ]
    )
    combined = pd.concat([history, new_row], ignore_index=True)
    featured = build_features(combined)
    last = featured.iloc[-1]

    feature_cols = [
        c
        for c in featured.columns
        if c not in ("post_count",)  # post_count is metadata, not a model feature
    ]
    return {col: float(last[col]) if not pd.isna(last[col]) else 0.0 for col in feature_cols}

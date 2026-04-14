"""
engines/forecasting/feature_builder.py

Single source of truth for the ML model input feature matrix.

ALL signals from every engine are registered here before being passed to
any model. When a new signal is added to any engine, it MUST be registered
in build_features() so all models see it consistently.

Input:
  - price_df:     hourly OHLCV DataFrame (UTC index) from storage
  - macro_df:     output of engines.macro.macro_features.build_features()
  - sentiment_df: output of engines.sentiment.sentiment_features.build_features()
  - onchain:      OnChainSnapshot for the coin being predicted

Output:
  - feature_df:   merged hourly DataFrame, float32, no NaNs, aligned index
"""

import structlog
import numpy as np
import pandas as pd
from typing import Optional

from engines.onchain.onchain_aggregator import OnChainSnapshot

logger = structlog.get_logger(__name__)

_PRICE_COLS = ["open", "high", "low", "close", "volume"]


def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive technical features from OHLCV columns."""
    out = df[_PRICE_COLS].copy()
    c = df["close"]

    out["return_1h"] = c.pct_change(1)
    out["return_4h"] = c.pct_change(4)
    out["return_24h"] = c.pct_change(24)
    out["return_72h"] = c.pct_change(72)

    out["vol_24h"] = c.rolling(24).std()
    out["vol_72h"] = c.rolling(72).std()

    out["ma_12h"] = c.rolling(12).mean()
    out["ma_24h"] = c.rolling(24).mean()
    out["ma_72h"] = c.rolling(72).mean()
    out["ma_168h"] = c.rolling(168).mean()

    out["ma_cross_12_24"] = out["ma_12h"] / out["ma_24h"] - 1
    out["ma_cross_24_72"] = out["ma_24h"] / out["ma_72h"] - 1

    high_24 = df["high"].rolling(24).max()
    low_24 = df["low"].rolling(24).min()
    rng = (high_24 - low_24).replace(0, np.nan)
    out["price_position_24h"] = (c - low_24) / rng

    v = df["volume"]
    out["volume_ma_24h"] = v.rolling(24).mean()
    out["volume_ratio_24h"] = v / out["volume_ma_24h"]

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    return out


def build_features(
    price_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    onchain: Optional[OnChainSnapshot] = None,
) -> pd.DataFrame:
    """Build the full feature matrix for a single coin.

    Args:
        price_df:     Hourly OHLCV DataFrame with UTC DatetimeIndex.
                      Required columns: open, high, low, close, volume.
        macro_df:     Hourly macro features from macro_features.build_features().
                      Reindexed onto price index and forward-filled.
        sentiment_df: Hourly sentiment features from sentiment_features.build_features().
                      Reindexed onto price index and forward-filled.
        onchain:      Latest OnChainSnapshot (scalar values broadcast to all rows).

    Returns:
        DataFrame with all features as float32, no NaNs, UTC hourly index.
    """
    if price_df.empty:
        logger.warning("feature_builder_empty_price_df")
        return pd.DataFrame()

    out = _price_features(price_df)

    if macro_df is not None and not macro_df.empty:
        macro_reindexed = macro_df.reindex(out.index, method="ffill")
        macro_reindexed.columns = [
            f"macro_{c}" if not c.startswith("macro_") else c for c in macro_reindexed.columns
        ]
        out = out.join(macro_reindexed, how="left")

    if sentiment_df is not None and not sentiment_df.empty:
        sent_reindexed = sentiment_df.reindex(out.index, method="ffill")
        sent_reindexed.columns = [
            f"sent_{c}" if not c.startswith("sent_") else c for c in sent_reindexed.columns
        ]
        out = out.join(sent_reindexed, how="left")

    if onchain is not None:
        if onchain.sol_tvl_usd is not None:
            out["onchain_tvl_usd"] = float(onchain.sol_tvl_usd)
        if onchain.sol_protocol_count is not None:
            out["onchain_protocol_count"] = float(onchain.sol_protocol_count)
        if onchain.sol_tps_total is not None:
            out["onchain_tps_total"] = float(onchain.sol_tps_total)
        if onchain.sol_tps_non_vote is not None:
            out["onchain_tps_non_vote"] = float(onchain.sol_tps_non_vote)
        if onchain.doge_transactions_24h is not None:
            out["onchain_txs_24h"] = float(onchain.doge_transactions_24h)
        if onchain.doge_hodling_addresses is not None:
            out["onchain_hodling_addresses"] = float(onchain.doge_hodling_addresses)
        if onchain.doge_nodes is not None:
            out["onchain_nodes"] = float(onchain.doge_nodes)
        if onchain.doge_mempool_txs is not None:
            out["onchain_mempool_txs"] = float(onchain.doge_mempool_txs)
        out["onchain_whale_signal"] = float(onchain.whale_signal)
        out["onchain_exchange_flow"] = float(onchain.exchange_flow_signal)

    out = out.ffill().fillna(0.0)
    out = out.astype(np.float32)

    logger.info("feature_builder_done", rows=len(out), cols=len(out.columns))
    return out


def get_feature_names(
    include_macro: bool = True,
    include_sentiment: bool = True,
    include_onchain: bool = True,
    coin: str = "SOL",
) -> list[str]:
    """Return expected feature column names in the order build_features() produces them."""
    names: list[str] = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1h",
        "return_4h",
        "return_24h",
        "return_72h",
        "vol_24h",
        "vol_72h",
        "ma_12h",
        "ma_24h",
        "ma_72h",
        "ma_168h",
        "ma_cross_12_24",
        "ma_cross_24_72",
        "price_position_24h",
        "volume_ma_24h",
        "volume_ratio_24h",
        "rsi_14",
    ]
    if include_macro:
        names += [
            "macro_fed_funds_rate",
            "macro_treasury_10y",
            "macro_dxy_close",
            "macro_m2_supply",
            "macro_macro_sentiment",
        ]
    if include_sentiment:
        names += [
            "sent_composite",
            "sent_composite_ma4h",
            "sent_composite_ma24h",
            "sent_composite_ma72h",
            "sent_composite_mom24h",
            "sent_sentiment_divergence",
        ]
    if include_onchain:
        if coin == "SOL":
            names += [
                "onchain_tvl_usd",
                "onchain_protocol_count",
                "onchain_tps_total",
                "onchain_tps_non_vote",
                "onchain_whale_signal",
                "onchain_exchange_flow",
            ]
        else:
            names += [
                "onchain_txs_24h",
                "onchain_hodling_addresses",
                "onchain_nodes",
                "onchain_mempool_txs",
                "onchain_whale_signal",
                "onchain_exchange_flow",
            ]
    return names

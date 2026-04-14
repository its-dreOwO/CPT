"""
scripts/train_models.py

Training entry point for all learnable ML models (LSTM, TFT, XGBoost, LightGBM).
TimesFM is zero-shot and never trained.

Loads historical price data from the database, builds the feature matrix
(technical indicators from price data), and trains the selected model(s).

Usage:
    python scripts/train_models.py --model all --coin SOL
    python scripts/train_models.py --model xgboost --coin DOGE
    python scripts/train_models.py --model lstm --coin SOL --epochs 100
"""

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import structlog
import pandas as pd

from config.constants import COINS, SEQUENCE_LENGTH
from config.logging_config import setup_logging
from storage.database import get_session
from storage.models import PriceData
from storage.price_repository import get_range
from engines.forecasting.feature_builder import build_features
from engines.forecasting import trainer

logger = structlog.get_logger(__name__)


def load_price_data(coin: str, days: int = 730) -> pd.DataFrame:
    """Load hourly OHLCV data from the database for the given coin.

    Args:
        coin: 'SOL' or 'DOGE'.
        days: How many days of history to load (default: 730 = 2 years).

    Returns:
        DataFrame with UTC DatetimeIndex and columns:
        open, high, low, close, volume.
        Returns empty DataFrame if no data found.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    with get_session() as session:
        rows = get_range(session, coin, start, end)
        # Extract data inside session context to avoid DetachedInstanceError
        data = [
            (r.timestamp, r.open, r.high, r.low, r.close, r.volume)
            for r in rows
        ]

    if not data:
        logger.warning("train_no_price_data", coin=coin, days=days)
        return pd.DataFrame()

    df = pd.DataFrame(
        data,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    logger.info("train_data_loaded", coin=coin, rows=len(df), start=df.index[0], end=df.index[-1])
    return df


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Train ML models for crypto price forecasting")
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "lstm", "tft", "xgboost", "lightgbm"],
        default="all",
        help="Model(s) to train (default: all)",
    )
    parser.add_argument(
        "--coin",
        type=str,
        choices=COINS,
        required=True,
        help="Coin to train on (SOL or DOGE)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Days of price history to load (default: 730)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for LSTM/TFT training (default: trainer defaults)",
    )
    parser.add_argument(
        "--price-only",
        action="store_true",
        default=True,
        help="Use price-derived technical features only (no macro/sentiment/onchain)",
    )
    args = parser.parse_args()

    coin = args.coin
    model_name = args.model

    print(f"\n=== Training {model_name} for {coin} ===")

    # Load price data from DB
    price_df = load_price_data(coin, days=args.days)
    if price_df.empty:
        print(f"ERROR: No price data found for {coin}. Run backfill_prices.py first.")
        sys.exit(1)

    # Check minimum data requirements
    min_rows = SEQUENCE_LENGTH * 24 + max(24, 72, 168) + 100  # ~1660 hours minimum
    if len(price_df) < min_rows:
        print(
            f"ERROR: Insufficient data. Need >= {min_rows} rows, got {len(price_df)}. "
            f"Run backfill_prices.py with more days."
        )
        sys.exit(1)

    # Build feature matrix (price-only: technical indicators from OHLCV)
    print("Building feature matrix...")
    feature_df = build_features(price_df)
    if feature_df.empty:
        print("ERROR: Feature building failed.")
        sys.exit(1)

    print(f"  Features: {len(feature_df.columns)} columns, {len(feature_df)} rows")

    # Dispatch to appropriate trainer
    print(f"\nTraining {model_name}...")

    if model_name == "xgboost":
        trainer.train_xgboost(feature_df, coin)
    elif model_name == "lightgbm":
        trainer.train_lightgbm(feature_df, coin)
    elif model_name == "lstm":
        epochs = args.epochs if args.epochs else 50
        trainer.train_lstm(feature_df, coin, epochs=epochs)
    elif model_name == "tft":
        epochs = args.epochs if args.epochs else 30
        trainer.train_tft(feature_df, coin, max_epochs=epochs)
    elif model_name == "all":
        trainer.train_all(feature_df, coin)
    else:
        print(f"ERROR: Unknown model '{model_name}'")
        sys.exit(1)

    print(f"\n=== Training complete for {coin} ({model_name}) ===")


if __name__ == "__main__":
    main()

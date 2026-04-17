"""
scripts/evaluate_models.py

Evaluation metrics for trained ML models using rolling walk-forward validation.

Loads trained model weights, runs inference on a held-out test period,
and compares predictions to actual prices. Computes MAE, RMSE,
directional accuracy, and Sharpe ratio per horizon.

Usage:
    python scripts/evaluate_models.py --coin SOL --lookback 90
    python scripts/evaluate_models.py --coin DOGE --lookback 60
"""

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import structlog
import numpy as np
import pandas as pd

from config.constants import COINS, PREDICTION_HORIZONS_HOURS
from config.logging_config import setup_logging
from storage.database import get_session
from storage.price_repository import get_range
from engines.forecasting.feature_builder import build_features
from engines.forecasting import (
    xgboost_model,
    lightgbm_model,
    lstm_model,
    transformer_model,
    evaluator as eval_module,
)

logger = structlog.get_logger(__name__)

# Models to evaluate (in order).
# TFT is excluded: its rolling evaluation requires future rows to be present
# in the input dataframe (pytorch-forecasting constraint). It works correctly
# at inference time via predictor.py which feeds the full history window.
_ALL_MODELS = ["xgboost", "lightgbm", "lstm"]


def load_price_data(coin: str, days: int = 730) -> pd.DataFrame:
    """Load hourly OHLCV data from the database."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    with get_session() as session:
        rows = get_range(session, coin, start, end)
        data = [(r.timestamp, r.open, r.high, r.low, r.close, r.volume) for r in rows]

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(
        data,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _evaluate_xgboost(feature_df: pd.DataFrame, coin: str, horizon_hours: list[int]) -> dict:
    """Evaluate XGBoost model on test data."""
    models = xgboost_model.load_latest(coin)
    if models is None:
        logger.warning("eval_xgb_no_weights", coin=coin)
        return {}

    results = {}
    for h in horizon_hours:
        # For each test point, predict and compare
        # Since XGBoost predicts price level directly
        X = feature_df.iloc[:-h].values.astype(np.float64)
        actuals = feature_df["close"].values[h:]

        if len(X) == 0:
            continue

        preds = np.array(
            [
                xgboost_model.predict(models, X[i])[f"target_{h}h" if h != 168 else "target_7d"]
                for i in range(len(X))
            ]
        )

        # current_prices for direction calc: price at prediction time
        current = feature_df["close"].values[:-h]

        results[h] = (current, actuals, preds)

    return results


def _evaluate_lightgbm(feature_df: pd.DataFrame, coin: str, horizon_hours: list[int]) -> dict:
    """Evaluate LightGBM model on test data."""
    models = lightgbm_model.load_latest(coin)
    if models is None:
        return {}

    results = {}
    for h in horizon_hours:
        X = feature_df.iloc[:-h].values.astype(np.float64)
        actuals = feature_df["close"].values[h:]

        if len(X) == 0:
            continue

        preds = np.array(
            [
                lightgbm_model.predict(models, X[i])[f"target_{h}h" if h != 168 else "target_7d"]
                for i in range(len(X))
            ]
        )
        current = feature_df["close"].values[:-h]

        results[h] = (current, actuals, preds)

    return results


def _evaluate_lstm(feature_df: pd.DataFrame, coin: str, horizon_hours: list[int]) -> dict:
    """Evaluate LSTM model on test data using sliding window."""
    model = lstm_model.load_latest(coin)
    if model is None:
        return {}

    seq_len = 168  # 7 days × 24 hours — matches training window in trainer.py
    n = len(feature_df)

    if n < seq_len + max(horizon_hours):
        logger.warning(
            "eval_lstm_insufficient_data", coin=coin, n=n, needed=seq_len + max(horizon_hours)
        )
        return {}

    results = {}
    for h in horizon_hours:
        key = f"target_{h}h" if h != 168 else "target_7d"
        current_prices = []
        actual_prices = []
        pred_prices = []

        # Sliding window: for each position, predict horizon h ahead
        for i in range(seq_len, n - max(horizon_hours)):
            seq = feature_df.iloc[i - seq_len : i].values.astype(np.float32)
            current_prices.append(feature_df["close"].iloc[i - 1])
            actual_prices.append(feature_df["close"].iloc[i - 1 + h])

            pred = lstm_model.predict(model, seq)
            pred_prices.append(pred[key])

        if len(current_prices) == 0:
            continue

        results[h] = (
            np.array(current_prices, dtype=np.float64),
            np.array(actual_prices, dtype=np.float64),
            np.array(pred_prices, dtype=np.float64),
        )

    return results


def _evaluate_tft(feature_df: pd.DataFrame, coin: str, horizon_hours: list[int]) -> dict:
    """Evaluate TFT model on test data."""
    model = transformer_model.load_latest(coin)
    if model is None:
        return {}

    # TFT needs full feature_df with 'close' column
    df = feature_df.copy()
    if "close" not in df.columns:
        return {}

    results = {}
    for h in horizon_hours:
        key = f"target_{h}h" if h != 168 else "target_7d"
        current_prices = []
        actual_prices = []
        pred_prices = []

        # Rolling prediction with TFT
        # TFT encoder needs _ENCODER_LEN // 2 to _ENCODER_LEN history
        min_history = transformer_model._ENCODER_LEN // 2
        for i in range(min_history, len(df) - max(horizon_hours)):
            window_df = df.iloc[i - min_history : i].copy()
            current_prices.append(df["close"].iloc[i - 1])
            actual_prices.append(df["close"].iloc[i - 1 + h])

            try:
                pred = transformer_model.predict(model, window_df, coin)
                pred_prices.append(pred[key])
            except Exception:
                pred_prices.append(df["close"].iloc[i - 1])  # fallback

        if len(current_prices) == 0:
            continue

        results[h] = (
            np.array(current_prices, dtype=np.float64),
            np.array(actual_prices, dtype=np.float64),
            np.array(pred_prices, dtype=np.float64),
        )

    return results


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate trained ML models")
    parser.add_argument(
        "--coin",
        type=str,
        choices=COINS,
        required=True,
        help="Coin to evaluate (SOL or DOGE)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=90,
        help="Days of test period (default: 90)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Total days of price history to load (default: 730)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=_ALL_MODELS,
        help="Models to evaluate (default: all trained models)",
    )
    args = parser.parse_args()

    coin = args.coin
    lookback_days = args.lookback

    print(f"\n=== Evaluating models for {coin} (test: last {lookback_days} days) ===")

    # Load price data
    price_df = load_price_data(coin, days=args.days)
    if price_df.empty:
        print(f"ERROR: No price data for {coin}. Run backfill_prices.py first.")
        sys.exit(1)

    # Build feature matrix
    print("Building feature matrix...")
    feature_df = build_features(price_df)
    if feature_df.empty:
        print("ERROR: Feature building failed.")
        sys.exit(1)

    print(f"  Features: {len(feature_df.columns)} columns, {len(feature_df)} rows")
    print(f"  Period: {feature_df.index[0]} to {feature_df.index[-1]}")

    # Split: last lookback_days = test, rest = train (for context)
    test_start = feature_df.index[-1] - timedelta(days=lookback_days)
    test_df = feature_df[feature_df.index >= test_start]
    print(f"  Test set: {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")

    horizon_hours = PREDICTION_HORIZONS_HOURS

    # Evaluate each model
    model_evals = {
        "xgboost": _evaluate_xgboost,
        "lightgbm": _evaluate_lightgbm,
        "lstm": _evaluate_lstm,
        "tft": _evaluate_tft,
    }

    reports = []

    for model_name in args.models:
        if model_name not in model_evals:
            print(f"  Skipping unknown model: {model_name}")
            continue

        print(f"\n--- Evaluating {model_name} ---")

        # Use test set data for evaluation
        try:
            horizon_data = model_evals[model_name](test_df, coin, horizon_hours)
        except Exception as exc:
            logger.error("eval_model_failed", model=model_name, coin=coin, error=str(exc))
            print(f"  FAILED: {exc}")
            continue

        if not horizon_data:
            print("  No results (model not trained or insufficient data)")
            continue

        report = eval_module.evaluate(coin, model_name, horizon_data)
        reports.append(report)

    # Summary
    if not reports:
        print("\nNo models could be evaluated.")
        return

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    best_report = eval_module.compare_reports(reports)

    for r in reports:
        passes = r.passes_threshold()
        marker = "PASS" if passes else "FAIL"
        print(f"\n  {r.model.upper():12s} [{marker}]")
        print(f"    Mean Dir. Accuracy: {r.mean_directional_accuracy:.4f} (target: >0.55)")
        print(f"    Mean Sharpe:        {r.mean_sharpe:.4f} (target: >1.0)")
        for h, m in sorted(r.horizons.items()):
            print(
                f"      {h:3d}h: MAE={m.mae:.6f}  RMSE={m.rmse:.6f}  "
                f"DirAcc={m.directional_accuracy:.4f}  Sharpe={m.sharpe:.4f}  (n={m.n_samples})"
            )

    if best_report:
        print(
            f"\n  Best model: {best_report.model.upper()} "
            f"(dir. accuracy: {best_report.mean_directional_accuracy:.4f})"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()

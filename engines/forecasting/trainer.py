"""
engines/forecasting/trainer.py

Training entry point for all learnable models (LSTM, TFT, XGBoost, LightGBM).
TimesFM is zero-shot and never imported here.

Called ONLY from scripts/train_models.py. Never imported by pipeline code.

Usage (via scripts/train_models.py):
    python scripts/train_models.py --model all --coin SOL
    python scripts/train_models.py --model lstm --coin DOGE
"""

import structlog
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Literal

from config.constants import PREDICTION_HORIZONS_HOURS, SEQUENCE_LENGTH

logger = structlog.get_logger(__name__)

ModelName = Literal["lstm", "tft", "xgboost", "lightgbm", "all"]


def _make_targets(close: np.ndarray) -> dict[int, np.ndarray]:
    """Build target arrays: future close price at each horizon offset."""
    targets: dict[int, np.ndarray] = {}
    for h in PREDICTION_HORIZONS_HOURS:
        future = np.roll(close, -h).astype(np.float32)
        targets[h] = future[:-h]  # drop last h rows (no future known)
    return targets


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Slide a window over X to produce (n_samples, seq_len, n_features)."""
    n = len(X) - seq_len
    return np.stack([X[i : i + seq_len] for i in range(n)], axis=0)


def train_xgboost(feature_df: pd.DataFrame, coin: str) -> None:
    from engines.forecasting import xgboost_model

    X = feature_df.values.astype(np.float32)
    close = feature_df["close"].values.astype(np.float32)
    y = _make_targets(close)
    # Trim X to match target length
    min_len = min(len(v) for v in y.values())
    X_trim = X[:min_len]
    y_trim = {h: v[:min_len] for h, v in y.items()}
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    xgboost_model.train(X_trim, y_trim, coin, date_tag)
    logger.info("trainer_xgb_done", coin=coin)


def train_lightgbm(feature_df: pd.DataFrame, coin: str) -> None:
    from engines.forecasting import lightgbm_model

    X = feature_df.values.astype(np.float32)
    close = feature_df["close"].values.astype(np.float32)
    y = _make_targets(close)
    min_len = min(len(v) for v in y.values())
    X_trim = X[:min_len]
    y_trim = {h: v[:min_len] for h, v in y.items()}
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    lightgbm_model.train(X_trim, y_trim, coin, date_tag)
    logger.info("trainer_lgbm_done", coin=coin)


def train_lstm(feature_df: pd.DataFrame, coin: str, epochs: int = 50) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from engines.forecasting import lstm_model
    from config.constants import LSTM_CONFIG

    X_raw = feature_df.values.astype(np.float32)

    # Z-score normalise every feature column so LSTM gradients stay on a
    # consistent scale.  Without this, 'close' (~$150) vs 'return_1h' (~0.01)
    # differ by 4+ orders of magnitude and the model fails to converge.
    feat_mean = X_raw.mean(axis=0)
    feat_std = X_raw.std(axis=0) + 1e-8
    X_full = (X_raw - feat_mean) / feat_std

    close = feature_df["close"].values.astype(np.float32)
    y_flat = _make_targets(close)

    # Use shorter sequence for training (7 days = 168 hours) to keep tensor size manageable.
    # Model still learns temporal patterns; at inference time it receives the full
    # SEQUENCE_LENGTH * 24 window via predictor.py.
    seq_len = 168  # 7 days × 24 hours
    max_h = max(PREDICTION_HORIZONS_HOURS)
    n_usable = len(X_full) - seq_len - max_h
    if n_usable < 100:
        logger.warning("trainer_lstm_insufficient_data", n_usable=n_usable)
        return

    # Subsample to keep DataLoader manageable: use every 6th position
    step = 6
    indices = list(range(0, n_usable, step))
    logger.info("trainer_lstm_building", coin=coin, n_total=n_usable, n_samples=len(indices), seq_len=seq_len)

    X_seq = np.stack([X_full[i : i + seq_len] for i in indices], axis=0)
    y_tensors = {}
    for h in PREDICTION_HORIZONS_HOURS:
        if h in y_flat:
            vals = [y_flat[h][i] for i in indices]
            y_tensors[h] = torch.tensor(vals, dtype=torch.float32)

    model = lstm_model.build_model(input_size=X_full.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(LSTM_CONFIG["learning_rate"]))
    criterion = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        *[y_tensors[h] for h in PREDICTION_HORIZONS_HOURS],
    )
    loader = DataLoader(dataset, batch_size=int(LSTM_CONFIG["batch_size"]), shuffle=True)

    logger.info("trainer_lstm_start", coin=coin, epochs=epochs, batches_per_epoch=len(loader))

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            x_b = batch[0].to(lstm_model.DEVICE)
            preds = model(x_b)
            loss = sum(
                criterion(preds[h], batch[i + 1].to(lstm_model.DEVICE))
                for i, h in enumerate(PREDICTION_HORIZONS_HOURS)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        logger.info("lstm_epoch", coin=coin, epoch=epoch + 1, loss=round(avg_loss, 6))

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    # Save normalisation stats alongside the weights so predict() can reproduce them
    lstm_model.save(model, coin, date_tag, mean=feat_mean, std=feat_std)
    logger.info("trainer_lstm_done", coin=coin)


def train_tft(feature_df: pd.DataFrame, coin: str, max_epochs: int = 30) -> None:
    import lightning.pytorch as pl
    from engines.forecasting import transformer_model

    # Use shorter encoder length for training (168 hours = 7 days) to keep it fast.
    # Inference uses the full _ENCODER_LEN.
    train_encoder_len = 168
    dataset = transformer_model._make_dataset(feature_df, coin, is_train=True, encoder_len=train_encoder_len)
    val_cutoff = int(len(feature_df) * 0.9)
    val_df = feature_df.iloc[val_cutoff:].copy()
    val_dataset = transformer_model._make_dataset(val_df, coin, is_train=False, encoder_len=train_encoder_len)

    train_loader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    logger.info("trainer_tft_start", coin=coin, train_size=len(dataset), val_size=len(val_dataset))

    model = transformer_model.build_model(dataset)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if transformer_model.DEVICE == "gpu" else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model, train_loader, val_loader)
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    transformer_model.save(trainer, coin, date_tag)
    logger.info("trainer_tft_done", coin=coin)


def train_all(feature_df: pd.DataFrame, coin: str) -> None:
    """Train all learnable models for a coin sequentially."""
    logger.info("trainer_start_all", coin=coin, rows=len(feature_df))
    train_xgboost(feature_df, coin)
    train_lightgbm(feature_df, coin)
    train_lstm(feature_df, coin)
    train_tft(feature_df, coin)
    logger.info("trainer_all_done", coin=coin)

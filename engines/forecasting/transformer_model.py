"""
engines/forecasting/transformer_model.py

Temporal Fusion Transformer (TFT) via pytorch-forecasting.
Multi-horizon attention model: uses price, macro, sentiment, on-chain features.
Weights saved as tft_{coin}_{YYYYMMDD}.ckpt
"""

import os
import glob
import structlog
import pandas as pd
import torch
import lightning.pytorch as pl
from typing import Optional
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE

from config.constants import PREDICTION_HORIZONS_HOURS, MODELS_DIR, SEQUENCE_LENGTH

logger = structlog.get_logger(__name__)
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# Predict up to max horizon (7d = 168h)
_MAX_HORIZON = max(PREDICTION_HORIZONS_HOURS)
_ENCODER_LEN = SEQUENCE_LENGTH * 24  # 60 days of hourly history


def _make_dataset(df: pd.DataFrame, coin: str, is_train: bool = True) -> TimeSeriesDataSet:
    """Wrap feature DataFrame into a pytorch-forecasting TimeSeriesDataSet."""
    df = df.copy()
    df["time_idx"] = range(len(df))
    df["group"] = coin
    df["target"] = df["close"].astype(float)

    time_varying = [c for c in df.columns if c not in ("time_idx", "group", "target")]

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        min_encoder_length=_ENCODER_LEN // 2,
        max_encoder_length=_ENCODER_LEN,
        min_prediction_length=1,
        max_prediction_length=_MAX_HORIZON,
        time_varying_unknown_reals=time_varying,
        allow_missing_timesteps=True,
    )


def build_model(dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.001,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=MAE(),
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )


def save(trainer: pl.Trainer, coin: str, date_tag: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"tft_{coin.lower()}_{date_tag}.ckpt")
    trainer.save_checkpoint(path)
    logger.info("tft_saved", coin=coin, path=path)
    return path


def load_latest(coin: str) -> Optional[TemporalFusionTransformer]:
    pattern = os.path.join(MODELS_DIR, f"tft_{coin.lower()}_*.ckpt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        logger.warning("tft_no_weights", coin=coin)
        return None
    model = TemporalFusionTransformer.load_from_checkpoint(candidates[-1])
    model.eval()
    logger.info("tft_loaded", coin=coin, path=candidates[-1])
    return model


def predict(model: TemporalFusionTransformer, df: pd.DataFrame, coin: str) -> dict[str, float]:
    """Run TFT inference. Returns forecasts at the three target horizons."""
    try:
        dataset = _make_dataset(df, coin, is_train=False)
        loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        raw = model.predict(loader, mode="raw", return_index=True)
        preds = raw.output.prediction[0, :, 0].cpu().numpy()
        return {
            "target_24h": float(preds[PREDICTION_HORIZONS_HOURS[0] - 1]),
            "target_72h": float(preds[PREDICTION_HORIZONS_HOURS[1] - 1]),
            "target_7d": float(preds[PREDICTION_HORIZONS_HOURS[2] - 1]),
        }
    except Exception:
        logger.warning("tft_inference_failed", exc_info=True)
        last = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
        return {"target_24h": last, "target_72h": last, "target_7d": last}

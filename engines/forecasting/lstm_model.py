"""
engines/forecasting/lstm_model.py

PyTorch LSTM for multi-horizon price forecasting.
Architecture: 2-layer LSTM (hidden=256) -> 3 output heads (24h/72h/7d).
Input shape: (batch, seq_len, n_features).
Weights saved as lstm_{coin}_{YYYYMMDD}.pt
"""

import os
import glob
import structlog
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from config.constants import PREDICTION_HORIZONS_HOURS, LSTM_CONFIG, MODELS_DIR

logger = structlog.get_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict(
            {str(h): nn.Linear(hidden_size, 1) for h in PREDICTION_HORIZONS_HOURS}
        )

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden)
        return {h: self.heads[str(h)](last).squeeze(-1) for h in PREDICTION_HORIZONS_HOURS}


def build_model(input_size: int) -> LSTMModel:
    return LSTMModel(
        input_size=input_size,
        hidden_size=int(LSTM_CONFIG["hidden_size"]),
        num_layers=int(LSTM_CONFIG["num_layers"]),
        dropout=float(LSTM_CONFIG["dropout"]),
    ).to(DEVICE)


def save(
    model: LSTMModel,
    coin: str,
    date_tag: str,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    close_col_idx: Optional[int] = None,
) -> str:
    """Save model weights and feature-normalisation stats to a single .pt file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"lstm_{coin.lower()}_{date_tag}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": model.lstm.input_size,
            "mean": mean,
            "std": std,
            "close_col_idx": close_col_idx,
        },
        path,
    )
    logger.info("lstm_saved", coin=coin, path=path)
    return path


def load_latest(coin: str) -> Optional[LSTMModel]:
    pattern = os.path.join(MODELS_DIR, f"lstm_{coin.lower()}_*.pt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        logger.warning("lstm_no_weights", coin=coin)
        return None
    # weights_only=False: checkpoint contains numpy arrays (mean/std) which are not
    # plain tensors; suppress the FutureWarning by being explicit.
    ckpt = torch.load(candidates[-1], map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    # Attach normalisation stats so predict() can de-mean the input
    model.feat_mean: Optional[np.ndarray] = ckpt.get("mean")  # type: ignore[assignment]
    model.feat_std: Optional[np.ndarray] = ckpt.get("std")  # type: ignore[assignment]
    model.close_col_idx: Optional[int] = ckpt.get("close_col_idx")  # type: ignore[assignment]
    model.eval()
    logger.info("lstm_loaded", coin=coin, path=candidates[-1])
    return model


def predict(model: LSTMModel, X_seq: np.ndarray) -> dict[str, float]:
    """Inference on a single sequence window.

    Args:
        model: Loaded LSTMModel in eval mode.
        X_seq: (seq_len, n_features) float32 array — raw (un-normalised) features.

    Returns:
        Dict with target_24h, target_72h, target_7d as absolute prices.
    """
    seq = X_seq.astype(np.float32)

    # Capture current close price before normalisation (needed to convert log returns back)
    close_col_idx = getattr(model, "close_col_idx", None)
    current_price = float(seq[-1, close_col_idx]) if close_col_idx is not None else None

    mean = getattr(model, "feat_mean", None)
    std = getattr(model, "feat_std", None)
    if mean is not None and std is not None:
        seq = (seq - mean) / std

    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        preds = model(x)

    if current_price is not None:
        # Model outputs log returns — convert back to absolute prices
        return {
            "target_24h": float(current_price * np.exp(preds[24].cpu().item())),
            "target_72h": float(current_price * np.exp(preds[72].cpu().item())),
            "target_7d": float(current_price * np.exp(preds[168].cpu().item())),
        }
    # Fallback for old checkpoints trained on absolute prices
    return {
        "target_24h": float(preds[24].cpu().item()),
        "target_72h": float(preds[72].cpu().item()),
        "target_7d": float(preds[168].cpu().item()),
    }

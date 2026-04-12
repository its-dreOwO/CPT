"""
engines/sentiment/finbert_scorer.py

FinBERT sentiment scorer (ProsusAI/finbert).
Best for formal financial and macro news text.
Weight in ensemble: 0.3 (see config/constants.py SENTIMENT_WEIGHTS).

Input must be pre-cleaned by text_preprocessor.clean().
Model is loaded once at first call and reused (singleton).
"""

import structlog
import torch
from transformers import pipeline, Pipeline
from typing import Optional

from config.constants import FINBERT_MODEL_ID

logger = structlog.get_logger(__name__)

_DEVICE = 0 if torch.cuda.is_available() else -1
_MAX_LENGTH = 512
_BATCH_SIZE = 32

# Label -> score mapping: positive=+1, negative=-1, neutral=0
_LABEL_MAP: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}

_pipeline: Optional[Pipeline] = None


def _get_pipeline() -> Pipeline:
    """Load FinBERT pipeline (once, then cached)."""
    global _pipeline
    if _pipeline is None:
        logger.info("finbert_loading", model=FINBERT_MODEL_ID, device=_DEVICE)
        _pipeline = pipeline(  # type: ignore[call-overload]
            "sentiment-analysis",
            model=FINBERT_MODEL_ID,
            device=_DEVICE,
            truncation=True,
            max_length=_MAX_LENGTH,
        )
        logger.info("finbert_loaded")
    return _pipeline


def score(text: str) -> float:
    """Score a single pre-cleaned text with FinBERT.

    Args:
        text: Pre-cleaned text from text_preprocessor.clean().

    Returns:
        Score in [-1.0, +1.0]. 0.0 if text is empty.
    """
    if not text:
        return 0.0
    result = _get_pipeline()(text)[0]
    label = result["label"].lower()
    return _LABEL_MAP.get(label, 0.0)


def score_batch(texts: list[str]) -> list[float]:
    """Score a batch of pre-cleaned texts with FinBERT.

    Args:
        texts: List of pre-cleaned strings.

    Returns:
        List of scores in [-1.0, +1.0], same length as input.
        Empty strings score as 0.0 without going through the model.
    """
    if not texts:
        return []

    pipe = _get_pipeline()
    non_empty = [(i, t) for i, t in enumerate(texts) if t]
    scores = [0.0] * len(texts)

    if non_empty:
        indices, batch = zip(*non_empty)
        results = pipe(list(batch), batch_size=_BATCH_SIZE)
        for idx, result in zip(indices, results):
            label = result["label"].lower()
            scores[idx] = _LABEL_MAP.get(label, 0.0)

    logger.debug("finbert_batch_scored", count=len(non_empty))
    return scores

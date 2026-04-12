"""
engines/sentiment/vader_scorer.py

VADER sentiment scorer. Fast, rule-based, no GPU needed.
Good for high-volume real-time scoring where transformer latency is a bottleneck.
Weight in ensemble: 0.2 (see config/constants.py SENTIMENT_WEIGHTS).

Input must be pre-cleaned by text_preprocessor.clean().
"""

import structlog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = structlog.get_logger(__name__)

# Module-level singleton -- SentimentIntensityAnalyzer is thread-safe
_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score(text: str) -> float:
    """Score a single pre-cleaned text using VADER.

    Args:
        text: Pre-cleaned text from text_preprocessor.clean().

    Returns:
        Compound score in [-1.0, +1.0].
        0.0 if text is empty.
    """
    if not text:
        return 0.0
    compound = _get_analyzer().polarity_scores(text)["compound"]
    return float(compound)


def score_batch(texts: list[str]) -> list[float]:
    """Score a list of pre-cleaned texts using VADER.

    Args:
        texts: List of pre-cleaned strings.

    Returns:
        List of compound scores in [-1.0, +1.0], same length as input.
    """
    analyzer = _get_analyzer()
    scores = [float(analyzer.polarity_scores(t)["compound"]) if t else 0.0 for t in texts]
    logger.debug("vader_batch_scored", count=len(scores))
    return scores

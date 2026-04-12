"""
engines/sentiment/sentiment_aggregator.py

Aggregates raw text from all sources into a single sentiment score per coin.

Pipeline for each coin:
  1. Collect raw posts from scrapers (Reddit, Telegram; Twitter disabled)
  2. Clean + filter via text_preprocessor
  3. Score with CryptoBERT (weight 0.5), FinBERT (weight 0.3), VADER (weight 0.2)
  4. Apply Elon DOGE multiplier (DOGE only)
  5. Return SentimentResult with per-source scores and composite score

Weights are defined in config/constants.py: SENTIMENT_WEIGHTS.
"""

import structlog
from dataclasses import dataclass, field
from typing import Optional

from config.constants import SENTIMENT_WEIGHTS
from engines.sentiment import text_preprocessor
from engines.sentiment import vader_scorer
from engines.sentiment import reddit_scraper
from engines.sentiment import finbert_scorer
from engines.sentiment import cryptobert_scorer
from engines.sentiment import elon_tracker

logger = structlog.get_logger(__name__)


@dataclass
class SentimentResult:
    """Aggregated sentiment for a single coin at a single point in time."""

    coin: str
    composite: float  # Final weighted score in [-1.0, +1.0]
    cryptobert_score: Optional[float] = None
    finbert_score: Optional[float] = None
    vader_score: Optional[float] = None
    elon_multiplier: float = 1.0
    post_count: int = 0
    sources: list[str] = field(default_factory=list)


def _mean_or_none(scores: list[float]) -> Optional[float]:
    """Return mean of non-empty list, else None."""
    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else None


def aggregate(coin: str, use_gpu_scorers: bool = True) -> SentimentResult:
    """Fetch, clean, score, and aggregate sentiment for a single coin.

    Args:
        coin: 'SOL' or 'DOGE'.
        use_gpu_scorers: If False, skip CryptoBERT and FinBERT (VADER only).
                         Useful for testing or on CPU-only machines.

    Returns:
        SentimentResult with composite score in [-1.0, +1.0].
        Returns a zero-score result if all scrapers fail.
    """
    all_texts: list[str] = []
    sources_used: list[str] = []

    # ── Reddit ──────────────────────────────────────────────────────────────
    try:
        reddit_posts = reddit_scraper.fetch_posts(coin)
        reddit_texts = reddit_scraper.extract_texts(reddit_posts)
        cleaned_reddit = text_preprocessor.clean_and_filter(reddit_texts)
        if cleaned_reddit:
            all_texts.extend(cleaned_reddit)
            sources_used.append("reddit")
            logger.info("sentiment_reddit_fetched", coin=coin, count=len(cleaned_reddit))
    except Exception:
        logger.warning("sentiment_reddit_failed", coin=coin, exc_info=True)

    # ── Telegram ────────────────────────────────────────────────────────────
    try:
        from engines.sentiment import telegram_scraper

        tg_posts = telegram_scraper.fetch_messages(coin)
        tg_texts = [p.get("text", "") for p in tg_posts if p.get("text")]
        cleaned_tg = text_preprocessor.clean_and_filter(tg_texts)
        if cleaned_tg:
            all_texts.extend(cleaned_tg)
            sources_used.append("telegram")
            logger.info("sentiment_telegram_fetched", coin=coin, count=len(cleaned_tg))
    except Exception:
        logger.warning("sentiment_telegram_failed", coin=coin, exc_info=True)

    # ── Twitter (disabled) ──────────────────────────────────────────────────
    # twitter_scraper.fetch_posts() returns [] — no texts added, no error logged here.
    # Enable when Twitter API access is upgraded.

    if not all_texts:
        logger.warning("sentiment_no_texts", coin=coin)
        return SentimentResult(coin=coin, composite=0.0, post_count=0)

    post_count = len(all_texts)

    # ── Score ────────────────────────────────────────────────────────────────
    cryptobert_score: Optional[float] = None
    finbert_score: Optional[float] = None
    vader_score: Optional[float] = None

    # VADER — always run (fast, no GPU)
    try:
        vader_scores = vader_scorer.score_batch(all_texts)
        vader_score = _mean_or_none(vader_scores)
    except Exception:
        logger.warning("sentiment_vader_failed", coin=coin, exc_info=True)

    if use_gpu_scorers:
        # CryptoBERT — social slang specialist
        try:
            cb_scores = cryptobert_scorer.score_batch(all_texts)
            cryptobert_score = _mean_or_none(cb_scores)
        except Exception:
            logger.warning("sentiment_cryptobert_failed", coin=coin, exc_info=True)

        # FinBERT — formal financial text specialist
        try:
            fb_scores = finbert_scorer.score_batch(all_texts)
            finbert_score = _mean_or_none(fb_scores)
        except Exception:
            logger.warning("sentiment_finbert_failed", coin=coin, exc_info=True)

    # ── Weighted composite ───────────────────────────────────────────────────
    weights = SENTIMENT_WEIGHTS  # {"cryptobert": 0.5, "finbert": 0.3, "vader": 0.2}
    total_weight = 0.0
    weighted_sum = 0.0

    if cryptobert_score is not None:
        weighted_sum += weights["cryptobert"] * cryptobert_score
        total_weight += weights["cryptobert"]
    if finbert_score is not None:
        weighted_sum += weights["finbert"] * finbert_score
        total_weight += weights["finbert"]
    if vader_score is not None:
        weighted_sum += weights["vader"] * vader_score
        total_weight += weights["vader"]

    composite = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # ── Elon multiplier (DOGE only) ──────────────────────────────────────────
    elon_mult = 1.0
    if coin == "DOGE":
        elon_mult = elon_tracker.get_doge_multiplier()
        composite = max(-1.0, min(1.0, composite * elon_mult))

    logger.info(
        "sentiment_aggregated",
        coin=coin,
        composite=round(composite, 4),
        cryptobert=round(cryptobert_score, 4) if cryptobert_score is not None else None,
        finbert=round(finbert_score, 4) if finbert_score is not None else None,
        vader=round(vader_score, 4) if vader_score is not None else None,
        post_count=post_count,
        sources=sources_used,
    )

    return SentimentResult(
        coin=coin,
        composite=composite,
        cryptobert_score=cryptobert_score,
        finbert_score=finbert_score,
        vader_score=vader_score,
        elon_multiplier=elon_mult,
        post_count=post_count,
        sources=sources_used,
    )


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()
    for c in ["SOL", "DOGE"]:
        result = aggregate(c, use_gpu_scorers=False)
        print(
            f"{result.coin}: composite={result.composite:.4f}  "
            f"vader={result.vader_score}  posts={result.post_count}  "
            f"sources={result.sources}"
        )

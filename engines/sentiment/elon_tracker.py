"""
engines/sentiment/elon_tracker.py

Elon Musk DOGE signal tracker.

Checks recent tweets from @elonmusk (via twitter_scraper) for DOGE-related keywords.
When a DOGE-keyword tweet is found within the last hour, returns a 3x multiplier
to amplify the DOGE sentiment score.

Twitter is currently disabled (free tier doesn't support search_recent_tweets).
This module returns a no-op multiplier (1.0) until Twitter access is restored.
See twitter_scraper.py for re-enablement instructions.

IMPORTANT: This multiplier ONLY applies to DOGE. Never call this for SOL.
"""

import structlog

logger = structlog.get_logger(__name__)

# Multiplier applied to DOGE sentiment when a fresh Elon DOGE tweet is detected
ELON_MULTIPLIER: float = 3.0

# How recent a tweet must be to trigger the multiplier (seconds)
_TWEET_FRESHNESS_SEC: int = 3600  # 1 hour


def get_doge_multiplier() -> float:
    """Return the sentiment multiplier for DOGE based on recent Elon tweets.

    When Twitter is enabled and @elonmusk has tweeted a DOGE keyword within the
    last hour, returns ELON_MULTIPLIER (3.0). Otherwise returns 1.0 (no effect).

    Twitter is currently disabled (401 Unauthorized on free tier). This function
    always returns 1.0 until Twitter access is upgraded. Once re-enabled, replace
    the early return below with the live check.

    Returns:
        3.0 if a fresh Elon DOGE tweet is detected, 1.0 otherwise.
    """
    # Twitter disabled — no-op until free-tier restriction is lifted
    # To re-enable: remove this early return and uncomment the block below
    logger.debug("elon_tracker_disabled_returning_noop")
    return 1.0

    # ── Re-enable block (unreachable until Twitter access is upgraded) ──────
    # from engines.sentiment.twitter_scraper import fetch_posts
    #
    # try:
    #     tweets = fetch_posts("DOGE", limit=10)  # Most recent 10 tweets
    #     cutoff = datetime.now(timezone.utc) - timedelta(seconds=_TWEET_FRESHNESS_SEC)
    #     for tweet in tweets:
    #         created_at = tweet.get("created_at")
    #         if created_at and created_at >= cutoff:
    #             text_lower = tweet.get("text", "").lower()
    #             if any(kw.lower() in text_lower for kw in ELON_DOGE_KEYWORDS):
    #                 logger.info("elon_doge_tweet_detected", multiplier=ELON_MULTIPLIER)
    #                 return ELON_MULTIPLIER
    # except Exception:
    #     logger.warning("elon_tracker_fetch_failed", exc_info=True)
    #
    # return 1.0
    # ────────────────────────────────────────────────────────────────────────

"""
engines/sentiment/twitter_scraper.py

Twitter/X scraper using Tweepy v2.
Currently DISABLED -- requires Twitter Basic tier ($100/mo) for search_recent_tweets.
Free tier returns 401 Unauthorized on all search endpoints.

To enable:
    1. Upgrade to Twitter Basic tier at developer.twitter.com
    2. Regenerate bearer token and set TWITTER_BEARER_TOKEN in .env
    3. Remove the early-return guard in fetch_tweets()
"""

import structlog
import tweepy
from typing import TypedDict

from config.settings import settings
from config.constants import TWITTER_SEARCH_TERMS
from utils.retry import retry
from utils.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

_rate_limiter = RateLimiter({"twitter": 300})


class Tweet(TypedDict):
    id: str
    text: str
    lang: str


@retry(max_attempts=2, min_wait_sec=5.0, max_wait_sec=15.0)
def fetch_tweets(coin: str, max_results: int = 100) -> list[Tweet]:
    """Fetch recent tweets for a coin using Twitter API v2.

    CURRENTLY DISABLED: Returns empty list until a valid Basic-tier
    bearer token is configured. All other pipeline stages are unaffected.

    Args:
        coin: 'SOL' or 'DOGE'.
        max_results: Tweets to fetch (10-100 per request on Basic tier).

    Returns:
        List of Tweet dicts, or empty list if disabled/failed.
    """
    if not settings.TWITTER_BEARER_TOKEN:
        logger.warning("twitter_scraper_disabled", reason="TWITTER_BEARER_TOKEN not set")
        return []

    # Guard: remove this block once a valid Basic-tier token is confirmed working
    logger.warning(
        "twitter_scraper_disabled",
        reason="Token present but API access unconfirmed (likely Free tier). "
        "Skipping to avoid 401 errors. Upgrade to Basic tier to enable.",
    )
    return []

    # --- Active implementation (unreachable until guard is removed) ---
    terms = TWITTER_SEARCH_TERMS.get(coin.upper(), [])
    if not terms:
        return []

    query = " OR ".join(terms) + " -is:retweet lang:en"
    client = tweepy.Client(bearer_token=settings.TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

    _rate_limiter.wait("twitter")
    response = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["lang", "created_at"],
    )

    if not response.data:
        return []

    tweets: list[Tweet] = [
        Tweet(id=str(t.id), text=t.text, lang=getattr(t, "lang", "en") or "en")
        for t in response.data
    ]
    logger.info("twitter_fetched", coin=coin, count=len(tweets))
    return tweets


def extract_texts(tweets: list[Tweet]) -> list[str]:
    """Extract raw text strings from a list of Tweet dicts.

    Args:
        tweets: List of Tweet TypedDicts.

    Returns:
        List of raw tweet text strings.
    """
    return [t["text"] for t in tweets if t.get("text")]

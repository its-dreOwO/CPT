"""
engines/sentiment/reddit_scraper.py

Scrapes Reddit using public JSON endpoints -- no API key required.
Uses httpx (not PRAW) against reddit.com/r/{sub}/hot.json.

Rate limit: ~30 req/min. Enforced via RateLimiter.
"""

import structlog
import httpx
from typing import TypedDict

from config.constants import REDDIT_SUBREDDITS
from utils.retry import retry
from utils.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

_HEADERS = {"User-Agent": "CPT-SentimentBot/1.0 (research project)"}
_BASE_URL = "https://www.reddit.com/r/{sub}/hot.json"
_rate_limiter = RateLimiter({"reddit": 30})


class RedditPost(TypedDict):
    id: str
    subreddit: str
    title: str
    selftext: str
    score: int
    num_comments: int
    url: str


@retry(max_attempts=3, min_wait_sec=2.0, max_wait_sec=8.0)
def _fetch_subreddit(subreddit: str, limit: int = 25) -> list[RedditPost]:
    """Fetch hot posts from a single subreddit.

    Args:
        subreddit: Subreddit name without r/ prefix.
        limit: Number of posts to fetch (max 100).

    Returns:
        List of RedditPost dicts. Empty list on failure.
    """
    _rate_limiter.wait("reddit")
    url = _BASE_URL.format(sub=subreddit)

    with httpx.Client(headers=_HEADERS, timeout=10.0, follow_redirects=True) as client:
        resp = client.get(url, params={"limit": limit})
        resp.raise_for_status()

    children = resp.json().get("data", {}).get("children", [])
    posts: list[RedditPost] = []
    for child in children:
        d = child.get("data", {})
        posts.append(
            RedditPost(
                id=d.get("id", ""),
                subreddit=subreddit,
                title=d.get("title", ""),
                selftext=d.get("selftext", ""),
                score=int(d.get("score", 0)),
                num_comments=int(d.get("num_comments", 0)),
                url=d.get("url", ""),
            )
        )

    logger.info("reddit_fetched", subreddit=subreddit, posts=len(posts))
    return posts


def fetch_posts(coin: str, limit_per_sub: int = 25) -> list[RedditPost]:
    """Fetch hot posts for a coin across all configured subreddits.

    Args:
        coin: 'SOL' or 'DOGE'.
        limit_per_sub: Posts to fetch per subreddit.

    Returns:
        Combined list of RedditPost dicts from all subreddits.
        Failed subreddits are skipped with a warning.
    """
    subreddits = REDDIT_SUBREDDITS.get(coin.upper(), [])
    if not subreddits:
        logger.warning("reddit_no_subreddits_configured", coin=coin)
        return []

    all_posts: list[RedditPost] = []
    for sub in subreddits:
        try:
            all_posts.extend(_fetch_subreddit(sub, limit=limit_per_sub))
        except Exception as exc:
            logger.warning("reddit_subreddit_failed", subreddit=sub, error=str(exc))

    logger.info("reddit_fetch_done", coin=coin, total_posts=len(all_posts))
    return all_posts


def extract_texts(posts: list[RedditPost]) -> list[str]:
    """Extract title + selftext from a list of posts for sentiment scoring.

    Args:
        posts: List of RedditPost dicts.

    Returns:
        List of raw text strings (title + body combined where available).
    """
    texts: list[str] = []
    for post in posts:
        parts = [post["title"]]
        if post["selftext"] and post["selftext"] not in ("[removed]", "[deleted]"):
            parts.append(post["selftext"])
        texts.append(" ".join(parts))
    return texts

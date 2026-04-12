"""
engines/sentiment/text_preprocessor.py

Cleans raw social text before passing to any scorer.
ALWAYS call clean() before scoring -- scorers expect pre-cleaned input.
"""

import re
import structlog
import emoji
from langdetect import detect, LangDetectException

logger = structlog.get_logger(__name__)

# Max chars to pass to transformer models (tokenizer hard limit is 512 tokens)
_MAX_CHARS = 1000

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")  # keep the word, drop the #
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(text: str) -> str:
    """Clean a single piece of raw social text.

    Steps:
        1. Strip control characters
        2. Remove URLs
        3. Remove @mentions
        4. Convert #hashtag -> hashtag (keep the word for sentiment)
        5. Convert emoji to text description (e.g. 🚀 -> 'rocket')
        6. Collapse whitespace
        7. Truncate to _MAX_CHARS

    Args:
        text: Raw text from Twitter, Reddit, or Telegram.

    Returns:
        Cleaned string, safe to pass to any scorer.
        Empty string if input is empty/whitespace only.
    """
    if not text or not text.strip():
        return ""

    text = _NON_ASCII_CTRL.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1 ", text)
    text = emoji.demojize(text, delimiters=(" ", " "))  # 🚀 -> " rocket "
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:_MAX_CHARS]


def is_english(text: str) -> bool:
    """Return True if langdetect thinks the text is English.

    Args:
        text: Pre-cleaned text (run clean() first).

    Returns:
        True if English or if detection fails (fail-open to avoid dropping too much).
    """
    if len(text) < 15:
        return True  # too short to detect reliably -- keep it
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True  # fail-open


def clean_and_filter(texts: list[str]) -> list[str]:
    """Clean a list of texts and drop non-English ones.

    Args:
        texts: Raw text strings from any social source.

    Returns:
        List of cleaned English strings with empty strings removed.
    """
    result: list[str] = []
    for raw in texts:
        cleaned = clean(raw)
        if cleaned and is_english(cleaned):
            result.append(cleaned)
    return result

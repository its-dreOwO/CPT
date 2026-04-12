"""
engines/sentiment/telegram_scraper.py

Scrapes public Telegram channels using Telethon.
Requires TELEGRAM_API_ID and TELEGRAM_API_HASH in .env.
A .session file is created on first run (interactive phone login required once).
"""

import asyncio
import structlog
from typing import TypedDict
from datetime import datetime, timezone, timedelta

from telethon import TelegramClient
from telethon.errors import FloodWaitError

from config.settings import settings

logger = structlog.get_logger(__name__)

# Public channels to monitor per coin
TELEGRAM_CHANNELS: dict[str, list[str]] = {
    "SOL": [
        "solana",  # Official Solana channel
        "solananews",
        "CryptoCompass",
    ],
    "DOGE": [
        "dogecoin",
        "dogecoinnews",
        "CryptoCompass",
    ],
}

_SESSION_FILE = "data/telegram_session"


class TelegramMessage(TypedDict):
    id: int
    channel: str
    text: str
    date: datetime


async def _fetch_channel(
    client: TelegramClient,
    channel: str,
    limit: int,
    since_hours: int,
) -> list[TelegramMessage]:
    """Fetch recent messages from a single public channel.

    Args:
        client: Connected Telethon client.
        channel: Channel username (without @).
        limit: Max messages to fetch.
        since_hours: Only fetch messages from the last N hours.

    Returns:
        List of TelegramMessage dicts.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    messages: list[TelegramMessage] = []

    try:
        async for msg in client.iter_messages(channel, limit=limit):
            if not msg.text:
                continue
            msg_date = (
                msg.date.replace(tzinfo=timezone.utc) if msg.date.tzinfo is None else msg.date
            )
            if msg_date < cutoff:
                break
            messages.append(
                TelegramMessage(
                    id=msg.id,
                    channel=channel,
                    text=msg.text,
                    date=msg_date,
                )
            )
    except FloodWaitError as exc:
        logger.warning("telegram_flood_wait", channel=channel, wait_seconds=exc.seconds)
        await asyncio.sleep(exc.seconds)
    except Exception as exc:
        logger.warning("telegram_channel_failed", channel=channel, error=str(exc))

    return messages


async def fetch_messages_async(
    coin: str, limit_per_channel: int = 50, since_hours: int = 4
) -> list[TelegramMessage]:
    """Fetch recent messages for a coin from all configured Telegram channels.

    Args:
        coin: 'SOL' or 'DOGE'.
        limit_per_channel: Max messages per channel.
        since_hours: Only return messages newer than this many hours.

    Returns:
        Combined list of TelegramMessage dicts.
        Returns empty list if API credentials are missing.
    """
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        logger.warning("telegram_scraper_disabled", reason="API credentials not set")
        return []

    channels = TELEGRAM_CHANNELS.get(coin.upper(), [])
    if not channels:
        return []

    all_messages: list[TelegramMessage] = []
    client = TelegramClient(
        _SESSION_FILE,
        int(settings.TELEGRAM_API_ID),
        settings.TELEGRAM_API_HASH,
    )

    try:
        await client.start()
        for channel in channels:
            msgs = await _fetch_channel(client, channel, limit_per_channel, since_hours)
            all_messages.extend(msgs)
            logger.info("telegram_channel_fetched", channel=channel, count=len(msgs))
    except Exception as exc:
        logger.error("telegram_fetch_failed", coin=coin, error=str(exc))
    finally:
        await client.disconnect()

    logger.info("telegram_fetch_done", coin=coin, total=len(all_messages))
    return all_messages


def fetch_messages(
    coin: str, limit_per_channel: int = 50, since_hours: int = 4
) -> list[TelegramMessage]:
    """Synchronous wrapper around fetch_messages_async.

    Args:
        coin: 'SOL' or 'DOGE'.
        limit_per_channel: Max messages per channel.
        since_hours: Only return messages newer than this many hours.

    Returns:
        Combined list of TelegramMessage dicts.
    """
    return asyncio.run(fetch_messages_async(coin, limit_per_channel, since_hours))


def extract_texts(messages: list[TelegramMessage]) -> list[str]:
    """Extract raw text strings from TelegramMessage dicts.

    Args:
        messages: List of TelegramMessage TypedDicts.

    Returns:
        List of raw message text strings.
    """
    return [m["text"] for m in messages if m.get("text")]

"""
config/settings.py

Single source of truth for all environment variables.
Loads from .env via Pydantic BaseSettings and exposes typed fields.

Usage:
    from config.settings import settings
    api_key = settings.FRED_API_KEY

NEVER use os.environ directly anywhere else in the codebase.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed environment configuration loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------
    # Database & Cache
    # ------------------------------------------------------------
    DATABASE_URL: str = "sqlite:///data/cpt.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------
    # Model Weights Directory
    # ------------------------------------------------------------
    MODEL_DIR: Path = Path("models")

    # ------------------------------------------------------------
    # ENGINE 1 — Macroeconomic
    # ------------------------------------------------------------
    FRED_API_KEY: str = Field(
        default="", description="FRED API key from https://fred.stlouisfed.org"
    )

    # ------------------------------------------------------------
    # ENGINE 2 — On-Chain Analytics
    # ------------------------------------------------------------
    # DeFiLlama -- free, no API key. SOL TVL and DeFi protocol metrics.
    DEFILLAMA_BASE_URL: str = Field(
        default="https://api.llama.fi",
        description="DeFiLlama API base URL (free, no key required)",
    )

    # Blockchair -- free tier, no API key. DOGE active addresses and tx volume.
    BLOCKCHAIR_BASE_URL: str = Field(
        default="https://api.blockchair.com",
        description="Blockchair API base URL (free tier, 1000 req/day)",
    )

    SOL_RPC_URL: str = Field(
        default="https://api.mainnet-beta.solana.com",
        description="Solana RPC endpoint (Alchemy or Helius recommended)",
    )

    DOGE_RPC_URL: str = Field(
        default="http://localhost:22555", description="Dogecoin Core JSON-RPC URL"
    )
    DOGE_RPC_USER: str = Field(default="", description="Dogecoin RPC username")
    DOGE_RPC_PASS: str = Field(default="", description="Dogecoin RPC password")

    # ------------------------------------------------------------
    # ENGINE 3 — Social Sentiment
    # ------------------------------------------------------------
    TWITTER_BEARER_TOKEN: str = Field(default="", description="Twitter/X API v2 Bearer token")

    REDDIT_CLIENT_ID: str = Field(default="", description="Reddit script app client ID")
    REDDIT_CLIENT_SECRET: str = Field(default="", description="Reddit script app secret")
    REDDIT_USER_AGENT: str = Field(default="CPT/1.0", description="Reddit user agent string")

    TELEGRAM_API_ID: str = Field(default="", description="Telegram API ID from my.telegram.org")
    TELEGRAM_API_HASH: str = Field(default="", description="Telegram API hash from my.telegram.org")

    # ------------------------------------------------------------
    # NOTIFICATIONS — Discord
    # ------------------------------------------------------------
    DISCORD_WEBHOOK_URL: str = Field(default="", description="Discord channel webhook URL")
    DISCORD_BOT_TOKEN: str = Field(default="", description="Optional Discord bot token")

    # ------------------------------------------------------------
    # NOTIFICATIONS — WhatsApp (Twilio)
    # ------------------------------------------------------------
    TWILIO_ACCOUNT_SID: str = Field(default="", description="Twilio account SID")
    TWILIO_AUTH_TOKEN: str = Field(default="", description="Twilio auth token")
    TWILIO_WHATSAPP_FROM: str = Field(
        default="whatsapp:+14155238886",
        description="Twilio WhatsApp sender (sandbox by default)",
    )
    TWILIO_WHATSAPP_TO: str = Field(
        default="", description="Your WhatsApp number (with country code)"
    )

    # ------------------------------------------------------------
    # NOTIFICATIONS — Zalo Official Account
    # ------------------------------------------------------------
    ZALO_APP_ID: str = Field(default="", description="Zalo app ID from developers.zalo.me")
    ZALO_APP_SECRET: str = Field(default="", description="Zalo app secret")
    ZALO_OA_ACCESS_TOKEN: str = Field(
        default="", description="Zalo OA access token (expires every 1h)"
    )
    ZALO_REFRESH_TOKEN: str = Field(default="", description="Zalo refresh token for auto-renewal")

    # ------------------------------------------------------------
    # Alert Thresholds (overridable via .env)
    # ------------------------------------------------------------
    ALERT_CONFIDENCE_THRESHOLD: float = 0.70
    ALERT_MIN_MOVE_PCT: float = 3.0
    WHALE_THRESHOLD_USD: float = 500_000.0
    ELON_DOGE_MULTIPLIER: float = 3.0

    # ------------------------------------------------------------
    # Scheduler Intervals (minutes)
    # ------------------------------------------------------------
    DATA_FETCH_INTERVAL_MIN: int = 15
    PREDICTION_INTERVAL_MIN: int = 60

    # ------------------------------------------------------------
    # Live Server
    # ------------------------------------------------------------
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    SERVER_RELOAD: bool = False  # Set True only in development


# Singleton instance — import this everywhere in the codebase.
settings = Settings()

"""
config/constants.py

Hard-coded values that don't change per environment.

Never scatter these values across engine files. If a value needs to change
at runtime based on env, put it in settings.py instead.
"""

from typing import Final

# ============================================================
# Tracked Assets
# ============================================================

COINS: Final[list[str]] = ["SOL", "DOGE"]

# CoinGecko IDs (for price data fallback)
COINGECKO_IDS: Final[dict[str, str]] = {
    "SOL": "solana",
    "DOGE": "dogecoin",
}

# ccxt symbols for Binance
CCXT_SYMBOLS: Final[dict[str, str]] = {
    "SOL": "SOL/USDT",
    "DOGE": "DOGE/USDT",
}

# yfinance tickers (fallback)
YFINANCE_TICKERS: Final[dict[str, str]] = {
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
}

# ============================================================
# Macroeconomic Data — FRED Series IDs
# ============================================================

FRED_SERIES: Final[dict[str, str]] = {
    "fed_funds_rate": "FEDFUNDS",  # Effective Federal Funds Rate
    "treasury_10y": "DGS10",  # 10-Year Treasury Constant Maturity Rate
    "m2_supply": "M2SL",  # M2 Money Supply
}

# yfinance symbol for US Dollar Index
DXY_TICKER: Final[str] = "DX-Y.NYB"

# ============================================================
# Exchange Wallets (for exchange flow detection)
# ============================================================
# These are KNOWN hot wallet addresses of major exchanges.
# Inflow to these addresses = selling pressure signal.
# KEEP THIS LIST UPDATED — wallets change as exchanges rotate addresses.

EXCHANGE_WALLETS_SOL: Final[dict[str, list[str]]] = {
    "binance": [
        # Add Binance SOL hot wallet addresses
        # Example: "2ojv9BAiHUrvsm9gxDe7fJSzbNZSJcxZvf8dqmWGHG8S",
    ],
    "coinbase": [
        # Add Coinbase SOL hot wallet addresses
    ],
    "kraken": [
        # Add Kraken SOL hot wallet addresses
    ],
}

EXCHANGE_WALLETS_DOGE: Final[dict[str, list[str]]] = {
    "binance": [
        # Add Binance DOGE hot wallet addresses
        # Example: "DHE42Ps8LSyoTb9zY6kXPgNc66ZzEBXH8A",
    ],
    "coinbase": [
        # Add Coinbase DOGE hot wallet addresses
    ],
    "kraken": [
        # Add Kraken DOGE hot wallet addresses
    ],
}

# Minimum USD value for a wallet to be classified as a whale
WHALE_THRESHOLD_USD: Final[float] = 500_000.0

# Known large SOL/DOGE wallets to monitor (populate as discovered)
# Format: plain Base58 address strings
WHALE_WALLETS_SOL: Final[list[str]] = [
    # Add known large SOL holder addresses here
]

WHALE_WALLETS_DOGE: Final[list[str]] = [
    # Add known large DOGE holder addresses here
]

# ============================================================
# Sentiment Analysis
# ============================================================

# Weighted average applied in sentiment_aggregator.py
SENTIMENT_WEIGHTS: Final[dict[str, float]] = {
    "cryptobert": 0.5,  # Best for social slang
    "finbert": 0.3,  # Best for macro/news text
    "vader": 0.2,  # Fast baseline
}

# HuggingFace model IDs — Sentiment
FINBERT_MODEL_ID: Final[str] = "ProsusAI/finbert"
CRYPTOBERT_MODEL_ID: Final[str] = "ElKulako/cryptobert"

# DOGE-specific keywords for elon_tracker.py
ELON_DOGE_KEYWORDS: Final[list[str]] = [
    "doge",
    "dogecoin",
    "shibe",
    "#dogecoin",
    "$doge",
]

# Elon Musk's Twitter handle (without @)
ELON_TWITTER_HANDLE: Final[str] = "elonmusk"

# Social media sources to scrape
TWITTER_SEARCH_TERMS: Final[dict[str, list[str]]] = {
    "SOL": ["$SOL", "#Solana", "Solana crypto"],
    "DOGE": ["$DOGE", "#Dogecoin", "Dogecoin"],
}

REDDIT_SUBREDDITS: Final[dict[str, list[str]]] = {
    "SOL": ["solana", "CryptoCurrency"],
    "DOGE": ["dogecoin", "CryptoCurrency"],
}

# ============================================================
# ML Forecasting
# ============================================================

# Sliding window for sequence models (days of history per sample)
SEQUENCE_LENGTH: Final[int] = 60

# Prediction horizons (in hours ahead from current time)
PREDICTION_HORIZONS_HOURS: Final[list[int]] = [24, 72, 168]  # 1d, 3d, 7d

# HuggingFace model ID — TimesFM (Google Time Series Foundation Model)
TIMESFM_MODEL_ID: Final[str] = "google/timesfm-2.5-200m-pytorch"
# Context length: 16,000 hourly steps (~667 days). Max horizon: 1,000 steps.
# Zero-shot: no training needed. Quantile output: 10th–90th percentile bands.
TIMESFM_CONTEXT_LEN: Final[int] = 512  # start conservative; increase after testing
TIMESFM_HORIZON_LEN: Final[int] = 168  # max horizon we need = 7d = 168 hours

# Default ensemble weights (5 models — tuned by Optuna during training)
# TimesFM gets top weight: pre-trained on 400B+ real-world time points, zero-shot
DEFAULT_ENSEMBLE_WEIGHTS: Final[dict[str, float]] = {
    "timesfm": 0.30,  # Foundation model, zero-shot, strongest raw price signal
    "tft": 0.25,  # Multi-horizon, uses all features (macro+onchain+sentiment)
    "lstm": 0.20,  # Sequence memory on full feature set
    "xgboost": 0.15,  # Regime detection, interpretable
    "lightgbm": 0.10,  # Tabular complement to XGBoost
}

# LSTM hyperparameters (defaults, tuned by Optuna)
LSTM_CONFIG: Final[dict[str, int | float]] = {
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 64,
}

# ============================================================
# API Rate Limits (requests per minute)
# ============================================================
# Used by utils/rate_limiter.py to stay within provider limits.

RATE_LIMITS_PER_MIN: Final[dict[str, int]] = {
    "fred": 120,
    "defillama": 30,  # Free, no key. Be polite.
    "blockchair": 30,  # Free tier: 1000 req/day hard cap, 30/min
    "twitter": 300,  # Basic tier
    "reddit": 30,  # Public JSON endpoints, be polite
    "discord": 30,  # Per webhook
    "solana_rpc": 100,  # Depends on provider
    "yfinance": 2000,  # No official limit, be polite
}

# ============================================================
# Cache TTLs (seconds) for storage/cache_manager.py
# ============================================================

CACHE_TTL_API_RESPONSE: Final[int] = 5 * 60  # 5 minutes
CACHE_TTL_SENTIMENT_SCORE: Final[int] = 15 * 60  # 15 minutes
CACHE_TTL_PREDICTION: Final[int] = 60 * 60  # 1 hour
CACHE_TTL_ALERT_DEDUP: Final[int] = 60 * 60  # 1 hour (prevents duplicate alerts)

# ============================================================
# Live Server
# ============================================================

# Minimum price move (%) to trigger a re-prediction immediately
LIVE_PREDICTION_TRIGGER_PCT: Final[float] = 1.0

# Maximum time (seconds) between predictions even without a price move
LIVE_PREDICTION_MAX_INTERVAL_SEC: Final[int] = 5 * 60  # 5 minutes

# WebSocket heartbeat interval (seconds)
WS_HEARTBEAT_SEC: Final[int] = 30

# ============================================================
# File Paths (relative to project root)
# ============================================================

DATA_DIR_RAW: Final[str] = "data/raw"
DATA_DIR_PROCESSED: Final[str] = "data/processed"
DATA_DIR_CACHE: Final[str] = "data/cache"
MODELS_DIR: Final[str] = "models"
LOGS_DIR: Final[str] = "logs"

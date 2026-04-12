# CLAUDE.md — CPT (Crypto Price Tracker/Predictor)

> This file is loaded automatically by Claude Code at the start of every session.
> It tells the AI assistant how this project is structured and how to work within it.

---

## Project Purpose

Predict 24h / 72h / 7d prices for **SOL (Solana)** and **DOGE (Dogecoin)** using four engines:
Macroeconomic tracking, On-Chain analytics, Social Sentiment analysis, and ML Forecasting.
Predictions are posted to **Discord**, **WhatsApp**, and **Zalo**.

---

## Always Read Before Coding

When starting any task in this project, read these files first:
- `OVERVIEW.md` — full system explanation and every-file reference
- `config/settings.py` — all environment variables and their types
- `config/constants.py` — thresholds, wallet addresses, coin IDs, weights
- `utils/retry.py` and `utils/rate_limiter.py` — required for all API calls
- `utils/time_utils.py` — required for all timestamp handling

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         server/app.py (FastAPI)      │
                    │  REST: /predictions, /health         │
                    │  WebSocket: /ws/prices, /ws/preds    │
                    └──────────────┬──────────────────────┘
                                   │ lifespan startup
                    ┌──────────────▼──────────────────────┐
                    │      pipeline/orchestrator.py        │
                    │  APScheduler inside FastAPI lifespan │
                    └──┬───────────────────┬──────────────┘
                       │                   │
          ┌────────────▼────┐   ┌──────────▼──────────────┐
          │  data_pipeline  │   │   prediction_pipeline   │
          │  (every 15 min) │   │  (every 5 min OR 1%+    │
          └────────────┬────┘   │   price move)           │
                       │        └──────────┬──────────────┘
                       ▼                   ▼
                   storage/          ensemble.py
               (SQLite + Redis)   (TimesFM+TFT+LSTM
                                   +XGB+LGBM)
                                        │
                               alert_pipeline.py
                                        │
                          Discord / WhatsApp / Zalo

engines/prices/price_stream.py  ←  ccxt WebSocket (live ticks)
Entry point: uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Module Responsibilities

### engines/macro/
- `fred_client.py` — fetch only. FRED series: FEDFUNDS, DGS10, M2SL.
- `dxy_tracker.py` — fetch only. yfinance symbol: `DX-Y.NYB`.
- `macro_aggregator.py` — merge all macro DataFrames onto UTC hourly index.
- `macro_features.py` — feature engineering only. No fetching.

### engines/onchain/
- `sol_rpc_client.py` — Solana RPC only. Endpoint: `settings.SOL_RPC_URL`.
- `doge_rpc_client.py` — Dogecoin Core JSON-RPC only.
- `whale_detector.py` — classify wallet movements. Threshold: `constants.WHALE_THRESHOLD_USD`.
- `exchange_flow.py` — compute net inflow/outflow. Wallet list: `constants.EXCHANGE_WALLETS`.
- `glassnode_client.py` — REPLACED. Now split into:
  - `defillama_client.py` — DeFiLlama REST API. SOL TVL, protocol count, DeFi activity. No API key.
  - `blockchair_client.py` — Blockchair REST API. DOGE active addresses, daily tx count. No API key.
- `onchain_aggregator.py` — merge SOL + DOGE on-chain signals into one DataFrame.

### engines/sentiment/
- Scrapers (`twitter_scraper`, `reddit_scraper`, `telegram_scraper`) — collect raw text ONLY.
  - **Reddit scraper** uses public JSON endpoints (`httpx`), NOT the Reddit API/PRAW. No API key needed.
- `text_preprocessor.py` — ALWAYS called before passing text to any scorer.
- Scorers (`finbert_scorer`, `cryptobert_scorer`, `vader_scorer`) — score pre-cleaned text ONLY.
- `elon_tracker.py` — monitors @elonmusk. Applies 3x DOGE multiplier. Never affects SOL.
- `sentiment_aggregator.py` — weighted average: CryptoBERT=0.5, FinBERT=0.3, VADER=0.2.
- `sentiment_features.py` — rolling windows and momentum features. No fetching or scoring.

### engines/prices/
- `price_stream.py` — ccxt WebSocket: subscribes to SOL/USDT and DOGE/USDT tick streams
  from Binance. On each tick, checks if price moved ≥ 1% from last prediction price.
  If so, triggers an immediate re-prediction via `prediction_pipeline.py`.
- `price_aggregator.py` — aggregates raw ticks into OHLCV candles at 1-minute and 1-hour
  resolutions for storage. Called by `price_stream.py`.

### engines/forecasting/
- `feature_builder.py` — **single source of truth** for model input features. All new signals
  must be registered here.
- `timesfm_model.py` — loads `google/timesfm-2.5-200m-pytorch` (zero-shot, no training).
  Input: raw hourly close price series (up to 16,000 points). Output: 24h/72h/7d quantile
  forecasts. GPU required for reasonable latency. Context length set in `constants.TIMESFM_CONTEXT_LEN`.
- Models (`lstm`, `transformer`, `xgboost`, `lightgbm`) — stateless at inference.
  Load weights from `models/` directory. Never retrain during prediction runs.
- `ensemble.py` — combines 5 model outputs. Returns `PredictionResult` dataclass.
  Weights: TimesFM=30%, TFT=25%, LSTM=20%, XGBoost=15%, LightGBM=10%.
  Never imports `trainer.py`.
- `trainer.py` — only imported by `scripts/train_models.py`. Never by pipeline code.
- `evaluator.py` — only imported by `scripts/evaluate_models.py`.

### server/
- `app.py` — FastAPI application. Starts the APScheduler and ccxt WebSocket in its
  `lifespan` context manager. This is the production entry point.
- `websocket_manager.py` — manages active WebSocket client connections. Broadcasts
  `PredictionResult` updates to all connected clients as JSON.
- `routes/predictions.py` — REST endpoints:
  `GET /predictions/{coin}` → latest prediction,
  `GET /predictions/{coin}/history` → last N predictions,
  `WS /ws/predictions` → stream live updates.
- `routes/health.py` — `GET /health` (liveness), `GET /status` (full engine status,
  last fetch times, model load status).

### pipeline/
- `orchestrator.py` — ONLY production entry point. APScheduler: fetch every 15 min,
  predict every 1 hour.
- `alert_pipeline.py` — sends notification only when `confidence >= 0.70 AND move >= 3%`.
  Uses Redis to deduplicate (1h TTL per coin).
- `data_pipeline.py` — if any engine fails, log warning and continue. Partial data is fine.

### notifications/
- `message_formatter.py` — ALL formatting logic lives here (Jinja2 templates). Never put
  platform-specific strings inside individual notifier files.
- `chart_generator.py` — saves PNGs to `data/cache/`. Returns file path for caller.
- `zalo_notifier.py` — handles Zalo token refresh (expires every 1h) internally.

### storage/
- All database writes go through repository classes (`price_repository`, `prediction_repository`).
  Never use raw SQLAlchemy sessions from engine code.
- `cache_manager.py` — Redis TTLs: API=5min, Sentiment=15min, Predictions=1h, Dedup=1h.

---

## Key APIs

| Service | Env Var(s) | Rate Limit | Notes |
|---------|-----------|-----------|-------|
| FRED | `FRED_API_KEY` | 120 req/min | Free -- verified working |
| DeFiLlama | None (no key) | ~30 req/min | Free -- SOL TVL, DeFi metrics |
| Blockchair | None (no key) | 30 req/min, 1000/day | Free tier -- DOGE active addresses, tx count |
| Twitter/X | `TWITTER_BEARER_TOKEN` | 500k tweets/month | Basic tier |
| Reddit | None (public JSON) | ~30 req/min | No API key -- uses `reddit.com/r/{sub}/hot.json` |
| Telegram | `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` | Soft limits | Session file required |
| Solana RPC | `SOL_RPC_URL` | Provider-dependent | Use Alchemy or Helius |
| Dogecoin RPC | `DOGE_RPC_URL`, `DOGE_RPC_USER`, `DOGE_RPC_PASS` | Local node | ~60 GB disk |
| Discord | `DISCORD_WEBHOOK_URL` | 30 req/min | Webhook only, no bot token needed |
| Twilio WhatsApp | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` | Per plan | Sandbox: +14155238886 |
| Zalo OA | `ZALO_OA_ACCESS_TOKEN`, `ZALO_APP_ID`, `ZALO_APP_SECRET`, `ZALO_REFRESH_TOKEN` | 2000/day | Token refreshes every 1h |

---

## Key APIs (additions)

| Service | Env Var(s) | Notes |
|---------|-----------|-------|
| TimesFM | HuggingFace (auto-download) | Zero-shot, no API key needed. ~800MB download on first run |
| FastAPI server | `SERVER_HOST`, `SERVER_PORT` | Default: 0.0.0.0:8000 |

## Run Commands

```bash
# Production (live server — replaces the old orchestrator command)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Development (auto-reload on file change)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# API endpoints (once server is running)
# GET  http://localhost:8000/predictions/SOL
# GET  http://localhost:8000/predictions/DOGE
# GET  http://localhost:8000/health
# GET  http://localhost:8000/status
# WS   ws://localhost:8000/ws/predictions
# WS   ws://localhost:8000/ws/prices

# CLI Interface (for now, until frontend is built)
python cli.py status                        # check server status
python cli.py predict SOL                   # check latest cached prediction
python cli.py notify enable discord         # toggle discord alerts on
python cli.py notify disable whatsapp       # toggle whatsapp alerts off

# Test individual engines
python -m engines.macro.macro_aggregator
python -m engines.onchain.onchain_aggregator
python -m engines.sentiment.sentiment_aggregator
python -m engines.forecasting.predictor

# One-time setup (run in this order)
python scripts/setup_db.py
python scripts/backfill_prices.py --coins SOL DOGE --days 1095
python scripts/train_models.py --model all --coin SOL
python scripts/train_models.py --model all --coin DOGE

# Evaluate model performance
python scripts/evaluate_models.py --lookback 90

# Shortcuts via Makefile
make run         # start production loop
make train       # train all models for both coins
make test        # run unit tests
make test-all    # run unit + integration tests
make lint        # ruff + mypy
make format      # black
```

---

## Code Conventions

### Language & Style
- Python **3.11+**. All functions must have **type hints**.
- Formatter: **Black** with `line-length=100`.
- Linter: **Ruff** (replaces flake8 + isort).
- Type checker: **mypy** (strict mode on `config/` and `utils/`).
- Docstrings: **Google style** on all public functions.
- Async: use `asyncio` throughout. Never use `threading` for I/O.

### Data Conventions
- All timestamps → UTC via `utils/time_utils.to_utc()`. No exceptions.
- All DataFrames → `DatetimeIndex` with `freq='1h'` UTC after resampling.
- All sentiment scores → normalized to `[-1.0, +1.0]` before aggregation.
- All prices → USD float.
- All model inputs → `float32` tensors or `float64` numpy arrays.

### Error Handling
- All API calls wrapped with `@retry` from `utils/retry.py`.
- Log errors with `structlog.get_logger()`. Never use bare `print()`.
- Never use bare `except Exception: pass`. Always log the exception.
- If a data source fails, return `None` and let downstream code handle it.
  The `feature_builder` handles `None` inputs gracefully with zero-fill.

### Secrets
- NEVER hardcode API keys or credentials.
- Always use `from config.settings import settings` then `settings.<KEY>`.
- `.env` is gitignored. Only commit `.env.example`.

### Model Weights
- Saved with date stamp: `lstm_sol_YYYYMMDD.pt`, `tft_doge_YYYYMMDD.ckpt`.
- `models/` directory is gitignored. Use DVC or manual backup for sharing.
- `predictor.py` always loads the most recently dated weights for each coin.

### Imports
- No circular imports. Dependency direction: `utils` ← `config` ← `engines` ← `pipeline` ← `notifications`.
- Never import from `pipeline/` inside `engines/`.
- Never import `trainer.py` outside of `scripts/`.

---

## Testing

```bash
pytest tests/unit/ -v                        # no API keys needed (mocked)
pytest tests/integration/ -v --timeout=30   # requires real .env
pytest tests/ -v -m "not live"              # safe for CI
```

- Mock all external calls in unit tests using `pytest-mock`.
- Integration tests that hit live APIs are marked `@pytest.mark.live`.
- Fixtures for common test objects (sample DataFrames, mock API responses) are in `tests/conftest.py`.

---

## Common Tasks for Claude

### Adding a New Data Signal
1. Add fetcher in the relevant engine (e.g., `engines/macro/new_signal.py`)
2. Register it in the engine's aggregator
3. Add feature engineering in the engine's `*_features.py`
4. Register the new feature column in `engines/forecasting/feature_builder.py`
5. Retrain models: `python scripts/train_models.py --model all --coin SOL`

### Live Prediction Trigger Logic
Re-prediction fires when EITHER condition is true:
1. Price moved ≥ `LIVE_PREDICTION_TRIGGER_PCT` (1%) since last prediction — triggered by `price_stream.py`
2. `LIVE_PREDICTION_MAX_INTERVAL_SEC` (5 min) elapsed — triggered by APScheduler in `orchestrator.py`

### Adding a New Notification Platform
1. Create `notifications/new_platform_notifier.py` inheriting `BaseNotifier`
2. Add a Jinja2 template in `notifications/message_formatter.py`
3. Add env vars to `config/settings.py` and `.env.example`
4. Call the new notifier from `pipeline/alert_pipeline.py`

### Debugging a Prediction Run
1. Check `logs/` for structured JSON errors from the last run
2. Run `python -m engines.forecasting.predictor` in isolation to test inference
3. Check Redis cache state with `cache_manager.get(key)` to confirm data freshness
4. Verify feature shape matches model's expected input in `feature_builder.py`

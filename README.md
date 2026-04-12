# CPT — Crypto Price Tracker / Predictor

A live forecasting server that continuously predicts price direction and magnitude for **SOL (Solana)** and **DOGE (Dogecoin)**. Streams real-time price ticks via WebSocket and re-forecasts on every 1%+ price move or every 5 minutes. Posts alerts to Discord, WhatsApp, and Zalo when confidence ≥ 70% and predicted move ≥ 3%.

---

## How It Works

```
FRED / yfinance      →  Macro Engine      ─┐
Solana / DOGE RPC    →  On-Chain Engine   ─┤→  Feature Builder  →  ML Ensemble  →  Alert
Twitter / Reddit     →  Sentiment Engine  ─┘
Binance WebSocket    →  Live Price Stream ──────────────────────────────────────→  Re-predict on 1%+ move
```

**5-model ensemble:**

| Model | Weight | Role |
|-------|--------|------|
| TimesFM 2.5 (Google) | 30% | Zero-shot foundation model, raw price signal |
| TFT (Temporal Fusion Transformer) | 25% | Multi-horizon, uses all features |
| LSTM | 20% | Sequential memory on full feature set |
| XGBoost | 15% | Regime detection, interpretable |
| LightGBM | 10% | Tabular complement to XGBoost |

Predictions are made for **24h / 72h / 7d** horizons. Alerts fire when `confidence ≥ 0.70` AND `predicted move ≥ 3%`.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Server | FastAPI + uvicorn |
| Scheduler | APScheduler (inside FastAPI lifespan) |
| Live prices | ccxt WebSocket (Binance) |
| ML models | PyTorch, pytorch-forecasting, XGBoost, LightGBM |
| Foundation model | TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`) |
| NLP / Sentiment | CryptoBERT, FinBERT, VADER |
| Database | SQLAlchemy + SQLite (PostgreSQL-ready) |
| Cache / Dedup | Redis |
| Notifications | Discord webhook, Twilio WhatsApp, Zalo OA |

---

## Project Structure

```
engines/
  macro/          # FRED, DXY, M2 supply
  onchain/        # Solana RPC, Blockchair (DOGE), DeFiLlama (SOL TVL)
  sentiment/      # Twitter, Reddit, Telegram + CryptoBERT / FinBERT / VADER
  forecasting/    # TimesFM, TFT, LSTM, XGBoost, LightGBM, ensemble
  prices/         # ccxt WebSocket live tick stream

pipeline/         # APScheduler orchestration
server/           # FastAPI app, REST + WebSocket routes
notifications/    # Discord, WhatsApp, Zalo
storage/          # SQLAlchemy ORM, Redis cache
utils/            # retry, rate limiter, time utils
config/           # settings (Pydantic), constants
```

---

## API Endpoints

```
GET  /health                        # liveness check
GET  /status                        # engine status, last fetch times
GET  /predictions/{coin}            # latest prediction for SOL or DOGE
GET  /predictions/{coin}/history    # last N predictions
WS   /ws/predictions                # live prediction stream
WS   /ws/prices                     # live price tick stream
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.11+. GPU recommended for TimesFM (CUDA 12+, ~1GB VRAM).

### 2. Configure environment

```bash
cp .env.example .env
# Fill in API keys — see .env.example for required keys
```

### 3. Initialize database

```bash
python scripts/setup_db.py
```

### 4. Backfill historical data (needed before training)

```bash
python scripts/backfill_prices.py --coins SOL DOGE --days 1095
```

### 5. Train models

```bash
python scripts/train_models.py --model all --coin SOL
python scripts/train_models.py --model all --coin DOGE
```

### 6. Start the live server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or via Makefile:

```bash
make run     # production
make dev     # development (auto-reload)
```

---

## Required API Keys

| Service | Purpose | Where to get |
|---------|---------|--------------|
| `FRED_API_KEY` | Interest rates, M2 supply | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `SOL_RPC_URL` | Solana on-chain data | [helius.dev](https://helius.dev) or [alchemy.com](https://alchemy.com) |
| `TWITTER_BEARER_TOKEN` | Crypto sentiment | [developer.twitter.com](https://developer.twitter.com) |
| `TELEGRAM_API_ID/HASH` | Telegram channel scraping | [my.telegram.org](https://my.telegram.org) |
| `DISCORD_WEBHOOK_URL` | Alert delivery | Discord Server Settings → Integrations → Webhooks |
| `TWILIO_*` | WhatsApp alerts | [console.twilio.com](https://console.twilio.com) |
| `ZALO_*` | Zalo OA alerts | [oa.zalo.me](https://oa.zalo.me) |

Reddit, DeFiLlama, and Blockchair require no API keys.

---

## Development

```bash
# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v --timeout=30

# Lint + format
ruff check . --fix
black --line-length 100 .
mypy config/ utils/

# Shortcuts
make test
make lint
make format
```

### Branch strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production — only merged from `develop` via PR |
| `develop` | Integration — all features land here |
| `feature/*` | One branch per engine/module |

---

## Build Progress

| Phase | Status |
|-------|--------|
| A — Foundation (config, utils, storage) | Complete |
| B — Data Collection (macro, prices) | In progress |
| C — Sentiment Engine | Pending |
| D — On-Chain Engine | Pending |
| E — ML Models | Pending |
| F — Live Server & Pipeline | Pending |
| G — Notifications | Pending |
| H — Testing & Hardening | Pending |

See [TASKS.md](TASKS.md) for the full task breakdown.

---

## License

Personal project — not licensed for redistribution.

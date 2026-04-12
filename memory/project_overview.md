---
name: CPT Project Overview
description: Full project context, architecture decisions, status tracker, and session notes for the Crypto Price Tracker/Predictor
type: project
---

# CPT — Project Memory
# Last Updated: 2026-04-11

---

## What This Project Does

A **live forecasting server** (FastAPI + uvicorn) that continuously predicts price direction
and magnitude for **SOL (Solana)** and **DOGE (Dogecoin)**. Streams real-time price ticks via
ccxt WebSocket and re-forecasts on every 1%+ price move (or every 5 min). Posts alerts to
Discord, WhatsApp, and Zalo when confidence ≥ 70% and predicted move ≥ 3%.
Exposes REST + WebSocket API for live prediction streaming.

**Target user:** Personal trading signals for the project owner.
**Platform:** Windows 11, Python 3.11+, runs locally via APScheduler.

---

## Build Status — Engine Completion

| Engine | Status | Notes |
|--------|--------|-------|
| Foundation (config, utils, storage) | **COMPLETE** | All A1-A13 tasks done |
| Engine 1: Macroeconomic | Not started | FRED + yfinance + DXY |
| Engine 2: On-Chain | Not started | Solana RPC + DOGE RPC + Glassnode |
| Engine 3: Sentiment | Not started | Twitter + Reddit + Telegram + NLP models |
| Engine 4: ML Forecasting | Not started | TimesFM 2.5 + LSTM + TFT + XGBoost + LightGBM + Ensemble |
| Live Price Stream | Not started | ccxt WebSocket, OHLCV aggregator |
| Live Server | Not started | FastAPI + uvicorn, REST + WebSocket API |
| Pipeline Orchestration | Not started | APScheduler inside FastAPI lifespan |
| Notifications | Not started | Discord, WhatsApp, Zalo |

---

## API Key Status

| Service | Env Var | Status |
|---------|---------|--------|
| FRED | `FRED_API_KEY` | DONE -- Obtained, verified, and loaded into .env |
| DeFiLlama | None (no key) | DONE -- Free, no key needed. Replaces Glassnode for SOL metrics |
| Blockchair | None (no key) | DONE -- Free tier, no key needed. Replaces Glassnode for DOGE metrics |
| Twitter/X | `TWITTER_BEARER_TOKEN` | Not obtained |
| Reddit | _No key needed_ | DONE -- Switched to public JSON scraping (no API key) |
| Telegram | `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | Not obtained |
| Solana RPC | `SOL_RPC_URL` | Not obtained (use Alchemy or Helius) |
| Dogecoin RPC | `DOGE_RPC_URL` / `DOGE_RPC_USER` / `DOGE_RPC_PASS` | Not obtained (local node) |
| Discord | `DISCORD_WEBHOOK_URL` | Not obtained |
| Twilio WhatsApp | `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_WHATSAPP_FROM` | Not obtained |
| Zalo OA | `ZALO_OA_ACCESS_TOKEN` / `ZALO_APP_ID` / `ZALO_APP_SECRET` / `ZALO_REFRESH_TOKEN` | Not obtained |

---

## Architecture Decisions & Rationale

### Why APScheduler instead of Airflow or Prefect?
Single-machine local deployment. No need for distributed orchestration infrastructure.
APScheduler runs in-process with zero extra setup. Revisit if deploying to a server.
**How to apply:** Keep all scheduling logic in `pipeline/orchestrator.py` only.

### Why Twilio for WhatsApp instead of the Meta Cloud API directly?
Twilio has a mature Python SDK, a free sandbox for testing, and simpler setup.
Meta Cloud API requires a Facebook Business Account verification process that can take weeks.
**How to apply:** Use Twilio sandbox (`+14155238886`) during development. Switch to a
verified sender only for production deployment.

### Why SQLite by default instead of PostgreSQL?
Local development simplicity — zero infrastructure to install. SQLAlchemy abstraction means
a single `DATABASE_URL` env var change switches to PostgreSQL with no code changes.
**How to apply:** Default `DATABASE_URL=sqlite:///data/cpt.db`. Set
`DATABASE_URL=postgresql://user:pass@localhost/cpt` in `.env` for production.

### Why CryptoBERT (weight 0.5) over FinBERT (weight 0.3) for sentiment?
CryptoBERT (`ElKulako/cryptobert`) was fine-tuned on 3.2M crypto-specific tweets. It
understands slang (HODL, moon, rekt, wen lambo, ngmi) that FinBERT misclassifies as neutral.
FinBERT is better for formal macro/news text. VADER handles real-time streaming where
transformer latency is a bottleneck.
**How to apply:** Weights are in `config/constants.py: SENTIMENT_WEIGHTS`. Adjustable
without code changes.

### Why a 3x Elon Musk multiplier for DOGE?
Historical data shows @elonmusk tweets mentioning DOGE cause 10–40% price swings within
hours. The multiplier amplifies the CryptoBERT score for DOGE only when an Elon tweet is
detected within the last 1 hour. The multiplier value is in `config/constants.py: ELON_DOGE_MULTIPLIER`.
**How to apply:** Only in `engines/sentiment/elon_tracker.py`. Never applied to SOL signals.

### Why TimesFM 2.5 gets the highest ensemble weight (30%)?
Google's TimesFM 2.5 is pre-trained on 400B+ real-world time points. It works zero-shot
with no training, supports 16,000-point context (667 days hourly), and outputs quantile
bands (10th–90th percentile). Model ID: `google/timesfm-2.5-200m-pytorch`. Uses ~1GB VRAM.
**How to apply:** Load once at server startup in `server/app.py lifespan`. Pass raw close
price array — no feature engineering. Use `freq=0` for hourly crypto data.

### Why a live server (FastAPI) instead of a pure batch scheduler?
"Always pulling data and live forecasting" requires sub-minute response to price moves.
APScheduler alone (1-hour batch) would miss flash crashes and sudden pumps. FastAPI with
ccxt WebSocket streams triggers re-prediction on every 1%+ price move regardless of schedule.
**How to apply:** Entry point is now `uvicorn server.app:app`, not `python -m pipeline.orchestrator`.
The orchestrator runs inside FastAPI's lifespan context.

### Why TFT is now 25% (was 35%)?
TFT weight reduced to make room for TimesFM (30%). TFT still provides unique value because
it uses the FULL feature set (macro + on-chain + sentiment) while TimesFM uses only raw price.
They are complementary. TFT also provides built-in variable importance across all signals.
**How to apply:** Weights in `config/constants.py: DEFAULT_ENSEMBLE_WEIGHTS`. Optuna
tunes these during `scripts/train_models.py`.

### Why Optuna for hyperparameter tuning?
Bayesian optimization converges faster than grid search on the large hyperparameter space
of LSTM (hidden size, layers, dropout, learning rate) and TFT (attention heads, hidden
size, dropout). Optuna also integrates directly with PyTorch Lightning.
**How to apply:** Tuning runs only in `scripts/train_models.py`, never in the live pipeline.

### Why Redis for caching instead of in-memory dict?
Redis TTL-based expiry is automatic and survives process restarts. Critical for the alert
deduplication use case — if the orchestrator restarts mid-hour, it won't re-send alerts.
**How to apply:** All caching goes through `storage/cache_manager.py`. Never use a plain
`dict` for TTL-based caching.

---

## Data Flow Summary

```
FRED / yfinance ──► macro_aggregator ──────────────────────┐
Solana/DOGE RPC ──► onchain_aggregator ────────────────────┤
Twitter/Reddit  ──► sentiment_aggregator ──────────────────┤
Binance OHLCV   ──► (backfill only)                        │
                                                            ▼
                                               feature_builder.py
                                                            │
                                    LSTM + TFT + XGBoost + LightGBM
                                                            │
                                                      ensemble.py
                                                            │
                                            confidence ≥ 0.70 + move ≥ 3%
                                                            │
                                     Discord / WhatsApp / Zalo
```

---

## Key File Quick Reference

| File | Purpose |
|------|---------|
| `config/settings.py` | All env vars as typed Pydantic fields |
| `config/constants.py` | Thresholds, wallet lists, coin IDs, weights |
| `utils/retry.py` | `@retry` decorator — wrap every API call |
| `utils/time_utils.py` | `to_utc()` — apply to every timestamp |
| `engines/forecasting/feature_builder.py` | Canonical model input — add all new signals here |
| `engines/forecasting/ensemble.py` | Final prediction output — `PredictionResult` dataclass |
| `engines/sentiment/elon_tracker.py` | DOGE-only Elon signal with 3x multiplier |
| `pipeline/orchestrator.py` | Production entry point — APScheduler setup |
| `notifications/message_formatter.py` | All Jinja2 message templates |
| `scripts/train_models.py` | One-time model training — never called by pipeline |

---

## Model Performance Log

Update after each `scripts/evaluate_models.py` run.

| Date | Coin | Model | 24h Dir. Accuracy | 7d MAE (USD) | Sharpe Ratio |
|------|------|-------|------------------|--------------|-------------|
| — | SOL | ensemble | TBD | TBD | TBD |
| — | DOGE | ensemble | TBD | TBD | TBD |

Target thresholds before production deployment:
- Directional accuracy > 55% (better than random)
- Sharpe ratio > 1.0 on backtested signal

---

## Known Limitations & Blockers

| Item | Detail |
|------|--------|
| Solana RPC | Public endpoints are heavily rate-limited. Alchemy or Helius paid plan recommended |
| Dogecoin node | Requires running `dogecoin-core` locally (~60 GB disk space) |
| Telegram scraper | Requires real phone number for `telethon` API registration |
| Twitter backfill | Free Twitter API only allows 7-day lookback. Academic API needed for 90+ day sentiment history |
| Model cold start | Models need at least 90 days of data before training is meaningful. Run `backfill_prices.py` first |
| Zalo token | Access token expires every 1 hour. Refresh logic is in `zalo_notifier.py` using the refresh token |

---

## Recommended Build Order

Build in this sequence to always have something runnable:

1. `config/settings.py` + `config/constants.py` — unblocks all other files
2. `utils/` (retry, rate_limiter, time_utils, crypto_utils, validators)
3. `storage/database.py` + `storage/models.py` → run `scripts/setup_db.py`
4. `engines/macro/` → test with `python -m engines.macro.macro_aggregator`
5. `scripts/backfill_prices.py` → populate 3 years of OHLCV data
6. `engines/sentiment/` (VADER + Reddit first, no GPU needed)
7. `engines/onchain/` (Glassnode first, no local node needed)
8. `engines/forecasting/` (XGBoost first, fastest to train)
9. `pipeline/orchestrator.py` → first end-to-end run
10. `notifications/discord_notifier.py` → first alert delivered

---

## Session Notes

> Append after each working session. Format: `YYYY-MM-DD: what was built, decisions made, blockers`

**2026-04-11 (session 1):** Project initialized. Created full folder tree (97 files including server/ and
engines/prices/), OVERVIEW.md, CLAUDE.md, memory file, requirements.txt, .env.example.
Config skeletons written. Integrated TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) as 5th
forecasting model with 30% ensemble weight. Switched architecture to live FastAPI server
(uvicorn) with ccxt WebSocket for real-time price ticks and re-prediction on 1%+ moves.
PyTorch CUDA confirmed working (RTX 4060 Laptop, CUDA 12.8). All 43 packages verified.
No engine code written yet. API keys still needed.

**2026-04-11 (session 2):** Started API key collection. FRED API key obtained and verified —
all 3 series (FEDFUNDS=3.64%, DGS10=4.29%, M2SL=$22,667.3B) returning data successfully.
Reddit approach changed: dropped PRAW dependency in favor of public JSON endpoint scraping
(`reddit.com/r/{sub}/hot.json`) — no API key needed, tested on r/solana, r/dogecoin,
r/CryptoCurrency, all returning posts. Updated `.env` (removed Reddit credentials),
TASKS.md, and this memory file. Remaining API keys: Discord, Twitter, Telegram,
Helius (SOL RPC), Twilio, Zalo.

**2026-04-12 (session 3):** Dropped Glassnode -- free tier no longer available.
Replaced with two fully free, no-key-needed alternatives:
- DeFiLlama (api.llama.fi): SOL chain TVL, DeFi protocol metrics. Rate limit ~30 req/min.
- Blockchair (api.blockchair.com): DOGE active addresses, daily tx count. 1000 req/day cap.
NVT and SOPR metrics dropped (were Glassnode-specific, not available for free elsewhere).
Updated: .env, .env.example, config/settings.py, config/constants.py, CLAUDE.md, OVERVIEW.md,
TASKS.md (D1 split into D1a/D1b), memory file. Skeleton file glassnode_client.py to be
renamed to defillama_client.py and blockchair_client.py when those modules are implemented.

**2026-04-12 (session 4):** Completed Phase A foundational utils (Tasks A2-A6) + Logging & DB (Tasks A7-A9):
- `time_utils.py` (aware UTC normalizations)
- `retry.py` (tenacity decorator)
- `rate_limiter.py` (token bucket)
- `crypto_utils.py` (percent change, returns, normalization)
- `validators.py` (Pydantic schemas for PredictionResult and NotificationConfig)
- `logging_config.py` (Structlog JSON setup)
- `storage/database.py` & `storage/models.py` (SQLAlchemy setup and ORM models for PriceData, Prediction, SentimentScore)
Also implemented `cli.py` at the project root to satisfy the user's request for a command-line interface to interact with the system (status, predictions, and notification toggles) until a web UI is built. Updated CLAUDE.md and TASKS.md accordingly.

**2026-04-12 (session 5):** Completed Phase A fully (A10-A13) + GitHub setup.
- `storage/price_repository.py` — `upsert_candle()`, `get_range()`
- `storage/prediction_repository.py` — `save()`, `get_latest()`, `get_history()`
- `storage/cache_manager.py` — Redis wrapper with TTL, graceful fallback if Redis not running
- `scripts/setup_db.py` — creates all ORM tables via `Base.metadata.create_all()`
- FRED API key loaded into `.env` (gitignored, not committed)
- GitHub repo created: https://github.com/its-dreOwO/CPT
- Branching strategy: `main` (production) ← `develop` (integration) ← `feature/*`
- CI workflow: `.github/workflows/ci.yml` — ruff + black + mypy + pytest on every PR
- PR template and issue templates added to `.github/`
- Branch protection rules to be configured manually on GitHub settings page

# CPT — Developer Onboarding Guide
# READ THIS BEFORE TOUCHING ANY CODE

---

## Section 1 — What This System Does

CPT (Crypto Price Tracker/Predictor) is a **live forecasting server** that continuously
predicts the price direction and magnitude of two cryptocurrencies — **SOL (Solana)** and
**DOGE (Dogecoin)** — using five AI models and four analytical data engines. It runs 24/7 as
a FastAPI server, streaming live price ticks and re-forecasting whenever the market moves.

### End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│              LIVE SERVER  (server/app.py — FastAPI + uvicorn)        │
│                                                                      │
│  REST  → GET /predictions/{coin}   WebSocket → /ws/predictions      │
│          GET /health                            /ws/prices           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ lifespan starts both:
              ┌────────────┴─────────────┐
              │                          │
┌─────────────▼──────────┐  ┌────────────▼──────────────────────────┐
│  ccxt WebSocket        │  │  APScheduler (pipeline/orchestrator)  │
│  engines/prices/       │  │                                       │
│  price_stream.py       │  │  • Macro fetch    every 15 min        │
│                        │  │  • On-chain fetch every 15 min        │
│  Live SOL/DOGE ticks   │  │  • Sentiment fetch every 15 min       │
│  from Binance WS       │  │  • Forced re-predict every 5 min      │
│                        │  │                                       │
│  Price moved ≥ 1%? ───►│  │                                       │
│  → trigger predict now │  └───────────────────────────────────────┘
└────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION                                │
│                                                                     │
│  FRED / yfinance  →  engines/macro/       (interest rates, DXY, M2)│
│  Solana/DOGE RPC  →  engines/onchain/     (whale moves, flow)       │
│  Twitter/Reddit   →  engines/sentiment/   (crowd mood, Elon signal) │
│  Binance ccxt WS  →  engines/prices/      (live ticks → OHLCV)      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE BUILDING                               │
│         engines/forecasting/feature_builder.py                      │
│   Combines all signals into a single feature matrix per coin        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 ML FORECASTING (on price move OR every 5 min)       │
│                                                                     │
│  TimesFM 2.5 (30%)  ← Google foundation model, zero-shot           │
│  TFT        (25%)   ← Multi-horizon, uses all features              │
│  LSTM       (20%)   ← Sequence memory on full feature set           │
│  XGBoost    (15%)   ← Regime detection, interpretable               │
│  LightGBM   (10%)   ← Tabular complement                            │
│                      ↓                                              │
│              engines/forecasting/ensemble.py                        │
│   Output: { coin, direction, magnitude %, confidence 0–1,          │
│             target_24h, target_72h, target_7d, quantile_bands }     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                   Broadcast to WebSocket clients immediately
                                │
                  confidence ≥ 0.70 AND predicted move ≥ 3%?
                                │ YES
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        NOTIFICATIONS                                │
│   → discord_notifier.py   (rich embeds + chart image)              │
│   → whatsapp_notifier.py  (Twilio API text + image)                │
│   → zalo_notifier.py      (Zalo OA API v2 message)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### What "Prediction" Means
- **Direction**: UP or DOWN relative to current price
- **Magnitude**: Estimated % change (e.g., +4.2%)
- **Confidence**: A 0–1 score from the ensemble. Only post alerts when ≥ 0.70
- **Horizons**: 24h, 72h, 7d forecasts are generated simultaneously

### Why SOL and DOGE Are Treated Differently
| Coin | Value Driver | Primary Signals |
|------|-------------|-----------------|
| SOL  | Utility / network usage | On-chain activity, developer growth, DeFi TVL, macro liquidity |
| DOGE | Sentiment / meme culture | Twitter/Reddit mood, Elon Musk tweets (3x multiplier), viral trends |

---

## Section 2 — Read This Before Coding (Checklist)

Complete every item before writing any code in this project. Each file is short — reading
them takes ~10 minutes total but prevents hours of debugging.

- [ ] **1. `config/settings.py`** — All environment variables live here as typed fields.
      If you need to access an API key or threshold, it comes from `settings`, not from
      `os.environ` directly. Understand what keys exist before you write any API client.

- [ ] **2. `config/constants.py`** — Hard-coded values that don't change per environment:
      known exchange wallet addresses for whale detection, coin identifiers, sentiment
      weight ratios, alert thresholds. Never scatter these in engine files.

- [ ] **3. `utils/retry.py`** — ALL external API calls must be wrapped with the `@retry`
      decorator defined here. It uses `tenacity` with exponential backoff. No bare
      `requests.get()` calls anywhere in the codebase.

- [ ] **4. `utils/rate_limiter.py`** — Token-bucket rate limiter used by all API clients
      to stay within provider limits. Import and call before sending requests.

- [ ] **5. `utils/time_utils.py`** — Every timestamp that enters the system must pass
      through `to_utc()`. All DataFrames must have UTC DatetimeIndex. This is not
      optional — mixing timezones causes silent data misalignment in feature building.

- [ ] **6. `engines/forecasting/feature_builder.py`** — The single source of truth for
      what features go into every model. If you add a new signal (e.g., a new on-chain
      metric), it gets added HERE. Never construct feature vectors ad-hoc in model files.

- [ ] **7. `storage/models.py`** — ORM schema: `PriceData`, `Prediction`, `SentimentScore`.
      Understand the table structure before writing any query or insert logic.

- [ ] **8. `pipeline/orchestrator.py`** — The main entry point. Understand the APScheduler
- [ ] **9. `server/app.py`** — understand the FastAPI lifespan: what starts, what order, how models load
- [ ] **10. `engines/prices/price_stream.py`** — understand how live ticks trigger re-predictions
      jobs and their intervals before modifying any timing logic.

---

## Section 3 — Every File Explained

### `config/`

| File | What It Does | Called By | Key Rule |
|------|-------------|-----------|----------|
| `settings.py` | Pydantic BaseSettings: loads `.env`, exposes all config as typed attributes. `settings.FRED_API_KEY`, `settings.SOL_RPC_URL`, etc. | Everything that needs config | Never call `os.environ` directly. Always use `settings.<KEY>` |
| `constants.py` | Hard-coded values: `WHALE_THRESHOLD_USD = 500_000`, exchange wallet lists, `SENTIMENT_WEIGHTS`, `ALERT_CONFIDENCE_THRESHOLD = 0.70`, coin CoinGecko IDs | All engines + pipeline | Never put numeric thresholds inside engine files |
| `logging_config.py` | Configures `structlog` for structured JSON output. Call `setup_logging()` once at startup in `orchestrator.py` | `pipeline/orchestrator.py` | Log with `structlog.get_logger()`, never with `print()` |

### `engines/macro/`

| File | What It Does | Called By | Calls Into | Key Rule |
|------|-------------|-----------|------------|----------|
| `fred_client.py` | Fetches FEDFUNDS (Fed Funds Rate), DGS10 (10yr Treasury), M2SL (M2 Money Supply) from the FRED REST API | `macro_aggregator.py` | `utils/retry.py`, `utils/rate_limiter.py` | Fetch and return raw DataFrame only. No processing here |
| `m2_supply.py` | Dedicated module for M2 money supply trend analysis (YoY growth rate) | `macro_aggregator.py` | `fred_client.py` | Thin wrapper; M2 data comes from FRED |
| `dxy_tracker.py` | Fetches DXY (US Dollar Index) from yfinance. Symbol: `DX-Y.NYB` | `macro_aggregator.py` | `utils/retry.py` | DXY inverse-correlates with crypto — rising DXY = bearish signal |
| `macro_aggregator.py` | Merges all macro DataFrames onto a shared UTC hourly DatetimeIndex. Handles forward-fill for weekly FRED data | `pipeline/data_pipeline.py` | all macro fetchers | Outputs a single DataFrame with columns: `fed_rate, dgs10, m2_supply, dxy` |
| `macro_features.py` | Engineers features from raw macro data: YoY % change, z-scores, 30/90-day rolling means, lag features | `engines/forecasting/feature_builder.py` | `macro_aggregator.py` | No fetching. Receives DataFrame, returns DataFrame with new columns |

### `engines/onchain/`

| File | What It Does | Called By | Calls Into | Key Rule |
|------|-------------|-----------|------------|----------|
| `sol_rpc_client.py` | Queries Solana JSON-RPC: `getAccountInfo`, `getSignaturesForAddress`, `getTransaction`. Endpoint from `settings.SOL_RPC_URL` | `whale_detector.py`, `exchange_flow.py` | `utils/retry.py`, `utils/rate_limiter.py` | Use Alchemy or Helius endpoint — public RPC is rate-limited and unreliable |
| `doge_rpc_client.py` | Queries Dogecoin Core via bitcoin-compatible JSON-RPC: `getbalance`, `listtransactions`, `getblockcount` | `whale_detector.py`, `exchange_flow.py` | `utils/retry.py` | Requires a local Dogecoin Core node (~60 GB). Credentials from `settings` |
| `whale_detector.py` | Identifies wallets with balance > `WHALE_THRESHOLD_USD` and classifies recent movements as ACCUMULATE / DISTRIBUTE / NEUTRAL | `onchain_aggregator.py` | `sol_rpc_client.py`, `doge_rpc_client.py` | Threshold in `config/constants.py`. Movement to exchange = bearish signal |
| `exchange_flow.py` | Computes `net_inflow = deposits - withdrawals` for known exchange hot wallets (list in `constants.py`). Positive = more selling pressure | `onchain_aggregator.py` | `sol_rpc_client.py`, `doge_rpc_client.py` | Exchange wallet addresses must be kept updated in `constants.py` |
| `defillama_client.py` | Calls DeFiLlama REST API (`api.llama.fi`): SOL chain TVL, active protocol count, DeFi inflow/outflow. No API key needed | `onchain_aggregator.py` | `utils/retry.py`, `utils/rate_limiter.py` | Free, no key. Rate-limit politely to 30 req/min |
| `blockchair_client.py` | Calls Blockchair REST API: DOGE daily active addresses, transaction count, avg tx value. No API key needed | `onchain_aggregator.py` | `utils/retry.py`, `utils/rate_limiter.py` | Free tier: 1000 req/day hard cap. Budget requests carefully |
| `onchain_aggregator.py` | Merges whale signals, exchange flow, DeFiLlama (SOL), and Blockchair (DOGE) metrics into a single feature DataFrame | `pipeline/data_pipeline.py` | all onchain modules | Outputs columns: `sol_whale_signal, sol_exchange_flow, sol_tvl, sol_defi_inflow, doge_whale_signal, doge_exchange_flow, doge_active_addresses, doge_tx_count` |

### `engines/sentiment/`

| File | What It Does | Called By | Calls Into | Key Rule |
|------|-------------|-----------|------------|----------|
| `twitter_scraper.py` | Tweepy v2: searches `$SOL`, `$DOGE`, `#Solana`, `#Dogecoin`. Returns list of raw tweet texts + metadata | `sentiment_aggregator.py` | `utils/retry.py`, `utils/rate_limiter.py` | Collect raw text ONLY. No scoring here |
| `reddit_scraper.py` | httpx: fetches top/new posts from `r/solana`, `r/dogecoin`, `r/CryptoCurrency` via public JSON endpoints (`/hot.json`, `/new.json`). No API key needed | `sentiment_aggregator.py` | `utils/retry.py` | Collect raw text ONLY. No scoring here. No PRAW dependency |
| `telegram_scraper.py` | Telethon: reads messages from public crypto Telegram channels | `sentiment_aggregator.py` | `utils/retry.py` | Requires phone-number-linked API registration. Store session files securely (gitignored) |
| `elon_tracker.py` | Monitors `@elonmusk` tweets for DOGE/Dogecoin keywords. Sets `elon_signal = True` in feature vector when detected within the last hour. Applies **3x multiplier** on CryptoBERT DOGE sentiment only | `sentiment_aggregator.py` | `twitter_scraper.py` | Only affects DOGE. Never apply to SOL. Multiplier defined in `constants.py` |
| `text_preprocessor.py` | Cleans raw social text: removes URLs, strips HTML, converts emoji to tokens, normalizes whitespace, detects/filters non-English | All scorers (called before every scoring call) | `emoji`, `langdetect`, `nltk` | ALWAYS called before passing text to any scorer. Never score raw text |
| `finbert_scorer.py` | Loads `ProsusAI/finbert`. Scores text → `{positive, negative, neutral}` confidence. Best for macro/news text | `sentiment_aggregator.py` | `transformers`, `torch` | GPU optional. Use for FRED news, macro headlines. Batch inputs for speed |
| `cryptobert_scorer.py` | Loads `ElKulako/cryptobert`. Scores text → sentiment label + confidence. Best for social media | `sentiment_aggregator.py` | `transformers`, `torch` | Primary scorer for Twitter/Reddit/Telegram. GPU recommended for throughput |
| `vader_scorer.py` | `vaderSentiment`: rule-based, no model download, instant scoring. Returns compound score in `[-1, +1]` | `sentiment_aggregator.py` | `vaderSentiment` | No GPU needed. Used as fast fallback and for real-time stream scoring |
| `sentiment_aggregator.py` | Runs preprocessor → all three scorers → weighted average. Weights: `CryptoBERT=0.5, FinBERT=0.3, VADER=0.2`. Applies elon multiplier for DOGE | `pipeline/data_pipeline.py` | all scrapers + scorers | Normalizes all scores to `[-1.0, +1.0]` before weighting |
| `sentiment_features.py` | Engineers rolling sentiment features: 1h/4h/24h rolling mean, momentum (current - 24h avg), bull/bear divergence between SOL and DOGE | `engines/forecasting/feature_builder.py` | `sentiment_aggregator.py` | No fetching or scoring. Receives scored DataFrame, returns feature DataFrame |

### `engines/prices/`

| File | What It Does | Called By | Calls Into | Key Rule |
|------|-------------|-----------|------------|----------|
| `price_stream.py` | Subscribes to Binance ccxt WebSocket for SOL/USDT and DOGE/USDT real-time ticks. On each tick, computes % change from last prediction price. If ≥ `LIVE_PREDICTION_TRIGGER_PCT` (1%), immediately triggers `prediction_pipeline.run()` | `server/app.py` (lifespan) | `ccxt`, `price_aggregator.py`, `pipeline/prediction_pipeline.py` | This is the real-time heartbeat. Never add non-price logic here |
| `price_aggregator.py` | Aggregates raw price ticks into 1-minute and 1-hour OHLCV candles. Writes completed candles to `storage/price_repository.py` | `price_stream.py` | `storage/price_repository.py`, `utils/time_utils.py` | A candle is only "complete" when the next candle starts. Never write partial candles |

### `server/`

| File | What It Does | Called By | Key Rule |
|------|-------------|-----------|----------|
| `app.py` | FastAPI application with `lifespan` context. On startup: loads all models, starts APScheduler, starts ccxt WebSocket stream. On shutdown: graceful cleanup. | `uvicorn server.app:app` | This is the ONLY production entry point. Do not run orchestrator directly |
| `websocket_manager.py` | Manages the set of active WebSocket client connections. `broadcast(data)` sends `PredictionResult` JSON to all connected clients. Thread-safe | `routes/predictions.py` | Use `asyncio.Lock` for the connection set. Never block the event loop |
| `routes/predictions.py` | `GET /predictions/{coin}` — latest cached prediction. `GET /predictions/{coin}/history` — last N from DB. `WS /ws/predictions` — subscribe to live updates | `app.py` router | Always read from Redis cache first, fall back to DB |
| `routes/health.py` | `GET /health` — simple 200 OK liveness check. `GET /status` — full engine status: last fetch times, model load status, Redis connectivity | `app.py` router | Never fails (catch all exceptions, return degraded status) |

### `engines/forecasting/`

| File | What It Does | Called By | Calls Into | Key Rule |
|------|-------------|-----------|------------|----------|
| `timesfm_model.py` | Loads `google/timesfm-2.5-200m-pytorch` in bfloat16 on GPU. Input: raw hourly close price array (up to 16,000 points). Output: `{mean_24h, mean_72h, mean_7d, quantiles}`. Zero-shot — no training needed | `ensemble.py` | `timesfm` package, `torch` | Load model ONCE at server startup in `app.py lifespan`. Never reload per-prediction |
| `feature_builder.py` | **Single source of truth** for model inputs. Calls macro_features, onchain_aggregator, sentiment_features and assembles the canonical feature matrix. Also adds technical indicators (RSI, MACD, Bollinger Bands via `ta`) | `predictor.py`, `trainer.py` | macro, onchain, sentiment feature modules | If you add any new signal to any engine, register it here. Never build feature vectors ad-hoc in model files |
| `lstm_model.py` | PyTorch LSTM: 2 stacked layers, hidden_size=256, dropout=0.2. 60-day sliding input window. Outputs 24h/72h/7d price delta predictions | `ensemble.py`, `trainer.py` | `torch` | Stateless at inference time. Loads weights from `models/lstm_{coin}.pt` |
| `transformer_model.py` | Temporal Fusion Transformer via `pytorch-forecasting`. Handles mixed static (coin type) and time-varying (all features) covariates. Built-in attention for variable importance | `ensemble.py`, `trainer.py` | `pytorch-forecasting`, `pytorch-lightning` | Best single model for multi-horizon forecasting. Slowest to train |
| `xgboost_model.py` | XGBoost gradient boosted trees. Tabular features only (no sequence). Outputs bull/bear regime classification + price direction | `ensemble.py`, `trainer.py` | `xgboost`, `scikit-learn` | Provides interpretability via feature importance. Fast to retrain |
| `lightgbm_model.py` | LightGBM: faster alternative to XGBoost, better on high-cardinality categorical features | `ensemble.py`, `trainer.py` | `lightgbm`, `scikit-learn` | Complement to XGBoost in the ensemble. Rarely needs separate tuning |
| `ensemble.py` | Combines LSTM + TFT + XGBoost + LightGBM using learned weights (default: 25/35/20/20). Returns `PredictionResult` dataclass | `pipeline/prediction_pipeline.py` | all model files | **Never import `trainer.py` here.** Inference only. If a model fails to load, log and assign weight=0 to that model |
| `trainer.py` | Full training loop: data loading, cross-validation, early stopping, model checkpointing with date-stamped filenames | `scripts/train_models.py` ONLY | `feature_builder.py`, all model files, `pytorch-lightning` | Never called by the prediction pipeline. Only imported by training scripts |
| `evaluator.py` | Computes model performance: MAE, RMSE, directional accuracy %, Sharpe ratio on held-out test set | `scripts/evaluate_models.py` | all model files | Use rolling walk-forward validation, not a single train/test split |
| `predictor.py` | Loads trained models, calls `feature_builder`, runs inference through all models, passes to `ensemble`. Entry point for generating a single prediction run | `pipeline/prediction_pipeline.py` | `feature_builder.py`, `ensemble.py` | Do not retrain or fine-tune here |

### `pipeline/`

| File | What It Does | Called By | Key Rule |
|------|-------------|-----------|----------|
| `orchestrator.py` | Sets up APScheduler jobs inside FastAPI lifespan. Data fetch every 15 min. Forced prediction every 5 min. Also triggered externally by `price_stream.py` on 1%+ moves | `server/app.py` lifespan | All pipeline modules | Do NOT run directly — always started by `server/app.py` |
| `data_pipeline.py` | Orchestrates macro + onchain + sentiment data fetch → process → store in database | `orchestrator.py` (every 15 min) | All engine aggregators, `storage/` repositories | If any engine fails, log the error and continue. Partial data is better than no data |
| `prediction_pipeline.py` | Loads latest features from DB, runs `predictor.py`, stores `PredictionResult`, broadcasts to WebSocket clients | `orchestrator.py` (every 5 min) OR `price_stream.py` (on 1%+ move) | `predictor.py`, `prediction_repository.py`, `server/websocket_manager.py` | Only run after `data_pipeline.py` has completed at least one successful cycle |
| `alert_pipeline.py` | Reads latest `PredictionResult`. If `confidence >= 0.70 AND abs(magnitude) >= 3%`, triggers notification. Deduplicates alerts using Redis (1h TTL) | `orchestrator.py` | `prediction_repository.py`, `cache_manager.py`, all notifiers | Check Redis before sending. Never send duplicate alerts within 1 hour for the same coin |

### `notifications/`

| File | What It Does | Key Rule |
|------|-------------|----------|
| `base_notifier.py` | Abstract base class. Defines `send(message: str, image_path: str | None)` interface | All notifiers must inherit this |
| `discord_notifier.py` | Sends rich embed messages + chart PNG to Discord via webhook URL or bot | Uses `DISCORD_WEBHOOK_URL` from settings |
| `whatsapp_notifier.py` | Sends messages via Twilio WhatsApp API. Use sandbox `+14155238886` for development | Requires approved Twilio WhatsApp sender for production |
| `zalo_notifier.py` | Posts to Zalo Official Account followers via OA API v2 REST calls (httpx). Token expires every 1h — refresh logic is inside this file | Vietnamese audience. `ZALO_OA_ACCESS_TOKEN` must be refreshed before each call |
| `message_formatter.py` | **All platform-specific formatting lives here.** Uses Jinja2 templates to produce platform-appropriate text. Called by each notifier | Never put formatting strings inside individual notifier files |
| `chart_generator.py` | Generates a matplotlib price forecast PNG. Saves to `data/cache/{coin}_{timestamp}.png` with 1h TTL | Returns file path. Caller attaches the file to the notification |

### `storage/`

| File | What It Does | Key Rule |
|------|-------------|----------|
| `database.py` | SQLAlchemy engine + session factory. Reads `DATABASE_URL` from settings (default: SQLite). Zero code change needed to switch to PostgreSQL | Always use `get_session()` context manager, never raw connections |
| `models.py` | ORM table definitions: `PriceData` (OHLCV), `Prediction` (model output), `SentimentScore` (aggregated score per coin per hour) | Read this before writing any query |
| `price_repository.py` | CRUD for `PriceData`. Methods: `upsert_candle()`, `get_range()` | All price DB access goes through here |
| `prediction_repository.py` | CRUD for `Prediction`. Methods: `save()`, `get_latest()`, `get_history()` | All prediction DB access goes through here |
| `cache_manager.py` | Redis wrapper. Methods: `set(key, value, ttl_seconds)`, `get(key)`, `exists(key)`. Used for API caching and alert deduplication | TTLs: API responses=5min, Sentiment=15min, Predictions=1h, Alert dedup=1h |

### `utils/`

| File | What It Does | Key Rule |
|------|-------------|----------|
| `retry.py` | `@retry` decorator using `tenacity`. Default: 3 attempts, exponential backoff starting at 2s. Retries on `httpx.HTTPError`, `requests.exceptions.RequestException` | Wrap EVERY external API call. Do not write bare HTTP calls without this |
| `rate_limiter.py` | Token-bucket rate limiter. Usage: `limiter.wait("fred")` before an API call. Limits configured in `config/constants.py` | One limiter instance per API service. Import and call before sending any request |
| `time_utils.py` | `to_utc(dt)` — converts any datetime to UTC. `now_utc()` — current UTC time. `resample_hourly(df)` — resamples DataFrame to 1h UTC DatetimeIndex | Every timestamp entering the system calls `to_utc()`. Every DataFrame output has `freq='1h'` UTC index |
| `crypto_utils.py` | `pct_change(old, new)` → % change. `normalize_price(series)` → min-max scaled. `compute_returns(df)` → log returns for model input | Use these instead of inline math to keep calculations consistent |
| `validators.py` | Pydantic schemas for validating API responses and model inputs before they enter the pipeline | Validate at system boundaries: after API fetch, before DB write, before model inference |

### `scripts/`

| File | When to Run | Key Rule |
|------|-------------|----------|
| `setup_db.py` | Once, before first run. Creates all SQLAlchemy tables | Run this first before anything else |
| `backfill_prices.py` | Once, to populate 3 years of SOL/DOGE OHLCV history from Binance via `ccxt`. Args: `--coins SOL DOGE --days 1095` | Models need at least 90 days to train meaningfully |
| `train_models.py` | After backfill. Trains LSTM + TFT + XGBoost + LightGBM per coin. Args: `--model all --coin SOL` | Uses `engines/forecasting/trainer.py`. Saves weights to `models/` with date stamp |
| `evaluate_models.py` | After training. Runs backtests with rolling walk-forward validation. Args: `--lookback 90` | Check directional accuracy > 55% and Sharpe > 1.0 before deploying |

---

## Section 4 — AI Model Quick Reference

| Model | HuggingFace ID | Input | Output | Loaded In | GPU Needed |
|-------|---------------|-------|--------|-----------|-----------|
| **TimesFM 2.5** | `google/timesfm-2.5-200m-pytorch` | Raw hourly close price tensor, up to 16,000 points. **No feature engineering needed.** | `mean_predictions` + quantile bands (10th-90th percentile) per horizon | `timesfm_model.py` | Yes (RTX 4060 OK) |
| **FinBERT** | `ProsusAI/finbert` | Cleaned news/macro text (max 512 tokens) | `{positive, negative, neutral}` + confidence float | `finbert_scorer.py` | No (slow on CPU) |
| **CryptoBERT** | `ElKulako/cryptobert` | Cleaned crypto social text (max 512 tokens) | Sentiment label + confidence float | `cryptobert_scorer.py` | Recommended |
| **VADER** | Built-in (no download) | Raw or lightly cleaned text | `compound` float in `[-1, +1]` | `vader_scorer.py` | No |
| **LSTM** | Custom PyTorch (local) | Float tensor `[batch, 60, num_features]` | Float tensor `[batch, 3]` (24h/72h/7d delta) | `lstm_model.py` | Optional |
| **TFT** | `pytorch-forecasting` library | `TimeSeriesDataSet` object | Multi-horizon point forecasts + quantiles | `transformer_model.py` | Recommended |
| **XGBoost** | Local `.json` weights | Tabular `numpy` array `[n_samples, num_features]` | Binary direction label + probability | `xgboost_model.py` | No |
| **LightGBM** | Local `.pkl` weights | Tabular `numpy` array `[n_samples, num_features]` | Binary direction label + probability | `lightgbm_model.py` | No |

### TimesFM Key Facts
- **Zero-shot**: No training or fine-tuning needed. Load and run immediately.
- **Context**: Up to 16,000 hourly points (~667 days). Start at 512 (`TIMESFM_CONTEXT_LEN`), increase as data accumulates.
- **Quantile output**: Returns 10th/20th/.../90th percentile bands — use these as confidence intervals in chart PNGs.
- **Download**: ~800MB on first run, cached to `~/.cache/huggingface/`.
- **GPU VRAM**: ~1GB in bfloat16. RTX 4060 8GB handles it comfortably alongside other models.
- **Frequency input**: Use `freq=0` (high frequency / sub-daily) for hourly crypto data.

### Model Loading Pattern (all models follow this)
```python
# At import time — load once, reuse across prediction calls
model = load_model(path=settings.MODEL_DIR / f"lstm_{coin}.pt")

# At inference time — stateless call
result = model.predict(features)
```

---

## Section 5 — Data Flow Diagram

```
External APIs                  Engines                   Storage
─────────────────────────────────────────────────────────────────────
FRED API  ──────────►  fred_client.py
yfinance  ──────────►  dxy_tracker.py    ─────►  macro_aggregator.py ─┐
                                                                        │
Solana RPC ─────────►  sol_rpc_client.py                              │
DOGE RPC  ──────────►  doge_rpc_client.py ────► onchain_aggregator.py─┤
DeFiLlama ──────────►  defillama_client.py (SOL TVL, free, no key)    │
Blockchair ──────────►  blockchair_client.py (DOGE metrics, free)     │
                                                                        │
Twitter   ──────────►  twitter_scraper.py                             │
Reddit    ──────────►  reddit_scraper.py  ────► sentiment_aggregator.py┤
Telegram  ──────────►  telegram_scraper.py                            │
                                                                        ├─► data_pipeline.py
Binance (ccxt) ──────► backfill_prices.py (one-time setup)            │       │
                                                                        │       ▼
                                                               price_repository.py
                                                               prediction_repository.py
                                                               cache_manager.py (Redis)
                                                                                │
                                                                                ▼
                                                            feature_builder.py
                                                                    │
                                                    ┌───────────────┼───────────────┐
                                                    ▼               ▼               ▼
                                               lstm_model   transformer_model   xgboost_model
                                               lightgbm_model
                                                    └───────────────┬───────────────┘
                                                                    ▼
                                                              ensemble.py
                                                          PredictionResult
                                                                    │
                                                        confidence > 0.70?
                                                                    │ YES
                                                                    ▼
                                                        alert_pipeline.py
                                                                    │
                                             ┌──────────────────────┼──────────────────┐
                                             ▼                      ▼                  ▼
                                   discord_notifier      whatsapp_notifier    zalo_notifier
```

---

## Section 6 — The 5 Non-Negotiable Coding Rules

> Breaking any of these rules introduces bugs that are extremely hard to trace.

**Rule 1: All timestamps → UTC**
```python
# WRONG
df.index = pd.to_datetime(df.index)

# RIGHT
from utils.time_utils import to_utc
df.index = to_utc(pd.to_datetime(df.index))
```

**Rule 2: All API calls → wrapped with @retry**
```python
# WRONG
response = requests.get(url)

# RIGHT
from utils.retry import retry
@retry
def fetch_data(url: str) -> dict:
    response = requests.get(url)
    return response.json()
```

**Rule 3: Never hardcode API keys**
```python
# WRONG
api_key = "abc123xyz"

# RIGHT
from config.settings import settings
api_key = settings.FRED_API_KEY
```

**Rule 4: Scrapers fetch, Scorers score — never mix**
```python
# WRONG — scorer calling Twitter API
class FinBERTScorer:
    def score_latest(self):
        tweets = requests.get(twitter_url)  # NO

# RIGHT — clear separation
texts = twitter_scraper.fetch()           # scraper fetches
clean = text_preprocessor.clean(texts)   # preprocessor cleans
scores = finbert_scorer.score(clean)     # scorer scores
```

**Rule 5: All DB writes go through repositories**
```python
# WRONG
session.add(Prediction(...))
session.commit()

# RIGHT
from storage.prediction_repository import PredictionRepository
repo = PredictionRepository(session)
repo.save(prediction_result)
```

---

## Section 7 — Testing Guide

```bash
# Start live server (production)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Start live server (development, auto-reload)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Unit tests (no API keys needed — all external calls are mocked)
pytest tests/unit/ -v

# Integration tests (requires .env with real API keys)
pytest tests/integration/ -v --timeout=30

# Tests that need live API connections are marked:
# @pytest.mark.live
# Skip them in CI:
pytest tests/ -v -m "not live"
```

**What each test file covers:**
| Test File | Tests |
|-----------|-------|
| `tests/unit/test_fred_client.py` | FRED response parsing, retry on 429, UTC index output |
| `tests/unit/test_whale_detector.py` | Threshold classification, ACCUMULATE/DISTRIBUTE logic |
| `tests/unit/test_finbert_scorer.py` | Score normalization, empty input handling, batch output shape |
| `tests/unit/test_ensemble.py` | Weight normalization, missing model fallback, PredictionResult fields |
| `tests/unit/test_message_formatter.py` | Discord/WhatsApp/Zalo template rendering, confidence formatting |
| `tests/integration/test_macro_pipeline.py` | Full macro fetch → aggregate → feature build chain |
| `tests/integration/test_sentiment_pipeline.py` | Scrape → preprocess → score → aggregate chain |
| `tests/integration/test_prediction_pipeline.py` | Feature build → all models → ensemble → PredictionResult |

---

## Section 8 — Notification Platform Notes

### Discord
- Uses webhook URL (`DISCORD_WEBHOOK_URL`) for posting, no bot token needed for basic use
- Rich embeds supported: title, description, color, fields, footer, image attachment
- Rate limit: 30 requests/minute per webhook
- Image: attach chart PNG as `files=` parameter in the webhook payload

### WhatsApp (via Twilio)
- **Development**: Use Twilio sandbox number `+14155238886`. Join sandbox by texting
  `join <your-sandbox-keyword>` from your WhatsApp number first
- **Production**: Requires a Twilio WhatsApp Business sender (Meta approval, ~1 week)
- Supports text + media (image URL or uploaded media SID)
- Cost: ~$0.005/message. Set a monthly budget cap in Twilio console

### Zalo (via Official Account API v2)
- Requires registering a Zalo Official Account (OA) at `oa.zalo.me`
- Target audience: Vietnamese users. Interface/docs are primarily in Vietnamese
- `ZALO_OA_ACCESS_TOKEN` expires every **1 hour** — the `zalo_notifier.py` file handles
  automatic token refresh using `ZALO_APP_ID` + `ZALO_APP_SECRET` + `ZALO_REFRESH_TOKEN`
- Rate limit: 2,000 messages/day on free tier
- Supports text messages and image attachments via separate upload + send API calls

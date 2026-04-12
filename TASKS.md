# CPT — Task Tracker
# Last Updated: 2026-04-11

---

## PHASE A — Foundation (Build First — Unblocks Everything)

- [x] **A1** Gather all API keys and fill in `.env`
  - [x] FRED API key (free, 1 min) -- https://fred.stlouisfed.org/docs/api/api_key.html -- Verified 2026-04-11
  - [x] Reddit -- switched to public JSON scraping (no API key needed) -- Verified 2026-04-11
  - [ ] Discord webhook URL (30 sec) — Server Settings → Integrations → Webhooks
  - [ ] Twitter/X Bearer Token — https://developer.twitter.com
  - [ ] Telegram API ID + hash — https://my.telegram.org/auth
  - [ ] Glassnode API key — https://glassnode.com
  - [ ] Solana RPC URL — https://helius.dev (free 100k req/day)
  - [ ] Twilio WhatsApp (sandbox) — https://console.twilio.com
  - [ ] Zalo OA access token — https://oa.zalo.me (requires business verification)

- [x] **A2** Implement `utils/time_utils.py` — `to_utc()`, `now_utc()`, `resample_hourly()`
- [x] **A3** Implement `utils/retry.py` — `@retry` decorator with tenacity exponential backoff
- [x] **A4** Implement `utils/rate_limiter.py` — token bucket limiter per API service
- [x] **A5** Implement `utils/crypto_utils.py` — `pct_change()`, `normalize_price()`, `compute_returns()`
- [x] **A6** Implement `utils/validators.py` — Pydantic schemas for API response validation
- [x] **A7** Implement `config/logging_config.py` — structlog JSON setup, `setup_logging()`
- [x] **A8** Implement `storage/database.py` — SQLAlchemy engine + session factory
- [x] **A9** Implement `storage/models.py` — ORM: PriceData, Prediction, SentimentScore
- [x] **A10** Implement `storage/price_repository.py` — `upsert_candle()`, `get_range()`
- [x] **A11** Implement `storage/prediction_repository.py` — `save()`, `get_latest()`, `get_history()`
- [x] **A12** Implement `storage/cache_manager.py` — Redis wrapper with TTL methods
- [x] **A13** Run `python scripts/setup_db.py` — initialize database schema

---

## PHASE B — Data Collection

- [ ] **B1** Implement `engines/macro/fred_client.py` — fetch FEDFUNDS, DGS10, M2SL
- [ ] **B2** Implement `engines/macro/m2_supply.py` — M2 trend analysis wrapper
- [ ] **B3** Implement `engines/macro/dxy_tracker.py` — fetch DXY from yfinance
- [ ] **B4** Implement `engines/macro/macro_aggregator.py` — merge on UTC hourly index
- [ ] **B5** Implement `engines/macro/macro_features.py` — YoY change, z-scores, lags
- [ ] **B6** Implement `scripts/backfill_prices.py` — 3 years of SOL/DOGE OHLCV from Binance
- [ ] **B7** Run `make backfill` — populate historical price data (needed before training)
- [ ] **B8** Implement `engines/prices/price_stream.py` — ccxt WebSocket for live SOL/DOGE ticks
- [ ] **B9** Implement `engines/prices/price_aggregator.py` — ticks → 1min/1hr OHLCV candles

---

## PHASE C — Sentiment Engine

- [ ] **C1** Implement `engines/sentiment/text_preprocessor.py` — URL removal, emoji, lang filter
- [ ] **C2** Implement `engines/sentiment/vader_scorer.py` — VADER compound score [-1, +1]
- [ ] **C3** Implement `engines/sentiment/reddit_scraper.py` — httpx JSON scraping: r/solana, r/dogecoin (no API key needed)
- [ ] **C4** Implement `engines/sentiment/twitter_scraper.py` — Tweepy v2: $SOL, $DOGE search
- [ ] **C5** Implement `engines/sentiment/telegram_scraper.py` — Telethon: public channels
- [ ] **C6** Implement `engines/sentiment/finbert_scorer.py` — ProsusAI/finbert inference
- [ ] **C7** Implement `engines/sentiment/cryptobert_scorer.py` — ElKulako/cryptobert inference
- [ ] **C8** Implement `engines/sentiment/elon_tracker.py` — @elonmusk DOGE signal + 3x multiplier
- [ ] **C9** Implement `engines/sentiment/sentiment_aggregator.py` — weighted average (0.5/0.3/0.2)
- [ ] **C10** Implement `engines/sentiment/sentiment_features.py` — rolling windows, momentum

---

## PHASE D — On-Chain Engine

- [ ] **D1a** Implement `engines/onchain/defillama_client.py` -- DeFiLlama REST API: SOL chain TVL, DeFi protocol count (free, no key)
- [ ] **D1b** Implement `engines/onchain/blockchair_client.py` -- Blockchair REST API: DOGE active addresses, daily tx count (free, 1000 req/day limit)
- [ ] **D2** Implement `engines/onchain/sol_rpc_client.py` — Solana RPC: account + tx queries
- [ ] **D3** Implement `engines/onchain/doge_rpc_client.py` — Dogecoin Core JSON-RPC
- [ ] **D4** Implement `engines/onchain/whale_detector.py` — classify wallets >$500k
- [ ] **D5** Implement `engines/onchain/exchange_flow.py` — net inflow/outflow to exchanges
- [ ] **D6** Implement `engines/onchain/onchain_aggregator.py` — merge SOL + DOGE on-chain

---

## PHASE E — ML Models

- [ ] **E1** Implement `engines/forecasting/feature_builder.py` — canonical feature matrix
- [ ] **E2** Implement `engines/forecasting/timesfm_model.py` — TimesFM 2.5 zero-shot inference
  - Model ID: `google/timesfm-2.5-200m-pytorch`
  - Input: raw hourly close price array, `freq=0`
  - Output: mean + quantile bands per horizon
- [ ] **E3** Implement `engines/forecasting/xgboost_model.py` — train + inference wrapper
- [ ] **E4** Implement `engines/forecasting/lightgbm_model.py` — train + inference wrapper
- [ ] **E5** Implement `engines/forecasting/lstm_model.py` — PyTorch LSTM (2 layers, hidden=256)
- [ ] **E6** Implement `engines/forecasting/transformer_model.py` — TFT via pytorch-forecasting
- [ ] **E7** Implement `engines/forecasting/trainer.py` — training loop, checkpointing
- [ ] **E8** Implement `engines/forecasting/evaluator.py` — MAE, RMSE, directional accuracy, Sharpe
- [ ] **E9** Implement `engines/forecasting/ensemble.py` — 5-model weighted combiner
- [ ] **E10** Implement `engines/forecasting/predictor.py` — inference entry point
- [ ] **E11** Run `make train-sol` then `make train-doge` — train LSTM, TFT, XGBoost, LightGBM
- [ ] **E12** Run `make evaluate` — verify directional accuracy >55% and Sharpe >1.0

---

## PHASE F — Live Server & Pipeline

- [ ] **F1** Implement `pipeline/data_pipeline.py` — fetch → process → store
- [ ] **F2** Implement `pipeline/prediction_pipeline.py` — features → predict → broadcast
- [ ] **F3** Implement `pipeline/alert_pipeline.py` — threshold check + Redis dedup
- [ ] **F4** Implement `pipeline/orchestrator.py` — APScheduler jobs inside FastAPI lifespan
- [ ] **F5** Implement `server/websocket_manager.py` — manage WS clients, broadcast predictions
- [ ] **F6** Implement `server/routes/health.py` — GET /health, GET /status
- [ ] **F7** Implement `server/routes/predictions.py` — GET + WS /predictions endpoints
- [ ] **F8** Implement `server/app.py` — FastAPI app with lifespan (load models, start scheduler + WS)
- [ ] **F9** Run `make dev` — first full end-to-end live server test

---

## PHASE G — Notifications

- [ ] **G1** Implement `notifications/chart_generator.py` — matplotlib price forecast PNGs
- [ ] **G2** Implement `notifications/message_formatter.py` — Jinja2 templates per platform
- [ ] **G3** Implement `notifications/base_notifier.py` — abstract base class
- [ ] **G4** Implement `notifications/discord_notifier.py` — webhook rich embeds + chart attach
- [ ] **G5** Implement `notifications/whatsapp_notifier.py` — Twilio API
- [ ] **G6** Implement `notifications/zalo_notifier.py` — Zalo OA API v2 + token refresh
- [ ] **G7** Send first real Discord alert — confirm end-to-end delivery

---

## PHASE H — Testing & Hardening

- [ ] **H1** Write unit tests (tests/unit/) for all 5 test files
- [ ] **H2** Write integration tests (tests/integration/) for all 3 pipelines
- [ ] **H3** Run `make test-all` — all tests passing
- [ ] **H4** Run `make lint` — ruff + mypy clean
- [ ] **H5** Stress test live server: simulate 1%+ price moves, confirm re-prediction triggers
- [ ] **H6** Confirm alert deduplication: same coin should not alert twice within 1 hour

---

## NOTES

- **Entry point:** `make run` (production) or `make dev` (development with auto-reload)
- **First run order:** A → A13 (setup-db) → B7 (backfill) → E11 (train) → F9 (server up)
- **TimesFM:** downloads ~800MB model weights on first inference call. Allow time on first run.
- **Dogecoin node:** Skip D3/D4/D5 DOGE RPC tasks initially — use Glassnode (D1) instead.
- **Skip for now:** Telegram (C5) — needs phone number. Add after other scrapers work.

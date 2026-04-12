.PHONY: run fetch train test lint format setup

# --- Production ---
run:
	uvicorn server.app:app --host 0.0.0.0 --port 8000

dev:
	uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# --- Data ---
fetch:
	python -m engines.macro.macro_aggregator
	python -m engines.onchain.onchain_aggregator
	python -m engines.sentiment.sentiment_aggregator

backfill:
	python scripts/backfill_prices.py --coins SOL DOGE --days 1095

# --- Models ---
train-sol:
	python scripts/train_models.py --model all --coin SOL

train-doge:
	python scripts/train_models.py --model all --coin DOGE

train: train-sol train-doge

evaluate:
	python scripts/evaluate_models.py --lookback 90

# --- Database ---
setup-db:
	python scripts/setup_db.py

# --- Tests ---
test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --timeout=30

test-all:
	pytest tests/ -v --timeout=30

# --- Code Quality ---
format:
	black --line-length 100 .

lint:
	ruff check .
	mypy .

# --- Environment ---
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

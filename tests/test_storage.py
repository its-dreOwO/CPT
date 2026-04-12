import sys
import os
import datetime

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.database import get_session
from storage.price_repository import upsert_candle, get_range
from storage.prediction_repository import save, get_latest, get_history
from utils.validators import PredictionResult
from utils.time_utils import now_utc
from storage.cache_manager import cache
from scripts.setup_db import setup

def test_database_and_repositories():
    print("==== 1. Setup DataBase ====")
    setup()
    print("Database is ready!\n")

    print("==== 2. Test Price Repository ====")
    with get_session() as session:
        now = now_utc()
        # insert
        candle = upsert_candle(session, "SOL", now, open=100.0, high=105.0, low=98.0, close=102.0, volume=10000.5)
        print(f"Upserted candle ID: {candle.id}")
        
        # update
        candle_updated = upsert_candle(session, "SOL", now, open=100.0, high=105.0, low=98.0, close=103.0, volume=10000.5)
        print(f"Updated candle Close: {candle_updated.close} (should be 103.0)")
        
        # get range
        start = now - datetime.timedelta(hours=1)
        end = now + datetime.timedelta(hours=1)
        candles = get_range(session, "SOL", start, end)
        print(f"Retrieved {len(candles)} candles in range")

    print("\n==== 3. Test Prediction Repository ====")
    with get_session() as session:
        pred_res = PredictionResult(
            coin="DOGE",
            timestamp=now_utc(),
            direction="bullish",
            magnitude_pct=5.5,
            confidence=0.85,
            target_24h=0.15,
            ensemble_weights_used={"timesfm": 0.5, "xgboost": 0.5}
        )
        saved_pred = save(session, pred_res)
        print(f"Saved prediction ID: {saved_pred.id}")
        
        latest = get_latest(session, "DOGE")
        print(f"Retrieved latest Prediction magnitude: {latest.magnitude_pct}%")
        
        history = get_history(session, "DOGE", limit=5)
        print(f"History length: {len(history)}")

    print("\n==== 4. Test Cache Manager (Redis) ====")
    # Might fail if Redis is not running locally, that's fine. We check if client exists.
    if cache.client:
        success = cache.set("test_key", {"status": "ok"}, ttl_seconds=10)
        print(f"Cache SET success: {success}")
        val = cache.get_json("test_key")
        print(f"Cache GET value: {val}")
        exists = cache.exists("test_key")
        print(f"Cache EXISTS: {exists}")
    else:
        print("Redis client not connected. Skipping Redis tests (expected if no local server running).")

    print("\nALL STORAGE TESTS COMPLETED ✅")

if __name__ == "__main__":
    test_database_and_repositories()

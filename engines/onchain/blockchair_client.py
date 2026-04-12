"""
engines/onchain/blockchair_client.py

Blockchair REST API client for Dogecoin on-chain metrics.
Free, no API key required. Rate limit: 30 req/min, 1000 req/day hard cap.

Provides:
  - Daily transaction count (24h)
  - Unique holding addresses (hodling_addresses)
  - Network node count
  - Mempool depth
  - Current DOGE market price (USD) — cross-check only

Endpoints used:
  GET https://api.blockchair.com/dogecoin/stats  -> chain summary stats
"""

import structlog
import httpx
from dataclasses import dataclass

from utils.retry import with_retry
from utils.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

_BASE = "https://api.blockchair.com"
_limiter = RateLimiter({"blockchair": 30})


@dataclass
class DogeStats:
    """Snapshot of Dogecoin on-chain statistics."""

    transactions_24h: int
    hodling_addresses: int
    nodes: int
    mempool_transactions: int
    blocks_24h: int
    hashrate_24h: float  # hashes/sec
    market_price_usd: float


@with_retry()
def fetch_stats() -> DogeStats:
    """Fetch current Dogecoin on-chain statistics from Blockchair.

    Returns:
        DogeStats dataclass. Returns zeroed instance on failure.
    """
    _limiter.acquire("blockchair")
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{_BASE}/dogecoin/stats")
        r.raise_for_status()

    data = r.json().get("data", {})

    stats = DogeStats(
        transactions_24h=int(data.get("transactions_24h", 0)),
        hodling_addresses=int(data.get("hodling_addresses", 0)),
        nodes=int(data.get("nodes", 0)),
        mempool_transactions=int(data.get("mempool_transactions", 0)),
        blocks_24h=int(data.get("blocks_24h", 0)),
        hashrate_24h=float(data.get("hashrate_24h", 0)),
        market_price_usd=float(data.get("market_price_usd", 0.0)),
    )

    logger.info(
        "blockchair_stats_fetched",
        transactions_24h=stats.transactions_24h,
        hodling_addresses=stats.hodling_addresses,
        nodes=stats.nodes,
    )
    return stats


@with_retry()
def fetch_address_balance(address: str) -> float:
    """Fetch the confirmed balance of a single DOGE address (in DOGE).

    Used by whale_detector.py to check known whale wallet balances.
    Rate-limit aware — each call consumes 1 of the 1000/day quota.

    Args:
        address: DOGE wallet address (Base58).

    Returns:
        Balance in DOGE as a float. Returns 0.0 if not found or on error.
    """
    _limiter.acquire("blockchair")
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{_BASE}/dogecoin/dashboards/address/{address}")
        r.raise_for_status()

    payload = r.json().get("data", {}).get(address, {})
    address_data = payload.get("address", {})
    # balance is in satoshis (1 DOGE = 1e8 satoshis)
    balance_satoshi = int(address_data.get("balance", 0))
    balance_doge = balance_satoshi / 1e8

    logger.debug(
        "blockchair_address_fetched", address=address[:8] + "...", balance_doge=balance_doge
    )
    return balance_doge


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()
    stats = fetch_stats()
    print(f"DOGE 24h txs:       {stats.transactions_24h:,}")
    print(f"Hodling addresses:  {stats.hodling_addresses:,}")
    print(f"Nodes:              {stats.nodes}")
    print(f"Mempool txs:        {stats.mempool_transactions}")
    print(f"Market price:       ${stats.market_price_usd:.4f}")

"""
engines/onchain/doge_rpc_client.py

BlockCypher REST API client for Dogecoin chain metrics.
Replaces the need for a local Dogecoin Core node (~60 GB).
Free, no API key required. Rate limit: 3 req/s, ~200 req/hr.

Provides:
  - Chain stats (block height, peer count, mempool depth, fee estimates)
  - Single address DOGE balance (for whale queries)

Endpoints used:
  GET https://api.blockcypher.com/v1/doge/main                        -> chain stats
  GET https://api.blockcypher.com/v1/doge/main/addrs/{address}/balance -> address balance
"""

import structlog
import httpx
from dataclasses import dataclass

from utils.retry import with_retry
from utils.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

_BASE = "https://api.blockcypher.com/v1/doge/main"
_limiter = RateLimiter({"blockcypher": 3})  # 3 req/s hard cap without key


@dataclass
class DogeChainStats:
    """Snapshot of Dogecoin chain state from BlockCypher."""

    block_height: int
    peer_count: int
    unconfirmed_count: int  # mempool depth
    high_fee_per_kb: int  # satoshis/KB
    medium_fee_per_kb: int
    low_fee_per_kb: int


@with_retry()
def fetch_chain_stats() -> DogeChainStats:
    """Fetch current Dogecoin chain statistics.

    Returns:
        DogeChainStats dataclass. Returns zeroed instance on failure.
    """
    _limiter.acquire("blockcypher")
    with httpx.Client(timeout=15) as client:
        r = client.get(_BASE)
        r.raise_for_status()

    data = r.json()
    stats = DogeChainStats(
        block_height=int(data.get("height", 0)),
        peer_count=int(data.get("peer_count", 0)),
        unconfirmed_count=int(data.get("unconfirmed_count", 0)),
        high_fee_per_kb=int(data.get("high_fee_per_kb", 0)),
        medium_fee_per_kb=int(data.get("medium_fee_per_kb", 0)),
        low_fee_per_kb=int(data.get("low_fee_per_kb", 0)),
    )

    logger.info(
        "doge_chain_stats_fetched",
        block_height=stats.block_height,
        peer_count=stats.peer_count,
        mempool=stats.unconfirmed_count,
    )
    return stats


@with_retry()
def fetch_address_balance(address: str) -> float:
    """Fetch the confirmed DOGE balance of a single address.

    Used by whale_detector.py for known whale wallet watchlist queries.
    Each call counts against the 200 req/hr BlockCypher free limit.

    Args:
        address: DOGE wallet address (Base58Check).

    Returns:
        Balance in DOGE. Returns 0.0 on error or if address not found.
    """
    _limiter.acquire("blockcypher")
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{_BASE}/addrs/{address}/balance")
        r.raise_for_status()

    data = r.json()
    # balance is in satoshis (1 DOGE = 1e8 satoshis)
    balance_satoshi = int(data.get("balance", 0))
    balance_doge = balance_satoshi / 1e8

    logger.debug("doge_balance_fetched", address=address[:8] + "...", doge=balance_doge)
    return balance_doge


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()
    stats = fetch_chain_stats()
    print(f"Block height:  {stats.block_height:,}")
    print(f"Peers:         {stats.peer_count}")
    print(f"Mempool txs:   {stats.unconfirmed_count}")
    print(f"High fee/KB:   {stats.high_fee_per_kb} sat")

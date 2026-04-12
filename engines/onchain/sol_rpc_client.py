"""
engines/onchain/sol_rpc_client.py

Solana public RPC client for network metrics.
Uses https://api.mainnet-beta.solana.com — free, no API key.
Rate limit: ~10 req/s for basic queries.

Provides:
  - Recent TPS (total and non-vote)
  - Epoch info (slot height, transaction count)
  - Circulating supply
  - Single address SOL balance (for whale queries)

Override endpoint via SOL_RPC_URL env var (e.g. Helius/Alchemy for higher limits).
"""

import structlog
import httpx
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from utils.retry import with_retry
from utils.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

_DEFAULT_RPC = "https://api.mainnet-beta.solana.com"
_limiter = RateLimiter({"solana_rpc": 100})

_RPC_URL: str = settings.SOL_RPC_URL or _DEFAULT_RPC


@dataclass
class SolNetworkStats:
    """Snapshot of Solana network statistics."""

    tps_total: float  # all transactions per second
    tps_non_vote: float  # non-vote (real user) transactions per second
    slot_height: int
    epoch: int
    transaction_count: int  # cumulative since genesis
    circulating_supply: float  # SOL


def _rpc(method: str, params: Optional[list] = None) -> dict:
    """Execute a single JSON-RPC call against the Solana endpoint."""
    _limiter.acquire("solana_rpc")
    payload: dict = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        payload["params"] = params

    with httpx.Client(timeout=15) as client:
        r = client.post(_RPC_URL, json=payload)
        r.raise_for_status()

    body = r.json()
    if "error" in body:
        raise RuntimeError(f"Solana RPC error [{method}]: {body['error']}")
    return body.get("result", {})


@with_retry()
def fetch_network_stats() -> SolNetworkStats:
    """Fetch current Solana network performance metrics.

    Returns:
        SolNetworkStats dataclass. Returns zeroed instance on failure.
    """
    # getRecentPerformanceSamples returns last N 60-second windows
    samples = _rpc("getRecentPerformanceSamples", [5])
    tps_total = 0.0
    tps_non_vote = 0.0
    if samples:
        tps_total = sum(s["numTransactions"] / s["samplePeriodSecs"] for s in samples) / len(
            samples
        )
        tps_non_vote = sum(
            s["numNonVoteTransactions"] / s["samplePeriodSecs"] for s in samples
        ) / len(samples)

    epoch_info = _rpc("getEpochInfo")
    supply_result = _rpc("getSupply")
    supply_value = supply_result.get("value", {}) if isinstance(supply_result, dict) else {}
    circulating = supply_value.get("circulating", 0) / 1e9  # lamports → SOL

    stats = SolNetworkStats(
        tps_total=round(tps_total, 2),
        tps_non_vote=round(tps_non_vote, 2),
        slot_height=int(epoch_info.get("absoluteSlot", 0)),
        epoch=int(epoch_info.get("epoch", 0)),
        transaction_count=int(epoch_info.get("transactionCount", 0)),
        circulating_supply=round(circulating, 2),
    )

    logger.info(
        "sol_rpc_stats_fetched",
        tps_total=stats.tps_total,
        tps_non_vote=stats.tps_non_vote,
        epoch=stats.epoch,
    )
    return stats


@with_retry()
def fetch_address_balance(address: str) -> float:
    """Fetch the SOL balance of a single wallet address (in SOL).

    Used by whale_detector.py for known whale wallet watchlist queries.

    Args:
        address: Base58 Solana wallet address.

    Returns:
        Balance in SOL. Returns 0.0 on error or if address not found.
    """
    result = _rpc("getBalance", [address])
    lamports = result.get("value", 0) if isinstance(result, dict) else 0
    balance_sol = lamports / 1e9
    logger.debug("sol_rpc_balance_fetched", address=address[:8] + "...", sol=balance_sol)
    return balance_sol


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()
    stats = fetch_network_stats()
    print(f"TPS (total):     {stats.tps_total}")
    print(f"TPS (non-vote):  {stats.tps_non_vote}")
    print(f"Slot height:     {stats.slot_height:,}")
    print(f"Epoch:           {stats.epoch}")
    print(f"Circulating SOL: {stats.circulating_supply:,.0f}")

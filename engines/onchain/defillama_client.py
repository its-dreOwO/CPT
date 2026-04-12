"""
engines/onchain/defillama_client.py

DeFiLlama REST API client for Solana on-chain metrics.
Free, no API key required. Rate limit: ~30 req/min.

Provides:
  - Current SOL chain TVL (USD)
  - Historical daily SOL TVL series
  - Count of active DeFi protocols on Solana

Endpoints used:
  GET https://api.llama.fi/v2/chains                     -> current TVL per chain
  GET https://api.llama.fi/v2/historicalChainTvl/Solana  -> daily TVL history
  GET https://api.llama.fi/protocols                     -> protocol list
"""

import structlog
import pandas as pd
import httpx

from utils.retry import with_retry
from utils.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

_BASE = "https://api.llama.fi"
_CHAIN_NAME = "Solana"
_limiter = RateLimiter({"defillama": 30})


@with_retry()
def fetch_current_tvl() -> float:
    """Fetch the current total TVL (USD) locked on the Solana chain.

    Returns:
        TVL in USD as a float. Returns 0.0 on failure.
    """
    _limiter.acquire("defillama")
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{_BASE}/v2/chains")
        r.raise_for_status()

    chains = r.json()
    for chain in chains:
        if chain.get("name") == _CHAIN_NAME:
            tvl = float(chain.get("tvl", 0.0))
            logger.info("defillama_tvl_fetched", tvl_usd=tvl)
            return tvl

    logger.warning("defillama_solana_chain_not_found")
    return 0.0


@with_retry()
def fetch_historical_tvl(days: int = 365) -> pd.DataFrame:
    """Fetch daily historical TVL for the Solana chain.

    Args:
        days: Number of most-recent days to return. Pass 0 for all history.

    Returns:
        DataFrame with UTC DatetimeIndex (daily) and column 'sol_tvl_usd'.
        Empty DataFrame on failure.
    """
    _limiter.acquire("defillama")
    with httpx.Client(timeout=20) as client:
        r = client.get(f"{_BASE}/v2/historicalChainTvl/{_CHAIN_NAME}")
        r.raise_for_status()

    records = r.json()  # [{"date": unix_ts, "tvl": float}, ...]
    if not records:
        logger.warning("defillama_historical_empty")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["date"], unit="s", utc=True)
    df = df.rename(columns={"tvl": "sol_tvl_usd"})
    df = df.set_index("timestamp")[["sol_tvl_usd"]].sort_index()

    if days > 0:
        df = df.iloc[-days:]

    logger.info("defillama_historical_fetched", rows=len(df))
    return df


@with_retry()
def fetch_protocol_count() -> int:
    """Count the number of active DeFi protocols deployed on Solana.

    Returns:
        Integer count of protocols. Returns 0 on failure.
    """
    _limiter.acquire("defillama")
    with httpx.Client(timeout=20) as client:
        r = client.get(f"{_BASE}/protocols")
        r.raise_for_status()

    protocols = r.json()
    sol_protocols = [p for p in protocols if _CHAIN_NAME in (p.get("chains") or [])]
    count = len(sol_protocols)
    logger.info("defillama_protocol_count", count=count)
    return count


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()
    print(f"Current SOL TVL: ${fetch_current_tvl():,.0f}")
    hist = fetch_historical_tvl(days=7)
    print(f"Historical TVL (last 7 days):\n{hist}")
    print(f"Active Solana protocols: {fetch_protocol_count()}")

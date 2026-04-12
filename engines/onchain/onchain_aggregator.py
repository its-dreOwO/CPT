"""
engines/onchain/onchain_aggregator.py

Aggregates all on-chain signals for SOL and DOGE into a single
OnChainSnapshot dataclass consumed by feature_builder.py.

Data sources:
  SOL:
    - DeFiLlama: chain TVL (USD), active protocol count
    - Solana public RPC: TPS (total + non-vote), epoch, circulating supply
    - whale_detector: net whale accumulation/distribution signal
    - exchange_flow: net inflow/outflow signal

  DOGE:
    - Blockchair: 24h tx count, hodling addresses, node count, mempool
    - BlockCypher: block height, peer count, mempool depth
    - whale_detector: net whale signal
    - exchange_flow: net flow signal

Usage:
    snapshot = aggregate("SOL", sol_price_usd=150.0, prev_balances={})
    snapshot = aggregate("DOGE", doge_price_usd=0.09, prev_balances={})
"""

import structlog
from dataclasses import dataclass, field
from typing import Optional

from engines.onchain.defillama_client import fetch_current_tvl, fetch_protocol_count
from engines.onchain.blockchair_client import fetch_stats as fetch_doge_stats
from engines.onchain.sol_rpc_client import fetch_network_stats as fetch_sol_stats
from engines.onchain.doge_rpc_client import fetch_chain_stats as fetch_doge_chain
from engines.onchain.whale_detector import scan as whale_scan, WhaleScan
from engines.onchain.exchange_flow import compute as exchange_flow_compute, ExchangeFlowResult

logger = structlog.get_logger(__name__)


@dataclass
class OnChainSnapshot:
    """All on-chain signals for a single coin at a single point in time."""

    coin: str

    # SOL-specific (None for DOGE)
    sol_tvl_usd: Optional[float] = None
    sol_protocol_count: Optional[int] = None
    sol_tps_total: Optional[float] = None
    sol_tps_non_vote: Optional[float] = None
    sol_circulating_supply: Optional[float] = None
    sol_epoch: Optional[int] = None

    # DOGE-specific (None for SOL)
    doge_transactions_24h: Optional[int] = None
    doge_hodling_addresses: Optional[int] = None
    doge_nodes: Optional[int] = None
    doge_mempool_txs: Optional[int] = None
    doge_block_height: Optional[int] = None
    doge_peer_count: Optional[int] = None

    # Shared signals (both coins)
    whale_signal: float = 0.0  # [-1,+1]: positive = accumulation
    exchange_flow_signal: float = 0.0  # [-1,+1]: positive = outflow (bullish)

    errors: list[str] = field(default_factory=list)


def aggregate(
    coin: str,
    price_usd: float = 0.0,
    prev_balances: Optional[dict[str, float]] = None,
) -> OnChainSnapshot:
    """Fetch and aggregate all on-chain signals for a single coin.

    Args:
        coin: 'SOL' or 'DOGE'.
        price_usd: Current asset price in USD (required for whale/flow signals).
        prev_balances: Previous address balance snapshot for change detection.
                       Pass {} or None on first run — whale/flow signals will be 0.

    Returns:
        OnChainSnapshot with all available signals. Failed fetches are logged
        as warnings and their fields left as None — downstream handles gracefully.
    """
    if prev_balances is None:
        prev_balances = {}

    snapshot = OnChainSnapshot(coin=coin)

    if coin == "SOL":
        _aggregate_sol(snapshot, price_usd, prev_balances)
    elif coin == "DOGE":
        _aggregate_doge(snapshot, price_usd, prev_balances)
    else:
        raise ValueError(f"Unknown coin: {coin}. Expected 'SOL' or 'DOGE'.")

    logger.info(
        "onchain_aggregated",
        coin=coin,
        whale_signal=snapshot.whale_signal,
        exchange_flow_signal=snapshot.exchange_flow_signal,
        errors=snapshot.errors,
    )
    return snapshot


def _aggregate_sol(
    snapshot: OnChainSnapshot,
    price_usd: float,
    prev_balances: dict[str, float],
) -> None:
    """Populate SOL-specific fields in the snapshot (mutates in place)."""

    # DeFiLlama — TVL
    try:
        snapshot.sol_tvl_usd = fetch_current_tvl()
    except Exception as e:
        snapshot.errors.append(f"defillama_tvl: {e}")
        logger.warning("onchain_sol_tvl_failed", exc_info=True)

    # DeFiLlama — protocol count
    try:
        snapshot.sol_protocol_count = fetch_protocol_count()
    except Exception as e:
        snapshot.errors.append(f"defillama_protocols: {e}")
        logger.warning("onchain_sol_protocols_failed", exc_info=True)

    # Solana public RPC — network stats
    try:
        rpc = fetch_sol_stats()
        snapshot.sol_tps_total = rpc.tps_total
        snapshot.sol_tps_non_vote = rpc.tps_non_vote
        snapshot.sol_circulating_supply = rpc.circulating_supply
        snapshot.sol_epoch = rpc.epoch
    except Exception as e:
        snapshot.errors.append(f"sol_rpc: {e}")
        logger.warning("onchain_sol_rpc_failed", exc_info=True)

    # Whale detector
    try:
        wscan: WhaleScan = whale_scan("SOL", price_usd, prev_balances)
        snapshot.whale_signal = wscan.net_signal
    except Exception as e:
        snapshot.errors.append(f"whale_sol: {e}")
        logger.warning("onchain_sol_whale_failed", exc_info=True)

    # Exchange flow
    try:
        flow: ExchangeFlowResult = exchange_flow_compute("SOL", price_usd, prev_balances)
        snapshot.exchange_flow_signal = flow.net_flow_signal
    except Exception as e:
        snapshot.errors.append(f"flow_sol: {e}")
        logger.warning("onchain_sol_flow_failed", exc_info=True)


def _aggregate_doge(
    snapshot: OnChainSnapshot,
    price_usd: float,
    prev_balances: dict[str, float],
) -> None:
    """Populate DOGE-specific fields in the snapshot (mutates in place)."""

    # Blockchair — chain stats
    try:
        bc = fetch_doge_stats()
        snapshot.doge_transactions_24h = bc.transactions_24h
        snapshot.doge_hodling_addresses = bc.hodling_addresses
        snapshot.doge_nodes = bc.nodes
        snapshot.doge_mempool_txs = bc.mempool_transactions
    except Exception as e:
        snapshot.errors.append(f"blockchair: {e}")
        logger.warning("onchain_doge_blockchair_failed", exc_info=True)

    # BlockCypher — chain stats (peer count, block height)
    try:
        bcy = fetch_doge_chain()
        snapshot.doge_block_height = bcy.block_height
        snapshot.doge_peer_count = bcy.peer_count
        if snapshot.doge_mempool_txs is None:
            snapshot.doge_mempool_txs = bcy.unconfirmed_count
    except Exception as e:
        snapshot.errors.append(f"blockcypher: {e}")
        logger.warning("onchain_doge_blockcypher_failed", exc_info=True)

    # Whale detector
    try:
        wscan: WhaleScan = whale_scan("DOGE", price_usd, prev_balances)
        snapshot.whale_signal = wscan.net_signal
    except Exception as e:
        snapshot.errors.append(f"whale_doge: {e}")
        logger.warning("onchain_doge_whale_failed", exc_info=True)

    # Exchange flow
    try:
        flow: ExchangeFlowResult = exchange_flow_compute("DOGE", price_usd, prev_balances)
        snapshot.exchange_flow_signal = flow.net_flow_signal
    except Exception as e:
        snapshot.errors.append(f"flow_doge: {e}")
        logger.warning("onchain_doge_flow_failed", exc_info=True)


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()
    for coin, price in [("SOL", 150.0), ("DOGE", 0.09)]:
        snap = aggregate(coin, price_usd=price)
        print(f"\n=== {coin} OnChain Snapshot ===")
        for k, v in snap.__dict__.items():
            if v is not None and k != "errors":
                print(f"  {k}: {v}")
        if snap.errors:
            print(f"  errors: {snap.errors}")

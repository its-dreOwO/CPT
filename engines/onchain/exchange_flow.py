"""
engines/onchain/exchange_flow.py

Computes net exchange flow signals for SOL and DOGE.

Net inflow to exchange wallets = selling pressure (bearish signal).
Net outflow from exchange wallets = accumulation (bullish signal).

Strategy:
  - Queries current balances of known exchange hot wallets
    (defined in config/constants.py: EXCHANGE_WALLETS_SOL / EXCHANGE_WALLETS_DOGE)
  - Compares to previous snapshot to compute delta per exchange
  - Aggregates into a single net_flow signal in [-1.0, +1.0]

Signal:
  +1.0 = strong net outflow from exchanges (coins leaving → accumulation)
  -1.0 = strong net inflow to exchanges (coins arriving → selling pressure)
   0.0 = balanced or no data
"""

import structlog
from dataclasses import dataclass, field

from config.constants import EXCHANGE_WALLETS_SOL, EXCHANGE_WALLETS_DOGE

logger = structlog.get_logger(__name__)

# Minimum balance delta (USD) to register as a meaningful flow
_MIN_FLOW_USD = 100_000.0


@dataclass
class ExchangeFlowResult:
    """Net exchange flow result for one coin."""

    coin: str
    net_flow_signal: float  # [-1.0, +1.0]: positive = outflow (bullish)
    inflow_usd: float
    outflow_usd: float
    wallets_checked: int
    exchange_deltas: dict[str, float] = field(default_factory=dict)  # exchange -> USD delta


def compute(
    coin: str,
    price_usd: float,
    previous_balances: dict[str, float],
) -> ExchangeFlowResult:
    """Compute net exchange flow signal for a coin.

    Args:
        coin: 'SOL' or 'DOGE'.
        price_usd: Current price in USD (used to convert balance changes to USD).
        previous_balances: Dict of {address: balance} from prior scan.
                           Pass empty dict on first run — returns zero signal.

    Returns:
        ExchangeFlowResult with net_flow_signal in [-1.0, +1.0].
    """
    if price_usd <= 0:
        logger.warning("exchange_flow_invalid_price", coin=coin)
        return ExchangeFlowResult(
            coin=coin, net_flow_signal=0.0, inflow_usd=0.0, outflow_usd=0.0, wallets_checked=0
        )

    wallet_map: dict[str, list[str]] = (
        EXCHANGE_WALLETS_SOL if coin == "SOL" else EXCHANGE_WALLETS_DOGE
    )

    # Flatten to list of (exchange_name, address)
    all_wallets = [(exchange, addr) for exchange, addrs in wallet_map.items() for addr in addrs]

    if not all_wallets:
        logger.debug("exchange_wallets_empty", coin=coin)
        return ExchangeFlowResult(
            coin=coin, net_flow_signal=0.0, inflow_usd=0.0, outflow_usd=0.0, wallets_checked=0
        )

    if coin == "SOL":
        from engines.onchain.sol_rpc_client import fetch_address_balance as _balance
    else:
        from engines.onchain.doge_rpc_client import fetch_address_balance as _balance

    total_inflow_usd = 0.0
    total_outflow_usd = 0.0
    exchange_deltas: dict[str, float] = {}
    wallets_checked = 0

    for exchange, address in all_wallets:
        try:
            current = _balance(address)
            prev = previous_balances.get(address)
            if prev is None:
                continue

            delta_coins = current - prev
            delta_usd = delta_coins * price_usd

            if abs(delta_usd) < _MIN_FLOW_USD:
                continue  # noise, ignore

            wallets_checked += 1
            exchange_deltas[exchange] = exchange_deltas.get(exchange, 0.0) + delta_usd

            if delta_usd > 0:
                total_inflow_usd += delta_usd  # coins arrived at exchange = inflow
            else:
                total_outflow_usd += abs(delta_usd)  # coins left exchange = outflow

        except Exception:
            logger.warning(
                "exchange_flow_balance_failed",
                coin=coin,
                exchange=exchange,
                address=address[:8] + "...",
            )

    total_flow = total_inflow_usd + total_outflow_usd
    net_flow_signal = 0.0
    if total_flow > 0:
        # Outflow is bullish (+), inflow is bearish (-)
        net_flow_signal = (total_outflow_usd - total_inflow_usd) / total_flow

    logger.info(
        "exchange_flow_computed",
        coin=coin,
        inflow_usd=round(total_inflow_usd, 2),
        outflow_usd=round(total_outflow_usd, 2),
        net_flow_signal=round(net_flow_signal, 4),
        wallets_checked=wallets_checked,
    )

    return ExchangeFlowResult(
        coin=coin,
        net_flow_signal=net_flow_signal,
        inflow_usd=total_inflow_usd,
        outflow_usd=total_outflow_usd,
        wallets_checked=wallets_checked,
        exchange_deltas=exchange_deltas,
    )

"""
engines/onchain/whale_detector.py

Detects whale wallet activity for SOL and DOGE.

A "whale" is any wallet holding > WHALE_THRESHOLD_USD worth of the asset.
Threshold is defined in config/constants.py: WHALE_THRESHOLD_USD = $500,000.

Strategy:
  - Maintains a hardcoded watchlist of known large wallets per coin
    (populated in config/constants.py: WHALE_WALLETS_SOL / WHALE_WALLETS_DOGE)
  - Queries current balances via sol_rpc_client / doge_rpc_client
  - Computes balance change vs. last snapshot to detect accumulation/distribution

Signal output:
  +1.0 = net whale accumulation (bullish)
  -1.0 = net whale distribution (selling pressure)
   0.0 = no significant change or insufficient data
"""

import structlog
from dataclasses import dataclass, field

from config.constants import WHALE_THRESHOLD_USD, WHALE_WALLETS_SOL, WHALE_WALLETS_DOGE

logger = structlog.get_logger(__name__)

# Minimum balance change (as % of total balance) to register as a move
_MIN_CHANGE_PCT = 5.0


@dataclass
class WhaleScan:
    """Result of a whale watchlist scan for one coin."""

    coin: str
    wallets_checked: int
    net_signal: float  # [-1.0, +1.0]
    accumulators: list[str] = field(default_factory=list)  # addresses accumulating
    distributors: list[str] = field(default_factory=list)  # addresses distributing


def scan(
    coin: str,
    price_usd: float,
    previous_balances: dict[str, float],
) -> WhaleScan:
    """Scan whale watchlist for significant balance changes.

    Args:
        coin: 'SOL' or 'DOGE'.
        price_usd: Current price of the asset in USD (to apply threshold filter).
        previous_balances: Dict of {address: balance} from the last scan.
                           Pass empty dict on first run — no signal will be emitted.

    Returns:
        WhaleScan with net_signal in [-1.0, +1.0].
        Returns zero-signal scan if watchlist is empty or price is 0.
    """
    if price_usd <= 0:
        logger.warning("whale_scan_invalid_price", coin=coin)
        return WhaleScan(coin=coin, wallets_checked=0, net_signal=0.0)

    wallets: list[str] = WHALE_WALLETS_SOL if coin == "SOL" else WHALE_WALLETS_DOGE

    if not wallets:
        logger.debug("whale_watchlist_empty", coin=coin)
        return WhaleScan(coin=coin, wallets_checked=0, net_signal=0.0)

    # Lazy import to avoid circular dependencies
    if coin == "SOL":
        from engines.onchain.sol_rpc_client import fetch_address_balance as _balance
    else:
        from engines.onchain.doge_rpc_client import fetch_address_balance as _balance

    accumulators: list[str] = []
    distributors: list[str] = []
    checked = 0

    for address in wallets:
        try:
            current = _balance(address)
            value_usd = current * price_usd

            if value_usd < WHALE_THRESHOLD_USD:
                continue  # below threshold — skip

            checked += 1
            prev = previous_balances.get(address)
            if prev is None or prev == 0:
                continue  # no baseline yet

            pct_change = ((current - prev) / prev) * 100
            if pct_change >= _MIN_CHANGE_PCT:
                accumulators.append(address)
            elif pct_change <= -_MIN_CHANGE_PCT:
                distributors.append(address)

        except Exception:
            logger.warning("whale_balance_failed", coin=coin, address=address[:8] + "...")

    total = len(accumulators) + len(distributors)
    net_signal = 0.0
    if total > 0:
        net_signal = (len(accumulators) - len(distributors)) / total

    logger.info(
        "whale_scan_complete",
        coin=coin,
        wallets_checked=checked,
        accumulators=len(accumulators),
        distributors=len(distributors),
        net_signal=round(net_signal, 4),
    )

    return WhaleScan(
        coin=coin,
        wallets_checked=checked,
        net_signal=net_signal,
        accumulators=accumulators,
        distributors=distributors,
    )

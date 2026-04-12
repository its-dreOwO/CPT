"""
engines/prices/price_aggregator.py

Aggregates raw price ticks into OHLCV candles at 1-minute and 1-hour resolutions.
Called by PriceStream on each incoming tick. Results are persisted to DB by the
orchestrator on each scheduler cycle.
"""

import structlog
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = structlog.get_logger(__name__)


@dataclass
class Tick:
    """A single raw price tick from the WebSocket stream."""

    coin: str
    timestamp: datetime
    price: float
    volume: float = 0.0


@dataclass
class OHLCVCandle:
    """A completed OHLCV candle at a given resolution."""

    coin: str
    timestamp: datetime  # candle open time (UTC, floored to resolution)
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceAggregator:
    """In-memory tick buffer that produces OHLCV candles on demand.

    Ticks are stored per-coin. Call get_candles() to aggregate them into
    OHLCV at the desired resolution. Call flush() after persisting to DB
    to clear the buffer and prevent unbounded memory growth.
    """

    def __init__(self) -> None:
        self._ticks: dict[str, list[Tick]] = {}

    def add_tick(self, coin: str, price: float, volume: float = 0.0) -> None:
        """Record a new price tick.

        Args:
            coin: 'SOL' or 'DOGE'.
            price: Last traded price in USDT.
            volume: Trade size in base currency units (optional).
        """
        if coin not in self._ticks:
            self._ticks[coin] = []

        self._ticks[coin].append(
            Tick(
                coin=coin,
                timestamp=datetime.now(timezone.utc),
                price=price,
                volume=volume,
            )
        )

    def get_candles(self, coin: str, resolution: str = "1h") -> pd.DataFrame:
        """Aggregate stored ticks into OHLCV candles at the given resolution.

        Args:
            coin: 'SOL' or 'DOGE'.
            resolution: pandas resample rule. Use '1min' for 1-minute or '1h' for 1-hour.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            UTC DatetimeIndex floored to the requested resolution.
            Empty DataFrame if no ticks are stored for this coin.
        """
        ticks = self._ticks.get(coin, [])
        if not ticks:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            [{"timestamp": t.timestamp, "price": t.price, "volume": t.volume} for t in ticks]
        )
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)

        ohlcv = df["price"].resample(resolution).ohlc()
        ohlcv["volume"] = df["volume"].resample(resolution).sum()
        ohlcv = ohlcv.dropna(subset=["open"])

        logger.debug(
            "candles_aggregated",
            coin=coin,
            resolution=resolution,
            candles=len(ohlcv),
            ticks=len(ticks),
        )
        return ohlcv

    def latest_price(self, coin: str) -> Optional[float]:
        """Return the most recent tick price for a coin, or None if no ticks.

        Args:
            coin: 'SOL' or 'DOGE'.
        """
        ticks = self._ticks.get(coin)
        if not ticks:
            return None
        return ticks[-1].price

    def tick_count(self, coin: str) -> int:
        """Return the number of buffered ticks for a coin."""
        return len(self._ticks.get(coin, []))

    def flush(self, coin: str) -> None:
        """Clear the tick buffer for a coin after candles have been persisted.

        Args:
            coin: 'SOL' or 'DOGE'.
        """
        if coin in self._ticks:
            count = len(self._ticks[coin])
            self._ticks[coin] = []
            logger.debug("tick_buffer_flushed", coin=coin, flushed=count)

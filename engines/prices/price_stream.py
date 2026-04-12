"""
engines/prices/price_stream.py

Live price tick stream via ccxt.pro WebSocket (Binance).
Subscribes to SOL/USDT and DOGE/USDT trade streams.

On each tick:
  - Passes the price to PriceAggregator for OHLCV candle building
  - Checks if price moved >= LIVE_PREDICTION_TRIGGER_PCT from last prediction price
  - If so, fires the on_trigger callback (wired to prediction_pipeline in orchestrator)
"""

import asyncio
import structlog
from typing import TYPE_CHECKING, Callable, Awaitable, Optional

import ccxt.pro as ccxtpro

from config.constants import CCXT_SYMBOLS, LIVE_PREDICTION_TRIGGER_PCT

if TYPE_CHECKING:
    from engines.prices.price_aggregator import PriceAggregator

logger = structlog.get_logger(__name__)

# Type alias for the trigger callback
TriggerCallback = Callable[[str, float], Awaitable[None]]


class PriceStream:
    """Streams live price ticks from Binance via ccxt.pro WebSocket.

    Usage::

        async def on_trigger(coin: str, price: float) -> None:
            await prediction_pipeline.run(coin)

        stream = PriceStream(on_trigger=on_trigger, aggregator=price_aggregator)
        await stream.start()   # runs until cancelled
    """

    def __init__(
        self,
        on_trigger: Optional[TriggerCallback] = None,
        aggregator: Optional["PriceAggregator"] = None,
    ) -> None:
        """
        Args:
            on_trigger: Async callback fired when price moves >= LIVE_PREDICTION_TRIGGER_PCT.
                        Signature: async def on_trigger(coin: str, price: float) -> None
            aggregator: PriceAggregator instance to feed ticks into for candle building.
        """
        self._exchange = ccxtpro.binance({"newUpdates": True})
        self._on_trigger = on_trigger
        self._aggregator = aggregator
        self._last_prediction_prices: dict[str, Optional[float]] = {
            coin: None for coin in CCXT_SYMBOLS
        }
        self._running = False

    async def _handle_tick(self, coin: str, price: float, volume: float) -> None:
        """Process one price tick: feed aggregator and check for trigger."""
        # Feed into candle aggregator if wired
        if self._aggregator is not None:
            self._aggregator.add_tick(coin, price, volume)

        # Check trigger threshold
        last = self._last_prediction_prices.get(coin)
        if last is None:
            self._last_prediction_prices[coin] = price
            return

        pct_move = abs(price - last) / last * 100
        if pct_move >= LIVE_PREDICTION_TRIGGER_PCT:
            logger.info(
                "price_trigger_fired",
                coin=coin,
                price=price,
                last_prediction_price=last,
                pct_move=round(pct_move, 3),
            )
            self._last_prediction_prices[coin] = price
            if self._on_trigger:
                await self._on_trigger(coin, price)

    async def _watch_symbol(self, symbol: str, coin: str) -> None:
        """Continuously watch one symbol's trade stream."""
        while self._running:
            try:
                trades = await self._exchange.watch_trades(symbol)
                if trades:
                    latest = trades[-1]
                    price = float(latest["price"])
                    volume = float(latest.get("amount", 0.0))
                    await self._handle_tick(coin, price, volume)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("price_stream_error", symbol=symbol, error=str(exc))
                await asyncio.sleep(5)  # brief back-off before reconnect

    async def start(self) -> None:
        """Start streaming all coins simultaneously. Runs until stop() is called."""
        self._running = True
        logger.info("price_stream_starting", symbols=list(CCXT_SYMBOLS.values()))
        await asyncio.gather(
            *[self._watch_symbol(symbol, coin) for coin, symbol in CCXT_SYMBOLS.items()]
        )

    async def stop(self) -> None:
        """Gracefully stop all streams and close the WebSocket connection."""
        self._running = False
        await self._exchange.close()
        logger.info("price_stream_stopped")

    def update_last_prediction_price(self, coin: str, price: float) -> None:
        """Update the reference price after a prediction is made.

        Called by prediction_pipeline immediately after each prediction run
        so the next trigger is based on the freshest prediction price.

        Args:
            coin: 'SOL' or 'DOGE'.
            price: The price at which the prediction was made.
        """
        self._last_prediction_prices[coin] = price
        logger.debug("last_prediction_price_updated", coin=coin, price=price)

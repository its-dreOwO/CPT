import asyncio
import time
from typing import Dict
from threading import Lock


class RateLimiter:
    """
    A simple token bucket rate limiter to keep our API calls within safe limits.
    Can be used both synchronously and asynchronously.
    """

    def __init__(self, limits_per_min: Dict[str, int]):
        self.limits = limits_per_min
        self._tokens: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}
        for service, limit in limits_per_min.items():
            self._tokens[service] = float(limit)

        self._lock = Lock()

    def _get_tokens(self, service: str) -> float:
        """Internal helper to calculate current tokens for a service."""
        if service not in self.limits:
            return 1.0  # default unbounded if not registered

        rate_per_sec = self.limits[service] / 60.0
        current_time = time.time()

        if service not in self._last_update:
            self._last_update[service] = current_time
            self._tokens[service] = float(self.limits[service])
            return self._tokens[service]

        elapsed = current_time - self._last_update[service]
        self._last_update[service] = current_time

        # Add new tokens based on elapsed time, capped at max limits
        self._tokens[service] = min(
            float(self.limits[service]), self._tokens[service] + elapsed * rate_per_sec
        )
        return self._tokens[service]

    def acquire(self, service: str) -> float:
        """
        Consumes one token for the service. Returns time to wait if none available.
        """
        with self._lock:
            if service not in self.limits:
                return 0.0

            tokens = self._get_tokens(service)
            if tokens >= 1.0:
                self._tokens[service] -= 1.0
                return 0.0

            # Need to wait
            rate_per_sec = self.limits[service] / 60.0
            wait_time = (1.0 - tokens) / rate_per_sec
            return wait_time

    def wait(self, service: str) -> None:
        """Synchronously wait for a token."""
        wait_time = self.acquire(service)
        if wait_time > 0:
            time.sleep(wait_time)

    async def await_token(self, service: str) -> None:
        """Asynchronously wait for a token."""
        wait_time = self.acquire(service)
        if wait_time > 0:
            await asyncio.sleep(wait_time)


# We will instantiate a global instance driven by the settings/constants in actual use.
# For now this module exposes the class.

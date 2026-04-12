import asyncio
import logging
from functools import wraps
from typing import Any, Callable

import httpx
import requests
from tenacity import (
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
    Retrying,
)

logger = logging.getLogger(__name__)

# Exceptions to retry on
RETRY_EXCEPTIONS = (
    httpx.HTTPError,
    requests.exceptions.RequestException,
    ConnectionError,
    TimeoutError,
)


def with_retry(
    max_attempts: int = 3, min_wait_sec: float = 2.0, max_wait_sec: float = 10.0
) -> Callable:
    """
    A decorator to retry a function if it raises common HTTP or network errors.
    Supports both synchronous and asynchronous functions.
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
                    wait=wait_exponential(multiplier=1, min=min_wait_sec, max=max_wait_sec),
                    stop=stop_after_attempt(max_attempts),
                    reraise=True,
                ):
                    with attempt:
                        if attempt.retry_state.attempt_number > 1:
                            logger.warning(
                                f"Retrying {func.__name__} (attempt {attempt.retry_state.attempt_number}/{max_attempts}) "
                                f"after error: {attempt.retry_state.outcome.exception()}"
                            )
                        return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                for attempt in Retrying(
                    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
                    wait=wait_exponential(multiplier=1, min=min_wait_sec, max=max_wait_sec),
                    stop=stop_after_attempt(max_attempts),
                    reraise=True,
                ):
                    with attempt:
                        if attempt.retry_state.attempt_number > 1:
                            logger.warning(
                                f"Retrying {func.__name__} (attempt {attempt.retry_state.attempt_number}/{max_attempts}) "
                                f"after error: {attempt.retry_state.outcome.exception()}"
                            )
                        return func(*args, **kwargs)

            return sync_wrapper

    return decorator

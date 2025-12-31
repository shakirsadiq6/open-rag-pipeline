"""
Retry utilities for robust error handling.
"""

import time
from functools import wraps
from typing import Any, Callable, Type, TypeVar

from open_rag_pipeline.exceptions import OpenRAGPipelineError
from open_rag_pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Factor to multiply delay by on each retry
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                            f"Retrying in {delay:.2f} seconds...",
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")

            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            raise OpenRAGPipelineError("Function failed after all retries")

        return wrapper

    return decorator


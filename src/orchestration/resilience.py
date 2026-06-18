import logging
import asyncio
from functools import wraps

logger = logging.getLogger("resilience")


class ValidationError(Exception):
    pass


class DataPipelineError(Exception):
    pass


class ModelError(Exception):
    pass


class DatabaseError(Exception):
    pass


class ExternalServiceError(Exception):
    pass


def retry_with_backoff(retries=3, backoff_factor=1.5):
    """
    Decorator for async retry logic with exponential backoff.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= retries:
                        logger.error(f"Max retries reached for {func.__name__}: {str(e)}")
                        raise e
                    sleep_time = backoff_factor**attempt
                    logger.warning(
                        f"Retry {attempt}/{retries} for {func.__name__} after {sleep_time}s due to: {str(e)}"
                    )
                    await asyncio.sleep(sleep_time)

        return wrapper

    return decorator

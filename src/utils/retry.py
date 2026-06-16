"""
Retry Utilities with Exponential Backoff

Provides decorators and context managers for retry logic with exponential backoff.
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_ms: int = 100,
        max_delay_ms: int = 5000,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay_ms: Initial delay in milliseconds
            max_delay_ms: Maximum delay in milliseconds
            exponential_base: Base for exponential backoff calculation
            jitter: Add random jitter to delays
        """
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay_ms(self, attempt: int) -> int:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Attempt number (0-indexed)
        
        Returns:
            Delay in milliseconds
        """
        # Exponential backoff: delay = initial_delay * (base ^ attempt)
        delay_ms = self.initial_delay_ms * (self.exponential_base ** attempt)
        
        # Cap at maximum delay
        delay_ms = min(delay_ms, self.max_delay_ms)
        
        # Add jitter if enabled
        if self.jitter:
            import random
            jitter_factor = random.uniform(0.8, 1.2)
            delay_ms *= jitter_factor
        
        return int(delay_ms)


def retry_sync(
    max_retries: int = 3,
    initial_delay_ms: int = 100,
    max_delay_ms: int = 5000,
    exponential_base: float = 2.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for synchronous functions with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay_ms: Initial delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to delays
        on_retry: Callback function called on retry (attempt_number, exception)
    
    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay_ms=initial_delay_ms,
        max_delay_ms=max_delay_ms,
        exponential_base=exponential_base,
        jitter=jitter,
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay_ms = config.get_delay_ms(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'attempt': attempt + 1,
                                'error': str(e),
                                'retry_delay_ms': delay_ms,
                            }
                        )
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        time.sleep(delay_ms / 1000.0)
                    else:
                        logger.error(
                            f"All retry attempts failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'attempts': max_retries + 1,
                                'error': str(e),
                            }
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


def retry_async(
    max_retries: int = 3,
    initial_delay_ms: int = 100,
    max_delay_ms: int = 5000,
    exponential_base: float = 2.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for async functions with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay_ms: Initial delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to delays
        on_retry: Callback function called on retry (attempt_number, exception)
    
    Returns:
        Decorated async function with retry logic
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay_ms=initial_delay_ms,
        max_delay_ms=max_delay_ms,
        exponential_base=exponential_base,
        jitter=jitter,
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay_ms = config.get_delay_ms(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'attempt': attempt + 1,
                                'error': str(e),
                                'retry_delay_ms': delay_ms,
                            }
                        )
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        await asyncio.sleep(delay_ms / 1000.0)
                    else:
                        logger.error(
                            f"All retry attempts failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'attempts': max_retries + 1,
                                'error': str(e),
                            }
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


class RetryContext:
    """Context manager for retry logic (synchronous)."""
    
    def __init__(
        self,
        operation_name: str,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
    ):
        """
        Initialize retry context manager.
        
        Args:
            operation_name: Name of the operation for logging
            config: RetryConfig instance (uses defaults if None)
            on_retry: Callback function called on retry
        """
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self.attempt = 0
    
    def __enter__(self):
        """Enter context."""
        self.attempt = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            return False
        
        if self.attempt < self.config.max_retries:
            delay_ms = self.config.get_delay_ms(self.attempt)
            logger.warning(
                f"Retry {self.operation_name}",
                extra={
                    'operation': self.operation_name,
                    'attempt': self.attempt + 1,
                    'error': str(exc_val),
                    'retry_delay_ms': delay_ms,
                }
            )
            
            if self.on_retry:
                self.on_retry(self.attempt + 1, exc_val)
            
            time.sleep(delay_ms / 1000.0)
            self.attempt += 1
            return True
        
        return False


class RetryContextAsync:
    """Context manager for retry logic (asynchronous)."""
    
    def __init__(
        self,
        operation_name: str,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
    ):
        """
        Initialize async retry context manager.
        
        Args:
            operation_name: Name of the operation for logging
            config: RetryConfig instance (uses defaults if None)
            on_retry: Callback function called on retry
        """
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self.attempt = 0
    
    async def __aenter__(self):
        """Enter context."""
        self.attempt = 0
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            return False
        
        if self.attempt < self.config.max_retries:
            delay_ms = self.config.get_delay_ms(self.attempt)
            logger.warning(
                f"Retry {self.operation_name}",
                extra={
                    'operation': self.operation_name,
                    'attempt': self.attempt + 1,
                    'error': str(exc_val),
                    'retry_delay_ms': delay_ms,
                }
            )
            
            if self.on_retry:
                self.on_retry(self.attempt + 1, exc_val)
            
            await asyncio.sleep(delay_ms / 1000.0)
            self.attempt += 1
            return True
        
        return False


# Example usage:
if __name__ == '__main__':
    @retry_sync(max_retries=3, initial_delay_ms=100)
    def unreliable_operation():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success"
    
    # Test decorator
    print(unreliable_operation())

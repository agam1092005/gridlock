"""
Circuit Breaker Pattern Implementation for Resilience

Implements the circuit breaker pattern to prevent cascading failures
when external dependencies (database, cache, APIs) are unavailable.
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Optional, TypeVar, Union

from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Prevents cascading failures by monitoring error rates and latency.
    State transitions:
    - CLOSED -> OPEN: When error rate > threshold or latency > limit
    - OPEN -> HALF_OPEN: After timeout (30 seconds by default)
    - HALF_OPEN -> CLOSED: If test request succeeds
    - HALF_OPEN -> OPEN: If test request fails
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: float = 0.5,  # 50% error rate
        failure_count_threshold: int = 5,  # minimum failures before opening
        timeout_seconds: int = 30,  # time to wait before half-open
        latency_threshold_ms: float = 500.0,  # latency threshold
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name (e.g., 'database', 'redis', 'model_api')
            failure_threshold: Error rate threshold (0.0-1.0) to open circuit
            failure_count_threshold: Minimum failures before opening
            timeout_seconds: Time to wait in OPEN state before trying HALF_OPEN
            latency_threshold_ms: Latency threshold in milliseconds
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_count_threshold = failure_count_threshold
        self.timeout_seconds = timeout_seconds
        self.latency_threshold_ms = latency_threshold_ms
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_open_time: Optional[float] = None
        self.latencies: list[float] = []
        self.max_latency_history = 100  # Keep last 100 latencies
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        total = self.failure_count + self.success_count
        if total == 0:
            return 0.0
        return self.failure_count / total
    
    def _should_open(self) -> bool:
        """Determine if circuit should open."""
        # Check failure count
        if self.failure_count < self.failure_count_threshold:
            return False
        
        # Check error rate
        error_rate = self._calculate_error_rate()
        if error_rate > self.failure_threshold:
            return True
        
        # Check recent latencies
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            if avg_latency > self.latency_threshold_ms:
                logger.warning(
                    f"Circuit breaker '{self.name}' detected high latency",
                    extra={'cb_latency_ms': avg_latency, 'threshold_ms': self.latency_threshold_ms}
                )
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_open_time is None:
            return False
        
        elapsed = time.time() - self.last_open_time
        return elapsed >= self.timeout_seconds
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker (synchronous).
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker '{self.name}' entering HALF_OPEN state",
                    extra={'cb_name': self.name}
                )
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Track latency
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.max_latency_history:
                self.latencies.pop(0)
            
            # Record success
            self.success_count += 1
            
            # Attempt to close circuit if in half-open
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' closed successfully",
                    extra={'cb_name': self.name}
                )
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure",
                extra={
                    'cb_name': self.name,
                    'error': str(e),
                    'failure_count': self.failure_count,
                    'error_rate': self._calculate_error_rate(),
                }
            )
            
            if self._should_open():
                self.state = CircuitState.OPEN
                self.last_open_time = time.time()
                logger.error(
                    f"Circuit breaker '{self.name}' is now OPEN",
                    extra={
                        'cb_name': self.name,
                        'error_rate': self._calculate_error_rate(),
                        'failure_count': self.failure_count,
                    }
                )
            
            raise
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker '{self.name}' entering HALF_OPEN state",
                    extra={'cb_name': self.name}
                )
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            start_time = time.time()
            result = await func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Track latency
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.max_latency_history:
                self.latencies.pop(0)
            
            # Record success
            self.success_count += 1
            
            # Attempt to close circuit if in half-open
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' closed successfully",
                    extra={'cb_name': self.name}
                )
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure",
                extra={
                    'cb_name': self.name,
                    'error': str(e),
                    'failure_count': self.failure_count,
                    'error_rate': self._calculate_error_rate(),
                }
            )
            
            if self._should_open():
                self.state = CircuitState.OPEN
                self.last_open_time = time.time()
                logger.error(
                    f"Circuit breaker '{self.name}' is now OPEN",
                    extra={
                        'cb_name': self.name,
                        'error_rate': self._calculate_error_rate(),
                        'failure_count': self.failure_count,
                    }
                )
            
            raise
    
    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            'circuit_name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'error_rate': self._calculate_error_rate(),
            'last_failure_time': self.last_failure_time,
            'last_open_time': self.last_open_time,
            'avg_latency_ms': (
                sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            ),
        }
    
    def reset(self):
        """Manually reset circuit breaker."""
        logger.info(f"Circuit breaker '{self.name}' manually reset", extra={'cb_name': self.name})
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_open_time = None
        self.latencies = []


class CircuitBreakerManager:
    """Manage multiple circuit breakers for different services."""
    
    def __init__(self):
        """Initialize manager."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
    
    def register(
        self,
        name: str,
        failure_threshold: float = 0.5,
        failure_count_threshold: int = 5,
        timeout_seconds: int = 30,
        latency_threshold_ms: float = 500.0,
    ) -> CircuitBreaker:
        """
        Register a circuit breaker.
        
        Args:
            name: Unique circuit breaker name
            failure_threshold: Error rate threshold (0.0-1.0)
            failure_count_threshold: Minimum failures before opening
            timeout_seconds: Time to wait in OPEN state
            latency_threshold_ms: Latency threshold in milliseconds
        
        Returns:
            CircuitBreaker instance
        """
        cb = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            failure_count_threshold=failure_count_threshold,
            timeout_seconds=timeout_seconds,
            latency_threshold_ms=latency_threshold_ms,
        )
        self.circuit_breakers[name] = cb
        logger.info(f"Registered circuit breaker '{name}'", extra={'cb_name': name})
        return cb
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_all_status(self) -> dict:
        """Get status of all circuit breakers."""
        return {
            name: cb.get_status()
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()


# Global manager instance
_manager_instance: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get or create circuit breaker manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CircuitBreakerManager()
    return _manager_instance


class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Prevents cascading failures by monitoring error rates and latency.
    State transitions:
    - CLOSED -> OPEN: When error rate > threshold or latency > limit
    - OPEN -> HALF_OPEN: After timeout (30 seconds by default)
    - HALF_OPEN -> CLOSED: If test request succeeds
    - HALF_OPEN -> OPEN: If test request fails
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: float = 0.5,  # 50% error rate
        failure_count_threshold: int = 5,  # minimum failures before opening
        timeout_seconds: int = 30,  # time to wait before half-open
        latency_threshold_ms: float = 500.0,  # latency threshold
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name (e.g., 'database', 'redis', 'model_api')
            failure_threshold: Error rate threshold (0.0-1.0) to open circuit
            failure_count_threshold: Minimum failures before opening
            timeout_seconds: Time to wait in OPEN state before trying HALF_OPEN
            latency_threshold_ms: Latency threshold in milliseconds
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_count_threshold = failure_count_threshold
        self.timeout_seconds = timeout_seconds
        self.latency_threshold_ms = latency_threshold_ms
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_open_time: Optional[float] = None
        self.latencies: list[float] = []
        self.max_latency_history = 100  # Keep last 100 latencies
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        total = self.failure_count + self.success_count
        if total == 0:
            return 0.0
        return self.failure_count / total
    
    def _should_open(self) -> bool:
        """Determine if circuit should open."""
        # Check failure count
        if self.failure_count < self.failure_count_threshold:
            return False
        
        # Check error rate
        error_rate = self._calculate_error_rate()
        if error_rate > self.failure_threshold:
            return True
        
        # Check recent latencies
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            if avg_latency > self.latency_threshold_ms:
                logger.warning(
                    f"Circuit breaker '{self.name}' detected high latency",
                    extra={'avg_latency_ms': avg_latency, 'threshold_ms': self.latency_threshold_ms}
                )
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_open_time is None:
            return False
        
        elapsed = time.time() - self.last_open_time
        return elapsed >= self.timeout_seconds
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker (synchronous).
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker '{self.name}' entering HALF_OPEN state",
                    extra={'name': self.name}
                )
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Track latency
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.max_latency_history:
                self.latencies.pop(0)
            
            # Record success
            self.success_count += 1
            
            # Attempt to close circuit if in half-open
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' closed successfully",
                    extra={'name': self.name}
                )
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure",
                extra={
                    'name': self.name,
                    'error': str(e),
                    'failure_count': self.failure_count,
                    'error_rate': self._calculate_error_rate(),
                }
            )
            
            if self._should_open():
                self.state = CircuitState.OPEN
                self.last_open_time = time.time()
                logger.error(
                    f"Circuit breaker '{self.name}' is now OPEN",
                    extra={
                        'name': self.name,
                        'error_rate': self._calculate_error_rate(),
                        'failure_count': self.failure_count,
                    }
                )
            
            raise
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker '{self.name}' entering HALF_OPEN state",
                    extra={'name': self.name}
                )
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            start_time = time.time()
            result = await func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Track latency
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.max_latency_history:
                self.latencies.pop(0)
            
            # Record success
            self.success_count += 1
            
            # Attempt to close circuit if in half-open
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' closed successfully",
                    extra={'name': self.name}
                )
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure",
                extra={
                    'name': self.name,
                    'error': str(e),
                    'failure_count': self.failure_count,
                    'error_rate': self._calculate_error_rate(),
                }
            )
            
            if self._should_open():
                self.state = CircuitState.OPEN
                self.last_open_time = time.time()
                logger.error(
                    f"Circuit breaker '{self.name}' is now OPEN",
                    extra={
                        'name': self.name,
                        'error_rate': self._calculate_error_rate(),
                        'failure_count': self.failure_count,
                    }
                )
            
            raise
    
    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'error_rate': self._calculate_error_rate(),
            'last_failure_time': self.last_failure_time,
            'last_open_time': self.last_open_time,
            'avg_latency_ms': (
                sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            ),
        }
    
    def reset(self):
        """Manually reset circuit breaker."""
        logger.info(f"Circuit breaker '{self.name}' manually reset", extra={'name': self.name})
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_open_time = None
        self.latencies = []


class CircuitBreakerManager:
    """Manage multiple circuit breakers for different services."""
    
    def __init__(self):
        """Initialize manager."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
    
    def register(
        self,
        name: str,
        failure_threshold: float = 0.5,
        failure_count_threshold: int = 5,
        timeout_seconds: int = 30,
        latency_threshold_ms: float = 500.0,
    ) -> CircuitBreaker:
        """
        Register a circuit breaker.
        
        Args:
            name: Unique circuit breaker name
            failure_threshold: Error rate threshold (0.0-1.0)
            failure_count_threshold: Minimum failures before opening
            timeout_seconds: Time to wait in OPEN state
            latency_threshold_ms: Latency threshold in milliseconds
        
        Returns:
            CircuitBreaker instance
        """
        cb = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            failure_count_threshold=failure_count_threshold,
            timeout_seconds=timeout_seconds,
            latency_threshold_ms=latency_threshold_ms,
        )
        self.circuit_breakers[name] = cb
        logger.info(f"Registered circuit breaker '{name}'", extra={'name': name})
        return cb
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_all_status(self) -> dict:
        """Get status of all circuit breakers."""
        return {
            name: cb.get_status()
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()


# Global manager instance
_manager_instance: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get or create circuit breaker manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CircuitBreakerManager()
    return _manager_instance

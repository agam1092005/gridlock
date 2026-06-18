"""Tests for circuit breaker pattern."""

import pytest

from src.utils import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
    get_circuit_breaker_manager,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state_closed(self):
        """Test that circuit starts in CLOSED state."""
        cb = CircuitBreaker("test")

        assert cb.state == CircuitState.CLOSED

    def test_successful_call(self):
        """Test successful call through circuit breaker."""
        cb = CircuitBreaker("test")

        def successful_func():
            return "success"

        result = cb.call(successful_func)

        assert result == "success"
        assert cb.success_count == 1
        assert cb.failure_count == 0

    def test_single_failure(self):
        """Test single failure doesn't open circuit."""
        cb = CircuitBreaker("test", failure_count_threshold=5)

        def failing_func():
            raise Exception("Failure")

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold reached."""
        cb = CircuitBreaker("test", failure_count_threshold=3)

        def failing_func():
            raise Exception("Failure")

        # Cause 3 failures
        for _ in range(3):
            with pytest.raises(Exception):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

    def test_open_circuit_rejects_requests(self):
        """Test that open circuit rejects new requests."""
        cb = CircuitBreaker("test", failure_count_threshold=1)

        def failing_func():
            raise Exception("Failure")

        # Cause failure to open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)

        # Next request should be rejected
        def dummy_func():
            return "should not execute"

        with pytest.raises(Exception, match="is OPEN"):
            cb.call(dummy_func)

    def test_half_open_state(self):
        """Test transition to HALF_OPEN state."""
        cb = CircuitBreaker(
            "test",
            failure_count_threshold=1,
            timeout_seconds=0.1,  # Short timeout for testing
        )

        def failing_func():
            raise Exception("Failure")

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        import time

        time.sleep(0.2)

        # Next call should transition to HALF_OPEN
        def dummy_func():
            return "success"

        result = cb.call(dummy_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_error_rate_detection(self):
        """Test that circuit opens based on error rate."""
        cb = CircuitBreaker(
            "test",
            failure_count_threshold=5,
            failure_threshold=0.6,  # 60% error rate
        )

        def failing_func():
            raise Exception("Failure")

        def success_func():
            return "success"

        # Cause 3 failures and 2 successes = 60% error rate
        for _ in range(3):
            with pytest.raises(Exception):
                cb.call(failing_func)

        for _ in range(2):
            cb.call(success_func)

        # 3 failures exceeds threshold, should open
        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

    def test_get_status(self):
        """Test getting circuit breaker status."""
        cb = CircuitBreaker("test_breaker")

        status = cb.get_status()

        assert status["name"] == "test_breaker"
        assert status["state"] == CircuitState.CLOSED.value
        assert status["failure_count"] == 0
        assert status["success_count"] == 0

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        cb = CircuitBreaker("test", failure_count_threshold=1)

        def failing_func():
            raise Exception("Failure")

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Manually reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_latency_threshold(self):
        """Test circuit opens based on latency threshold."""
        cb = CircuitBreaker(
            "test",
            failure_count_threshold=5,
            latency_threshold_ms=100.0,
        )

        import time

        def slow_func():
            time.sleep(0.15)  # 150ms > 100ms threshold
            return "done"

        # Multiple slow calls should trigger latency threshold
        for _ in range(5):
            cb.call(slow_func)

        # After 5 slow calls, should open
        with pytest.raises(Exception, match="is OPEN"):
            cb.call(lambda: "test")

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerManager:
    """Test circuit breaker manager."""

    def test_register_circuit_breaker(self):
        """Test registering a circuit breaker."""
        manager = CircuitBreakerManager()

        cb = manager.register("database")

        assert cb.name == "database"
        assert manager.get("database") is cb

    def test_get_nonexistent_circuit_breaker(self):
        """Test getting nonexistent circuit breaker returns None."""
        manager = CircuitBreakerManager()

        cb = manager.get("nonexistent")

        assert cb is None

    def test_get_all_status(self):
        """Test getting status of all circuit breakers."""
        manager = CircuitBreakerManager()

        manager.register("database")
        manager.register("redis")

        all_status = manager.get_all_status()

        assert len(all_status) == 2
        assert "database" in all_status
        assert "redis" in all_status

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        manager = CircuitBreakerManager()

        db_cb = manager.register("database", failure_count_threshold=1)
        redis_cb = manager.register("redis", failure_count_threshold=1)

        # Open both circuits
        def failing_func():
            raise Exception("Failure")

        with pytest.raises(Exception):
            db_cb.call(failing_func)
        with pytest.raises(Exception):
            redis_cb.call(failing_func)

        assert db_cb.state == CircuitState.OPEN
        assert redis_cb.state == CircuitState.OPEN

        # Reset all
        manager.reset_all()

        assert db_cb.state == CircuitState.CLOSED
        assert redis_cb.state == CircuitState.CLOSED

    def test_custom_configuration(self):
        """Test registering with custom configuration."""
        manager = CircuitBreakerManager()

        cb = manager.register(
            "api",
            failure_threshold=0.3,
            failure_count_threshold=3,
            timeout_seconds=60,
            latency_threshold_ms=1000.0,
        )

        assert cb.failure_threshold == 0.3
        assert cb.failure_count_threshold == 3
        assert cb.timeout_seconds == 60
        assert cb.latency_threshold_ms == 1000.0

    def test_global_manager_singleton(self):
        """Test that global manager is a singleton."""
        manager1 = get_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()

        assert manager1 is manager2

"""Tests for retry utilities."""

import asyncio
import time

import pytest

from src.utils import (
    RetryConfig,
    RetryContext,
    RetryContextAsync,
    retry_async,
    retry_sync,
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_basic_delay_calculation(self):
        """Test basic exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay_ms=100,
            exponential_base=2.0,
            jitter=False,
        )

        assert config.get_delay_ms(0) == 100
        assert config.get_delay_ms(1) == 200
        assert config.get_delay_ms(2) == 400

    def test_max_delay_cap(self):
        """Test that delays are capped at maximum."""
        config = RetryConfig(
            initial_delay_ms=100,
            exponential_base=2.0,
            max_delay_ms=500,
            jitter=False,
        )

        assert config.get_delay_ms(0) == 100
        assert config.get_delay_ms(1) == 200
        assert config.get_delay_ms(2) == 400
        assert config.get_delay_ms(3) == 500  # Capped at max
        assert config.get_delay_ms(4) == 500  # Still capped

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delays."""
        config = RetryConfig(
            initial_delay_ms=100,
            jitter=True,
        )

        delays = [config.get_delay_ms(0) for _ in range(10)]

        # All delays should be close to 100 but not exactly equal
        assert all(80 <= d <= 120 for d in delays)
        assert len(set(delays)) > 1  # Should have some variation


class TestRetrySyncDecorator:
    """Test synchronous retry decorator."""

    def test_successful_first_attempt(self):
        """Test function succeeds on first attempt."""
        call_count = 0

        @retry_sync(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test function retries on failure."""
        call_count = 0

        @retry_sync(max_retries=2, initial_delay_ms=10)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First attempt fails")
            return "success"

        result = failing_then_success()

        assert result == "success"
        assert call_count == 2

    def test_failure_after_retries_exhausted(self):
        """Test that exception is raised after all retries exhausted."""
        call_count = 0

        @retry_sync(max_retries=2, initial_delay_ms=10)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retries = []

        def on_retry(attempt, exception):
            retries.append((attempt, str(exception)))

        call_count = 0

        @retry_sync(max_retries=2, initial_delay_ms=10, on_retry=on_retry)
        def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"

        result = fails_then_succeeds()

        assert result == "success"
        assert len(retries) == 2
        assert retries[0][0] == 1
        assert retries[1][0] == 2


class TestRetryAsyncDecorator:
    """Test asynchronous retry decorator."""

    @pytest.mark.asyncio
    async def test_async_successful_first_attempt(self):
        """Test async function succeeds on first attempt."""
        call_count = 0

        @retry_async(max_retries=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_failure(self):
        """Test async function retries on failure."""
        call_count = 0

        @retry_async(max_retries=2, initial_delay_ms=10)
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First attempt fails")
            return "success"

        result = await failing_then_success()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_failure_after_retries_exhausted(self):
        """Test that exception is raised after all async retries exhausted."""
        call_count = 0

        @retry_async(max_retries=2, initial_delay_ms=10)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            await always_fails()

        assert call_count == 3  # Initial + 2 retries


class TestRetryContext:
    """Test synchronous retry context manager."""

    def test_retry_context_success(self):
        """Test retry context with successful operation."""
        attempts = 0

        def operation_that_fails():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("First attempt")
            return "success"

        config = RetryConfig(max_retries=3, initial_delay_ms=10)
        ctx = RetryContext("test_op", config=config)

        for attempt in range(config.max_retries + 1):
            try:
                with ctx:
                    result = operation_that_fails()
                    break
            except ValueError:
                if attempt == config.max_retries:
                    raise

        assert result == "success"
        assert attempts == 2

    def test_retry_context_exhaustion(self):
        """Test retry context exhaustion."""
        config = RetryConfig(max_retries=2, initial_delay_ms=10)
        attempts = 0

        def failing_operation():
            nonlocal attempts
            attempts += 1
            raise ValueError("Always fails")

        ctx = RetryContext("test_op", config=config)
        with pytest.raises(ValueError):
            for attempt in range(config.max_retries + 1):
                try:
                    with ctx:
                        failing_operation()
                except ValueError:
                    if attempt == config.max_retries:
                        raise


class TestRetryContextAsync:
    """Test asynchronous retry context manager."""

    @pytest.mark.asyncio
    async def test_retry_context_async_success(self):
        """Test async retry context with successful operation."""
        attempts = 0

        async def operation_that_fails():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("First attempt")
            return "success"

        config = RetryConfig(max_retries=3, initial_delay_ms=10)
        ctx = RetryContextAsync("test_op", config=config)

        for attempt in range(config.max_retries + 1):
            try:
                async with ctx:
                    result = await operation_that_fails()
                    break
            except ValueError:
                if attempt == config.max_retries:
                    raise

        assert result == "success"
        assert attempts == 2


class TestRetryTiming:
    """Test retry timing behavior."""

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff adds appropriate delays."""
        call_times = []

        @retry_sync(
            max_retries=2,
            initial_delay_ms=50,
            exponential_base=2.0,
            jitter=False,
        )
        def slow_failing_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Failure")
            return "success"

        start_time = time.time()
        slow_failing_func()
        total_time = time.time() - start_time

        # Should have delays: ~50ms after first call, ~100ms after second
        # Total should be roughly 150ms
        assert total_time >= 0.1  # At least 100ms
        assert len(call_times) == 3

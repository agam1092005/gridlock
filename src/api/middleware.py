import time
import logging
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger("api")


class TimingAndLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request timing, logging, and basic rate limiting."""

    def __init__(self, app):
        super().__init__(app)
        # Simple in-memory rate limiting (use Redis in production)
        self.request_counts = defaultdict(lambda: {"count": 0, "reset_time": time.time() + 60})
        self.failed_auth = defaultdict(lambda: {"count": 0, "blocked_until": 0})

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract API key from Authorization header
        auth_header = request.headers.get("Authorization")
        api_key = None
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix

        # Rate limiting: Check if API key is blocked due to auth failures
        if api_key:
            blocked_data = self.failed_auth[api_key]
            if blocked_data["blocked_until"] > time.time():
                remaining_seconds = blocked_data["blocked_until"] - time.time()
                logger.warning(f"IP blocked due to failed auth attempts: {request.client}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many failed authentication attempts. Retry after {remaining_seconds:.0f} seconds",
                    headers={"Retry-After": str(int(remaining_seconds))},
                )

        # Basic per-minute rate limiting (1000 requests/minute = ~16.7 req/sec)
        current_time = time.time()
        key = api_key or request.client.host if request.client else "unknown"

        rate_data = self.request_counts[key]

        # Reset counter if minute has passed
        if current_time >= rate_data["reset_time"]:
            rate_data["count"] = 0
            rate_data["reset_time"] = current_time + 60

        # Check rate limit (1000 requests per minute)
        if rate_data["count"] >= 1000:
            logger.warning(f"Rate limit exceeded for key: {key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: 1000 requests per minute",
                headers={"Retry-After": str(int(rate_data["reset_time"] - current_time))},
            )

        rate_data["count"] += 1

        start_time = time.time()

        try:
            response = await call_next(request)
        except HTTPException as e:
            # Track authentication failures
            if e.status_code == status.HTTP_401_UNAUTHORIZED and api_key:
                blocked_data = self.failed_auth[api_key]
                blocked_data["count"] += 1

                # Block after 5 failed attempts for 15 minutes
                if blocked_data["count"] >= 5:
                    blocked_data["blocked_until"] = time.time() + 900  # 15 minutes
                    logger.error(f"API key blocked after 5 failed auth attempts: {api_key[:8]}...")
            raise
        except Exception as e:
            logger.error(f"Request {request_id} failed with error: {str(e)}", exc_info=True)
            raise

        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            f"Request {request_id} completed | "
            f"Method: {request.method} | "
            f"Path: {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Latency: {process_time:.2f}ms"
        )
        return response


def verify_api_key(api_key: str) -> bool:
    """
    Verify API key against database.
    In production, this would query a database for valid API keys.

    Args:
        api_key: The API key to verify

    Returns:
        bool: True if key is valid and active, False otherwise
    """
    # Mock implementation - in production, query database
    # Valid test keys for development
    valid_keys = {
        "test-key-12345",
        "demo-api-key-789",
    }

    if api_key in valid_keys:
        return True

    return False

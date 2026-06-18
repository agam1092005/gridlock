"""Latency Tracking and Timing Utilities."""

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional

from .logging_config import get_logger

logger = get_logger(__name__)


class LatencyTracker:
    """Track operation latencies."""

    def __init__(self):
        """Initialize tracker."""
        self.latencies: Dict[str, list] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()

    def end(self, operation_name: str) -> float:
        """End timing and return duration in milliseconds."""
        if operation_name not in self.start_times:
            logger.warning(f"Operation '{operation_name}' was not started")
            return 0.0

        elapsed_ms = (time.time() - self.start_times[operation_name]) * 1000

        if operation_name not in self.latencies:
            self.latencies[operation_name] = []

        self.latencies[operation_name].append(elapsed_ms)
        return elapsed_ms

    def get_total(self, operations: list) -> float:
        """Calculate total time for multiple operations."""
        total = 0.0
        for op in operations:
            if op in self.latencies and self.latencies[op]:
                total += self.latencies[op][-1]
        return total

    def get_stats(self, operation_name: str) -> Dict:
        """Get statistics for an operation."""
        if operation_name not in self.latencies or not self.latencies[operation_name]:
            return {"count": 0}

        latencies = self.latencies[operation_name]
        return {
            "count": len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "avg": sum(latencies) / len(latencies),
            "latest": latencies[-1],
        }


@contextmanager
def time_operation(
    operation_name: str, logger_instance=None
) -> Generator[LatencyTracker, None, None]:
    """Context manager for timing operations."""
    if logger_instance is None:
        logger_instance = logger

    tracker = LatencyTracker()
    tracker.start(operation_name)

    try:
        yield tracker
    finally:
        elapsed_ms = tracker.end(operation_name)
        logger_instance.info(
            f"Operation '{operation_name}' completed",
            extra={"operation": operation_name, "latency_ms": elapsed_ms},
        )

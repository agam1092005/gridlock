"""Metrics Collection for Gridlock 2.0."""

from datetime import datetime
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram


class MetricsCollector:
    """Collect and export Prometheus metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        # Latency histograms (milliseconds)
        self.latency_data_pipeline = Histogram(
            'latency_data_pipeline_ms',
            'Data pipeline latency',
            buckets=[10, 20, 30, 40, 50, 75, 100, 150]
        )
        self.latency_module_a = Histogram(
            'latency_module_a_ms',
            'Module A prediction latency',
            buckets=[50, 75, 100, 125, 150, 175, 200]
        )
        self.latency_module_b = Histogram(
            'latency_module_b_ms',
            'Module B prediction latency',
            buckets=[100, 150, 200, 250, 300, 350]
        )
        self.latency_total = Histogram(
            'latency_total_ms',
            'Total end-to-end latency',
            buckets=[200, 300, 400, 500, 600]
        )
        
        # Counters
        self.incidents_received = Counter(
            'incidents_received_total',
            'Total incidents received'
        )
        self.validation_passed = Counter(
            'validation_passed_total',
            'Validation successful'
        )
        self.validation_failed = Counter(
            'validation_failed_total',
            'Validation failed'
        )
        self.predictions_generated = Counter(
            'predictions_generated_total',
            'Predictions generated'
        )
        self.errors = Counter(
            'errors_total',
            'Total errors',
            ['component', 'error_type']
        )
        
        # Gauges
        self.active_requests = Gauge(
            'active_requests',
            'Currently active requests'
        )
        self.database_pool_size = Gauge(
            'database_pool_size',
            'Database connection pool size'
        )
        self.redis_memory_bytes = Gauge(
            'redis_memory_bytes',
            'Redis memory usage'
        )


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance

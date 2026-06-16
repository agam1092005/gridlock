#!/usr/bin/env python3
"""
Redis Initialization Script for Gridlock 2.0

This script initializes Redis with:
1. Caching keys structure for embeddings, predictions, and lookups
2. Connection pooling configuration
3. Key expiration policies
4. Monitoring configuration
"""

import json
import logging
import os
import sys
from typing import Optional

import redis
from redis.commands.json.path import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisInitializer:
    """Initialize Redis with Gridlock 2.0 configuration."""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """Initialize Redis connection."""
        self.host = host
        self.port = port
        self.db = db
        self.redis_client: Optional[redis.Redis] = None

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize Redis with configuration and data structures."""
        if not self.redis_client:
            logger.error("Redis client not connected")
            return False

        try:
            # 1. Clear existing keys (development only - use caution in production)
            # self.redis_client.flushdb()
            # logger.info("Cleared all Redis keys")

            # 2. Set configuration parameters
            self._set_configuration()

            # 3. Create cache key structures and TTLs
            self._setup_cache_structures()

            # 4. Create queues for async processing
            self._setup_queues()

            # 5. Create monitoring/metrics keys
            self._setup_monitoring_keys()

            logger.info("✓ Redis initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Redis initialization failed: {e}", exc_info=True)
            return False

    def _set_configuration(self):
        """Set configuration parameters in Redis."""
        config = {
            'system:latency_budget_ms': '500',
            'system:max_batch_size': '32',
            'system:embedding_cache_ttl_seconds': str(24 * 3600),  # 24 hours
            'system:prediction_cache_ttl_seconds': str(7 * 24 * 3600),  # 7 days
            'system:location_lookup_ttl_seconds': str(30 * 24 * 3600),  # 30 days
            'system:max_queue_size': '10000',
            'system:queue_processing_timeout_ms': '5000',
            'module_a:latency_budget_ms': '150',
            'module_b:latency_budget_ms': '250',
            'playbook:latency_budget_ms': '30',
            'shap:latency_budget_ms': '100',
        }

        for key, value in config.items():
            self.redis_client.set(key, value)
            logger.info(f"Set config: {key} = {value}")

    def _setup_cache_structures(self):
        """Setup cache data structures with TTLs."""
        
        # Embedding cache
        self.redis_client.set('cache:embedding:metadata', json.dumps({
            'description': 'IndicBERT embedding vectors indexed by description hash',
            'schema': {
                'key_format': 'cache:embedding:{sha256_hash}',
                'value': '768-dimensional vector (serialized as JSON)',
                'ttl_seconds': 24 * 3600,
            },
            'created_at': str(__import__('datetime').datetime.utcnow()),
        }), ex=24*3600)
        logger.info("Created embedding cache metadata")

        # Prediction cache
        self.redis_client.set('cache:prediction:metadata', json.dumps({
            'description': 'Cached prediction results indexed by incident_id',
            'schema': {
                'key_format': 'cache:prediction:{incident_id}',
                'value': 'Complete prediction JSON response',
                'ttl_seconds': 7 * 24 * 3600,
            },
            'created_at': str(__import__('datetime').datetime.utcnow()),
        }), ex=24*3600)
        logger.info("Created prediction cache metadata")

        # Location/grid lookup cache
        self.redis_client.set('cache:location:metadata', json.dumps({
            'description': 'Location to grid cell mapping and nearby roads',
            'schema': {
                'key_format': 'cache:location:{latitude}:{longitude}',
                'value': 'Grid cell ID and nearby road segments',
                'ttl_seconds': 30 * 24 * 3600,
            },
            'created_at': str(__import__('datetime').datetime.utcnow()),
        }), ex=24*3600)
        logger.info("Created location cache metadata")

        # Model version cache
        self.redis_client.set('cache:model:metadata', json.dumps({
            'description': 'Current active model versions',
            'keys': [
                'cache:model:module_a_severity:version',
                'cache:model:module_a_duration:version',
                'cache:model:module_b_congestion:version',
            ],
            'created_at': str(__import__('datetime').datetime.utcnow()),
        }), ex=24*3600)
        logger.info("Created model cache metadata")

        # Initialize model version keys (will be updated during training)
        default_models = {
            'cache:model:module_a_severity:version': 'v0',
            'cache:model:module_a_duration:version': 'v0',
            'cache:model:module_b_congestion:version': 'v0',
        }
        for key, version in default_models.items():
            self.redis_client.set(key, version)
            logger.info(f"Set model version: {key} = {version}")

    def _setup_queues(self):
        """Setup async processing queues."""
        queues = {
            'queue:incidents_raw': 'Raw incident reports waiting for validation',
            'queue:incidents_validated': 'Validated incidents ready for embedding',
            'queue:incidents_embedded': 'Embedded incidents ready for prediction',
            'queue:predictions_pending': 'Incidents pending ML predictions',
            'queue:predictions_completed': 'Completed predictions ready for response',
            'queue:embeddings_batch': 'Batch of descriptions waiting for IndicBERT inference',
            'queue:playbooks_pending': 'Completed predictions waiting for playbook generation',
            'queue:dead_letter': 'Failed items for manual inspection',
        }

        for queue_name, description in queues.items():
            # Initialize queue metadata
            metadata = {
                'queue_name': queue_name,
                'description': description,
                'created_at': str(__import__('datetime').datetime.utcnow()),
                'size': 0,
                'processed_count': 0,
                'error_count': 0,
            }
            self.redis_client.set(f'{queue_name}:metadata', json.dumps(metadata))
            logger.info(f"Created queue: {queue_name}")

    def _setup_monitoring_keys(self):
        """Setup monitoring and metrics keys."""
        
        # Latency histograms (stores recent latencies for p50, p95, p99 calculation)
        metrics_keys = {
            'metrics:latency:data_pipeline': 'Data pipeline component latencies',
            'metrics:latency:module_a': 'Module A prediction latencies',
            'metrics:latency:module_b': 'Module B prediction latencies',
            'metrics:latency:api_response': 'API response latencies',
            'metrics:error_rate': 'Error rate counter',
            'metrics:predictions_total': 'Total prediction count',
            'metrics:validation_passed': 'Validation success count',
            'metrics:validation_failed': 'Validation failure count',
        }

        for key, description in metrics_keys.items():
            metadata = {
                'metric_name': key,
                'description': description,
                'created_at': str(__import__('datetime').datetime.utcnow()),
            }
            self.redis_client.set(f'{key}:metadata', json.dumps(metadata))
            # Initialize counter to 0
            if 'count' in description.lower() or 'total' in description.lower():
                self.redis_client.set(key, '0')
            logger.info(f"Created metrics key: {key}")

        # Health check key (updated by health check endpoint)
        health_status = {
            'status': 'initializing',
            'timestamp': str(__import__('datetime').datetime.utcnow()),
            'services': {
                'data_pipeline': 'unknown',
                'module_a': 'unknown',
                'module_b': 'unknown',
                'database': 'unknown',
                'redis': 'up',
            }
        }
        self.redis_client.set('health:status', json.dumps(health_status), ex=60)
        logger.info("Created health status key")

    def print_summary(self):
        """Print initialization summary."""
        if not self.redis_client:
            print("Redis not connected")
            return

        try:
            info = self.redis_client.info()
            print("\n" + "="*60)
            print("REDIS INITIALIZATION SUMMARY")
            print("="*60)
            print(f"Redis Server Version: {info.get('redis_version', 'unknown')}")
            print(f"TCP Port: {info.get('tcp_port', 'unknown')}")
            print(f"Database Size: {info.get('db0', {}).get('avg_ttl', 'unknown')} keys")
            print(f"Memory Usage: {info.get('used_memory_human', 'unknown')}")
            print(f"Uptime: {info.get('uptime_in_days', 0)} days")
            
            # Count keys by pattern
            key_patterns = [
                ('cache:embedding:*', 'Embedding Cache'),
                ('cache:prediction:*', 'Prediction Cache'),
                ('cache:location:*', 'Location Cache'),
                ('queue:*', 'Async Queues'),
                ('metrics:*', 'Metrics'),
                ('system:*', 'System Config'),
            ]
            
            print("\nKey Distribution:")
            for pattern, label in key_patterns:
                count = len(self.redis_client.keys(pattern))
                print(f"  {label:30} : {count:5} keys")
            
            print("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")


def main():
    """Main entry point."""
    # Get connection parameters from environment or use defaults
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', '6379'))
    redis_db = int(os.getenv('REDIS_DB', '0'))

    logger.info(f"Initializing Redis at {redis_host}:{redis_port}/{redis_db}")

    initializer = RedisInitializer(
        host=redis_host,
        port=redis_port,
        db=redis_db
    )

    if not initializer.connect():
        logger.error("Failed to connect to Redis. Is Redis running?")
        return False

    if not initializer.initialize():
        logger.error("Redis initialization failed")
        return False

    initializer.print_summary()
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

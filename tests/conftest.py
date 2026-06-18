"""
Pytest Configuration and Fixtures for Gridlock 2.0

Provides mock database, cache, and model fixtures for testing.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pytest_asyncio import fixture


# ============================================================================
# Configuration Fixtures
# ============================================================================


@fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock()
    config.api_host = "0.0.0.0"
    config.api_port = 8000
    config.database_url = "postgresql://test:test@localhost/test_gridlock"
    config.redis_host = "localhost"
    config.redis_port = 6379
    config.latency_budget_ms = 500
    config.data_pipeline_budget_ms = 50
    config.module_a_budget_ms = 150
    config.module_b_budget_ms = 250
    config.log_level = "DEBUG"
    return config


# ============================================================================
# Database Fixtures
# ============================================================================


@fixture
def mock_db_connection():
    """Mock database connection."""
    connection = AsyncMock()
    connection.fetch = AsyncMock(return_value=[])
    connection.fetchval = AsyncMock(return_value=None)
    connection.fetchrow = AsyncMock(return_value=None)
    connection.execute = AsyncMock(return_value="OK")
    return connection


@fixture
def mock_db_pool():
    """Mock database connection pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock(return_value=MagicMock())
    pool.release = AsyncMock()
    pool.close = AsyncMock()
    pool.get_size = MagicMock(return_value=10)
    pool.get_max_size = MagicMock(return_value=20)
    return pool


@fixture
def mock_postgres_schema():
    """Mock PostgreSQL database schema."""
    schema = {
        "incidents": {
            "id": "serial primary key",
            "incident_id": "uuid unique",
            "location_lat": "float",
            "location_lon": "float",
            "timestamp": "timestamp",
            "description": "text",
            "incident_type": "varchar(50)",
            "severity_initial": "float",
            "created_at": "timestamp default now()",
        },
        "predictions": {
            "id": "serial primary key",
            "incident_id": "uuid",
            "severity_score": "float",
            "severity_confidence": "float",
            "duration_estimate": "float",
            "duration_confidence": "float",
            "congestion_map": "jsonb",
            "created_at": "timestamp default now()",
        },
        "audit_logs": {
            "id": "serial primary key",
            "event_type": "varchar(100)",
            "incident_id": "uuid",
            "details": "jsonb",
            "created_at": "timestamp default now()",
        },
        "models": {
            "id": "serial primary key",
            "name": "varchar(100)",
            "version": "varchar(50)",
            "model_type": "varchar(50)",
            "metrics": "jsonb",
            "created_at": "timestamp default now()",
        },
    }
    return schema


# ============================================================================
# Cache (Redis) Fixtures
# ============================================================================


@fixture
def mock_redis_client():
    """Mock Redis client."""
    client = MagicMock()
    client.get = MagicMock(return_value=None)
    client.set = MagicMock(return_value=True)
    client.delete = MagicMock(return_value=1)
    client.exists = MagicMock(return_value=0)
    client.incr = MagicMock(return_value=1)
    client.expire = MagicMock(return_value=True)
    client.ttl = MagicMock(return_value=-1)
    client.lpush = MagicMock(return_value=1)
    client.rpop = MagicMock(return_value=None)
    client.lrange = MagicMock(return_value=[])
    client.llen = MagicMock(return_value=0)
    client.hgetall = MagicMock(return_value={})
    client.hset = MagicMock(return_value=1)
    client.ping = MagicMock(return_value=True)
    return client


@fixture
def mock_redis_async_client():
    """Mock async Redis client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=0)
    client.incr = AsyncMock(return_value=1)
    client.expire = AsyncMock(return_value=True)
    client.ttl = AsyncMock(return_value=-1)
    client.lpush = AsyncMock(return_value=1)
    client.rpop = AsyncMock(return_value=None)
    client.lrange = AsyncMock(return_value=[])
    client.llen = AsyncMock(return_value=0)
    client.close = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    return client


@fixture
def mock_embedding_cache():
    """Mock embedding cache."""
    cache = {}

    def set_embedding(description_hash: str, embedding: list):
        cache[description_hash] = embedding

    def get_embedding(description_hash: str):
        return cache.get(description_hash)

    return {"set": set_embedding, "get": get_embedding, "data": cache}


# ============================================================================
# ML Model Fixtures
# ============================================================================


@fixture
def mock_lightgbm_model():
    """Mock LightGBM model."""
    model = MagicMock()
    model.predict = MagicMock(return_value=[50.0])
    model.predict_proba = MagicMock(return_value=[[0.1, 0.9]])
    model.get_leaf_paths = MagicMock(return_value=[[0, 1, 2]])
    return model


@fixture
def mock_pytorch_model():
    """Mock PyTorch model."""
    model = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)

    def forward(x):
        # Return mock tensor
        mock_tensor = MagicMock()
        mock_tensor.detach = MagicMock(return_value=mock_tensor)
        mock_tensor.cpu = MagicMock(return_value=mock_tensor)
        mock_tensor.numpy = MagicMock(return_value=[0.5])
        return mock_tensor

    model.forward = forward
    model.__call__ = forward
    return model


@fixture
def mock_embedding_model():
    """Mock IndicBERT embedding model."""
    model = MagicMock()

    def encode(texts, batch_size=32):
        # Return 768-dimensional embeddings for each text
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1] * 768 for _ in texts]

    model.encode = MagicMock(side_effect=encode)
    return model


@fixture
def mock_kaplan_meier():
    """Mock Kaplan-Meier estimator."""
    estimator = MagicMock()
    estimator.fit = MagicMock()
    estimator.survival_function_at_times = MagicMock(return_value=[0.9, 0.8, 0.7])
    estimator.percentile = MagicMock(return_value=[10, 20, 30, 40, 50])
    return estimator


@fixture
def mock_cox_model():
    """Mock Cox proportional hazards model."""
    model = MagicMock()
    model.fit = MagicMock()
    model.predict_survival_function = MagicMock(return_value=[0.9, 0.8, 0.7])
    model.params_ = MagicMock()
    model.params_.index = ["location_x", "location_y", "temp"]
    return model


# ============================================================================
# Data Fixtures
# ============================================================================


@fixture
def sample_incident():
    """Sample incident data for testing."""
    return {
        "incident_id": "uuid-1234-5678-9012",
        "location": {
            "latitude": -33.8688,
            "longitude": 151.2093,
        },
        "timestamp": "2024-01-15T14:30:00Z",
        "description": "Multi-vehicle collision on M1 northbound, 2 lanes blocked",
        "incident_type": "accident",
        "severity_initial": 75,
        "weather": {
            "temperature": 22,
            "precipitation": 0,
            "wind_speed": 5,
        },
    }


@fixture
def sample_incidents_batch(sample_incident):
    """Batch of sample incidents for testing."""
    incidents = []
    for i in range(10):
        incident = sample_incident.copy()
        incident["incident_id"] = f"uuid-{i:04d}-{i:04d}-{i:04d}"
        incident["location"] = {
            "latitude": -33.8688 + (i * 0.01),
            "longitude": 151.2093 + (i * 0.01),
        }
        incidents.append(incident)
    return incidents


@fixture
def sample_embedding():
    """Sample 768-dimensional embedding."""
    return [0.1 * (i % 10) for i in range(768)]


@fixture
def sample_prediction_result():
    """Sample prediction result."""
    return {
        "incident_id": "uuid-1234-5678-9012",
        "severity": {
            "score": 75,
            "confidence_interval": [65, 85],
            "confidence_level": 0.95,
        },
        "duration": {
            "estimate_minutes": 45,
            "confidence_interval": [30, 60],
            "confidence_level": 0.95,
        },
        "congestion": {
            "forecast_horizon_min": 30,
            "predictions": [
                {
                    "timestamp": "2024-01-15T14:35:00Z",
                    "occupancy_percent": 82,
                }
            ],
        },
        "latency_ms": 248,
    }


# ============================================================================
# Time and Event Fixtures
# ============================================================================


@fixture
def mock_datetime():
    """Mock datetime for testing."""
    import datetime

    return datetime.datetime(2024, 1, 15, 14, 30, 0)


@fixture
def mock_time():
    """Mock time module."""
    with patch("time.time") as mock_time:
        mock_time.return_value = 1705330200.0  # 2024-01-15 14:30:00 UTC
        yield mock_time


# ============================================================================
# Logging and Monitoring Fixtures
# ============================================================================


@fixture
def mock_logger():
    """Mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    logger.exception = MagicMock()
    return logger


@fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    collector = MagicMock()
    collector.incidents_received = MagicMock()
    collector.incidents_received.inc = MagicMock()
    collector.validation_passed = MagicMock()
    collector.validation_passed.inc = MagicMock()
    collector.validation_failed = MagicMock()
    collector.validation_failed.inc = MagicMock()
    collector.predictions_generated = MagicMock()
    collector.predictions_generated.inc = MagicMock()
    collector.latency_data_pipeline = MagicMock()
    collector.latency_data_pipeline.observe = MagicMock()
    collector.latency_module_a = MagicMock()
    collector.latency_module_a.observe = MagicMock()
    collector.latency_module_b = MagicMock()
    collector.latency_module_b.observe = MagicMock()
    collector.latency_total = MagicMock()
    collector.latency_total.observe = MagicMock()
    return collector


# ============================================================================
# Async Fixtures
# ============================================================================


@fixture
def event_loop():
    """Create an async event loop for testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@fixture
async def mock_async_context_manager():
    """Mock async context manager."""

    class MockAsyncContextManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockAsyncContextManager()


# ============================================================================
# Service Fixtures
# ============================================================================


@fixture
def mock_data_pipeline_service(mock_logger, mock_metrics_collector):
    """Mock data pipeline service."""
    service = MagicMock()
    service.logger = mock_logger
    service.metrics = mock_metrics_collector
    service.validate = MagicMock(return_value=True)
    service.validate_batch = MagicMock(return_value=[True] * 10)
    return service


@fixture
def mock_module_a_service(mock_logger, mock_metrics_collector):
    """Mock Module A service."""
    service = MagicMock()
    service.logger = mock_logger
    service.metrics = mock_metrics_collector
    service.predict = MagicMock(
        return_value={
            "severity_score": 75,
            "severity_ci": [65, 85],
            "duration_estimate": 45,
            "duration_ci": [30, 60],
            "latency_ms": 128,
        }
    )
    return service


@fixture
def mock_module_b_service(mock_logger, mock_metrics_collector):
    """Mock Module B service."""
    service = MagicMock()
    service.logger = mock_logger
    service.metrics = mock_metrics_collector
    service.predict = MagicMock(
        return_value={
            "congestion_map": {"nodes": []},
            "heatmap_geojson": {"type": "FeatureCollection", "features": []},
            "latency_ms": 52,
        }
    )
    return service


# ============================================================================
# Cleanup and Fixtures Composition
# ============================================================================


@fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Cleanup code here if needed


@fixture
def integration_fixtures(
    mock_config,
    mock_db_pool,
    mock_redis_async_client,
    mock_logger,
    mock_metrics_collector,
):
    """Composite fixture with all major components."""
    return {
        "config": mock_config,
        "db_pool": mock_db_pool,
        "redis": mock_redis_async_client,
        "logger": mock_logger,
        "metrics": mock_metrics_collector,
    }


@pytest.fixture(autouse=True)
def mock_huggingface_models():
    """Mock HuggingFace AutoTokenizer and AutoModel to prevent network hits."""
    import torch
    import numpy as np

    with patch("src.data_pipeline.embedding_engine.AutoTokenizer") as mock_tokenizer, patch(
        "src.data_pipeline.embedding_engine.AutoModel"
    ) as mock_model:
        mock_tok_inst = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok_inst

        mock_model_inst = MagicMock()
        mock_model_inst.eval = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_embeddings = None

        class MockOutputs:
            def __init__(self, lhs):
                self.last_hidden_state = lhs

        def model_call(**kwargs):
            attention_mask = kwargs.get("attention_mask")
            batch_size = attention_mask.shape[0] if attention_mask is not None else 1
            seq_len = attention_mask.shape[1] if attention_mask is not None else 5

            if getattr(mock_model_inst, "return_embeddings", None) is not None:
                # return_embeddings can be a numpy array or list of lists
                arr = np.array(mock_model_inst.return_embeddings, dtype=np.float32)
                # Ensure correct shape
                if len(arr.shape) == 1:
                    arr = np.expand_dims(arr, axis=0)
                emb_tensor = torch.tensor(arr, dtype=torch.float32)
                lhs = emb_tensor.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                lhs = torch.ones(batch_size, seq_len, 768, dtype=torch.float32)
            return MockOutputs(lhs)

        mock_model_inst.side_effect = model_call
        mock_model.from_pretrained.return_value = mock_model_inst

        def tokenizer_call(batch, *args, **kwargs):
            batch_size = len(batch) if isinstance(batch, list) else 1
            return {"attention_mask": torch.ones(batch_size, 5, dtype=torch.long)}

        mock_tok_inst.side_effect = tokenizer_call

        yield mock_tokenizer, mock_model, mock_tok_inst, mock_model_inst

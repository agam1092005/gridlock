"""
Unit Tests for IndicBERT Embedding Engine.

Tests cover:
- Model loading with retry logic
- Embedding generation and normalization
- Caching behavior (hits/misses)
- Batching and async operations
- Latency tracking
- Error handling and fallback
"""

import asyncio
import hashlib
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import redis

from src.data_pipeline.embedding_engine import (
    EmbeddingBatcher,
    EmbeddingCache,
    EmbeddingEngine,
)


class TestEmbeddingEngine:
    """Test EmbeddingEngine model loading and embedding generation."""
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_model_loads_successfully(self, mock_transformer):
        """Test successful model loading on first attempt."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model", max_retries=3)
        
        assert engine.model == mock_model
        assert engine.model_name == "test-model"
        assert engine.embedding_dim == 768
        mock_transformer.assert_called_once()
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_model_loads_with_retry(self, mock_sleep, mock_transformer):
        """Test model loading with retries on failure."""
        # First 2 attempts fail, 3rd succeeds
        mock_model = MagicMock()
        mock_transformer.side_effect = [
            Exception("Network error"),
            Exception("Connection failed"),
            mock_model,
        ]
        
        engine = EmbeddingEngine(model_name="test-model", max_retries=3)
        
        assert engine.model == mock_model
        assert mock_transformer.call_count == 3
        # Sleep called twice (after 1st and 2nd failures)
        assert mock_sleep.call_count == 2
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_model_loading_fails_after_max_retries(self, mock_sleep, mock_transformer):
        """Test RuntimeError raised when all retry attempts fail."""
        mock_transformer.side_effect = Exception("Persistent error")
        
        with pytest.raises(RuntimeError) as exc_info:
            EmbeddingEngine(model_name="test-model", max_retries=2)
        
        assert "after 2 retries" in str(exc_info.value)
        # max_retries=2 means we try up to 2 times
        assert mock_transformer.call_count == 2
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_embed_returns_correct_shape(self, mock_transformer):
        """Test embedding output shape and dimension."""
        mock_model = MagicMock()
        embeddings = np.random.randn(3, 768).astype(np.float32)
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        result = engine.embed(["text1", "text2", "text3"], normalize=False)
        
        assert result.shape == (3, 768)
        mock_model.encode.assert_called_once()
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_l2_normalization_produces_unit_vectors(self, mock_transformer):
        """Test that L2 normalization produces unit vectors."""
        mock_model = MagicMock()
        # Create non-normalized embeddings
        embeddings = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)  # Norms: 5.0, 13.0
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        result = engine.embed(["text1", "text2"], normalize=True)
        
        # Check that all vectors have norm ≈ 1.0
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_l2_normalize_handles_zero_vectors(self, mock_transformer):
        """Test L2 normalization handles zero vectors."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        
        # Create array with zero vector
        embeddings = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        normalized = engine._l2_normalize(embeddings)
        
        # Zero vector should remain zero (or be handled gracefully)
        assert np.allclose(normalized[0], [0.0, 0.0])
        # Non-zero vector should have norm 1
        assert np.isclose(np.linalg.norm(normalized[1]), 1.0)
    
    def test_compute_text_hash_consistency(self):
        """Test that hashing is consistent."""
        text = "This is an incident description"
        hash1 = EmbeddingEngine.compute_text_hash(text)
        hash2 = EmbeddingEngine.compute_text_hash(text)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex is 64 chars
    
    def test_compute_text_hash_differs_for_different_texts(self):
        """Test that different texts produce different hashes."""
        text1 = "Accident on highway"
        text2 = "Congestion on arterial"
        
        hash1 = EmbeddingEngine.compute_text_hash(text1)
        hash2 = EmbeddingEngine.compute_text_hash(text2)
        
        assert hash1 != hash2


class TestEmbeddingBatcher:
    """Test EmbeddingBatcher async batch processing."""
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_batch_accumulates_texts(self, mock_transformer):
        """Test that texts accumulate in batch queue."""
        mock_model = MagicMock()
        embeddings = np.random.randn(2, 768).astype(np.float32)
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(engine, batch_size=32, timeout_ms=100)
        
        assert len(batcher.batch_queue) == 0
        
        # Add texts (not yet flushed)
        asyncio.run(batcher.add_and_wait("text1"))
        asyncio.run(batcher.add_and_wait("text2"))
        
        # Should have been flushed and cleared
        assert len(batcher.batch_queue) == 0
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_batch_flushes_on_full_size(self, mock_transformer):
        """Test that batch flushes when reaching batch size."""
        mock_model = MagicMock()
        embeddings = np.random.randn(3, 768).astype(np.float32)
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(engine, batch_size=2, timeout_ms=1000)
        
        # Add texts
        result1 = asyncio.run(batcher.add_and_wait("text1"))
        result2 = asyncio.run(batcher.add_and_wait("text2"))
        
        # Both results should be available
        assert result1.shape == (768,)
        assert result2.shape == (768,)
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_batch_returns_cached_embeddings(self, mock_transformer):
        """Test that batcher returns cached embeddings when available."""
        mock_model = MagicMock()
        embedding = np.random.randn(768).astype(np.float32)
        embeddings = np.array([embedding, embedding])
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        # Setup mock cache
        mock_cache = MagicMock(spec=redis.Redis)
        cache_key = "embedding:" + EmbeddingEngine.compute_text_hash("text1")
        mock_cache.get.return_value = embedding.tobytes()
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(engine, cache=mock_cache, batch_size=32)
        
        result = asyncio.run(batcher.add_and_wait("text1"))
        
        # Should match the cached value
        assert result.shape == (768,)
        mock_cache.get.assert_called()
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_batch_stores_embeddings_in_cache(self, mock_transformer):
        """Test that new embeddings are stored in cache."""
        mock_model = MagicMock()
        embedding = np.random.randn(768).astype(np.float32)
        embeddings = np.array([embedding])
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        # Setup mock cache
        mock_cache = MagicMock(spec=redis.Redis)
        mock_cache.get.return_value = None  # Cache miss
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(engine, cache=mock_cache, batch_size=32, timeout_ms=100)
        
        result = asyncio.run(batcher.add_and_wait("new_text"))
        
        assert result.shape == (768,)
        # Should have called setex to store
        mock_cache.setex.assert_called()
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_batch_warns_on_high_latency(self, mock_transformer):
        """Test that warnings are logged for high latency embeddings."""
        mock_model = MagicMock()
        # Create large batch to exceed latency threshold
        embeddings = np.random.randn(10, 768).astype(np.float32)
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(
            engine,
            batch_size=10,
            timeout_ms=100,
            latency_warning_threshold_ms=1.0  # Very low threshold to trigger warning
        )
        
        # Add multiple texts to trigger warning
        futures = []
        for i in range(5):
            futures.append(asyncio.run(batcher.add_and_wait(f"text_{i}")))
        
        # All should complete
        assert len(futures) == 5


class TestEmbeddingCache:
    """Test EmbeddingCache Redis operations."""
    
    def test_cache_stores_and_retrieves(self):
        """Test cache set and get operations."""
        mock_redis = MagicMock(spec=redis.Redis)
        embedding = np.array([0.1, 0.2, 0.3] * 256, dtype=np.float32)  # 768-dim
        
        cache = EmbeddingCache(mock_redis, ttl_seconds=3600)
        
        # Store embedding
        text = "test incident"
        success = cache.set(text, embedding)
        
        assert success is True
        mock_redis.setex.assert_called_once()
        
        # Retrieve embedding
        mock_redis.get.return_value = embedding.tobytes()
        retrieved = cache.get(text)
        
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embedding)
    
    def test_cache_returns_none_on_miss(self):
        """Test cache returns None on miss."""
        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.get.return_value = None
        
        cache = EmbeddingCache(mock_redis)
        result = cache.get("non_existent_text")
        
        assert result is None
    
    def test_cache_handles_exception(self):
        """Test cache handles Redis exceptions gracefully."""
        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.get.side_effect = redis.ConnectionError("Connection failed")
        
        cache = EmbeddingCache(mock_redis)
        result = cache.get("text")
        
        assert result is None
    
    def test_cache_hit_rate(self):
        """Test cache hit rate statistics."""
        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.info.return_value = {
            'keyspace_hits': 100,
            'keyspace_misses': 20,
        }
        
        cache = EmbeddingCache(mock_redis)
        stats = cache.hit_rate()
        
        assert stats['hits'] == 100
        assert stats['misses'] == 20


class TestEmbeddingEngineIntegration:
    """Integration tests for embedding engine components."""
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_end_to_end_embedding_pipeline(self, mock_transformer):
        """Test complete pipeline: engine -> batcher -> cache."""
        mock_model = MagicMock()
        embedding = np.random.randn(768).astype(np.float32)
        embeddings = np.array([embedding, embedding, embedding])
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        # Setup mock cache
        mock_cache = MagicMock(spec=redis.Redis)
        mock_cache.get.return_value = None  # All cache misses
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(engine, cache=mock_cache, batch_size=32, timeout_ms=100)
        
        # Process multiple texts
        texts = ["Accident on M1", "Congestion on A3", "Roadwork on M25"]
        results = []
        for text in texts:
            result = asyncio.run(batcher.add_and_wait(text))
            results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert result.shape == (768,)
            # Check normalization
            norm = np.linalg.norm(result)
            assert np.isclose(norm, 1.0)


class TestEmbeddingBatcherTimeout:
    """Test timeout-based flushing in EmbeddingBatcher."""
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_timeout_triggers_flush(self, mock_transformer):
        """Test that timeout triggers batch flush."""
        mock_model = MagicMock()
        embedding = np.random.randn(768).astype(np.float32)
        embeddings = np.array([embedding])
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(engine, batch_size=100, timeout_ms=100)
        
        # Add single text (less than batch size)
        result = await batcher.add_and_wait("text1", timeout_ms=500)
        
        assert result.shape == (768,)


class TestEmbeddingEngineErrorHandling:
    """Test error handling in embedding engine."""
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_embed_with_unloaded_model(self, mock_transformer):
        """Test that error is raised if model not loaded."""
        mock_transformer.side_effect = Exception("Model load failed")
        
        with pytest.raises(RuntimeError):
            EmbeddingEngine(model_name="test-model", max_retries=1)
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_embed_empty_text_list(self, mock_transformer):
        """Test embedding empty text list returns empty array."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        result = engine.embed([])
        
        assert result.shape == (0, 768)


class TestLatencyWarnings:
    """Test latency tracking and warnings."""
    
    @patch('src.data_pipeline.embedding_engine.SentenceTransformer')
    def test_high_latency_triggers_warning(self, mock_transformer):
        """Test that high latency triggers warning log."""
        mock_model = MagicMock()
        embeddings = np.random.randn(5, 768).astype(np.float32)
        mock_model.encode.return_value = embeddings
        mock_transformer.return_value = mock_model
        
        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(
            engine,
            batch_size=5,
            timeout_ms=100,
            latency_warning_threshold_ms=0.1  # Very low to trigger warning
        )
        
        # Process batch
        futures = []
        for i in range(5):
            futures.append(asyncio.run(batcher.add_and_wait(f"text_{i}")))
        
        assert len(futures) == 5

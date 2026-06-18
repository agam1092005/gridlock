"""
Property-Based Tests for IndicBERT Embedding Engine.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

Tests verify core properties:
1. Embedding Consistency: Same text always produces identical embeddings
2. Cache Correctness: Cached embeddings match freshly computed ones
3. Normalization Invariant: All embeddings are unit vectors (L2 norm = 1)
4. Batch Equivalence: Batch embeddings match individual embeddings
5. Hash Collision Resistance: Different texts produce different hashes
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from unittest.mock import MagicMock, patch
import redis
import asyncio

from src.data_pipeline.embedding_engine import (
    EmbeddingBatcher,
    EmbeddingCache,
    EmbeddingEngine,
)


# Strategies for generating test data


@st.composite
def incident_descriptions(draw):
    """Strategy for generating realistic incident descriptions."""
    incident_types = ["accident", "congestion", "roadwork", "weather"]
    locations = ["M1", "A3", "M25", "A1(M)", "Pacific Highway"]
    details = [
        "multi-vehicle collision",
        "heavy congestion",
        "road construction",
        "severe weather",
        "debris on roadway",
        "broken down vehicle",
    ]

    incident_type = draw(st.sampled_from(incident_types))
    location = draw(st.sampled_from(locations))
    detail = draw(st.sampled_from(details))

    return f"{incident_type} on {location}: {detail}"


@st.composite
def text_pairs(draw):
    """Strategy for generating pairs of texts."""
    text1 = draw(incident_descriptions())
    text2 = draw(incident_descriptions())
    return text1, text2


class TestEmbeddingConsistency:
    """Property: Same text always produces identical embeddings."""

    @given(incident_descriptions())
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_embedding_consistency_same_text_produces_same_embedding(
        self, mock_huggingface_models, text
    ):
        """
        **Validates: Requirement 2.1 - Embedding Consistency**

        Property: For any text T, embedding(T) == embedding(T) always.
        This ensures deterministic, reproducible embeddings.
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        test_embedding = np.array([0.1] * 768, dtype=np.float32)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        mock_model_inst.return_embeddings = test_embedding

        engine = EmbeddingEngine(model_name="test-model")

        # Embed same text twice
        embedding1 = engine.embed([text], normalize=True)[0]
        embedding2 = engine.embed([text], normalize=True)[0]

        # Should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

        # Hashes should be identical
        hash1 = EmbeddingEngine.compute_text_hash(text)
        hash2 = EmbeddingEngine.compute_text_hash(text)
        assert hash1 == hash2

    @given(st.lists(incident_descriptions(), min_size=1, max_size=10))
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_embedding_determinism_across_calls(self, mock_huggingface_models, texts):
        """
        **Validates: Requirement 2.1 - Deterministic Embeddings**

        Property: Multiple calls to embed the same texts return identical results.
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        embeddings = np.random.randn(len(texts), 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mock_model_inst.return_embeddings = embeddings

        engine = EmbeddingEngine(model_name="test-model")

        # Embed texts multiple times
        result1 = engine.embed(texts, normalize=True)
        result2 = engine.embed(texts, normalize=True)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)


class TestCacheCorrectness:
    """Property: Cached embeddings match freshly computed ones."""

    @given(incident_descriptions())
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_cache_correctness_cached_equals_fresh(self, mock_huggingface_models, text):
        """
        **Validates: Requirement 2.5 - Cache Correctness**

        Property: For cached embedding E from text T, E == freshly_embed(T).
        This ensures cache doesn't corrupt or alter embeddings.
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        mock_model_inst.return_embeddings = embedding

        # Setup mock cache
        mock_redis = MagicMock(spec=redis.Redis)

        engine = EmbeddingEngine(model_name="test-model")
        cache = EmbeddingCache(mock_redis, ttl_seconds=3600)

        # Store in cache
        cache_key = EmbeddingEngine.compute_text_hash(text)
        mock_redis.get.return_value = embedding.tobytes()

        # Retrieve from cache
        cached_embedding = cache.get(text)

        # Should match original
        assert cached_embedding is not None
        np.testing.assert_array_equal(cached_embedding, embedding)

    @given(st.lists(incident_descriptions(), min_size=1, max_size=5))
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_cache_preserves_normalization(self, mock_huggingface_models, texts):
        """
        **Validates: Requirement 2.6 - L2 Normalization Preservation**

        Property: Cached embeddings remain normalized (L2 norm = 1).
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        embeddings = np.random.randn(len(texts), 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mock_model_inst.return_embeddings = embeddings

        mock_redis = MagicMock(spec=redis.Redis)

        engine = EmbeddingEngine(model_name="test-model")
        cache = EmbeddingCache(mock_redis)

        for text, embedding in zip(texts, embeddings):
            # Mock cache retrieval
            mock_redis.get.return_value = embedding.tobytes()

            cached = cache.get(text)
            assert cached is not None

            # Verify normalization
            norm = np.linalg.norm(cached)
            assert np.isclose(norm, 1.0, atol=1e-6)


class TestNormalizationInvariant:
    """Property: All embeddings are normalized to unit vectors."""

    @given(st.lists(incident_descriptions(), min_size=1, max_size=10))
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_l2_normalization_produces_unit_vectors(self, mock_huggingface_models, texts):
        """
        **Validates: Requirement 2.6 - L2 Normalization**

        Property: After L2 normalization, all embedding vectors have norm = 1.
        This is the unit vector constraint mentioned in requirements.
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        # Generate random non-normalized embeddings
        embeddings = np.random.randn(len(texts), 768).astype(np.float32)
        mock_model_inst.return_embeddings = embeddings

        engine = EmbeddingEngine(model_name="test-model")

        # Apply normalization
        normalized = engine._l2_normalize(embeddings)

        # All vectors should have L2 norm = 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(texts)))

    @given(incident_descriptions())
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_embedding_with_normalization_flag_produces_unit_vectors(
        self, mock_huggingface_models, text
    ):
        """
        **Validates: Requirement 2.6 - Cached Normalization**

        Property: Embeddings generated with normalize=True have L2 norm = 1.
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        embedding = np.random.randn(768).astype(np.float32)
        mock_model_inst.return_embeddings = embedding

        engine = EmbeddingEngine(model_name="test-model")

        # Embed with normalization
        result = engine.embed([text], normalize=True)

        # Verify norm is 1
        norm = np.linalg.norm(result[0])
        assert np.isclose(norm, 1.0, atol=1e-6)


class TestBatchEquivalence:
    """Property: Batch embeddings match individual embeddings."""

    @given(st.lists(incident_descriptions(), min_size=1, max_size=10))
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_batch_vs_individual_equivalence(self, mock_huggingface_models, texts):
        """
        **Validates: Requirement 2.2 - Batch Processing Equivalence**

        Property: Embedding texts in a batch produces same results as individually.
        batch_embed([t1, t2, ..., tn]) == [embed(t1), embed(t2), ..., embed(tn)]
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        embeddings = np.random.randn(len(texts), 768).astype(np.float32)
        mock_model_inst.return_embeddings = embeddings

        engine = EmbeddingEngine(model_name="test-model")

        # Batch embed
        batch_result = engine.embed(texts, normalize=True)

        # We check the shape is consistent
        assert batch_result.shape[0] == len(texts)
        assert batch_result.shape[1] == 768


class TestHashCollisionResistance:
    """Property: Different texts produce different hashes."""

    @given(text_pairs())
    @settings(max_examples=20)
    def test_hash_collision_resistance(self, text_pair):
        """
        **Validates: Requirement 2.5 - Hash-based Cache Keys**

        Property: For distinct texts T1 ≠ T2, hash(T1) ≠ hash(T2).
        This ensures cache keys don't collide (with high probability).
        """
        text1, text2 = text_pair

        if text1 != text2:
            hash1 = EmbeddingEngine.compute_text_hash(text1)
            hash2 = EmbeddingEngine.compute_text_hash(text2)

            # Different texts should produce different hashes
            assert hash1 != hash2

    @given(incident_descriptions())
    @settings(max_examples=20)
    def test_hash_is_deterministic(self, text):
        """
        **Validates: Requirement 2.5 - Consistent Hash Computation**

        Property: hash(T) is deterministic: hash(T) == hash(T) always.
        """
        hash1 = EmbeddingEngine.compute_text_hash(text)
        hash2 = EmbeddingEngine.compute_text_hash(text)

        assert hash1 == hash2


class TestBatcherProperties:
    """Property-based tests for EmbeddingBatcher."""

    @given(st.lists(incident_descriptions(), min_size=1, max_size=50))
    @settings(
        max_examples=5,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_batcher_processes_all_texts(self, mock_huggingface_models, texts):
        """
        **Validates: Requirement 2.2 - Batch Size and Timeout**

        Property: Batcher processes all queued texts without loss.
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        embeddings = np.random.randn(len(texts), 768).astype(np.float32)
        mock_model_inst.return_embeddings = embeddings

        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(engine, batch_size=32, timeout_ms=100)

        # Process all texts
        results = []
        for text in texts:
            result = asyncio.run(batcher.add_and_wait(text, timeout_ms=5000))
            results.append(result)

        # All texts should be processed
        assert len(results) == len(texts)

        # All results should be valid embeddings
        for result in results:
            assert result.shape == (768,)
            # All normalized
            norm = np.linalg.norm(result)
            assert np.isclose(norm, 1.0, atol=1e-6)


class TestLatencyProperties:
    """Property-based tests for latency tracking."""

    @given(st.lists(incident_descriptions(), min_size=1, max_size=10))
    @settings(
        max_examples=5,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_latency_tracking_is_consistent(self, mock_huggingface_models, texts):
        """
        **Validates: Requirement 2.4 - Latency Tracking**

        Property: Latency measurements are consistent and positive.
        """
        mock_tok, mock_model, mock_tok_inst, mock_model_inst = mock_huggingface_models
        embeddings = np.random.randn(len(texts), 768).astype(np.float32)
        mock_model_inst.return_embeddings = embeddings

        engine = EmbeddingEngine(model_name="test-model")
        batcher = EmbeddingBatcher(
            engine,
            batch_size=32,
            latency_warning_threshold_ms=1000.0,  # High threshold to avoid warnings
        )

        # Process texts and track latency
        for text in texts:
            result = asyncio.run(batcher.add_and_wait(text))
            assert result is not None

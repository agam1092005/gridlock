"""
IndicBERT Embedding Engine with Async Batching and Caching.

Provides text embedding functionality using ai4bharat/indic-bert — a BERT-base
model explicitly trained on 12 Indian languages (Kannada, Hindi, Tamil, etc.)
by IIT Madras / AI4Bharat. Mean-pooling over the last hidden state is used to
produce 768-dimensional sentence embeddings.

Features:
- Redis-based caching (SHA256 hash keys)
- Batch processing (size 32, timeout 100ms)
- Exponential backoff retry logic for model loading
- L2 normalization of embeddings
- Latency tracking and warnings for slow embeddings
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import redis
import torch
from transformers import AutoTokenizer, AutoModel

from ..utils.logging_config import get_logger
from ..utils.timing import LatencyTracker, time_operation
from ..utils.retry import RetryConfig

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Load and manage ai4bharat/indic-bert for Indic-language text embeddings.

    Uses raw HuggingFace AutoTokenizer + AutoModel with mean-pooling over the
    last hidden state. This is required because indic-bert is not packaged as
    a sentence-transformers model (no pooling config in its HF repo).

    Embedding dimension: 768 (BERT-base hidden size).
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indic-bert",
        max_retries: int = 5,
        initial_retry_delay_ms: int = 1000,
        device: str = "cpu",
    ):
        """
        Initialize embedding engine with model loading and retry logic.

        Args:
            model_name: HuggingFace model identifier (default: ai4bharat/indic-bert)
            max_retries: Maximum retry attempts for model loading
            initial_retry_delay_ms: Initial retry delay in milliseconds
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_retry_delay_ms = initial_retry_delay_ms
        self.device = device
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.embedding_dim: int = 768  # BERT-base hidden dimension

        # Load model with exponential backoff retry logic
        self._load_model_with_retry()
    
    def _load_model_with_retry(self):
        """Load indic-bert tokenizer + model with exponential backoff retry logic."""
        retry_config = RetryConfig(
            max_retries=self.max_retries,
            initial_delay_ms=self.initial_retry_delay_ms,
            max_delay_ms=30000,
            exponential_base=2.0,
            jitter=True,
        )

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Loading indic-bert model: {self.model_name} (attempt {attempt + 1}/{self.max_retries})",
                    extra={
                        'model_name': self.model_name,
                        'attempt': attempt + 1,
                        'device': self.device,
                    }
                )

                # ai4bharat/indic-bert is a gated repo — pass HF token if set.
                # Set HF_TOKEN in your .env (or HUGGING_FACE_HUB_TOKEN) to avoid
                # passing it on every CLI invocation.
                import os
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
                self.model = AutoModel.from_pretrained(self.model_name, token=hf_token)
                self.model.eval()  # type: ignore
                if self.device != "cpu" and torch.cuda.is_available():
                    self.model = self.model.to(self.device)  # type: ignore

                logger.info(
                    "indic-bert loaded successfully (768-dim, Indic language support)",
                    extra={'model_name': self.model_name}
                )
                return

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Failed to load indic-bert (attempt {attempt + 1}/{self.max_retries}): {e}",
                    extra={
                        'model_name': self.model_name,
                        'attempt': attempt + 1,
                        'error': str(e),
                    }
                )

                if attempt < self.max_retries - 1:
                    delay_ms = retry_config.get_delay_ms(attempt)
                    logger.info(f"Retrying in {delay_ms}ms", extra={'delay_ms': delay_ms})
                    time.sleep(delay_ms / 1000.0)

        logger.error(
            f"Failed to load indic-bert after {self.max_retries} attempts",
            extra={
                'model_name': self.model_name,
                'max_retries': self.max_retries,
                'error': str(last_exception),
            }
        )
        raise RuntimeError(
            f"Failed to load indic-bert '{self.model_name}' after {self.max_retries} retries: {str(last_exception)}"
        )
    
    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool the last hidden state weighted by the attention mask.
        This is the standard way to produce sentence embeddings from a raw BERT model.
        """
        # Expand mask to match hidden state shape: (batch, seq_len) → (batch, seq_len, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Generate sentence embeddings for a batch of texts using indic-bert.

        Args:
            texts: List of text strings to embed (supports Kannada, Hindi, etc.)
            normalize: Whether to apply L2 normalization

        Returns:
            Array of shape (len(texts), 768) with embeddings
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        batch_size = min(32, len(texts))
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            if self.device != "cpu" and torch.cuda.is_available():
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean-pool over token dimension (ignore [PAD] tokens via attention mask)
            batch_embeddings = self._mean_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            ).cpu().numpy()
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        if normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings  # type: ignore
    
    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to embeddings (unit vector constraint).
        
        Args:
            embeddings: Array of shape (n, d) with embeddings
        
        Returns:
            L2-normalized embeddings of same shape
        """
        # Compute L2 norm for each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        # Normalize
        normalized = embeddings / norms
        return normalized  # type: ignore
    
    @staticmethod
    def compute_text_hash(text: str) -> str:
        """
        Compute SHA256 hash of text for use as cache key.
        
        Args:
            text: Text to hash
        
        Returns:
            Hex-encoded SHA256 hash
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


class EmbeddingBatcher:
    """
    Batch text embeddings with timeout-based flushing.
    
    Features:
    - Accumulates texts up to batch size (default 32)
    - Flushes on timeout (default 100ms) or batch full
    - Async interface for non-blocking operations
    - Returns embeddings via futures
    """
    
    def __init__(
        self,
        engine: EmbeddingEngine,
        cache: Optional[redis.Redis] = None,
        batch_size: int = 32,
        timeout_ms: int = 100,
        cache_ttl_seconds: int = 86400,
        latency_warning_threshold_ms: float = 50.0,
    ):
        """
        Initialize embedding batcher.
        
        Args:
            engine: EmbeddingEngine instance
            cache: Redis client for caching embeddings
            batch_size: Maximum batch size before flushing
            timeout_ms: Timeout in milliseconds for flushing
            cache_ttl_seconds: TTL for cached embeddings in seconds
            latency_warning_threshold_ms: Warn if embedding exceeds this latency per sample
        """
        self.engine = engine
        self.cache = cache
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.cache_ttl_seconds = cache_ttl_seconds
        self.latency_warning_threshold_ms = latency_warning_threshold_ms
        
        # Queue for batching
        self.batch_queue: List[Tuple[str, asyncio.Future]] = []
        self.last_flush_time = time.time()
        
        # Background task for timeout-based flushing
        self.flush_task: Optional[asyncio.Task] = None
        self.should_stop = False
    
    async def add_and_wait(
        self,
        text: str,
        timeout_ms: int = 5000,
    ) -> np.ndarray:
        """
        Add text to batch and wait for embedding.
        
        Args:
            text: Text to embed
            timeout_ms: Timeout for waiting on embedding
        
        Returns:
            Embedding vector of shape (768,)
        
        Raises:
            TimeoutError: If embedding not available within timeout
            RuntimeError: If model not loaded or embedding fails
        """
        # Create future for result
        future: asyncio.Future = asyncio.Future()
        
        # Add to batch queue
        self.batch_queue.append((text, future))
        
        # Check if we need to flush
        time_since_last_flush = (time.time() - self.last_flush_time) * 1000
        if len(self.batch_queue) >= self.batch_size or time_since_last_flush >= self.timeout_ms:
            await self.flush()
        
        # Schedule flush if timeout hasn't been scheduled
        if self.flush_task is None or self.flush_task.done():
            self.flush_task = asyncio.create_task(self._scheduled_flush())
        
        try:
            embedding = await asyncio.wait_for(
                future,
                timeout=timeout_ms / 1000.0
            )
            return embedding  # type: ignore
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout waiting for embedding after {timeout_ms}ms",
                extra={'timeout_ms': timeout_ms, 'text_length': len(text)}
            )
            raise TimeoutError(f"Embedding not available within {timeout_ms}ms")
    
    async def _scheduled_flush(self):
        """Periodically flush batch based on timeout."""
        try:
            await asyncio.sleep(self.timeout_ms / 1000.0)
            if self.batch_queue:
                await self.flush()
        except asyncio.CancelledError:
            pass
    
    async def flush(self):
        """
        Flush current batch and process embeddings.
        
        Retrieves from cache where possible, computes missing embeddings,
        and stores new embeddings in cache.
        """
        if not self.batch_queue:
            return
        
        # Split batch from queue
        batch = self.batch_queue[:self.batch_size]
        self.batch_queue = self.batch_queue[self.batch_size:]
        self.last_flush_time = time.time()
        
        texts = [text for text, _ in batch]
        futures = [future for _, future in batch]
        
        logger.debug(
            f"Flushing embedding batch with {len(texts)} texts",
            extra={'batch_size': len(texts)}
        )
        
        # Check cache first
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for idx, text in enumerate(texts):
            cache_key = EmbeddingEngine.compute_text_hash(text)
            
            if self.cache:
                try:
                    cached = self.cache.get(cache_key)
                    if cached:
                        embedding = np.frombuffer(cached, dtype=np.float32)  # type: ignore
                        embeddings.append((idx, embedding))
                        continue
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {str(e)}")
            
            # Not in cache, will need to embed
            texts_to_embed.append(text)
            indices_to_embed.append(idx)
        
        # Sort embeddings by original index for correct ordering
        embeddings_dict: Dict[int, np.ndarray] = {idx: emb for idx, emb in embeddings}
        
        # Compute missing embeddings
        if texts_to_embed:
            start_time = time.time()
            new_embeddings = self.engine.embed(texts_to_embed, normalize=True)
            embedding_time_ms = (time.time() - start_time) * 1000
            per_sample_latency_ms = embedding_time_ms / len(texts_to_embed)
            
            # Log warning if exceeds latency threshold
            if per_sample_latency_ms > self.latency_warning_threshold_ms:
                logger.warning(
                    f"Embedding latency exceeded threshold",
                    extra={
                        'per_sample_latency_ms': per_sample_latency_ms,
                        'threshold_ms': self.latency_warning_threshold_ms,
                        'batch_size': len(texts_to_embed),
                    }
                )
            
            # Cache new embeddings
            for idx, text, embedding in zip(indices_to_embed, texts_to_embed, new_embeddings):
                embeddings_dict[idx] = embedding
                
                if self.cache:
                    try:
                        cache_key = EmbeddingEngine.compute_text_hash(text)
                        # Store as bytes
                        embedding_bytes = embedding.astype(np.float32).tobytes()
                        self.cache.setex(
                            cache_key,
                            self.cache_ttl_seconds,
                            embedding_bytes
                        )
                    except Exception as e:
                        logger.warning(f"Cache store failed for text: {str(e)}")
        
        # Resolve futures in original order
        for idx, future in enumerate(futures):
            if not future.done():
                embedding = embeddings_dict.get(idx)
                if embedding is not None:
                    future.set_result(embedding)
                else:
                    future.set_exception(RuntimeError(f"Embedding not found for text at index {idx}"))
    
    async def close(self):
        """Close batcher and flush any pending embeddings."""
        self.should_stop = True
        
        # Flush remaining batch
        if self.batch_queue:
            await self.flush()
        
        # Cancel flush task
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass


class EmbeddingCache:
    """Redis-backed cache for embeddings with hash-based keys."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 86400,  # 24 hours default
        key_prefix: str = "embedding:",
    ):
        """
        Initialize embedding cache.
        
        Args:
            redis_client: Redis client instance
            ttl_seconds: Time-to-live for cached embeddings
            key_prefix: Prefix for cache keys
        """
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.
        
        Args:
            text: Text to look up
        
        Returns:
            Embedding as numpy array, or None if not cached
        """
        cache_key = self.key_prefix + EmbeddingEngine.compute_text_hash(text)
        
        try:
            cached = self.redis.get(cache_key)
            if cached:
                return np.frombuffer(cached, dtype=np.float32)  # type: ignore
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        
        return None
    
    def set(self, text: str, embedding: np.ndarray) -> bool:
        """
        Store embedding in cache.
        
        Args:
            text: Text associated with embedding
            embedding: Embedding vector
        
        Returns:
            True if successful, False otherwise
        """
        cache_key = self.key_prefix + EmbeddingEngine.compute_text_hash(text)
        
        try:
            embedding_bytes = embedding.astype(np.float32).tobytes()
            self.redis.setex(cache_key, self.ttl_seconds, embedding_bytes)
            return True
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
            return False
    
    def hit_rate(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            info = self.redis.info('stats')
            return {
                'hits': info.get('keyspace_hits', 0),  # type: ignore
                'misses': info.get('keyspace_misses', 0),  # type: ignore
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {str(e)}")
            return {'hits': 0, 'misses': 0}

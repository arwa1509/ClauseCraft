"""
Embedding Service
==================
Generates vector embeddings for document chunks and queries
using Sentence Transformers.
"""

import numpy as np
from typing import Optional
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Embedding will use fallback.")

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class Embedder:
    """
    Generates dense vector embeddings using Sentence Transformers.
    Supports batch embedding with progress tracking.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model = None
        self.dimension = EMBEDDING_DIMENSION
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        if not ST_AVAILABLE:
            logger.warning("sentence-transformers not available, using random embeddings")
            return

        try:
            self.model = SentenceTransformer(self.model_name)
            # Get actual dimension from model
            test_emb = self.model.encode(["test"])
            self.dimension = test_emb.shape[1]
            logger.info(
                f"✅ Loaded embedding model: {self.model_name} "
                f"(dim={self.dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            numpy array of shape (dimension,)
        """
        if self.model is None:
            # Fallback: deterministic pseudo-random embedding
            return self._fallback_embed(text)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.zeros((0, self.dimension))

        if self.model is None:
            return np.array([self._fallback_embed(t) for t in texts])

        if show_progress:
            logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

        if show_progress:
            logger.info(f"✅ Generated {len(embeddings)} embeddings (dim={self.dimension})")
        return embeddings

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-random embedding as fallback."""
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))

    def similarity_batch(self, query: str, texts: list[str]) -> list[float]:
        """Compute similarity between a query and multiple texts."""
        query_emb = self.embed(query)
        text_embs = self.embed_batch(texts, show_progress=False)
        return list(np.dot(text_embs, query_emb))

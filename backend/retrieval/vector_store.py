"""
FAISS Vector Store
===================
Manages the FAISS index for dense vector retrieval.
Supports building, saving, loading, and searching.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Vector store will use brute-force search.")

from config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_DIMENSION


class VectorStore:
    """
    FAISS-based vector store for dense retrieval.
    Stores embeddings alongside chunk metadata.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        self.index_path = index_path or FAISS_INDEX_PATH
        self.metadata_path = metadata_path or METADATA_PATH
        self.dimension = dimension

        self.index = None
        self.metadata: list[dict] = []  # Parallel list with FAISS vectors
        self._embeddings: Optional[np.ndarray] = None  # For brute-force fallback

        self._load()

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: list[dict],
        use_ivf: bool = False,
    ):
        """
        Build a FAISS index from embeddings and chunk metadata.

        Args:
            embeddings: numpy array of shape (n, dimension)
            chunks: List of chunk dicts with text and metadata
            use_ivf: If True, use IVF index for large datasets
        """
        n = len(embeddings)
        self.dimension = embeddings.shape[1] if n > 0 else self.dimension

        logger.info(f"Building vector index: {n} vectors, dim={self.dimension}")

        if FAISS_AVAILABLE:
            if use_ivf and n > 1000:
                # IVF index for large datasets
                nlist = min(int(np.sqrt(n)), 100)
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
                self.index.train(embeddings.astype(np.float32))
                self.index.add(embeddings.astype(np.float32))
                self.index.nprobe = min(10, nlist)
            else:
                # Flat index (exact search)
                self.index = faiss.IndexFlatIP(self.dimension)
                if n > 0:
                    self.index.add(embeddings.astype(np.float32))
        else:
            # Brute-force fallback
            self._embeddings = embeddings.astype(np.float32)

        # Store metadata
        self.metadata = []
        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "entities": chunk.get("entities", []),
            })

        logger.info(f"✅ Vector index built: {n} vectors")

    def add_vectors(self, embeddings: np.ndarray, chunks: list[dict]):
        """Add new vectors to an existing index."""
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embeddings.astype(np.float32))
        elif self._embeddings is not None:
            self._embeddings = np.vstack(
                [self._embeddings, embeddings.astype(np.float32)]
            )
        else:
            self.build_index(embeddings, chunks)
            return

        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "entities": chunk.get("entities", []),
            })

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[dict]:
        """
        Search the index for most similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,)
            top_k: Number of results to return

        Returns:
            List of dicts with text, metadata, and score
        """
        if not self.metadata:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            # FAISS search
            k = min(top_k, self.index.ntotal)
            if k == 0:
                return []
            scores, indices = self.index.search(query, k)
            scores = scores[0]
            indices = indices[0]
        elif self._embeddings is not None:
            # Brute-force search
            similarities = np.dot(self._embeddings, query.T).flatten()
            k = min(top_k, len(similarities))
            indices = np.argsort(similarities)[::-1][:k]
            scores = similarities[indices]
        else:
            return []

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
                "metadata": meta["metadata"],
                "entities": meta.get("entities", []),
                "score": float(score),
                "rank": len(results) + 1,
            })

        return results

    def save(self):
        """Save index and metadata to disk."""
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"💾 FAISS index saved: {self.index.ntotal} vectors")
        elif self._embeddings is not None:
            np.save(str(self.index_path) + ".npy", self._embeddings)
            logger.info(f"💾 Embeddings saved: {len(self._embeddings)} vectors")

        # Save metadata
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, default=str)
        )
        logger.info(f"💾 Metadata saved: {len(self.metadata)} chunks")

    def _load(self):
        """Load index and metadata from disk."""
        # Load metadata
        if self.metadata_path.exists():
            try:
                self.metadata = json.loads(self.metadata_path.read_text())
                logger.info(f"📂 Loaded metadata: {len(self.metadata)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = []

        # Load FAISS index
        if FAISS_AVAILABLE and self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"📂 FAISS index loaded: {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
        elif Path(str(self.index_path) + ".npy").exists():
            try:
                self._embeddings = np.load(str(self.index_path) + ".npy")
                logger.info(f"📂 Embeddings loaded: {len(self._embeddings)} vectors")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        if FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        elif self._embeddings is not None:
            return len(self._embeddings)
        return 0

    def clear(self):
        """Clear the vector store."""
        self.index = None
        self._embeddings = None
        self.metadata = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        logger.info("🗑️ Vector store cleared")

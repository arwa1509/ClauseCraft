"""
Cross-Encoder Reranker
=======================
Uses a cross-encoder model to re-score (query, passage) pairs
for more accurate relevance ranking.
"""

from typing import Optional
from loguru import logger

try:
    from sentence_transformers import CrossEncoder
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False
    logger.warning("CrossEncoder not available. Reranking will use original scores.")

from config import CROSS_ENCODER_MODEL, RERANK_TOP_K


class CrossEncoderReranker:
    """
    Cross-encoder based reranker.
    Scores each (query, passage) pair independently for better accuracy
    than bi-encoder retrieval alone.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or CROSS_ENCODER_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model."""
        if not CE_AVAILABLE:
            logger.warning("CrossEncoder not available, reranking disabled")
            return

        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info(f"✅ Loaded cross-encoder: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[dict]:
        """
        Rerank passages using the cross-encoder.

        Args:
            query: The search query
            passages: List of passage dicts with 'text' field
            top_k: Number of top results to return

        Returns:
            Reranked list of passage dicts with updated scores
        """
        if not passages:
            return []

        if self.model is None:
            # Fallback: return passages sorted by original score
            logger.debug("Cross-encoder not available, returning original ranking")
            return sorted(
                passages, key=lambda x: x.get("score", 0), reverse=True
            )[:top_k]

        # Create (query, passage) pairs
        pairs = [(query, p.get("text", "")) for p in passages]

        # Score all pairs
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return passages[:top_k]

        # Attach scores and sort
        scored_passages = []
        for passage, ce_score in zip(passages, scores):
            scored_passages.append({
                **passage,
                "cross_encoder_score": float(ce_score),
                "original_score": passage.get("score", 0),
                "score": float(ce_score),  # Use CE score as primary
            })

        # Sort by cross-encoder score
        scored_passages.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        # Update ranks
        for i, p in enumerate(scored_passages[:top_k]):
            p["rank"] = i + 1

        logger.debug(
            f"Reranked {len(passages)} passages → top {top_k} "
            f"(best CE score: {scored_passages[0]['cross_encoder_score']:.4f})"
        )

        return scored_passages[:top_k]

    def score_pair(self, query: str, passage: str) -> float:
        """Score a single (query, passage) pair."""
        if self.model is None:
            return 0.0

        try:
            score = self.model.predict([(query, passage)])
            return float(score[0])
        except Exception:
            return 0.0

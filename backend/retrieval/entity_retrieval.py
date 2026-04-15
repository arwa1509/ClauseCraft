"""
Entity-Based Retrieval
=======================
Retrieves chunks based on entity matching using the entity index.
Complements dense retrieval for precise entity-based search.
"""

import json
from typing import Optional
from loguru import logger

from ner.entity_index import EntityIndex
from config import ENTITY_TOP_K, METADATA_PATH


class EntityRetriever:
    """
    Retrieves chunks that contain specific entities.
    Uses the entity index for fast lookup.
    """

    def __init__(self, entity_index: Optional[EntityIndex] = None):
        self.entity_index = entity_index or EntityIndex()
        self._chunk_cache: dict = {}

    def _load_chunks(self) -> dict:
        """Load chunk texts from metadata file for returning full results."""
        if not self._chunk_cache:
            if METADATA_PATH.exists():
                try:
                    chunks = json.loads(METADATA_PATH.read_text())
                    self._chunk_cache = {c["chunk_id"]: c for c in chunks}
                except Exception as e:
                    logger.warning(f"Failed to load chunk cache: {e}")
        return self._chunk_cache

    def retrieve(
        self,
        entity_texts: list[str],
        top_k: int = ENTITY_TOP_K,
        use_fuzzy: bool = True,
    ) -> list[dict]:
        """
        Retrieve chunks containing specific entities.

        Args:
            entity_texts: List of entity texts to search for
            top_k: Maximum number of results
            use_fuzzy: Whether to use fuzzy matching

        Returns:
            List of chunk dicts with text, metadata, score
        """
        chunk_scores: dict[str, float] = {}
        chunk_entities: dict[str, list] = {}

        for entity_text in entity_texts:
            if use_fuzzy:
                matches = self.entity_index.lookup_fuzzy(entity_text)
            else:
                matches = [
                    {**m, "similarity": 1.0}
                    for m in self.entity_index.lookup(entity_text)
                ]

            for match in matches:
                chunk_id = match["chunk_id"]
                similarity = match.get("similarity", 1.0)

                # Accumulate scores (more entity matches → higher score)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + similarity

                if chunk_id not in chunk_entities:
                    chunk_entities[chunk_id] = []
                chunk_entities[chunk_id].append({
                    "text": entity_text,
                    "label": match.get("label", ""),
                    "similarity": similarity,
                })

        # Sort by accumulated score
        sorted_chunks = sorted(
            chunk_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Build results with full chunk data
        chunks = self._load_chunks()
        results = []

        for chunk_id, score in sorted_chunks:
            chunk_data = chunks.get(chunk_id, {})
            results.append({
                "chunk_id": chunk_id,
                "text": chunk_data.get("text", ""),
                "metadata": chunk_data.get("metadata", {}),
                "entities": chunk_data.get("entities", []),
                "score": score / len(entity_texts),  # Normalize by query entity count
                "matched_entities": chunk_entities.get(chunk_id, []),
                "rank": len(results) + 1,
            })

        logger.debug(
            f"Entity retrieval: {entity_texts} → {len(results)} results"
        )

        return results

"""
Entity Index
==============
Inverted index mapping entities to chunk IDs for fast entity-based retrieval.
Persistent storage with incremental update support.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Optional
from loguru import logger

from config import ENTITY_INDEX_PATH


class EntityIndex:
    """
    Inverted index: entity → list of chunk_ids.
    Supports efficient entity-based document retrieval.
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or ENTITY_INDEX_PATH
        # entity_key → [{"chunk_id": str, "label": str, "text": str}]
        self.index: dict[str, list[dict]] = defaultdict(list)
        # label → set of entity_keys
        self.label_index: dict[str, set] = defaultdict(set)

        self._load()

    def _make_key(self, text: str) -> str:
        """Create a normalized key for an entity."""
        return text.strip().lower()

    def add_entities(self, chunk_id: str, entities: list[dict]):
        """
        Add entities from a chunk to the index.

        Args:
            chunk_id: ID of the chunk these entities belong to
            entities: List of entity dicts (text, label, ...)
        """
        for entity in entities:
            key = self._make_key(entity["text"])
            label = entity.get("label", "UNKNOWN")

            # Avoid duplicate entries
            existing_chunks = {e["chunk_id"] for e in self.index[key]}
            if chunk_id not in existing_chunks:
                self.index[key].append({
                    "chunk_id": chunk_id,
                    "label": label,
                    "text": entity["text"],
                })

            self.label_index[label].add(key)

    def lookup(self, entity_text: str) -> list[dict]:
        """
        Look up chunks containing a specific entity.

        Args:
            entity_text: The entity text to look up

        Returns:
            List of chunk references
        """
        key = self._make_key(entity_text)
        return self.index.get(key, [])

    def lookup_fuzzy(self, entity_text: str, threshold: float = 0.7) -> list[dict]:
        """
        Fuzzy lookup: find chunks with entities similar to the query.

        Args:
            entity_text: The entity text to search for
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of chunk references with similarity scores
        """
        query_key = self._make_key(entity_text)
        results = []

        # Exact match first
        if query_key in self.index:
            for ref in self.index[query_key]:
                results.append({**ref, "similarity": 1.0})

        # Fuzzy matching
        query_tokens = set(query_key.split())
        for key in self.index:
            if key == query_key:
                continue
            key_tokens = set(key.split())

            # Jaccard similarity
            if query_tokens and key_tokens:
                intersection = query_tokens & key_tokens
                union = query_tokens | key_tokens
                similarity = len(intersection) / len(union)

                if similarity >= threshold:
                    for ref in self.index[key]:
                        results.append({**ref, "similarity": similarity})

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def get_entities_by_label(self, label: str) -> list[str]:
        """Get all entities of a specific type."""
        return list(self.label_index.get(label, set()))

    def get_all_entities(self) -> dict:
        """Get all entities grouped by label."""
        result = {}
        for label, keys in self.label_index.items():
            result[label] = list(keys)
        return result

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_entities": len(self.index),
            "total_references": sum(len(v) for v in self.index.values()),
            "entities_by_label": {
                label: len(keys) for label, keys in self.label_index.items()
            },
        }

    def save(self):
        """Save index to disk."""
        data = {
            "index": dict(self.index),
            "label_index": {k: list(v) for k, v in self.label_index.items()},
        }
        self.index_path.write_text(json.dumps(data, indent=2))
        logger.info(f"💾 Entity index saved: {len(self.index)} entities")

    def _load(self):
        """Load index from disk if exists."""
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text())
                self.index = defaultdict(list, data.get("index", {}))
                self.label_index = defaultdict(
                    set,
                    {k: set(v) for k, v in data.get("label_index", {}).items()},
                )
                logger.info(f"📂 Entity index loaded: {len(self.index)} entities")
            except Exception as e:
                logger.warning(f"Failed to load entity index: {e}")

    def clear(self):
        """Clear the entire index."""
        self.index = defaultdict(list)
        self.label_index = defaultdict(set)
        if self.index_path.exists():
            self.index_path.unlink()
        logger.info("🗑️ Entity index cleared")

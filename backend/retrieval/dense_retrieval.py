"""
Hybrid dense retrieval with conservative score blending.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from config import DENSE_TOP_K, METADATA_PATH


class DenseRetriever:
    """
    Retrieve passages using vector similarity plus lightweight lexical support.

    The scoring intentionally favors semantic retrieval and keeps structural
    heuristics as small tie-breakers rather than dominant ranking signals.
    """

    def __init__(
        self,
        embedder: Optional["Embedder"] = None,
        vector_store: Optional["VectorStore"] = None,
    ):
        self.embedder = embedder or __import__(
            "retrieval.embedder", fromlist=["Embedder"]
        ).Embedder()
        self.vector_store = vector_store or __import__(
            "retrieval.vector_store", fromlist=["VectorStore"]
        ).VectorStore()
        self._bm25 = None
        self._corpus_chunks = []
        self._build_bm25()

    def _build_bm25(self):
        self._bm25 = None
        self._corpus_chunks = []
        if not METADATA_PATH.exists():
            return
        with open(METADATA_PATH, "r", encoding="utf-8") as handle:
            self._corpus_chunks = json.load(handle)
        tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in self._corpus_chunks]
        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25 index on {len(tokenized_corpus)} chunks")

    def rebuild_bm25(self):
        """Refresh lexical retrieval from the latest on-disk chunk metadata."""
        self._build_bm25()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9\.\-/]*", text.lower())

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        max_score = max(values)
        min_score = min(values)
        if max_score == min_score:
            return {key: 1.0 for key in scores}
        return {
            key: (value - min_score) / (max_score - min_score)
            for key, value in scores.items()
        }

    def _get_bm25_scores(self, query: str) -> dict[str, float]:
        if not self._bm25:
            return {}
        query_terms = self._tokenize(query)
        raw_scores = self._bm25.get_scores(query_terms)
        scores = {
            self._corpus_chunks[i]["chunk_id"]: float(score)
            for i, score in enumerate(raw_scores)
            if score > 0
        }
        return self._normalize_scores(scores)

    def _structure_bonus(self, chunk_data: dict, query_intent: str) -> float:
        text = chunk_data.get("text", "").lower()
        meta = chunk_data.get("metadata", {})
        bonus = 0.0

        if query_intent == "section" and "section" in text:
            bonus += 0.08
        if query_intent == "definition" and any(
            marker in text for marker in {"means", "refers to", "includes"}
        ):
            bonus += 0.06
        if query_intent == "condition" and any(
            marker in text for marker in {"provided that", "subject to", "shall"}
        ):
            bonus += 0.06

        chunk_idx = meta.get("chunk_index", 0)
        total_chunks = max(meta.get("total_chunks", 1), 1)
        if chunk_idx / total_chunks <= 0.15:
            bonus += 0.03

        if "prayer" in text:
            bonus -= 0.08

        return bonus

    def retrieve(
        self, query: str, top_k: int = DENSE_TOP_K, query_intent: str = "general"
    ) -> list[dict]:
        if self.vector_store.size == 0:
            logger.warning("Vector store is empty. Process documents first.")
            return []

        query_embedding = self.embedder.embed(query)
        dense_results = self.vector_store.search(query_embedding, top_k=top_k * 3)
        dense_scores = {result["chunk_id"]: float(result["score"]) for result in dense_results}
        dense_scores = self._normalize_scores(dense_scores)
        dense_data = {result["chunk_id"]: result for result in dense_results}

        bm25_scores = self._get_bm25_scores(query)
        all_ids = set(dense_scores) | set(sorted(bm25_scores, key=bm25_scores.get, reverse=True)[: top_k * 2])

        final_results = []
        for chunk_id in all_ids:
            chunk_data = dense_data.get(chunk_id)
            if not chunk_data:
                chunk_data = next(
                    (chunk for chunk in self._corpus_chunks if chunk.get("chunk_id") == chunk_id),
                    None,
                )
                if not chunk_data:
                    continue
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": chunk_data.get("text", ""),
                    "metadata": chunk_data.get("metadata", {}),
                    "entities": chunk_data.get("entities", []),
                }

            semantic_score = dense_scores.get(chunk_id, 0.0)
            lexical_score = bm25_scores.get(chunk_id, 0.0)
            entity_overlap = self._entity_overlap(query, chunk_data.get("entities", []))
            structure_bonus = self._structure_bonus(chunk_data, query_intent)

            hybrid_score = (
                (0.7 * semantic_score)
                + (0.2 * lexical_score)
                + (0.08 * entity_overlap)
                + structure_bonus
            )

            final_results.append(
                {
                    **chunk_data,
                    "score": round(min(max(hybrid_score, 0.0), 1.0), 4),
                }
            )

        final_results.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(final_results[:top_k], start=1):
            item["rank"] = rank
        return final_results[:top_k]

    def _entity_overlap(self, query: str, entities: list[dict]) -> float:
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return 0.0
        overlap = 0
        for entity in entities:
            entity_terms = set(self._tokenize(entity.get("text", "")))
            if entity_terms and entity_terms & query_terms:
                overlap += 1
        return min(overlap * 0.25, 1.0)

    def retrieve_with_expansion(
        self,
        query: str,
        expanded_queries: list[str],
        top_k: int = DENSE_TOP_K,
    ) -> list[dict]:
        all_results = {}
        for result in self.retrieve(query, top_k):
            all_results[result["chunk_id"]] = result

        for expanded_query in expanded_queries:
            for result in self.retrieve(expanded_query, max(1, top_k // 2)):
                current = all_results.get(result["chunk_id"])
                if current is None or result["score"] > current["score"]:
                    all_results[result["chunk_id"]] = result

        results = sorted(all_results.values(), key=lambda item: item["score"], reverse=True)
        return results[:top_k]

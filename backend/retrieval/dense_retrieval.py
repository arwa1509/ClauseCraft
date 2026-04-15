"""
Dense Retrieval
================
FAISS-based dense vector retrieval for semantic search.
"""

import json
from typing import Optional
from loguru import logger
import math
from rank_bm25 import BM25Okapi
import numpy as np

from config import DENSE_TOP_K, METADATA_PATH


class DenseRetriever:
    """
    Hybrid retriever using Dense FAISS search + BM25 keyword matching
    for semantic + strict keyword search.
    """

    def __init__(
        self,
        embedder: Optional["Embedder"] = None,
        vector_store: Optional["VectorStore"] = None,
    ):
        self.embedder = embedder or __import__('retrieval.embedder', fromlist=['Embedder']).Embedder()
        self.vector_store = vector_store or __import__('retrieval.vector_store', fromlist=['VectorStore']).VectorStore()
        self._bm25 = None
        self._corpus_chunks = []
        self._build_bm25()
        
    def _build_bm25(self):
        """Build BM25 index over all local chunks."""
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                self._corpus_chunks = json.load(f)
            
            tokenized_corpus = [doc.get("text", "").lower().split() for doc in self._corpus_chunks]
            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
                logger.info(f"✅ Built BM25 search index on {len(tokenized_corpus)} chunks")

    def _get_bm25_scores(self, query: str) -> dict:
        if not self._bm25:
            return {}
        
        # Stopword removal and split
        stopwords = {"this", "is", "a", "the", "does", "what", "are", "in", "by", "of", "deal", "with", "handled"}
        query_terms = [w for w in query.lower().split() if w not in stopwords]
        
        # Specific domain boosting
        boosted_terms = []
        for term in query_terms:
            boosted_terms.append(term)
            if term in ["maintenance", "custody", "jurisdiction"]:
                boosted_terms.extend([term, term]) # triple weight

        scores = self._bm25.get_scores(boosted_terms)
        
        return {
            self._corpus_chunks[i]["chunk_id"]: float(score)
            for i, score in enumerate(scores)
            if score > 0
        }

    def retrieve(
        self, query: str, top_k: int = DENSE_TOP_K
    ) -> list[dict]:
        """
        Perform hybrid FAISS + BM25 keyword retrieval.
        Score = 0.5 * Semantic + 0.3 * Keyword
        """
        if self.vector_store.size == 0:
            logger.warning("Vector store is empty. Process documents first.")
            return []

        # 1. Semantic Search
        query_embedding = self.embedder.embed(query)
        dense_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        dense_scores = {r["chunk_id"]: r["score"] for r in dense_results}
        chunk_data = {r["chunk_id"]: r for r in dense_results}

        # 2. BM25 Keyword Search
        bm25_scores = self._get_bm25_scores(query)

        # Build complete set of chunk IDs
        all_ids = set()
        for cid in dense_scores.keys():
            all_ids.add(cid)
            
        bm25_sorted_keys = []
        if bm25_scores:
            bm25_sorted_keys = sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:top_k]
            for cid in bm25_sorted_keys:
                all_ids.add(cid)

        final_results = []
        for cid in all_ids:
            s_score = dense_scores.get(cid, 0.0)
            k_score = bm25_scores.get(cid, 0.0)
            
            # Hybrid Weighting: Normalize roughly to handle scale differences
            hybrid_score = (0.5 * s_score) + (0.3 * (k_score * 0.1)) 

            c_data = chunk_data.get(cid)
            if not c_data:
                # If it's a BM25 only grab, fish it out of the corpus cache
                c_idx = next((i for i, c in enumerate(self._corpus_chunks) if c["chunk_id"] == cid), None)
                if c_idx is not None:
                    cd = self._corpus_chunks[c_idx]
                    c_data = {
                        "chunk_id": cid,
                        "text": cd.get("text", ""),
                        "metadata": cd.get("metadata", {}),
                        "entities": cd.get("entities", [])
                    }
                else: 
                    continue
            
            c_data["score"] = hybrid_score
            final_results.append(c_data)

        # Sort and trim
        final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]

        logger.debug(
            f"Hybrid Dense+BM25 retrieval: query='{query[:50]}...' → {len(final_results)} results"
        )

        return final_results

    def retrieve_with_expansion(
        self,
        query: str,
        expanded_queries: list[str],
        top_k: int = DENSE_TOP_K,
    ) -> list[dict]:
        """
        Retrieve with query expansion — search with multiple query variants
        and merge results.

        Args:
            query: Original query
            expanded_queries: List of expanded/reformulated queries
            top_k: Number of results

        Returns:
            Merged and deduplicated results
        """
        all_results = {}

        # Search with original query
        for result in self.retrieve(query, top_k):
            cid = result["chunk_id"]
            if cid not in all_results or result["score"] > all_results[cid]["score"]:
                all_results[cid] = result

        # Search with expanded queries
        for eq in expanded_queries:
            for result in self.retrieve(eq, top_k // 2):
                cid = result["chunk_id"]
                if cid not in all_results or result["score"] > all_results[cid]["score"]:
                    all_results[cid] = result

        # Sort by score and return top_k
        results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

"""
Retrieval API Router
=====================
Endpoints for retrieval operations.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.dense_retrieval import DenseRetriever
from retrieval.entity_retrieval import EntityRetriever
from retrieval.fusion import reciprocal_rank_fusion
from ner.entity_index import EntityIndex

router = APIRouter()

_components = {}


def _get_components():
    if not _components:
        _components["embedder"] = Embedder()
        _components["store"] = VectorStore()
        _components["dense"] = DenseRetriever(_components["embedder"], _components["store"])
        _components["entity_index"] = EntityIndex()
        _components["entity"] = EntityRetriever(_components["entity_index"])
    return _components


class RetrievalRequest(BaseModel):
    query: str
    top_k: Optional[int] = 20
    method: Optional[str] = "hybrid"  # dense | entity | hybrid
    entity_texts: Optional[list[str]] = None


@router.post("/search")
async def search(request: RetrievalRequest):
    """Search for relevant chunks using dense, entity, or hybrid retrieval."""
    comp = _get_components()

    dense_results = []
    entity_results = []

    if request.method in ("dense", "hybrid"):
        dense_results = comp["dense"].retrieve(request.query, top_k=request.top_k)

    if request.method in ("entity", "hybrid") and request.entity_texts:
        entity_results = comp["entity"].retrieve(
            request.entity_texts, top_k=request.top_k
        )

    if request.method == "hybrid" and dense_results and entity_results:
        results = reciprocal_rank_fusion(dense_results, entity_results)
    elif dense_results:
        results = dense_results
    elif entity_results:
        results = entity_results
    else:
        results = []

    return {
        "query": request.query,
        "method": request.method,
        "results": results[:request.top_k],
        "total": len(results),
        "dense_count": len(dense_results),
        "entity_count": len(entity_results),
    }


@router.get("/stats")
async def get_stats():
    """Get retrieval system statistics."""
    comp = _get_components()
    return {
        "vector_store_size": comp["store"].size,
        "entity_index_stats": comp["entity_index"].get_stats(),
    }

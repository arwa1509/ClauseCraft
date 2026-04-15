"""
Ranking API Router
===================
Endpoints for reranking operations.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ranking.cross_encoder import CrossEncoderReranker

router = APIRouter()

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


class RerankRequest(BaseModel):
    query: str
    passages: list[dict]
    top_k: Optional[int] = 5


@router.post("/rerank")
async def rerank(request: RerankRequest):
    """Rerank passages using cross-encoder."""
    reranker = _get_reranker()
    results = reranker.rerank(
        request.query, request.passages, top_k=request.top_k
    )
    return {
        "query": request.query,
        "reranked": results,
        "total": len(results),
    }


class ScoreRequest(BaseModel):
    query: str
    passage: str


@router.post("/score")
async def score_pair(request: ScoreRequest):
    """Score a single query-passage pair."""
    reranker = _get_reranker()
    score = reranker.score_pair(request.query, request.passage)
    return {"query": request.query, "score": score}

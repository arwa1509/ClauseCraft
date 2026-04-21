"""
Generation API Router
======================
Endpoints for RAG-based answer generation.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from generation.agentic_router import AgenticRouter

router = APIRouter()

_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        _generator = AgenticRouter()
    return _generator


class GenerateRequest(BaseModel):
    query: str
    passages: list[dict]
    intent: Optional[str] = "general"
    entities: Optional[list[dict]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1


@router.post("/generate")
async def generate_answer(request: GenerateRequest):
    """Generate an answer from query and retrieved passages."""
    gen = _get_generator()
    answer = gen.run(
        query=request.query,
        passages=request.passages,
        intent=request.intent,
        entities=request.entities,
    )
    return {
        "query": request.query,
        "answer": answer,
        "intent": request.intent,
        "num_passages": len(request.passages),
    }

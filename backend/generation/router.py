"""
Generation API Router
======================
Endpoints for RAG-based answer generation.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from generation.rag_generator import RAGGenerator

router = APIRouter()

_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
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
    answer = gen.generate(
        query=request.query,
        passages=request.passages,
        intent=request.intent,
        entities=request.entities,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return {
        "query": request.query,
        "answer": answer,
        "intent": request.intent,
        "num_passages": len(request.passages),
    }

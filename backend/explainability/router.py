"""
Explainability API Router
===========================
Endpoints for explainability operations.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from explainability.claim_mapper import ClaimMapper
from explainability.confidence import ConfidenceScorer
from explainability.highlighter import EntityHighlighter

router = APIRouter()


class ExplainRequest(BaseModel):
    answer: dict | str
    source_passages: list[dict]
    query_entities: Optional[list[dict]] = []


class HighlightRequest(BaseModel):
    text: str
    entities: list[dict]
    format: Optional[str] = "annotations"  # annotations | html | markdown


@router.post("/explain")
async def explain_answer(request: ExplainRequest):
    """Map answer claims to source passages and compute confidence."""
    mapper = ClaimMapper()
    scorer = ConfidenceScorer()

    # Map claims to sources
    claim_mappings = mapper.map_claims(request.answer, request.source_passages)

    # Compute confidence
    confidence = scorer.compute(
        request.source_passages, request.query_entities or [], claim_mappings
    )

    # Per-claim confidence
    per_claim = scorer.compute_per_claim(claim_mappings)

    return {
        "claim_mappings": claim_mappings,
        "confidence": confidence,
        "per_claim_confidence": per_claim,
        "total_claims": len(claim_mappings),
    }


@router.post("/highlight")
async def highlight_entities(request: HighlightRequest):
    """Highlight entities in text."""
    highlighter = EntityHighlighter()
    result = highlighter.highlight(request.text, request.entities, request.format)
    return {"result": result}


@router.get("/entity-legend")
async def get_entity_legend():
    """Get the entity type color legend."""
    return {"legend": EntityHighlighter.get_entity_legend()}

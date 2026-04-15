"""
Evaluation API Router
======================
Endpoints for evaluation and quality metrics.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from evaluation.metrics import EvaluationMetrics
from evaluation.hallucination import HallucinationDetector

router = APIRouter()

metrics = EvaluationMetrics()
_hallucination_detector = None


def _get_hallucination_detector():
    global _hallucination_detector
    if _hallucination_detector is None:
        _hallucination_detector = HallucinationDetector()
    return _hallucination_detector


class HallucinationRequest(BaseModel):
    answer: str
    source_passages: list[dict]


class EntityF1Request(BaseModel):
    predicted: list[dict]
    gold: list[dict]


class RougeLRequest(BaseModel):
    predicted: str
    reference: str


@router.post("/hallucination")
async def check_hallucination(request: HallucinationRequest):
    """Check generated answer for hallucinations."""
    detector = _get_hallucination_detector()
    return detector.detect(request.answer, request.source_passages)


@router.post("/entity-f1")
async def compute_entity_f1(request: EntityF1Request):
    """Compute entity-level F1 score."""
    return metrics.entity_f1(request.predicted, request.gold)


@router.post("/rouge-l")
async def compute_rouge_l(request: RougeLRequest):
    """Compute ROUGE-L score."""
    return metrics.rouge_l(request.predicted, request.reference)


@router.get("/dataset")
async def check_eval_dataset(path: Optional[str] = None):
    """Check if evaluation dataset is available."""
    return metrics.evaluate_dataset(path)

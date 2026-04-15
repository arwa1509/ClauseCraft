"""
NER API Router
===============
Endpoints for Named Entity Recognition operations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from ner.rule_based import RuleBasedNER
from ner.ml_based import MLBasedNER
from ner.entity_index import EntityIndex
from ner.entity_linker import EntityLinker

router = APIRouter()

# Lazy singletons
_ner_components = {}


def _get_components():
    if not _ner_components:
        _ner_components["rule"] = RuleBasedNER()
        _ner_components["ml"] = MLBasedNER()
        _ner_components["index"] = EntityIndex()
        _ner_components["linker"] = EntityLinker()
    return _ner_components


class NERRequest(BaseModel):
    text: str
    use_rule_based: Optional[bool] = True
    use_ml_based: Optional[bool] = True


class FineTuneRequest(BaseModel):
    dataset_path: Optional[str] = None
    n_iter: Optional[int] = 30
    dropout: Optional[float] = 0.35


@router.post("/extract")
async def extract_entities(request: NERRequest):
    """Extract named entities from text using rule-based and/or ML methods."""
    components = _get_components()

    entities = []

    if request.use_rule_based:
        rule_entities = components["rule"].extract(request.text)
        entities.extend(rule_entities)

    if request.use_ml_based:
        ml_entities = components["ml"].extract(request.text)
        entities.extend(ml_entities)

    # Deduplicate
    seen = set()
    unique = []
    for e in entities:
        key = (e["text"].lower(), e["label"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return {
        "text": request.text,
        "entities": unique,
        "count": len(unique),
        "methods_used": {
            "rule_based": request.use_rule_based,
            "ml_based": request.use_ml_based,
        },
    }


@router.post("/fine-tune")
async def fine_tune_ner(request: FineTuneRequest):
    """Fine-tune the ML NER model with custom legal training data."""
    components = _get_components()
    result = components["ml"].fine_tune(
        training_data_path=request.dataset_path,
        n_iter=request.n_iter,
        dropout=request.dropout,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/index/stats")
async def get_entity_index_stats():
    """Get entity index statistics."""
    components = _get_components()
    return components["index"].get_stats()


@router.get("/index/entities")
async def get_all_entities():
    """Get all indexed entities grouped by label."""
    components = _get_components()
    return components["index"].get_all_entities()


@router.get("/index/lookup/{entity_text}")
async def lookup_entity(entity_text: str, fuzzy: bool = False):
    """Look up chunks containing a specific entity."""
    components = _get_components()

    if fuzzy:
        results = components["index"].lookup_fuzzy(entity_text)
    else:
        results = components["index"].lookup(entity_text)

    return {
        "entity": entity_text,
        "fuzzy": fuzzy,
        "results": results,
        "count": len(results),
    }

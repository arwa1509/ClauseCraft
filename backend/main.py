"""
ClauseCraft backend application.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from config import (
    DENSE_TOP_K,
    ENTITY_TOP_K,
    LOG_FILE,
    LOG_LEVEL,
    QUERY_HISTORY_PATH,
    RERANK_TOP_K,
)
from evaluation.router import router as evaluation_router
from explainability.claim_mapper import ClaimMapper
from explainability.confidence import ConfidenceScorer
from explainability.highlighter import EntityHighlighter
from explainability.router import router as explainability_router
from generation.agentic_router import AgenticRouter
from generation.router import router as generation_router
from ingestion.router import router as ingestion_router
from ner.entity_index import EntityIndex
from ner.ml_based import MLBasedNER
from ner.router import router as ner_router
from ner.rule_based import RuleBasedNER
from ranking.cross_encoder import CrossEncoderReranker
from ranking.router import router as ranking_router
from retrieval.dense_retrieval import DenseRetriever
from retrieval.embedder import Embedder
from retrieval.entity_retrieval import EntityRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.router import router as retrieval_router
from retrieval.vector_store import VectorStore

logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    ),
)
logger.add(str(LOG_FILE), rotation="10 MB", retention="7 days", level="DEBUG")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ClauseCraft starting up")
    if QUERY_HISTORY_PATH.exists():
        try:
            QUERY_HISTORY_PATH.unlink()
            logger.info("Cleared previous query history for a fresh session")
        except Exception as exc:
            logger.warning(f"Could not clear query history on startup: {exc}")
    yield
    logger.info("ClauseCraft shutting down")


app = FastAPI(
    title="ClauseCraft Legal RAG System",
    description="Grounded legal document analysis with entity-aware retrieval",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router, prefix="/api/ingestion", tags=["Ingestion"])
app.include_router(ner_router, prefix="/api/ner", tags=["NER"])
app.include_router(retrieval_router, prefix="/api/retrieval", tags=["Retrieval"])
app.include_router(ranking_router, prefix="/api/ranking", tags=["Ranking"])
app.include_router(generation_router, prefix="/api/generation", tags=["Generation"])
app.include_router(
    explainability_router,
    prefix="/api/explainability",
    tags=["Explainability"],
)
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["Evaluation"])


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "ClauseCraft Legal RAG System"}


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_entity_retrieval: Optional[bool] = True
    use_reranking: Optional[bool] = True


class Citation(BaseModel):
    id: int
    chunk_id: str = ""
    doc_name: str = "Unknown"
    page_num: Optional[int] = None
    section: Optional[str] = None
    source: str = "local"
    url: Optional[str] = None


class SupportingPassage(BaseModel):
    chunk_id: str = ""
    text: str = ""
    metadata: dict = Field(default_factory=dict)
    score: float = 0.0


class EvidencePoint(BaseModel):
    text: str
    citation_ids: list[int] = Field(default_factory=list)
    chunk_id: str = ""
    source: str = "local"
    page_num: Optional[int] = None
    section: Optional[str] = None


class AnswerSegment(BaseModel):
    text: str
    citation_ids: list[int] = Field(default_factory=list)
    chunk_id: str = ""
    score: float = 0.0
    page_num: Optional[int] = None
    section: Optional[str] = None


class AnswerPayload(BaseModel):
    simple_answer: str
    markdown_answer: str = ""
    answer_text: str = ""
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_points: list[EvidencePoint] = Field(default_factory=list)
    answer_segments: list[AnswerSegment] = Field(default_factory=list)
    confidence: float = 0.0
    answer_type: str = "grounded_extractive"
    router_confidence: float = 0.0
    source: str = "local"
    web_augmented: bool = False
    web_results_count: int = 0
    web_error: Optional[str] = None


class QueryResponse(BaseModel):
    answer: AnswerPayload
    sources: list
    entities: list
    confidence: float
    explanation: dict
    query_entities: list
    query_intent: str
    processing_time: float


_pipeline_components = {}
_pipeline_lock = Lock()
_history_lock = Lock()


def _get_pipeline():
    if not _pipeline_components:
        with _pipeline_lock:
            if not _pipeline_components:
                logger.info("Initializing pipeline components")
                _pipeline_components["rule_ner"] = RuleBasedNER()
                _pipeline_components["ml_ner"] = MLBasedNER()
                _pipeline_components["entity_index"] = EntityIndex()
                _pipeline_components["embedder"] = Embedder()
                _pipeline_components["vector_store"] = VectorStore()
                _pipeline_components["dense_retriever"] = DenseRetriever(
                    _pipeline_components["embedder"],
                    _pipeline_components["vector_store"],
                )
                _pipeline_components["entity_retriever"] = EntityRetriever(
                    _pipeline_components["entity_index"]
                )
                _pipeline_components["reranker"] = CrossEncoderReranker()
                _pipeline_components["generator"] = AgenticRouter(
                    embedder=_pipeline_components["embedder"]
                )
                _pipeline_components["claim_mapper"] = ClaimMapper(
                    _pipeline_components["embedder"]
                )
                _pipeline_components["confidence_scorer"] = ConfidenceScorer()
                _pipeline_components["highlighter"] = EntityHighlighter()
    return _pipeline_components


def _detect_intent(question: str, entities: list) -> str:
    q_lower = question.lower()
    if any(w in q_lower for w in ["what is", "define", "meaning of", "definition"]):
        return "definition"
    if "section" in q_lower or any(e.get("label") == "STATUTE" for e in entities):
        return "section"
    if any(w in q_lower for w in ["why", "reason"]):
        return "reasoning"
    if any(w in q_lower for w in ["can", "when", "if"]):
        return "condition"
    if any(w in q_lower for w in ["how to", "procedure", "process", "steps"]):
        return "procedure"
    if any(w in q_lower for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"
    if any(w in q_lower for w in ["case", "judgment", "ruling", "verdict", "held"]):
        return "case_based"
    if any(w in q_lower for w in ["who", "which court", "which judge"]):
        return "factual"
    return "general"


def _save_query_history(question: str, response: dict):
    with _history_lock:
        history = []
        if QUERY_HISTORY_PATH.exists():
            try:
                history = json.loads(QUERY_HISTORY_PATH.read_text())
            except Exception:
                history = []
        history.append(
            {
                "question": question,
                "answer": response.get("answer", {}).get("simple_answer", ""),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": response.get("confidence", 0),
                "num_sources": len(response.get("sources", [])),
            }
        )
        history = history[-500:]
        QUERY_HISTORY_PATH.write_text(json.dumps(history, indent=2))


def _merge_entities(list1: list, list2: list) -> list:
    seen = set()
    merged = []
    for entity in list1 + list2:
        key = (entity.get("text", "").lower(), entity.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(entity)
    return merged


def _empty_answer() -> dict:
    return {
        "simple_answer": (
            "I could not find enough support in the indexed documents to answer this question."
        ),
        "markdown_answer": (
            "I could not find enough support in the indexed documents to answer this question."
        ),
        "answer_text": "",
        "supporting_passages": [],
        "key_entities": [],
        "citations": [],
        "evidence_points": [],
        "answer_segments": [],
        "confidence": 0.0,
        "answer_type": "insufficient_context",
        "router_confidence": 0.0,
        "source": "local",
        "web_augmented": False,
        "web_results_count": 0,
        "web_error": None,
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Query Pipeline"])
async def full_query_pipeline(request: QueryRequest):
    start_time = time.time()
    pipeline = _get_pipeline()

    logger.info(f"Query: {request.question}")
    rule_entities = pipeline["rule_ner"].extract(request.question)
    ml_entities = pipeline["ml_ner"].extract(request.question)
    query_entities = _merge_entities(rule_entities, ml_entities)

    intent = _detect_intent(request.question, query_entities)

    dense_results = pipeline["dense_retriever"].retrieve(
        request.question,
        top_k=DENSE_TOP_K,
        query_intent=intent,
    )

    entity_results = []
    if request.use_entity_retrieval and query_entities:
        entity_names = [entity["text"] for entity in query_entities]
        entity_results = pipeline["entity_retriever"].retrieve(
            entity_names,
            top_k=ENTITY_TOP_K,
        )

    fused_results = (
        reciprocal_rank_fusion(dense_results, entity_results)
        if entity_results
        else dense_results
    )

    top_k = request.top_k or RERANK_TOP_K
    reranked = (
        pipeline["reranker"].rerank(request.question, fused_results, top_k=top_k)
        if request.use_reranking and fused_results
        else fused_results[:top_k]
    )

    answer = (
        pipeline["generator"].run(
            query=request.question,
            passages=reranked,
            intent=intent,
            entities=query_entities,
        )
        if reranked
        else _empty_answer()
    )

    claim_mapping = pipeline["claim_mapper"].map_claims(answer, reranked)
    pipeline_confidence = pipeline["confidence_scorer"].compute(
        reranked,
        query_entities,
        claim_mapping,
    )

    answer_text_for_ner = " ".join(
        part for part in [answer.get("simple_answer", ""), answer.get("answer_text", "")]
        if part
    )
    answer_entities = pipeline["rule_ner"].extract(answer_text_for_ner)
    answer_entities += pipeline["ml_ner"].extract(answer_text_for_ner)
    answer_entities = _merge_entities(answer_entities, [])

    highlighted_sources = []
    for passage in reranked:
        passage_text = passage.get("text", "")
        passage_entities = _merge_entities(
            pipeline["rule_ner"].extract(passage_text),
            pipeline["ml_ner"].extract(passage_text),
        )
        highlighted_sources.append(
            {
                "chunk_id": passage.get("chunk_id", ""),
                "text": passage_text,
                "metadata": passage.get("metadata", {}),
                "score": passage.get("score", 0),
                "entities": passage_entities,
                "highlighted_text": pipeline["highlighter"].highlight(
                    passage_text,
                    passage_entities,
                ),
            }
        )

    response_data = {
        "answer": answer,
        "sources": highlighted_sources,
        "entities": answer_entities,
        "confidence": round(
            max(float(answer.get("confidence", 0.0)), pipeline_confidence),
            4,
        ),
        "explanation": {
            "claim_mapping": claim_mapping,
            "intent": intent,
            "num_dense_results": len(dense_results),
            "num_entity_results": len(entity_results),
            "num_fused_results": len(fused_results),
            "num_reranked": len(reranked),
            "method": "Dense + Entity -> Fusion -> Cross-Encoder -> Grounded Extractive Answering",
        },
        "query_entities": query_entities,
        "query_intent": intent,
        "processing_time": round(time.time() - start_time, 3),
    }

    _save_query_history(request.question, response_data)
    return QueryResponse(**response_data)


@app.get("/api/query/history", tags=["Query Pipeline"])
async def get_query_history():
    if QUERY_HISTORY_PATH.exists():
        return json.loads(QUERY_HISTORY_PATH.read_text())
    return []


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

"""
Legal RAG System - Main FastAPI Application
============================================
Explainable Retrieval-Augmented Generation with Named Entity Awareness
for Legal Document Analysis.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from config import LOG_LEVEL, LOG_FILE

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(str(LOG_FILE), rotation="10 MB", retention="7 days", level="DEBUG")

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal RAG System",
    description="Explainable Retrieval-Augmented Generation with Named Entity Awareness for Legal Document Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Import and Mount Routers ─────────────────────────────────────────────────
from ingestion.router import router as ingestion_router
from ner.router import router as ner_router
from retrieval.router import router as retrieval_router
from ranking.router import router as ranking_router
from generation.router import router as generation_router
from explainability.router import router as explainability_router
from evaluation.router import router as evaluation_router

app.include_router(ingestion_router, prefix="/api/ingestion", tags=["Ingestion"])
app.include_router(ner_router, prefix="/api/ner", tags=["NER"])
app.include_router(retrieval_router, prefix="/api/retrieval", tags=["Retrieval"])
app.include_router(ranking_router, prefix="/api/ranking", tags=["Ranking"])
app.include_router(generation_router, prefix="/api/generation", tags=["Generation"])
app.include_router(explainability_router, prefix="/api/explainability", tags=["Explainability"])
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["Evaluation"])


# ─── Startup & Shutdown Events ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Legal RAG System starting up...")
    logger.info("📦 Loading models (this may take a moment on first run)...")
    # Models are loaded lazily on first use to speed up startup


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Legal RAG System shutting down...")


# ─── Health Check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Legal RAG System"}


# ─── Query Endpoint (Main Pipeline) ───────────────────────────────────────────
from pydantic import BaseModel
from typing import Optional
import json
import time

from ner.rule_based import RuleBasedNER
from ner.ml_based import MLBasedNER
from ner.entity_index import EntityIndex
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.dense_retrieval import DenseRetriever
from retrieval.entity_retrieval import EntityRetriever
from retrieval.fusion import reciprocal_rank_fusion
from ranking.cross_encoder import CrossEncoderReranker
from generation.rag_generator import RAGGenerator
from generation.agentic_router import AgenticRouter
from explainability.claim_mapper import ClaimMapper
from explainability.confidence import ConfidenceScorer
from explainability.highlighter import EntityHighlighter
from config import RERANK_TOP_K, DENSE_TOP_K, ENTITY_TOP_K, QUERY_HISTORY_PATH


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_entity_retrieval: Optional[bool] = True
    use_reranking: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    sources: list
    entities: list
    confidence: float
    explanation: dict
    query_entities: list
    query_intent: str
    processing_time: float


# Lazy-loaded singletons
_pipeline_components = {}


def _get_pipeline():
    """Lazily initialize all pipeline components."""
    if not _pipeline_components:
        logger.info("Initializing pipeline components...")
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
        _pipeline_components["generator"] = AgenticRouter()
        _pipeline_components["claim_mapper"] = ClaimMapper(
            _pipeline_components["embedder"]
        )
        _pipeline_components["confidence_scorer"] = ConfidenceScorer()
        _pipeline_components["highlighter"] = EntityHighlighter()
        logger.info("✅ All pipeline components initialized")
    return _pipeline_components


def _detect_intent(question: str, entities: list) -> str:
    """Detect query intent based on keywords and structure."""
    q_lower = question.lower()
    if any(w in q_lower for w in ["what is", "define", "meaning of", "definition"]):
        return "definition"
    elif any(w in q_lower for w in ["how to", "procedure", "process", "steps"]):
        return "procedure"
    elif any(w in q_lower for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"
    elif any(w in q_lower for w in ["case", "judgment", "ruling", "verdict", "held"]):
        return "case_based"
    elif any(w in q_lower for w in ["who", "which court", "which judge"]):
        return "factual"
    else:
        return "general"


def _save_query_history(question: str, response: dict):
    """Save query and response to history file."""
    history = []
    if QUERY_HISTORY_PATH.exists():
        try:
            history = json.loads(QUERY_HISTORY_PATH.read_text())
        except Exception:
            history = []
    history.append({
        "question": question,
        "answer": response.get("answer", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": response.get("confidence", 0),
        "num_sources": len(response.get("sources", [])),
    })
    QUERY_HISTORY_PATH.write_text(json.dumps(history, indent=2))


@app.post("/api/query", response_model=QueryResponse, tags=["Query Pipeline"])
async def full_query_pipeline(request: QueryRequest):
    """
    Main query endpoint — runs the full RAG pipeline:
    1. NER on query
    2. Intent detection
    3. Hybrid retrieval (dense + entity)
    4. Cross-encoder reranking
    5. RAG generation
    6. Explainability (claim mapping, confidence, highlighting)
    """
    start_time = time.time()
    pipeline = _get_pipeline()

    # Step 1: Extract entities from query
    logger.info(f"📝 Query: {request.question}")
    rule_entities = pipeline["rule_ner"].extract(request.question)
    ml_entities = pipeline["ml_ner"].extract(request.question)
    # Merge entities (deduplicate)
    query_entities = _merge_entities(rule_entities, ml_entities)
    logger.info(f"🏷️ Query entities: {query_entities}")

    # Step 2: Detect intent
    intent = _detect_intent(request.question, query_entities)
    logger.info(f"🎯 Query intent: {intent}")

    # Step 3: Dense retrieval
    dense_results = pipeline["dense_retriever"].retrieve(
        request.question, top_k=DENSE_TOP_K
    )
    logger.info(f"🔍 Dense retrieval: {len(dense_results)} results")

    # Step 4: Entity-based retrieval (if enabled)
    entity_results = []
    if request.use_entity_retrieval and query_entities:
        entity_names = [e["text"] for e in query_entities]
        entity_results = pipeline["entity_retriever"].retrieve(
            entity_names, top_k=ENTITY_TOP_K
        )
        logger.info(f"🏷️ Entity retrieval: {len(entity_results)} results")

    # Step 5: Fusion
    if entity_results:
        fused_results = reciprocal_rank_fusion(dense_results, entity_results)
    else:
        fused_results = dense_results
    logger.info(f"🔀 Fused results: {len(fused_results)} chunks")

    # Step 6: Cross-encoder reranking (if enabled)
    top_k = request.top_k or RERANK_TOP_K
    if request.use_reranking and fused_results:
        reranked = pipeline["reranker"].rerank(
            request.question, fused_results, top_k=top_k
        )
        logger.info(f"📊 Reranked to top {len(reranked)} results")
    else:
        reranked = fused_results[:top_k]

    # Step 7: RAG Generation
    if reranked:
        answer = pipeline["generator"].run(
            query=request.question,
            passages=reranked,
            intent=intent,
            entities=query_entities,
        )
    else:
        answer = "No relevant passages were found in the document corpus to answer this question."
    logger.info(f"✍️ Answer generated ({len(answer)} chars)")

    # Step 8: Explainability
    # Claim → source mapping
    claim_mapping = pipeline["claim_mapper"].map_claims(answer, reranked)

    # Confidence scoring
    confidence = pipeline["confidence_scorer"].compute(
        reranked, query_entities, claim_mapping
    )

    # Entity highlighting
    answer_entities = pipeline["rule_ner"].extract(answer)
    answer_entities += pipeline["ml_ner"].extract(answer)
    answer_entities = _merge_entities(answer_entities, [])

    highlighted_sources = []
    for passage in reranked:
        passage_entities = pipeline["rule_ner"].extract(passage.get("text", ""))
        highlighted_sources.append({
            "text": passage.get("text", ""),
            "metadata": passage.get("metadata", {}),
            "score": passage.get("score", 0),
            "entities": passage_entities,
            "highlighted_text": pipeline["highlighter"].highlight(
                passage.get("text", ""), passage_entities
            ),
        })

    processing_time = time.time() - start_time

    response_data = {
        "answer": answer,
        "sources": highlighted_sources,
        "entities": answer_entities,
        "confidence": confidence,
        "explanation": {
            "claim_mapping": claim_mapping,
            "intent": intent,
            "num_dense_results": len(dense_results),
            "num_entity_results": len(entity_results),
            "num_fused_results": len(fused_results),
            "num_reranked": len(reranked),
            "method": "Hybrid (Dense + Entity) → RRF → Cross-Encoder → RAG",
        },
        "query_entities": query_entities,
        "query_intent": intent,
        "processing_time": round(processing_time, 3),
    }

    # Save to history
    _save_query_history(request.question, response_data)

    return QueryResponse(**response_data)


@app.get("/api/query/history", tags=["Query Pipeline"])
async def get_query_history():
    """Get all past queries and their results."""
    if QUERY_HISTORY_PATH.exists():
        return json.loads(QUERY_HISTORY_PATH.read_text())
    return []


def _merge_entities(list1: list, list2: list) -> list:
    """Merge two entity lists, deduplicating by text+label."""
    seen = set()
    merged = []
    for e in list1 + list2:
        key = (e.get("text", "").lower(), e.get("label", ""))
        if key not in seen:
            seen.add(key)
            merged.append(e)
    return merged


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

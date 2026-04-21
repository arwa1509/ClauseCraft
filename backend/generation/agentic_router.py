"""
Agentic Fallback Engine (LangGraph)
=====================================
A stateful agentic router that evaluates retrieval confidence and
automatically falls back to a Tavily web-search tool when confidence
drops below the configured threshold (default: 0.75).

Graph States:
    EVALUATE  → run confidence check on retrieved passages
    LOCAL     → answer from local FAISS index (high confidence path)
    WEB_SEARCH → query Tavily for supplemental case-law (low confidence path)
    GENERATE  → call RAGGenerator with the best context
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, TypedDict

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# LangGraph imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed – AgenticRouter will run in degraded mode.")

# ──────────────────────────────────────────────────────────────────────────────
# Tavily web-search tool
# ──────────────────────────────────────────────────────────────────────────────
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None
    TAVILY_AVAILABLE = False
    logger.warning("tavily-python not installed – web-search fallback disabled.")

# ──────────────────────────────────────────────────────────────────────────────
# Internal imports
# ──────────────────────────────────────────────────────────────────────────────
from generation.rag_generator import RAGGenerator

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = float(os.getenv("AGENTIC_CONFIDENCE_THRESHOLD", "0.75"))
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))


# ──────────────────────────────────────────────────────────────────────────────
# Graph State Schema
# ──────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """Mutable state object passed between LangGraph nodes."""
    query: str
    passages: list[dict]          # retrieved from local FAISS / entity index
    intent: str
    entities: list[dict]
    confidence: float             # score from confidence.py or rag_generator
    web_results: list[dict]       # results fetched from Tavily
    final_context: list[dict]    # merged passages used for generation
    answer: str                   # final JSON answer string
    route: str                    # "local" | "web_search"
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: derive confidence from passages
# ──────────────────────────────────────────────────────────────────────────────

def _derive_confidence(passages: list[dict]) -> float:
    """
    Compute a confidence score from the top-ranked passage.
    Replicates the sigmoid normalization used in rag_generator.py so the
    agentic layer is consistent with the generation layer.
    """
    import math
    if not passages:
        return 0.0
    raw = passages[0].get("score", 0.0)
    if isinstance(raw, (int, float)):
        if raw < 0 or raw > 1:
            return round(1 / (1 + math.exp(-raw)), 4)
        return round(float(raw), 4)
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph Node Functions
# ──────────────────────────────────────────────────────────────────────────────

def node_evaluate_confidence(state: AgentState) -> AgentState:
    """
    Node 1 – EVALUATE
    Compute confidence from the retrieved passages and decide routing.
    """
    passages = state.get("passages", [])
    confidence = _derive_confidence(passages)
    route = "local" if confidence >= CONFIDENCE_THRESHOLD else "web_search"

    logger.info(
        f"[AgentEval] confidence={confidence:.3f} threshold={CONFIDENCE_THRESHOLD} "
        f"→ route={route}"
    )
    return {**state, "confidence": confidence, "route": route}


def node_use_local_context(state: AgentState) -> AgentState:
    """
    Node 2 – LOCAL
    High-confidence path: use local FAISS passages as-is.
    """
    logger.info("[AgentLocal] Using local FAISS context.")
    return {**state, "final_context": state.get("passages", []), "web_results": []}


def node_web_search(state: AgentState) -> AgentState:
    """
    Node 3 – WEB_SEARCH
    Low-confidence path: query Tavily for up-to-date case law, then merge
    the web snippets with any local passages so the LLM gets cross-source context.
    """
    query = state["query"]
    
    # We only arrive here if confidence was below threshold (i.e. no relevant document match).
    logger.info(f"[AgentWeb] Triggering Tavily web search for: {query!r}")

    web_results: list[dict] = []

    if not TAVILY_AVAILABLE:
        logger.warning("[AgentWeb] Tavily not available – skipping web search.")
    elif not TAVILY_API_KEY:
        logger.warning("[AgentWeb] TAVILY_API_KEY not set – skipping web search.")
    else:
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(
                query=f"legal case law India: {query}",
                max_results=WEB_SEARCH_MAX_RESULTS,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )

            for item in response.get("results", []):
                web_results.append({
                    "text": item.get("content", ""),
                    "score": item.get("score", 0.5),
                    "metadata": {
                        "source": "tavily_web",
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "doc_name": item.get("title", "Web Result"),
                    },
                    "entities": [],
                })

            logger.info(f"[AgentWeb] Fetched {len(web_results)} web results.")
        except Exception as exc:
            logger.error(f"[AgentWeb] Tavily search failed: {exc}")

    # Merge: web results first (fresher), then local passages as fallback
    local_passages = state.get("passages", [])
    merged = web_results + local_passages

    # Since we are fetching from Web, let's flag the system to not strictly filter 
    # it out downstream as 'not in context'. We assign intent='external_definition' 
    # to help the generator.
    if web_results:
         state["intent"] = "external_web"

    return {**state, "web_results": web_results, "final_context": merged}


def node_generate_answer(state: AgentState) -> AgentState:
    """
    Node 4 – GENERATE
    Call RAGGenerator with the final_context (either local or web-augmented).
    """
    generator = RAGGenerator()
    answer = generator.generate(
        query=state["query"],
        passages=state["final_context"],
        intent=state.get("intent", "general"),
        entities=state.get("entities", []),
    )

    # Patch confidence and source information into the JSON answer
    try:
        answer_dict = json.loads(answer)
        answer_dict["confidence"] = state["confidence"]
        answer_dict["source"] = state["route"]
        answer_dict["web_augmented"] = bool(state.get("web_results"))
        answer = json.dumps(answer_dict)
    except (json.JSONDecodeError, TypeError):
        pass  # answer stays as-is if it's not valid JSON

    logger.info(
        f"[AgentGen] Answer generated. source={state['route']} "
        f"web_results={len(state.get('web_results', []))}"
    )
    return {**state, "answer": answer}


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge: decide which context path to take
# ──────────────────────────────────────────────────────────────────────────────

def route_decision(state: AgentState) -> str:
    """Return the next node name based on the route field."""
    return state.get("route", "local")


# ──────────────────────────────────────────────────────────────────────────────
# Build the LangGraph StateGraph
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph():
    """Construct and compile the LangGraph agent graph."""
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("evaluate", node_evaluate_confidence)
    graph.add_node("local", node_use_local_context)
    graph.add_node("web_search", node_web_search)
    graph.add_node("generate", node_generate_answer)

    # Entry point
    graph.set_entry_point("evaluate")

    # Conditional routing after evaluation
    graph.add_conditional_edges(
        "evaluate",
        route_decision,
        {
            "local": "local",
            "web_search": "web_search",
        },
    )

    # Both context paths converge at generate node
    graph.add_edge("local", "generate")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Module-level compiled graph (lazy init to avoid import-time failures)
_COMPILED_GRAPH = None


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class AgenticRouter:
    """
    High-level wrapper around the LangGraph agent.

    Usage::

        router = AgenticRouter()
        answer_json = router.run(
            query="What is the penalty for murder under IPC?",
            passages=[...],          # from your dual-pipeline retriever
            intent="penalty_query",
            entities=[...],
        )
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self._graph = _get_graph()

    def run(
        self,
        query: str,
        passages: list[dict],
        intent: str = "general",
        entities: Optional[list[dict]] = None,
    ) -> str:
        """
        Execute the agentic graph and return the answer as a JSON string.

        Falls back to a direct RAGGenerator call if LangGraph is unavailable.
        """
        if entities is None:
            entities = []

        if self._graph is None:
            logger.warning(
                "LangGraph not available – using direct RAGGenerator (no agentic routing)."
            )
            generator = RAGGenerator()
            return generator.generate(
                query=query,
                passages=passages,
                intent=intent,
                entities=entities,
            )

        initial_state: AgentState = {
            "query": query,
            "passages": passages,
            "intent": intent,
            "entities": entities,
            "confidence": 0.0,
            "web_results": [],
            "final_context": [],
            "answer": "",
            "route": "local",
            "error": None,
        }

        try:
            final_state = self._graph.invoke(initial_state)
            return final_state["answer"]
        except Exception as exc:
            logger.error(f"[AgenticRouter] Graph execution failed: {exc}")
            # Graceful degradation
            generator = RAGGenerator()
            return generator.generate(
                query=query,
                passages=passages,
                intent=intent,
                entities=entities,
            )

    def get_route_info(
        self,
        query: str,
        passages: list[dict],
    ) -> dict[str, Any]:
        """
        Introspect what route the agent would select without running generation.
        Useful for explainability dashboards.
        """
        confidence = _derive_confidence(passages)
        route = "local" if confidence >= self.confidence_threshold else "web_search"
        return {
            "confidence": confidence,
            "threshold": self.confidence_threshold,
            "route": route,
            "web_augmented": route == "web_search",
        }

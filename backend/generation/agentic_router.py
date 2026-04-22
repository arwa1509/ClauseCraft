"""
Agentic routing for grounded answer generation.

The system remains corpus-grounded by default. Optional web fallback is
available only when explicitly enabled by configuration.

Each AgenticRouter instance compiles its own LangGraph with the embedder
baked in via closure, so RAGGenerator gets semantic scoring on every call.
"""

from __future__ import annotations

import os
from typing import Any, Optional, TypedDict

from loguru import logger
from dotenv import load_dotenv

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed; agentic routing will run in degraded mode.")

try:
    from tavily import TavilyClient

    TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None
    TAVILY_AVAILABLE = False
    logger.warning("tavily-python not installed; web fallback is unavailable.")

from generation.rag_generator import RAGGenerator

_CASE_INTENTS = frozenset({"case_based", "reasoning", "factual"})


class AgentState(TypedDict):
    query: str
    passages: list[dict]
    intent: str
    entities: list[dict]
    confidence: float
    web_results: list[dict]
    final_context: list[dict]
    answer: dict
    route: str
    error: Optional[str]


def _load_runtime_settings() -> dict[str, Any]:
    load_dotenv(override=True)
    return {
        "confidence_threshold": float(os.getenv("AGENTIC_CONFIDENCE_THRESHOLD", "0.75")),
        "web_search_max_results": int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", "").strip(),
        "web_fallback_enabled": os.getenv(
            "AGENTIC_WEB_FALLBACK_ENABLED", "false"
        ).lower() == "true",
    }


def _derive_confidence(passages: list[dict]) -> float:
    if not passages:
        return 0.0
    top_passages = passages[:3]
    weights = [0.6, 0.3, 0.1]
    score = 0.0
    total_weight = 0.0
    for weight, passage in zip(weights, top_passages):
        score += weight * float(passage.get("score", 0.0))
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return round(min(max(score / total_weight, 0.0), 1.0), 4)


def node_evaluate_confidence(state: AgentState) -> AgentState:
    settings = _load_runtime_settings()
    confidence = _derive_confidence(state.get("passages", []))
    route = "local"
    if settings["web_fallback_enabled"] and confidence < settings["confidence_threshold"]:
        route = "web_search"
    logger.info(
        f"[AgentEval] confidence={confidence:.3f} "
        f"threshold={settings['confidence_threshold']:.3f} route={route}"
    )
    return {**state, "confidence": confidence, "route": route}


def node_use_local_context(state: AgentState) -> AgentState:
    return {**state, "final_context": state.get("passages", []), "web_results": []}


def node_web_search(state: AgentState) -> AgentState:
    settings = _load_runtime_settings()
    query = state["query"]
    web_results: list[dict] = []

    if not settings["web_fallback_enabled"]:
        return {
            **state,
            "web_results": [],
            "final_context": state.get("passages", []),
            "error": "web_fallback_disabled",
        }

    if not TAVILY_AVAILABLE or not settings["tavily_api_key"]:
        logger.warning("[AgentWeb] Web fallback requested but Tavily is unavailable.")
        return {
            **state,
            "web_results": [],
            "final_context": state.get("passages", []),
            "error": "tavily_not_configured",
        }

    try:
        client = TavilyClient(api_key=settings["tavily_api_key"])
        suffix = " case law judgment" if state.get("intent") in _CASE_INTENTS else " legal"
        response = client.search(
            query=f"{query}{suffix}",
            max_results=settings["web_search_max_results"],
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )
        for item in response.get("results", []):
            web_results.append({
                "chunk_id": item.get("url", ""),
                "text": item.get("content", ""),
                "score": min(max(float(item.get("score", 0.5)), 0.0), 1.0),
                "metadata": {
                    "doc_name": item.get("title", "Web Result"),
                    "source": "tavily_web",
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                },
                "entities": [],
            })
    except Exception as exc:
        logger.error(f"[AgentWeb] Tavily search failed: {exc}")
        return {
            **state,
            "web_results": [],
            "final_context": state.get("passages", []),
            "error": f"web_search_failed: {exc}",
        }

    merged = web_results + state.get("passages", [])
    return {
        **state,
        "web_results": web_results,
        "final_context": merged,
        "intent": "external_web" if web_results else state.get("intent", "general"),
        "error": None if web_results else "web_search_returned_no_results",
    }


def route_decision(state: AgentState) -> str:
    return state.get("route", "local")


class AgenticRouter:
    """
    Orchestrates retrieval → generation with an optional web fallback.

    Pass ``embedder`` to enable semantic sentence scoring inside RAGGenerator.
    The graph is compiled once per instance with the embedder captured in a
    closure, so no global state is needed.
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        embedder=None,
    ):
        settings = _load_runtime_settings()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings["confidence_threshold"]
        )
        self.embedder = embedder
        self._graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def _build_graph(self):
        embedder = self.embedder  # captured in closure — do not rename

        def node_generate_answer(state: AgentState) -> AgentState:
            generator = RAGGenerator(embedder=embedder)
            answer = generator.generate(
                query=state["query"],
                passages=state["final_context"],
                intent=state.get("intent", "general"),
                entities=state.get("entities", []),
            )
            answer["router_confidence"] = state["confidence"]
            answer["source"] = state["route"]
            answer["web_augmented"] = bool(state.get("web_results"))
            answer["web_results_count"] = len(state.get("web_results", []))
            answer["web_error"] = state.get("error")
            return {**state, "answer": answer}

        graph = StateGraph(AgentState)
        graph.add_node("evaluate", node_evaluate_confidence)
        graph.add_node("local", node_use_local_context)
        graph.add_node("web_search", node_web_search)
        graph.add_node("generate", node_generate_answer)
        graph.set_entry_point("evaluate")
        graph.add_conditional_edges(
            "evaluate",
            route_decision,
            {"local": "local", "web_search": "web_search"},
        )
        graph.add_edge("local", "generate")
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def run(
        self,
        query: str,
        passages: list[dict],
        intent: str = "general",
        entities: Optional[list[dict]] = None,
    ) -> dict:
        entities = entities or []

        if self._graph is None:
            generator = RAGGenerator(embedder=self.embedder)
            answer = generator.generate(
                query=query,
                passages=passages,
                intent=intent,
                entities=entities,
            )
            answer["router_confidence"] = _derive_confidence(passages)
            answer["source"] = "local"
            answer["web_augmented"] = False
            answer["web_results_count"] = 0
            answer["web_error"] = "langgraph_unavailable"
            return answer

        initial_state: AgentState = {
            "query": query,
            "passages": passages,
            "intent": intent,
            "entities": entities,
            "confidence": 0.0,
            "web_results": [],
            "final_context": [],
            "answer": {},
            "route": "local",
            "error": None,
        }

        try:
            final_state = self._graph.invoke(initial_state)
            return final_state["answer"]
        except Exception as exc:
            logger.error(f"[AgenticRouter] Graph execution failed: {exc}")
            generator = RAGGenerator(embedder=self.embedder)
            answer = generator.generate(
                query=query,
                passages=passages,
                intent=intent,
                entities=entities,
            )
            answer["router_confidence"] = _derive_confidence(passages)
            answer["source"] = "local"
            answer["web_augmented"] = False
            answer["web_results_count"] = 0
            answer["web_error"] = f"graph_execution_failed: {exc}"
            return answer

    def get_route_info(self, query: str, passages: list[dict]) -> dict[str, Any]:
        settings = _load_runtime_settings()
        confidence = _derive_confidence(passages)
        route = (
            "web_search"
            if settings["web_fallback_enabled"] and confidence < self.confidence_threshold
            else "local"
        )
        return {
            "query": query,
            "confidence": confidence,
            "threshold": self.confidence_threshold,
            "route": route,
            "web_augmented": route == "web_search",
            "web_fallback_enabled": settings["web_fallback_enabled"],
            "tavily_configured": bool(settings["tavily_api_key"]),
        }

"""
Legal Prompt Templates
=======================
Structured prompts for different query intents and legal reasoning tasks.
"""


def get_system_prompt() -> str:
    """Base system prompt for the legal RAG assistant."""
    return """You are a precise legal assistant that answers questions based ONLY on the provided context passages. 

CRITICAL RULES:
1. Answer ONLY using information from the provided context passages.
2. If the context does not contain enough information, say "Based on the provided documents, I cannot find sufficient information to answer this question."
3. NEVER fabricate legal citations, section numbers, case names, or any legal information.
4. Always cite which passage(s) support each claim in your answer.
5. Use precise legal terminology.
6. Structure your answer clearly with proper formatting.
7. If multiple passages provide relevant information, synthesize them coherently.
8. Distinguish between direct quotes and your interpretations."""


def get_qa_prompt(
    query: str,
    passages: list[dict],
    intent: str = "general",
    entities: list[dict] = None,
) -> str:
    """
    Build the full prompt for RAG generation.

    Args:
        query: User's question
        passages: Retrieved and reranked passages
        intent: Detected query intent
        entities: Entities extracted from query
    """
    # Format context passages
    context_parts = []
    for i, passage in enumerate(passages, 1):
        meta = passage.get("metadata", {})
        source = meta.get("doc_name", "Unknown")
        section = meta.get("section", "")
        page = meta.get("page_num", "")

        source_info = f"[Source: {source}"
        if section:
            source_info += f", {section}"
        if page:
            source_info += f", Page {page}"
        source_info += "]"

        context_parts.append(
            f"--- Passage {i} {source_info} ---\n{passage.get('text', '')}"
        )

    context = "\n\n".join(context_parts)

    # Intent-specific instructions
    intent_instructions = _get_intent_instructions(intent)

    # Entity context
    entity_context = ""
    if entities:
        entity_strs = [f"{e['text']} ({e['label']})" for e in entities]
        entity_context = f"\nKey entities identified in the query: {', '.join(entity_strs)}\n"

    prompt = f"""{intent_instructions}
{entity_context}
CONTEXT PASSAGES:
{context}

QUESTION: {query}

ANSWER (cite passage numbers, e.g., [Passage 1]):"""

    return prompt


def _get_intent_instructions(intent: str) -> str:
    """Get intent-specific instructions for the LLM."""
    instructions = {
        "definition": (
            "The user is asking for a legal definition or meaning. "
            "Provide a clear, precise definition as found in the context. "
            "Include the source statute or provision if mentioned."
        ),
        "procedure": (
            "The user is asking about a legal procedure or process. "
            "Provide step-by-step information as described in the context. "
            "Include relevant section numbers and requirements."
        ),
        "comparison": (
            "The user is asking to compare legal concepts, provisions, or cases. "
            "Structure your answer to clearly highlight similarities and differences. "
            "Use the context to support each comparison point."
        ),
        "case_based": (
            "The user is asking about a legal case, judgment, or ruling. "
            "Provide case details including parties, court, key holdings, and ratio decidendi "
            "as found in the context."
        ),
        "factual": (
            "The user is asking a factual legal question. "
            "Provide a direct, accurate answer based on the context. "
            "Cite specific provisions, sections, or case references."
        ),
        "general": (
            "Answer the legal question comprehensively using the provided context. "
            "Structure your answer clearly and cite sources."
        ),
    }
    return instructions.get(intent, instructions["general"])


def get_claim_extraction_prompt(answer: str) -> str:
    """Prompt to extract individual claims from an answer."""
    return f"""Extract each distinct factual claim from the following legal answer.
Return each claim as a separate line.
Only extract factual claims, not opinions or hedging language.

Answer:
{answer}

Claims (one per line):"""


def get_summarization_prompt(text: str, max_length: int = 200) -> str:
    """Prompt for legal text summarization."""
    return f"""Summarize the following legal text in {max_length} words or less.
Focus on key legal provisions, parties, and outcomes.
Maintain legal precision.

Text:
{text}

Summary:"""

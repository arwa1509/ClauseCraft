"""
Grounded answer generator for legal retrieval results.

Sentences are scored using a blend of semantic similarity (bi-encoder
embeddings) and lexical overlap, so the generator understands meaning
rather than relying on exact term matches.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from loguru import logger

from sentence_utils import split_sentences


class RAGGenerator:
    """
    Build grounded answers from retrieved passages.

    When an embedder is supplied, sentence selection is driven primarily by
    semantic similarity to the query (65%) with lexical overlap as a
    secondary signal (35%). Without an embedder it falls back to lexical-only.
    """

    def __init__(self, provider: Optional[str] = None, embedder=None):
        self.provider = provider or "extractive"
        self.embedder = embedder

    def generate(
        self,
        query: str,
        passages: list[dict],
        intent: str = "general",
        entities: Optional[list[dict]] = None,
        max_sentences: int = 5,
    ) -> dict:
        if not passages:
            return {
                "simple_answer": (
                    "I could not find enough support in the indexed documents to "
                    "answer this question."
                ),
                "markdown_answer": (
                    "I could not find enough support in the indexed documents to answer "
                    "this question."
                ),
                "answer_text": "",
                "supporting_passages": [],
                "key_entities": [],
                "citations": [],
                "evidence_points": [],
                "answer_segments": [],
                "confidence": 0.0,
                "answer_type": "insufficient_context",
            }

        entities = entities or []
        query_terms = self._query_terms(query, entities)

        # ── Pre-compute query embedding once ──────────────────────────────────
        query_emb = None
        if self.embedder is not None:
            try:
                query_emb = self.embedder.embed(query)
            except Exception as exc:
                logger.warning(f"Query embedding failed, falling back to lexical: {exc}")

        # ── Collect all candidate sentences with position tracking ────────────
        all_sentence_data: list[dict] = []
        for passage_rank, passage in enumerate(passages[:5], start=1):
            text = passage.get("text", "")
            for sent_idx, sentence in enumerate(self._split_sentences(text)):
                all_sentence_data.append({
                    "text": sentence,
                    "passage": passage,
                    "passage_rank": passage_rank,
                    "sent_idx": sent_idx,
                })

        # ── Batch semantic scoring (one forward pass for all sentences) ────────
        semantic_scores = [0.0] * len(all_sentence_data)
        use_semantic = False
        if query_emb is not None and all_sentence_data:
            try:
                sent_texts = [s["text"] for s in all_sentence_data]
                sent_embs = self.embedder.embed_batch(sent_texts, show_progress=False)
                raw = np.dot(sent_embs, query_emb)
                semantic_scores = [float(max(0.0, s)) for s in raw]
                use_semantic = True
                logger.debug(
                    "Semantic sentence scoring: %d sentences, top sim=%.3f",
                    len(sent_texts),
                    max(semantic_scores) if semantic_scores else 0.0,
                )
            except Exception as exc:
                logger.warning(f"Batch semantic scoring failed: {exc}")

        # ── Score and filter ───────────────────────────────────────────────────
        candidate_sentences: list[dict] = []
        for i, sent_data in enumerate(all_sentence_data):
            score = self._sentence_score(
                sent_data["text"],
                query_terms,
                intent,
                semantic_sim=semantic_scores[i],
                use_semantic=use_semantic,
            )
            if score <= 0:
                continue
            candidate_sentences.append({
                **sent_data,
                "score": round(min(max(score, 0.0), 1.2), 4),
            })

        candidate_sentences.sort(
            key=lambda item: (
                item["score"],
                item["passage"].get("score", 0),
                -item["passage_rank"],
            ),
            reverse=True,
        )

        selected = self._select_diverse_sentences(candidate_sentences, max_sentences)
        best_passages = self._select_supporting_passages(passages)

        if not selected:
            best_passage = passages[0]
            fallback_text = self._clean_sentence(best_passage.get("text", ""))[:350]
            if fallback_text and not fallback_text.endswith("."):
                fallback_text += "."
            confidence = min(max(best_passage.get("score", 0.0), 0.0), 1.0)
            return {
                "simple_answer": (
                    "The top retrieved passage may be relevant, but I do not have "
                    "enough sentence-level support to answer confidently."
                ),
                "markdown_answer": (
                    "The top retrieved passage may be relevant, but I do not have enough "
                    "sentence-level support to answer confidently.\n\n"
                    "## Best Available Passage\n"
                    f"> {fallback_text}"
                ),
                "answer_text": fallback_text,
                "supporting_passages": best_passages,
                "key_entities": self._collect_entities(passages[:2]),
                "citations": self._citations_from_passages(best_passages),
                "evidence_points": [
                    {
                        "text": fallback_text,
                        "citation_ids": [1] if best_passages else [],
                        "chunk_id": best_passage.get("chunk_id", ""),
                    }
                ],
                "answer_segments": [],
                "confidence": round(confidence, 4),
                "answer_type": "weak_support",
            }

        ordered_sentences = self._restore_original_order(selected)
        answer_text = " ".join(item["text"] for item in ordered_sentences).strip()
        simple_answer = self._format_simple_answer(answer_text)

        cited_passages: list[dict] = []
        seen_ids: set[str] = set()
        for item in ordered_sentences:
            chunk_id = item["passage"].get("chunk_id", "")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                cited_passages.append(item["passage"])
        if not cited_passages:
            cited_passages = passages[:1]

        citations = self._citations_from_passages(cited_passages)
        citation_ids_by_chunk = {
            citation["chunk_id"]: citation["id"] for citation in citations if citation.get("chunk_id")
        }
        answer_segments = self._build_answer_segments(ordered_sentences, citation_ids_by_chunk)
        evidence_points = self._build_evidence_points(ordered_sentences, citation_ids_by_chunk)
        markdown_answer = self._build_markdown_answer(
            query=query,
            answer_text=answer_text,
            answer_segments=answer_segments,
            evidence_points=evidence_points,
            citations=citations,
        )
        confidence = self._estimate_confidence(ordered_sentences, cited_passages)

        result = {
            "simple_answer": simple_answer,
            "markdown_answer": markdown_answer,
            "answer_text": answer_text,
            "supporting_passages": self._select_supporting_passages(cited_passages),
            "key_entities": self._collect_entities(cited_passages),
            "citations": citations,
            "evidence_points": evidence_points,
            "answer_segments": answer_segments,
            "confidence": confidence,
            "answer_type": "grounded_extractive",
        }
        logger.debug(
            "Generated grounded answer: %d citation(s), confidence=%.3f, semantic=%s",
            len(result["citations"]),
            result["confidence"],
            use_semantic,
        )
        return result

    # ── Scoring helpers ────────────────────────────────────────────────────────

    def _sentence_score(
        self,
        sentence: str,
        query_terms: set[str],
        intent: str,
        semantic_sim: float = 0.0,
        use_semantic: bool = False,
    ) -> float:
        sent_terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\.\-/]*", sentence.lower()))
        if not sent_terms:
            return 0.0

        overlap = len(query_terms & sent_terms)
        coverage = overlap / max(len(query_terms), 1)
        precision = overlap / max(len(sent_terms), 1)
        lexical = (0.7 * coverage) + (0.3 * precision)

        if use_semantic:
            score = 0.35 * lexical + 0.65 * semantic_sim
        else:
            score = lexical

        sentence_lower = sentence.lower()
        if intent == "definition" and any(
            m in sentence_lower for m in {"means", "refers to", "defined as"}
        ):
            score += 0.12
        if intent == "section" and "section" in sentence_lower:
            score += 0.12
        if intent == "condition" and any(
            m in sentence_lower for m in {"if", "provided that", "subject to", "shall"}
        ):
            score += 0.12
        if intent == "reasoning" and any(
            m in sentence_lower for m in {"because", "therefore", "since", "held"}
        ):
            score += 0.12

        if len(sentence) < 25:
            score -= 0.1
        if len(sentence) > 400:
            score -= 0.08
        if sentence_lower.startswith("prayer") or "it is therefore prayed" in sentence_lower:
            score -= 0.4

        return round(min(max(score, 0.0), 1.2), 4)

    def _query_terms(self, query: str, entities: list[dict]) -> set[str]:
        stop_words = {
            "a", "an", "and", "are", "be", "by", "can", "does", "for", "how",
            "if", "in", "is", "it", "of", "on", "or", "the", "to", "what",
            "when", "which", "who", "why",
        }
        tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\.\-/]*", query.lower()))
        tokens -= stop_words
        for entity in entities:
            entity_text = entity.get("text", "").lower()
            tokens.update(
                t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\.\-/]*", entity_text)
                if t not in stop_words
            )
        return tokens

    def _split_sentences(self, text: str) -> list[str]:
        clean = re.sub(r"\s+", " ", text).strip()
        return [
            self._clean_sentence(piece)
            for piece in split_sentences(clean)
            if self._clean_sentence(piece)
        ]

    def _clean_sentence(self, sentence: str) -> str:
        sentence = re.sub(r"^\d+\.\s*", "", sentence.strip())
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.strip(" -")

    def _select_diverse_sentences(
        self, candidates: list[dict], max_sentences: int
    ) -> list[dict]:
        selected: list[dict] = []
        seen_texts: set[str] = set()
        for candidate in candidates:
            normalized = candidate["text"].lower()
            if normalized in seen_texts:
                continue
            if any(self._near_duplicate(normalized, item["text"].lower()) for item in selected):
                continue
            selected.append(candidate)
            seen_texts.add(normalized)
            if len(selected) >= max_sentences:
                break
        return selected

    def _near_duplicate(self, left: str, right: str) -> bool:
        left_terms = set(left.split())
        right_terms = set(right.split())
        if not left_terms or not right_terms:
            return False
        return len(left_terms & right_terms) / len(left_terms | right_terms) > 0.8

    def _restore_original_order(self, selected: list[dict]) -> list[dict]:
        return sorted(selected, key=lambda item: (item["passage_rank"], item["sent_idx"]))

    def _format_simple_answer(self, answer_text: str) -> str:
        if not answer_text:
            return "I could not extract a supported answer from the retrieved passages."
        answer_text = answer_text.strip()
        if not answer_text.endswith("."):
            answer_text += "."
        return answer_text

    def _build_answer_segments(
        self,
        ordered_sentences: list[dict],
        citation_ids_by_chunk: dict[str, int],
    ) -> list[dict]:
        segments = []
        for item in ordered_sentences:
            chunk_id = item["passage"].get("chunk_id", "")
            citation_id = citation_ids_by_chunk.get(chunk_id)
            segments.append(
                {
                    "text": item["text"],
                    "citation_ids": [citation_id] if citation_id else [],
                    "chunk_id": chunk_id,
                    "score": round(float(item.get("score", 0.0)), 4),
                    "page_num": item["passage"].get("metadata", {}).get("page_num"),
                    "section": item["passage"].get("metadata", {}).get("section"),
                }
            )
        return segments

    def _build_evidence_points(
        self,
        ordered_sentences: list[dict],
        citation_ids_by_chunk: dict[str, int],
    ) -> list[dict]:
        evidence_points = []
        for item in ordered_sentences:
            chunk_id = item["passage"].get("chunk_id", "")
            citation_id = citation_ids_by_chunk.get(chunk_id)
            evidence_points.append(
                {
                    "text": item["text"],
                    "citation_ids": [citation_id] if citation_id else [],
                    "chunk_id": chunk_id,
                    "source": item["passage"].get("metadata", {}).get("source", "local"),
                    "page_num": item["passage"].get("metadata", {}).get("page_num"),
                    "section": item["passage"].get("metadata", {}).get("section"),
                }
            )
        return evidence_points

    def _build_markdown_answer(
        self,
        query: str,
        answer_text: str,
        answer_segments: list[dict],
        evidence_points: list[dict],
        citations: list[dict],
    ) -> str:
        del query
        if not answer_segments:
            return answer_text

        summary_lines = []
        for segment in answer_segments:
            refs = " ".join(f"[{cid}]" for cid in segment.get("citation_ids", []))
            summary_lines.append(f"{segment['text']} {refs}".strip())

        evidence_lines = []
        for point in evidence_points:
            refs = " ".join(f"[{cid}]" for cid in point.get("citation_ids", []))
            location = []
            if point.get("section"):
                location.append(str(point["section"]))
            if point.get("page_num") is not None:
                location.append(f"p.{point['page_num']}")
            location_text = f" ({', '.join(location)})" if location else ""
            evidence_lines.append(f"- {point['text']} {refs}{location_text}".strip())

        citation_lines = []
        for citation in citations:
            parts = [f"[{citation['id']}] {citation.get('doc_name', 'Unknown')}"]
            if citation.get("section"):
                parts.append(f"section {citation['section']}")
            if citation.get("page_num") is not None:
                parts.append(f"page {citation['page_num']}")
            if citation.get("source"):
                parts.append(f"source {citation['source']}")
            citation_lines.append(f"- {', '.join(parts)}")

        return "\n\n".join(
            [
                "## Answer",
                " ".join(summary_lines),
                "## Evidence",
                "\n".join(evidence_lines),
                "## Citations",
                "\n".join(citation_lines),
            ]
        )

    def _select_supporting_passages(self, passages: list[dict]) -> list[dict]:
        result: list[dict] = []
        seen: set[str] = set()
        for passage in passages[:3]:
            chunk_id = passage.get("chunk_id", "")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            result.append({
                "chunk_id": chunk_id,
                "text": passage.get("text", ""),
                "metadata": passage.get("metadata", {}),
                "score": round(float(passage.get("score", 0.0)), 4),
            })
        return result

    def _collect_entities(self, passages: list[dict]) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()
        for passage in passages:
            for entity in passage.get("entities", []):
                text = entity.get("text", "").strip()
                label = entity.get("label", "").strip()
                if not text:
                    continue
                value = f"{text} ({label})" if label else text
                if value.lower() not in seen:
                    seen.add(value.lower())
                    entities.append(value)
        return entities[:8]

    def _citations_from_passages(self, passages: list[dict]) -> list[dict]:
        citations = []
        for idx, passage in enumerate(passages, start=1):
            meta = passage.get("metadata", {})
            citations.append({
                "id": idx,
                "chunk_id": passage.get("chunk_id", ""),
                "doc_name": meta.get("doc_name", "Unknown"),
                "page_num": meta.get("page_num"),
                "section": meta.get("section"),
                "source": meta.get("source", "local"),
                "url": meta.get("url"),
            })
        return citations

    def _estimate_confidence(self, sentences: list[dict], passages: list[dict]) -> float:
        if not sentences:
            return 0.0
        sentence_score = sum(item["score"] for item in sentences) / len(sentences)
        passage_score = sum(float(p.get("score", 0.0)) for p in passages) / len(passages)
        confidence = (0.55 * min(max(sentence_score, 0.0), 1.0)) + (
            0.45 * min(max(passage_score, 0.0), 1.0)
        )
        return round(min(max(confidence, 0.0), 1.0), 4)

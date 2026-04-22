"""
Claim-to-evidence mapping utilities.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from sentence_utils import split_sentences


class ClaimMapper:
    def __init__(self, embedder=None):
        self.embedder = embedder

    def map_claims(
        self,
        answer: str | dict,
        source_passages: list[dict],
    ) -> list[dict]:
        if not answer or not source_passages:
            return []

        claims = self._extract_claims(answer)
        if not claims:
            return []

        mappings = []
        for claim in claims:
            best_match = self._find_best_source(claim, source_passages)
            if best_match:
                mappings.append(best_match)

        logger.debug(f"Mapped {len(mappings)} claims to source passages")
        return mappings

    def _extract_claims(self, answer: str | dict) -> list[str]:
        if isinstance(answer, dict):
            answer = " ".join(
                part
                for part in [answer.get("simple_answer", ""), answer.get("answer_text", "")]
                if part
            )

        clean = re.sub(r"\*\*.*?\*\*", "", answer)
        clean = re.sub(r"\[Passage \d+\]", "", clean)
        clean = re.sub(r"\[.*?\]", "", clean)

        claims = []
        for sentence in split_sentences(clean):
            if len(sentence) > 20 and not sentence.startswith("Note:") and not sentence.startswith("*"):
                claims.append(sentence)
        return claims

    def _find_best_source(
        self,
        claim: str,
        source_passages: list[dict],
    ) -> Optional[dict]:
        if self.embedder:
            return self._find_best_source_semantic(claim, source_passages)
        return self._find_best_source_lexical(claim, source_passages)

    def _find_best_source_semantic(
        self,
        claim: str,
        source_passages: list[dict],
    ) -> Optional[dict]:
        try:
            passage_texts = [passage.get("text", "") for passage in source_passages]
            similarities = self.embedder.similarity_batch(claim, passage_texts)
            if not similarities:
                return None

            best_idx = max(range(len(similarities)), key=lambda idx: similarities[idx])
            best_sim = float(similarities[best_idx])
            if best_sim < 0.1:
                return None

            best_passage = source_passages[best_idx]
            return {
                "claim": claim,
                "source_passage_idx": best_idx,
                "source_chunk_id": best_passage.get("chunk_id", ""),
                "similarity": round(best_sim, 4),
                "source_text": best_passage.get("text", "")[:300],
                "evidence_snippet": self._find_evidence_snippet(claim, best_passage.get("text", "")),
                "source_metadata": best_passage.get("metadata", {}),
            }
        except Exception as exc:
            logger.warning(f"Semantic claim mapping failed: {exc}")
            return self._find_best_source_lexical(claim, source_passages)

    def _find_best_source_lexical(
        self,
        claim: str,
        source_passages: list[dict],
    ) -> Optional[dict]:
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "of",
            "to",
            "for",
            "and",
            "or",
            "not",
            "with",
            "that",
            "this",
            "on",
            "at",
            "by",
            "from",
            "as",
            "be",
            "it",
        }
        claim_words = set(claim.lower().split()) - stop_words
        if not claim_words:
            return None

        best_idx = -1
        best_score = 0.0
        for idx, passage in enumerate(source_passages):
            passage_words = set(passage.get("text", "").lower().split()) - stop_words
            if not passage_words:
                continue
            score = len(claim_words & passage_words) / len(claim_words | passage_words)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0 or best_score < 0.05:
            return None

        best_passage = source_passages[best_idx]
        return {
            "claim": claim,
            "source_passage_idx": best_idx,
            "source_chunk_id": best_passage.get("chunk_id", ""),
            "similarity": round(best_score, 4),
            "source_text": best_passage.get("text", "")[:300],
            "evidence_snippet": self._find_evidence_snippet(claim, best_passage.get("text", "")),
            "source_metadata": best_passage.get("metadata", {}),
        }

    def _find_evidence_snippet(self, claim: str, passage_text: str, window: int = 200) -> str:
        if not passage_text:
            return ""

        claim_words = set(claim.lower().split())
        best_sentence = ""
        best_overlap = 0
        for sentence in split_sentences(passage_text):
            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words & sentence_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()
        return best_sentence[:window] if best_sentence else passage_text[:window]

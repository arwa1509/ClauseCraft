"""
Claim-to-Evidence Mapper
=========================
Maps individual claims in the generated answer to source passages.
Enables claim-level verification and explainability.
"""

import re
from typing import Optional
from loguru import logger


class ClaimMapper:
    """
    Maps each claim/sentence in the answer to supporting evidence passages.
    Uses semantic similarity to find the best matching passage for each claim.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder

    def map_claims(
        self,
        answer: str,
        source_passages: list[dict],
    ) -> list[dict]:
        """
        Split the answer into claims and map each to source passages.

        Args:
            answer: Generated answer text
            source_passages: List of source passage dicts

        Returns:
            List of claim mappings:
            [{"claim": str, "source_passage_idx": int, "similarity": float, "source_text": str}]
        """
        if not answer or not source_passages:
            return []

        # Step 1: Extract claims (sentences) from the answer
        claims = self._extract_claims(answer)

        if not claims:
            return []

        # Step 2: Map each claim to its best matching source passage
        mappings = []

        for claim in claims:
            best_match = self._find_best_source(claim, source_passages)
            if best_match:
                mappings.append(best_match)

        logger.debug(f"Mapped {len(mappings)} claims to source passages")
        return mappings

    def _extract_claims(self, answer: str) -> list[str]:
        """
        Extract individual claims/sentences from the answer.
        Removes non-claim content like headers and notes.
        """
        # Remove markdown formatting
        clean = re.sub(r"\*\*.*?\*\*", "", answer)
        clean = re.sub(r"\[Passage \d+\]", "", clean)
        clean = re.sub(r"\[.*?\]", "", clean)

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", clean)

        # Filter out short or non-claim sentences
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and not s.startswith("Note:") and not s.startswith("*"):
                claims.append(s)

        return claims

    def _find_best_source(
        self,
        claim: str,
        source_passages: list[dict],
    ) -> Optional[dict]:
        """Find the source passage that best supports a claim."""

        if self.embedder:
            # Use semantic similarity
            return self._find_best_source_semantic(claim, source_passages)
        else:
            # Use lexical overlap
            return self._find_best_source_lexical(claim, source_passages)

    def _find_best_source_semantic(
        self,
        claim: str,
        source_passages: list[dict],
    ) -> Optional[dict]:
        """Find best source using embedding similarity."""
        try:
            passage_texts = [p.get("text", "") for p in source_passages]
            similarities = self.embedder.similarity_batch(claim, passage_texts)

            best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            best_sim = similarities[best_idx]

            if best_sim < 0.1:  # Minimum threshold
                return None

            # Find the most relevant sentence within the passage
            best_passage = source_passages[best_idx]
            evidence_snippet = self._find_evidence_snippet(
                claim, best_passage.get("text", "")
            )

            return {
                "claim": claim,
                "source_passage_idx": best_idx,
                "source_chunk_id": best_passage.get("chunk_id", ""),
                "similarity": round(float(best_sim), 4),
                "source_text": best_passage.get("text", "")[:300],
                "evidence_snippet": evidence_snippet,
                "source_metadata": best_passage.get("metadata", {}),
            }
        except Exception as e:
            logger.warning(f"Semantic claim mapping failed: {e}")
            return self._find_best_source_lexical(claim, source_passages)

    def _find_best_source_lexical(
        self,
        claim: str,
        source_passages: list[dict],
    ) -> Optional[dict]:
        """Find best source using word overlap (Jaccard similarity)."""
        claim_words = set(claim.lower().split())

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
                       "to", "for", "and", "or", "not", "with", "that", "this",
                       "on", "at", "by", "from", "as", "be", "it"}
        claim_words -= stop_words

        if not claim_words:
            return None

        best_idx = -1
        best_score = 0

        for i, passage in enumerate(source_passages):
            passage_words = set(passage.get("text", "").lower().split()) - stop_words
            if not passage_words:
                continue

            # Jaccard similarity
            intersection = claim_words & passage_words
            union = claim_words | passage_words
            score = len(intersection) / len(union) if union else 0

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0 or best_score < 0.05:
            return None

        best_passage = source_passages[best_idx]
        return {
            "claim": claim,
            "source_passage_idx": best_idx,
            "source_chunk_id": best_passage.get("chunk_id", ""),
            "similarity": round(best_score, 4),
            "source_text": best_passage.get("text", "")[:300],
            "evidence_snippet": self._find_evidence_snippet(
                claim, best_passage.get("text", "")
            ),
            "source_metadata": best_passage.get("metadata", {}),
        }

    def _find_evidence_snippet(
        self, claim: str, passage_text: str, window: int = 200
    ) -> str:
        """Find the most relevant snippet within a passage for a claim."""
        if not passage_text:
            return ""

        claim_words = set(claim.lower().split())
        sentences = passage_text.split(".")

        best_sentence = ""
        best_overlap = 0

        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(claim_words & sent_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sent.strip()

        return best_sentence[:window] if best_sentence else passage_text[:window]

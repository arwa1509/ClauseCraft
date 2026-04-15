"""
Hallucination Detection
========================
Detects hallucinated content in generated answers using
Natural Language Inference (NLI) models.
"""

import re
from typing import Optional
from loguru import logger

try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from config import NLI_MODEL


class HallucinationDetector:
    """
    Detects hallucinations by checking if answer claims
    are entailed by the source passages using NLI.

    Labels:
    - ENTAILMENT: claim is supported by evidence
    - CONTRADICTION: claim contradicts evidence
    - NEUTRAL: claim is not supported (potential hallucination)
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or NLI_MODEL
        self.nli_model = None
        self._load_model()

    def _load_model(self):
        """Load the NLI model."""
        if not HF_AVAILABLE:
            logger.warning("transformers not available, hallucination detection limited")
            return

        try:
            self.nli_model = hf_pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )
            logger.info(f"✅ NLI model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            self.nli_model = None

    def detect(
        self,
        answer: str,
        source_passages: list[dict],
    ) -> dict:
        """
        Detect hallucinations in the generated answer.

        Args:
            answer: Generated answer text
            source_passages: Source passages used for generation

        Returns:
            Dict with hallucination analysis
        """
        # Extract claims from answer
        claims = self._extract_claims(answer)

        if not claims:
            return {
                "hallucination_rate": 0.0,
                "claims": [],
                "analysis": "No extractable claims found",
            }

        # Combine source passages into evidence text
        evidence = " ".join(p.get("text", "") for p in source_passages)

        if not evidence.strip():
            return {
                "hallucination_rate": 1.0,
                "claims": [{"claim": c, "label": "UNSUPPORTED", "score": 0.0} for c in claims],
                "analysis": "No source evidence available",
            }

        # Check each claim against evidence
        results = []
        for claim in claims:
            result = self._check_claim(claim, evidence)
            results.append(result)

        # Compute hallucination rate
        unsupported = sum(
            1 for r in results if r["label"] in ("NEUTRAL", "CONTRADICTION", "UNSUPPORTED")
        )
        hallucination_rate = unsupported / len(results) if results else 0

        return {
            "hallucination_rate": round(hallucination_rate, 4),
            "total_claims": len(results),
            "supported": sum(1 for r in results if r["label"] == "ENTAILMENT"),
            "unsupported": unsupported,
            "claims": results,
        }

    def _check_claim(self, claim: str, evidence: str) -> dict:
        """Check a single claim against the evidence."""
        if self.nli_model:
            return self._check_claim_nli(claim, evidence)
        else:
            return self._check_claim_lexical(claim, evidence)

    def _check_claim_nli(self, claim: str, evidence: str) -> dict:
        """Check claim using NLI model."""
        try:
            # Truncate evidence to model max length
            max_evidence_len = 1024
            if len(evidence) > max_evidence_len:
                # Find the most relevant part of evidence
                claim_words = set(claim.lower().split())
                evidence_sents = evidence.split(".")
                scored_sents = []
                for sent in evidence_sents:
                    overlap = len(set(sent.lower().split()) & claim_words)
                    scored_sents.append((overlap, sent))
                scored_sents.sort(reverse=True)
                evidence = ". ".join(s for _, s in scored_sents[:10])

            result = self.nli_model(
                evidence,
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis=claim,
            )

            top_label = result["labels"][0].upper()
            top_score = result["scores"][0]

            return {
                "claim": claim,
                "label": top_label,
                "score": round(top_score, 4),
                "all_scores": {
                    l.upper(): round(s, 4)
                    for l, s in zip(result["labels"], result["scores"])
                },
            }
        except Exception as e:
            logger.warning(f"NLI check failed for claim: {e}")
            return self._check_claim_lexical(claim, evidence)

    def _check_claim_lexical(self, claim: str, evidence: str) -> dict:
        """Fallback: check claim using lexical overlap."""
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())

        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
                       "to", "for", "and", "or", "not", "with", "that", "this"}
        claim_words -= stop_words
        evidence_words -= stop_words

        if not claim_words:
            return {"claim": claim, "label": "NEUTRAL", "score": 0.5}

        overlap = len(claim_words & evidence_words) / len(claim_words)

        if overlap >= 0.6:
            label = "ENTAILMENT"
        elif overlap >= 0.3:
            label = "NEUTRAL"
        else:
            label = "UNSUPPORTED"

        return {
            "claim": claim,
            "label": label,
            "score": round(overlap, 4),
        }

    def _extract_claims(self, answer: str) -> list[str]:
        """Extract verifiable claims from the answer."""
        # Remove markdown and citations
        clean = re.sub(r"\*\*.*?\*\*", "", answer)
        clean = re.sub(r"\[.*?\]", "", clean)

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", clean)

        # Filter
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and not s.startswith("Note"):
                claims.append(s)

        return claims

"""
Hallucination detection using NLI when available.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

try:
    from transformers import pipeline as hf_pipeline

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from config import NLI_MODEL
from sentence_utils import split_sentences


class HallucinationDetector:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or NLI_MODEL
        self.nli_model = None
        self._load_model()

    def _load_model(self):
        if not HF_AVAILABLE:
            logger.warning("transformers not available; hallucination detection limited")
            return

        try:
            self.nli_model = hf_pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
            )
            logger.info(f"NLI model loaded: {self.model_name}")
        except Exception as exc:
            logger.warning(f"Failed to load NLI model: {exc}")
            self.nli_model = None

    def detect(self, answer: str | dict, source_passages: list[dict]) -> dict:
        claims = self._extract_claims(answer)
        if not claims:
            return {
                "hallucination_rate": 0.0,
                "claims": [],
                "analysis": "No extractable claims found",
            }

        evidence = " ".join(passage.get("text", "") for passage in source_passages)
        if not evidence.strip():
            return {
                "hallucination_rate": 1.0,
                "claims": [{"claim": claim, "label": "UNSUPPORTED", "score": 0.0} for claim in claims],
                "analysis": "No source evidence available",
            }

        results = [self._check_claim(claim, evidence) for claim in claims]
        unsupported = sum(
            1 for result in results if result["label"] in {"NEUTRAL", "CONTRADICTION", "UNSUPPORTED"}
        )
        hallucination_rate = unsupported / len(results) if results else 0.0
        return {
            "hallucination_rate": round(hallucination_rate, 4),
            "total_claims": len(results),
            "supported": sum(1 for result in results if result["label"] == "ENTAILMENT"),
            "unsupported": unsupported,
            "claims": results,
        }

    def _check_claim(self, claim: str, evidence: str) -> dict:
        if self.nli_model:
            return self._check_claim_nli(claim, evidence)
        return self._check_claim_lexical(claim, evidence)

    def _check_claim_nli(self, claim: str, evidence: str) -> dict:
        try:
            evidence = self._trim_evidence_for_claim(claim, evidence)
            result = self.nli_model({"text": evidence, "text_pair": claim})[0]
            normalized = self._normalize_nli_scores(result)
            top_label = max(normalized, key=normalized.get)
            top_score = normalized[top_label]
            return {
                "claim": claim,
                "label": top_label,
                "score": round(top_score, 4),
                "all_scores": {label: round(score, 4) for label, score in normalized.items()},
            }
        except Exception as exc:
            logger.warning(f"NLI check failed for claim: {exc}")
            return self._check_claim_lexical(claim, evidence)

    def _normalize_nli_scores(self, scores: list[dict]) -> dict[str, float]:
        label_map = {}
        for item in scores:
            label = item.get("label", "").upper()
            score = float(item.get("score", 0.0))
            if "ENTAIL" in label:
                label_map["ENTAILMENT"] = score
            elif "CONTRAD" in label:
                label_map["CONTRADICTION"] = score
            elif "NEUTRAL" in label:
                label_map["NEUTRAL"] = score
            elif label in {"LABEL_0", "LABEL_1", "LABEL_2"}:
                # Common MNLI label order: contradiction, neutral, entailment
                mapped = {
                    "LABEL_0": "CONTRADICTION",
                    "LABEL_1": "NEUTRAL",
                    "LABEL_2": "ENTAILMENT",
                }[label]
                label_map[mapped] = score
        return label_map or {"NEUTRAL": 0.0}

    def _trim_evidence_for_claim(self, claim: str, evidence: str) -> str:
        claim_words = set(claim.lower().split())
        scored_sentences = []
        for sentence in split_sentences(evidence):
            overlap = len(set(sentence.lower().split()) & claim_words)
            scored_sentences.append((overlap, sentence))
        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in scored_sentences[:8]]
        return " ".join(selected)[:1500]

    def _check_claim_lexical(self, claim: str, evidence: str) -> dict:
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "for", "and", "or", "not", "with", "that", "this"}
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
        return {"claim": claim, "label": label, "score": round(overlap, 4)}

    def _extract_claims(self, answer: str | dict) -> list[str]:
        if isinstance(answer, dict):
            answer = " ".join(
                part
                for part in [answer.get("simple_answer", ""), answer.get("answer_text", "")]
                if part
            )

        clean = re.sub(r"\*\*.*?\*\*", "", answer)
        clean = re.sub(r"\[.*?\]", "", clean)
        claims = []
        for sentence in split_sentences(clean):
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.startswith("Note"):
                claims.append(sentence)
        return claims

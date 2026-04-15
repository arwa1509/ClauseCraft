"""
Confidence Scorer
==================
Computes confidence scores for generated answers based on
retrieval quality, entity coverage, and claim-evidence alignment.
"""

from loguru import logger


class ConfidenceScorer:
    """
    Multi-factor confidence scoring for RAG outputs.

    Factors:
    1. Retrieval score quality (how well passages match the query)
    2. Entity coverage (what fraction of query entities are found in passages)
    3. Claim-evidence alignment (how well each claim maps to a source)
    4. Source diversity (answers from multiple sources are more reliable)
    """

    def __init__(
        self,
        retrieval_weight: float = 0.3,
        entity_weight: float = 0.25,
        claim_weight: float = 0.3,
        diversity_weight: float = 0.15,
    ):
        self.retrieval_weight = retrieval_weight
        self.entity_weight = entity_weight
        self.claim_weight = claim_weight
        self.diversity_weight = diversity_weight

    def compute(
        self,
        passages: list[dict],
        query_entities: list[dict],
        claim_mappings: list[dict],
    ) -> float:
        """
        Compute overall confidence score.

        Args:
            passages: Retrieved/reranked passages
            query_entities: Entities extracted from the query
            claim_mappings: Claim-to-evidence mappings

        Returns:
            Confidence score between 0 and 1
        """
        if not passages:
            return 0.0

        # Factor 1: Retrieval quality
        retrieval_score = self._retrieval_quality(passages)

        # Factor 2: Entity coverage
        entity_score = self._entity_coverage(passages, query_entities)

        # Factor 3: Claim-evidence alignment
        claim_score = self._claim_alignment(claim_mappings)

        # Factor 4: Source diversity
        diversity_score = self._source_diversity(passages)

        # Weighted combination
        confidence = (
            self.retrieval_weight * retrieval_score
            + self.entity_weight * entity_score
            + self.claim_weight * claim_score
            + self.diversity_weight * diversity_score
        )

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        logger.debug(
            f"Confidence: {confidence:.3f} "
            f"(retrieval={retrieval_score:.3f}, entity={entity_score:.3f}, "
            f"claim={claim_score:.3f}, diversity={diversity_score:.3f})"
        )

        return round(confidence, 4)

    def _retrieval_quality(self, passages: list[dict]) -> float:
        """
        Score based on retrieval scores of top passages.
        Higher scores = higher confidence.
        """
        if not passages:
            return 0.0

        scores = [p.get("score", 0) for p in passages]
        if not scores:
            return 0.0

        # Average of top passage scores, normalized
        avg_score = sum(scores) / len(scores)

        # Cross-encoder scores are typically in [-10, 10] range
        # Dense retrieval scores are typically in [0, 1] range
        # Normalize to [0, 1]
        if avg_score > 1:
            # Likely cross-encoder scores
            normalized = min(1.0, (avg_score + 10) / 20)
        elif avg_score < 0:
            normalized = max(0.0, (avg_score + 10) / 20)
        else:
            normalized = avg_score

        # Bonus for high top-1 score
        top_score = max(scores) if scores else 0
        if top_score > 1:
            top_normalized = min(1.0, (top_score + 10) / 20)
        else:
            top_normalized = top_score

        return 0.6 * normalized + 0.4 * top_normalized

    def _entity_coverage(
        self, passages: list[dict], query_entities: list[dict]
    ) -> float:
        """
        Score based on how many query entities appear in retrieved passages.
        """
        if not query_entities:
            return 0.7  # Neutral score if no entities in query

        query_entity_texts = {e["text"].lower() for e in query_entities}

        # Collect all entities from passages
        passage_entity_texts = set()
        for p in passages:
            for e in p.get("entities", []):
                passage_entity_texts.add(e.get("text", "").lower())
            # Also check raw text
            p_text = p.get("text", "").lower()
            for qe in query_entity_texts:
                if qe in p_text:
                    passage_entity_texts.add(qe)

        # Coverage ratio
        covered = query_entity_texts & passage_entity_texts
        coverage = len(covered) / len(query_entity_texts) if query_entity_texts else 0

        return coverage

    def _claim_alignment(self, claim_mappings: list[dict]) -> float:
        """
        Score based on how well answer claims align with source passages.
        """
        if not claim_mappings:
            return 0.3  # Low confidence if no claims mapped

        similarities = [m.get("similarity", 0) for m in claim_mappings]

        # Average similarity
        avg_sim = sum(similarities) / len(similarities)

        # Fraction of well-supported claims (similarity > 0.3)
        well_supported = sum(1 for s in similarities if s > 0.3) / len(similarities)

        return 0.5 * avg_sim + 0.5 * well_supported

    def _source_diversity(self, passages: list[dict]) -> float:
        """
        Score based on diversity of sources.
        Answers from multiple documents are more trustworthy.
        """
        if not passages:
            return 0.0

        # Count unique source documents
        sources = set()
        for p in passages:
            meta = p.get("metadata", {})
            sources.add(meta.get("doc_name", "unknown"))

        # Multiple sources = higher confidence
        n_sources = len(sources)
        if n_sources >= 3:
            return 1.0
        elif n_sources == 2:
            return 0.8
        elif n_sources == 1:
            return 0.5
        else:
            return 0.0

    def compute_per_claim(
        self, claim_mappings: list[dict]
    ) -> list[dict]:
        """
        Compute confidence scores for each individual claim.
        """
        result = []
        for mapping in claim_mappings:
            sim = mapping.get("similarity", 0)
            confidence = min(1.0, sim * 1.5)  # Scale up, cap at 1
            result.append({
                "claim": mapping.get("claim", ""),
                "confidence": round(confidence, 4),
                "source_passage_idx": mapping.get("source_passage_idx"),
                "supported": confidence > 0.4,
            })
        return result

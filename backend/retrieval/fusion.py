"""
Reciprocal Rank Fusion (RRF)
==============================
Combines results from multiple retrieval methods into a single ranked list.
"""

from loguru import logger
from config import RRF_K


def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d:
        score(d) = Σ weight_i / (k + rank_i(d))

    where k is a constant (default 60) and rank_i(d) is the rank of d
    in the i-th result list.

    Args:
        *result_lists: Variable number of result lists from different retrievers
        k: RRF constant (higher = more uniform weighting)
        weights: Optional per-list weights (default: equal)

    Returns:
        Merged and sorted list of results
    """
    if not result_lists:
        return []

    # Default equal weights
    if weights is None:
        weights = [1.0] * len(result_lists)

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0

        for rank, result in enumerate(results, start=1):
            chunk_id = result.get("chunk_id", "")
            if not chunk_id:
                continue

            # RRF formula
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + weight / (k + rank)

            # Keep the best metadata (from the result with highest original score)
            if chunk_id not in chunk_data or result.get("score", 0) > chunk_data[chunk_id].get("score", 0):
                chunk_data[chunk_id] = result

    # Build merged results
    merged = []
    for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        data = chunk_data[chunk_id]
        merged.append({
            "chunk_id": chunk_id,
            "text": data.get("text", ""),
            "metadata": data.get("metadata", {}),
            "entities": data.get("entities", []),
            "score": rrf_score,
            "original_score": data.get("score", 0),
            "rank": len(merged) + 1,
        })

    logger.debug(
        f"RRF fusion: {[len(r) for r in result_lists]} lists → {len(merged)} merged results"
    )

    return merged

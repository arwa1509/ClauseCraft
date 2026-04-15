"""
Evaluation Metrics
===================
Computes retrieval and generation quality metrics.
Supports: Precision@K, Recall@K, MRR, Entity F1, ROUGE-L.
"""

import json
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

from sklearn.metrics import f1_score, precision_score, recall_score
from config import QA_DATASETS_DIR


class EvaluationMetrics:
    """
    Computes evaluation metrics for retrieval and generation quality.
    """

    def precision_at_k(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 5,
    ) -> float:
        """Compute Precision@K: fraction of retrieved docs that are relevant."""
        retrieved_k = retrieved_ids[:k]
        if not retrieved_k:
            return 0.0
        relevant_count = sum(1 for r in retrieved_k if r in relevant_ids)
        return relevant_count / len(retrieved_k)

    def recall_at_k(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 5,
    ) -> float:
        """Compute Recall@K: fraction of relevant docs that are retrieved."""
        if not relevant_ids:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        relevant_count = sum(1 for r in retrieved_k if r in relevant_ids)
        return relevant_count / len(relevant_ids)

    def mrr(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:
        """
        Compute Mean Reciprocal Rank.
        MRR = 1/rank of first relevant result.
        """
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return 1.0 / i
        return 0.0

    def entity_f1(
        self,
        predicted_entities: list[dict],
        gold_entities: list[dict],
    ) -> dict:
        """
        Compute entity-level precision, recall, and F1.

        Args:
            predicted_entities: Entities from NER
            gold_entities: Ground truth entities

        Returns:
            Dict with precision, recall, f1
        """
        pred_set = {
            (e["text"].lower(), e.get("label", ""))
            for e in predicted_entities
        }
        gold_set = {
            (e["text"].lower(), e.get("label", ""))
            for e in gold_entities
        }

        if not pred_set and not gold_set:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if not pred_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not gold_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        true_positives = len(pred_set & gold_set)
        precision = true_positives / len(pred_set)
        recall = true_positives / len(gold_set)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def rouge_l(
        self,
        predicted: str,
        reference: str,
    ) -> dict:
        """
        Compute ROUGE-L score between predicted and reference text.
        """
        if not ROUGE_AVAILABLE:
            logger.warning("rouge-score not installed, computing basic overlap")
            return self._basic_rouge(predicted, reference)

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, predicted)

        return {
            "rouge_l_precision": round(scores["rougeL"].precision, 4),
            "rouge_l_recall": round(scores["rougeL"].recall, 4),
            "rouge_l_f1": round(scores["rougeL"].fmeasure, 4),
        }

    def _basic_rouge(self, predicted: str, reference: str) -> dict:
        """Basic ROUGE-L approximation using LCS."""
        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()

        if not pred_words or not ref_words:
            return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

        # LCS length
        lcs_len = self._lcs_length(pred_words, ref_words)

        precision = lcs_len / len(pred_words)
        recall = lcs_len / len(ref_words)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return {
            "rouge_l_precision": round(precision, 4),
            "rouge_l_recall": round(recall, 4),
            "rouge_l_f1": round(f1, 4),
        }

    def _lcs_length(self, x: list, y: list) -> int:
        """Compute length of Longest Common Subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def evaluate_dataset(self, qa_data_path: Optional[str] = None) -> dict:
        """
        Run full evaluation on a QA dataset.

        Expected format (JSON):
        [
            {
                "question": "What is Section 302?",
                "answer": "Section 302 deals with punishment for murder...",
                "relevant_chunks": ["chunk_id_1", "chunk_id_2"],
                "entities": [{"text": "Section 302", "label": "STATUTE"}]
            }
        ]
        """
        # Find QA dataset
        if qa_data_path:
            data_path = Path(qa_data_path)
        else:
            data_files = list(QA_DATASETS_DIR.glob("*.json"))
            if not data_files:
                return {"error": "No QA evaluation data found"}
            data_path = data_files[0]

        try:
            with open(data_path) as f:
                qa_data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to load QA data: {e}"}

        if not qa_data:
            return {"error": "QA dataset is empty"}

        return {
            "dataset": str(data_path),
            "total_questions": len(qa_data),
            "message": "Use the /api/evaluation/run endpoint with the full pipeline to evaluate.",
        }

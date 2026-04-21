"""
Ragas-Powered Evaluation Framework
=====================================
Integrates the Ragas library to evaluate ClauseCraft's RAG pipeline on two
critical dimensions:

    1. Faithfulness       – measures hallucination (does the answer stay
                            grounded in the retrieved context?)
    2. Context Precision  – measures retrieval quality (are the retrieved
                            chunks actually relevant to the question?)

The module is designed as a drop-in upgrade to the existing EvaluationMetrics
class, preserving all original metrics (Precision@K, Recall@K, MRR, Entity F1,
ROUGE-L) while adding the Ragas suite.

Usage (standalone script):
    python -m evaluation.metrics --qa-path data/qa_datasets/sample.json

Environment Variables:
    OPENAI_API_KEY   – required by Ragas (uses OpenAI for LLM-based judging)
                       If not set, Ragas falls back to heuristic evaluation.
    RAGAS_BATCH_SIZE – number of samples per Ragas evaluation batch (default 4)
"""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import Optional

from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

from config import QA_DATASETS_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Ragas imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
    )
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    RAGAS_AVAILABLE = True
    logger.info("✅ Ragas library loaded.")
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "ragas not installed – Faithfulness and Context Precision metrics "
        "will use lightweight heuristic fallbacks."
    )

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI LLM wrapper for Ragas (optional)
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
RAGAS_BATCH_SIZE = int(os.getenv("RAGAS_BATCH_SIZE", "4"))


def _build_ragas_llm():
    """Build a LangChain-compatible LLM for Ragas, or return None."""
    if not OPENAI_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            temperature=0,
        )
    except ImportError:
        logger.warning("langchain-openai not installed – Ragas will use defaults.")
        return None


def _build_ragas_embeddings():
    """Build a LangChain-compatible Embeddings for Ragas, or return None."""
    if not OPENAI_API_KEY:
        return None
    try:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic fallbacks (used when Ragas / OpenAI are unavailable)
# ──────────────────────────────────────────────────────────────────────────────

def _heuristic_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Lightweight faithfulness proxy.
    Measures the fraction of answer tokens that can be found in any context chunk.
    Range: 0.0 (hallucinated) – 1.0 (fully grounded).
    """
    if not answer or not contexts:
        return 0.0
    answer_tokens = set(answer.lower().split())
    context_tokens = set(" ".join(contexts).lower().split())
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return round(overlap / len(answer_tokens), 4)


def _heuristic_context_precision(question: str, contexts: list[str]) -> float:
    """
    Lightweight context precision proxy.
    Measures what fraction of retrieved context chunks contain question keywords.
    """
    if not question or not contexts:
        return 0.0
    q_tokens = set(question.lower().split())
    relevant = sum(
        1 for c in contexts
        if q_tokens & set(c.lower().split())
    )
    return round(relevant / len(contexts), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation class
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationMetrics:
    """
    Unified evaluation suite for the ClauseCraft RAG pipeline.

    Original metrics: Precision@K, Recall@K, MRR, Entity F1, ROUGE-L.
    New Ragas metrics: Faithfulness, Context Precision.
    """

    # ── Original metrics (unchanged) ─────────────────────────────────────────

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
        """Compute Mean Reciprocal Rank."""
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return 1.0 / i
        return 0.0

    def entity_f1(
        self,
        predicted_entities: list[dict],
        gold_entities: list[dict],
    ) -> dict:
        """Compute entity-level precision, recall, and F1."""
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
        if not pred_set or not gold_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def rouge_l(self, predicted: str, reference: str) -> dict:
        """Compute ROUGE-L score between predicted and reference text."""
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
        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()
        if not pred_words or not ref_words:
            return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}
        lcs_len = self._lcs_length(pred_words, ref_words)
        precision = lcs_len / len(pred_words)
        recall = lcs_len / len(ref_words)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "rouge_l_precision": round(precision, 4),
            "rouge_l_recall": round(recall, 4),
            "rouge_l_f1": round(f1, 4),
        }

    def _lcs_length(self, x: list, y: list) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # ── NEW: Ragas-powered metrics ────────────────────────────────────────────

    def faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> float:
        """
        Measure faithfulness (anti-hallucination).

        Uses Ragas Faithfulness metric if available; falls back to a
        token-overlap heuristic.

        Args:
            question: The user's query.
            answer:   The LLM-generated answer.
            contexts: The retrieved text chunks passed to the LLM.

        Returns:
            Float in [0.0, 1.0]. Higher is better (less hallucination).
        """
        if not RAGAS_AVAILABLE:
            score = _heuristic_faithfulness(answer, contexts)
            logger.debug(f"[Faithfulness/heuristic] {score:.4f}")
            return score

        try:
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            dataset = EvaluationDataset(samples=[sample])
            llm = _build_ragas_llm()
            emb = _build_ragas_embeddings()

            metric = Faithfulness(llm=llm) if llm else Faithfulness()
            results = evaluate(dataset=dataset, metrics=[metric])
            score = float(results["faithfulness"][0])
            logger.info(f"[Faithfulness/ragas] {score:.4f}")
            return round(score, 4)
        except Exception as exc:
            logger.warning(f"Ragas Faithfulness failed ({exc}), using heuristic.")
            return _heuristic_faithfulness(answer, contexts)

    def context_precision(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> float:
        """
        Measure context precision (retrieval quality).

        Uses Ragas LLMContextPrecisionWithoutReference if available; falls back
        to a keyword-overlap heuristic.

        Args:
            question: The user's query.
            answer:   The generated answer (used as implicit reference).
            contexts: The retrieved text chunks.

        Returns:
            Float in [0.0, 1.0]. Higher is better.
        """
        if not RAGAS_AVAILABLE:
            score = _heuristic_context_precision(question, contexts)
            logger.debug(f"[ContextPrecision/heuristic] {score:.4f}")
            return score

        try:
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            dataset = EvaluationDataset(samples=[sample])
            llm = _build_ragas_llm()

            metric = (
                LLMContextPrecisionWithoutReference(llm=llm)
                if llm
                else LLMContextPrecisionWithoutReference()
            )
            results = evaluate(dataset=dataset, metrics=[metric])
            score = float(results["llm_context_precision_without_reference"][0])
            logger.info(f"[ContextPrecision/ragas] {score:.4f}")
            return round(score, 4)
        except Exception as exc:
            logger.warning(f"Ragas ContextPrecision failed ({exc}), using heuristic.")
            return _heuristic_context_precision(question, contexts)

    def evaluate_rag_sample(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        retrieved_ids: Optional[list[str]] = None,
        relevant_ids: Optional[set[str]] = None,
        predicted_entities: Optional[list[dict]] = None,
        gold_entities: Optional[list[dict]] = None,
        reference_answer: Optional[str] = None,
    ) -> dict:
        """
        Run the complete evaluation suite on a single RAG sample.

        Returns a unified dict with all metric scores.
        """
        result: dict = {}

        # Ragas metrics
        result["faithfulness"] = self.faithfulness(question, answer, contexts)
        result["context_precision"] = self.context_precision(question, answer, contexts)

        # Retrieval metrics (if ground truth available)
        if retrieved_ids is not None and relevant_ids is not None:
            result["precision_at_5"] = self.precision_at_k(retrieved_ids, relevant_ids, k=5)
            result["recall_at_5"] = self.recall_at_k(retrieved_ids, relevant_ids, k=5)
            result["mrr"] = self.mrr(retrieved_ids, relevant_ids)

        # Entity F1 (if ground truth available)
        if predicted_entities is not None and gold_entities is not None:
            result["entity_f1"] = self.entity_f1(predicted_entities, gold_entities)

        # ROUGE-L (if reference answer available)
        if reference_answer:
            result["rouge_l"] = self.rouge_l(answer, reference_answer)

        return result

    def evaluate_dataset(
        self,
        qa_data_path: Optional[str] = None,
        rag_results: Optional[list[dict]] = None,
    ) -> dict:
        """
        Run full evaluation on a QA dataset.

        Args:
            qa_data_path: Path to JSON file of QA samples.
            rag_results:  Pre-computed list of RAG outputs with keys:
                          question, answer, contexts (list[str]).

        QA JSON format::

            [
                {
                    "question": "What is Section 302?",
                    "answer": "Section 302 deals with...",   # optional reference
                    "relevant_chunks": ["chunk_id_1", ...],
                    "entities": [{"text": "Section 302", "label": "STATUTE"}]
                }
            ]

        RAG results format::

            [
                {
                    "question": "...",
                    "answer": "...",     # generated
                    "contexts": ["retrieved chunk 1", "retrieved chunk 2"]
                }
            ]
        """
        # Load QA dataset
        if rag_results is None:
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
        else:
            qa_data = rag_results

        if not qa_data:
            return {"error": "QA dataset is empty"}

        all_faithfulness: list[float] = []
        all_context_precision: list[float] = []
        all_mrr: list[float] = []
        sample_results: list[dict] = []

        for item in qa_data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            contexts = item.get("contexts", [])
            reference = item.get("reference_answer", answer)

            if not question or not answer:
                continue

            sample_eval = self.evaluate_rag_sample(
                question=question,
                answer=answer,
                contexts=contexts if contexts else [answer],  # graceful
                reference_answer=reference,
            )
            all_faithfulness.append(sample_eval["faithfulness"])
            all_context_precision.append(sample_eval["context_precision"])
            if "mrr" in sample_eval:
                all_mrr.append(sample_eval["mrr"])

            sample_results.append({
                "question": question,
                "scores": sample_eval,
            })

        def _avg(lst: list[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        aggregate = {
            "total_samples": len(sample_results),
            "ragas_faithfulness_mean": _avg(all_faithfulness),
            "ragas_context_precision_mean": _avg(all_context_precision),
            "ragas_available": RAGAS_AVAILABLE,
            "sample_results": sample_results,
        }
        if all_mrr:
            aggregate["mean_reciprocal_rank"] = _avg(all_mrr)

        logger.info(
            f"Evaluation complete – "
            f"faithfulness={aggregate['ragas_faithfulness_mean']:.4f}, "
            f"context_precision={aggregate['ragas_context_precision_mean']:.4f}"
        )
        return aggregate


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point: python -m evaluation.metrics --qa-path ...
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Ragas evaluation on ClauseCraft RAG outputs."
    )
    parser.add_argument(
        "--qa-path",
        type=str,
        default=None,
        help="Path to the QA evaluation JSON file.",
    )
    parser.add_argument(
        "--rag-results",
        type=str,
        default=None,
        help=(
            "Path to a JSON file of pre-computed RAG results "
            "[{question, answer, contexts}]."
        ),
    )
    args = parser.parse_args()

    rag_results_data = None
    if args.rag_results:
        with open(args.rag_results) as f:
            rag_results_data = json.load(f)

    metrics = EvaluationMetrics()
    results = metrics.evaluate_dataset(
        qa_data_path=args.qa_path,
        rag_results=rag_results_data,
    )
    print(json.dumps(results, indent=2))

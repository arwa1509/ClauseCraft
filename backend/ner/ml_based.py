"""
ML-assisted NER using spaCy with conservative legal label mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. ML-based NER will return no results.")

from config import MODELS_DIR, NER_DATASETS_DIR, SPACY_MODEL

SPACY_LABEL_MAP = {
    "ORG": "ORG",
    "PERSON": "PERSON",
    "DATE": "DATE",
    "LAW": "STATUTE",
    "GPE": "LOCATION",
    "MONEY": "MONEY",
    "CARDINAL": None,
    "ORDINAL": None,
}

COURT_KEYWORDS = {"court", "tribunal", "bench", "commission"}
JUDGE_PREFIXES = (
    "justice ",
    "judge ",
    "hon'ble justice ",
    "honourable justice ",
    "honorable justice ",
)
JUDGE_TITLES = {"justice", "judge", "cj", "c.j."}


class MLBasedNER:
    def __init__(self):
        self.nlp = None
        self._resume_training = False
        self._load_model()

    def _load_model(self):
        if not SPACY_AVAILABLE:
            return

        custom_model_path = MODELS_DIR / "legal_ner_model"
        if custom_model_path.exists():
            try:
                self.nlp = spacy.load(str(custom_model_path))
                self._resume_training = True
                logger.info(f"Loaded custom legal NER model from {custom_model_path}")
                return
            except Exception as exc:
                logger.warning(f"Failed to load custom legal model: {exc}")

        try:
            self.nlp = spacy.load(SPACY_MODEL)
            self._resume_training = True
            logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
        except OSError:
            logger.warning(f"spaCy model '{SPACY_MODEL}' not found; trying download.")
            try:
                spacy.cli.download(SPACY_MODEL)
                self.nlp = spacy.load(SPACY_MODEL)
                self._resume_training = True
            except Exception as exc:
                logger.error(f"Failed to download spaCy model: {exc}")
                self.nlp = None
                self._resume_training = False

    def extract(self, text: str) -> list[dict]:
        if not self.nlp or not text:
            return []

        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            legal_label = self._map_label(ent.label_, ent.text)
            if legal_label is None:
                continue
            entities.append(
                {
                    "text": ent.text.strip(),
                    "label": legal_label,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "ml_based",
                    "original_label": ent.label_,
                }
            )
        return entities

    def _map_label(self, spacy_label: str, text: str) -> Optional[str]:
        text_lower = text.lower()
        mapped = SPACY_LABEL_MAP.get(spacy_label, spacy_label)
        if mapped is None:
            return None

        if spacy_label == "ORG":
            if any(keyword in text_lower for keyword in COURT_KEYWORDS):
                return "COURT"
            return "ORG"

        if spacy_label == "PERSON":
            if self._looks_like_judge(text_lower):
                return "JUDGE"
            return "PERSON"

        if spacy_label in {"NORP", "FAC", "PRODUCT"}:
            return None

        return mapped

    def _looks_like_judge(self, text_lower: str) -> bool:
        if text_lower.startswith(JUDGE_PREFIXES):
            return True
        text_parts = text_lower.replace(",", " ").split()
        if not text_parts:
            return False
        return text_parts[0] in JUDGE_TITLES

    def fine_tune(
        self,
        training_data_path: Optional[str] = None,
        n_iter: int = 30,
        dropout: float = 0.35,
    ) -> dict:
        if not SPACY_AVAILABLE:
            return {"error": "spaCy not available"}

        if training_data_path:
            data_path = Path(training_data_path)
        else:
            data_files = list(NER_DATASETS_DIR.glob("*.json"))
            if not data_files:
                return {"error": "No NER training data found"}
            data_path = data_files[0]

        with open(data_path, "r", encoding="utf-8") as handle:
            training_data = json.load(handle)
        if not training_data:
            return {"error": "Training data is empty"}

        if self.nlp is None:
            self.nlp = spacy.blank("en")
            self._resume_training = False

        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        legal_labels = [
            "STATUTE",
            "PROVISION",
            "COURT",
            "JUDGE",
            "CASE_CITATION",
            "LEGAL_ACTION",
            "DATE",
            "PARTY",
            "ACT",
            "PENALTY",
            "JURISDICTION",
            "ORG",
            "PERSON",
        ]
        for label in legal_labels:
            ner.add_label(label)

        from spacy.training import Example

        train_examples = []
        for item in training_data:
            doc = self.nlp.make_doc(item["text"])
            annotations = {"entities": [tuple(entity) for entity in item.get("entities", [])]}
            try:
                train_examples.append(Example.from_dict(doc, annotations))
            except Exception as exc:
                logger.warning(f"Skipping bad training example: {exc}")

        if not train_examples:
            return {"error": "No valid training examples"}

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = (
                self.nlp.resume_training()
                if self._resume_training
                else self.nlp.begin_training()
            )
            losses_history = []
            for iteration in range(n_iter):
                import random

                random.shuffle(train_examples)
                losses = {}
                for batch_start in range(0, len(train_examples), 8):
                    batch = train_examples[batch_start : batch_start + 8]
                    self.nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)
                losses_history.append(losses.get("ner", 0))
                if (iteration + 1) % 10 == 0:
                    logger.info(
                        f"Iteration {iteration + 1}/{n_iter}, loss={losses.get('ner', 0):.4f}"
                    )

        output_path = MODELS_DIR / "legal_ner_model"
        output_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(str(output_path))
        self._resume_training = True
        return {
            "status": "success",
            "iterations": n_iter,
            "final_loss": losses_history[-1] if losses_history else None,
            "examples": len(train_examples),
            "model_path": str(output_path),
        }

"""
ML-Based Legal NER
===================
Uses spaCy (and optionally transformers) for ML-based entity recognition.
Supports fine-tuning with custom legal NER datasets.
"""

import json
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    import spacy
    from spacy.tokens import DocBin
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. ML-based NER will be limited.")

from config import SPACY_MODEL, MODELS_DIR, NER_DATASETS_DIR


# Mapping from spaCy default labels to our legal labels
SPACY_LABEL_MAP = {
    "ORG": "COURT",         # Organizations → Courts (when legal context)
    "PERSON": "JUDGE",       # Persons → Judges (when legal context)
    "DATE": "DATE",
    "LAW": "STATUTE",
    "GPE": "JURISDICTION",
    "MONEY": "PENALTY",
    "CARDINAL": None,        # Skip cardinal numbers
    "ORDINAL": None,
}

# Legal context keywords to refine entity labeling
COURT_KEYWORDS = {"court", "tribunal", "bench", "division", "commission"}
JUDGE_KEYWORDS = {"justice", "judge", "hon", "honourable", "cj", "j."}


class MLBasedNER:
    """
    ML-based Named Entity Recognition using spaCy.
    Falls back to a lightweight approach if spaCy is unavailable.
    """

    def __init__(self):
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Load the spaCy model."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, ML NER will return empty results")
            return

        # Check for fine-tuned model first
        custom_model_path = MODELS_DIR / "legal_ner_model"
        if custom_model_path.exists():
            try:
                self.nlp = spacy.load(str(custom_model_path))
                logger.info(f"✅ Loaded custom legal NER model from {custom_model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load custom model: {e}")

        # Fall back to default spaCy model
        try:
            self.nlp = spacy.load(SPACY_MODEL)
            logger.info(f"✅ Loaded spaCy model: {SPACY_MODEL}")
        except OSError:
            logger.warning(f"spaCy model '{SPACY_MODEL}' not found. Downloading...")
            try:
                spacy.cli.download(SPACY_MODEL)
                self.nlp = spacy.load(SPACY_MODEL)
                logger.info(f"✅ Downloaded and loaded spaCy model: {SPACY_MODEL}")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                self.nlp = None

    def extract(self, text: str) -> list[dict]:
        """
        Extract entities using spaCy NER.

        Args:
            text: Input text

        Returns:
            List of entity dicts with text, label, start, end
        """
        if not self.nlp or not text:
            return []

        # Process with spaCy
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            # Map spaCy label to our legal label
            legal_label = self._map_label(ent.label_, ent.text)

            if legal_label is None:
                continue

            entities.append({
                "text": ent.text.strip(),
                "label": legal_label,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "ml_based",
                "original_label": ent.label_,
            })

        return entities

    def _map_label(self, spacy_label: str, text: str) -> Optional[str]:
        """
        Map spaCy labels to legal entity labels with context awareness.
        """
        text_lower = text.lower()

        # Direct mapping
        if spacy_label in SPACY_LABEL_MAP:
            mapped = SPACY_LABEL_MAP[spacy_label]
            if mapped is None:
                return None

            # Refine ORG → COURT if it looks like a court
            if spacy_label == "ORG":
                if any(kw in text_lower for kw in COURT_KEYWORDS):
                    return "COURT"
                return "ORG"

            # Refine PERSON → JUDGE if preceded by "Justice" etc.
            if spacy_label == "PERSON":
                if any(kw in text_lower for kw in JUDGE_KEYWORDS):
                    return "JUDGE"
                return "PARTY"

            return mapped

        # For labels not in our map
        if spacy_label == "NORP":
            return None  # Nationalities not needed
        if spacy_label == "FAC":
            return None  # Facilities not needed
        if spacy_label == "PRODUCT":
            return None

        return spacy_label

    def fine_tune(
        self,
        training_data_path: Optional[str] = None,
        n_iter: int = 30,
        dropout: float = 0.35,
    ) -> dict:
        """
        Fine-tune the NER model with custom legal training data.

        Expected training data format (JSON):
        [
            {
                "text": "Section 302 of IPC deals with murder",
                "entities": [[0, 11, "STATUTE"], [15, 18, "ACT"]]
            },
            ...
        ]

        Args:
            training_data_path: Path to training data JSON
            n_iter: Number of training iterations
            dropout: Dropout rate

        Returns:
            Training metrics dict
        """
        if not SPACY_AVAILABLE:
            return {"error": "spaCy not available"}

        # Find training data
        if training_data_path:
            data_path = Path(training_data_path)
        else:
            # Look for default dataset
            data_files = list(NER_DATASETS_DIR.glob("*.json"))
            if not data_files:
                return {"error": "No NER training data found"}
            data_path = data_files[0]

        logger.info(f"Fine-tuning NER model with {data_path}")

        # Load training data
        with open(data_path) as f:
            training_data = json.load(f)

        if not training_data:
            return {"error": "Training data is empty"}

        # Create or get the NER pipe
        if self.nlp is None:
            self.nlp = spacy.blank("en")

        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        # Add entity labels
        legal_labels = [
            "STATUTE", "PROVISION", "COURT", "JUDGE",
            "CASE_CITATION", "LEGAL_ACTION", "DATE",
            "PARTY", "ACT", "PENALTY",
        ]
        for label in legal_labels:
            ner.add_label(label)

        # Convert training data to spaCy format
        train_examples = []
        from spacy.training import Example

        for item in training_data:
            text = item["text"]
            entities = item.get("entities", [])
            doc = self.nlp.make_doc(text)
            annotations = {"entities": [tuple(e) for e in entities]}
            try:
                example = Example.from_dict(doc, annotations)
                train_examples.append(example)
            except Exception as e:
                logger.warning(f"Skipping bad training example: {e}")

        if not train_examples:
            return {"error": "No valid training examples"}

        # Training loop
        other_pipes = [p for p in self.nlp.pipe_names if p != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            losses_history = []

            for i in range(n_iter):
                import random
                random.shuffle(train_examples)
                losses = {}

                for batch_start in range(0, len(train_examples), 8):
                    batch = train_examples[batch_start:batch_start + 8]
                    self.nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)

                losses_history.append(losses.get("ner", 0))
                if (i + 1) % 10 == 0:
                    logger.info(f"  Iteration {i+1}/{n_iter}, Loss: {losses.get('ner', 0):.4f}")

        # Save fine-tuned model
        output_path = MODELS_DIR / "legal_ner_model"
        output_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(str(output_path))
        logger.info(f"✅ Fine-tuned model saved to {output_path}")

        return {
            "status": "success",
            "iterations": n_iter,
            "final_loss": losses_history[-1] if losses_history else None,
            "examples": len(train_examples),
            "model_path": str(output_path),
        }

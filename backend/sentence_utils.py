"""
Shared sentence segmentation utilities.

Uses the configured spaCy model's statistical sentence boundary detection when
available, which handles legal abbreviations like "Sec.", "Hon.", and "vs."
more reliably than a pure period-based sentencizer. Falls back to the basic
sentencizer if the model is not installed.
"""

from __future__ import annotations

import threading

import spacy

from config import SPACY_MODEL

_sentencizer = None
_sentencizer_lock = threading.Lock()


def _get_sentencizer():
    global _sentencizer
    if _sentencizer is None:
        with _sentencizer_lock:
            if _sentencizer is None:
                try:
                    nlp = spacy.load(SPACY_MODEL, exclude=["ner"])
                    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")
                except OSError:
                    nlp = spacy.blank("en")
                    nlp.add_pipe("sentencizer")
                _sentencizer = nlp
    return _sentencizer


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    nlp = _get_sentencizer()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

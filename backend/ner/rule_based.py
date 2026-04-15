"""
Rule-Based Legal NER
=====================
Pattern-based extraction of legal entities using regex.
Supports fine-grained legal entity types including nested entities.

Entity Types:
- STATUTE: Section 302 IPC, Section 420 CrPC
- PROVISION: Article 14, Article 21
- COURT: Supreme Court, High Court of Delhi
- JUDGE: Justice R.M. Lodha
- CASE_CITATION: AIR 2014 SC 1863, (2020) 5 SCC 1
- LEGAL_ACTION: bail, appeal, writ petition
- DATE: 1st January 2020, 01/01/2020
- PARTY: Appellant, Respondent, Petitioner
- ACT: Indian Penal Code, Companies Act 2013
- PENALTY: imprisonment, fine, compensation
"""

import re
from typing import Optional
from loguru import logger


class RuleBasedNER:
    """
    Regex-based Named Entity Recognition for legal documents.
    Handles nested entities (e.g., "Section 302 of the Indian Penal Code"
    yields both STATUTE and ACT entities).
    """

    def __init__(self):
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> list[tuple[str, re.Pattern]]:
        """Build regex patterns for each entity type."""
        return [
            # ── STATUTE ─────────────────────────────────────────────
            ("STATUTE", re.compile(
                r"(?:Section|Sec\.?|S\.)\s+\d+[A-Z]?"
                r"(?:\s*[\(\[][\divxIVX]+[\)\]])?"      # Sub-sections
                r"(?:\s+(?:of|under|r/w)\s+(?:the\s+)?"
                r"(?:Indian\s+Penal\s+Code|IPC|Cr\.?P\.?C\.?|CPC|"
                r"Code\s+of\s+Criminal\s+Procedure|"
                r"Code\s+of\s+Civil\s+Procedure|"
                r"Evidence\s+Act|"
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act(?:\s*,?\s*\d{4})?))?",
                re.IGNORECASE,
            )),

            # ── PROVISION (Constitutional Articles) ─────────────────
            ("PROVISION", re.compile(
                r"(?:Article|Art\.?)\s+\d+[A-Z]?"
                r"(?:\s*[\(\[][\divx]+[\)\]])?"
                r"(?:\s+of\s+the\s+Constitution(?:\s+of\s+India)?)?",
                re.IGNORECASE,
            )),

            # ── ACT ────────────────────────────────────────────────
            ("ACT", re.compile(
                r"(?:the\s+)?"
                r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+)"
                r"(?:Act|Ordinance|Code|Rules|Regulations)"
                r"(?:\s*,?\s*\d{4})?",
            )),

            # ── CASE_CITATION ──────────────────────────────────────
            ("CASE_CITATION", re.compile(
                r"(?:"
                r"AIR\s+\d{4}\s+[A-Z]+(?:\s+\d+)?"                     # AIR citations
                r"|\(\d{4}\)\s+\d+\s+SCC\s+\d+"                         # SCC citations
                r"|\d{4}\s+(?:SCC|SCR|AIR)\s+[\(\d]+"                   # Year-first
                r"|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.?\s+"            # Party v Party
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"
                r"(?:\s*[\(\[]\d{4}[\)\]])?"                             # Optional year
                r"|\[\d{4}\]\s+\d+\s+[A-Z]+\s+\d+"                      # [Year] citations
                r")",
            )),

            # ── COURT ──────────────────────────────────────────────
            ("COURT", re.compile(
                r"(?:"
                r"(?:Hon'?ble\s+)?Supreme\s+Court(?:\s+of\s+India)?"
                r"|(?:Hon'?ble\s+)?High\s+Court(?:\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?"
                r"|(?:Hon'?ble\s+)?District\s+Court(?:\s+of\s+[A-Z][a-z]+)?"
                r"|(?:Hon'?ble\s+)?Sessions\s+Court"
                r"|(?:Hon'?ble\s+)?Magistrate(?:'s)?\s+Court"
                r"|National\s+(?:Company\s+Law\s+)?Tribunal"
                r"|NCLAT|NCLT|NGT|SAT|CAT|ITAT|CESTAT"
                r"|(?:Hon'?ble\s+)?(?:Civil|Criminal|Family|Labour)\s+Court"
                r")",
                re.IGNORECASE,
            )),

            # ── JUDGE ──────────────────────────────────────────────
            ("JUDGE", re.compile(
                r"(?:"
                r"(?:(?:Hon'?ble|Honourable)\s+)?"
                r"(?:Chief\s+)?Justice\s+"
                r"(?:(?:Dr\.?\s+)?[A-Z]\.?\s*)+[A-Z][a-z]+"
                r"(?:\s+[A-Z][a-z]+)*"
                r"|(?:J\.?\s*,?\s*(?:and\s+)?)+[A-Z][a-z]+"  # "Lodha, J."
                r")",
            )),

            # ── LEGAL_ACTION ───────────────────────────────────────
            ("LEGAL_ACTION", re.compile(
                r"\b(?:"
                r"bail|anticipatory\s+bail|regular\s+bail"
                r"|appeal|cross[\s-]?appeal|special\s+leave\s+petition"
                r"|writ\s+petition|habeas\s+corpus"
                r"|FIR|first\s+information\s+report|chargesheet|charge\s*sheet"
                r"|conviction|acquittal|sentence|sentencing"
                r"|remand|parole|probation"
                r"|injunction|stay\s+order|interim\s+order"
                r"|review\s+petition|curative\s+petition"
                r"|arbitration|mediation|conciliation"
                r"|summons|warrant|notice"
                r"|plea\s+bargain(?:ing)?"
                r"|cognizable\s+offence|non[\s-]?cognizable"
                r"|bailable|non[\s-]?bailable"
                r")\b",
                re.IGNORECASE,
            )),

            # ── DATE ───────────────────────────────────────────────
            ("DATE", re.compile(
                r"(?:"
                r"\d{1,2}(?:st|nd|rd|th)?\s+"
                r"(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s*,?\s*\d{4}"
                r"|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
                r"|(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2}\s*,?\s*\d{4}"
                r")",
                re.IGNORECASE,
            )),

            # ── PARTY ─────────────────────────────────────────────
            ("PARTY", re.compile(
                r"\b(?:"
                r"(?:the\s+)?(?:appellant|respondent|petitioner|defendant|"
                r"plaintiff|complainant|accused|prosecution|"
                r"applicant|opposite\s+party|intervener|"
                r"claimant|counter[\s-]?claimant)"
                r"s?"
                r")\b",
                re.IGNORECASE,
            )),

            # ── PENALTY ───────────────────────────────────────────
            ("PENALTY", re.compile(
                r"\b(?:"
                r"(?:rigorous|simple)?\s*imprisonment"
                r"(?:\s+(?:for\s+)?(?:life|\d+\s+(?:years?|months?|days?)))?"
                r"|(?:fine|penalty)\s+of\s+(?:Rs\.?\s*)?\d[\d,]*"
                r"|death\s+(?:sentence|penalty)"
                r"|compensation\s+of\s+(?:Rs\.?\s*)?\d[\d,]*"
                r"|damages"
                r")\b",
                re.IGNORECASE,
            )),
        ]

    def extract(self, text: str) -> list[dict]:
        """
        Extract all legal entities from text.

        Args:
            text: Input text to process

        Returns:
            List of entity dicts: [{"text": str, "label": str, "start": int, "end": int}]
        """
        if not text:
            return []

        entities = []

        for label, pattern in self.patterns:
            for match in pattern.finditer(text):
                entity_text = match.group().strip()
                if len(entity_text) < 2:
                    continue

                entities.append({
                    "text": entity_text,
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "rule_based",
                })

        # Remove overlapping entities (keep longer match)
        entities = self._resolve_overlaps(entities)

        return entities

    def _resolve_overlaps(self, entities: list[dict]) -> list[dict]:
        """
        Remove overlapping entities, preferring longer matches.
        For nested entities, keep both if different labels.
        """
        if not entities:
            return []

        # Sort by start position, then by length (longer first)
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

        resolved = []
        for entity in entities:
            # Check if this entity overlaps with any already resolved
            is_overlap = False
            for existing in resolved:
                # Same label and overlapping → skip
                if (
                    entity["label"] == existing["label"]
                    and entity["start"] < existing["end"]
                    and entity["end"] > existing["start"]
                ):
                    is_overlap = True
                    break

            if not is_overlap:
                resolved.append(entity)

        return resolved

    def extract_with_context(
        self, text: str, context_window: int = 50
    ) -> list[dict]:
        """
        Extract entities with surrounding context for better understanding.

        Args:
            text: Input text
            context_window: Number of characters of context on each side

        Returns:
            List of entity dicts with added "context" field
        """
        entities = self.extract(text)

        for entity in entities:
            start = max(0, entity["start"] - context_window)
            end = min(len(text), entity["end"] + context_window)
            entity["context"] = text[start:end]

        return entities

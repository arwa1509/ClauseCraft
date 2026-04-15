"""
Text Cleaner Module
====================
Cleans and normalizes extracted text from legal documents.
Handles encoding issues, OCR artifacts, and structural detection.
"""

import re
import unicodedata
from typing import Optional
from loguru import logger


class TextCleaner:
    """
    Cleans raw extracted text from legal documents.
    Detects document structure (sections, clauses, articles).
    """

    # Common legal section patterns
    SECTION_PATTERNS = [
        # "Section 1.", "Section 1:", "SECTION 1"
        re.compile(r"^(SECTION|Section|Sec\.?)\s+(\d+[\.\:]?\s*.*)", re.MULTILINE),
        # "Article 1", "ARTICLE I"
        re.compile(r"^(ARTICLE|Article)\s+([IVXLCDM\d]+[\.\:]?\s*.*)", re.MULTILINE),
        # "Clause 1", "CLAUSE 1.1"
        re.compile(r"^(CLAUSE|Clause)\s+([\d\.]+[\.\:]?\s*.*)", re.MULTILINE),
        # "Chapter 1", "CHAPTER I"
        re.compile(r"^(CHAPTER|Chapter)\s+([IVXLCDM\d]+[\.\:]?\s*.*)", re.MULTILINE),
        # "Part I", "PART 1"
        re.compile(r"^(PART|Part)\s+([IVXLCDM\d]+[\.\:]?\s*.*)", re.MULTILINE),
        # Numbered headings: "1.", "1.1", "1.1.1"
        re.compile(r"^(\d+(?:\.\d+)*)\.\s+([A-Z].*)", re.MULTILINE),
    ]

    # OCR artifact patterns
    OCR_ARTIFACTS = [
        (re.compile(r"[|]"), "I"),   # Pipe often mis-OCR'd as I
        (re.compile(r"(?<!\w)0(?=\w)"), "O"),  # Leading 0 as O
        (re.compile(r"\x0c"), "\n"),  # Form feed
    ]

    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps to raw text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Step 1: Fix encoding
        text = self._fix_encoding(text)

        # Step 2: Remove control characters (keep newlines and tabs)
        text = self._remove_control_chars(text)

        # Step 3: Fix OCR artifacts
        text = self._fix_ocr_artifacts(text)

        # Step 4: Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Step 5: Fix hyphenation (word split across lines)
        text = self._fix_hyphenation(text)

        # Step 6: Normalize whitespace
        text = self._normalize_whitespace(text)

        # Step 7: Remove repeated lines (common in scanned docs)
        text = self._remove_repeated_lines(text)

        return text.strip()

    def detect_sections(self, text: str) -> list[dict]:
        """
        Detect document sections/clauses/articles and their positions.

        Returns:
            List of dicts: [{"type": str, "number": str, "title": str, "start": int}]
        """
        sections = []

        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(text):
                section_type = match.group(1).strip().upper()
                section_id = match.group(2).strip() if match.lastindex >= 2 else ""

                sections.append({
                    "type": section_type,
                    "number": section_id.split()[0] if section_id else "",
                    "title": section_id,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Sort by position in text
        sections.sort(key=lambda s: s["start"])

        logger.debug(f"Detected {len(sections)} sections in text")
        return sections

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Handle common mis-encoded characters
        replacements = {
            "\u2018": "'", "\u2019": "'",  # Smart quotes
            "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "—",  # Dashes
            "\u2026": "...",               # Ellipsis
            "\xa0": " ",                    # Non-breaking space
            "\u200b": "",                   # Zero-width space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        return "".join(
            c for c in text
            if c in ("\n", "\t", "\r") or not unicodedata.category(c).startswith("C")
        )

    def _fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR misrecognitions."""
        for pattern, replacement in self.OCR_ARTIFACTS:
            text = pattern.sub(replacement, text)
        return text

    def _fix_hyphenation(self, text: str) -> str:
        """Rejoin words split across lines by hyphenation."""
        # "judg-\nment" → "judgment"
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        # Replace tabs with spaces
        text = text.replace("\t", "    ")
        # Collapse multiple spaces within lines
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _remove_repeated_lines(self, text: str) -> str:
        """Remove lines that repeat more than 3 times (common in scanned docs)."""
        lines = text.split("\n")
        counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped:
                counts[stripped] = counts.get(stripped, 0) + 1

        # Remove lines appearing more than 3 times
        filtered = []
        removed_lines = {k for k, v in counts.items() if v > 3 and len(k) < 100}
        for line in lines:
            if line.strip() not in removed_lines:
                filtered.append(line)

        return "\n".join(filtered)

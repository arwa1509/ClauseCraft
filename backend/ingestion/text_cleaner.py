"""
Text cleaning and structure detection utilities for legal documents.
"""

from __future__ import annotations

import re
import unicodedata

from loguru import logger


class TextCleaner:
    """
    Clean raw text while preserving as much legal formatting as possible.
    """

    SECTION_PATTERNS = [
        re.compile(r"^(SECTION|Section|Sec\.?)\s+(\d+[\.:]?\s*.*)", re.MULTILINE),
        re.compile(r"^(ARTICLE|Article)\s+([IVXLCDM\d]+[\.:]?\s*.*)", re.MULTILINE),
        re.compile(r"^(CLAUSE|Clause)\s+([\d\.]+[\.:]?\s*.*)", re.MULTILINE),
        re.compile(r"^(CHAPTER|Chapter)\s+([IVXLCDM\d]+[\.:]?\s*.*)", re.MULTILINE),
        re.compile(r"^(PART|Part)\s+([IVXLCDM\d]+[\.:]?\s*.*)", re.MULTILINE),
        re.compile(r"^(\d+(?:\.\d+)*)\.\s+([A-Z].*)", re.MULTILINE),
    ]

    OCR_ARTIFACTS = [
        (re.compile(r"\x0c"), "\n"),
    ]

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self._fix_encoding(text)
        text = self._remove_control_chars(text)
        text = self._fix_ocr_artifacts(text)
        text = unicodedata.normalize("NFKC", text)
        text = self._fix_hyphenation(text)
        text = self._normalize_whitespace(text)
        text = self._remove_repeated_lines(text)
        return text.strip()

    def detect_sections(self, text: str) -> list[dict]:
        sections = []
        seen_positions = set()

        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(text):
                section_type = match.group(1).strip().upper()
                section_id = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
                key = (match.start(), match.end(), match.group(0).strip().lower())
                if key in seen_positions:
                    continue
                seen_positions.add(key)
                sections.append(
                    {
                        "type": section_type,
                        "number": section_id.split()[0] if section_id else "",
                        "title": section_id,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        sections.sort(key=lambda section: section["start"])
        logger.debug(f"Detected {len(sections)} sections in text")
        return sections

    def _fix_encoding(self, text: str) -> str:
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\xa0": " ",
            "\u200b": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_control_chars(self, text: str) -> str:
        return "".join(
            char
            for char in text
            if char in ("\n", "\t", "\r") or not unicodedata.category(char).startswith("C")
        )

    def _fix_ocr_artifacts(self, text: str) -> str:
        for pattern, replacement in self.OCR_ARTIFACTS:
            text = pattern.sub(replacement, text)
        return text

    def _fix_hyphenation(self, text: str) -> str:
        return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    def _normalize_whitespace(self, text: str) -> str:
        text = text.replace("\t", "    ")
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _remove_repeated_lines(self, text: str) -> str:
        lines = text.split("\n")
        counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped:
                counts[stripped] = counts.get(stripped, 0) + 1

        removed_lines = {line for line, count in counts.items() if count > 3 and len(line) < 100}
        filtered = [line for line in lines if line.strip() not in removed_lines]
        return "\n".join(filtered)

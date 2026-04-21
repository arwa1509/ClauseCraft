"""
Structure-Aware PDF Parser (LlamaParse)
========================================
Refactored ingestion layer that uses LlamaParse as the primary parser.
LlamaParse preserves:
  - Section / Clause hierarchies  → output as Markdown headings
  - Tables                        → output as Markdown pipe tables
  - Lists, bullet points          → output as Markdown lists

Output contract (same shape as original pdfplumber parser so chunker.py
and the rest of the pipeline remain unchanged):

    [
        {
            "page_num": int,        # 1-indexed (or chunk index for LlamaParse)
            "text": str,            # Markdown-formatted text
            "metadata": {
                "source_file": str,
                "total_pages": int,
                "parser": "llamaparse" | "pdfplumber" | "pypdf2",
                "has_tables": bool,
                "sections": list[str],   # detected section/heading titles
            }
        },
        ...
    ]

Environment Variables:
    LLAMA_CLOUD_API_KEY   – required for LlamaParse
    LLAMAPARSE_LANGUAGE   – default "en" (ISO 639-1 code)
    LLAMAPARSE_RESULT_TYPE – "markdown" (default) | "text"
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# LlamaParse (primary parser)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from llama_parse import LlamaParse
    from llama_index.core import SimpleDirectoryReader
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LlamaParse = None
    SimpleDirectoryReader = None
    LLAMAPARSE_AVAILABLE = False
    logger.warning(
        "llama-parse / llama-index-core not installed. "
        "Falling back to pdfplumber/PyPDF2."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Legacy fallback parsers (kept for robustness)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed.")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PdfReader = None
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not installed.")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")
LLAMAPARSE_LANGUAGE: str = os.getenv("LLAMAPARSE_LANGUAGE", "en")
LLAMAPARSE_RESULT_TYPE: str = os.getenv("LLAMAPARSE_RESULT_TYPE", "markdown")

# Regex patterns for structural analysis of the Markdown output
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_SECTION_KEYWORDS = re.compile(
    r"\b(section|clause|article|paragraph|schedule|annexure|exhibit)\s*[\d\.]+",
    re.IGNORECASE,
)


def _extract_section_titles(text: str) -> list[str]:
    """Extract heading titles from Markdown text."""
    return [m.group(2).strip() for m in _HEADING_RE.finditer(text)]


def _has_tables(text: str) -> bool:
    """Return True if the text contains at least one Markdown table row."""
    return bool(_TABLE_RE.search(text))


def _split_markdown_by_heading(
    content: str,
    source_file: str,
    total_pages: int,
) -> list[dict]:
    """
    Split LlamaParse markdown output into logical chunks by top-level heading.
    Each heading (# or ##) becomes a 'page' in the output contract.
    If no headings are found, the content is returned as a single page.
    """
    # Split on any Markdown heading (H1–H3 treated as section boundaries)
    parts = re.split(r"(?=^#{1,3} )", content, flags=re.MULTILINE)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [{
            "page_num": 1,
            "text": content.strip(),
            "metadata": {
                "source_file": source_file,
                "total_pages": total_pages,
                "parser": "llamaparse",
                "has_tables": _has_tables(content),
                "sections": _extract_section_titles(content),
            },
        }]

    pages = []
    for idx, part in enumerate(parts, 1):
        pages.append({
            "page_num": idx,
            "text": part,
            "metadata": {
                "source_file": source_file,
                "total_pages": len(parts),
                "parser": "llamaparse",
                "has_tables": _has_tables(part),
                "sections": _extract_section_titles(part),
            },
        })
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Main PDFParser class
# ──────────────────────────────────────────────────────────────────────────────

class PDFParser:
    """
    Structure-aware PDF parser with LlamaParse as the primary engine.

    Priority chain:
        1. LlamaParse (API-based, cloud) – best structural fidelity
        2. pdfplumber (local)            – good for flat text + basic tables
        3. PyPDF2 (local)               – last resort
    """

    def __init__(self, force_fallback: bool = False):
        """
        Args:
            force_fallback: Skip LlamaParse and use pdfplumber/PyPDF2 directly.
                            Useful for offline/air-gapped environments.
        """
        self.force_fallback = force_fallback

        # Header / footer noise patterns (used only for pdfplumber / PyPDF2 paths)
        self._header_pattern = re.compile(
            r"^(page\s*\d+|confidential|draft|privileged|©.*\d{4}).*$",
            re.IGNORECASE | re.MULTILINE,
        )
        self._footer_pattern = re.compile(
            r"^.*?(page\s*\d+\s*of\s*\d+|\d+\s*$)",
            re.IGNORECASE | re.MULTILINE,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, file_path: str | Path) -> list[dict]:
        """
        Parse a PDF and return a list of structured page dicts.

        Returns:
            List[Dict] with keys: page_num, text (markdown), metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info(f"Parsing PDF: {file_path.name}")

        # ── LlamaParse (primary) ──────────────────────────────────────────────
        if (
            LLAMAPARSE_AVAILABLE
            and LLAMA_CLOUD_API_KEY
            and not self.force_fallback
        ):
            try:
                return self._parse_with_llamaparse(file_path)
            except Exception as exc:
                logger.warning(
                    f"LlamaParse failed ({exc}). Falling back to pdfplumber."
                )

        # ── pdfplumber (secondary) ────────────────────────────────────────────
        if PDFPLUMBER_AVAILABLE:
            logger.info("Using pdfplumber parser.")
            return self._parse_with_pdfplumber(file_path)

        # ── PyPDF2 (tertiary) ─────────────────────────────────────────────────
        if PYPDF2_AVAILABLE:
            logger.warning("Using PyPDF2 fallback (pdfplumber not available).")
            return self._parse_with_pypdf2(file_path)

        raise ImportError(
            "No PDF library available. "
            "Set LLAMA_CLOUD_API_KEY and install llama-parse, "
            "or install pdfplumber / PyPDF2."
        )

    def parse_to_full_text(self, file_path: str | Path) -> str:
        """Parse PDF and return the full concatenated Markdown text."""
        pages = self.parse(file_path)
        return "\n\n".join(p["text"] for p in pages if p["text"])

    # ──────────────────────────────────────────────────────────────────────────
    # Private: LlamaParse
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_with_llamaparse(self, file_path: Path) -> list[dict]:
        """
        Parse using LlamaParse cloud API.

        LlamaParse returns rich Markdown with:
          - ## headings for clauses/sections
          - | col | col | tables preserved
          - Nested list structures
        """
        logger.info("  [LlamaParse] Submitting document to LlamaParse...")

        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type=LLAMAPARSE_RESULT_TYPE,   # "markdown"
            language=LLAMAPARSE_LANGUAGE,
            verbose=False,
            # Instruct the parser to treat legal section markers as headings
            parsing_instruction=(
                "This is a legal document. "
                "Treat 'Section', 'Clause', 'Article', 'Schedule', "
                "and 'Annexure' markers as Markdown headings (##). "
                "Preserve all tabular data as Markdown pipe tables. "
                "Do not summarise or omit any content."
            ),
            # Premium parsing for complex layouts
            premium_mode=True,
        )

        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader(
            input_files=[str(file_path)],
            file_extractor=file_extractor,
        )

        # LlamaIndex returns a list of Document objects
        docs = reader.load_data()
        logger.info(f"  [LlamaParse] Received {len(docs)} document chunk(s).")

        if not docs:
            raise ValueError("LlamaParse returned no documents.")

        # Concatenate all doc text into one Markdown string
        full_markdown = "\n\n".join(d.text for d in docs if d.text)

        # Split by headings to produce page-level chunks
        pages = _split_markdown_by_heading(
            content=full_markdown,
            source_file=file_path.name,
            total_pages=len(docs),
        )

        logger.info(
            f"  [LlamaParse] Produced {len(pages)} structured section(s) "
            f"from {file_path.name}"
        )
        return pages

    # ──────────────────────────────────────────────────────────────────────────
    # Private: pdfplumber (enhanced with table → Markdown conversion)
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_with_pdfplumber(self, file_path: Path) -> list[dict]:
        """Parse with pdfplumber, converting tables to Markdown pipe syntax."""
        pages = []
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"  Total pages: {total_pages}")

            for i, page in enumerate(pdf.pages):
                # Extract tables first
                tables_md = self._extract_tables_as_markdown(page)

                # Extract plain text
                text = page.extract_text() or ""
                text = self._clean_page_text(text)

                # Attach Markdown tables after the page text
                if tables_md:
                    text = text + "\n\n" + "\n\n".join(tables_md)

                # Annotate legal section headings
                text = self._annotate_sections(text)

                sections = _extract_section_titles(text)
                has_tbl = bool(tables_md)

                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "metadata": {
                        "source_file": file_path.name,
                        "total_pages": total_pages,
                        "page_width": page.width,
                        "page_height": page.height,
                        "parser": "pdfplumber",
                        "has_tables": has_tbl,
                        "sections": sections,
                    },
                })

        logger.info(f"  Extracted {len(pages)} pages from {file_path.name}")
        return pages

    def _extract_tables_as_markdown(self, page) -> list[str]:
        """Convert pdfplumber tables to Markdown pipe table format."""
        raw_tables = page.extract_tables() or []
        md_tables: list[str] = []
        for table in raw_tables:
            if not table:
                continue
            rows: list[str] = []
            for r_idx, row in enumerate(table):
                cells = [str(c or "").replace("\n", " ").strip() for c in row]
                rows.append("| " + " | ".join(cells) + " |")
                if r_idx == 0:
                    # Header separator row
                    rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            md_tables.append("\n".join(rows))
        return md_tables

    def _annotate_sections(self, text: str) -> str:
        """
        Heuristically promote legal section markers to Markdown headings.
        E.g. "Section 3.2 – Penalty" → "## Section 3.2 – Penalty"
        """
        def _to_heading(m: re.Match) -> str:
            # Avoid double-annotating existing headings
            start = m.start()
            preceding = text[max(0, start - 2): start]
            if preceding.strip().startswith("#"):
                return m.group(0)
            return f"\n## {m.group(0).strip()}\n"

        return _SECTION_KEYWORDS.sub(_to_heading, text)

    # ──────────────────────────────────────────────────────────────────────────
    # Private: PyPDF2 fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_with_pypdf2(self, file_path: Path) -> list[dict]:
        """Parse using PyPDF2 as last-resort fallback."""
        pages = []
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        logger.info(f"  Total pages: {total_pages}")

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = self._clean_page_text(text)
            text = self._annotate_sections(text)

            pages.append({
                "page_num": i + 1,
                "text": text,
                "metadata": {
                    "source_file": file_path.name,
                    "total_pages": total_pages,
                    "parser": "pypdf2",
                    "has_tables": False,
                    "sections": _extract_section_titles(text),
                },
            })

        logger.info(f"  Extracted {len(pages)} pages from {file_path.name}")
        return pages

    # ──────────────────────────────────────────────────────────────────────────
    # Shared text cleaning (for non-LlamaParse paths)
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_page_text(self, text: str) -> str:
        """Remove noise: headers, footers, page numbers, watermarks."""
        if not text:
            return ""
        text = self._header_pattern.sub("", text)
        text = self._footer_pattern.sub("", text)
        text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(
            r"(CONFIDENTIAL|DRAFT|PRIVILEGED)\s*", "", text, flags=re.IGNORECASE
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

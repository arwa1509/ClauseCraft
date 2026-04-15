"""
PDF Parser Module
==================
Converts PDF files to clean text with page-level metadata.
Uses pdfplumber for robust text extraction including tables and complex layouts.
"""

import re
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.warning("pdfplumber not installed. PDF parsing will be unavailable.")

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
    logger.warning("PyPDF2 not installed. Fallback PDF parsing unavailable.")


class PDFParser:
    """
    Extracts text from PDF files with page-level metadata.
    Primary: pdfplumber (better for complex layouts, tables).
    Fallback: PyPDF2.
    """

    def __init__(self):
        self.header_pattern = re.compile(
            r"^(page\s*\d+|confidential|draft|privileged|©.*\d{4}).*$",
            re.IGNORECASE | re.MULTILINE,
        )
        self.footer_pattern = re.compile(
            r"^.*?(page\s*\d+\s*of\s*\d+|\d+\s*$)",
            re.IGNORECASE | re.MULTILINE,
        )

    def parse(self, file_path: str | Path) -> list[dict]:
        """
        Parse a PDF file and return a list of page dicts.

        Returns:
            List of dicts: [{"page_num": int, "text": str, "metadata": dict}]
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info(f"Parsing PDF: {file_path.name}")

        # Try pdfplumber first (better quality)
        if pdfplumber is not None:
            return self._parse_with_pdfplumber(file_path)
        elif PdfReader is not None:
            logger.warning("Using PyPDF2 fallback (pdfplumber not available)")
            return self._parse_with_pypdf2(file_path)
        else:
            raise ImportError(
                "No PDF library available. Install pdfplumber or PyPDF2."
            )

    def _parse_with_pdfplumber(self, file_path: Path) -> list[dict]:
        """Parse PDF using pdfplumber."""
        pages = []
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"  Total pages: {total_pages}")

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                # Clean the extracted text
                text = self._clean_page_text(text, i + 1, total_pages)

                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "metadata": {
                        "source_file": file_path.name,
                        "total_pages": total_pages,
                        "page_width": page.width,
                        "page_height": page.height,
                    },
                })

        logger.info(f"  Extracted {len(pages)} pages from {file_path.name}")
        return pages

    def _parse_with_pypdf2(self, file_path: Path) -> list[dict]:
        """Parse PDF using PyPDF2 as fallback."""
        pages = []
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        logger.info(f"  Total pages: {total_pages}")

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = self._clean_page_text(text, i + 1, total_pages)

            pages.append({
                "page_num": i + 1,
                "text": text,
                "metadata": {
                    "source_file": file_path.name,
                    "total_pages": total_pages,
                },
            })

        logger.info(f"  Extracted {len(pages)} pages from {file_path.name}")
        return pages

    def _clean_page_text(
        self, text: str, page_num: int, total_pages: int
    ) -> str:
        """
        Remove common noise from extracted page text:
        - Headers/footers
        - Page numbers
        - Watermarks
        - Excessive whitespace
        """
        if not text:
            return ""

        # Remove headers
        text = self.header_pattern.sub("", text)

        # Remove footers
        text = self.footer_pattern.sub("", text)

        # Remove standalone page numbers
        text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

        # Remove watermark-like repeated text
        text = re.sub(r"(CONFIDENTIAL|DRAFT|PRIVILEGED)\s*", "", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        return text

    def parse_to_full_text(self, file_path: str | Path) -> str:
        """Parse PDF and return the full concatenated text."""
        pages = self.parse(file_path)
        return "\n\n".join(p["text"] for p in pages if p["text"])

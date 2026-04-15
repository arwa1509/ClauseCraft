"""
Structure-Aware Chunker
========================
Splits legal documents into semantically meaningful chunks while
preserving context and metadata. Supports section/clause/paragraph splitting.
"""

import re
import hashlib
from typing import Optional
from loguru import logger

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from ingestion.text_cleaner import TextCleaner


class Chunk:
    """Represents a document chunk with metadata."""

    def __init__(
        self,
        text: str,
        chunk_id: str,
        doc_name: str,
        page_num: Optional[int] = None,
        section: Optional[str] = None,
        section_num: Optional[str] = None,
        chunk_index: int = 0,
        total_chunks: int = 0,
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.doc_name = doc_name
        self.page_num = page_num
        self.section = section
        self.section_num = section_num
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": {
                "doc_name": self.doc_name,
                "page_num": self.page_num,
                "section": self.section,
                "section_num": self.section_num,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
            },
        }


class StructureAwareChunker:
    """
    Chunks legal documents using structure-aware splitting.

    Strategy:
    1. Try to split by detected sections (Section, Article, Clause)
    2. Within sections, split by paragraphs
    3. If paragraphs too large, use sliding window
    4. Always preserve context with overlap
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.cleaner = TextCleaner()

    def chunk_document(
        self,
        text: str,
        doc_name: str,
        pages: Optional[list[dict]] = None,
    ) -> list[Chunk]:
        """
        Chunk a document into semantically meaningful pieces.

        Args:
            text: Full document text
            doc_name: Name of the source document
            pages: Optional list of page dicts with page_num and text

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        # Detect sections
        sections = self.cleaner.detect_sections(text)

        if sections:
            logger.info(f"  Found {len(sections)} sections, using structure-aware chunking")
            chunks = self._chunk_by_sections(text, doc_name, sections, pages)
        else:
            logger.info("  No sections detected, using paragraph-based chunking")
            chunks = self._chunk_by_paragraphs(text, doc_name, pages)

        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.info(f"  Created {len(chunks)} chunks for {doc_name}")
        return chunks

    def _chunk_by_sections(
        self,
        text: str,
        doc_name: str,
        sections: list[dict],
        pages: Optional[list[dict]] = None,
    ) -> list[Chunk]:
        """Split text by detected section boundaries."""
        chunks = []
        chunk_idx = 0

        for i, section in enumerate(sections):
            # Get section text (from this section start to next section start)
            start = section["start"]
            end = sections[i + 1]["start"] if i + 1 < len(sections) else len(text)
            section_text = text[start:end].strip()

            if not section_text:
                continue

            # If section is small enough, keep as single chunk
            if len(section_text) <= self.chunk_size:
                if len(section_text) >= self.min_chunk_size:
                    page_num = self._find_page_num(start, text, pages)
                    chunk_id = self._make_chunk_id(doc_name, chunk_idx)
                    chunks.append(Chunk(
                        text=section_text,
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        page_num=page_num,
                        section=section.get("type"),
                        section_num=section.get("number"),
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1
            else:
                # Section too large — split by paragraphs within section
                sub_chunks = self._split_large_text(
                    section_text, doc_name, chunk_idx,
                    section_type=section.get("type"),
                    section_num=section.get("number"),
                    base_offset=start,
                    pages=pages,
                    full_text=text,
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)

        # Handle text before first section
        if sections and sections[0]["start"] > 0:
            preamble = text[:sections[0]["start"]].strip()
            if len(preamble) >= self.min_chunk_size:
                pre_chunks = self._split_large_text(
                    preamble, doc_name, 0,
                    section_type="PREAMBLE",
                    section_num="0",
                    base_offset=0,
                    pages=pages,
                    full_text=text,
                )
                # Prepend and reindex
                for i, c in enumerate(chunks):
                    c.chunk_index = i + len(pre_chunks)
                chunks = pre_chunks + chunks

        return chunks

    def _chunk_by_paragraphs(
        self,
        text: str,
        doc_name: str,
        pages: Optional[list[dict]] = None,
    ) -> list[Chunk]:
        """Split text by paragraphs, merging small ones."""
        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_text = ""
        chunk_idx = 0

        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current and start new
            if current_text and len(current_text) + len(para) + 2 > self.chunk_size:
                if len(current_text) >= self.min_chunk_size:
                    chunk_id = self._make_chunk_id(doc_name, chunk_idx)
                    page_num = self._find_page_num(
                        text.find(current_text[:50]), text, pages
                    )
                    chunks.append(Chunk(
                        text=current_text,
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        page_num=page_num,
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1

                # Keep overlap from end of current text
                overlap_text = current_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_text = overlap_text + "\n\n" + para if overlap_text else para
            else:
                current_text = current_text + "\n\n" + para if current_text else para

        # Save last chunk
        if current_text and len(current_text) >= self.min_chunk_size:
            chunk_id = self._make_chunk_id(doc_name, chunk_idx)
            chunks.append(Chunk(
                text=current_text,
                chunk_id=chunk_id,
                doc_name=doc_name,
                chunk_index=chunk_idx,
            ))

        return chunks

    def _split_large_text(
        self,
        text: str,
        doc_name: str,
        start_idx: int,
        section_type: Optional[str] = None,
        section_num: Optional[str] = None,
        base_offset: int = 0,
        pages: Optional[list[dict]] = None,
        full_text: Optional[str] = None,
    ) -> list[Chunk]:
        """Split large text using sliding window with overlap."""
        chunks = []
        idx = start_idx

        # Try paragraph-based first
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > self.chunk_size:
                if len(current) >= self.min_chunk_size:
                    chunk_id = self._make_chunk_id(doc_name, idx)
                    page_num = self._find_page_num(
                        base_offset, full_text or text, pages
                    )
                    chunks.append(Chunk(
                        text=current,
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        page_num=page_num,
                        section=section_type,
                        section_num=section_num,
                        chunk_index=idx,
                    ))
                    idx += 1

                overlap = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = overlap + "\n\n" + para if overlap else para
            else:
                current = current + "\n\n" + para if current else para

        if current and len(current) >= self.min_chunk_size:
            chunk_id = self._make_chunk_id(doc_name, idx)
            chunks.append(Chunk(
                text=current,
                chunk_id=chunk_id,
                doc_name=doc_name,
                section=section_type,
                section_num=section_num,
                chunk_index=idx,
            ))

        # If no paragraphs found, fall back to character-based sliding window
        if not chunks and len(text) >= self.min_chunk_size:
            pos = 0
            while pos < len(text):
                end = min(pos + self.chunk_size, len(text))
                chunk_text = text[pos:end].strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_id = self._make_chunk_id(doc_name, idx)
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        section=section_type,
                        section_num=section_num,
                        chunk_index=idx,
                    ))
                    idx += 1
                pos += self.chunk_size - self.chunk_overlap

        return chunks

    def _make_chunk_id(self, doc_name: str, chunk_idx: int) -> str:
        """Generate a unique chunk ID."""
        raw = f"{doc_name}_{chunk_idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _find_page_num(
        self,
        char_offset: int,
        full_text: str,
        pages: Optional[list[dict]],
    ) -> Optional[int]:
        """Estimate page number from character offset."""
        if not pages or char_offset < 0:
            return None

        running_offset = 0
        for page in pages:
            page_len = len(page.get("text", "")) + 2  # +2 for separator
            if running_offset + page_len > char_offset:
                return page.get("page_num")
            running_offset += page_len

        return pages[-1].get("page_num") if pages else None

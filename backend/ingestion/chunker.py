"""
Structure-aware chunking for legal documents.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from loguru import logger

from config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE
from ingestion.text_cleaner import TextCleaner


class Chunk:
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
        char_start: Optional[int] = None,
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.doc_name = doc_name
        self.page_num = page_num
        self.section = section
        self.section_num = section_num
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.char_start = char_start

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
        if not text.strip():
            return []

        sections = self.cleaner.detect_sections(text)
        if sections:
            logger.info(f"  Found {len(sections)} sections, using structure-aware chunking")
            chunks = self._chunk_by_sections(text, doc_name, sections, pages)
        else:
            logger.info("  No sections detected, using paragraph-based chunking")
            chunks = self._chunk_by_paragraphs(text, doc_name, pages)

        self._finalize_chunks(chunks, doc_name, text, pages)
        logger.info(f"  Created {len(chunks)} chunks for {doc_name}")
        return chunks

    def _finalize_chunks(
        self,
        chunks: list[Chunk],
        doc_name: str,
        full_text: str,
        pages: Optional[list[dict]],
    ):
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.chunk_id = self._make_chunk_id(doc_name, index)
            chunk.total_chunks = total_chunks
            if chunk.page_num is None:
                chunk.page_num = self._find_page_num(
                    chunk.char_start if chunk.char_start is not None else 0,
                    full_text,
                    pages,
                )

    def _chunk_by_sections(
        self,
        text: str,
        doc_name: str,
        sections: list[dict],
        pages: Optional[list[dict]] = None,
    ) -> list[Chunk]:
        chunks = []

        if sections and sections[0]["start"] > 0:
            preamble = text[: sections[0]["start"]].strip()
            if len(preamble) >= self.min_chunk_size:
                chunks.extend(
                    self._split_large_text(
                        preamble,
                        doc_name,
                        0,
                        section_type="PREAMBLE",
                        section_num="0",
                        base_offset=0,
                        pages=pages,
                        full_text=text,
                    )
                )

        for i, section in enumerate(sections):
            start = section["start"]
            end = sections[i + 1]["start"] if i + 1 < len(sections) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue

            if len(section_text) <= self.chunk_size:
                if len(section_text) >= self.min_chunk_size:
                    chunks.append(
                        Chunk(
                            text=section_text,
                            chunk_id="",
                            doc_name=doc_name,
                            page_num=self._find_page_num(start, text, pages),
                            section=section.get("type"),
                            section_num=section.get("number"),
                            char_start=start,
                        )
                    )
            else:
                chunks.extend(
                    self._split_large_text(
                        section_text,
                        doc_name,
                        0,
                        section_type=section.get("type"),
                        section_num=section.get("number"),
                        base_offset=start,
                        pages=pages,
                        full_text=text,
                    )
                )

        return chunks

    def _chunk_by_paragraphs(
        self,
        text: str,
        doc_name: str,
        pages: Optional[list[dict]] = None,
    ) -> list[Chunk]:
        paragraphs = self._extract_paragraphs(text, 0)
        chunks = []
        current_parts = []
        current_start = None

        for para_text, para_start in paragraphs:
            proposed = "\n\n".join(current_parts + [para_text]) if current_parts else para_text
            if current_parts and len(proposed) > self.chunk_size:
                chunk_text = "\n\n".join(current_parts)
                if len(chunk_text) >= self.min_chunk_size and current_start is not None:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id="",
                            doc_name=doc_name,
                            page_num=self._find_page_num(current_start, text, pages),
                            char_start=current_start,
                        )
                    )
                overlap_text = chunk_text[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current_parts = [part for part in [overlap_text, para_text] if part]
                current_start = max(current_start or 0, len(text[:para_start]) - len(overlap_text))
            else:
                if current_start is None:
                    current_start = para_start
                current_parts.append(para_text)

        if current_parts and current_start is not None:
            chunk_text = "\n\n".join(current_parts)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id="",
                        doc_name=doc_name,
                        page_num=self._find_page_num(current_start, text, pages),
                        char_start=current_start,
                    )
                )

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
        del start_idx
        chunks = []
        paragraphs = self._extract_paragraphs(text, base_offset)
        current_parts = []
        current_start = None

        for para_text, para_start in paragraphs:
            proposed = "\n\n".join(current_parts + [para_text]) if current_parts else para_text
            if current_parts and len(proposed) > self.chunk_size:
                chunk_text = "\n\n".join(current_parts)
                if len(chunk_text) >= self.min_chunk_size and current_start is not None:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id="",
                            doc_name=doc_name,
                            page_num=self._find_page_num(current_start, full_text or text, pages),
                            section=section_type,
                            section_num=section_num,
                            char_start=current_start,
                        )
                    )
                overlap_text = chunk_text[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current_parts = [part for part in [overlap_text, para_text] if part]
                current_start = max(base_offset, para_start - len(overlap_text))
            else:
                if current_start is None:
                    current_start = para_start
                current_parts.append(para_text)

        if current_parts and current_start is not None:
            chunk_text = "\n\n".join(current_parts)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id="",
                        doc_name=doc_name,
                        page_num=self._find_page_num(current_start, full_text or text, pages),
                        section=section_type,
                        section_num=section_num,
                        char_start=current_start,
                    )
                )

        if not chunks and len(text) >= self.min_chunk_size:
            pos = 0
            while pos < len(text):
                end = min(pos + self.chunk_size, len(text))
                chunk_text = text[pos:end].strip()
                if len(chunk_text) >= self.min_chunk_size:
                    absolute_start = base_offset + pos
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id="",
                            doc_name=doc_name,
                            page_num=self._find_page_num(
                                absolute_start,
                                full_text or text,
                                pages,
                            ),
                            section=section_type,
                            section_num=section_num,
                            char_start=absolute_start,
                        )
                    )
                pos += max(1, self.chunk_size - self.chunk_overlap)

        return chunks

    def _extract_paragraphs(self, text: str, base_offset: int) -> list[tuple[str, int]]:
        paragraphs = []
        for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
            para_text = match.group(0).strip()
            if para_text:
                paragraphs.append((para_text, base_offset + match.start()))
        return paragraphs

    def _make_chunk_id(self, doc_name: str, chunk_idx: int) -> str:
        raw = f"{doc_name}_{chunk_idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _find_page_num(
        self,
        char_offset: int,
        full_text: str,
        pages: Optional[list[dict]],
    ) -> Optional[int]:
        del full_text
        if not pages or char_offset < 0:
            return None

        running_offset = 0
        for page in pages:
            page_len = len(page.get("text", ""))
            if running_offset + page_len >= char_offset:
                return page.get("page_num")
            running_offset += page_len + 2
        return pages[-1].get("page_num") if pages else None

"""
Ingestion API Router
=====================
Endpoints for uploading, processing, and managing legal documents.
"""

import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from loguru import logger

from config import DOCUMENTS_DIR, PROCESSED_DIR, METADATA_PATH
from ingestion.pdf_parser import PDFParser
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import StructureAwareChunker

router = APIRouter()

# Module-level instances
pdf_parser = PDFParser()
text_cleaner = TextCleaner()
chunker = StructureAwareChunker()
TRACE_PATH = PROCESSED_DIR / "ingestion_trace.json"

# Processing state
_processing_status = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "documents": [],
    "current_stage": "idle",
    "metrics": {},
}


class ProcessingStatus(BaseModel):
    status: str
    progress: float
    message: str
    documents: list
    current_stage: str = "idle"
    metrics: dict = {}


def _get_all_chunks() -> list[dict]:
    """Load all processed chunks from metadata file."""
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return []


def _save_all_chunks(chunks: list[dict]):
    """Save all chunks to metadata file."""
    METADATA_PATH.write_text(json.dumps(chunks, indent=2, default=str))


def _load_traces() -> list[dict]:
    if TRACE_PATH.exists():
        try:
            return json.loads(TRACE_PATH.read_text())
        except Exception as exc:
            logger.warning(f"Failed to load ingestion traces: {exc}")
    return []


def _save_traces(traces: list[dict]):
    TRACE_PATH.write_text(json.dumps(traces, indent=2, default=str))


def _build_metrics_snapshot() -> dict:
    chunks = _get_all_chunks()
    docs = []
    for ext in ["*.pdf", "*.txt", "*.json"]:
        docs.extend(DOCUMENTS_DIR.glob(ext))

    doc_type_counts = Counter((doc.suffix.lower() or "unknown") for doc in docs)
    chunks_by_doc = Counter(chunk.get("metadata", {}).get("doc_name", "Unknown") for chunk in chunks)
    sections_by_doc = Counter()
    page_refs = set()
    max_chunk_length = 0
    total_chunk_chars = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        total_chunk_chars += len(text)
        max_chunk_length = max(max_chunk_length, len(text))
        if metadata.get("section"):
            sections_by_doc[metadata.get("doc_name", "Unknown")] += 1
        if metadata.get("page_num") is not None:
            page_refs.add((metadata.get("doc_name", "Unknown"), metadata.get("page_num")))

    entity_stats = {"total_entities": 0, "total_references": 0, "entities_by_label": {}}
    entity_path = PROCESSED_DIR / "entity_index.json"
    if entity_path.exists():
        try:
            from ner.entity_index import EntityIndex

            entity_stats = EntityIndex().get_stats()
        except Exception as exc:
            logger.warning(f"Failed to load entity stats for metrics snapshot: {exc}")

    per_document = []
    for doc in sorted(docs, key=lambda item: item.name.lower()):
        per_document.append(
            {
                "name": doc.name,
                "type": doc.suffix.lower(),
                "size_bytes": doc.stat().st_size,
                "chunks": chunks_by_doc.get(doc.name, 0),
                "sections": sections_by_doc.get(doc.name, 0),
            }
        )

    avg_chunk_chars = round(total_chunk_chars / len(chunks), 1) if chunks else 0.0
    faiss_index = PROCESSED_DIR / "faiss_index.bin"
    npy_index = PROCESSED_DIR / "faiss_index.bin.npy"

    return {
        "documents_total": len(docs),
        "chunks_total": len(chunks),
        "average_chunk_chars": avg_chunk_chars,
        "max_chunk_chars": max_chunk_length,
        "documents_by_type": dict(doc_type_counts),
        "entity_total": entity_stats.get("total_entities", 0),
        "entity_references_total": entity_stats.get("total_references", 0),
        "entities_by_label": entity_stats.get("entities_by_label", {}),
        "page_citations_total": len(page_refs),
        "vector_index": {
            "present": faiss_index.exists() or npy_index.exists(),
            "chunk_vectors": len(chunks),
        },
        "per_document": per_document,
    }


def _preview(text: str, limit: int = 240) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _build_document_trace(
    file_path: Path,
    pages: Optional[list[dict]],
    raw_text: str,
    clean_text: str,
    chunk_dicts: list[dict],
) -> dict:
    parser = "text"
    page_samples = []
    section_titles = []
    if pages:
        parser = pages[0].get("metadata", {}).get("parser", "pdf")
        for page in pages[:8]:
            metadata = page.get("metadata", {})
            page_samples.append(
                {
                    "page_num": page.get("page_num"),
                    "char_count": len(page.get("text", "")),
                    "has_tables": metadata.get("has_tables", False),
                    "sections": metadata.get("sections", []),
                    "preview": _preview(page.get("text", "")),
                }
            )
            section_titles.extend(metadata.get("sections", []))

    chunk_samples = []
    for chunk in chunk_dicts[:12]:
        meta = chunk.get("metadata", {})
        chunk_samples.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "page_num": meta.get("page_num"),
                "section": meta.get("section"),
                "section_num": meta.get("section_num"),
                "char_count": len(chunk.get("text", "")),
                "preview": _preview(chunk.get("text", "")),
            }
        )

    return {
        "document_name": file_path.name,
        "document_type": file_path.suffix.lower(),
        "parser": parser,
        "raw_char_count": len(raw_text),
        "clean_char_count": len(clean_text),
        "page_count": len(pages) if pages else 1,
        "chunk_count": len(chunk_dicts),
        "detected_sections": sorted({title for title in section_titles if title})[:20],
        "raw_text_preview": _preview(raw_text),
        "clean_text_preview": _preview(clean_text),
        "page_samples": page_samples,
        "chunk_samples": chunk_samples,
        "entity_total": 0,
        "entity_labels": {},
        "sample_entities": [],
    }


def _refresh_runtime_pipeline():
    """
    Reload live in-memory indices after a full indexing run.

    This keeps the API query pipeline in sync with newly processed documents
    without requiring an application restart.
    """
    try:
        from main import _pipeline_components

        if not _pipeline_components:
            return

        vector_store = _pipeline_components.get("vector_store")
        if vector_store is not None:
            vector_store.index = None
            vector_store._embeddings = None
            vector_store.metadata = []
            vector_store._load()

        entity_index = _pipeline_components.get("entity_index")
        if entity_index is not None:
            entity_index._load()

        dense_retriever = _pipeline_components.get("dense_retriever")
        if dense_retriever is not None:
            dense_retriever.rebuild_bm25()

        logger.info("Refreshed live pipeline indices after ingestion")
    except Exception as exc:
        logger.warning(f"Could not refresh live pipeline indices: {exc}")


def _process_document(file_path: Path) -> tuple[list[dict], dict]:
    """
    Process a single document through the full ingestion pipeline.

    Steps:
    1. Parse (PDF → text, or read TXT/JSON)
    2. Clean text
    3. Chunk into structured pieces
    4. Return chunk dicts
    """
    suffix = file_path.suffix.lower()
    doc_name = file_path.name

    logger.info(f"📄 Processing document: {doc_name}")

    # Step 1: Parse
    if suffix == ".pdf":
        pages = pdf_parser.parse(file_path)
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
    elif suffix == ".txt":
        full_text = file_path.read_text(encoding="utf-8", errors="ignore")
        pages = None
    elif suffix == ".json":
        data = json.loads(file_path.read_text())
        if isinstance(data, dict):
            full_text = data.get("text", data.get("content", json.dumps(data)))
        elif isinstance(data, list):
            full_text = "\n\n".join(
                item.get("text", item.get("content", str(item)))
                for item in data
                if isinstance(item, dict)
            )
        else:
            full_text = str(data)
        pages = None
    else:
        logger.warning(f"  Unsupported file type: {suffix}")
        return [], {
            "document_name": doc_name,
            "document_type": suffix,
            "parser": "unsupported",
            "error": f"Unsupported file type: {suffix}",
        }

    if not full_text.strip():
        logger.warning(f"  No text extracted from {doc_name}")
        return [], {
            "document_name": doc_name,
            "document_type": suffix,
            "parser": "empty",
            "error": "No text extracted",
        }

    # Step 2: Clean
    clean_text = text_cleaner.clean(full_text)
    logger.info(f"  Cleaned text: {len(clean_text)} chars")

    # Step 3: Chunk
    chunks = chunker.chunk_document(clean_text, doc_name, pages)
    chunk_dicts = [c.to_dict() for c in chunks]
    trace = _build_document_trace(file_path, pages, full_text, clean_text, chunk_dicts)

    logger.info(f"  ✅ Created {len(chunk_dicts)} chunks from {doc_name}")
    return chunk_dicts, trace


def _process_all_documents():
    """Background task: process all documents in the documents directory."""
    global _processing_status

    _processing_status = {
        "status": "processing",
        "progress": 0,
        "message": "Starting document processing...",
        "documents": [],
        "current_stage": "parsing",
        "metrics": _build_metrics_snapshot(),
    }

    # Find all documents
    doc_files = []
    for ext in ["*.pdf", "*.txt", "*.json"]:
        doc_files.extend(DOCUMENTS_DIR.glob(ext))

    if not doc_files:
        _processing_status = {
            "status": "idle",
            "progress": 0,
            "message": "No documents found in the documents directory.",
            "documents": [],
            "current_stage": "idle",
            "metrics": _build_metrics_snapshot(),
        }
        return

    all_chunks = []
    traces = []
    total = len(doc_files)

    for i, file_path in enumerate(doc_files):
        _processing_status["message"] = f"Processing {file_path.name} ({i+1}/{total})"
        _processing_status["progress"] = (i / total) * 100
        _processing_status["current_stage"] = "parsing"

        try:
            chunks, trace = _process_document(file_path)
            all_chunks.extend(chunks)
            traces.append(trace)
            _processing_status["documents"].append({
                "name": file_path.name,
                "chunks": len(chunks),
                "status": "success",
            })
        except Exception as e:
            traces.append(
                {
                    "document_name": file_path.name,
                    "document_type": file_path.suffix.lower(),
                    "parser": "error",
                    "error": str(e),
                }
            )
            logger.error(f"  ❌ Error processing {file_path.name}: {e}")
            _processing_status["documents"].append({
                "name": file_path.name,
                "chunks": 0,
                "status": f"error: {str(e)}",
            })

    # Save all chunks
    _save_all_chunks(all_chunks)
    _save_traces(traces)

    _processing_status["status"] = "completed"
    _processing_status["progress"] = 100
    _processing_status["message"] = f"Processed {total} documents → {len(all_chunks)} chunks"
    _processing_status["current_stage"] = "processed"
    _processing_status["metrics"] = _build_metrics_snapshot()

    logger.info(f"✅ Ingestion complete: {total} docs → {len(all_chunks)} chunks")


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a legal document (PDF, TXT, or JSON)."""
    allowed_extensions = {".pdf", ".txt", ".json"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}",
        )

    # Save file
    dest = DOCUMENTS_DIR / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"📁 Uploaded: {file.filename} ({len(content)} bytes)")

    return {
        "message": f"File '{file.filename}' uploaded successfully",
        "file_name": file.filename,
        "file_size": len(content),
        "path": str(dest),
    }


@router.post("/process")
async def process_documents(background_tasks: BackgroundTasks):
    """Process all uploaded documents (runs in background)."""
    if _processing_status.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Document processing is already in progress",
        )

    background_tasks.add_task(_process_all_documents)

    return {"message": "Document processing started in background"}


@router.post("/process-and-index")
async def process_and_index(background_tasks: BackgroundTasks):
    """Process documents AND build NER + vector indices (full pipeline)."""
    from ner.entity_index import EntityIndex
    from ner.rule_based import RuleBasedNER
    from ner.ml_based import MLBasedNER
    from retrieval.embedder import Embedder
    from retrieval.vector_store import VectorStore

    if _processing_status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Already processing")

    def full_pipeline():
        global _processing_status

        # Step 1: Process documents
        _process_all_documents()

        if _processing_status["status"] != "completed":
            return

        _processing_status["message"] = "Building NER entity index..."
        _processing_status["status"] = "processing"
        _processing_status["current_stage"] = "entity_indexing"

        # Load chunks
        chunks = _get_all_chunks()
        if not chunks:
            _processing_status["message"] = "No chunks to index"
            _processing_status["metrics"] = _build_metrics_snapshot()
            return

        # Step 2: NER on all chunks → entity index
        rule_ner = RuleBasedNER()
        ml_ner = MLBasedNER()
        entity_idx = EntityIndex()
        traces_by_doc = {
            trace.get("document_name"): trace for trace in _load_traces() if trace.get("document_name")
        }

        for chunk in chunks:
            text = chunk.get("text", "")
            entities = rule_ner.extract(text) + ml_ner.extract(text)
            # Deduplicate
            seen = set()
            unique = []
            for e in entities:
                key = (e["text"].lower(), e["label"])
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            chunk["entities"] = unique
            entity_idx.add_entities(chunk["chunk_id"], unique)

            doc_trace = traces_by_doc.get(chunk.get("metadata", {}).get("doc_name"))
            if doc_trace is not None:
                doc_trace["entity_total"] = doc_trace.get("entity_total", 0) + len(unique)
                label_counts = doc_trace.setdefault("entity_labels", {})
                sample_entities = doc_trace.setdefault("sample_entities", [])
                for entity in unique:
                    label = entity.get("label", "UNKNOWN")
                    label_counts[label] = label_counts.get(label, 0) + 1
                    if len(sample_entities) < 12:
                        sample_entities.append(
                            {
                                "text": entity.get("text", ""),
                                "label": label,
                                "source": entity.get("source", ""),
                            }
                        )

        entity_idx.save()
        _save_all_chunks(chunks)
        _save_traces(list(traces_by_doc.values()))

        _processing_status["message"] = "Building vector embeddings..."
        _processing_status["current_stage"] = "vector_indexing"

        # Step 3: Embeddings → FAISS
        embedder = Embedder()
        store = VectorStore()

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)
        store.build_index(embeddings, chunks)
        store.save()
        _refresh_runtime_pipeline()

        _processing_status["status"] = "completed"
        _processing_status["progress"] = 100
        _processing_status["current_stage"] = "completed"
        _processing_status["message"] = (
            f"Full pipeline complete: {len(chunks)} chunks indexed, "
            f"{len(entity_idx.index)} entities mapped"
        )
        _processing_status["metrics"] = _build_metrics_snapshot()
        logger.info(f"✅ Full pipeline complete")

    background_tasks.add_task(full_pipeline)
    return {"message": "Full processing pipeline started in background"}


@router.get("/status", response_model=ProcessingStatus)
async def get_processing_status():
    """Get current document processing status."""
    return ProcessingStatus(**_processing_status)


@router.get("/metrics")
async def get_ingestion_metrics():
    """Get detailed corpus and indexing metrics for the frontend."""
    snapshot = _build_metrics_snapshot()
    return {
        "status": _processing_status.get("status", "idle"),
        "current_stage": _processing_status.get("current_stage", "idle"),
        "message": _processing_status.get("message", ""),
        "metrics": snapshot,
    }


@router.get("/traces")
async def get_ingestion_traces():
    """Get detailed parser, cleaning, chunking, and NER traces per document."""
    return {
        "status": _processing_status.get("status", "idle"),
        "current_stage": _processing_status.get("current_stage", "idle"),
        "documents": _load_traces(),
    }


@router.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    docs = []
    for ext in ["*.pdf", "*.txt", "*.json"]:
        for f in DOCUMENTS_DIR.glob(ext):
            docs.append({
                "name": f.name,
                "size": f.stat().st_size,
                "type": f.suffix,
                "modified": f.stat().st_mtime,
            })
    return {"documents": docs, "total": len(docs)}


@router.get("/chunks")
async def list_chunks():
    """List all processed chunks with metadata."""
    chunks = _get_all_chunks()
    return {
        "chunks": chunks,
        "total": len(chunks),
    }


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete an uploaded document."""
    file_path = DOCUMENTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

    file_path.unlink()
    logger.info(f"🗑️ Deleted document: {filename}")

    return {"message": f"Document '{filename}' deleted successfully"}

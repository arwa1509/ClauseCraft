"""
Ingestion API Router
=====================
Endpoints for uploading, processing, and managing legal documents.
"""

import json
import shutil
import time
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

# Processing state
_processing_status = {"status": "idle", "progress": 0, "message": "", "documents": []}


class ProcessingStatus(BaseModel):
    status: str
    progress: float
    message: str
    documents: list


def _get_all_chunks() -> list[dict]:
    """Load all processed chunks from metadata file."""
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return []


def _save_all_chunks(chunks: list[dict]):
    """Save all chunks to metadata file."""
    METADATA_PATH.write_text(json.dumps(chunks, indent=2, default=str))


def _process_document(file_path: Path) -> list[dict]:
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
        return []

    if not full_text.strip():
        logger.warning(f"  No text extracted from {doc_name}")
        return []

    # Step 2: Clean
    clean_text = text_cleaner.clean(full_text)
    logger.info(f"  Cleaned text: {len(clean_text)} chars")

    # Step 3: Chunk
    chunks = chunker.chunk_document(clean_text, doc_name, pages)
    chunk_dicts = [c.to_dict() for c in chunks]

    logger.info(f"  ✅ Created {len(chunk_dicts)} chunks from {doc_name}")
    return chunk_dicts


def _process_all_documents():
    """Background task: process all documents in the documents directory."""
    global _processing_status

    _processing_status = {
        "status": "processing",
        "progress": 0,
        "message": "Starting document processing...",
        "documents": [],
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
        }
        return

    all_chunks = []
    total = len(doc_files)

    for i, file_path in enumerate(doc_files):
        _processing_status["message"] = f"Processing {file_path.name} ({i+1}/{total})"
        _processing_status["progress"] = (i / total) * 100

        try:
            chunks = _process_document(file_path)
            all_chunks.extend(chunks)
            _processing_status["documents"].append({
                "name": file_path.name,
                "chunks": len(chunks),
                "status": "success",
            })
        except Exception as e:
            logger.error(f"  ❌ Error processing {file_path.name}: {e}")
            _processing_status["documents"].append({
                "name": file_path.name,
                "chunks": 0,
                "status": f"error: {str(e)}",
            })

    # Save all chunks
    _save_all_chunks(all_chunks)

    _processing_status["status"] = "completed"
    _processing_status["progress"] = 100
    _processing_status["message"] = f"Processed {total} documents → {len(all_chunks)} chunks"

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

        # Load chunks
        chunks = _get_all_chunks()
        if not chunks:
            _processing_status["message"] = "No chunks to index"
            return

        # Step 2: NER on all chunks → entity index
        rule_ner = RuleBasedNER()
        ml_ner = MLBasedNER()
        entity_idx = EntityIndex()

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

        entity_idx.save()
        _save_all_chunks(chunks)

        _processing_status["message"] = "Building vector embeddings..."

        # Step 3: Embeddings → FAISS
        embedder = Embedder()
        store = VectorStore()

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)
        store.build_index(embeddings, chunks)
        store.save()

        _processing_status["status"] = "completed"
        _processing_status["progress"] = 100
        _processing_status["message"] = (
            f"Full pipeline complete: {len(chunks)} chunks indexed, "
            f"{len(entity_idx.index)} entities mapped"
        )
        logger.info(f"✅ Full pipeline complete")

    background_tasks.add_task(full_pipeline)
    return {"message": "Full processing pipeline started in background"}


@router.get("/status", response_model=ProcessingStatus)
async def get_processing_status():
    """Get current document processing status."""
    return ProcessingStatus(**_processing_status)


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

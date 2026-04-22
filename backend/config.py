"""
Central configuration for the Legal RAG system.
All paths, model names, and hyperparameters are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
NER_DATASETS_DIR = DATA_DIR / "ner_datasets"
QA_DATASETS_DIR = DATA_DIR / "qa_datasets"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [DOCUMENTS_DIR, NER_DATASETS_DIR, QA_DATASETS_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2

# ─── Cross-Encoder Model ──────────────────────────────────────────────────────
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ─── spaCy NER Model ──────────────────────────────────────────────────────────
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# ─── LLM Configuration ────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | ollama | huggingface
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "microsoft/phi-2")

# ─── NLI Model (Hallucination Detection) ──────────────────────────────────────
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")

# ─── Chunking Parameters ──────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "150"))

# ─── Retrieval Parameters ─────────────────────────────────────────────────────
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_index.bin"
METADATA_PATH = PROCESSED_DIR / "chunk_metadata.json"
ENTITY_INDEX_PATH = PROCESSED_DIR / "entity_index.json"
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "20"))
ENTITY_TOP_K = int(os.getenv("ENTITY_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF constant

# ─── Query History ─────────────────────────────────────────────────────────────
QUERY_HISTORY_PATH = PROCESSED_DIR / "query_history.json"

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

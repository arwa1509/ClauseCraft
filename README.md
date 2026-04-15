# Explainable Legal RAG with Named Entity Awareness

A production-quality legal question-answering system that combines Named Entity Recognition (NER), Retrieval-Augmented Generation (RAG), and sophisticated explainability modules.

## Features

- **Document Ingestion**: Supports PDF, TXT, and JSON files with structure-aware chunking.
- **Legal NER**: Extracts STATUTE, COURT, JUDGE, CASE_CITATION, and more using hybrid Rule-based + ML pipelines.
- **Hybrid Retrieval**: Combines FAISS-based dense vector search with entity-based inverted index lookup.
- **Advanced Ranking**: Uses Cross-Encoder reranking for precision relevance.
- **Explainable Results**: Claim-to-evidence mapping, confidence scoring, and color-coded entity highlighting.
- **Modern UI**: Clean, premium dashboard built with React and Tailwind CSS.

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **OpenAI API Key** (or local Ollama/HuggingFace setup)

---

## Setup Instructions

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Configuration:**
Copy `.env.example` to `.env` and add your OpenAI API Key:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=...
```

**Run Backend:**
```bash
uvicorn main:app --reload
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Usage Guide

1. **Upload Documents**: Go to the **Documents** page and upload your legal PDFs or text files.
2. **Process Pipeline**: Click **"Process & Index"**. This will:
   - Partition text into structured chunks.
   - Extract legal entities using NER.
   - Generate vector embeddings for the corpus.
3. **Ask Questions**: Navigate to the **Legal Query** page.
   - Enter a question (e.g., "What is the provision for bail in criminal cases?").
   - View the synthesized answer with claim-to-source citations.
4. **Verification**: Use the **Explainability Panel** and **Sources** tab to verify claims against the original document text.
5. **Analytics**: Check the **Analytics** page to explore the knowledge graph extracted from your corpus.

---

## Project Structure

- `/backend`: FastAPI server, NLP pipeline, retrieval logic.
- `/frontend`: React + Tailwind source code.
- `/backend/data`: Document storage and processed indices.
- `/backend/models`: Custom spaCy and ML models.

---

## Technology Stack

- **NLP**: spaCy, Sentence-Transformers, Transformers (NLI).
- **Retrieval**: FAISS, Custom Entity Index.
- **Backend**: FastAPI, Pydantic, Loguru.
- **Frontend**: Vite, React, Tailwind CSS, Lucide, Framer Motion.

"""FastAPI service: ask a question, get quoted spans with citations.

POST /query {"question": "...", "top_k": 5}
  -> {"status": "answered" | "insufficient_evidence",
      "answer_spans": [{"text": "...", "citation": {...}}],
      "citations": [...],
      "latency_ms": 123}

GET /health -> {"status": "ok", "chunks": N, "docs": M}

The retrieval pipeline loads once at startup from data/artifacts/. Build the
artifacts first with: python scripts/build_index.py
"""

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import answer as answer_module
from src import embed, index as vector_index, ingest, retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
UI_FILE = Path(__file__).resolve().parent / "ui.html"

app = FastAPI(title="EvidenceRAG", version="0.1.0")

# The demo page is served from this app, but also allow file:// and local
# static servers during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_retriever = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to answer.")
    top_k: int = Field(5, ge=1, le=20, description="Retrieved chunks to consider.")


class Citation(BaseModel):
    doc_id: str
    page: int
    char_start: int
    char_end: int
    score: float


class AnswerSpan(BaseModel):
    text: str
    citation: Citation


class QueryResponse(BaseModel):
    status: str
    answer_spans: list[AnswerSpan]
    citations: list[Citation]
    latency_ms: int


def load_pipeline():
    """Load chunks, FAISS index, embedder, and wrap them in a Retriever."""
    chunks = ingest.load_chunks(ARTIFACTS_DIR / "chunks.jsonl")
    faiss_index = vector_index.load_index(ARTIFACTS_DIR / "index.faiss")
    embedder = embed.Embedder()
    return retrieve.Retriever(chunks, embedder, faiss_index)


@app.on_event("startup")
def startup():
    global _retriever
    _retriever = load_pipeline()


@app.get("/")
def home():
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/health")
def health():
    chunks = _retriever.chunks
    docs = sorted({chunk["doc_id"] for chunk in chunks})
    return {"status": "ok", "chunks": len(chunks), "docs": docs}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    started = time.perf_counter()
    ranked = _retriever.hybrid_search(request.question, top_k=request.top_k)
    result = answer_module.extractive_answer(request.question, ranked, _retriever.embedder)
    latency_ms = int((time.perf_counter() - started) * 1000)
    citations = [span["citation"] for span in result["spans"]]
    return QueryResponse(
        status=result["status"],
        answer_spans=result["spans"],
        citations=citations,
        latency_ms=latency_ms,
    )

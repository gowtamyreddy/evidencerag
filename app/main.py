"""FastAPI service: ask a question, get quoted spans with citations.

POST /query {"question": "...", "top_k": 5}
  -> {"status": "answered" | "insufficient_evidence",
      "answer_spans": [{"text": "...", "citation": {...}}],
      "citations": [...],
      "latency_ms": 123}

GET /health -> {"status": "ok", "chunks": N, "docs": [...], "corpus": "default" | "upload"}

GET / -> the demo UI (app/ui.html)

POST /upload (multipart form field "file", a .zip of PDFs)
  -> 202 {"status": "building", "docs": [...]}
  The zip replaces the active corpus: PDFs are ingested, embedded, and
  indexed in the background; GET /upload/status reports progress and
  /query answers 503 until the new index is swapped in.

POST /corpus/reset -> restore the built-in corpus from data/artifacts/.

The default corpus loads once at startup from data/artifacts/. Build it
first with: python scripts/build_index.py
"""

import io
import threading
import time
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import answer as answer_module
from src import embed, index as vector_index, ingest, retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
UI_FILE = Path(__file__).resolve().parent / "ui.html"

MAX_ZIP_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_PDFS = 50

app = FastAPI(title="EvidenceRAG", version="0.2.0")

# The demo page is served from this app, but also allow file:// and local
# static servers during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_retriever = None
_default_retriever = None
_state_lock = threading.Lock()
_state = {"building": False, "corpus": "default", "error": None}


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


def build_retriever(pdf_paths, embedder):
    """Run the full offline pipeline in memory: ingest -> embed -> index."""
    all_chunks = []
    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        pages = ingest.read_pdf(pdf_path, doc_id)
        all_chunks.extend(ingest.chunk_pages(doc_id, pages))
    if not all_chunks:
        raise ValueError("No extractable text found in the uploaded PDFs.")
    vectors = embedder.embed_passages([chunk["text"] for chunk in all_chunks])
    faiss_index = vector_index.build_index(vectors)
    return retrieve.Retriever(all_chunks, embedder, faiss_index)


def _extract_pdfs(zip_bytes):
    """Extract PDF members of an uploaded zip to data/uploads/.

    Member names are flattened to their basename, so a zip can never write
    outside the uploads directory. Returns the list of extracted paths.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for old in UPLOADS_DIR.glob("*.pdf"):
        old.unlink()
    extracted = []
    seen_stems = set()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if not name.lower().endswith(".pdf"):
                continue
            stem = Path(name).stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            target = UPLOADS_DIR / name
            target.write_bytes(archive.read(member))
            extracted.append(target)
            if len(extracted) >= MAX_PDFS:
                break
    return extracted


def _build_and_swap(pdf_paths):
    """Background job: build a retriever from uploaded PDFs and swap it in."""
    global _retriever
    try:
        new_retriever = build_retriever(pdf_paths, _default_retriever.embedder)
        with _state_lock:
            _retriever = new_retriever
            _state.update(building=False, corpus="upload", error=None)
    except Exception as exc:  # surfaced to the UI via /upload/status
        with _state_lock:
            _state.update(building=False, error=str(exc))


@app.on_event("startup")
def startup():
    global _retriever, _default_retriever
    chunks = ingest.load_chunks(ARTIFACTS_DIR / "chunks.jsonl")
    faiss_index = vector_index.load_index(ARTIFACTS_DIR / "index.faiss")
    embedder = embed.Embedder()
    _default_retriever = retrieve.Retriever(chunks, embedder, faiss_index)
    _retriever = _default_retriever


@app.get("/")
def home():
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/health")
def health():
    with _state_lock:
        state = dict(_state)
    docs = sorted({chunk["doc_id"] for chunk in _retriever.chunks})
    return {
        "status": "ok",
        "chunks": len(_retriever.chunks),
        "docs": docs,
        "corpus": state["corpus"],
        "building": state["building"],
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    with _state_lock:
        building = _state["building"]
    if building:
        raise HTTPException(status_code=503, detail="Index is building; try again shortly.")
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


@app.post("/upload", status_code=202)
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip file of PDFs.")
    zip_bytes = await file.read()
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="Zip is larger than 100 MB.")
    try:
        pdf_paths = _extract_pdfs(zip_bytes)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="That file is not a valid zip.")
    if not pdf_paths:
        raise HTTPException(status_code=400, detail="No PDFs found in the zip.")
    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
        _state.update(building=True, error=None)
    background_tasks.add_task(_build_and_swap, pdf_paths)
    return {"status": "building", "docs": [path.stem for path in pdf_paths]}


@app.get("/upload/status")
def upload_status():
    with _state_lock:
        state = dict(_state)
    docs = sorted({chunk["doc_id"] for chunk in _retriever.chunks})
    return {
        "building": state["building"],
        "corpus": state["corpus"],
        "error": state["error"],
        "docs": docs,
        "chunks": len(_retriever.chunks),
    }


@app.post("/corpus/reset")
def corpus_reset():
    global _retriever
    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
        _retriever = _default_retriever
        _state.update(corpus="default", error=None)
    return {"status": "ok", "corpus": "default"}

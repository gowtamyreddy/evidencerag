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

POST /fetch {"url": "https://..."} -> 202 {"status": "building", "doc": "..."}
  Fetches a PDF or web page, appends it to the dynamic corpus, and rebuilds
  the index in the background. Repeats of the same URL get a numeric suffix.

POST /corpus/reset -> restore the built-in corpus from data/artifacts/.

The default corpus loads once at startup from data/artifacts/. Build it
first with: python scripts/build_index.py
"""

import io
import re
import threading
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import requests
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


class FetchRequest(BaseModel):
    url: str = Field(..., min_length=8, description="A PDF link or web page to add to the corpus.")


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


def build_retriever_from_dir(uploads_dir, embedder):
    """Run the full offline pipeline in memory over an uploads directory."""
    pdf_paths = sorted(uploads_dir.glob("*.pdf"))
    txt_paths = sorted(uploads_dir.glob("*.txt"))
    if not pdf_paths and not txt_paths:
        raise ValueError("No PDFs or text files found in the uploads directory.")
    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(ingest.chunk_pages(pdf_path.stem, ingest.read_pdf(pdf_path, pdf_path.stem)))
    for txt_path in txt_paths:
        all_chunks.extend(ingest.chunk_pages(txt_path.stem, ingest.read_text_file(txt_path)))
    if not all_chunks:
        raise ValueError("No extractable text found in the uploaded documents.")
    vectors = embedder.embed_passages([chunk["text"] for chunk in all_chunks])
    faiss_index = vector_index.build_index(vectors)
    return retrieve.Retriever(all_chunks, embedder, faiss_index)


def _unique_stem(uploads_dir, stem):
    """Avoid overwriting: 2309.15217 -> 2309.15217-2 if taken."""
    candidate, n = stem, 1
    while (uploads_dir / f"{candidate}.pdf").exists() or (uploads_dir / f"{candidate}.txt").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return candidate


def _extract_pdfs(zip_bytes):
    """Extract PDF members of an uploaded zip to data/uploads/.

    Member names are flattened to their basename, so a zip can never write
    outside the uploads directory. Returns the list of extracted paths.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    extracted = []
    seen_stems = set()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            members = archive.namelist()
    except zipfile.BadZipFile:
        raise
    # The zip is valid: safe to replace the previous dynamic corpus.
    for old in list(UPLOADS_DIR.glob("*.pdf")) + list(UPLOADS_DIR.glob("*.txt")):
        old.unlink()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in members:
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


def _build_and_swap():
    """Background job: rebuild the retriever from data/uploads/ and swap it in."""
    global _retriever
    try:
        new_retriever = build_retriever_from_dir(UPLOADS_DIR, _default_retriever.embedder)
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
    background_tasks.add_task(_build_and_swap)
    return {"status": "building", "docs": [path.stem for path in pdf_paths]}


class _TextExtractor(HTMLParser):
    """Minimal readable-text extractor: drop scripts, styles, noscript."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "header", "footer", "nav"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "header", "footer", "nav") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self):
        return ingest.clean_text(" ".join(self.parts))


def _doc_id_from_url(url):
    """Stable doc id for a fetched URL. arXiv ids keep their familiar form."""
    arxiv = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
    if arxiv:
        return arxiv.group(1)
    parts = urlsplit(url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (parts.netloc + parts.path).strip("-/"))[:60].strip("-")
    return slug or "web-doc"


@app.post("/fetch", status_code=202)
async def fetch_url(request: FetchRequest, background_tasks: BackgroundTasks):
    """Fetch a URL (PDF or web page) and add it to the dynamic corpus.

    The document is appended to data/uploads/ and the whole uploads folder is
    rebuilt in the background; poll GET /upload/status for progress. Repeats
    of the same URL get a numeric suffix instead of overwriting.
    """
    if not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://.")
    try:
        response = requests.get(
            request.url,
            headers={"User-Agent": "EvidenceRAG demo (corpus fetch)"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch that URL: {exc}")
    if len(response.content) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="Fetched file is larger than 100 MB.")
    content_type = response.headers.get("Content-Type", "").lower()

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if "application/pdf" in content_type or request.url.lower().endswith(".pdf"):
        stem = _unique_stem(UPLOADS_DIR, _doc_id_from_url(request.url))
        target = UPLOADS_DIR / f"{stem}.pdf"
        target.write_bytes(response.content)
    elif "text/html" in content_type or "text/plain" in content_type or not content_type:
        extractor = _TextExtractor()
        extractor.feed(response.text)
        text = extractor.text()
        if len(text) < 200:
            raise HTTPException(status_code=400, detail="That page has too little readable text to index.")
        stem = _unique_stem(UPLOADS_DIR, _doc_id_from_url(request.url))
        target = UPLOADS_DIR / f"{stem}.txt"
        target.write_text(text, encoding="utf-8")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type ({content_type}); send a PDF or an HTML page.",
        )

    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
        _state.update(building=True, error=None)
    background_tasks.add_task(_build_and_swap)
    return {"status": "building", "doc": stem, "source": "pdf" if target.suffix == ".pdf" else "page"}


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

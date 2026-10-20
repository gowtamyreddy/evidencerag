# EvidenceRAG

A document question-answering system that proves where its answers came from. Every answer is a quoted span from the corpus with a citation pointing at the exact document, page, and character offsets. Retrieval is evaluated with a golden set, and the test suite shows the failure cases on purpose: ambiguous and unanswerable questions should return "insufficient evidence", not a confident wrong answer.

The v1 corpus is twenty papers: arXiv papers on retrieval-augmented generation and LLM evaluation, plus voice-AI and medical-imaging papers, so the system can answer questions about its own design space and neighboring domains. `data/papers/CONTENT_NOTES.md` records a manifest audit noting that five files do not match the paper their filename claims; the golden-question set is grounded only in the verified files.

## Architecture

```
                      +------------------+
                      |  arXiv PDFs      |
                      |  data/papers/    |
                      +--------+---------+
                               |
                     scripts/fetch_corpus.py
                               v
                      +------------------+
                      |  src/ingest.py   |  clean text, overlapping chunks,
                      |  chunks.jsonl    |  keep doc_id / page / char offsets
                      +--------+---------+
                               |
                     scripts/build_index.py
                               |
              +----------------+----------------+
              |                                 |
   src/embed.py                       src/index.py
   all-MiniLM-L6-v2,                  FAISS IndexFlatIP over
   L2-normalized vectors              normalized vectors (exact cosine)
              |                                 |
              +----------------+----------------+
                               v
                      +------------------+
                      | src/retrieve.py  |  dense (FAISS) + BM25,
                      | hybrid_search    |  fused with reciprocal rank fusion
                      +--------+---------+
                               |
                      src/answer.py        extractive: score each sentence
                               |           against the query, quote the best
                               v           1-3 above threshold
                      +------------------+
                      | app/main.py      |  POST /query -> answer spans +
                      | FastAPI          |  citations + latency
                      +------------------+

                      src/eval.py          golden JSONL -> hit@k, MRR,
                      tests/               citation precision; failure cases
```

## Demo UI and dynamic upload (v1.1)

Open `http://localhost:8000/` for the browser demo: a search box with clickable examples, a 🎙 voice button (the browser's built-in speech recognition fills the box and auto-asks; Chrome, Edge, Safari), quoted answer spans with citations, and a corpus upload card. Uploading a `.zip` of PDFs replaces the active corpus: the same ingest → embed → FAISS pipeline runs as a background job (`POST /upload`), the UI polls `GET /upload/status` until the index swaps in, and `/query` answers 503 while the build runs. `POST /fetch` accepts a URL instead: a PDF link or a web page is downloaded, its text extracted, appended to the dynamic corpus (repeats get a numeric suffix), and the index rebuilt the same way. `POST /corpus/reset` restores the built-in corpus. The static golden eval (hit@1, MRR) still applies per-corpus: the same harness can score whatever corpus is active.

## How to run

```bash
# 1. Create the environment (CPU torch first, then the rest)
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

# 2. Fetch the corpus (~12 arXiv PDFs)
.venv/bin/python scripts/fetch_corpus.py

# 3. Build chunks, embeddings, and the FAISS index
.venv/bin/python scripts/build_index.py

# 4. Run the tests
.venv/bin/python -m pytest -q

# 5. Evaluate retrieval on the golden set
.venv/bin/python src/eval.py

# 6. Start the API
.venv/bin/python -m uvicorn app.main:app --port 8000
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What problem does retrieval-augmented generation solve?"}'
```

## Key design decisions

**Extractive answers, no LLM.** Answers are quoted sentences, not generated text. This removes hallucinated wording entirely and needs no API key, which keeps the demo runnable anywhere. The honest tradeoff: answers are limited to what is written verbatim, so the system quotes evidence rather than synthesizing it. That is a deliberate v1 scope choice; a later phase can add an LLM that must stay grounded in the quoted spans.

**Hybrid retrieval (dense + BM25).** Dense retrieval handles paraphrase ("how is answer quality measured" matching "we evaluate generation quality"), while BM25 is exact on rare terms like paper names and metric names ("MRR", "RAGAS"). Each fails where the other succeeds. Reciprocal rank fusion combines them using only ranks, so no score normalization between two incomparable scoring systems is needed.

**FAISS over pgvector for v1.** FAISS runs in-process with zero infrastructure: one command builds the index, one command serves it. The vector store is swappable by design; only `src/index.py` touches FAISS, so moving to pgvector later means replacing four functions (`build_index`, `search`, `save_index`, `load_index`).

**Normalized embeddings everywhere.** The embedder L2-normalizes on the way out, so inner product equals cosine similarity. FAISS uses inner product, and sentence scoring uses a dot product; one normalization point keeps both correct.

**"Insufficient evidence" as a first-class result.** If no sentence scores above the threshold, the API says so and names the nearest chunks. A wrong confident answer is the worst outcome for a citation system; refusing is the feature.

**Deterministic.** FAISS `IndexFlatIP` is exact search (no approximation, no training randomness), and the same model on the same text always produces the same vectors.

## Failure cases

These are tested, not hidden:

1. **Unanswerable questions.** Asked about something the corpus does not cover (e.g. "What is the capital of France?"), the best sentence score falls below the threshold and the system returns `insufficient_evidence` with the nearest chunks flagged for inspection.
2. **Ambiguous questions.** A vague query like "retrieval" matches many chunks weakly and none strongly, so confidence stays low and the system refuses rather than picking a random meaning.
3. **Single-signal false positives.** A chunk that ranks highly in only one of dense/BM25 can look convincing. RRF fusion requires agreement between both signals, which the tests verify with synthetic rankings.
4. **Sentence-boundary quotes.** Answers are sentence quotes, so a question needing synthesis across distant sentences gets the relevant sentences quoted side by side instead of a fused claim. This is a known v1 limitation; citations stay exact because of it.

## Project layout

```
evidencerag/
  src/ingest.py        PDF/text -> cleaned, overlapping chunks with offsets
  src/embed.py         sentence-transformers wrapper, normalized vectors
  src/index.py         FAISS index build/search/save/load
  src/retrieve.py      dense + BM25 + reciprocal rank fusion
  src/answer.py        extractive sentence scoring with citations
  src/eval.py          golden-set metrics: hit@k, MRR, citation precision
  app/main.py          FastAPI: POST /query, GET /health
  scripts/fetch_corpus.py   arXiv download for the demo corpus
  scripts/build_index.py    ingest -> embed -> index, writes data/artifacts/
  data/papers/         downloaded PDFs (not committed)
  data/artifacts/      chunks.jsonl, embeddings.npy, index.faiss (not committed)
  data/golden_sample.jsonl   6 golden questions with expected docs
  tests/test_retrieval.py    golden test, failure cases, RRF unit test
```

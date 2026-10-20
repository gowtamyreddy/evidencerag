# EvidenceRAG

A document question-answering system that proves where its answers came from. Every answer is a verbatim quote from the corpus with a citation (document, page, character offsets) that resolves back to the source text, and when no sentence clears the evidence threshold the system answers "insufficient evidence" instead of guessing. There is no LLM and no API key: retrieval over real papers, measured on a hand-written golden set.

## Example

Actual response from the API running the verified corpus described below (only the machine-dependent `latency_ms` field is omitted):

```json
POST /query  {"question": "How many Wikipedia passages does Dense Passage Retrieval index?"}

{
  "status": "answered",
  "reranked": true,
  "answer_spans": [
    {
      "text": "We then split each article into multiple, disjoint text blocks of 100 words as passages, serving as our basic retrieval units, following (Wang et al., 2019), which results in 21,015,324 passages in the end.5 Each passage is also prepended with the title of the Wikipedia article where the passage is from, along with an [SEP] token.",
      "citation": {"doc_id": "2004.04906", "page": 4, "char_start": 866, "char_end": 1198, "score": 0.7063}
    }
  ]
}
```

The full response also carries the retrieved chunk text plus the quote's offsets within it (`rel_start`/`rel_end`), which is what the demo UI uses to show the sentence highlighted in context. **What the offsets mean:** character positions in the *cleaned, extracted page text* (the text pypdf produces, with whitespace collapsed), not byte offsets into the PDF file. Within that coordinate system, quotes are exact: the test suite asserts `chunk_text[rel_start:rel_end]` equals the quoted span for live API responses.

## Features

- Hybrid retrieval: dense embeddings (FAISS) + BM25 keyword search, fused with reciprocal rank fusion
- Optional cross-encoder reranking of the fused top 20 (on by default; one checkbox or flag to disable)
- Extractive answers: the best 1–3 sentences, quoted verbatim, each with a resolvable citation
- First-class refusal: below the evidence threshold, the answer is `insufficient_evidence` plus the nearest chunks for inspection
- Browser demo UI with a source-context drawer (the quoted sentence highlighted inside its chunk)
- Bring your own corpus: upload a zip of PDFs, or fetch a PDF/web page by URL; the index rebuilds in the background and swaps in atomically
- Voice questions transcribed locally by faster-whisper; audio never leaves the machine
- Golden-set evaluation with hit@k, MRR, and citation precision, in text or JSON
- Corpus integrity validation: every curated paper's file, SHA-256, and title checked offline against the manifest

## Architecture

```
   PDFs / uploads / URLs          scripts/fetch_corpus.py (arXiv)
            |                     scripts/fetch_bulk_corpus.py (optional expansion)
            v
   src/ingest.py        clean text (incl. lone-surrogate stripping), overlapping
                        chunks, each with doc_id / page / char offsets
            |
   scripts/build_index.py
            |
   src/embed.py  all-MiniLM-L6-v2, L2-normalized  +  src/index.py  FAISS IndexFlatIP
            v
   src/retrieve.py      dense top-20 + BM25 top-20, reciprocal rank fusion (k=60)
            |
   src/rerank.py        cross-encoder reorders the fused top-20 (optional)
            |
   src/answer.py        score each sentence against the query; quote the best
                        1-3 above threshold, else insufficient_evidence
            v
   app/main.py          FastAPI: /query /health /upload /fetch /stt /corpus/reset
   app/ui.html          demo page; header counts are read live from /health

   src/eval.py          golden JSONL -> hit@k, MRR, citation precision (+ --json)
   scripts/validate_corpus.py   manifest vs files vs titles (offline)
```

## The corpus, and what the counts mean

Four different numbers get conflated in this project, so here they are, defined:

| Count | What it is | Value |
|---|---|---|
| Curated corpus | Papers selected, identity-verified, and used for the golden evaluation | **20 papers** |
| Fetchable from arXiv | Subset `scripts/fetch_corpus.py` downloads from a clean checkout | 16 of the 20 |
| Contributed papers | Journal/conference PDFs added by hand; not redistributed in this repo | 4 of the 20 |
| Bulk expansion (optional) | Recent arXiv papers fetched by `scripts/fetch_bulk_corpus.py` as realistic distractors; titles recorded from Semantic Scholar, contents not individually audited | ~110 more |
| Indexed chunks | What the numbers above become after chunking (curated 20-paper build) | 3,475 chunks |
| Golden questions | Hand-written evaluation questions (16 answerable + 2 unanswerable) | 18 |

Document identity is not taken on faith. An early audit found five files whose contents did not match their filenames, because the fetch list itself carried wrong arXiv IDs (a graphene-physics paper was serving as "REALM"). The fetch list is corrected, the wrong files are removed, and `data/papers/manifest.json` now records each curated paper's canonical title, authors, source, page count, and SHA-256. `python scripts/validate_corpus.py` re-verifies files, hashes, and titles offline and currently passes 20/20. The full audit trail is in `data/papers/CONTENT_NOTES.md`.

Reproducibility, stated plainly: a clean checkout can fetch 16 of the 20 curated papers and rebuild everything. The 4 contributed PDFs are not in the repo, so a 16-paper build scores the 5 golden questions about them as misses; the eval prints a warning naming exactly those questions. The evaluation table below is from the full verified 20-paper corpus.

## How to run

```bash
# 1. Create the environment (CPU torch first, then the rest)
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

# 2. Fetch the arXiv portion of the curated corpus (16 PDFs)
.venv/bin/python scripts/fetch_corpus.py

# 3. (Optional) verify corpus identity against the manifest
.venv/bin/python scripts/validate_corpus.py

# 4. Build chunks, embeddings, and the FAISS index
.venv/bin/python scripts/build_index.py

# 5. Run the tests (also runs in GitHub Actions on every push)
.venv/bin/python -m pytest -q

# 6. Evaluate retrieval on the golden set (add --rerank, --json out.json)
.venv/bin/python -m src.eval data/golden_questions_v1.jsonl

# 7. Start the API + demo UI
.venv/bin/python -m uvicorn app.main:app --port 8000
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many Wikipedia passages does Dense Passage Retrieval index?"}'
```

## Evaluation

The golden set (`data/golden_questions_v1.jsonl`) is 18 hand-written questions grounded in the verified corpus: 15 factual, 1 comparison, 2 unanswerable. It is a regression/smoke suite for this corpus, not a general benchmark: one author, one corpus, small n. Metrics are computed over the 16 answerable questions; the 2 unanswerable ones are excluded because the correct behavior for them is refusal, which the API tests cover.

Metric definitions (unit of evaluation: one question; ranks are over retrieved chunks, window = top 5):

- **hit@k** — fraction of questions where at least one chunk from an expected document appears in the top k.
- **MRR** — mean over questions of 1/(rank of the first chunk from an expected document); 0 if none is in the window.
- **Citation precision** — per question, the fraction of the top-5 chunks that come from an expected document, averaged. This measures whether the citations the UI would show point at the right paper.

Measured on the verified 20-paper corpus (3,475 chunks), same questions, same index, both modes (2026-10-20):

| metric | stage 1 only | stage 1 + rerank |
|---|---|---|
| hit@1 | 0.812 | 1.000 |
| hit@3 | 0.937 | 1.000 |
| hit@5 | 1.000 | 1.000 |
| MRR | 0.887 | 1.000 |
| citation precision | 0.863 | 0.988 |

Retrieval latency on the machine that ran this (a shared cloud VM, models already loaded): mean 69 ms stage 1 only, mean 1,767 ms with the cross-encoder. Treat those as orders of magnitude, not specifications; the eval prints its own timings on your hardware and `--json` writes the full per-question results.

An earlier version of this table (hit@1 0.812 → 1.000, MRR 0.896 → 1.000, precision 0.850 → 0.975) was measured on the pre-audit corpus that contained the mislabeled files. It is superseded by the table above; the interesting part is *why* the stage-1 numbers moved, which is the failure analysis below.

### Failure analysis

With the corpus corrected, stage 1 (no reranker) misses rank 1 on 3 of 16 questions, and the misses are now honest hard negatives rather than data errors:

1. **Self-RAG factual** — the expected paper ranks 2nd behind CRAG (2401.15884). Before the corpus fix, rank 1 was a mislabeled *survey* that discussed Self-RAG; now it is a genuinely related paper about retrieval decisions, whose chunks legitimately compete on retrieval vocabulary. The reranker, which reads question and chunk together, resolves it.
2. **Self-RAG vs RAPTOR comparison** — the expected primary paper ranks 5th in stage 1: comparison questions dilute their terms across two topics, and generic retrieval vocabulary dominates the fusion. Reranker resolves it.
3. **PanNuke hand-crafted features** — ranks 2nd behind the stock-market survey; broad terms (data scale, feature selection) overlap across unrelated papers. Reranker resolves it.

The pattern worth remembering: stage 1 is a recall machine with vocabulary-level judgment; every observed rank-1 failure is an ordering failure among *plausible* chunks, which is exactly the gap a cross-encoder fills, and exactly why hit@3/hit@5 were already high before it.

## Demo UI and dynamic corpora

Open `http://localhost:8000/`: a search box with example questions, quoted answers with citations, a "View source in context" drawer per span, a voice button (local faster-whisper via `POST /stt`), and corpus controls. The header counts (documents, chunks) are read live from `GET /health`, so they always describe the corpus actually loaded, curated or uploaded.

Uploading a `.zip` of PDFs replaces the active corpus: ingest → embed → FAISS runs as a background job (`POST /upload`), the UI polls `GET /upload/status`, `/query` answers 503 while building, and the new retriever swaps in only when the build succeeds. `POST /fetch` adds one URL (PDF or web page) to the dynamic corpus the same way; repeats get a numeric suffix. `POST /corpus/reset` restores the built-in corpus. One corpus is active at a time; that is a deliberate demo simplification, and `docs/PRODUCTION.md` describes the multi-user design.

## Key design decisions

**Extractive answers, no LLM.** Answers are quoted sentences, not generated text. This removes hallucinated wording entirely and needs no API key, which keeps the demo runnable anywhere. The honest tradeoff: answers are limited to what is written verbatim, so the system quotes evidence rather than synthesizing it. That is a deliberate scope choice; a later phase could add an LLM constrained to the quoted spans.

**Hybrid retrieval (dense + BM25).** Dense retrieval handles paraphrase; BM25 is exact on rare terms like paper names and metric names. Each fails where the other succeeds. Reciprocal rank fusion combines them using only ranks, so no score normalization between two incomparable scoring systems is needed.

**Two-stage ranking.** Stage 1 optimizes recall over the whole corpus; the cross-encoder reranker reorders the fused top 20 by reading question and chunk together. The failure analysis above is what justifies the second stage's latency cost, and the eval measures both configurations so the justification stays honest.

**FAISS over pgvector for v1.** FAISS runs in-process with zero infrastructure. The vector store is swappable by design; only `src/index.py` touches FAISS. `docs/PRODUCTION.md` covers when and how that changes (approximate indexes around a million chunks, managed stores beyond).

**Normalized embeddings everywhere.** The embedder L2-normalizes on the way out, so inner product equals cosine similarity in both FAISS and sentence scoring; one normalization point keeps both correct.

**"Insufficient evidence" as a first-class result.** If no sentence scores above the threshold (0.5), the API says so and names the nearest chunks. A wrong confident answer is the worst outcome for a citation system; refusing is the feature.

**Deterministic core.** FAISS `IndexFlatIP` is exact search (no approximation), and the same model on the same text produces the same vectors, so evaluation results are reproducible run to run.

## Security and privacy

- **Everything is local.** No document, question, or audio leaves the machine; there is no telemetry and no third-party API. The only outbound requests are the ones you explicitly trigger (corpus fetches, URL ingestion).
- **Uploaded files are untrusted input.** Zip members are flattened to basenames (no path traversal), capped per-file and in total extracted size (zip-bomb guard), and must carry real PDF magic bytes; a rejected upload never wipes the working corpus. Malformed or encrypted PDFs fail the build, surface as an error via `/upload/status`, and never crash the API.
- **URL ingestion is SSRF-hardened.** `/fetch` validates every hop: scheme allowlist, DNS resolution checked so hosts resolving to private, loopback, link-local, or reserved addresses (including the cloud metadata range) are refused; redirects are followed manually and re-validated; downloads are streamed with a hard size cap and PDF content is verified by magic bytes. One residual risk is documented in the code: a DNS-rebinding race between the check and the connect is not fully closed (that needs pinned-IP connections with manual SNI handling), so do not expose `/fetch` on a trusted network segment without a proxy that adds that protection.
- **Deployment assumption.** This is a local/demo server: CORS is permissive for local development and there is no authentication. Before exposing it beyond localhost, restrict CORS origins and put it behind an authenticated proxy. `docs/PRODUCTION.md` designs the authenticated version.

## Known limitations

- Answers quote sentences; a sentence truncated by a chunk boundary can be quoted mid-thought, and the naive sentence splitter breaks on abbreviations ("et al."). Quotes remain exact substrings, but they are not always complete sentences.
- One active corpus at a time; uploads replace rather than merge (fetches append to the dynamic corpus).
- The golden set is small and single-author; treat its metrics as regression signals for this corpus, not as claims about retrieval quality in general.
- PDF text extraction is pypdf: scanned-image PDFs produce no text and therefore no chunks (the affected pages are simply absent, and `validate_corpus.py` title checks would flag a fully-scanned curated paper).
- The cross-encoder adds seconds per query on CPU; stage 1 alone is the low-latency configuration.

## Project layout

```
evidencerag/
  src/ingest.py        PDF/text -> cleaned, overlapping chunks with offsets
  src/embed.py         sentence-transformers wrapper, normalized vectors
  src/index.py         FAISS index build/search/save/load
  src/retrieve.py      dense + BM25 + reciprocal rank fusion
  src/rerank.py        cross-encoder rerank stage over the fused top-20
  src/answer.py        extractive sentence scoring with citations
  src/eval.py          golden-set metrics: hit@k, MRR, citation precision
  app/main.py          FastAPI service + ingestion endpoints
  app/ui.html          demo page
  scripts/fetch_corpus.py      arXiv download for the curated corpus (16 papers)
  scripts/fetch_bulk_corpus.py optional ~110-paper distractor expansion
  scripts/build_index.py       ingest -> embed -> index, writes data/artifacts/
  scripts/validate_corpus.py   offline manifest/file/title verification
  data/papers/         PDFs + manifest.json + CONTENT_NOTES.md (PDFs not committed)
  data/artifacts/      chunks.jsonl, embeddings.npy, index.faiss (not committed)
  data/golden_questions_v1.jsonl   the 18-question golden set
  tests/               retrieval + API, eval metrics, security (SSRF/upload)
  docs/PRODUCTION.md   production architecture: auth, database, caching, scale
```

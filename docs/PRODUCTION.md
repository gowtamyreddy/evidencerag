# EvidenceRAG in Production

How this project grows from a working demo into a product: what breaks, what
to build in what order, and the trade-offs behind each choice. Companion to
README.md (what the system is) — this document is about operating it for
real users at real scale.

## The production mindset

A demo answers "can it work once, for me?" Production answers five harder
questions, continuously:

1. **Identity** — who is using it, and what are they allowed to do?
2. **Durability** — if the process dies tonight, what is lost?
3. **Load** — what happens at 10x users and 100x data, and which part
   breaks first?
4. **Cost & latency** — what does one question cost in time and compute,
   and which parts of that cost can be paid once instead of every time
   (caching)?
5. **Evidence** — how do we *know* it works: logs, metrics, evaluations,
   user counts.

Every design choice below is a trade-off among those five. There are no
free wins; there are only wins whose price you have chosen on purpose.

## Where the demo stands

Single FastAPI process. Corpus artifacts (chunks.jsonl, embeddings.npy,
index.faiss) are files on disk, loaded into memory. Uploads rebuild the
index in a background task and swap the retriever atomically. No accounts,
no database, no query log, no cache. Correct and demonstrable — and every
one of those "no"s is a production gap.

What the demo already gets right, and production keeps: the background
ingest pattern (`POST /upload` -> job -> `GET /upload/status` -> atomic
swap) *is* the production ingestion pattern, just with a real queue later.
The vector store is isolated in `src/index.py`, so it can be replaced
without touching retrieval logic. The golden-set evaluation is the
production quality gate.

## Target architecture

```
            ┌────────────┐   static UI (cache headers)
  browser ──┤  FastAPI   ├──────────────────────────────┐
            │  (stateless│                                │
            │   replicas)│                                ▼
            └─────┬──────┘                        ┌──────────────┐
                  │                               │  SQLite DB   │
        ┌─────────┼──────────────┐                │ users        │
        ▼         ▼              ▼                │ documents    │
   exact-match  semantic    retrieval             │ queries log  │
   answer cache answer cache pipeline             │ feedback     │
        │         │         (FAISS+BM25,          └──────────────┘
        └────┬────┘          rerank, extract)
             ▼                    │
        cached answer ◄───────────┘
   ingestion: upload -> job queue -> embed -> index swap (per user corpus)
   identity:  Google OAuth (OIDC) -> session -> per-user corpora & limits
```

### 1. Identity: Google sign-in

- Sign in with Google (OpenID Connect, via `authlib`). Scopes: email and
  profile only — the minimum that identifies a person.
- Why accounts at all: per-user corpora (uploads are already per-corpus in
  the demo; accounts make them *owned*), abuse control, rate limits, and
  the user metrics below. A demo nobody signs into can still be impressive;
  a product nobody signs into has no users you can count.
- Friction trade-off, decided deliberately: **querying the built-in corpus
  stays open with no login** (a demo that demands sign-up before the first
  question loses most visitors). Sign-in is required to **upload your own
  corpus** and unlocks higher rate limits. Freemium gate at the expensive
  operation, not at the front door.
- Sessions: signed cookie, http-only. No passwords are ever stored — Google
  holds them; we hold a user row: `id, google_sub, email, name,
  created_at, last_seen_at`.

### 2. Database: SQLite first, on purpose

SQLite is the right first database here: zero operations, one file,
surprisingly capable, and this workload is read-heavy with light writes
(sign-ups, document registry, query logs). All access goes through
SQLAlchemy models, so the move to Postgres is a connection-string change
when one of SQLite's real limits arrives: concurrent writers at scale, or
more than one app replica needing the same database.

Schema:

- `users` — as above.
- `documents` — `id, owner_id (nullable = built-in corpus), doc_id, title,
  source (upload/url/builtin), pages, chunks, status, created_at`. This is
  the registry the demo currently keeps implicitly as "files in a folder."
- `queries` — `id, user_id (nullable), question, status (answered /
  insufficient_evidence), latency_ms, cache_hit (none/exact/semantic),
  corpus_version, created_at`. This table *is* the analytics: user counts,
  questions per day, refusal rate, p50/p95 latency.
- `feedback` — `query_id, helpful (bool)`. One click in the UI ("was this
  quote actually the answer?") turns user traffic into evaluation data.
- `corpora` / `corpus_version` — every corpus mutation bumps a version
  integer. Cache keys include it (see below); it is what makes cache
  invalidation a fact instead of a guess.

Deployment note: on Hugging Face Spaces free tier the filesystem is
ephemeral, so a SQLite file resets when the Space restarts. Acceptable for
the public demo (sign-ups are illustrative), and the honest upgrade path is
a free external Postgres (Neon/Supabase) or paid persistent storage. The
code does not change — only `DATABASE_URL`.

### 3. Caching: pay each cost once

A cold question costs ~1-4 s (retrieval + cross-encoder rerank). Most demo
and early-product traffic is repeats: everyone asks the example questions.
Three layers, cheapest first:

1. **Exact-match answer cache.** Key: `sha256(normalized_question +
   corpus_version)`. Normalization: lowercase, collapse whitespace, strip
   trailing punctuation. A hit returns the stored answer in <50 ms. Stored
   in SQLite (later Redis — same key design).
2. **Semantic cache.** The interesting one, for "same or *similar*"
   questions. Embed the incoming question; if its cosine similarity to a
   previously answered question **on the same corpus version** is
   >= 0.95, serve that answer and log `cache_hit=semantic`. The threshold
   is the whole trade-off: too low and a *different* question gets a
   confident, wrong, cached answer — the exact failure this product exists
   to prevent. 0.95 starts conservative; every semantic hit is logged so
   the threshold is tuned from data (the queries log), not vibes. Question
   embeddings live in a small FAISS index of their own.
3. **Embedding cache.** Question and chunk embeddings are pure functions
   of text; cache by text hash so reruns, evals, and the semantic cache
   never pay for the same embedding twice.

Invalidation rule for all layers: **the corpus version is part of every
key**, so a corpus change orphans old entries instead of serving stale
answers. No TTL guessing.

### 4. Huge data: what breaks, in order

Measured anchors from the laptop build: ~170 chunks per paper, 1.5 KB of
vector per chunk. So 10,000 papers ≈ 1.7M chunks ≈ 2.6 GB of vectors — the
in-memory exact FAISS index is already the wrong tool there.

- **~1M chunks:** switch `IndexFlatIP` (exact) to an approximate index
  (`IndexIVFPQ` or HNSW). Small recall cost, 10-30x memory cut, and the
  swap is contained in `src/index.py` by design. Validate recall against
  the golden set before and after — the eval decides, not the brochure.
- **Ingestion:** uploads stop being a background task in the web process
  and become a job queue (RQ/Celery): the API accepts the upload, workers
  embed, the index swaps per corpus when the job lands. Users poll status
  (the demo's `/upload/status` shape survives unchanged).
- **Serving:** once the index and DB are external to the process, the API
  is stateless and replicates behind a load balancer. The reranker is the
  expensive per-query stage: it scales horizontally, and the caches above
  exist precisely to keep most questions from ever reaching it.
- **10M documents:** sharded ANN (or a managed store: Qdrant / pgvector /
  Milvus), BM25 moves to OpenSearch, per-tenant indexes. The design seams
  for this already exist; see the README scale section.

### 5. Knowing it works: metrics that matter

From the `queries` table and the eval harness, reviewed weekly:

- **Product:** sign-ups, daily question-askers, questions per user,
  upload conversion (signed in -> uploaded a corpus).
- **Quality:** golden-set hit@1 / MRR / citation precision per release
  (release gate: no regression), live insufficient-evidence rate (a spike
  means the corpus or retrieval broke, not that users got worse at
  asking).
- **Performance:** p50/p95 latency split by cache layer; cache hit rate is
  the cheapest capacity there is.
- **Trust:** feedback ratio (helpful / total) — the product's core claim,
  measured on real traffic.

### 6. Hygiene

Structured logging (one JSON line per request with query id), config from
environment only, rate limiting per user/IP (stricter for anonymous),
upload validation already in place (zip checked, 100 MB / 50 PDF caps),
CI running the test suite and the golden eval on every push, and an ADR
(architecture decision record) — one short file per real decision in
`docs/adr/` — so every trade-off above is explainable in an interview from
the record, not from memory.

## Build sequence

1. **Data layer** — SQLAlchemy + SQLite: users, documents, queries,
   feedback; query logging; corpus_version. (No visible change; everything
   after this stands on it.)
2. **Exact + semantic cache** — biggest user-visible win: repeat questions
   go from seconds to milliseconds.
3. **Google sign-in** — accounts, per-user corpora, sign-in gate on
   uploads, admin stats from the queries table.
4. **ANN index + job queue** — when the corpus, not the code, demands it.

## The positioning this buys

The demo proves retrieval skill. This document — and each phase shipped —
proves product engineering: identity, data modeling, cache design with an
explicit correctness trade-off, a scaling path with measured anchors, and
quality gates. "I built a RAG demo" is a claim. "I run a citation-verified
QA product with N signed-up users, a semantic cache tuned from logged
traffic, and published accuracy on a golden set" is a record.

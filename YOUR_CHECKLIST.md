# EvidenceRAG — Your Side (Meganaa's checklist)

## The deal

Boone builds the machinery. You own the judgment: the corpus, the golden questions, the failure analysis, and the story. Interviewers don't hire you for code someone else wrote; they hire you for decisions you can defend.

## Phase 0 — Corpus + golden set (tonight, ~1 hr)

- [ ] Corpus: 20 curated papers (16 fetch from arXiv via scripts/fetch_corpus.py, 4 contributed journal PDFs). All identity-verified — see data/papers/CONTENT_NOTES.md and scripts/validate_corpus.py.
- [ ] Skim 3–4 papers (abstracts + conclusions are enough). Write each paper's one-line claim in your own words.
- [ ] Write 20–30 golden questions, one JSON object per line:
  `{"question": "...", "expected_doc_id": "<arXiv id>", "expected_keywords": ["distinctive", "phrases"], "type": "factual|comparison|ambiguous|unanswerable"}`
- [ ] Type guide:
  - `factual` (10–12): answerable from one paper. Write the question AFTER reading the paper's claim, so the expected doc is honest.
  - `comparison` (5–6): needs two papers, e.g. "How does FiD's use of retrieved passages differ from RAG's?"
  - `ambiguous` (4–5): keywords match many papers, e.g. "What is dense passage retrieval?" Expect the system to spread confidence thin.
  - `unanswerable` (4–5): plausible but absent, e.g. "How does RAG handle video inputs?" Must return "insufficient evidence", never a guess.
- [ ] Save as `data/golden.jsonl` in the repo, or paste the lines to me and I'll place them.

## Phase 1 — Learn the machine (days 2–3, ~1.5 hrs, with Boone)

- [ ] Walkthrough: I explain each module; you interrupt until you can explain it back.
- [ ] Be able to whiteboard the pipeline: ingest → chunk → embed → FAISS + BM25 → RRF fusion → sentence scoring → cited spans.
- [ ] For each decision below, know the why and the tradeoff. We'll fill this in together.
- [ ] Review the retrieval comparison run on YOUR golden set: which strategy won, and can you say why in one sentence?

| Decision | Why | Tradeoff |
|---|---|---|
| Chunks ~500 chars, ~100 overlap | small enough for precise citations, big enough for context | very long reasoning chains get split |
| all-MiniLM-L6-v2 embeddings | fast, free, 384-dim, good enough | weaker than large models on nuance |
| FAISS, not pgvector | zero-infrastructure demo, one command | single machine, no concurrent writes |
| Hybrid dense + BM25, fused with RRF | dense misses exact terms, BM25 misses paraphrase | two indexes to maintain |
| Extractive answers, not generative | citations are provable, no API key needed | can't synthesize across spans |

## Phase 2 — Failure analysis (days 4–6, ~2 hrs, yours to own)

- [ ] Run the eval on your golden set. Open every miss, not just the score.
- [ ] Categorize each miss: chunking too coarse / query too vague / embedding blind spot / genuinely unanswerable.
- [ ] Write 3–5 sentences per category for the README. This section is your best interview material; "here's where it breaks and what I'd try next" is what senior interviewers actually ask about.

## Phase 3 — Story + polish (days 7–10, ~2 hrs)

- [ ] README sections: architecture, results table, failure cases, "what I'd do next" (cross-encoder reranker, generative answers with citations, pgvector for multi-user).
- [ ] 2-minute demo script: problem → what I built → how I measured it → where it breaks → what's next.
- [ ] Optional: Streamlit UI polish, short demo video.

## Time budget

~6–8 hours total across 8–10 days, in small sittings. The job sprint comes first; this fits around it.

# Corpus content notes (audited 2026-10-19; corrected 2026-10-20)

## Resolution (2026-10-20)

Root cause found: the arXiv ID list in `scripts/fetch_corpus.py` itself
carried five wrong IDs, so the script faithfully downloaded the wrong
papers. The fetch list is corrected, the wrong files are removed, and the
intended papers are in place, each verified against its own first page
(title + authors) and recorded with a SHA-256 in `manifest.json` (v2
format). `scripts/validate_corpus.py` re-checks files, hashes, and titles
offline; it currently passes 20/20.

| File (removed) | Claimed | Actually contained | Replacement |
|---|---|---|---|
| 2002.08904.pdf | REALM | Twisted double bilayer graphene physics (He et al.) | 2002.08909 = REALM (Guu et al.) |
| 2002.08525.pdf | Fusion-in-Decoder | NN convection models for climate (Beucler et al.) | 2007.01282 = FiD (Izacard & Grave) |
| 2104.08691.pdf | RETRO | Parameter-efficient prompt tuning (Lester et al.) | 2112.04426 = RETRO (Borgeaud et al.) |
| 2312.10997.pdf | RAPTOR | RAG survey (Gao et al.) | — (survey was never an intended corpus paper) |
| 2401.18059.pdf | Corrective RAG | RAPTOR (Sarthi et al.) — the file was RAPTOR all along | kept; CRAG added as 2401.15884 (Yan et al.) |

Consequences for the evaluation: golden questions never referenced the
removed files as expected documents, and the one question expecting
2401.18059 now has a true label (the file really is RAPTOR). On the
corrected corpus, stage-1 metrics moved slightly (hit@1 0.812, MRR 0.887,
citation precision 0.863) and the misses changed character: the Self-RAG
question's rank-1 slot is now held by CRAG (2401.15884), a genuinely
related paper, instead of the mislabeled survey. The cross-encoder
reranker still resolves every miss (hit@1 1.000, citation precision
0.988). Metrics were re-measured on the corrected corpus; see the
README's evaluation table.

## Historical audit (2026-10-19, superseded by the fix above)

`manifest.json` titles did NOT match the actual PDF content for five
files. Golden questions were grounded only in the verified files:
2005.11401 (RAG), 2004.04906 (DPR), 2309.15217 (RAGAs), 2307.03172 (Lost
in the Middle), 2310.11511 (Self-RAG), 2306.05685 (LLM-as-a-Judge),
2303.16634 (G-Eval), 2407.04051 (FunAudioLLM), 2410.08035 (IntrinsicVoice),
2501.18096 (MILS), 2505.19134 (golden questions), gsad-sdee-2022,
electronics-10-02717, pannuke-handcrafted-features,
cancer-ensemble-segmentation.

## Not in the corpus (uploaded 2026-10-19, parked by decision)

- 2609.38419 — Colorectal Cancer Segmentation (Caglar & Temizel)
- NeurIPS-2025 LLM lossless image compression paper (P2-LLM)
- SN-FPN pathology segmentation paper (Lee, Koganti et al.)

Their golden questions live in `data/golden_questions_new_papers.jsonl`,
kept separate until a decision is made to ingest them.

## Keyword uniqueness traps (checked corpus-wide 2026-10-19)

- "reflection tokens" appears in 2310.11511 AND (formerly) the survey
  2312.10997; the survey is removed, but "critique tokens" remains the
  safer Self-RAG keyword.
- "CosyVoice" appears in 2407.04051 and 2410.08035; "SenseVoice" is
  FunAudioLLM-only.
- "0.618" appears in gsad-sdee-2022 and pannuke-handcrafted-features;
  "golden section" is GSAD-only.
- DPR and RAG both use 21M 100-word Wikipedia chunks but different dumps
  (DPR: Dec 2018, 21,015,324 passages; RAG: Dec 2016, 21M docs) — pair
  numbers with dump date.

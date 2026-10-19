# Corpus content notes (audited 2026-10-19)

`manifest.json` titles do NOT always match the actual PDF content.
Golden questions must ground `expected_doc_id` in the ACTUAL content, not the manifest title.

## Mislabeled files (do not use until fixed)

| File | Manifest says | Actually contains |
|---|---|---|
| 2002.08904.pdf | REALM | Twisted double bilayer graphene physics (Minhao He et al.) |
| 2002.08525.pdf | Fusion-in-Decoder | Physically-consistent NN convection models, climate (Beucler et al.) |
| 2104.08691.pdf | RETRO | Parameter-efficient prompt tuning (Lester et al.) |
| 2312.10997.pdf | RAPTOR | "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.) |
| 2401.18059.pdf | Corrective RAG (CRAG) | RAPTOR (Sarthi et al.) — actual RAPTOR paper lives here |

Notes:
- There is NO Corrective RAG paper in the corpus; nothing covers Correct/Incorrect/Ambiguous retrieval actions.
- 2312.10997 (RAG survey) IS RAG-relevant but its ID is misleading; its invented taxonomy is "Naive RAG / Advanced RAG / Modular RAG".
- 2401.18059 is a truthful ID for RAPTOR (that is RAPTOR's real arXiv ID).

## Verified-clean files used in golden_questions_v1.jsonl

2005.11401 (RAG), 2004.04906 (DPR), 2309.15217 (RAGAs), 2307.03172 (Lost in the Middle),
2310.11511 (Self-RAG), 2306.05685 (LLM-as-a-Judge), 2303.16634 (G-Eval), 2407.04051 (FunAudioLLM),
2410.08035 (IntrinsicVoice), 2501.18096 (MILS), 2505.19134 (golden questions),
gsad-sdee-2022, electronics-10-02717, pannuke-handcrafted-features, cancer-ensemble-segmentation.

## Uploaded 2026-10-19 (not yet in corpus — pending her call on ingestion)

| Proposed ID | File | Content |
|---|---|---|
| 2609.38419 | 2609.38419v1.pdf | Colorectal Cancer Segmentation with Adaptive Augmentation and Multi-Resolution Ensemble Models (Caglar & Temizel) |
| llm-lossless-image-compression | NeurIPS-2025-...-Paper-Conference.pdf | Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need (P2-LLM) |
| sn-fpn-pathology-segmentation | SN-FPN_...pdf | SN-FPN: Self-Attention Nested Feature Pyramid Network for Digital Pathology Image Segmentation (KSU: Sanghoon Lee, Sai Chandana Koganti — same group as the other two KSU papers; uses BCSS like cancer-ensemble-segmentation) |

Golden questions for these three live in `data/golden_questions_new_papers.jsonl` (kept separate until ingested).
Note: all other 2026-10-19 uploads were byte- or text-identical to corpus files — no version conflicts.

## Keyword uniqueness traps (checked corpus-wide 2026-10-19)

- "reflection tokens" appears in 2310.11511 AND the survey 2312.10997; use "critique tokens" (Self-RAG only).
- "CosyVoice" appears in 2407.04051 and 2410.08035; "SenseVoice" is FunAudioLLM-only.
- "0.618" appears in gsad-sdee-2022 and pannuke-handcrafted-features; "golden section" is GSAD-only.
- DPR and RAG both use 21M 100-word Wikipedia chunks but different dumps (DPR: Dec 2018, 21,015,324 passages; RAG: Dec 2016, 21M docs) — pair numbers with dump date.
- RAG-Sequence/RAG-Token = 2005.11401 only; Naive/Advanced/Modular RAG = survey 2312.10997 only.
- "DeepLabv3" is now shared by THREE papers (cancer-ensemble-segmentation, 2609.38419, SN-FPN) — never use it alone as a keyword; pair with PgEns / SN-FPN / adaptive augmentation.
- "diffusion" appears in 2407.04051 and 2501.18096 but neither discusses learning-rate schedules — the v1 diffusion unanswerable still holds.
- "class probability-guided" appears in BOTH KSU papers (pannuke overlaps cancer) — use "PgEns" (cancer-only) instead.
- "CosyVoice" appears in 2407.04051 and 2410.08035; "SenseVoice" is FunAudioLLM-only.
- P2-LLM renders as "P2 -LLM" in extracted text — keyword used is "next-pixel prediction" instead.

"""Tests for retrieval, fusion, and the insufficient-evidence failure mode.

The fixture index is tiny and synthetic on purpose: these tests run in
seconds with no corpus download, and every score is deterministic because
the same model on the same text always produces the same vectors.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import answer as answer_module  # noqa: E402
from src import embed, index as vector_index  # noqa: E402
from src import ingest, retrieve  # noqa: E402

SYNTHETIC_DOCS = {
    "doc_a": (
        "Retrieval augmented generation combines a retriever with a "
        "sequence to sequence model. The retriever finds relevant Wikipedia "
        "passages and the generator conditions on them."
    ),
    "doc_b": (
        "Evaluation of language models uses benchmarks like MT-Bench. Human "
        "preference judgments are collected to rank model outputs."
    ),
    "doc_c": (
        "Dense passage retrieval trains two BERT encoders with contrastive "
        "loss. Hard negatives mined from BM25 improve training."
    ),
}

# (question, expected_doc_id): the fixture's miniature golden set.
GOLDEN_QUESTIONS = [
    ("How does retrieval augmented generation work?", "doc_a"),
    ("What loss is used to train dense passage retrieval?", "doc_c"),
    ("How are language models evaluated?", "doc_b"),
]


@pytest.fixture(scope="module")
def retriever():
    """Build a 3-chunk index from the synthetic docs. Built once per run."""
    chunks = []
    for doc_id, text in SYNTHETIC_DOCS.items():
        chunks.append(
            {
                "doc_id": doc_id,
                "page": 1,
                "char_start": 0,
                "char_end": len(text),
                "text": text,
            }
        )
    embedder = embed.Embedder()
    vectors = embedder.embed_passages([chunk["text"] for chunk in chunks])
    faiss_index = vector_index.build_index(vectors)
    return retrieve.Retriever(chunks, embedder, faiss_index)


def test_golden_sample_hit_at_3(retriever):
    """Every golden question finds its expected doc in the top 3."""
    for question, expected_doc_id in GOLDEN_QUESTIONS:
        ranked = retriever.hybrid_search(question, top_k=3)
        doc_ids = [chunk["doc_id"] for chunk, _score in ranked]
        assert expected_doc_id in doc_ids, f"missed {expected_doc_id} for: {question}"


def test_answerable_question_returns_cited_spans(retriever):
    """A good question returns quoted spans, each with a valid citation."""
    question = "How does retrieval augmented generation work?"
    ranked = retriever.hybrid_search(question, top_k=3)
    result = answer_module.extractive_answer(question, ranked, retriever.embedder)

    assert result["status"] == "answered"
    assert len(result["spans"]) >= 1
    for span in result["spans"]:
        citation = span["citation"]
        assert citation["doc_id"] == "doc_a"
        assert citation["page"] == 1
        assert 0 <= citation["char_start"] < citation["char_end"]
        assert 0.0 <= citation["score"] <= 1.0
        assert len(span["text"].split()) >= 4


def test_unanswerable_question_returns_insufficient_evidence(retriever):
    """A question the corpus cannot answer must not get a confident answer."""
    question = "What is the capital of France?"
    ranked = retriever.hybrid_search(question, top_k=3)
    result = answer_module.extractive_answer(question, ranked, retriever.embedder)

    assert result["status"] == "insufficient_evidence"
    assert result["spans"] == []
    assert "nearest_chunks" in result


def test_ambiguous_question_returns_insufficient_evidence(retriever):
    """A vague question matches weakly everywhere; refuse instead of guessing."""
    question = "What about training?"
    ranked = retriever.hybrid_search(question, top_k=3)
    result = answer_module.extractive_answer(question, ranked, retriever.embedder)

    assert result["status"] == "insufficient_evidence"
    assert result["spans"] == []


def test_rrf_prefers_agreement_over_single_signal():
    """A chunk ranked highly by BOTH signals beats one ranked #1 by only one.

    chunk_2 is #2 in dense and #2 in BM25: 1/61 + 1/61.
    chunk_1 is #1 in dense only: 1/61.
    Agreement must win, even though chunk_1 has the single best rank.
    """
    dense_ranking = ["chunk_1", "chunk_2", "chunk_3"]
    bm25_ranking = ["chunk_2", "chunk_3", "chunk_4"]
    fused = retrieve.reciprocal_rank_fusion([dense_ranking, bm25_ranking])
    ordered_ids = [chunk_id for chunk_id, _score in fused]

    assert ordered_ids[0] == "chunk_2"
    assert ordered_ids.index("chunk_1") > ordered_ids.index("chunk_2")


def test_chunk_offsets_reconstruct_text():
    """char_start/char_end must slice back to the exact chunk text."""
    text = "The quick brown fox jumps over the lazy dog near the river bank."
    for piece, start, end in ingest.chunk_text(text, chunk_size=20, overlap=5):
        assert text[start:end] == piece

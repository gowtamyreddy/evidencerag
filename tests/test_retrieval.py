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
from src import rerank as rerank_module  # noqa: E402

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
        # The source-context view relies on the chunk text carrying the
        # quoted sentence at exactly the relative offsets.
        assert span["chunk_text"][span["rel_start"]:span["rel_end"]] == span["text"]
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


class _StubReranker:
    """Scores chunks by a fixed keyword so ordering is fully deterministic."""

    def __init__(self, keyword):
        self.keyword = keyword

    def score(self, query, texts):
        return [1.0 if self.keyword in text else 0.0 for text in texts]


def test_rerank_reorders_by_cross_encoder_score():
    """A chunk the fusion stages ranked last must win if the cross-encoder
    scores it highest — that reordering is the entire point of stage two."""
    chunks = [
        {"doc_id": "generic", "page": 1, "char_start": 0, "char_end": 10,
         "text": "retrieval retrieval retrieval generic words"},
        {"doc_id": "specific", "page": 1, "char_start": 0, "char_end": 10,
         "text": "the precise answer mentions goldenneedle here"},
    ]
    fused = [(chunks[0], 0.9), (chunks[1], 0.1)]  # fusion prefers generic
    ranked = rerank_module.rerank("q", fused, _StubReranker("goldenneedle"), top_k=2)
    assert ranked[0][0]["doc_id"] == "specific"
    assert ranked[0][1] == 1.0


def test_clean_text_strips_lone_surrogates():
    """Regression test for the 130-paper build crash: arXiv 2606.22352
    (a vision-tokens paper) extracts with unpaired UTF-16 surrogates for
    its math glyphs. They are legal Python str content but cannot be
    UTF-8 encoded, so the embedding tokenizer died with a misleading
    TypeError. clean_text must remove them."""
    dirty = "Vision Tokens \ud835\ud835 D and \ud800 more text"
    cleaned = ingest.clean_text(dirty)
    cleaned.encode("utf-8")  # raises if a lone surrogate survived
    assert "Vision" in cleaned and "more" in cleaned


def test_sentence_offsets_handle_repeated_sentences():
    """The same sentence twice in one chunk must get two distinct spans,
    each slicing back to the sentence — find-from-cursor, not find-from-0."""
    text = "Results improve with scale. Results improve with scale. Other text here now."
    spans = answer_module._sentences_with_offsets(text)
    repeated = [(s, a, b) for s, a, b in spans if s.startswith("Results improve")]
    assert len(repeated) == 2
    assert repeated[0][1] < repeated[1][1]
    for sentence, start, end in spans:
        assert text[start:end] == sentence


def test_empty_pages_produce_no_chunks():
    """Blank pages (scanned-image pages, separators) must not create
    chunks, and must not shift the page numbers of later pages."""
    pages = [(1, ""), (2, "   "), (3, "Real content lives on this page here.")]
    chunks = ingest.chunk_pages("doc_x", pages)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 3


def test_unicode_text_chunks_reconstruct():
    """Curly quotes, em dashes, and accented characters survive chunking
    with offsets that still slice back to the exact chunk text."""
    text = "L'évaluation «hybride» — dense et lexical — améliore la précision des citations."
    for piece, start, end in ingest.chunk_text(text, chunk_size=30, overlap=5):
        assert text[start:end] == piece

"""Evaluate retrieval quality against a golden JSONL file.

Golden entries look like:
  {"question": "...", "expected_doc_ids": ["2005.11401"], "type": "factual"}

Metrics:
- hit@k: share of questions where a chunk from an expected doc is in the top k.
- MRR: mean reciprocal rank of the first chunk from an expected doc (0 if none
  in the top k).
- citation precision: share of the returned top-5 chunks that come from an
  expected doc. This measures whether the citations we would show are
  pointing at the right document.

Unanswerable and ambiguous questions are reported separately, because for
them the correct behavior is insufficient_evidence, not retrieval.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import embed, index as vector_index, ingest, retrieve  # noqa: E402

EVAL_TOP_K = 5


def load_golden(path):
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Accept both "expected_doc_id" (a single string) and
            # "expected_doc_ids" (a list); normalize to a list.
            if "expected_doc_ids" not in entry and "expected_doc_id" in entry:
                doc_id = entry["expected_doc_id"]
                entry["expected_doc_ids"] = [doc_id] if doc_id else []
            entries.append(entry)
    return entries


def load_retriever(artifacts_dir):
    artifacts_dir = Path(artifacts_dir)
    chunks = ingest.load_chunks(artifacts_dir / "chunks.jsonl")
    faiss_index = vector_index.load_index(artifacts_dir / "index.faiss")
    embedder = embed.Embedder()
    return retrieve.Retriever(chunks, embedder, faiss_index)


def rank_of_expected(ranked_chunks, expected_doc_ids):
    """Position (1-based) of the first chunk from an expected doc, or None."""
    for position, (chunk, _score) in enumerate(ranked_chunks, start=1):
        if chunk["doc_id"] in expected_doc_ids:
            return position
    return None


def evaluate(retriever, golden, top_k=EVAL_TOP_K):
    """Run every golden question and aggregate the metrics.

    Unanswerable and ambiguous questions are excluded from the retrieval
    metrics: for them the correct behavior is insufficient_evidence, not
    retrieval, so scoring them as misses would punish honest refusal.
    """
    answerable = [
        entry for entry in golden
        if entry.get("type") not in ("unanswerable", "ambiguous")
    ]
    graceful = [
        entry for entry in golden
        if entry.get("type") in ("unanswerable", "ambiguous")
    ]
    results = []
    for entry in answerable:
        ranked = retriever.hybrid_search(entry["question"], top_k=top_k)
        rank = rank_of_expected(ranked, entry["expected_doc_ids"])
        hits = sum(1 for chunk, _ in ranked if chunk["doc_id"] in entry["expected_doc_ids"])
        results.append(
            {
                "question": entry["question"],
                "type": entry.get("type", "factual"),
                "rank": rank,
                "precision": hits / top_k,
            }
        )
    return results


def summarize(results):
    total = len(results)
    summary = {"n": total}
    for k in (1, 3, 5):
        summary[f"hit@{k}"] = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k) / total
    summary["mrr"] = sum(1.0 / r["rank"] for r in results if r["rank"] is not None) / total
    summary["citation_precision"] = sum(r["precision"] for r in results) / total
    return summary


def print_table(summary):
    print(f"{'metric':<20}{'value':>8}")
    print("-" * 28)
    print(f"{'questions':<20}{summary['n']:>8}")
    for key in ("hit@1", "hit@3", "hit@5", "mrr", "citation_precision"):
        print(f"{key:<20}{summary[key]:>8.3f}")


def main():
    artifacts_dir = PROJECT_ROOT / "data" / "artifacts"
    golden_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "golden_sample.jsonl"
    print("Loading retriever...")
    retriever = load_retriever(artifacts_dir)
    golden = load_golden(golden_path)
    print(f"Evaluating {len(golden)} golden questions...\n")
    results = evaluate(retriever, golden)
    for entry in results:
        rank = entry["rank"] if entry["rank"] is not None else "miss"
        print(f"[{entry['type']:<10}] rank={rank:<4} {entry['question'][:70]}")
    print()
    print_table(summarize(results))
    graceful = [e for e in golden if e.get("type") in ("unanswerable", "ambiguous")]
    if graceful:
        print("\nGraceful-failure questions (excluded from metrics above):")
        for entry in graceful:
            print(f"  [{entry['type']:<12}] {entry['question'][:70]}")


if __name__ == "__main__":
    main()

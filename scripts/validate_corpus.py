"""Validate the curated corpus against data/papers/manifest.json.

For every manifest entry, this checks — fully offline — that:
  1. the PDF file exists,
  2. its SHA-256 matches the manifest (the file is the file we audited),
  3. the paper's own first two pages contain its claimed title
     (word-coverage check, robust to PDF text-extraction spacing noise).

Why this exists: the corpus once contained five files whose contents did
not match their filenames/manifest titles, because the fetch list carried
wrong arXiv IDs. The bad files silently poisoned retrieval and the golden
evaluation. Run this after any corpus change:

    python scripts/validate_corpus.py

Exit code 0 means every curated document is what the manifest claims.
The bulk corpus (scripts/fetch_bulk_corpus.py) is validated separately:
its manifest records the title Semantic Scholar returned for each ID.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import ingest  # noqa: E402

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
MANIFEST = PAPERS_DIR / "manifest.json"
TITLE_WORD_MIN_LEN = 4
TITLE_COVERAGE_REQUIRED = 0.8


def _squash(text):
    """Lowercase and strip everything but alphanumerics (extraction-proof)."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def title_coverage(title, pages_text):
    """Share of the title's significant words found in the paper's text."""
    haystack = _squash(pages_text)
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) >= TITLE_WORD_MIN_LEN]
    if not words:
        return 1.0
    found = sum(1 for w in words if _squash(w) in haystack)
    return found / len(words)


def validate_entry(doc_id, entry):
    problems = []
    path = PAPERS_DIR / f"{doc_id}.pdf"
    if not path.exists():
        return [f"{doc_id}: file missing"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.get("sha256"):
        problems.append(f"{doc_id}: sha256 mismatch (file changed since audit)")
    pages = ingest.read_pdf(path, doc_id)
    first_text = " ".join(text for _num, text in pages[:2])
    coverage = title_coverage(entry["title"], first_text)
    if coverage < TITLE_COVERAGE_REQUIRED:
        problems.append(
            f"{doc_id}: title coverage {coverage:.2f} < {TITLE_COVERAGE_REQUIRED} "
            f"(claimed title not found in the paper's first pages)"
        )
    return problems


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"Validating {len(manifest)} curated documents against manifest.json...\n")
    all_problems = []
    for doc_id, entry in manifest.items():
        if not isinstance(entry, dict) or "title" not in entry:
            all_problems.append(f"{doc_id}: manifest entry is not in v2 format")
            print(f"  FORMAT  {doc_id}")
            continue
        problems = validate_entry(doc_id, entry)
        if problems:
            all_problems.extend(problems)
            print(f"  FAIL    {doc_id}")
        else:
            print(f"  ok      {doc_id} — {entry['title'][:64]}")
    print()
    if all_problems:
        print(f"{len(all_problems)} problem(s):")
        for problem in all_problems:
            print(f"  - {problem}")
        sys.exit(1)
    print(f"All {len(manifest)} documents verified: files, hashes, and titles match.")


if __name__ == "__main__":
    main()

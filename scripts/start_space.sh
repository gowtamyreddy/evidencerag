#!/usr/bin/env bash
# First start on a fresh Space: fetch the corpus and build the index.
# Later starts find data/artifacts/ already built and skip straight to serving.
set -e
cd "$(dirname "$0")/.."

if [ ! -f data/artifacts/index.faiss ]; then
  echo "First start: fetching corpus..."
  python scripts/fetch_corpus.py
  python scripts/fetch_bulk_corpus.py
  echo "Building index..."
  python scripts/build_index.py
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 7860

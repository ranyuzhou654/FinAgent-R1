#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INDEX_DIR="data/indexes/bm25"
CORPUS_DIR="data/corpus"

mkdir -p "$INDEX_DIR"

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$CORPUS_DIR" \
  --index "$INDEX_DIR" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions \
  --storeDocvectors \
  --storeRaw

echo "BM25 index build complete: $INDEX_DIR"


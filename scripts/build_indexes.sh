#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python scripts/build_corpus.py
bash scripts/build_bm25_index.sh
python scripts/build_dense_index.py

echo "Index build finished."


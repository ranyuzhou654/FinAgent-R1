#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INDEX_DIR="data/indexes/bm25"
CORPUS_DIR="data/corpus"
CORPUS_FILE="$CORPUS_DIR/financial_passages.jsonl"

errors=()

if ! command -v python >/dev/null 2>&1; then
  errors+=("python is not available in PATH.")
elif ! python -c "import pyserini" >/dev/null 2>&1; then
  errors+=("pyserini is not installed in the current Python environment. Run: pip install pyserini")
fi

if ! command -v java >/dev/null 2>&1; then
  errors+=("java is not available in PATH. Install JDK 21 and expose it via JAVA_HOME/PATH.")
else
  JAVA_VERSION_OUTPUT="$(java -version 2>&1 | head -n1)"
  JAVA_MAJOR="$(printf '%s\n' "$JAVA_VERSION_OUTPUT" | sed -E 's/.*version "([0-9]+)(\.[^"]*)?".*/\1/')"
  if [[ ! "$JAVA_MAJOR" =~ ^[0-9]+$ ]]; then
    errors+=("failed to parse Java version from: $JAVA_VERSION_OUTPUT")
  elif (( JAVA_MAJOR < 21 )); then
    errors+=("Pyserini BM25 indexing requires Java 21+, but found: $JAVA_VERSION_OUTPUT")
  fi
fi

if [[ ! -f "$CORPUS_FILE" ]]; then
  errors+=("missing corpus file: $CORPUS_FILE. Run: python scripts/build_corpus.py")
fi

if (( ${#errors[@]} > 0 )); then
  echo "[BM25] Cannot build the index because prerequisites are missing:"
  for err in "${errors[@]}"; do
    echo "  - $err"
  done
  echo
  echo "macOS example:"
  echo "  brew install --cask temurin@21"
  echo "  export JAVA_HOME=\$(/usr/libexec/java_home -v 21)"
  echo "  export PATH=\"\$JAVA_HOME/bin:\$PATH\""
  exit 1
fi

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

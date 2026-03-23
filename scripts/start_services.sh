#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python tools/retrieval_server.py &
RETRIEVAL_PID=$!

python demo/backend/main.py &
BACKEND_PID=$!

echo "Retrieval server PID: $RETRIEVAL_PID"
echo "Backend server PID: $BACKEND_PID"
echo "Use 'kill $RETRIEVAL_PID $BACKEND_PID' to stop both services."

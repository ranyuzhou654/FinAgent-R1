#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B}"
export MAX_TURNS="${MAX_TURNS:-5}"

exec uvicorn demo.backend.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --reload

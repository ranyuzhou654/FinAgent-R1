#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export WANDB_PROJECT="${WANDB_PROJECT:-FinAgent-R1}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/vendor/Search-R1:${PYTHONPATH:-}"

PYTHONUNBUFFERED=1 python training/finagent_verl_main.py "$@"


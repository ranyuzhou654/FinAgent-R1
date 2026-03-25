#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/raw
cd data/raw

echo "== Downloading FinQA =="
python - <<'PY'
from datasets import load_dataset

dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
dataset.save_to_disk("finqa_hf")
print("Saved FinQA to data/raw/finqa_hf")
for split, items in dataset.items():
    print(f"  {split}: {len(items)}")
PY

echo "== Downloading ConvFinQA =="
if [ ! -d "ConvFinQA" ]; then
  git clone https://github.com/czyssrs/ConvFinQA.git
else
  echo "ConvFinQA already exists, skipping clone."
fi

echo "== Downloading optional Fino1 reasoning paths =="
python - <<'PY'
from datasets import load_dataset

try:
    dataset = load_dataset("TheFinAI/Fino1_Reasoning_Path_FinQA_v2")
    dataset.save_to_disk("fino1_hf")
    print("Saved Fino1 to data/raw/fino1_hf")
except Exception as exc:
    print(f"Skipped Fino1 download: {exc}")
PY

echo "== Data download complete =="


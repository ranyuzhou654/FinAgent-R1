# FinAgent-R1

FinAgent-R1 is a financial multi-tool agent project inspired by Search-R1. The
system trains an LLM to decide when to search report passages, run financial
calculations, and query structured tables with SQL.

## Project Structure

The repository is organized into five layers:

1. Data preparation for FinQA-style report reasoning.
2. Tool backends for retrieval, calculation, and SQL access.
3. SFT cold-start training for tool-tag formatting.
4. GRPO reinforcement learning with multi-tool rewards.
5. Evaluation and demo services.

## Quick Start

1. Create a Python 3.10+ environment.
2. Install dependencies from `requirements.txt`.
3. Download and preprocess data with the scripts in `scripts/`.
4. Build indexes and the SQLite database.
5. Convert FinQA into veRL parquet data with `python scripts/prepare_verl_finqa_data.py`.
6. Test tools before training.
7. Launch multi-turn GRPO with `bash scripts/train_grpo_verl.sh`.

## Notes

- The repository defaults to `faiss-cpu` for portability. Replace it with a GPU
  build in your training environment if needed.
- Unsloth is optional. A pure TRL fallback training script is included.
- Search-R1 is vendored in `vendor/Search-R1/` and reused as the veRL training runtime.
- Large datasets, indexes, checkpoints, and experiment logs are gitignored.

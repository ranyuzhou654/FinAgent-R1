# FinAgent-R1

FinAgent-R1 is a financial multi-tool agent project inspired by Search-R1. The
system trains an LLM to decide when to search report passages, run financial
calculations, and query structured tables with SQL.

## Project Structure

The repository is organized into six layers:

1. Data preparation for FinQA-style report reasoning.
2. Tool backends for retrieval, calculation, and SQL access.
3. Optional SFT cold-start scripts for tool-tag formatting.
4. Search-R1-style veRL multi-turn GRPO as the main training path.
5. Optional TRL / Unsloth compatibility training scripts.
6. Evaluation plus FastAPI, Gradio, and React demo surfaces.

## Quick Start

1. Create a Python 3.10+ environment.
2. Install dependencies from `requirements.txt`.
3. Download and preprocess data with the scripts in `scripts/`.
4. Build indexes and the SQLite database.
5. Convert FinQA into veRL parquet data with `python scripts/prepare_verl_finqa_data.py`.
6. Test tools with `python scripts/test_tools.py`.
7. Launch the main veRL training path with `bash scripts/train_grpo_verl.sh`.

## Optional Pipelines

- Generate SFT seed data with `python scripts/generate_sft_data.py`.
- Run SFT cold-start with `python training/sft_coldstart.py`.
- Smoke-test the SFT adapter with `python scripts/test_sft_model.py`.
- Run pure TRL GRPO with `python training/grpo_train.py`.
- Run Unsloth GRPO with `python training/grpo_train_unsloth.py`.

## Demo Surfaces

- FastAPI backend: `python demo/backend/main.py`
- Gradio app: `python demo/gradio_app.py`
- React frontend: `cd demo/frontend && npm install && npm run dev`
- Docker Compose: `docker compose up --build`

## Evaluation

- Main evaluation: `python eval/evaluate.py <model_path> --max-samples 50`
- Ablation runner: `python eval/ablation.py`

## Notes

- The repository defaults to `faiss-cpu` for portability. Replace it with a GPU
  build in your training environment if needed.
- veRL is the primary RL training path. TRL and Unsloth are compatibility paths.
- Search-R1 is vendored in `vendor/Search-R1/` and reused as the veRL training runtime.
- Large datasets, indexes, checkpoints, and experiment logs are gitignored.

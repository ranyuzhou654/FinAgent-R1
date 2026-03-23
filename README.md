# FinAgent-R1 🏦🔎🧮

[中文说明 🇨🇳](docs/README.zh-CN.md)

FinAgent-R1 is a financial multi-tool agent project inspired by Search-R1. The
system trains an LLM to decide when to search report passages, run financial
calculations, and query structured tables with SQL.

## ✨ What This Repo Contains

- 📚 A FinQA-oriented data pipeline that converts raw report reasoning examples
  into JSONL and veRL parquet datasets.
- 🛠️ Three tool backends: report search, financial calculation, and SQLite access.
- 🤖 A Search-R1-style veRL multi-turn GRPO training path as the primary RL setup.
- 🧪 Optional SFT, TRL, and Unsloth compatibility scripts for smaller-scale runs.
- 🖥️ Evaluation plus FastAPI, Gradio, React, and Docker demo surfaces.

## 🧱 Project Structure

The repository is organized into six layers:

1. 📦 Data preparation for FinQA-style report reasoning.
2. 🔍 Tool backends for retrieval, calculation, and SQL access.
3. 📝 Optional SFT cold-start scripts for tool-tag formatting.
4. 🚀 Search-R1-style veRL multi-turn GRPO as the main training path.
5. 🔁 Optional TRL / Unsloth compatibility training scripts.
6. 📊 Evaluation plus FastAPI, Gradio, and React demo surfaces.

## ⚡ Quick Start

1. Create a Python 3.10+ environment.
2. Install dependencies from `requirements.txt`.
3. Download and preprocess data with the scripts in `scripts/`.
4. Build indexes and the SQLite database.
5. Convert FinQA into veRL parquet data with `python scripts/prepare_verl_finqa_data.py`.
6. Test tools with `python scripts/test_tools.py`.
7. Launch the main veRL training path with `bash scripts/train_grpo_verl.sh`.

## 📘 Usage Guide

### 1. 🧰 Prepare The Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to use the React frontend:

```bash
cd demo/frontend
npm install
cd ../..
```

### 2. 🏗️ Build The Data Pipeline

Raw FinQA data is expected under `data/raw/finqa_hf`. Once the dataset is in
place, run the repository pipeline in this order:

```bash
# Optional: inspect the raw dataset structure first
python scripts/explore_data.py

# Convert FinQA into train/validation/test JSONL
python scripts/prepare_training_data.py

# Build retrieval corpus and SQL artifacts
python scripts/build_corpus.py
python scripts/build_sql_database.py
python scripts/build_question_table_map.py

# Build BM25 + dense retrieval indexes
bash scripts/build_indexes.sh

# Convert processed JSONL into veRL parquet
python scripts/prepare_verl_finqa_data.py
```

Expected outputs:

- `data/processed/train.jsonl`, `validation.jsonl`, `test.jsonl`
- `data/corpus/financial_passages.jsonl`
- `data/tables/financial_data.db`
- `data/tables/question_table_map.json`
- `data/indexes/...`
- `data/verl/finqa/train.parquet`, `validation.parquet`, `test.parquet`

### 3. ✅ Verify The Tool Stack

Before training, validate that retrieval, calculator, and SQL execution all
work from the command line:

```bash
python scripts/test_tools.py
```

### 4. 🤖 Run The Main Training Path

The primary training path is veRL-based multi-turn GRPO.

Terminal 1:

```bash
python tools/retrieval_server.py
```

Terminal 2:

```bash
bash scripts/train_grpo_verl.sh
```

Common overrides:

```bash
bash scripts/train_grpo_verl.sh \
  trainer.total_training_steps=100 \
  trainer.save_freq=50 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B
```

The veRL config lives in `configs/verl_ppo_finqa.yaml`. By default, checkpoints
go to `checkpoints/grpo_verl`.

## 🧪 Optional Pipelines

- 🪴 Generate SFT seed data with `python scripts/generate_sft_data.py`.
- 🧠 Run SFT cold-start with `python training/sft_coldstart.py`.
- 🔎 Smoke-test the SFT adapter with `python scripts/test_sft_model.py`.
- 🔁 Run pure TRL GRPO with `python training/grpo_train.py`.
- ⚙️ Run Unsloth GRPO with `python training/grpo_train_unsloth.py`.

Recommended order if you want the optional branch:

```bash
python scripts/generate_sft_data.py
python training/sft_coldstart.py
python scripts/test_sft_model.py
python training/grpo_train.py
```

## 🖥️ Demo Surfaces

- FastAPI backend: `python demo/backend/main.py`
- Gradio app: `python demo/gradio_app.py`
- React frontend: `cd demo/frontend && npm install && npm run dev`
- Docker Compose: `docker compose up --build`

Typical local demo flow:

Terminal 1:

```bash
python tools/retrieval_server.py
```

Terminal 2:

```bash
MODEL_PATH=Qwen/Qwen2.5-3B python demo/backend/main.py
```

Terminal 3, choose one:

```bash
python demo/gradio_app.py
```

or

```bash
cd demo/frontend
npm run dev
```

The backend exposes:

- `GET /api/health`
- `POST /api/ask`
- `POST /api/ask_stream`

## 📏 Evaluation

- Main evaluation: `python eval/evaluate.py <model_path> --max-samples 50`
- Ablation runner: `python eval/ablation.py`

Examples:

```bash
python eval/evaluate.py Qwen/Qwen2.5-3B --max-samples 20
python eval/ablation.py
```

`eval/evaluate.py` currently reports:

- ✅ Execution Accuracy
- 🧩 Heuristic Program Accuracy
- 🛠️ Tool Usage Rate
- 🔀 Multi-Tool Rate

## 🔄 Pipeline And Principles

### 🏋️ Training Pipeline

The repository follows this high-level flow:

1. Raw FinQA examples are converted into processed JSONL with prompt, answer,
   program, context, and table fields.
2. The report passages are materialized into a retrieval corpus, while tables
   are loaded into SQLite and linked to question IDs.
3. Retrieval indexes are built for BM25 and dense search.
4. Processed JSONL is converted into Search-R1-style veRL parquet, where each
   row contains chat prompts plus reward metadata.
5. veRL samples multi-turn trajectories where the model emits tool tags such as
   `<search>...</search>` or `<sql>...</sql>`.
6. Tool outputs are fed back into the context as `<observation>...</observation>`
   so the model can continue reasoning before answering.
7. Rule-based rewards score answer correctness and agent behavior, and GRPO
   updates the policy.

### 🧭 Inference Pipeline

At evaluation time or in the demo backend, the agent loop is:

1. Build the system + user prompt.
2. Generate the next assistant segment.
3. Detect the first tool tag in the generated text.
4. Execute the corresponding backend:
   `search -> retrieval`, `calculate -> calculator`, `sql -> SQLite`.
5. Append the tool result as an observation.
6. Repeat until the model emits `<answer>...</answer>` or the turn budget ends.

### 🆚 Why Multi-Turn RL Instead Of Plain RAG

This repository is not just a prompt-augmented RAG baseline. The model is
trained to decide:

- whether it needs a tool at all
- which tool to call
- when to stop calling tools
- how to integrate observations across multiple turns

That is the main reason the veRL path is the primary training setup.

### 🎯 Reward Design

The current rule-based reward stack combines:

- answer accuracy
- format and structured final-answer compliance
- whether tools were used when needed
- whether multiple tools were coordinated
- penalties for invalid retries, no-tool answers, and overuse

The veRL path consumes these rewards through `training/finagent_verl_main.py`,
while the optional TRL / Unsloth scripts use compatibility wrappers in
`training/reward_functions.py`.

## 📝 Notes

- The repository defaults to `faiss-cpu` for portability. Replace it with a GPU
  build in your training environment if needed.
- veRL is the primary RL training path. TRL and Unsloth are compatibility paths.
- Search-R1 is vendored in `vendor/Search-R1/` and reused as the veRL training runtime.
- Large datasets, indexes, checkpoints, and experiment logs are gitignored.

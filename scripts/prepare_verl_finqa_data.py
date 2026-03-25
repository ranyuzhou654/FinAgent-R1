"""Convert processed FinQA JSONL files into Search-R1-style veRL parquet data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
QUESTION_TABLE_MAP_PATH = ROOT_DIR / "data" / "tables" / "question_table_map.json"
OUTPUT_DIR = ROOT_DIR / "data" / "verl" / "finqa"


SYSTEM_PROMPT = """You are a financial analysis agent.
You must reason inside <think>...</think>.
You may use the following tools multiple times:
- <search>query</search> to search the financial report corpus. Results come back inside <observation>...</observation>.
- <calculate>expression</calculate> to run financial calculations or FinQA programs.
- <sql>query</sql> to query the SQLite table for this question.
When you finish, provide the final answer inside <answer>...</answer>."""


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def _load_table_map() -> dict[str, str]:
    if not QUESTION_TABLE_MAP_PATH.exists():
        return {}
    with QUESTION_TABLE_MAP_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_user_message(example: dict, table_name: str | None) -> str:
    lines = [f"Question: {example['question']}"]
    if table_name:
        lines.append(f"SQL table for this question: {table_name}")
        lines.append(f"If you need schema, you may call <sql>DESCRIBE {table_name}</sql>.")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    question_table_map = _load_table_map()

    for split in ("train", "validation", "test"):
        input_path = PROCESSED_DIR / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Processed file not found: {input_path}")

        rows = []
        skipped = 0
        for index, example in enumerate(_load_jsonl(input_path)):
            if not str(example.get("answer", "")).strip():
                skipped += 1
                continue
            table_name = question_table_map.get(example["id"])
            rows.append(
                {
                    "data_source": "finqa",
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_message(example, table_name)},
                    ],
                    "ability": "financial-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {
                            "target": str(example["answer"]),
                            "program": example.get("program", ""),
                            "question_id": example["id"],
                        },
                    },
                    "question_id": example["id"],
                    "question": example["question"],
                    "answer": str(example["answer"]),
                    "program": example.get("program", ""),
                    "table_name": table_name,
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "question_id": example["id"],
                        "table_name": table_name,
                    },
                }
            )

        output_path = OUTPUT_DIR / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        print(f"{split}: {len(rows)} rows -> {output_path} (skipped {skipped} empty-answer samples)")


if __name__ == "__main__":
    main()


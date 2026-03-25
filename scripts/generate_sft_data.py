"""Generate rule-based SFT seed data for the optional cold-start pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "train.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "sft_seed.jsonl"
QUESTION_TABLE_MAP_PATH = ROOT_DIR / "data" / "tables" / "question_table_map.json"

SYSTEM_PROMPT = (
    "You are a financial analysis agent with access to search, calculate, and SQL tools. "
    "Use <think>...</think> for reasoning, tool tags for actions, and "
    "<answer>...</answer> for the final answer."
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def load_question_table_map() -> dict[str, str]:
    if not QUESTION_TABLE_MAP_PATH.exists():
        return {}
    with QUESTION_TABLE_MAP_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_assistant_trace(example: dict, table_name: str | None = None) -> str:
    question = example["question"].strip()
    answer = str(example["answer"]).strip()
    program = str(example.get("program", "")).strip()
    context = str(example.get("context", "")).strip()

    context_excerpt = context[:500].strip() if context else "Relevant report passage not available in seed generation."
    search_query = question[:100]
    needs_sql = bool(table_name)
    needs_calculate = bool(program)

    parts = [
        "<think>I should gather the relevant report evidence before answering.</think>",
        f"<search>{search_query}</search>",
        f"<observation>{context_excerpt}</observation>",
        "<think>I have the report context and can decide whether I need structured data or calculation.</think>",
    ]

    if needs_sql:
        parts.extend(
            [
                f"<sql>DESCRIBE {table_name}</sql>",
                f"<observation>Schema loaded for {table_name}.</observation>",
                "<think>The table schema confirms where the structured values live.</think>",
            ]
        )

    if needs_calculate:
        parts.extend(
            [
                f"<calculate>{program}</calculate>",
                f"<observation>Result: {answer}</observation>",
                "<think>The computation matches the target quantity.</think>",
            ]
        )

    parts.append(f"<answer>{answer}</answer>")
    return "\n".join(parts)


def build_sft_example(example: dict, table_map: dict[str, str]) -> dict:
    table_name = table_map.get(example["id"])
    user_lines = [f"Question: {example['question']}"]
    if table_name:
        user_lines.append(f"Relevant SQL table: {table_name}")
    return {
        "id": example["id"],
        "question_id": example["id"],
        "answer": str(example["answer"]),
        "program": example.get("program", ""),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_lines)},
            {
                "role": "assistant",
                "content": build_assistant_trace(example, table_name=table_name),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Processed training data not found: {input_path}")

    records = load_jsonl(input_path)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    table_map = load_question_table_map()
    selected = records[: args.max_samples]
    sft_rows = [build_sft_example(example, table_map) for example in selected]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in sft_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sft_rows)} SFT seed rows to {output_path}")


if __name__ == "__main__":
    main()

"""Map question IDs to the generated SQLite table names."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from build_sql_database import table_name_from_question


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = ROOT_DIR / "data" / "raw" / "finqa_hf"
OUTPUT_PATH = ROOT_DIR / "data" / "tables" / "question_table_map.json"


def main() -> None:
    dataset = load_from_disk(str(RAW_DATASET_PATH))
    question_to_table: dict[str, str] = {}

    for split in ("train", "validation", "test"):
        for example in dataset[split]:
            question_to_table[example["id"]] = table_name_from_question(example["id"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(question_to_table, handle, ensure_ascii=False, indent=2)

    print(f"Question-table map written: {len(question_to_table)} entries -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


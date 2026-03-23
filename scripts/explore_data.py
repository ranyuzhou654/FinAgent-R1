"""Inspect FinQA dataset structure and basic statistics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from datasets import load_from_disk


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "raw" / "finqa_hf"

OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "greater",
    "exp",
    "table_sum",
    "table_average",
    "table_max",
    "table_min",
]


def flatten_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def compact_preview(value: object, limit: int = 200) -> str:
    text = " ".join(flatten_text(value).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def get_program(example: dict) -> str:
    return str(example.get("program") or example.get("program_re") or "")


def preview_table(table: list[list[str]], limit: int = 5) -> str:
    rows = ["   " + " | ".join(str(cell) for cell in row) for row in table[:limit]]
    if len(table) > limit:
        rows.append(f"   ... ({len(table)} rows total)")
    return "\n".join(rows)


def main() -> None:
    dataset = load_from_disk(str(DATASET_PATH))
    example = dataset["train"][0]

    print("=" * 60)
    print("FinQA example structure")
    print("=" * 60)
    print(f"\n1. ID: {example['id']}")
    print(f"\n2. pre_text (first 200 chars):\n{compact_preview(example.get('pre_text', []))}")
    print(f"\n3. table:\n{preview_table(example['table'])}")
    print(f"\n4. post_text (first 200 chars):\n{compact_preview(example.get('post_text', []))}")
    print(f"\n5. question: {example['question']}")
    print(f"\n6. answer: {example['answer']}")
    print(f"\n7. program:\n   {get_program(example) or '<missing>'}")
    print(f"\n8. program_re:\n   {example.get('program_re', '<missing>')}")
    print(f"\n9. gold_inds:\n   {example.get('gold_inds')}")
    print(f"\n10. available keys:\n   {sorted(example.keys())}")

    print("\n" + "=" * 60)
    print("Dataset statistics")
    print("=" * 60)
    for split in ("train", "validation", "test"):
        print(f"  {split}: {len(dataset[split])}")

    op_counter: Counter[str] = Counter()
    for item in dataset["train"]:
        program = get_program(item)
        for op in OPS:
            if op in program:
                op_counter[op] += 1

    print("\nOperation distribution:")
    for op, count in op_counter.most_common():
        print(f"  {op}: {count}")


if __name__ == "__main__":
    main()

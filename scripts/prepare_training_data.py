"""Convert FinQA into JSONL files suitable for training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = ROOT_DIR / "data" / "raw" / "finqa_hf"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

AGENT_SYSTEM_PROMPT = """You are a financial analysis agent. You can use the following tools to answer questions about financial reports:

Tools:
1. <search>query</search> - Search the financial report knowledge base for relevant passages
2. <calculate>expression</calculate> - Execute financial calculations such as PE ratio, growth rate, or percentage change
3. <sql>SQL query</sql> - Query structured financial data tables

Rules:
- Use <think>...</think> for reasoning
- Tool results will be returned in <observation>...</observation>
- Give the final answer in <answer>your answer</answer>
- You may call tools multiple times or not at all
- For numerical answers, provide the exact number

Question: {question}"""


def stringify_table(table: list[list[str]]) -> str:
    if not table:
        return ""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def process_example(example: dict) -> dict:
    table_text = stringify_table(example.get("table") or [])
    context = (
        f"{example.get('pre_text', '').strip()}\n\n"
        f"[Table]\n{table_text}\n\n"
        f"{example.get('post_text', '').strip()}"
    ).strip()

    return {
        "id": example["id"],
        "prompt": AGENT_SYSTEM_PROMPT.format(question=example["question"]),
        "question": example["question"],
        "answer": str(example["answer"]),
        "program": example.get("program", ""),
        "context": context,
        "table": example.get("table", []),
        "pre_text": example.get("pre_text", ""),
        "post_text": example.get("post_text", ""),
        "gold_inds": example.get("gold_inds", {}),
    }


def main() -> None:
    dataset = load_from_disk(str(RAW_DATASET_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for example in dataset[split]:
                handle.write(json.dumps(process_example(example), ensure_ascii=False) + "\n")
                count += 1
        print(f"{split}: {count} examples -> {output_path}")

    print("Training data preparation complete.")


if __name__ == "__main__":
    main()


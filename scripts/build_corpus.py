"""Build a retrieval corpus from FinQA report text and table content."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = ROOT_DIR / "data" / "raw" / "finqa_hf"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"


def normalize_text(text: object) -> str:
    if isinstance(text, list):
        return " ".join(" ".join(str(item).split()) for item in text if str(item).strip())
    return " ".join(str(text).split())


def stringify_table(table: list[list[str]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def main() -> None:
    dataset = load_from_disk(str(RAW_DATASET_PATH))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    corpus: dict[str, dict] = {}
    passage_id = 0

    for split in ("train", "validation", "test"):
        for example in dataset[split]:
            report_id = example["id"].rsplit("-", 1)[0]

            candidates = [
                ("pre_text", normalize_text(example.get("pre_text", "")), 50),
                ("post_text", normalize_text(example.get("post_text", "")), 50),
                ("table", normalize_text(stringify_table(example.get("table", []))), 30),
            ]

            for source, content, min_len in candidates:
                if len(content) <= min_len:
                    continue
                dedupe_key = content[:200]
                if dedupe_key in corpus:
                    continue
                corpus[dedupe_key] = {
                    "id": f"p_{passage_id}",
                    "contents": content,
                    "report_id": report_id,
                    "source": source,
                }
                passage_id += 1

    output_file = CORPUS_DIR / "financial_passages.jsonl"
    with output_file.open("w", encoding="utf-8") as handle:
        for doc in corpus.values():
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    passage_map = {doc["id"]: doc["contents"] for doc in corpus.values()}
    with (CORPUS_DIR / "passage_map.json").open("w", encoding="utf-8") as handle:
        json.dump(passage_map, handle, ensure_ascii=False, indent=2)

    print(f"Corpus build complete: {len(corpus)} passages")
    print(f"  passages: {output_file}")
    print(f"  map: {CORPUS_DIR / 'passage_map.json'}")


if __name__ == "__main__":
    main()

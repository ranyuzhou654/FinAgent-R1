"""Generate SFT seed data using DeepSeek API for diverse, high-quality traces."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "train.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "sft_seed.jsonl"
QUESTION_TABLE_MAP_PATH = ROOT_DIR / "data" / "tables" / "question_table_map.json"

SYSTEM_PROMPT = (
    "You are a financial analysis agent with access to search, calculate, and SQL tools. "
    "Use <think>...</think> for reasoning, tool tags for actions, and "
    "<answer>...</answer> for the final answer."
)

GENERATION_PROMPT = """\
You are helping create training data for a financial agent. Given the information below, \
write a realistic multi-turn agent trace that uses tool tags to solve the question.

## Available tools
- <search>query</search> → searches financial report passages, returns text in <observation>...</observation>
- <calculate>expression</calculate> → runs a math expression, returns result in <observation>...</observation>
- <sql>SQL query</sql> → queries a SQLite table, returns rows in <observation>...</observation>

## Rules
1. Start with <think>...</think> to reason about what to do.
2. Call one or more tools as needed. Each tool call is followed by a simulated <observation>.
3. Use <think> between tool calls to reason about next steps.
4. End with <answer>EXACT_ANSWER</answer>.
5. The final answer MUST be exactly: {answer}
6. Be natural and varied — do NOT use the same phrasing every time.
7. Keep the trace concise (3-8 tool calls max).

## Input
Question: {question}
Gold answer: {answer}
Gold program: {program}
Context excerpt (first 400 chars): {context}
{table_info}

## Output
Write ONLY the assistant trace (no extra explanation). Start directly with <think>."""


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


def call_deepseek(client: OpenAI, prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=os.getenv("SFT_GEN_MODEL", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": "You generate training data for a financial agent. Output only the agent trace."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.9,
                top_p=0.95,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return None


def validate_trace(trace: str, answer: str) -> bool:
    """Check that the trace has basic structure and contains the answer."""
    has_think = "<think>" in trace
    has_tool = any(tag in trace for tag in ["<search>", "<calculate>", "<sql>"])
    has_answer = "<answer>" in trace and "</answer>" in trace
    return has_think and has_tool and has_answer


def generate_one(
    example: dict,
    table_name: str | None,
    client: OpenAI,
) -> dict | None:
    context = str(example.get("context", ""))[:400]
    table_info = f"SQL table name: {table_name}" if table_name else "No SQL table available."

    prompt = GENERATION_PROMPT.format(
        question=example["question"],
        answer=example["answer"],
        program=example.get("program", "N/A"),
        context=context,
        table_info=table_info,
    )

    trace = call_deepseek(client, prompt)
    if trace is None:
        return None

    # Basic validation
    if not validate_trace(trace, str(example["answer"])):
        # Retry once with stricter instruction
        trace = call_deepseek(client, prompt + "\n\nIMPORTANT: You MUST include <think>, at least one tool tag, and <answer>.")
        if trace is None or not validate_trace(trace, str(example["answer"])):
            return None

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
            {"role": "assistant", "content": trace},
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8, help="Parallel API calls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY env)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: DeepSeek)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Processed training data not found: {input_path}")

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY env or pass --api-key")

    base_url = args.base_url or os.getenv("SFT_GEN_BASE_URL", "https://api.deepseek.com")

    client = OpenAI(api_key=api_key, base_url=base_url)

    records = load_jsonl(input_path)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    selected = records[: args.max_samples]
    table_map = load_question_table_map()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0

    print(f"Generating {len(selected)} SFT traces with {args.workers} parallel workers...")

    with open(output_path, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_one, ex, table_map.get(ex["id"]), client): ex["id"]
                for ex in selected
            }
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  Exception for {eid}: {e}")
                    failed += 1

                total = success + failed
                if total % 50 == 0:
                    print(f"  Progress: {total}/{len(selected)} (success={success}, failed={failed})")

    print(f"\nDone! Wrote {success} SFT rows to {output_path} (failed: {failed})")


if __name__ == "__main__":
    main()

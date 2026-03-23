"""Offline evaluation for FinAgent-R1 chat models."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.tool_dispatcher import multi_turn_agent_rollout
from training.reward_functions import answers_match, extract_answer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILE = ROOT_DIR / "data" / "processed" / "test.jsonl"
TOOL_CALL_PATTERN = re.compile(r"<(search|calculate|sql)>(.*?)</\1>", re.DOTALL)


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_fn_factory(model, tokenizer):
    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return generate


def normalize_program(program: str) -> str:
    return " ".join(str(program).strip().lower().split())


def extract_tool_calls(text: str) -> list[tuple[str, str]]:
    return [(tool, content.strip()) for tool, content in TOOL_CALL_PATTERN.findall(text)]


def program_matches(predicted_trace: str, gold_program: str) -> bool:
    normalized_gold = normalize_program(gold_program)
    if not normalized_gold:
        return False
    predicted_calculations = [
        normalize_program(content)
        for tool, content in extract_tool_calls(predicted_trace)
        if tool == "calculate"
    ]
    if normalized_gold in predicted_calculations:
        return True
    if normalized_gold.startswith("table_"):
        return any(tool == "sql" for tool, _ in extract_tool_calls(predicted_trace))
    return False


def evaluate(model_path: str, test_file: str | Path = DEFAULT_TEST_FILE, max_samples: int | None = None) -> dict:
    model, tokenizer = load_model(model_path)
    generate = generate_fn_factory(model, tokenizer)

    test_items = []
    with Path(test_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            test_items.append(json.loads(line))
    if max_samples is not None:
        test_items = test_items[:max_samples]

    metrics = defaultdict(int)
    detailed = []

    for index, item in enumerate(test_items, start=1):
        rollout = multi_turn_agent_rollout(
            generate_fn=generate,
            prompt=item["prompt"],
            question_id=item.get("id"),
            max_turns=5,
        )
        predicted_answer = extract_answer(rollout["full_text"])
        is_correct = answers_match(predicted_answer, item["answer"])
        is_program_correct = program_matches(rollout["full_text"], item.get("program", ""))
        metrics["total"] += 1
        if is_correct:
            metrics["correct"] += 1
        if item.get("program"):
            metrics["program_total"] += 1
            if is_program_correct:
                metrics["program_correct"] += 1
        if rollout["tool_trace"]:
            metrics["used_tools"] += 1
            tool_types = {entry["tool"] for entry in rollout["tool_trace"]}
            if len(tool_types) >= 2:
                metrics["multi_tool"] += 1
            for tool_name in tool_types:
                metrics[f"tool_{tool_name}"] += 1
        detailed.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "gold": item["answer"],
                "pred": predicted_answer,
                "correct": is_correct,
                "program_correct": is_program_correct,
                "tools_used": [entry["tool"] for entry in rollout["tool_trace"]],
                "num_tool_calls": len(rollout["tool_trace"]),
            }
        )
        if index % 20 == 0:
            accuracy = metrics["correct"] / metrics["total"] * 100
            print(f"[{index}/{len(test_items)}] accuracy={accuracy:.1f}%")

    total = metrics["total"] or 1
    program_total = metrics["program_total"] or 1
    metrics["execution_accuracy"] = metrics["correct"] / total
    metrics["program_accuracy"] = metrics["program_correct"] / program_total
    metrics["tool_usage_rate"] = metrics["used_tools"] / total
    metrics["multi_tool_rate"] = metrics["multi_tool"] / total

    output_dir = ROOT_DIR / "eval"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"results_{Path(model_path).name}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"metrics": dict(metrics), "results": detailed}, handle, ensure_ascii=False, indent=2)

    print(f"Execution Accuracy: {metrics['execution_accuracy'] * 100:.1f}%")
    print(f"Program Accuracy: {metrics['program_accuracy'] * 100:.1f}%")
    print(f"Tool Usage Rate: {metrics['tool_usage_rate'] * 100:.1f}%")
    print(f"Multi-Tool Rate: {metrics['multi_tool_rate'] * 100:.1f}%")
    print(f"Saved evaluation results to {output_path}")
    return dict(metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()
    evaluate(args.model_path, test_file=args.test_file, max_samples=args.max_samples)

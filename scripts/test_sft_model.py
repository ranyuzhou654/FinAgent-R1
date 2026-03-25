"""Load the optional SFT adapter and inspect tool-tag generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sft_config.yaml"
DEFAULT_TEST_QUESTIONS = [
    "What was the percentage change in total revenue from 2018 to 2019?",
    "Calculate the gross margin ratio given revenue of $5.2 billion and COGS of $3.1 billion.",
    "Which SQL table should I inspect to answer a balance-sheet style question?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_sft_model(base_model_name: str, adapter_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_path),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter_path = Path(args.model_path or Path(config["output_dir"]) / "final")
    if not adapter_path.exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {adapter_path}")

    model, tokenizer = load_sft_model(config["model_name"], adapter_path)
    device = model.device

    system_prompt = (
        "You are a financial analysis agent with access to search, calculate, and SQL tools. "
        "Use <think>...</think> for reasoning, tool tags for actions, and "
        "<answer>...</answer> for the final answer."
    )

    for question in DEFAULT_TEST_QUESTIONS:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        inputs = {"input_ids": input_ids}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
            )
        response = tokenizer.decode(outputs[0][input_ids.shape[1] :], skip_special_tokens=True)
        print("=" * 80)
        print(f"Question: {question}")
        print(response[:800])
        print(
            {
                "think": "<think>" in response,
                "search": "<search>" in response,
                "calculate": "<calculate>" in response,
                "sql": "<sql>" in response,
                "answer": "<answer>" in response,
            }
        )


if __name__ == "__main__":
    main()

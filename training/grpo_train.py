"""Compatibility GRPO training entrypoint built on pure TRL."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.reward_functions import accuracy_reward, agent_behavior_reward, tool_efficiency_reward
from training.void_turn_filter import void_turn_penalty_reward


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "grpo_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_training_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "prompt": item["prompt"],
                    "answer": str(item["answer"]),
                    "question": item.get("question", ""),
                    "id": item.get("id"),
                }
            )
    return rows


def build_trainer(model, tokenizer, dataset: Dataset, config: dict):
    from trl import GRPOConfig, GRPOTrainer

    training_cfg = config["training"]
    trainer_args = GRPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=str(training_cfg["lr_scheduler_type"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        num_generations=int(training_cfg["num_generations"]),
        max_prompt_length=int(training_cfg["max_prompt_length"]),
        max_completion_length=int(training_cfg["max_completion_length"]),
        bf16=bool(training_cfg["bf16"]) and torch.cuda.is_available(),
        gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        save_total_limit=int(training_cfg["save_total_limit"]),
        report_to=[] if str(training_cfg["report_to"]).lower() == "none" else [str(training_cfg["report_to"])],
        run_name=str(training_cfg["run_name"]),
        max_grad_norm=float(training_cfg["max_grad_norm"]),
    )
    trainer_kwargs = {
        "model": model,
        "args": trainer_args,
        "train_dataset": dataset,
        "reward_funcs": [
            accuracy_reward,
            agent_behavior_reward,
            tool_efficiency_reward,
            void_turn_penalty_reward,
        ],
        "peft_config": LoraConfig(
            r=int(config["lora"]["r"]),
            lora_alpha=int(config["lora"]["alpha"]),
            target_modules=list(config["lora"]["target_modules"]),
            lora_dropout=float(config["lora"]["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    }
    signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    return GRPOTrainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = args.model_path or config["model_name"]
    train_data_path = Path(config["train_data_path"])
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    dataset = Dataset.from_list(load_training_rows(train_data_path))
    trainer = build_trainer(model, tokenizer, dataset, config)
    trainer.train()

    final_dir = Path(config["output_dir"]) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved TRL GRPO checkpoint to {final_dir}")


if __name__ == "__main__":
    main()

"""Optional SFT cold-start training for FinAgent-R1 tool formatting."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sft_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def render_messages(messages: list[dict], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    chunks = []
    for message in messages:
        role = message["role"].upper()
        chunks.append(f"{role}: {message['content']}")
    return "\n\n".join(chunks)


def tokenize_example(example: dict, tokenizer, max_seq_length: int) -> dict:
    text = render_messages(example["messages"], tokenizer)
    tokenized = tokenizer(text, truncation=True, max_length=max_seq_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = config["model_name"]
    output_dir = Path(config["output_dir"])
    seed_data_path = Path(config["seed_data_path"])
    max_seq_length = int(config.get("max_seq_length", 2048))

    if not seed_data_path.exists():
        raise FileNotFoundError(f"SFT seed data not found: {seed_data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        target_modules=list(config["lora"]["target_modules"]),
        lora_dropout=float(config["lora"]["dropout"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(seed_data_path), split="train")
    tokenized_dataset = dataset.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
        remove_columns=dataset.column_names,
        num_proc=4,
        load_from_cache_file=False,
    )

    training_cfg = config["training"]
    report_to = str(training_cfg["report_to"]).strip()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=str(training_cfg["lr_scheduler_type"]),
        warmup_steps=int(training_cfg["warmup_steps"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        save_total_limit=int(training_cfg["save_total_limit"]),
        bf16=bool(training_cfg["bf16"]) and torch.cuda.is_available(),
        gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
        report_to=[] if report_to.lower() == "none" else [item.strip() for item in report_to.split(",") if item.strip()],
        run_name=str(training_cfg["run_name"]),
        max_grad_norm=float(training_cfg["max_grad_norm"]),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),
    )

    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved SFT adapter to {final_dir}")


if __name__ == "__main__":
    main()

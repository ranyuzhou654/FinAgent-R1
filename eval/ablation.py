"""Run a small ablation table over a set of model checkpoints."""

from __future__ import annotations

from eval.evaluate import evaluate


EXPERIMENTS = [
    ("Qwen/Qwen2.5-3B", "Base model"),
    ("checkpoints/sft/final", "SFT cold-start"),
    ("checkpoints/grpo/final", "GRPO (TRL/Unsloth)"),
    ("checkpoints/grpo_verl", "GRPO (veRL mainline)"),
]


def main() -> None:
    for model_path, label in EXPERIMENTS:
        print("=" * 60)
        print(f"{label}: {model_path}")
        print("=" * 60)
        try:
            evaluate(model_path, max_samples=50)
        except Exception as exc:
            print(f"Failed: {exc}")


if __name__ == "__main__":
    main()

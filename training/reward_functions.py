"""Reward utilities shared by veRL training and offline evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_TAGS = ("think", "search", "calculate", "sql", "observation", "answer")


@dataclass
class RewardWeights:
    accuracy: float = 1.0
    format: float = 0.1
    tool_use: float = 0.15
    multi_tool: float = 0.1
    reasoning_after_observation: float = 0.05
    no_tool_penalty: float = -0.05
    overuse_penalty: float = -0.1
    invalid_action_penalty: float = -0.05
    max_tool_calls: int = 4
    tolerance: float = 0.01


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(value: Any) -> str:
    normalized = str(value).strip().lower()
    normalized = normalized.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return str(round(float(normalized), 6))
    except ValueError:
        return " ".join(normalized.split())


def answers_match(prediction: str, gold: str, tolerance: float = 0.01) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True

    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        if gold_num == 0:
            return abs(pred_num) < tolerance
        return abs(pred_num - gold_num) / abs(gold_num) < tolerance
    except ValueError:
        return gold_norm in pred_norm or pred_norm in gold_norm


def has_balanced_tags(text: str) -> bool:
    for tag in SUPPORTED_TAGS:
        if len(re.findall(rf"<{tag}>", text)) != len(re.findall(rf"</{tag}>", text)):
            return False
    return True


def count_tool_calls(text: str) -> dict[str, int]:
    return {
        "search": text.count("<search>"),
        "calculate": text.count("<calculate>"),
        "sql": text.count("<sql>"),
    }


def is_structured_response(text: str) -> bool:
    if not has_balanced_tags(text):
        return False
    answer_matches = list(re.finditer(r"<answer>.*?</answer>", text, re.DOTALL))
    if not answer_matches:
        return False
    final_answer = answer_matches[-1]
    trailing = text[final_answer.end() :].strip()
    return trailing == ""


def observation_followed_by_reasoning(text: str) -> bool:
    return re.search(r"</observation>\s*<think>", text, re.DOTALL) is not None


def has_invalid_retry(text: str) -> bool:
    lowered = text.lower()
    return "invalid" in lowered and "let me try again" in lowered


def compute_behavior_bonus(text: str, weights: RewardWeights) -> float:
    tool_counts = count_tool_calls(text)
    total_tool_calls = sum(tool_counts.values())
    tool_types_used = sum(count > 0 for count in tool_counts.values())

    score = 0.0
    if is_structured_response(text):
        score += weights.format
    if total_tool_calls > 0:
        score += weights.tool_use
    else:
        score += weights.no_tool_penalty
    if tool_types_used >= 2:
        score += weights.multi_tool
    if total_tool_calls > weights.max_tool_calls:
        score += weights.overuse_penalty
    if observation_followed_by_reasoning(text):
        score += weights.reasoning_after_observation
    if has_invalid_retry(text):
        score += weights.invalid_action_penalty
    return score


def compute_finagent_score(solution_str: str, ground_truth: dict[str, Any], weights: RewardWeights | None = None) -> float:
    weights = weights or RewardWeights()
    answer = extract_answer(solution_str)
    gold_answer = str(ground_truth.get("target", ""))
    accuracy = weights.accuracy if answers_match(answer, gold_answer, tolerance=weights.tolerance) else 0.0
    bonus = compute_behavior_bonus(solution_str, weights)
    return accuracy + bonus


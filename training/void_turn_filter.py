"""Helpers for detecting and penalizing void turns in Agent trajectories."""

from __future__ import annotations

import re
from typing import Any

from training.reward_functions import completion_to_text


ACTION_PATTERN = re.compile(r"<(search|calculate|sql|answer)>.*?</\1>", re.DOTALL)


def split_turns(trajectory: str) -> list[str]:
    segments = re.split(r"</observation>", trajectory)
    return [segment.strip() for segment in segments if segment.strip()]


def has_void_turn(trajectory: str) -> bool:
    for turn in split_turns(trajectory):
        has_action = ACTION_PATTERN.search(turn) is not None
        has_reasoning = "<think>" in turn and "</think>" in turn
        if not has_action and not has_reasoning:
            return True
    return False


def filter_void_trajectories(trajectories: list[str], rewards: list[float]) -> tuple[list[str], list[float]]:
    filtered_trajectories: list[str] = []
    filtered_rewards: list[float] = []
    for trajectory, reward in zip(trajectories, rewards):
        if has_void_turn(trajectory):
            continue
        filtered_trajectories.append(trajectory)
        filtered_rewards.append(reward)
    return filtered_trajectories, filtered_rewards


def void_turn_penalty_reward(completions: list[Any], penalty: float = -1.0, **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        trajectory = completion_to_text(completion)
        rewards.append(penalty if has_void_turn(trajectory) else 0.0)
    return rewards

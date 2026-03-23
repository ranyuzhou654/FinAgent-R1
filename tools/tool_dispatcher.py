"""Shared tool-dispatch helpers for evaluation, demo, and scripted rollout."""

from __future__ import annotations

import re
from typing import Callable

from tools.calculator_tool import execute_calculate
from tools.search_tool import execute_search
from tools.sql_tool import execute_sql


TOOL_REGISTRY = {
    "search": {
        "tag": "search",
        "fn": execute_search,
        "description": "Search the financial report knowledge base",
    },
    "calculate": {
        "tag": "calculate",
        "fn": execute_calculate,
        "description": "Execute financial calculations",
    },
    "sql": {
        "tag": "sql",
        "fn": execute_sql,
        "description": "Query structured financial data tables",
    },
}


def detect_tool_call(text: str) -> tuple[str | None, str | None, str | None]:
    earliest_match = None
    earliest_tool = None
    for tool_name, config in TOOL_REGISTRY.items():
        pattern = rf"<{config['tag']}>(.*?)</{config['tag']}>"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            continue
        if earliest_match is None or match.start() < earliest_match.start():
            earliest_match = match
            earliest_tool = tool_name
    if earliest_match is None or earliest_tool is None:
        return None, None, None
    return earliest_tool, earliest_match.group(1).strip(), text[earliest_match.end() :]


def execute_tool(tool_name: str, query: str, question_id: str | None = None) -> str:
    if tool_name not in TOOL_REGISTRY:
        return f"Tool execution error: unknown tool '{tool_name}'"
    fn = TOOL_REGISTRY[tool_name]["fn"]
    try:
        if tool_name == "sql":
            result = fn(query, question_id=question_id)
        else:
            result = fn(query)
        if len(result) > 1500:
            result = result[:1500] + "\n... [truncated]"
        return result
    except Exception as exc:
        return f"Tool execution error ({tool_name}): {exc}"


def format_observation(result: str) -> str:
    return f"\n<observation>\n{result}\n</observation>\n"


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def multi_turn_agent_rollout(
    generate_fn: Callable[[str], str],
    prompt: str,
    question_id: str | None = None,
    max_turns: int = 5,
    max_total_chars: int = 16000,
) -> dict:
    full_text = ""
    tool_trace = []
    token_mask_segments = []
    current_input = prompt

    for turn in range(max_turns):
        new_text = generate_fn(current_input)
        full_text += new_text
        token_mask_segments.append((new_text, False))

        tool_name, query, _ = detect_tool_call(new_text)
        if tool_name:
            result = execute_tool(tool_name, query, question_id=question_id)
            observation = format_observation(result)
            full_text += observation
            token_mask_segments.append((observation, True))
            tool_trace.append(
                {
                    "turn": turn + 1,
                    "tool": tool_name,
                    "query": query,
                    "result": result[:500],
                }
            )
            current_input = prompt + full_text
        else:
            if "<answer>" in new_text:
                break
            if len(full_text) > max_total_chars:
                break

    return {
        "full_text": full_text,
        "tool_trace": tool_trace,
        "token_mask_segments": token_mask_segments,
        "answer": extract_answer(full_text),
    }


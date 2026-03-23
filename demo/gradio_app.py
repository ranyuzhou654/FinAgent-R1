"""Gradio front-end for the FinAgent-R1 backend."""

from __future__ import annotations

import os

import gradio as gr
import requests


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/ask")


def ask_agent(question: str) -> str:
    response = requests.post(BACKEND_URL, json={"question": question}, timeout=120)
    response.raise_for_status()
    payload = response.json()

    lines = [f"**Answer:** {payload['answer']}", "", f"**Tool Calls:** {payload['num_tool_calls']}"]
    for trace in payload["tool_trace"]:
        lines.append("")
        lines.append(f"**Turn {trace['turn']}** `{trace['tool']}` -> `{trace['query']}`")
        lines.append("```")
        lines.append(trace["result"][:300])
        lines.append("```")
    lines.append("")
    lines.append("**Full Reasoning:**")
    lines.append("```")
    lines.append(payload["full_text"][:2000])
    lines.append("```")
    return "\n".join(lines)


demo = gr.Interface(
    fn=ask_agent,
    inputs=gr.Textbox(label="Financial Question", lines=3),
    outputs=gr.Markdown(label="Agent Response"),
    title="FinAgent-R1",
    description="Financial multi-tool agent demo.",
    examples=[
        ["What was the percentage change in total revenue from 2018 to 2019?"],
        ["Calculate the CAGR from 100 to 150 over 3 years."],
        ["Use SQL to inspect the current financial table schema."],
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

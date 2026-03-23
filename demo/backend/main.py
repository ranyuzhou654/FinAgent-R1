"""Minimal FastAPI backend for FinAgent-R1 demos."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.tool_dispatcher import detect_tool_call, execute_tool, extract_answer, format_observation


MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-3B")
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))

app = FastAPI(title="FinAgent-R1 Demo", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    question_id: str | None = None


def build_prompt(question: str) -> str:
    return f"""You are a financial analysis agent.

Use <think>...</think> for reasoning.
Use <search>...</search>, <calculate>...</calculate>, and <sql>...</sql> to call tools.
Return the final answer inside <answer>...</answer>.

Question: {question}"""


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_generate_fn():
    model, tokenizer = load_model()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return generate


def iter_agent_rollout_events(
    generate_fn,
    request: QueryRequest,
    max_turns: int = MAX_TURNS,
) -> Iterator[dict]:
    prompt = build_prompt(request.question)
    full_text = ""
    tool_trace = []
    current_input = prompt

    for turn in range(max_turns):
        new_text = generate_fn(current_input)
        full_text += new_text
        yield {
            "type": "assistant",
            "turn": turn + 1,
            "content": new_text,
        }

        tool_name, query, _ = detect_tool_call(new_text)
        if tool_name:
            result = execute_tool(tool_name, query, question_id=request.question_id)
            observation = format_observation(result)
            full_text += observation
            trace_entry = {
                "turn": turn + 1,
                "tool": tool_name,
                "query": query,
                "result": result[:500],
            }
            tool_trace.append(trace_entry)
            yield {
                "type": "tool",
                **trace_entry,
            }
            yield {
                "type": "observation",
                "turn": turn + 1,
                "content": observation.strip(),
            }
            current_input = prompt + full_text
            continue

        if "<answer>" in new_text:
            break
        if len(full_text) > 16000:
            break

    yield {
        "type": "done",
        "answer": extract_answer(full_text),
        "full_text": full_text,
        "tool_trace": tool_trace,
        "num_tool_calls": len(tool_trace),
    }


def run_agent_rollout(generate_fn, request: QueryRequest) -> dict:
    final_event = None
    for event in iter_agent_rollout_events(generate_fn, request):
        if event["type"] == "done":
            final_event = event
    assert final_event is not None
    return final_event


@app.get("/api/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/api/ask")
def ask(request: QueryRequest):
    try:
        generate = build_generate_fn()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    rollout = run_agent_rollout(generate, request)
    return {
        "answer": rollout["answer"],
        "full_text": rollout["full_text"],
        "tool_trace": rollout["tool_trace"],
        "num_tool_calls": rollout["num_tool_calls"],
    }


@app.post("/api/ask_stream")
async def ask_stream(request: QueryRequest):
    try:
        generate = build_generate_fn()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    async def event_generator():
        try:
            for event in iter_agent_rollout_events(generate, request):
                yield {
                    "event": event["type"],
                    "data": json.dumps(event, ensure_ascii=False),
                }
                await asyncio.sleep(0)
        except Exception as exc:  # pragma: no cover - defensive SSE fallback
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

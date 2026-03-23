"""Minimal FastAPI backend for FinAgent-R1 demos."""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.tool_dispatcher import multi_turn_agent_rollout


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


@app.get("/api/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/api/ask")
def ask(request: QueryRequest):
    try:
        generate = build_generate_fn()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    prompt = f"""You are a financial analysis agent.

Use <think>...</think> for reasoning.
Use <search>...</search>, <calculate>...</calculate>, and <sql>...</sql> to call tools.
Return the final answer inside <answer>...</answer>.

Question: {request.question}"""

    rollout = multi_turn_agent_rollout(
        generate_fn=generate,
        prompt=prompt,
        question_id=request.question_id,
        max_turns=MAX_TURNS,
    )
    return {
        "answer": rollout["answer"],
        "full_text": rollout["full_text"],
        "tool_trace": rollout["tool_trace"],
        "num_tool_calls": len(rollout["tool_trace"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


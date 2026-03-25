"""Search-R1-compatible retrieval service for FinAgent-R1."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from search_tool import get_search_tool, search_service_payload


app = FastAPI(title="FinAgent-R1 Retrieval Service", version="1.0")


class BatchQueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = True
    method: str = "hybrid"


class SingleQueryRequest(BaseModel):
    query: str
    topk: int | None = None
    method: str = "hybrid"


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(request: BatchQueryRequest) -> dict[str, Any]:
    return search_service_payload(
        queries=request.queries,
        method=request.method,
        topk=request.topk,
        return_scores=request.return_scores,
    )


@app.post("/search")
def search(request: SingleQueryRequest) -> dict[str, Any]:
    result = get_search_tool().batch_search(
        queries=[request.query],
        method=request.method,
        topk=request.topk,
        return_scores=True,
    )[0]
    return {
        "results": result,
        "query": request.query,
        "method": request.method,
        "topk": request.topk or get_search_tool().topk,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)


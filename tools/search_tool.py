"""Financial retrieval utilities for FinAgent-R1."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BM25_INDEX = ROOT_DIR / "data" / "indexes" / "bm25"
DEFAULT_DENSE_INDEX = ROOT_DIR / "data" / "indexes" / "dense_hnsw.index"
DEFAULT_CORPUS_PATH = ROOT_DIR / "data" / "corpus" / "financial_passages.jsonl"
DEFAULT_PASSAGE_MAP = ROOT_DIR / "data" / "corpus" / "passage_map.json"
DEFAULT_FAISS_ID_MAP = ROOT_DIR / "data" / "indexes" / "faiss_id_map.json"
DEFAULT_DENSE_MODEL = "BAAI/bge-m3"


@dataclass
class SearchResult:
    document: dict[str, Any]
    score: float


class FinancialSearchTool:
    """Hybrid retriever backed by BM25 and an optional dense index."""

    def __init__(
        self,
        bm25_index_path: Path | str = DEFAULT_BM25_INDEX,
        dense_index_path: Path | str = DEFAULT_DENSE_INDEX,
        corpus_path: Path | str = DEFAULT_CORPUS_PATH,
        passage_map_path: Path | str = DEFAULT_PASSAGE_MAP,
        faiss_id_map_path: Path | str = DEFAULT_FAISS_ID_MAP,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        topk: int = 3,
    ) -> None:
        self.bm25_index_path = Path(bm25_index_path)
        self.dense_index_path = Path(dense_index_path)
        self.corpus_path = Path(corpus_path)
        self.passage_map_path = Path(passage_map_path)
        self.faiss_id_map_path = Path(faiss_id_map_path)
        self.dense_model_name = dense_model_name
        self.topk = topk

        self._bm25 = None
        self._dense_model = None
        self._dense_index = None
        self._doc_map: dict[str, dict[str, Any]] | None = None
        self._faiss_ids: list[str] | None = None

    def _ensure_doc_map(self) -> None:
        if self._doc_map is not None:
            return

        doc_map: dict[str, dict[str, Any]] = {}
        if self.corpus_path.exists():
            with self.corpus_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    contents = item["contents"]
                    doc_map[item["id"]] = {
                        "id": item["id"],
                        "title": f"{item.get('report_id', 'report')}:{item.get('source', 'passage')}",
                        "text": contents,
                        "contents": contents,
                        "report_id": item.get("report_id"),
                        "source": item.get("source"),
                    }
        elif self.passage_map_path.exists():
            with self.passage_map_path.open("r", encoding="utf-8") as handle:
                passage_map = json.load(handle)
            for pid, contents in passage_map.items():
                doc_map[pid] = {
                    "id": pid,
                    "title": pid,
                    "text": contents,
                    "contents": contents,
                    "report_id": None,
                    "source": None,
                }
        self._doc_map = doc_map

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if not self.bm25_index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {self.bm25_index_path}")

        from pyserini.search.lucene import LuceneSearcher

        self._bm25 = LuceneSearcher(str(self.bm25_index_path))

    def _ensure_dense(self) -> None:
        if self._dense_index is not None and self._dense_model is not None:
            return
        if not self.dense_index_path.exists():
            raise FileNotFoundError(f"Dense index not found: {self.dense_index_path}")
        if not self.faiss_id_map_path.exists():
            raise FileNotFoundError(f"FAISS id map not found: {self.faiss_id_map_path}")

        import faiss
        from sentence_transformers import SentenceTransformer

        self._dense_index = faiss.read_index(str(self.dense_index_path))
        self._dense_model = SentenceTransformer(self.dense_model_name)
        with self.faiss_id_map_path.open("r", encoding="utf-8") as handle:
            self._faiss_ids = json.load(handle)

    def _doc_from_hit(self, docid: str, contents: str, source: str) -> dict[str, Any]:
        self._ensure_doc_map()
        doc = (self._doc_map or {}).get(docid)
        if doc is not None:
            return doc
        return {
            "id": docid,
            "title": f"{source}:{docid}",
            "text": contents,
            "contents": contents,
            "report_id": None,
            "source": source,
        }

    def search_bm25(self, query: str, topk: int | None = None) -> list[SearchResult]:
        self._ensure_bm25()
        topk = topk or self.topk
        hits = self._bm25.search(query, k=topk)
        results: list[SearchResult] = []

        for hit in hits:
            raw = hit.raw
            if raw is None:
                raw_doc = self._bm25.doc(hit.docid)
                raw = raw_doc.raw() if raw_doc else None
            if raw is None:
                continue
            parsed = json.loads(raw)
            doc = self._doc_from_hit(hit.docid, parsed.get("contents", ""), "bm25")
            results.append(SearchResult(document=doc, score=float(hit.score)))
        return results

    def search_dense(self, query: str, topk: int | None = None) -> list[SearchResult]:
        self._ensure_dense()
        self._ensure_doc_map()
        topk = topk or self.topk

        embeddings = self._dense_model.encode(
            [f"Represent this sentence for searching relevant passages: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self._dense_index.search(embeddings, topk)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._faiss_ids or []):
                continue
            doc_id = self._faiss_ids[idx]
            doc = (self._doc_map or {}).get(doc_id)
            if doc is None:
                continue
            results.append(SearchResult(document=doc, score=float(score)))
        return results

    def search_hybrid(self, query: str, topk: int | None = None) -> list[SearchResult]:
        topk = topk or self.topk
        result_sets: list[list[SearchResult]] = []

        try:
            result_sets.append(self.search_bm25(query, topk=topk))
        except Exception:
            pass

        try:
            result_sets.append(self.search_dense(query, topk=topk))
        except Exception:
            pass

        if not result_sets:
            return []

        rrf_scores: dict[str, float] = {}
        docs: dict[str, dict[str, Any]] = {}
        raw_scores: dict[str, float] = {}
        for results in result_sets:
            for rank, item in enumerate(results, start=1):
                doc_id = item.document["id"]
                docs[doc_id] = item.document
                raw_scores[doc_id] = max(raw_scores.get(doc_id, float("-inf")), item.score)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)

        sorted_ids = sorted(rrf_scores, key=lambda doc_id: (rrf_scores[doc_id], raw_scores[doc_id]), reverse=True)
        return [
            SearchResult(document=docs[doc_id], score=raw_scores[doc_id])
            for doc_id in sorted_ids[:topk]
        ]

    def search(
        self,
        query: str,
        method: str = "hybrid",
        topk: int | None = None,
    ) -> list[SearchResult]:
        if method == "bm25":
            return self.search_bm25(query, topk=topk)
        if method == "dense":
            return self.search_dense(query, topk=topk)
        return self.search_hybrid(query, topk=topk)

    def batch_search(
        self,
        queries: list[str],
        method: str = "hybrid",
        topk: int | None = None,
        return_scores: bool = True,
    ) -> list[list[dict[str, Any]]]:
        output: list[list[dict[str, Any]]] = []
        for query in queries:
            results = self.search(query, method=method, topk=topk)
            if return_scores:
                output.append(
                    [
                        {"document": item.document, "score": item.score}
                        for item in results
                    ]
                )
            else:
                output.append([item.document for item in results])
        return output

    def execute(self, query: str, method: str = "hybrid", topk: int | None = None) -> str:
        results = self.search(query, method=method, topk=topk)
        if not results:
            return "No relevant financial information found."
        parts = []
        for idx, item in enumerate(results, start=1):
            parts.append(
                f"[{idx}] {item.document['contents'][:700]}"
            )
        return "\n\n".join(parts)


_SEARCH_TOOL: FinancialSearchTool | None = None


def get_search_tool() -> FinancialSearchTool:
    global _SEARCH_TOOL
    if _SEARCH_TOOL is None:
        _SEARCH_TOOL = FinancialSearchTool()
    return _SEARCH_TOOL


def execute_search(query: str, method: str = "hybrid", topk: int | None = None) -> str:
    return get_search_tool().execute(query, method=method, topk=topk)


def search_service_payload(
    queries: list[str],
    method: str = "hybrid",
    topk: int | None = None,
    return_scores: bool = True,
) -> dict[str, Any]:
    start_time = time.time()
    results = get_search_tool().batch_search(
        queries=queries,
        method=method,
        topk=topk,
        return_scores=return_scores,
    )
    return {
        "result": results,
        "latency_ms": round((time.time() - start_time) * 1000, 1),
        "method": method,
        "topk": topk or get_search_tool().topk,
    }


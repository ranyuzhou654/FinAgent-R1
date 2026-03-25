"""Encode corpus passages with BGE and build a FAISS HNSW index."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT_DIR / "data" / "corpus" / "financial_passages.jsonl"
INDEX_DIR = ROOT_DIR / "data" / "indexes"
INDEX_PATH = INDEX_DIR / "dense_hnsw.index"
ID_MAP_PATH = INDEX_DIR / "faiss_id_map.json"
MODEL_NAME = "BAAI/bge-m3"


def load_passages() -> tuple[list[str], list[str]]:
    passages: list[str] = []
    passage_ids: list[str] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            passages.append(doc["contents"][:512])
            passage_ids.append(doc["id"])
    return passages, passage_ids


def main() -> None:
    print(f"Loading dense encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    passages, passage_ids = load_passages()
    print(f"Loaded {len(passages)} passages")

    embeddings = []
    batch_size = 128
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        batch_embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        if start == 0 or (start // batch_size) % 10 == 0:
            print(f"  encoded {min(start + batch_size, len(passages))}/{len(passages)}")

    matrix = np.vstack(embeddings).astype("float32")
    dimension = matrix.shape[1]
    print(f"Embedding dimension: {dimension}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(matrix)

    faiss.write_index(index, str(INDEX_PATH))
    with ID_MAP_PATH.open("w", encoding="utf-8") as handle:
        json.dump(passage_ids, handle, ensure_ascii=False, indent=2)

    print(f"Dense index build complete: {INDEX_PATH}")
    print(f"ID map written: {ID_MAP_PATH}")


if __name__ == "__main__":
    main()

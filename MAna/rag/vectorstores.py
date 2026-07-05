"""FAISS helpers and resilient vector-database batch writes."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .retrieval import make_stable_id


def normalize_embeddings(vectors: Any) -> np.ndarray:
    """Return float32 L2-normalized vectors without mutating input."""
    values = np.asarray(vectors, dtype=np.float32).copy()
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def build_faiss_cosine_index(embeddings: Any):
    """Build ``IndexFlatIP`` over normalized vectors for cosine search."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS support requires the RAG extra: "
            "pip install 'M_Ana_package[rag]'"
        ) from exc
    vectors = normalize_embeddings(embeddings)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_faiss(index: Any, query_embeddings: Any, top_k: int = 5):
    """Search a cosine-configured FAISS index."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    queries = normalize_embeddings(query_embeddings)
    if hasattr(index, "ntotal"):
        top_k = min(top_k, int(index.ntotal))
    return index.search(queries, top_k)


def _encode_batch(model: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    if hasattr(model, "backend") and hasattr(model, "fit_transform"):
        result = model.encode(texts) if model.is_fitted else model.fit_transform(texts)
        values = result.vectors
        return values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    try:
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
    except TypeError:
        return normalize_embeddings(model.encode(list(texts)))


def upsert_embedding_batches(
    index: Any,
    records: Sequence[Dict[str, Any]],
    embedding_model: Any,
    *,
    batch_size: int = 64,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    namespace: Optional[str] = None,
    text_key: str = "text",
    id_key: str = "id",
    metadata_key: str = "metadata",
) -> int:
    """Encode and upsert records in deterministic, retried batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    written = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [str(record[text_key]) for record in batch]
        vectors = normalize_embeddings(_encode_batch(embedding_model, texts, batch_size))
        payload = []
        for record, text, vector in zip(batch, texts, vectors):
            metadata = dict(record.get(metadata_key, {}) or {})
            metadata.setdefault("text", text)
            record_id = str(record.get(id_key) or make_stable_id(text, metadata, prefix="vec"))
            payload.append(
                {"id": record_id, "values": vector.tolist(), "metadata": metadata}
            )

        for attempt in range(max_retries + 1):
            try:
                kwargs = {"vectors": payload}
                if namespace is not None:
                    kwargs["namespace"] = namespace
                index.upsert(**kwargs)
                written += len(payload)
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"vector upsert failed after {max_retries + 1} attempts"
                    ) from exc
                time.sleep(retry_delay * (2**attempt))
    return written

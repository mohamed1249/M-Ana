"""Persistent semantic retrieval built on the M-Ana NLP embedding layer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from ..nlp.embeddings import TextEmbedder


def make_stable_id(
    text: object,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    prefix: str = "doc",
) -> str:
    """Create a deterministic identifier from text and metadata."""
    payload = json.dumps(
        {"text": str(text), "metadata": metadata or {}},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


@dataclass(frozen=True)
class SearchHit:
    """One semantic retrieval result."""

    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticIndex:
    """In-memory semantic index with persistence and metadata."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        backend: str = "sentence-transformers",
        normalize_vectors: bool = True,
        embedder: Optional[TextEmbedder] = None,
        **embedder_kwargs: Any,
    ) -> None:
        self.embedder = embedder or TextEmbedder(
            backend=backend,
            model_name=model_name,
            normalize_vectors=normalize_vectors,
            **embedder_kwargs,
        )
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.vectors: Any = None
        self.is_fitted = False

    def fit(
        self,
        texts: Sequence[object],
        metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        ids: Optional[Sequence[object]] = None,
    ) -> "SemanticIndex":
        """Encode a corpus and retain stable IDs and metadata."""
        if not texts:
            raise ValueError("texts cannot be empty")
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata must have the same length as texts")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids must have the same length as texts")

        self.texts = [str(text) for text in texts]
        self.metadata = [
            dict(metadata[index] or {}) if metadata is not None else {}
            for index in range(len(texts))
        ]
        if ids is None:
            candidates = [
                make_stable_id(text, self.metadata[index])
                for index, text in enumerate(self.texts)
            ]
            seen: Dict[str, int] = {}
            self.ids = []
            for candidate in candidates:
                seen[candidate] = seen.get(candidate, 0) + 1
                suffix = f"-{seen[candidate]}" if seen[candidate] > 1 else ""
                self.ids.append(f"{candidate}{suffix}")
        else:
            self.ids = [str(value) for value in ids]
            if len(set(self.ids)) != len(self.ids):
                raise ValueError("ids must be unique")

        result = self.embedder.fit_transform(self.texts)
        self.vectors = result.vectors
        self.is_fitted = True
        return self

    def search(self, query: object, top_k: int = 5, min_score: Optional[float] = None) -> List[SearchHit]:
        """Return query-to-corpus cosine matches without an all-pairs matrix."""
        if not self.is_fitted:
            raise RuntimeError("fit or load must be called before search")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self.embedder.encode([query]).vectors
        scores = cosine_similarity(query_vector, self.vectors).ravel()
        order = np.argsort(scores)[::-1]
        hits: List[SearchHit] = []
        for index in order:
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            hits.append(
                SearchHit(
                    id=self.ids[index],
                    text=self.texts[index],
                    score=score,
                    metadata=dict(self.metadata[index]),
                    index=int(index),
                )
            )
            if len(hits) >= min(top_k, len(self.texts)):
                break
        return hits

    def save(self, path: str) -> Path:
        """Save vectors, corpus metadata, and fitted TF-IDF state."""
        if not self.is_fitted:
            raise RuntimeError("fit must be called before save")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        config = {
            "backend": self.embedder.backend,
            "model_name": self.embedder.model_name,
            "normalize_vectors": self.embedder.normalize_vectors,
            "texts": self.texts,
            "metadata": self.metadata,
            "ids": self.ids,
        }
        (destination / "index.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if sparse.issparse(self.vectors):
            sparse.save_npz(destination / "vectors.npz", self.vectors)
        else:
            np.save(destination / "vectors.npy", np.asarray(self.vectors))

        if self.embedder.backend == "tfidf":
            joblib.dump(self.embedder, destination / "embedder.joblib")
        elif self.embedder.cleaner is not None:
            joblib.dump(self.embedder.cleaner, destination / "cleaner.joblib")
        return destination

    @classmethod
    def load(cls, path: str) -> "SemanticIndex":
        """Load an index saved by :meth:`save`."""
        source = Path(path)
        config = json.loads((source / "index.json").read_text(encoding="utf-8"))
        if config["backend"] == "tfidf":
            embedder = joblib.load(source / "embedder.joblib")
        else:
            cleaner_path = source / "cleaner.joblib"
            cleaner = joblib.load(cleaner_path) if cleaner_path.exists() else None
            embedder = TextEmbedder(
                backend=config["backend"],
                model_name=config["model_name"],
                cleaner=cleaner,
                normalize_vectors=config["normalize_vectors"],
            )
            embedder.is_fitted = True

        instance = cls(embedder=embedder)
        instance.texts = config["texts"]
        instance.metadata = config["metadata"]
        instance.ids = config["ids"]
        vector_npz = source / "vectors.npz"
        instance.vectors = (
            sparse.load_npz(vector_npz)
            if vector_npz.exists()
            else np.load(source / "vectors.npy")
        )
        instance.is_fitted = True
        return instance

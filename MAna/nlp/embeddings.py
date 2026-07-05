"""Text embedding interfaces and semantic-search helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .preprocessing import TextCleaner


@dataclass
class EmbeddingResult:
    """Dense or sparse text embeddings."""

    vectors: Any
    texts: List[str]
    backend: str
    model_name: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.vectors.shape

    def similarity_matrix(self) -> np.ndarray:
        return cosine_similarity(self.vectors)

    def search(self, query_vector: Any, top_k: int = 5) -> pd.DataFrame:
        """Rank embedded texts against one query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        scores = cosine_similarity(query_vector, self.vectors).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return pd.DataFrame(
            {
                "index": order,
                "text": [self.texts[index] for index in order],
                "score": scores[order],
            }
        )


class TextEmbedder:
    """Create TF-IDF or sentence-transformer embeddings."""

    def __init__(
        self,
        backend: str = "tfidf",
        *,
        model_name: str = "all-MiniLM-L6-v2",
        cleaner: Optional[TextCleaner] = None,
        normalize_vectors: bool = True,
        **kwargs: Any,
    ) -> None:
        backend = backend.lower()
        if backend not in {"tfidf", "sentence-transformers"}:
            raise ValueError("backend must be 'tfidf' or 'sentence-transformers'")
        self.backend = backend
        self.model_name = model_name if backend != "tfidf" else "tfidf"
        self.cleaner = cleaner
        self.normalize_vectors = normalize_vectors
        self.kwargs = kwargs
        self.model: Any = None
        self.is_fitted = False

    def _prepare(self, texts: Sequence[object]) -> List[str]:
        values = ["" if text is None else str(text) for text in texts]
        return self.cleaner.transform(values) if self.cleaner is not None else values

    def _load_sentence_model(self):
        if self.model is None:
            os.environ.setdefault("USE_TF", "0")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Sentence-transformer embeddings require the NLP extra: "
                    "pip install 'M_Ana_package[nlp]'"
                ) from exc
            self.model = SentenceTransformer(self.model_name, **self.kwargs)
        return self.model

    def fit(self, texts: Sequence[object]) -> "TextEmbedder":
        prepared = self._prepare(texts)
        if self.backend == "tfidf":
            self.model = TfidfVectorizer(**self.kwargs)
            self.model.fit(prepared)
        else:
            self._load_sentence_model()
        self.is_fitted = True
        return self

    def encode(self, texts: Sequence[object]) -> EmbeddingResult:
        """Encode text after fitting the embedder."""
        if not self.is_fitted:
            raise RuntimeError("fit or fit_transform must be called before encode")
        prepared = self._prepare(texts)
        if self.backend == "tfidf":
            vectors = self.model.transform(prepared)
            if self.normalize_vectors:
                vectors = normalize(vectors)
        else:
            model = self._load_sentence_model()
            vectors = model.encode(
                prepared,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_vectors,
            )
        return EmbeddingResult(
            vectors=vectors,
            texts=prepared,
            backend=self.backend,
            model_name=self.model_name,
        )

    def fit_transform(self, texts: Sequence[object]) -> EmbeddingResult:
        prepared = self._prepare(texts)
        if self.backend == "tfidf":
            self.model = TfidfVectorizer(**self.kwargs)
            vectors = self.model.fit_transform(prepared)
            if self.normalize_vectors:
                vectors = normalize(vectors)
        else:
            model = self._load_sentence_model()
            vectors = model.encode(
                prepared,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_vectors,
            )
        self.is_fitted = True
        return EmbeddingResult(
            vectors=vectors,
            texts=prepared,
            backend=self.backend,
            model_name=self.model_name,
        )


def semantic_search(
    query: object,
    documents: Sequence[object],
    *,
    embedder: Optional[TextEmbedder] = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Embed and rank documents for a text query."""
    if not documents:
        return pd.DataFrame(columns=["index", "text", "score"])

    engine = embedder or TextEmbedder()
    if engine.backend == "tfidf" and not engine.is_fitted:
        combined = [query, *documents]
        result = engine.fit_transform(combined)
        query_vector = result.vectors[0]
        corpus = EmbeddingResult(
            vectors=result.vectors[1:],
            texts=result.texts[1:],
            backend=result.backend,
            model_name=result.model_name,
        )
    else:
        if not engine.is_fitted:
            engine.fit(documents)
        corpus = engine.encode(documents)
        query_vector = engine.encode([query]).vectors
    return corpus.search(query_vector, top_k=min(top_k, len(documents)))


def segment_text(text: object, n_segments: int = 4) -> List[str]:
    """Split text into a fixed number of contiguous word segments."""
    if n_segments <= 0:
        raise ValueError("n_segments must be greater than zero")
    words = str(text or "").split()
    base, remainder = divmod(len(words), n_segments)
    segments: List[str] = []
    start = 0
    for index in range(n_segments):
        size = base + (1 if index < remainder else 0)
        segments.append(" ".join(words[start : start + size]))
        start += size
    return segments


def encode_segmented_texts(
    texts: Sequence[object],
    n_segments: int = 4,
    *,
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    aggregation: str = "concatenate",
    normalize_vectors: bool = True,
    encoder: Optional[Any] = None,
    **model_kwargs: Any,
) -> np.ndarray:
    """Encode contiguous sections of each text and combine their vectors.

    ``concatenate`` preserves segment position, while ``mean`` produces the
    original embedding width. Supplying ``encoder`` makes the function usable
    with any object that implements ``encode``.
    """
    if aggregation not in {"concatenate", "mean"}:
        raise ValueError("aggregation must be 'concatenate' or 'mean'")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not texts:
        return np.empty((0, 0), dtype=float)

    segments = [segment_text(text, n_segments) for text in texts]
    flat_segments = [segment for document in segments for segment in document]
    nonempty = np.asarray([bool(segment.strip()) for segment in flat_segments])

    if encoder is None:
        os.environ.setdefault("USE_TF", "0")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Segmented embeddings require the NLP extra: "
                "pip install 'M_Ana_package[nlp]'"
            ) from exc
        encoder = SentenceTransformer(model_name, **model_kwargs)

    vectors = np.asarray(
        encoder.encode(
            flat_segments,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ),
        dtype=float,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(flat_segments):
        raise ValueError("encoder returned an unexpected embedding shape")
    vectors[~nonempty] = 0.0
    segmented = vectors.reshape(len(texts), n_segments, vectors.shape[1])

    if aggregation == "concatenate":
        combined = segmented.reshape(len(texts), -1)
    else:
        counts = nonempty.reshape(len(texts), n_segments).sum(axis=1, keepdims=True)
        counts = np.maximum(counts, 1)
        combined = segmented.sum(axis=1) / counts

    if normalize_vectors:
        combined = normalize(combined)
    return np.asarray(combined)

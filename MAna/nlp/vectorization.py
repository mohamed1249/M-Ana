"""Classical text vectorization and similarity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import TextCleaner


@dataclass
class VectorizationResult:
    """Vectorized documents and their feature vocabulary."""

    matrix: Any
    feature_names: np.ndarray
    documents: List[str]
    method: str
    vectorizer: Any

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def to_dataframe(self, dense: bool = False) -> pd.DataFrame:
        """Convert vectors to a pandas DataFrame."""
        columns = self.feature_names
        if dense:
            values = self.matrix.toarray() if hasattr(self.matrix, "toarray") else self.matrix
            return pd.DataFrame(values, columns=columns)
        if hasattr(self.matrix, "tocsr"):
            return pd.DataFrame.sparse.from_spmatrix(self.matrix, columns=columns)
        return pd.DataFrame(self.matrix, columns=columns)

    def top_terms(self, document_index: int, n: int = 10) -> List[tuple[str, float]]:
        """Return the highest-weight terms for one document."""
        if n <= 0:
            raise ValueError("n must be greater than zero")
        if not 0 <= document_index < len(self.documents):
            raise IndexError("document_index is out of range")

        row = self.matrix[document_index]
        values = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
        nonzero = np.flatnonzero(values)
        ranked = nonzero[np.argsort(values[nonzero])[::-1]][:n]
        return [(str(self.feature_names[index]), float(values[index])) for index in ranked]

    def similarity_matrix(self) -> np.ndarray:
        """Calculate pairwise cosine similarity between documents."""
        return cosine_similarity(self.matrix)


class TextVectorizer:
    """Unified wrapper for TF-IDF, count, and hashing vectorizers."""

    def __init__(
        self,
        method: str = "tfidf",
        *,
        cleaner: Optional[TextCleaner] = None,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Optional[Union[str, Sequence[str]]] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        **kwargs: Any,
    ) -> None:
        self.method = method.lower()
        self.cleaner = cleaner
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.kwargs = kwargs
        self.vectorizer = self._create_vectorizer()
        self.is_fitted = False

    def _create_vectorizer(self):
        kwargs = dict(self.kwargs)
        n_features = kwargs.pop("n_features", self.max_features or 2**18)
        alternate_sign = False
        if self.method == "hashing":
            alternate_sign = kwargs.pop("alternate_sign", False)
        common: Dict[str, Any] = {
            "ngram_range": self.ngram_range,
            "stop_words": self.stop_words,
            **kwargs,
        }
        if self.method == "tfidf":
            return TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                **common,
            )
        if self.method == "count":
            return CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                **common,
            )
        if self.method == "hashing":
            return HashingVectorizer(
                n_features=n_features,
                alternate_sign=alternate_sign,
                **common,
            )
        raise ValueError("method must be 'tfidf', 'count', or 'hashing'")

    def _prepare(self, documents: Sequence[object]) -> List[str]:
        values = ["" if value is None else str(value) for value in documents]
        return self.cleaner.transform(values) if self.cleaner is not None else values

    def _feature_names(self) -> np.ndarray:
        if hasattr(self.vectorizer, "get_feature_names_out"):
            return self.vectorizer.get_feature_names_out()
        return np.asarray(
            [f"feature_{index}" for index in range(self.vectorizer.n_features)],
            dtype=object,
        )

    def fit(self, documents: Sequence[object]) -> "TextVectorizer":
        """Fit the vocabulary."""
        prepared = self._prepare(documents)
        if self.method != "hashing":
            self.vectorizer.fit(prepared)
        self.is_fitted = True
        return self

    def fit_transform(self, documents: Sequence[object]) -> VectorizationResult:
        """Fit and vectorize documents."""
        prepared = self._prepare(documents)
        matrix = (
            self.vectorizer.transform(prepared)
            if self.method == "hashing"
            else self.vectorizer.fit_transform(prepared)
        )
        self.is_fitted = True
        return VectorizationResult(
            matrix=matrix,
            feature_names=self._feature_names(),
            documents=prepared,
            method=self.method,
            vectorizer=self.vectorizer,
        )

    def transform(self, documents: Sequence[object]) -> VectorizationResult:
        """Vectorize new documents using the fitted vocabulary."""
        if not self.is_fitted:
            raise RuntimeError("fit or fit_transform must be called before transform")
        prepared = self._prepare(documents)
        matrix = self.vectorizer.transform(prepared)
        return VectorizationResult(
            matrix=matrix,
            feature_names=self._feature_names(),
            documents=prepared,
            method=self.method,
            vectorizer=self.vectorizer,
        )

    def get_feature_names_out(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("the vectorizer has not been fitted")
        return self._feature_names()


def vectorize_texts(
    documents: Sequence[object],
    method: str = "tfidf",
    **kwargs: Any,
) -> VectorizationResult:
    """Convenience function for fitting and vectorizing a corpus."""
    return TextVectorizer(method=method, **kwargs).fit_transform(documents)


def cosine_similarity_matrix(vectors: Any) -> np.ndarray:
    """Calculate pairwise cosine similarity for a vector matrix."""
    return cosine_similarity(vectors)

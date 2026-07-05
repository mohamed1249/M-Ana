"""Classical topic modeling with NMF and Latent Dirichlet Allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF

from .preprocessing import TextCleaner
from .vectorization import TextVectorizer, VectorizationResult


@dataclass(frozen=True)
class Topic:
    """Human-readable topic terms and weights."""

    id: int
    terms: List[str]
    weights: List[float]

    @property
    def label(self) -> str:
        return " / ".join(self.terms[:3])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "terms": list(self.terms),
            "weights": list(self.weights),
        }


@dataclass
class TopicModelResult:
    """Topic model output for a corpus."""

    document_topics: np.ndarray
    topics: List[Topic]
    documents: List[str]
    method: str
    model: Any
    vectorization: VectorizationResult

    def assignments(self) -> pd.DataFrame:
        """Return each document's dominant topic and confidence."""
        topic_ids = self.document_topics.argmax(axis=1)
        confidence = self.document_topics.max(axis=1)
        labels = {topic.id: topic.label for topic in self.topics}
        return pd.DataFrame(
            {
                "document": self.documents,
                "topic": topic_ids,
                "topic_label": [labels[index] for index in topic_ids],
                "confidence": confidence,
            }
        )

    def topics_frame(self) -> pd.DataFrame:
        return pd.DataFrame([topic.to_dict() for topic in self.topics])


class TopicModeler:
    """Fit NMF or LDA topics with sensible vectorizer defaults."""

    def __init__(
        self,
        n_topics: int = 5,
        *,
        method: str = "nmf",
        n_top_words: int = 10,
        max_features: Optional[int] = 5000,
        cleaner: Optional[TextCleaner] = None,
        random_state: int = 42,
        vectorizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if n_topics <= 0:
            raise ValueError("n_topics must be greater than zero")
        if n_top_words <= 0:
            raise ValueError("n_top_words must be greater than zero")
        if method.lower() not in {"nmf", "lda"}:
            raise ValueError("method must be 'nmf' or 'lda'")

        self.n_topics = n_topics
        self.method = method.lower()
        self.n_top_words = n_top_words
        self.max_features = max_features
        self.cleaner = cleaner or TextCleaner(
            stop_words="english", min_token_length=2
        )
        self.random_state = random_state
        self.vectorizer_kwargs = dict(vectorizer_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.vectorizer: Optional[TextVectorizer] = None
        self.model: Any = None
        self.result_: Optional[TopicModelResult] = None

    def _build(self) -> None:
        vectorizer_method = "tfidf" if self.method == "nmf" else "count"
        self.vectorizer = TextVectorizer(
            method=vectorizer_method,
            cleaner=self.cleaner,
            max_features=self.max_features,
            **self.vectorizer_kwargs,
        )
        if self.method == "nmf":
            self.model = NMF(
                n_components=self.n_topics,
                init="nndsvda",
                random_state=self.random_state,
                max_iter=400,
                **self.model_kwargs,
            )
        else:
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                learning_method="batch",
                **self.model_kwargs,
            )

    def _extract_topics(self, feature_names: np.ndarray) -> List[Topic]:
        topics: List[Topic] = []
        for topic_id, component in enumerate(self.model.components_):
            indices = component.argsort()[::-1][: self.n_top_words]
            topics.append(
                Topic(
                    id=topic_id,
                    terms=[str(feature_names[index]) for index in indices],
                    weights=[float(component[index]) for index in indices],
                )
            )
        return topics

    def fit_transform(self, documents: Sequence[object]) -> TopicModelResult:
        """Fit a topic model and return per-document topic weights."""
        self._build()
        assert self.vectorizer is not None
        vectorization = self.vectorizer.fit_transform(documents)
        n_documents, n_features = vectorization.shape
        if self.n_topics > min(n_documents, n_features):
            raise ValueError(
                "n_topics cannot exceed the smaller of document and feature counts "
                f"({min(n_documents, n_features)})"
            )

        document_topics = self.model.fit_transform(vectorization.matrix)
        topics = self._extract_topics(vectorization.feature_names)
        self.result_ = TopicModelResult(
            document_topics=document_topics,
            topics=topics,
            documents=vectorization.documents,
            method=self.method,
            model=self.model,
            vectorization=vectorization,
        )
        return self.result_

    def fit(self, documents: Sequence[object]) -> "TopicModeler":
        self.fit_transform(documents)
        return self

    def transform(self, documents: Sequence[object]) -> np.ndarray:
        """Infer topic weights for new documents."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("fit must be called before transform")
        vectors = self.vectorizer.transform(documents)
        return self.model.transform(vectors.matrix)


def model_topics(
    documents: Sequence[object],
    n_topics: int = 5,
    method: str = "nmf",
    **kwargs: Any,
) -> TopicModelResult:
    """Convenience function for fitting a topic model."""
    return TopicModeler(n_topics=n_topics, method=method, **kwargs).fit_transform(
        documents
    )

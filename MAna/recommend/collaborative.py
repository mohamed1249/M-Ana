"""Sparse item-neighborhood collaborative filtering."""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .base import BaseRecommender, recommendation_frame, validate_top_n
from .data import InteractionMatrix, build_interaction_matrix


class ItemKNNRecommender(BaseRecommender):
    """Recommend items from cosine similarity in user-interaction space."""

    def __init__(
        self,
        *,
        user_column: str = "user_id",
        item_column: str = "item_id",
        score_column: Optional[str] = None,
        aggregation: str = "binary",
        n_neighbors: int = 50,
        minimum_similarity: float = 0.0,
    ) -> None:
        super().__init__()
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than zero")
        if not np.isfinite(minimum_similarity):
            raise ValueError("minimum_similarity must be finite")
        self.user_column = user_column
        self.item_column = item_column
        self.score_column = score_column
        self.aggregation = aggregation
        self.n_neighbors = int(n_neighbors)
        self.minimum_similarity = float(minimum_similarity)
        self.interactions: Optional[InteractionMatrix] = None
        self.item_vectors: Any = None

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        interaction_matrix = build_interaction_matrix(
            interactions,
            user_column=self.user_column,
            item_column=self.item_column,
            score_column=self.score_column,
            aggregation=self.aggregation,
        )
        self.interactions = interaction_matrix
        self.item_vectors = normalize(
            interaction_matrix.matrix.T.tocsr(),
            axis=1,
        ).tocsr()
        self._mark_fitted()
        return self

    def _item_similarities(self, item_index: int) -> np.ndarray:
        if self.item_vectors is None:
            raise RuntimeError("fit must be called before computing similarities")
        similarities = (self.item_vectors @ self.item_vectors[item_index].T).toarray().ravel()
        similarities[item_index] = 0.0
        return similarities

    def _neighbor_indices(self, similarities: np.ndarray) -> np.ndarray:
        eligible = np.flatnonzero(similarities > self.minimum_similarity)
        if eligible.size <= self.n_neighbors:
            return eligible
        local = np.argpartition(-similarities[eligible], self.n_neighbors - 1)[
            : self.n_neighbors
        ]
        return eligible[local]

    def recommend(
        self,
        user_id: Hashable,
        *,
        top_n: int = 10,
        exclude_items: Iterable[Hashable] = (),
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Aggregate each history item's nearest-neighbor evidence."""
        self._require_fitted()
        top_n = validate_top_n(top_n)
        interactions = self.interactions
        if interactions is None:
            raise RuntimeError("fitted interaction data is unavailable")
        user_index = interactions.user_index(user_id)
        history = interactions.matrix.getrow(user_index)
        if history.nnz == 0:
            return recommendation_frame([], [], top_n=top_n, source="item_knn")

        scores: np.ndarray = np.zeros(len(interactions.item_ids), dtype=float)
        for item_index, interaction_weight in zip(history.indices, history.data):
            similarities = self._item_similarities(int(item_index))
            neighbors = self._neighbor_indices(similarities)
            scores[neighbors] += float(interaction_weight) * similarities[neighbors]

        blocked = set(exclude_items)
        if exclude_seen:
            blocked.update(
                interactions.item_ids[index] for index in history.indices
            )
        candidates = [
            (item_id, float(scores[index]))
            for index, item_id in enumerate(interactions.item_ids)
            if item_id not in blocked and scores[index] > self.minimum_similarity
        ]
        return recommendation_frame(
            [item for item, _ in candidates],
            [score for _, score in candidates],
            top_n=top_n,
            source="item_knn",
        )

    def similar_items(
        self,
        item_id: Hashable,
        *,
        top_n: int = 10,
        exclude_items: Iterable[Hashable] = (),
    ) -> pd.DataFrame:
        """Return interaction-nearest neighbors for one catalog item."""
        self._require_fitted()
        top_n = validate_top_n(top_n)
        interactions = self.interactions
        if interactions is None:
            raise RuntimeError("fitted interaction data is unavailable")
        item_index = interactions.item_index(item_id)
        similarities = self._item_similarities(item_index)
        blocked = set(exclude_items) | {item_id}
        candidates = [
            (candidate, float(similarities[index]))
            for index, candidate in enumerate(interactions.item_ids)
            if candidate not in blocked and similarities[index] > self.minimum_similarity
        ]
        return recommendation_frame(
            [item for item, _ in candidates],
            [score for _, score in candidates],
            top_n=top_n,
            source="item_knn",
        )


__all__ = ["ItemKNNRecommender"]

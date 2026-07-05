"""Content feature preparation and profile-based recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

from .base import BaseRecommender, recommendation_frame, validate_top_n


def combine_text_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    separator: str = " ",
    weights: Optional[Mapping[str, int]] = None,
) -> pd.Series:
    """Combine text fields, optionally repeating fields to increase importance."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if not columns:
        raise ValueError("columns cannot be empty")
    field_weights = dict(weights or {})
    if any(int(value) < 0 for value in field_weights.values()):
        raise ValueError("text feature weights cannot be negative")

    def combine(row: pd.Series) -> str:
        parts = []
        for column in columns:
            value = "" if pd.isna(row[column]) else str(row[column]).strip()
            parts.extend([value] * int(field_weights.get(column, 1)))
        return separator.join(part for part in parts if part)

    return frame[list(columns)].apply(combine, axis=1)


def concatenate_feature_embeddings(
    *embedding_matrices: Any,
    weights: Optional[Sequence[float]] = None,
    normalize_rows: bool = True,
) -> np.ndarray:
    """Weight and concatenate aligned dense embedding matrices."""
    if not embedding_matrices:
        raise ValueError("At least one embedding matrix is required")
    matrices = [np.asarray(matrix, dtype=np.float32) for matrix in embedding_matrices]
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("All embedding matrices must be two-dimensional")
    if len({matrix.shape[0] for matrix in matrices}) != 1:
        raise ValueError("All embedding matrices must have the same row count")
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("embedding matrices must contain finite values")

    feature_weights = [1.0] * len(matrices) if weights is None else list(weights)
    if len(feature_weights) != len(matrices):
        raise ValueError("weights must match the number of embedding matrices")
    if any(not np.isfinite(weight) or weight < 0 for weight in feature_weights):
        raise ValueError("weights must be finite and non-negative")
    if not any(feature_weights):
        raise ValueError("At least one feature weight must be positive")

    combined = np.concatenate(
        [matrix * float(weight) for matrix, weight in zip(matrices, feature_weights)],
        axis=1,
    )
    return normalize(combined).astype(np.float32) if normalize_rows else combined


@dataclass
class NumericFeatureTransformer:
    columns: Sequence[str]
    medians: pd.Series
    scaler: Optional[StandardScaler]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        numeric = frame[list(self.columns)].apply(pd.to_numeric, errors="coerce")
        values = numeric.fillna(self.medians).fillna(0.0).to_numpy(dtype=np.float32)
        if self.scaler is not None:
            values = self.scaler.transform(values).astype(np.float32)
        return values


def prepare_numeric_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    standardize: bool = True,
) -> Tuple[np.ndarray, NumericFeatureTransformer]:
    """Median-impute numeric metadata and return a reusable transformer."""
    if not columns:
        raise ValueError("columns cannot be empty")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    values = numeric.fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
    scaler = StandardScaler() if standardize else None
    if scaler is not None:
        values = scaler.fit_transform(values).astype(np.float32)
    return values, NumericFeatureTransformer(list(columns), medians, scaler)


class ContentRecommender(BaseRecommender):
    """Recommend similar items or items matching a user's content profile."""

    def __init__(self) -> None:
        super().__init__()
        self.item_ids: List[Hashable] = []
        self.item_to_index: Dict[Hashable, int] = {}
        self.features: Any = None
        self.metadata: Dict[Hashable, Dict[str, Any]] = {}
        self.user_histories: Dict[Hashable, Sequence[Hashable]] = {}

    def fit(
        self,
        item_ids: Sequence[Hashable],
        features: Any,
        *,
        metadata: Optional[Any] = None,
        user_histories: Optional[Mapping[Hashable, Sequence[Hashable]]] = None,
    ) -> "ContentRecommender":
        ids = list(item_ids)
        if not ids:
            raise ValueError("item_ids cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("item_ids must be unique")
        matrix = features.tocsr().astype(np.float32) if sparse.issparse(features) else np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("feature rows must align with item_ids")
        if sparse.issparse(matrix):
            if not np.isfinite(matrix.data).all():
                raise ValueError("features must contain finite values")
        elif not np.isfinite(matrix).all():
            raise ValueError("features must contain finite values")

        self.item_ids = ids
        self.item_to_index = {item_id: index for index, item_id in enumerate(ids)}
        self.features = normalize(matrix, axis=1)
        if metadata is None:
            self.metadata = {item_id: {} for item_id in ids}
        elif isinstance(metadata, pd.DataFrame):
            if len(metadata) != len(ids):
                raise ValueError("metadata rows must align with item_ids")
            self.metadata = {
                item_id: dict(metadata.iloc[index]) for index, item_id in enumerate(ids)
            }
        else:
            metadata_values = list(metadata)
            if len(metadata_values) != len(ids):
                raise ValueError("metadata rows must align with item_ids")
            self.metadata = {
                item_id: dict(metadata_values[index] or {})
                for index, item_id in enumerate(ids)
            }
        self.user_histories = {
            user_id: list(items) for user_id, items in (user_histories or {}).items()
        }
        self._mark_fitted()
        return self

    def _known_items(
        self,
        items: Sequence[Hashable],
        weights: Optional[Sequence[float]] = None,
    ) -> Tuple[Sequence[int], np.ndarray]:
        if weights is not None and len(weights) != len(items):
            raise ValueError("weights must align with items")
        indices = []
        selected_weights = []
        for position, item_id in enumerate(items):
            if item_id in self.item_to_index:
                indices.append(self.item_to_index[item_id])
                selected_weights.append(1.0 if weights is None else float(weights[position]))
        if not indices:
            raise ValueError("None of the supplied items exist in the catalog")
        weight_array = np.asarray(selected_weights, dtype=float)
        if not np.isfinite(weight_array).all() or np.any(weight_array < 0):
            raise ValueError("profile weights must be finite and non-negative")
        if np.isclose(weight_array.sum(), 0.0):
            raise ValueError("At least one profile weight must be positive")
        return indices, weight_array

    def _weighted_profile(self, indices: Sequence[int], weights: np.ndarray) -> np.ndarray:
        selected = self.features[list(indices)]
        if sparse.issparse(selected):
            profile = np.asarray(selected.multiply(weights[:, None]).sum(axis=0)).ravel()
        else:
            profile = np.average(selected, axis=0, weights=weights)
        norm = np.linalg.norm(profile)
        return profile / norm if norm else profile

    def profile(
        self,
        liked_items: Sequence[Hashable],
        *,
        weights: Optional[Sequence[float]] = None,
        disliked_items: Sequence[Hashable] = (),
        dislike_weight: float = 0.5,
    ) -> np.ndarray:
        """Build a normalized positive-minus-negative content profile."""
        self._require_fitted()
        if dislike_weight < 0:
            raise ValueError("dislike_weight cannot be negative")
        indices, selected_weights = self._known_items(liked_items, weights)
        profile = self._weighted_profile(indices, selected_weights)
        if disliked_items:
            negative_indices, negative_weights = self._known_items(disliked_items)
            profile = profile - dislike_weight * self._weighted_profile(
                negative_indices, negative_weights
            )
            norm = np.linalg.norm(profile)
            if norm:
                profile = profile / norm
        return profile

    def recommend_from_profile(
        self,
        profile: Any,
        *,
        top_n: int = 10,
        exclude_items: Iterable[Hashable] = (),
    ) -> pd.DataFrame:
        self._require_fitted()
        top_n = validate_top_n(top_n)
        vector = np.asarray(profile, dtype=float).reshape(1, -1)
        if vector.shape[1] != self.features.shape[1]:
            raise ValueError("profile width must match feature width")
        if not np.isfinite(vector).all() or np.isclose(np.linalg.norm(vector), 0.0):
            raise ValueError("profile must be finite and non-zero")
        scores = cosine_similarity(vector, self.features).ravel()
        blocked = set(exclude_items)
        candidates = [
            (item_id, float(scores[index]))
            for index, item_id in enumerate(self.item_ids)
            if item_id not in blocked
        ]
        return recommendation_frame(
            [item for item, _ in candidates],
            [score for _, score in candidates],
            top_n=top_n,
            source="content",
            metadata=self.metadata,
        )

    def recommend_from_items(
        self,
        liked_items: Sequence[Hashable],
        *,
        top_n: int = 10,
        weights: Optional[Sequence[float]] = None,
        disliked_items: Sequence[Hashable] = (),
        dislike_weight: float = 0.5,
        exclude_seen: bool = True,
        exclude_items: Iterable[Hashable] = (),
    ) -> pd.DataFrame:
        profile = self.profile(
            liked_items,
            weights=weights,
            disliked_items=disliked_items,
            dislike_weight=dislike_weight,
        )
        blocked = set(exclude_items)
        if exclude_seen:
            blocked.update(liked_items)
            blocked.update(disliked_items)
        return self.recommend_from_profile(profile, top_n=top_n, exclude_items=blocked)

    def recommend(
        self,
        user_id: Hashable,
        *,
        top_n: int = 10,
        exclude_items: Iterable[Hashable] = (),
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        self._require_fitted()
        if user_id not in self.user_histories:
            raise KeyError(f"Unknown user_id: {user_id!r}")
        return self.recommend_from_items(
            self.user_histories[user_id],
            top_n=top_n,
            exclude_seen=exclude_seen,
            exclude_items=exclude_items,
        )

    def similar_items(
        self,
        item_id: Hashable,
        *,
        top_n: int = 10,
        exclude_items: Iterable[Hashable] = (),
    ) -> pd.DataFrame:
        self._require_fitted()
        try:
            index = self.item_to_index[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id: {item_id!r}") from exc
        profile = self.features[index]
        if sparse.issparse(profile):
            profile = profile.toarray()
        blocked = set(exclude_items) | {item_id}
        return self.recommend_from_profile(profile, top_n=top_n, exclude_items=blocked)


__all__ = [
    "ContentRecommender",
    "NumericFeatureTransformer",
    "combine_text_features",
    "concatenate_feature_embeddings",
    "prepare_numeric_features",
]

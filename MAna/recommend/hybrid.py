"""Score-calibrated and reciprocal-rank hybrid recommenders."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ..rag.ranking import fuse_rankings
from .base import BaseRecommender, recommendation_frame, validate_top_n


def weighted_score_fusion(
    score_frames: Mapping[str, pd.DataFrame],
    *,
    weights: Optional[Mapping[str, float]] = None,
    item_column: str = "item_id",
    score_column: str = "score",
    normalization: str = "minmax",
    top_n: int = 10,
) -> pd.DataFrame:
    """Normalize and combine scored candidate lists from several sources."""
    top_n = validate_top_n(top_n)
    if normalization not in {"minmax", "rank"}:
        raise ValueError("normalization must be 'minmax' or 'rank'")
    resolved_weights = dict(weights or {})
    combined: Dict[Hashable, float] = defaultdict(float)
    components: Dict[Hashable, Dict[str, float]] = defaultdict(dict)
    active_weight = 0.0

    for name, frame in score_frames.items():
        weight = float(resolved_weights.get(name, 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("hybrid weights must be finite and non-negative")
        if weight == 0 or frame.empty:
            continue
        missing = [column for column in (item_column, score_column) if column not in frame.columns]
        if missing:
            raise KeyError(f"Columns not found in {name!r}: {missing}")
        local = (
            frame[[item_column, score_column]]
            .dropna()
            .groupby(item_column, as_index=False, sort=False)[score_column]
            .max()
        )
        if local.empty:
            continue
        values = local[score_column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name!r} contains non-finite scores")
        if normalization == "minmax":
            spread = float(values.max() - values.min()) if len(values) else 0.0
            normalized = (
                (values - values.min()) / spread if spread > 0 else np.ones(len(values))
            )
        else:
            order = np.argsort(-values, kind="mergesort")
            ranks: np.ndarray = np.empty(len(values), dtype=float)
            ranks[order] = np.arange(1, len(values) + 1)
            normalized = 1.0 / ranks
        active_weight += weight
        for item_id, score in zip(local[item_column], normalized):
            component_score = float(score)
            components[item_id][name] = component_score
            combined[item_id] += weight * component_score

    if not combined:
        return recommendation_frame([], [], top_n=top_n, source="hybrid")
    denominator = active_weight if active_weight > 0 else 1.0
    ranked = recommendation_frame(
        list(combined),
        [score / denominator for score in combined.values()],
        top_n=top_n,
        source="hybrid",
    )
    ranked["component_scores"] = [dict(components[item]) for item in ranked["item_id"]]
    return ranked


class HybridRecommender(BaseRecommender):
    """Fuse any recommenders sharing MAna's recommendation DataFrame contract."""

    def __init__(
        self,
        recommenders: Mapping[str, Any],
        *,
        weights: Optional[Mapping[str, float]] = None,
        method: str = "weighted",
        normalization: str = "minmax",
        rank_constant: int = 60,
    ) -> None:
        super().__init__()
        if not recommenders:
            raise ValueError("recommenders cannot be empty")
        if method not in {"weighted", "rrf"}:
            raise ValueError("method must be 'weighted' or 'rrf'")
        if rank_constant < 0:
            raise ValueError("rank_constant cannot be negative")
        self.recommenders = dict(recommenders)
        self.weights = dict(weights or {})
        unknown_weights = set(self.weights) - set(self.recommenders)
        if unknown_weights:
            raise KeyError(f"Weights supplied for unknown recommenders: {sorted(unknown_weights)}")
        if any(not np.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("hybrid weights must be finite and non-negative")
        self.method = method
        self.normalization = normalization
        self.rank_constant = rank_constant
        self._mark_fitted()

    def recommend(
        self,
        user_id: Hashable,
        *,
        top_n: int = 10,
        candidate_pool_size: Optional[int] = None,
        exclude_items: Iterable[Hashable] = (),
        recommendation_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> pd.DataFrame:
        top_n = validate_top_n(top_n)
        pool_size = candidate_pool_size or max(50, top_n * 5)
        pool_size = validate_top_n(pool_size)
        source_kwargs = dict(recommendation_kwargs or {})
        frames: Dict[str, pd.DataFrame] = {}
        for name, recommender in self.recommenders.items():
            callback = recommender.recommend if hasattr(recommender, "recommend") else recommender
            if not callable(callback):
                raise TypeError(f"Recommender {name!r} is not callable")
            kwargs = {
                "user_id": user_id,
                "top_n": pool_size,
                "exclude_items": exclude_items,
                **dict(source_kwargs.get(name, {})),
            }
            frame = callback(**kwargs)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"Recommender {name!r} must return a pandas DataFrame")
            frames[name] = frame

        if self.method == "weighted":
            return weighted_score_fusion(
                frames,
                weights=self.weights,
                normalization=self.normalization,
                top_n=top_n,
            )

        rankings = {
            name: frame["item_id"].tolist()
            for name, frame in frames.items()
            if not frame.empty
        }
        fused = fuse_rankings(
            rankings,
            weights=self.weights,
            k=self.rank_constant,
            top_k=top_n,
        )
        ranked = recommendation_frame(
            [item.item_id for item in fused],
            [item.score for item in fused],
            top_n=top_n,
            source="hybrid_rrf",
        )
        rank_details = {item.item_id: dict(item.ranks) for item in fused}
        ranked["component_ranks"] = [rank_details[item] for item in ranked["item_id"]]
        return ranked


__all__ = ["HybridRecommender", "weighted_score_fusion"]

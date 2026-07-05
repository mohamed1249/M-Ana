"""Popularity and Bayesian-rating cold-start recommendations."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Optional, Set

import numpy as np
import pandas as pd

from .base import BaseRecommender, recommendation_frame, validate_top_n
from .data import build_user_histories, validate_interactions


class PopularityRecommender(BaseRecommender):
    """Recommend globally popular items with optional rating shrinkage and decay."""

    def __init__(
        self,
        *,
        user_column: str = "user_id",
        item_column: str = "item_id",
        score_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        score_mode: str = "bayesian",
        shrinkage: float = 10.0,
        half_life_days: Optional[float] = None,
    ) -> None:
        super().__init__()
        if score_mode not in {"count", "mean", "bayesian"}:
            raise ValueError("score_mode must be 'count', 'mean', or 'bayesian'")
        if shrinkage < 0:
            raise ValueError("shrinkage cannot be negative")
        if half_life_days is not None and half_life_days <= 0:
            raise ValueError("half_life_days must be greater than zero")
        if half_life_days is not None and timestamp_column is None:
            raise ValueError("timestamp_column is required when half_life_days is set")
        self.user_column = user_column
        self.item_column = item_column
        self.score_column = score_column
        self.timestamp_column = timestamp_column
        self.score_mode = score_mode
        self.shrinkage = float(shrinkage)
        self.half_life_days = half_life_days
        self.item_scores = pd.DataFrame()
        self.user_histories: Dict[Hashable, Set[Hashable]] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        data = validate_interactions(
            interactions,
            user_column=self.user_column,
            item_column=self.item_column,
            score_column=self.score_column,
            timestamp_column=self.timestamp_column,
        )
        if data.empty:
            raise ValueError("interactions cannot be empty")

        data = data.copy()
        if self.half_life_days is None:
            data["__weight__"] = 1.0
        else:
            latest = data[self.timestamp_column].max()
            age_days = (latest - data[self.timestamp_column]).dt.total_seconds() / 86_400
            data["__weight__"] = np.power(0.5, age_days / self.half_life_days)

        grouped = data.groupby(self.item_column, sort=False)
        summary = grouped.size().rename("interaction_count").to_frame()
        summary["weighted_count"] = grouped["__weight__"].sum()

        mode = self.score_mode
        if self.score_column is None:
            mode = "count"
        if mode == "count":
            summary["score"] = summary["weighted_count"]
        else:
            data["__weighted_score__"] = data[self.score_column] * data["__weight__"]
            weighted_sum = data.groupby(self.item_column, sort=False)["__weighted_score__"].sum()
            summary["mean_score"] = weighted_sum / summary["weighted_count"]
            if mode == "mean":
                summary["score"] = summary["mean_score"]
            else:
                global_mean = float(
                    data["__weighted_score__"].sum() / data["__weight__"].sum()
                )
                summary["score"] = (
                    summary["weighted_count"] * summary["mean_score"]
                    + self.shrinkage * global_mean
                ) / (summary["weighted_count"] + self.shrinkage)

        self.item_scores = summary.reset_index()
        histories = build_user_histories(
            data,
            user_column=self.user_column,
            item_column=self.item_column,
            score_column=self.score_column,
            timestamp_column=self.timestamp_column,
        )
        self.user_histories = {user: set(items) for user, items in histories.items()}
        self._mark_fitted()
        return self

    def recommend(
        self,
        user_id: Optional[Hashable] = None,
        *,
        top_n: int = 10,
        exclude_items: Iterable[Hashable] = (),
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Rank popular items, optionally removing the user's history."""
        self._require_fitted()
        top_n = validate_top_n(top_n)
        blocked = set(exclude_items)
        if exclude_seen and user_id is not None:
            blocked.update(self.user_histories.get(user_id, set()))
        candidates = self.item_scores[~self.item_scores[self.item_column].isin(blocked)]
        ranked = recommendation_frame(
            candidates[self.item_column].tolist(),
            candidates["score"].tolist(),
            top_n=top_n,
            source="popularity",
        )
        details = self.item_scores.rename(columns={self.item_column: "item_id"}).drop(
            columns="score"
        )
        return ranked.merge(details, on="item_id", how="left", sort=False)


__all__ = ["PopularityRecommender"]

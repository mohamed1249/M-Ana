"""Shared contracts and result formatting for recommender models."""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


ItemId = Hashable


def validate_top_n(top_n: int) -> int:
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
        raise TypeError("top_n must be an integer")
    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    return top_n


def recommendation_frame(
    item_ids: Sequence[ItemId],
    scores: Sequence[float],
    *,
    top_n: int,
    source: str,
    metadata: Optional[Mapping[ItemId, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Build a consistently ranked recommendation DataFrame."""
    top_n = validate_top_n(top_n)
    if len(item_ids) != len(scores):
        raise ValueError("item_ids and scores must have the same length")

    frame = pd.DataFrame(
        {
            "item_id": list(item_ids),
            "score": np.asarray(scores, dtype=float),
        }
    )
    if frame.empty:
        return pd.DataFrame(columns=["item_id", "score", "rank", "source"])
    if not np.isfinite(frame["score"]).all():
        raise ValueError("recommendation scores must be finite")

    frame = frame.groupby("item_id", as_index=False, sort=False)["score"].max()
    frame["_tie_breaker"] = frame["item_id"].map(str)
    frame = (
        frame.sort_values(
            ["score", "_tie_breaker"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(top_n)
        .drop(columns="_tie_breaker")
        .reset_index(drop=True)
    )
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["source"] = source
    if metadata is not None:
        frame["metadata"] = [dict(metadata.get(item_id, {})) for item_id in frame["item_id"]]
    return frame


class BaseRecommender:
    """Small fitted-state mixin shared by recommender implementations."""

    def __init__(self) -> None:
        self.is_fitted = False

    def _mark_fitted(self) -> None:
        self.is_fitted = True

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("fit must be called before requesting recommendations")


__all__ = [
    "BaseRecommender",
    "ItemId",
    "recommendation_frame",
    "validate_top_n",
]

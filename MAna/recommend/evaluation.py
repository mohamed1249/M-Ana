"""Offline top-k recommender evaluation and catalog diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Hashable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..rag.evaluation import retrieval_metrics
from .data import validate_interactions


def catalog_coverage(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    catalog_items: Sequence[Hashable],
    *,
    k: Optional[int] = None,
) -> float:
    """Fraction of the catalog appearing in at least one recommendation list."""
    catalog = set(catalog_items)
    if not catalog:
        raise ValueError("catalog_items cannot be empty")
    recommended = {
        item
        for items in recommendations.values()
        for item in (items if k is None else items[:k])
        if item in catalog
    }
    return float(len(recommended) / len(catalog))


def intra_list_diversity(
    recommended: Sequence[Hashable],
    item_ids: Sequence[Hashable],
    features: Any,
    *,
    k: Optional[int] = None,
) -> float:
    """Return one minus mean pairwise cosine similarity for known items."""
    ids = list(item_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids must be unique")
    id_to_index = {item_id: index for index, item_id in enumerate(ids)}
    selected_ids = list(recommended if k is None else recommended[:k])
    indices = [id_to_index[item] for item in selected_ids if item in id_to_index]
    if len(indices) < 2:
        return 0.0
    matrix = features[indices]
    similarities = cosine_similarity(matrix)
    upper = similarities[np.triu_indices_from(similarities, k=1)]
    return float(1 - np.mean(upper))


def evaluate_rankings(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    relevant_items: Mapping[Hashable, Sequence[Hashable]],
    *,
    k: int = 10,
    catalog_items: Optional[Sequence[Hashable]] = None,
) -> Dict[str, Any]:
    """Average top-k ranking metrics across users with relevant holdout items."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    users = [
        user
        for user in recommendations
        if user in relevant_items and len(relevant_items[user]) > 0
    ]
    if not users:
        raise ValueError("No evaluable users are shared by both mappings")

    per_user = [
        retrieval_metrics(recommendations[user], relevant_items[user], k=k)
        for user in users
    ]
    result: Dict[str, Any] = {
        "users_evaluated": len(users),
        "precision_at_k": float(np.mean([metrics["precision"] for metrics in per_user])),
        "recall_at_k": float(np.mean([metrics["recall"] for metrics in per_user])),
        "hit_rate_at_k": float(np.mean([metrics["hit_rate"] for metrics in per_user])),
        "mrr_at_k": float(np.mean([metrics["reciprocal_rank"] for metrics in per_user])),
        "map_at_k": float(np.mean([metrics["average_precision"] for metrics in per_user])),
        "ndcg_at_k": float(np.mean([metrics["ndcg"] for metrics in per_user])),
    }
    if catalog_items is not None:
        result["catalog_coverage"] = catalog_coverage(
            recommendations,
            catalog_items,
            k=k,
        )
    return result


def evaluate_recommender(
    recommender: Any,
    test_interactions: pd.DataFrame,
    *,
    user_column: str = "user_id",
    item_column: str = "item_id",
    k: int = 10,
    catalog_items: Optional[Sequence[Hashable]] = None,
    recommendation_kwargs: Optional[Mapping[str, Any]] = None,
    on_error: str = "raise",
) -> Dict[str, Any]:
    """Generate recommendations for test users and evaluate their held-out items."""
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    data = validate_interactions(
        test_interactions,
        user_column=user_column,
        item_column=item_column,
    )
    relevant = {
        user: group[item_column].drop_duplicates().tolist()
        for user, group in data.groupby(user_column, sort=False)
    }
    recommendations: Dict[Hashable, Sequence[Hashable]] = {}
    skipped = 0
    for user_id in relevant:
        try:
            frame = recommender.recommend(
                user_id=user_id,
                top_n=k,
                **dict(recommendation_kwargs or {}),
            )
        except (KeyError, ValueError, RuntimeError):
            if on_error == "raise":
                raise
            skipped += 1
            continue
        if not isinstance(frame, pd.DataFrame) or "item_id" not in frame.columns:
            raise TypeError("recommender.recommend must return a DataFrame with item_id")
        recommendations[user_id] = frame["item_id"].tolist()

    filtered_relevant = {user: relevant[user] for user in recommendations}
    result = evaluate_rankings(
        recommendations,
        filtered_relevant,
        k=k,
        catalog_items=catalog_items,
    )
    result["users_skipped"] = skipped
    return result


__all__ = [
    "catalog_coverage",
    "evaluate_rankings",
    "evaluate_recommender",
    "intra_list_diversity",
]

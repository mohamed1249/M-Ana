"""Evaluation metrics for ranked retrieval results."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def _item_id(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("id", item.get("item_id", str(item)))
    if hasattr(item, "id"):
        return item.id
    if hasattr(item, "item_id"):
        return item.item_id
    return item


def _hashable_id(item: Any) -> Any:
    value = _item_id(item)
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def retrieval_metrics(
    retrieved: Sequence[Any],
    relevant: Sequence[Any],
    *,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate binary-relevance metrics for one ranked retrieval query.

    Results may be raw IDs, mappings with ``id``, or objects such as
    :class:`SearchHit`. Duplicate retrieved IDs are counted only once.
    """
    if k is not None and k <= 0:
        raise ValueError("k must be greater than zero")

    relevant_ids = {_hashable_id(item) for item in relevant}
    if not relevant_ids:
        raise ValueError("relevant must contain at least one item")

    ranked_ids = []
    seen = set()
    for item in retrieved:
        item_id = _hashable_id(item)
        duplicate = item_id in seen
        if duplicate:
            continue
        seen.add(item_id)
        ranked_ids.append(item_id)

    cutoff = len(ranked_ids) if k is None else k
    top_ids = ranked_ids[:cutoff]
    relevance = np.asarray(
        [1.0 if item_id in relevant_ids else 0.0 for item_id in top_ids],
        dtype=float,
    )
    hits = int(relevance.sum())
    precision_denominator = cutoff if cutoff > 0 else 1
    precision = hits / precision_denominator
    recall = hits / len(relevant_ids)

    relevant_ranks = np.flatnonzero(relevance) + 1
    reciprocal_rank = 1 / int(relevant_ranks[0]) if relevant_ranks.size else 0.0
    precision_at_relevant = [
        float(relevance[:rank].sum() / rank) for rank in relevant_ranks
    ]
    average_precision_denominator = min(len(relevant_ids), cutoff)
    average_precision = (
        sum(precision_at_relevant) / average_precision_denominator
        if average_precision_denominator
        else 0.0
    )

    discounts = np.log2(np.arange(2, len(relevance) + 2))
    dcg = float(np.sum(relevance / discounts)) if relevance.size else 0.0
    ideal_hits = min(len(relevant_ids), cutoff)
    ideal_discounts = np.log2(np.arange(2, ideal_hits + 2))
    ideal_dcg = float(np.sum(1 / ideal_discounts)) if ideal_hits else 0.0

    return {
        "k": int(cutoff),
        "retrieved": int(len(top_ids)),
        "relevant": int(len(relevant_ids)),
        "relevant_retrieved": hits,
        "precision": float(precision),
        "recall": float(recall),
        "hit_rate": float(hits > 0),
        "reciprocal_rank": float(reciprocal_rank),
        "average_precision": float(average_precision),
        "ndcg": float(dcg / ideal_dcg) if ideal_dcg else 0.0,
    }


__all__ = ["retrieval_metrics"]

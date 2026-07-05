"""Ranking fusion utilities shared by retrieval and recommenders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RankedItem:
    item_id: Hashable
    score: float
    ranks: Dict[str, int] = field(default_factory=dict)


def fuse_rankings(
    rankings: Mapping[str, Sequence[Hashable]],
    *,
    weights: Optional[Mapping[str, float]] = None,
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[RankedItem]:
    """Fuse ranked stable IDs using weighted reciprocal-rank fusion."""
    if k < 0:
        raise ValueError("k cannot be negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    weights = dict(weights or {})
    scores: Dict[Hashable, float] = {}
    observed_ranks: Dict[Hashable, Dict[str, int]] = {}

    for name, items in rankings.items():
        weight = float(weights.get(name, 1.0))
        if weight < 0:
            raise ValueError("ranking weights cannot be negative")
        seen = set()
        for rank, item_id in enumerate(items, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
            observed_ranks.setdefault(item_id, {})[name] = rank

    results = [
        RankedItem(item_id=item_id, score=score, ranks=observed_ranks[item_id])
        for item_id, score in scores.items()
    ]
    results.sort(key=lambda item: (-item.score, str(item.item_id)))
    return results[:top_k] if top_k is not None else results


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Hashable]],
    *,
    weights: Optional[Mapping[str, float]] = None,
    k: int = 60,
    top_k: Optional[int] = None,
    return_scores: bool = False,
):
    """Return fused IDs, or rich :class:`RankedItem` values on request."""
    results = fuse_rankings(rankings, weights=weights, k=k, top_k=top_k)
    return results if return_scores else [item.item_id for item in results]

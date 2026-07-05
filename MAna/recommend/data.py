"""Interaction validation, sparse matrices, histories, and temporal splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")


def validate_interactions(
    interactions: pd.DataFrame,
    *,
    user_column: str = "user_id",
    item_column: str = "item_id",
    score_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """Validate and copy an interaction table without mutating the input."""
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame")
    required = [user_column, item_column]
    if score_column is not None:
        required.append(score_column)
    if timestamp_column is not None:
        required.append(timestamp_column)
    _require_columns(interactions, required)

    data = interactions.copy()
    if data[[user_column, item_column]].isna().any().any():
        raise ValueError("user and item identifiers cannot be missing")

    if score_column is not None:
        numeric = pd.to_numeric(data[score_column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{score_column!r} must contain finite numeric values")
        data[score_column] = numeric.astype(float)

    if timestamp_column is not None:
        timestamps = pd.to_datetime(data[timestamp_column], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError(f"{timestamp_column!r} contains invalid timestamps")
        data[timestamp_column] = timestamps
    return data


@dataclass(frozen=True)
class InteractionMatrix:
    """Sparse user-item matrix plus stable identifier mappings."""

    matrix: sparse.csr_matrix
    user_ids: List[Hashable]
    item_ids: List[Hashable]
    user_to_index: Dict[Hashable, int]
    item_to_index: Dict[Hashable, int]

    def user_index(self, user_id: Hashable) -> int:
        try:
            return self.user_to_index[user_id]
        except KeyError as exc:
            raise KeyError(f"Unknown user_id: {user_id!r}") from exc

    def item_index(self, item_id: Hashable) -> int:
        try:
            return self.item_to_index[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id: {item_id!r}") from exc


def build_interaction_matrix(
    interactions: pd.DataFrame,
    *,
    user_column: str = "user_id",
    item_column: str = "item_id",
    score_column: Optional[str] = None,
    aggregation: str = "binary",
) -> InteractionMatrix:
    """Build a CSR user-item matrix with deterministic first-seen ordering."""
    if aggregation not in {"binary", "sum", "mean", "max"}:
        raise ValueError("aggregation must be 'binary', 'sum', 'mean', or 'max'")
    data = validate_interactions(
        interactions,
        user_column=user_column,
        item_column=item_column,
        score_column=score_column,
    )
    if data.empty:
        raise ValueError("interactions cannot be empty")

    value_column = "__value__"
    data[value_column] = 1.0 if score_column is None else data[score_column].astype(float)
    group_columns = [user_column, item_column]
    if aggregation == "binary":
        grouped = data.groupby(group_columns, sort=False, as_index=False)[value_column].max()
        grouped[value_column] = 1.0
    else:
        grouped = data.groupby(group_columns, sort=False, as_index=False)[value_column].agg(aggregation)

    user_ids = pd.unique(data[user_column]).tolist()
    item_ids = pd.unique(data[item_column]).tolist()
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    rows = grouped[user_column].map(user_to_index).to_numpy(dtype=int)
    columns = grouped[item_column].map(item_to_index).to_numpy(dtype=int)
    values = grouped[value_column].to_numpy(dtype=float)
    matrix = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(user_ids), len(item_ids)),
        dtype=float,
    )
    return InteractionMatrix(
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
    )


def build_user_histories(
    interactions: pd.DataFrame,
    *,
    user_column: str = "user_id",
    item_column: str = "item_id",
    score_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    minimum_interactions: int = 1,
    minimum_score: Optional[float] = None,
) -> Dict[Hashable, List[Hashable]]:
    """Create de-duplicated user histories in temporal or score order."""
    if minimum_interactions <= 0:
        raise ValueError("minimum_interactions must be greater than zero")
    if minimum_score is not None and score_column is None:
        raise ValueError("score_column is required when minimum_score is provided")

    data = validate_interactions(
        interactions,
        user_column=user_column,
        item_column=item_column,
        score_column=score_column,
        timestamp_column=timestamp_column,
    )
    if minimum_score is not None:
        data = data[data[score_column] >= minimum_score]

    histories: Dict[Hashable, List[Hashable]] = {}
    for user_id, group in data.groupby(user_column, sort=False):
        if timestamp_column is not None:
            group = group.sort_values(timestamp_column, kind="mergesort")
        elif score_column is not None:
            group = group.sort_values(score_column, ascending=False, kind="mergesort")
        items = group[item_column].drop_duplicates().tolist()
        if len(items) >= minimum_interactions:
            histories[user_id] = items
    return histories


def temporal_train_test_split(
    interactions: pd.DataFrame,
    *,
    timestamp_column: str,
    user_column: str = "user_id",
    item_column: str = "item_id",
    score_column: Optional[str] = None,
    holdout: int = 1,
    minimum_interactions: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each eligible user's latest interactions without future leakage."""
    if holdout <= 0:
        raise ValueError("holdout must be greater than zero")
    if minimum_interactions <= holdout:
        raise ValueError("minimum_interactions must be greater than holdout")

    data = validate_interactions(
        interactions,
        user_column=user_column,
        item_column=item_column,
        score_column=score_column,
        timestamp_column=timestamp_column,
    ).reset_index(drop=True)
    data["__row_order__"] = np.arange(len(data))
    test_indices: List[int] = []
    for _, group in data.groupby(user_column, sort=False):
        if len(group) < minimum_interactions:
            continue
        ordered = group.sort_values(
            [timestamp_column, "__row_order__"],
            kind="mergesort",
        )
        test_indices.extend(ordered.tail(holdout).index.tolist())

    test_mask = data.index.isin(test_indices)
    columns = [column for column in data.columns if column != "__row_order__"]
    train = data.loc[~test_mask, columns].reset_index(drop=True)
    test = data.loc[test_mask, columns].reset_index(drop=True)
    return train, test


__all__ = [
    "InteractionMatrix",
    "build_interaction_matrix",
    "build_user_histories",
    "temporal_train_test_split",
    "validate_interactions",
]

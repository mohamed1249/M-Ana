"""Panel and grouped time-series feature engineering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import pandas as pd


def _as_columns(value: Union[str, Sequence[str]], name: str) -> list[str]:
    columns = [value] if isinstance(value, str) else list(value)
    if not columns:
        raise ValueError(f"{name} cannot be empty")
    return columns


def create_grouped_lag_features(
    data: pd.DataFrame,
    group_by: Union[str, Sequence[str]],
    columns: Union[str, Sequence[str]],
    lags: Union[int, Sequence[int]],
    *,
    time_col: Optional[str] = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Create lags independently within each panel group.

    When ``time_col`` is provided, observations are sorted chronologically
    inside each group before shifting, preventing accidental cross-time or
    cross-entity leakage.
    """
    groups = _as_columns(group_by, "group_by")
    targets = _as_columns(columns, "columns")
    lag_values = list(range(1, lags + 1)) if isinstance(lags, int) else list(lags)
    if not lag_values or any(not isinstance(lag, int) or lag <= 0 for lag in lag_values):
        raise ValueError("lags must contain positive integers")

    required = [*groups, *targets, *([time_col] if time_col else [])]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")

    result = data.copy()
    if time_col is not None:
        result[time_col] = pd.to_datetime(result[time_col])
        result = result.sort_values([*groups, time_col], kind="stable")

    created: list[str] = []
    grouped = result.groupby(groups, sort=False, observed=True)
    for target in targets:
        for lag in sorted(set(lag_values)):
            feature_name = f"{target}_lag_{lag}"
            result[feature_name] = grouped[target].shift(lag)
            created.append(feature_name)

    return result.dropna(subset=created) if dropna else result


def aggregate_panel(
    data: pd.DataFrame,
    date_col: str,
    value_cols: Union[str, Sequence[str]],
    *,
    group_by: Optional[Union[str, Sequence[str]]] = None,
    frequency: str = "D",
    aggregation: Union[str, Mapping[str, Any]] = "sum",
) -> pd.DataFrame:
    """Aggregate transactional observations into regular panel periods."""
    values = _as_columns(value_cols, "value_cols")
    groups = [] if group_by is None else _as_columns(group_by, "group_by")
    required = [date_col, *groups, *values]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")

    result = data.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    groupers: list[Any] = [pd.Grouper(key=date_col, freq=frequency), *groups]
    aggregated = (
        result.groupby(groupers, as_index=False, observed=True)[values]
        .agg(aggregation)
        .sort_values([*groups, date_col] if groups else [date_col], kind="stable")
        .reset_index(drop=True)
    )
    return aggregated

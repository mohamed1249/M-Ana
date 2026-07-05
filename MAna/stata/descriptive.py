"""Descriptive statistics and frequency summaries.

These helpers provide compact, reusable summaries for exploratory analysis while
consistently excluding missing and non-finite numeric observations.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats


NumberSequence = Union[Sequence[float], pd.Series, np.ndarray]


def _numeric_sample(values: NumberSequence) -> tuple[np.ndarray, int]:
    """Return finite one-dimensional values and the number excluded."""
    array = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(array)
    sample = array[finite]
    if sample.size == 0:
        raise ValueError("At least one finite numeric observation is required.")
    return sample, int(array.size - sample.size)


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")


def descriptive_statistics(
    values: NumberSequence,
    *,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Summarize a numeric sample and estimate a t-based mean interval.

    Missing values and positive or negative infinity are excluded. ``mode`` is
    the smallest mode when a sample is multimodal; all tied values are also
    available in ``modes``.
    """
    _validate_confidence(confidence)
    sample, excluded = _numeric_sample(values)
    count = sample.size
    q1, median, q3 = np.percentile(sample, [25, 50, 75])
    mode_values, mode_counts = np.unique(sample, return_counts=True)
    modes = mode_values[mode_counts == mode_counts.max()]

    if count > 1:
        variance = float(np.var(sample, ddof=1))
        standard_deviation = float(np.std(sample, ddof=1))
        standard_error = float(standard_deviation / np.sqrt(count))
        critical = float(stats.t.ppf((1 + confidence) / 2, count - 1))
        margin = critical * standard_error
        mean_ci = (float(np.mean(sample) - margin), float(np.mean(sample) + margin))
    else:
        variance = np.nan
        standard_deviation = np.nan
        standard_error = np.nan
        mean_ci = (np.nan, np.nan)

    has_variation = count > 1 and not np.isclose(np.ptp(sample), 0.0)
    skewness = (
        float(stats.skew(sample, bias=False))
        if count > 2 and has_variation
        else np.nan
    )
    excess_kurtosis = (
        float(stats.kurtosis(sample, fisher=True, bias=False))
        if count > 3 and has_variation
        else np.nan
    )

    return {
        "count": int(count),
        "excluded": excluded,
        "mean": float(np.mean(sample)),
        "median": float(median),
        "mode": float(modes[0]),
        "modes": [float(value) for value in modes],
        "mode_count": int(mode_counts.max()),
        "minimum": float(np.min(sample)),
        "maximum": float(np.max(sample)),
        "range": float(np.ptp(sample)),
        "variance": variance,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "confidence": float(confidence),
        "mean_ci": mean_ci,
    }


def grouped_descriptive_statistics(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: Union[str, Sequence[str]],
) -> pd.DataFrame:
    """Return count, center, spread, and quartiles for DataFrame groups."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    groups = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    if not groups:
        raise ValueError("At least one group column is required.")

    required = [value_column, *groups]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    data = frame[required].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data[value_column] = data[value_column].replace([np.inf, -np.inf], np.nan)

    result = (
        data.groupby(groups, dropna=False, observed=True, sort=False)[value_column]
        .agg(
            count="count",
            mean="mean",
            median="median",
            standard_deviation="std",
            minimum="min",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )
    result["iqr"] = result["q3"] - result["q1"]
    return result


def frequency_table(
    values: Iterable[Any],
    *,
    include_missing: bool = True,
) -> pd.DataFrame:
    """Create counts, proportions, percentages, and cumulative percentages."""
    series = pd.Series(list(values), dtype="object")
    counts = series.value_counts(dropna=not include_missing)
    result = counts.rename_axis("value").reset_index(name="count")

    if result.empty:
        result["proportion"] = pd.Series(dtype=float)
        result["percentage"] = pd.Series(dtype=float)
        result["cumulative_percentage"] = pd.Series(dtype=float)
        return result

    result["proportion"] = result["count"] / result["count"].sum()
    result["percentage"] = result["proportion"] * 100
    result["cumulative_percentage"] = result["percentage"].cumsum()
    return result


__all__ = [
    "descriptive_statistics",
    "frequency_table",
    "grouped_descriptive_statistics",
]

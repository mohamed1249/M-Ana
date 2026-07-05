"""Correlation, partial correlation, and simple regression helpers."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats


NumberSequence = Union[Sequence[float], pd.Series, np.ndarray]


def _paired_numeric(
    x: NumberSequence,
    y: NumberSequence,
    *,
    minimum: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y, dtype=float).ravel()
    if x_array.size != y_array.size:
        raise ValueError("Paired inputs must have the same length.")
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    x_clean, y_clean = x_array[keep], y_array[keep]
    if x_clean.size < minimum:
        raise ValueError(f"At least {minimum} paired finite observations are required.")
    return x_clean, y_clean


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _validate_nonconstant(x: np.ndarray, y: np.ndarray) -> None:
    if np.isclose(np.ptp(x), 0.0) or np.isclose(np.ptp(y), 0.0):
        raise ValueError("Correlation requires variation in both inputs.")


def correlation_test(
    x: NumberSequence,
    y: NumberSequence,
    *,
    method: str = "pearson",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test Pearson, Spearman, or Kendall association between paired values."""
    _validate_alpha(alpha)
    x_clean, y_clean = _paired_numeric(x, y)
    _validate_nonconstant(x_clean, y_clean)

    method = method.lower()
    tests = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }
    if method not in tests:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    coefficient, p_value = tests[method](x_clean, y_clean)
    p_value = float(p_value)
    return {
        "method": method,
        "n": int(x_clean.size),
        "coefficient": float(coefficient),
        "p_value": p_value,
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
        "decision": "reject_null" if p_value < alpha else "fail_to_reject_null",
    }


def partial_correlation(
    frame: pd.DataFrame,
    x: str,
    y: str,
    covariates: Union[str, Sequence[str]],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Estimate Pearson correlation after linearly controlling for covariates.

    The p-value uses ``n - k - 2`` degrees of freedom, where ``k`` is the
    number of covariates.
    """
    _validate_alpha(alpha)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    controls = [covariates] if isinstance(covariates, str) else list(covariates)
    if not controls:
        raise ValueError("At least one covariate is required.")
    if len(set(controls)) != len(controls):
        raise ValueError("covariates must not contain duplicates.")
    if x == y or x in controls or y in controls:
        raise ValueError("x, y, and covariates must refer to distinct columns.")

    columns = [x, y, *controls]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    data = frame[columns].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    degrees_of_freedom = len(data) - len(controls) - 2
    if degrees_of_freedom < 1:
        raise ValueError("Not enough complete observations for the covariates.")

    design = np.column_stack(
        [np.ones(len(data), dtype=float), data[controls].to_numpy(dtype=float)]
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Covariates are perfectly collinear.")

    x_values = data[x].to_numpy(dtype=float)
    y_values = data[y].to_numpy(dtype=float)
    x_residual = x_values - design @ np.linalg.lstsq(design, x_values, rcond=None)[0]
    y_residual = y_values - design @ np.linalg.lstsq(design, y_values, rcond=None)[0]
    _validate_nonconstant(x_residual, y_residual)

    coefficient, _ = stats.pearsonr(x_residual, y_residual)
    coefficient = float(coefficient)
    coefficient = float(np.clip(coefficient, -1.0, 1.0))
    if np.isclose(abs(coefficient), 1.0):
        p_value = 0.0
    else:
        test_statistic = coefficient * np.sqrt(
            degrees_of_freedom / (1 - coefficient**2)
        )
        p_value = float(2 * stats.t.sf(abs(test_statistic), degrees_of_freedom))

    return {
        "x": x,
        "y": y,
        "covariates": controls,
        "n": int(len(data)),
        "degrees_of_freedom": int(degrees_of_freedom),
        "coefficient": coefficient,
        "p_value": p_value,
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
        "decision": "reject_null" if p_value < alpha else "fail_to_reject_null",
    }


def correlation_matrix(
    frame: pd.DataFrame,
    *,
    method: str = "pearson",
    minimum_periods: int = 1,
) -> pd.DataFrame:
    """Return a correlation matrix for numeric DataFrame columns only."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    method = method.lower()
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")
    if minimum_periods < 1:
        raise ValueError("minimum_periods must be at least 1.")

    numeric = frame.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    return numeric.corr(method=method, min_periods=minimum_periods)


def linear_regression_test(
    x: NumberSequence,
    y: NumberSequence,
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Fit a simple least-squares line and return inferential diagnostics."""
    _validate_alpha(alpha)
    x_clean, y_clean = _paired_numeric(x, y)
    if np.isclose(np.ptp(x_clean), 0.0):
        raise ValueError("Regression requires variation in x.")

    result = stats.linregress(x_clean, y_clean)
    p_value = float(result.pvalue)
    return {
        "n": int(x_clean.size),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_value": float(result.rvalue),
        "r_squared": float(result.rvalue**2),
        "p_value": p_value,
        "slope_standard_error": float(result.stderr),
        "intercept_standard_error": float(result.intercept_stderr),
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
        "decision": "reject_null" if p_value < alpha else "fail_to_reject_null",
    }


__all__ = [
    "correlation_matrix",
    "correlation_test",
    "linear_regression_test",
    "partial_correlation",
]

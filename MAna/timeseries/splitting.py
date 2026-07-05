"""Leakage-safe chronological splitting and scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ScaledTimeSeriesSplit:
    """Chronological arrays, fitted train-only scalers, and source indices."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_scaler: Any
    target_scaler: Any
    train_index: pd.Index
    test_index: pd.Index
    target_was_series: bool = False

    def inverse_transform_targets(self, values: Any):
        restored = self.target_scaler.inverse_transform(
            np.asarray(values).reshape(-1, 1)
            if self.target_was_series
            else np.asarray(values)
        )
        return restored.ravel() if self.target_was_series else restored


def _test_count(length: int, test_size: Union[int, float], gap: int) -> int:
    if gap < 0:
        raise ValueError("gap cannot be negative")
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("float test_size must be between 0 and 1")
        count = max(1, int(length * test_size))
    else:
        count = int(test_size)
        if count <= 0:
            raise ValueError("integer test_size must be greater than zero")
    if length - count - gap <= 0:
        raise ValueError("test_size and gap leave no training observations")
    return count


def chronological_split_and_scale(
    features: pd.DataFrame,
    targets: Union[pd.Series, pd.DataFrame],
    test_size: Union[int, float] = 0.2,
    *,
    gap: int = 0,
    feature_scaler: Optional[Any] = None,
    target_scaler: Optional[Any] = None,
) -> ScaledTimeSeriesSplit:
    """Split chronologically, then fit scalers on training observations only."""
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    if not features.index.equals(targets.index):
        raise ValueError("features and targets must have aligned indices")

    test_count = _test_count(len(features), test_size, gap)
    train_count = len(features) - test_count - gap
    X_train_raw = features.iloc[:train_count]
    X_test_raw = features.iloc[train_count + gap :]
    target_was_series = isinstance(targets, pd.Series)
    target_frame = targets.to_frame() if target_was_series else targets
    y_train_raw = target_frame.iloc[:train_count]
    y_test_raw = target_frame.iloc[train_count + gap :]

    feature_scaler = feature_scaler or StandardScaler()
    target_scaler = target_scaler or StandardScaler()
    X_train = feature_scaler.fit_transform(X_train_raw)
    X_test = feature_scaler.transform(X_test_raw)
    y_train = target_scaler.fit_transform(y_train_raw)
    y_test = target_scaler.transform(y_test_raw)

    return ScaledTimeSeriesSplit(
        X_train=np.asarray(X_train),
        X_test=np.asarray(X_test),
        y_train=np.asarray(y_train).ravel() if target_was_series else np.asarray(y_train),
        y_test=np.asarray(y_test).ravel() if target_was_series else np.asarray(y_test),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_index=X_train_raw.index.copy(),
        test_index=X_test_raw.index.copy(),
        target_was_series=target_was_series,
    )

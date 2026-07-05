"""Numerical time-series diagnostics."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd


def periodogram_spectrum(
    data: Union[pd.Series, Sequence[float], np.ndarray],
    *,
    sampling_rate: float = 1.0,
    detrend: str = "linear",
) -> pd.DataFrame:
    """Return non-zero spectral frequencies, power, and implied periods."""
    from scipy.signal import periodogram

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than zero")
    values = pd.Series(data, dtype=float).dropna().to_numpy()
    if len(values) < 3:
        raise ValueError("at least three non-missing observations are required")
    frequencies, power = periodogram(
        values,
        fs=sampling_rate,
        detrend=detrend,
        window="boxcar",
        scaling="spectrum",
    )
    nonzero = frequencies > 0
    frequencies = frequencies[nonzero]
    return pd.DataFrame(
        {
            "frequency": frequencies,
            "power": power[nonzero],
            "period": 1.0 / frequencies,
        }
    )


def dominant_periods(
    data: Union[pd.Series, Sequence[float], np.ndarray],
    n: int = 5,
    *,
    sampling_rate: float = 1.0,
    detrend: str = "linear",
) -> pd.DataFrame:
    """Return the strongest spectral periods, ranked by power."""
    if n <= 0:
        raise ValueError("n must be greater than zero")
    spectrum = periodogram_spectrum(
        data, sampling_rate=sampling_rate, detrend=detrend
    )
    return spectrum.nlargest(n, "power").reset_index(drop=True)

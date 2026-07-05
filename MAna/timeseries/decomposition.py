"""
Time Series Decomposition Module

Extract trend, seasonal, and residual components from time series.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass


# ==================== DECOMPOSITION RESULT ====================

@dataclass
class DecompositionResult:
    """
    Container for decomposition results.

    Attributes:
    -----------
    observed : pd.Series
        Original time series.
    trend : pd.Series
        Trend component.
    seasonal : pd.Series
        Seasonal component.
    residual : pd.Series
        Residual component.
    """
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series

    def plot(self, figsize: Tuple[int, int] = (14, 10)):
        """Plot all components."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        self.observed.plot(ax=axes[0], color='#3498db', title='Observed')
        axes[0].set_ylabel('Observed', fontweight='bold')
        axes[0].grid(alpha=0.3)

        self.trend.plot(ax=axes[1], color='#e74c3c', title='Trend')
        axes[1].set_ylabel('Trend', fontweight='bold')
        axes[1].grid(alpha=0.3)

        self.seasonal.plot(ax=axes[2], color='#2ecc71', title='Seasonal')
        axes[2].set_ylabel('Seasonal', fontweight='bold')
        axes[2].grid(alpha=0.3)

        self.residual.plot(ax=axes[3], color='#9b59b6', title='Residual')
        axes[3].set_ylabel('Residual', fontweight='bold')
        axes[3].set_xlabel('Date', fontweight='bold')
        axes[3].grid(alpha=0.3)

        plt.suptitle('Time Series Decomposition', fontweight='bold', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.show()


# ==================== SEASONAL DECOMPOSITION ====================

def seasonal_decompose(
    data: Union[pd.Series, pd.DataFrame],
    model: str = 'additive',
    period: Optional[int] = None,
    extrapolate_trend: str = 'freq'
) -> DecompositionResult:
    """
    Classical seasonal decomposition.

    Separates time series into trend, seasonal, and residual components.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data with DatetimeIndex.
    model : str
        'additive' or 'multiplicative'.
        Additive: Y = Trend + Seasonal + Residual
        Multiplicative: Y = Trend * Seasonal * Residual
    period : int, optional
        Seasonal period (e.g., 7 for weekly, 12 for monthly).
        If None, will try to infer from data.
    extrapolate_trend : str
        How to handle trend at boundaries ('freq' or int).

    Returns:
    --------
    DecompositionResult
        Object with trend, seasonal, residual components.

    Examples:
    ---------
    >>> # Weekly seasonality
    >>> result = seasonal_decompose(df['sales'], period=7)
    >>> result.plot()
    >>>
    >>> # Monthly seasonality with multiplicative model
    >>> result = seasonal_decompose(df['sales'], model='multiplicative', period=12)
    >>> print(result.trend.head())
    """
    from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose

    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    # Infer period if not provided
    if period is None:
        from .preprocessing import detect_frequency
        freq = detect_frequency(data)

        if freq == 'D':
            period = 7  # Weekly seasonality for daily data
        elif freq == 'M':
            period = 12  # Yearly seasonality for monthly data
        elif freq == 'H':
            period = 24  # Daily seasonality for hourly data
        else:
            raise ValueError("Cannot infer period. Please specify period parameter.")

        print(f"[INFO] Inferred period: {period}")

    # Perform decomposition
    decomposition = sm_decompose(
        data,
        model=model,
        period=period,
        extrapolate_trend=extrapolate_trend
    )

    return DecompositionResult(
        observed=decomposition.observed,
        trend=decomposition.trend,
        seasonal=decomposition.seasonal,
        residual=decomposition.resid
    )


# ==================== STL DECOMPOSITION ====================

def stl_decompose(
    data: Union[pd.Series, pd.DataFrame],
    period: Optional[int] = None,
    seasonal: int = 7,
    trend: Optional[int] = None,
    robust: bool = False
) -> DecompositionResult:
    """
    STL (Seasonal-Trend decomposition using Loess) decomposition.

    More robust than classical decomposition, handles complex patterns better.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    period : int, optional
        Seasonal period.
    seasonal : int
        Length of seasonal smoother (must be odd).
    trend : int, optional
        Length of trend smoother (must be odd).
    robust : bool
        Use robust fitting (resistant to outliers).

    Returns:
    --------
    DecompositionResult
        Decomposition components.

    Examples:
    ---------
    >>> # Robust STL decomposition
    >>> result = stl_decompose(df['sales'], period=7, robust=True)
    >>> result.plot()
    """
    from statsmodels.tsa.seasonal import STL

    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    # Infer period if not provided
    if period is None:
        from .preprocessing import detect_frequency
        freq = detect_frequency(data)

        if freq == 'D':
            period = 7
        elif freq == 'M':
            period = 12
        elif freq == 'H':
            period = 24
        else:
            raise ValueError("Cannot infer period. Please specify period parameter.")

        print(f"[INFO] Inferred period: {period}")

    # Perform STL decomposition
    stl = STL(
        data,
        period=period,
        seasonal=seasonal,
        trend=trend,
        robust=robust
    )

    result = stl.fit()

    return DecompositionResult(
        observed=data,
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid
    )


# ==================== TREND EXTRACTION ====================

def trend_decompose(
    data: Union[pd.Series, pd.DataFrame],
    method: str = 'ma',
    window: int = 7
) -> pd.Series:
    """
    Extract trend component only.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    method : str
        'ma' (moving average), 'ewma' (exponential), or 'loess'.
    window : int
        Window size for smoothing.

    Returns:
    --------
    pd.Series
        Trend component.

    Examples:
    ---------
    >>> # Moving average trend
    >>> trend = trend_decompose(df['sales'], method='ma', window=30)
    >>>
    >>> # Exponential weighted moving average
    >>> trend = trend_decompose(df['sales'], method='ewma', window=14)
    """
    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    if method == 'ma':
        # Moving average
        trend = data.rolling(window=window, center=True).mean()

    elif method == 'ewma':
        # Exponential weighted moving average
        trend = data.ewm(span=window).mean()

    elif method == 'loess':
        # LOESS smoothing
        from statsmodels.nonparametric.smoothers_lowess import lowess

        # LOESS expects (x, y) format
        x = np.arange(len(data))
        y = data.values

        # Fraction of data for smoothing
        frac = window / len(data)

        smoothed = lowess(y, x, frac=frac, return_sorted=False)
        trend = pd.Series(smoothed, index=data.index, name='trend')

    else:
        raise ValueError(f"Unknown method: {method}")

    return trend


# ==================== DETRENDING / DESEASONALIZING ====================

def detrend(
    data: Union[pd.Series, pd.DataFrame],
    method: str = 'difference',
    window: int = 7
) -> pd.Series:
    """
    Remove trend from time series.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    method : str
        'difference' (subtract previous value) or 'linear' (linear detrend).
    window : int
        For 'ma', window for moving average detrending.

    Returns:
    --------
    pd.Series
        Detrended series.

    Examples:
    ---------
    >>> # First-order differencing
    >>> detrended = detrend(df['sales'], method='difference')
    >>>
    >>> # Linear detrending
    >>> detrended = detrend(df['sales'], method='linear')
    """
    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    if method == 'difference':
        # First-order differencing
        return data.diff()

    elif method == 'linear':
        # Linear detrending
        from scipy import signal
        detrended = signal.detrend(data.values)
        return pd.Series(detrended, index=data.index, name=data.name)

    elif method == 'ma':
        # Subtract moving average
        trend = data.rolling(window=window, center=True).mean()
        return data - trend

    else:
        raise ValueError(f"Unknown method: {method}")


def deseasonalize(
    data: Union[pd.Series, pd.DataFrame],
    period: int,
    model: str = 'additive'
) -> pd.Series:
    """
    Remove seasonal component from time series.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    period : int
        Seasonal period (e.g., 7, 12, 365).
    model : str
        'additive' or 'multiplicative'.

    Returns:
    --------
    pd.Series
        Deseasonalized series.

    Examples:
    ---------
    >>> # Remove weekly seasonality
    >>> deseasoned = deseasonalize(df['sales'], period=7)
    >>>
    >>> # Multiplicative deseasonalization
    >>> deseasoned = deseasonalize(df['sales'], period=12, model='multiplicative')
    """
    # Decompose
    result = seasonal_decompose(data, model=model, period=period)

    # Remove seasonal component
    if model == 'additive':
        deseasonalized = result.observed - result.seasonal
    else:  # multiplicative
        deseasonalized = result.observed / result.seasonal

    return deseasonalized

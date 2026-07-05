"""
Time Series Anomaly Detection Module

Detect outliers and anomalies in time series data.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List
from scipy import stats
import warnings


# ==================== STATISTICAL METHODS ====================

def detect_anomalies_zscore(
    data: Union[pd.Series, pd.DataFrame],
    threshold: float = 3.0,
    window: Optional[int] = None
) -> pd.Series:
    """
    Detect anomalies using Z-score method.

    Points with |z-score| > threshold are flagged as anomalies.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    threshold : float
        Z-score threshold (default: 3.0).
    window : int, optional
        Use rolling window for local anomaly detection.

    Returns:
    --------
    pd.Series
        Boolean mask (True = anomaly).

    Examples:
    ---------
    >>> # Global anomalies
    >>> anomalies = detect_anomalies_zscore(df['sales'], threshold=3)
    >>> print(f"Found {anomalies.sum()} anomalies")
    >>>
    >>> # Local anomalies (rolling window)
    >>> anomalies = detect_anomalies_zscore(df['sales'], threshold=2.5, window=30)
    """
    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    if window is None:
        # Global z-score
        z_scores = np.abs(stats.zscore(data.dropna()))
        anomalies = pd.Series(False, index=data.index)
        anomalies.loc[data.dropna().index] = z_scores > threshold
    else:
        # Rolling z-score
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        anomalies = z_scores > threshold

    print(f"[OK] Detected {anomalies.sum()} anomalies using Z-score (threshold={threshold})")

    return anomalies


def detect_anomalies_iqr(
    data: Union[pd.Series, pd.DataFrame],
    multiplier: float = 1.5,
    window: Optional[int] = None
) -> pd.Series:
    """
    Detect anomalies using IQR (Interquartile Range) method.

    Points outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are anomalies.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    multiplier : float
        IQR multiplier (default: 1.5 for outliers, 3.0 for extreme outliers).
    window : int, optional
        Use rolling window.

    Returns:
    --------
    pd.Series
        Boolean mask (True = anomaly).

    Examples:
    ---------
    >>> # Standard IQR
    >>> anomalies = detect_anomalies_iqr(df['sales'])
    >>>
    >>> # More conservative (extreme outliers only)
    >>> anomalies = detect_anomalies_iqr(df['sales'], multiplier=3.0)
    """
    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    if window is None:
        # Global IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        anomalies = (data < lower_bound) | (data > upper_bound)
    else:
        # Rolling IQR
        Q1 = data.rolling(window=window).quantile(0.25)
        Q3 = data.rolling(window=window).quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        anomalies = (data < lower_bound) | (data > upper_bound)

    print(f"[OK] Detected {anomalies.sum()} anomalies using IQR (multiplier={multiplier})")

    return anomalies


def detect_anomalies_mad(
    data: Union[pd.Series, pd.DataFrame],
    threshold: float = 3.5
) -> pd.Series:
    """
    Detect anomalies using MAD (Median Absolute Deviation).

    More robust to outliers than Z-score.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    threshold : float
        MAD threshold (default: 3.5).

    Returns:
    --------
    pd.Series
        Boolean mask (True = anomaly).

    Examples:
    ---------
    >>> anomalies = detect_anomalies_mad(df['sales'], threshold=3.5)
    """
    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    median = data.median()
    mad = np.median(np.abs(data - median))

    # Modified Z-score
    modified_z_scores = 0.6745 * (data - median) / mad

    anomalies = np.abs(modified_z_scores) > threshold

    print(f"[OK] Detected {anomalies.sum()} anomalies using MAD (threshold={threshold})")

    return anomalies


# ==================== MACHINE LEARNING METHODS ====================

def detect_anomalies_isolation_forest(
    data: Union[pd.Series, pd.DataFrame],
    contamination: float = 0.1,
    n_estimators: int = 100,
    add_features: bool = True
) -> pd.Series:
    """
    Detect anomalies using Isolation Forest (ML method).

    Good for complex, multivariate patterns.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    contamination : float
        Expected proportion of anomalies (0.1 = 10%).
    n_estimators : int
        Number of trees in forest.
    add_features : bool
        Add lag/rolling features automatically.

    Returns:
    --------
    pd.Series
        Boolean mask (True = anomaly).

    Examples:
    ---------
    >>> # Basic usage
    >>> anomalies = detect_anomalies_isolation_forest(df['sales'], contamination=0.05)
    >>>
    >>> # With feature engineering
    >>> anomalies = detect_anomalies_isolation_forest(
    ...     df['sales'],
    ...     contamination=0.1,
    ...     add_features=True
    ... )
    """
    from sklearn.ensemble import IsolationForest

    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    # Create features
    if add_features:
        from .features import create_lag_features, create_rolling_features

        df = data.to_frame()
        df = create_lag_features(df, lags=[1, 7], dropna=False)
        df = create_rolling_features(df, windows=[7], functions=['mean', 'std'], columns=[data.name])
        df = df.dropna()

        X = df.drop(columns=[data.name])
    else:
        X = data.values.reshape(-1, 1)

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42
    )

    predictions = iso_forest.fit_predict(X)

    # -1 = anomaly, 1 = normal
    anomalies = pd.Series(False, index=data.index)

    if add_features:
        anomalies.loc[df.index] = (predictions == -1)
    else:
        anomalies = pd.Series(predictions == -1, index=data.index)

    print(f"[OK] Detected {anomalies.sum()} anomalies using Isolation Forest")

    return anomalies


def detect_anomalies_lof(
    data: Union[pd.Series, pd.DataFrame],
    n_neighbors: int = 20,
    contamination: float = 0.1
) -> pd.Series:
    """
    Detect anomalies using LOF (Local Outlier Factor).

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    n_neighbors : int
        Number of neighbors to consider.
    contamination : float
        Expected proportion of anomalies.

    Returns:
    --------
    pd.Series
        Boolean mask (True = anomaly).

    Examples:
    ---------
    >>> anomalies = detect_anomalies_lof(df['sales'], n_neighbors=20)
    """
    from sklearn.neighbors import LocalOutlierFactor

    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    X = data.values.reshape(-1, 1)

    # Fit LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = lof.fit_predict(X)

    anomalies = pd.Series(predictions == -1, index=data.index)

    print(f"[OK] Detected {anomalies.sum()} anomalies using LOF")

    return anomalies


# ==================== PROPHET-BASED ANOMALY DETECTION ====================

def detect_anomalies_prophet(
    data: Union[pd.Series, pd.DataFrame],
    threshold: float = 0.95,
    freq: str = 'D'
) -> pd.Series:
    """
    Detect anomalies using Facebook Prophet.

    Points outside prediction intervals are flagged as anomalies.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data with DatetimeIndex.
    threshold : float
        Confidence interval threshold (0.95 = 95% CI).
    freq : str
        Frequency ('D' for daily, 'H' for hourly, etc.).

    Returns:
    --------
    pd.Series
        Boolean mask (True = anomaly).

    Examples:
    ---------
    >>> # Prophet-based detection
    >>> anomalies = detect_anomalies_prophet(df['sales'], threshold=0.99)
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("prophet is required. Install with: pip install prophet")

    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have single column")

    # Prepare data for Prophet
    df_prophet = pd.DataFrame({
        'ds': data.index,
        'y': data.values
    })

    # Fit Prophet
    model = Prophet(interval_width=threshold)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    # Predict
    forecast = model.predict(df_prophet)

    # Find anomalies (outside prediction interval)
    anomalies = (
        (data.values < forecast['yhat_lower'].values) |
        (data.values > forecast['yhat_upper'].values)
    )

    anomalies = pd.Series(anomalies, index=data.index)

    print(f"[OK] Detected {anomalies.sum()} anomalies using Prophet")

    return anomalies


# ==================== COMBINED DETECTION ====================

def detect_anomalies_ensemble(
    data: Union[pd.Series, pd.DataFrame],
    methods: List[str] = ['zscore', 'iqr', 'isolation_forest'],
    min_votes: int = 2
) -> pd.Series:
    """
    Ensemble anomaly detection (voting from multiple methods).

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    methods : list
        Methods to use: 'zscore', 'iqr', 'mad', 'isolation_forest', 'lof'.
    min_votes : int
        Minimum votes needed to flag as anomaly.

    Returns:
    --------
    pd.Series
        Boolean mask (True = anomaly).

    Examples:
    ---------
    >>> # Ensemble with 3 methods, need 2 votes
    >>> anomalies = detect_anomalies_ensemble(
    ...     df['sales'],
    ...     methods=['zscore', 'iqr', 'isolation_forest'],
    ...     min_votes=2
    ... )
    """
    votes = pd.DataFrame(index=data.index)

    for method in methods:
        if method == 'zscore':
            votes[method] = detect_anomalies_zscore(data)
        elif method == 'iqr':
            votes[method] = detect_anomalies_iqr(data)
        elif method == 'mad':
            votes[method] = detect_anomalies_mad(data)
        elif method == 'isolation_forest':
            votes[method] = detect_anomalies_isolation_forest(data)
        elif method == 'lof':
            votes[method] = detect_anomalies_lof(data)
        else:
            warnings.warn(f"Unknown method: {method}, skipping")

    # Count votes
    vote_count = votes.sum(axis=1)
    anomalies = vote_count >= min_votes

    print(f"\n[OK] Ensemble detected {anomalies.sum()} anomalies")
    print(f"  Methods: {methods}")
    print(f"  Min votes: {min_votes}")

    return anomalies

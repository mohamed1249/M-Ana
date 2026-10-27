"""
Time Series Preprocessing Module

Functions for preparing and validating time series data:
- Data validation and quality checks
- DateTime index creation and fixing
- Missing timestamp handling
- Frequency detection and resampling
- Stationarity testing
- Data splitting for time series
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from datetime import timedelta
import warnings
from scipy import stats


# ==================== DATA VALIDATION ====================

def validate_timeseries(
    data: Union[pd.Series, pd.DataFrame],
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    freq: Optional[str] = None,
    allow_duplicates: bool = False,
    allow_missing: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate time series data and return quality report.

    Checks:
    - DateTime index presence
    - Sorted order
    - Duplicate timestamps
    - Missing values
    - Data type consistency
    - Frequency regularity

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    date_column : str, optional
        Date column name (if DataFrame without datetime index).
    value_column : str, optional
        Value column name (if DataFrame).
    freq : str, optional
        Expected frequency (e.g., 'D', 'H', 'M').
    allow_duplicates : bool
        Allow duplicate timestamps.
    allow_missing : bool
        Allow missing values.
    verbose : bool
        Print validation report.

    Returns:
    --------
    dict
        Validation report with issues found.

    Examples:
    ---------
    >>> # Validate a time series
    >>> report = validate_timeseries(df['sales'], verbose=True)
    >>>
    >>> if report['is_valid']:
    ...     print("✓ Data is valid!")
    ... else:
    ...     print(f"⚠️  Found {len(report['issues'])} issues")
    """
    issues = []
    warnings_list = []

    # Convert to Series if needed
    if isinstance(data, pd.DataFrame):
        if value_column is None and len(data.columns) == 1:
            value_column = data.columns[0]

        if value_column:
            series = data[value_column]
        else:
            raise ValueError("For DataFrame input, specify value_column")
    else:
        series = data

    # Check 1: DateTime index
    if not isinstance(series.index, pd.DatetimeIndex):
        if date_column and isinstance(data, pd.DataFrame):
            issues.append("DateTime index not set. Use create_datetime_index() to fix.")
        else:
            issues.append("Index is not DatetimeIndex. Convert with pd.to_datetime().")

    # Check 2: Sorted order
    if isinstance(series.index, pd.DatetimeIndex):
        if not series.index.is_monotonic_increasing:
            issues.append("Index is not sorted. Sort with .sort_index().")

    # Check 3: Duplicate timestamps
    if isinstance(series.index, pd.DatetimeIndex):
        duplicates = series.index.duplicated().sum()
        if duplicates > 0:
            if not allow_duplicates:
                issues.append(f"Found {duplicates} duplicate timestamps.")
            else:
                warnings_list.append(f"Found {duplicates} duplicate timestamps (allowed).")

    # Check 4: Missing values
    missing_count = series.isna().sum()
    missing_pct = (missing_count / len(series)) * 100
    if missing_count > 0:
        if not allow_missing:
            issues.append(f"Found {missing_count} missing values ({missing_pct:.1f}%).")
        else:
            warnings_list.append(f"Found {missing_count} missing values ({missing_pct:.1f}%).")

    # Check 5: Data type
    if not pd.api.types.is_numeric_dtype(series):
        issues.append(f"Values are not numeric (dtype: {series.dtype}).")

    # Check 6: Frequency regularity
    if isinstance(series.index, pd.DatetimeIndex) and len(series) > 2:
        inferred_freq = pd.infer_freq(series.index)

        if inferred_freq is None:
            warnings_list.append("Could not infer frequency. Timestamps may be irregular.")
        elif freq and inferred_freq != freq:
            warnings_list.append(f"Inferred frequency '{inferred_freq}' differs from expected '{freq}'.")

    # Check 7: Constant values
    if series.nunique() == 1:
        warnings_list.append("All values are constant (no variance).")

    # Check 8: Outliers (simple check)
    if pd.api.types.is_numeric_dtype(series):
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            warnings_list.append(f"Found {outliers} potential outliers (|z-score| > 3).")

    # Build report
    report = {
        'is_valid': len(issues) == 0,
        'n_records': len(series),
        'n_missing': missing_count,
        'missing_pct': missing_pct,
        'n_duplicates': duplicates if isinstance(series.index, pd.DatetimeIndex) else None,
        'dtype': str(series.dtype),
        'inferred_freq': inferred_freq if isinstance(series.index, pd.DatetimeIndex) else None,
        'date_range': (series.index.min(), series.index.max()) if isinstance(series.index, pd.DatetimeIndex) else None,
        'issues': issues,
        'warnings': warnings_list
    }

    # Print report
    if verbose:
        print("=" * 70)
        print("TIME SERIES VALIDATION REPORT")
        print("=" * 70)
        print("\n📊 Dataset Overview:")
        print(f"  Records: {report['n_records']:,}")
        if report['date_range']:
            print(f"  Date Range: {report['date_range'][0]} to {report['date_range'][1]}")
        print(f"  Data Type: {report['dtype']}")
        if report['inferred_freq']:
            print(f"  Frequency: {report['inferred_freq']}")

        print("\n📈 Data Quality:")
        print(f"  Missing Values: {report['n_missing']:,} ({report['missing_pct']:.1f}%)")
        if report['n_duplicates'] is not None:
            print(f"  Duplicate Timestamps: {report['n_duplicates']:,}")

        if report['is_valid']:
            print("\n[OK] VALIDATION PASSED")
        else:
            print("\n✗ VALIDATION FAILED")

        if issues:
            print(f"\n[WARN] Issues Found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        if warnings_list:
            print(f"\n⚡ Warnings ({len(warnings_list)}):")
            for i, warning in enumerate(warnings_list, 1):
                print(f"  {i}. {warning}")

        print("=" * 70)

    return report


# ==================== DATETIME INDEX CREATION ====================

def create_datetime_index(
    data: pd.DataFrame,
    date_column: str,
    format: Optional[str] = None,
    drop_date_column: bool = True,
    sort: bool = True
) -> pd.DataFrame:
    """
    Convert a date column to DatetimeIndex.

    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame.
    date_column : str
        Name of date column.
    format : str, optional
        DateTime format string (e.g., '%Y-%m-%d').
    drop_date_column : bool
        Drop the original date column after conversion.
    sort : bool
        Sort by datetime index.

    Returns:
    --------
    pd.DataFrame
        DataFrame with DatetimeIndex.

    Examples:
    ---------
    >>> # Convert 'date' column to index
    >>> df = create_datetime_index(df, 'date')
    >>>
    >>> # With specific format
    >>> df = create_datetime_index(df, 'timestamp', format='%Y-%m-%d %H:%M:%S')
    """
    df = data.copy()

    # Convert to datetime
    if format:
        df[date_column] = pd.to_datetime(df[date_column], format=format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])

    # Set as index
    df = df.set_index(date_column)

    # Sort
    if sort:
        df = df.sort_index()

    # Drop original column (it's now the index)
    # Note: drop_date_column parameter is misleading here since it's already the index
    # Keeping for backward compatibility

    print(f"[OK] Created DatetimeIndex from '{date_column}'")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    print(f"  Records: {len(df):,}")

    return df


# ==================== FREQUENCY DETECTION ====================

def detect_frequency(
    data: Union[pd.Series, pd.DataFrame],
    return_timedelta: bool = False
) -> Union[str, timedelta, None]:
    """
    Detect the frequency of time series data.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex.
    return_timedelta : bool
        Return timedelta instead of frequency string.

    Returns:
    --------
    str, timedelta, or None
        Detected frequency or None if irregular.

    Examples:
    ---------
    >>> freq = detect_frequency(df)
    >>> print(f"Detected frequency: {freq}")  # 'D' for daily, 'H' for hourly, etc.
    >>>
    >>> # Get as timedelta
    >>> td = detect_frequency(df, return_timedelta=True)
    >>> print(f"Time between observations: {td}")
    """
    if isinstance(data, pd.DataFrame):
        index = data.index
    else:
        index = data.index

    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    if len(index) < 3:
        warnings.warn("Need at least 3 observations to reliably detect frequency")
        return None

    # Try pandas infer_freq
    freq = pd.infer_freq(index)

    if freq is None:
        # Calculate most common time difference
        diffs = index.to_series().diff().dropna()
        most_common_diff = diffs.mode()[0] if len(diffs) > 0 else None

        if most_common_diff and return_timedelta:
            return most_common_diff
        elif most_common_diff:
            # Try to map to frequency string
            if most_common_diff == timedelta(days=1):
                return 'D'
            elif most_common_diff == timedelta(hours=1):
                return 'H'
            elif most_common_diff == timedelta(minutes=1):
                return 'T'
            elif most_common_diff == timedelta(seconds=1):
                return 'S'
            else:
                return None
        return None

    if return_timedelta:
        # Convert fixed frequency strings to Timedelta without relying on the
        # deprecated DateOffset.delta attribute.
        offset = pd.tseries.frequencies.to_offset(freq)
        try:
            return pd.Timedelta(offset.nanos, unit="ns")
        except ValueError:
            # Calendar-dependent offsets such as months do not have one fixed
            # Timedelta representation.
            return None

    return freq


# ==================== RESAMPLING ====================

def resample_timeseries(
    data: Union[pd.Series, pd.DataFrame],
    freq: str,
    agg_func: Union[str, Dict] = 'mean',
    fill_method: Optional[str] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Resample time series to different frequency.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data with DatetimeIndex.
    freq : str
        Target frequency ('D', 'W', 'M', 'Q', 'Y', 'H', 'T', 'S').
        D: Daily, W: Weekly, M: Monthly, Q: Quarterly, Y: Yearly
        H: Hourly, T: Minutely, S: Secondly
    agg_func : str or dict
        Aggregation function(s): 'mean', 'sum', 'min', 'max', 'first', 'last'
        For DataFrame, can be dict mapping columns to functions.
    fill_method : str, optional
        Method to fill missing values: 'ffill', 'bfill', 'interpolate'.

    Returns:
    --------
    pd.Series or pd.DataFrame
        Resampled time series.

    Examples:
    ---------
    >>> # Resample hourly to daily (mean)
    >>> daily = resample_timeseries(hourly_data, 'D', agg_func='mean')
    >>>
    >>> # Resample daily to weekly (sum)
    >>> weekly = resample_timeseries(daily_data, 'W', agg_func='sum')
    >>>
    >>> # Different aggregations per column
    >>> df_resampled = resample_timeseries(
    ...     df, 'M',
    ...     agg_func={'sales': 'sum', 'customers': 'mean'}
    ... )
    >>>
    >>> # With forward fill
    >>> monthly = resample_timeseries(daily_data, 'M', agg_func='mean', fill_method='ffill')
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Resample
    resampled = data.resample(freq)

    # Aggregate
    if isinstance(agg_func, str):
        if agg_func == 'mean':
            result = resampled.mean()
        elif agg_func == 'sum':
            result = resampled.sum()
        elif agg_func == 'min':
            result = resampled.min()
        elif agg_func == 'max':
            result = resampled.max()
        elif agg_func == 'first':
            result = resampled.first()
        elif agg_func == 'last':
            result = resampled.last()
        elif agg_func == 'median':
            result = resampled.median()
        elif agg_func == 'count':
            result = resampled.count()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
    else:
        # Dict of functions
        result = resampled.agg(agg_func)

    # Fill missing values if requested
    if fill_method:
        if fill_method == 'ffill':
            result = result.ffill()
        elif fill_method == 'bfill':
            result = result.bfill()
        elif fill_method == 'interpolate':
            result = result.interpolate()
        else:
            raise ValueError(f"Unknown fill method: {fill_method}")

    print(f"[OK] Resampled from {len(data):,} to {len(result):,} records (frequency: {freq})")

    return result


# ==================== MISSING TIMESTAMPS ====================

def fill_missing_timestamps(
    data: Union[pd.Series, pd.DataFrame],
    freq: Optional[str] = None,
    method: str = 'ffill',
    limit: Optional[int] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Fill missing timestamps to create regular time series.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex.
    freq : str, optional
        Frequency to use. If None, inferred from data.
    method : str
        Fill method: 'ffill', 'bfill', 'interpolate', 'zero', 'mean'.
    limit : int, optional
        Maximum number of consecutive NaNs to fill.

    Returns:
    --------
    pd.Series or pd.DataFrame
        Time series with all timestamps filled.

    Examples:
    ---------
    >>> # Fill missing days with forward fill
    >>> df_complete = fill_missing_timestamps(df, freq='D', method='ffill')
    >>>
    >>> # Fill with interpolation
    >>> df_complete = fill_missing_timestamps(df, method='interpolate')
    >>>
    >>> # Fill with zeros
    >>> df_complete = fill_missing_timestamps(df, method='zero')
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Detect frequency if not provided
    if freq is None:
        freq = detect_frequency(data)
        if freq is None:
            raise ValueError("Could not detect frequency. Please specify freq parameter.")
        print(f"ℹ️  Using detected frequency: {freq}")

    # Create complete date range
    date_range = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=freq
    )

    # Reindex to include all dates
    if isinstance(data, pd.Series):
        result = data.reindex(date_range)
    else:
        result = data.reindex(date_range)

    # Count missing timestamps
    missing_count = result.isna().sum()
    if isinstance(missing_count, pd.Series):
        missing_count = missing_count.iloc[0] if len(missing_count) > 0 else 0

    # Fill missing values
    if method == 'ffill':
        result = result.ffill(limit=limit)
    elif method == 'bfill':
        result = result.bfill(limit=limit)
    elif method == 'interpolate':
        result = result.interpolate(limit=limit)
    elif method == 'zero':
        result = result.fillna(0)
    elif method == 'mean':
        if isinstance(result, pd.Series):
            result = result.fillna(result.mean())
        else:
            result = result.fillna(result.mean())
    else:
        raise ValueError(f"Unknown fill method: {method}")

    added_timestamps = len(result) - len(data)
    print(f"[OK] Added {added_timestamps} missing timestamps")
    print(f"  Original: {len(data):,} records")
    print(f"  Complete: {len(result):,} records")

    return result


# ==================== STATIONARITY TESTING ====================

def test_stationarity(
    data: Union[pd.Series, np.ndarray],
    test: str = 'adf',
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test if time series is stationary.

    Stationarity means:
    - Constant mean over time
    - Constant variance over time
    - No seasonality

    Parameters:
    -----------
    data : pd.Series or array
        Time series data.
    test : str
        Statistical test: 'adf' (Augmented Dickey-Fuller) or 'kpss'.
    alpha : float
        Significance level (default: 0.05).
    verbose : bool
        Print test results.

    Returns:
    --------
    dict
        Test results including statistic, p-value, and conclusion.

    Examples:
    ---------
    >>> # Test for stationarity
    >>> result = test_stationarity(df['sales'])
    >>>
    >>> if result['is_stationary']:
    ...     print("✓ Data is stationary")
    ... else:
    ...     print("⚠️  Data is non-stationary, apply differencing")
    """
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import adfuller, kpss

    if isinstance(data, pd.Series):
        data = data.dropna().values

    if test.lower() == 'adf':
        # Augmented Dickey-Fuller test
        # H0: Series has a unit root (non-stationary)
        # H1: Series is stationary
        result = adfuller(data, autolag='AIC')

        test_stat = result[0]
        p_value = result[1]
        critical_values = result[4]

        # Reject H0 if p-value < alpha (series is stationary)
        is_stationary = p_value < alpha

        if verbose:
            print("=" * 70)
            print("AUGMENTED DICKEY-FULLER TEST")
            print("=" * 70)
            print("\nNull Hypothesis: Series has a unit root (non-stationary)")
            print("Alternative: Series is stationary")
            print(f"\nTest Statistic: {test_stat:.4f}")
            print(f"P-value: {p_value:.6f}")
            print("\nCritical Values:")
            for key, value in critical_values.items():
                print(f"  {key}: {value:.4f}")

            print(f"\n{'='*70}")
            if is_stationary:
                print("[OK] CONCLUSION: Series is STATIONARY")
                print(f"  (p-value {p_value:.4f} < {alpha})")
            else:
                print("[WARN] CONCLUSION: Series is NON-STATIONARY")
                print(f"  (p-value {p_value:.4f} >= {alpha})")
                print("\n  Suggestion: Try differencing the series")
            print("=" * 70)

        return {
            'test': 'ADF',
            'statistic': test_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'conclusion': 'stationary' if is_stationary else 'non-stationary'
        }

    elif test.lower() == 'kpss':
        # KPSS test
        # H0: Series is stationary
        # H1: Series has a unit root (non-stationary)
        # statsmodels raises InterpolationWarning when the statistic sits
        # outside its lookup table. The returned p-value is still the documented
        # bounded value, so suppress the expected warning for cleaner notebooks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            result = kpss(data, regression='c', nlags='auto')

        test_stat = result[0]
        p_value = result[1]
        critical_values = result[3]

        # Reject H0 if p-value < alpha (series is non-stationary)
        is_stationary = p_value >= alpha

        if verbose:
            print("=" * 70)
            print("KPSS TEST")
            print("=" * 70)
            print("\nNull Hypothesis: Series is stationary")
            print("Alternative: Series has a unit root (non-stationary)")
            print(f"\nTest Statistic: {test_stat:.4f}")
            print(f"P-value: {p_value:.6f}")
            print("\nCritical Values:")
            for key, value in critical_values.items():
                print(f"  {key}: {value:.4f}")

            print(f"\n{'='*70}")
            if is_stationary:
                print("[OK] CONCLUSION: Series is STATIONARY")
                print(f"  (p-value {p_value:.4f} >= {alpha})")
            else:
                print("[WARN] CONCLUSION: Series is NON-STATIONARY")
                print(f"  (p-value {p_value:.4f} < {alpha})")
                print("\n  Suggestion: Try differencing the series")
            print("=" * 70)

        return {
            'test': 'KPSS',
            'statistic': test_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'conclusion': 'stationary' if is_stationary else 'non-stationary'
        }

    else:
        raise ValueError(f"Unknown test: {test}. Use 'adf' or 'kpss'.")


# ==================== DIFFERENCING ====================

def difference_series(
    data: Union[pd.Series, pd.DataFrame],
    periods: int = 1,
    seasonal_periods: Optional[int] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Apply differencing to make series stationary.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    periods : int
        Number of periods to difference (default: 1).
    seasonal_periods : int, optional
        Apply seasonal differencing at this period.

    Returns:
    --------
    pd.Series or pd.DataFrame
        Differenced series.

    Examples:
    ---------
    >>> # First-order differencing
    >>> df_diff = difference_series(df['sales'], periods=1)
    >>>
    >>> # Second-order differencing
    >>> df_diff2 = difference_series(df['sales'], periods=2)
    >>>
    >>> # Seasonal differencing (weekly)
    >>> df_seasonal = difference_series(df['sales'], seasonal_periods=7)
    """
    # Regular differencing
    result = data.diff(periods=periods)

    # Seasonal differencing
    if seasonal_periods:
        result = result.diff(periods=seasonal_periods)

    # Drop NaN values created by differencing
    result = result.dropna()

    print("[OK] Applied differencing")
    print(f"  Original length: {len(data):,}")
    print(f"  Differenced length: {len(result):,}")

    return result


# ==================== TRAIN/TEST SPLIT ====================

def train_test_split_ts(
    data: Union[pd.Series, pd.DataFrame],
    test_size: Union[int, float] = 0.2,
    gap: int = 0
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """
    Split time series into train and test sets.

    Unlike regular train_test_split, this maintains temporal order
    and optionally adds a gap between train and test.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    test_size : int or float
        If float: proportion of data for test (e.g., 0.2 = 20%).
        If int: number of observations for test.
    gap : int
        Number of observations to skip between train and test.

    Returns:
    --------
    tuple
        (train, test) datasets.

    Examples:
    ---------
    >>> # 80/20 split
    >>> train, test = train_test_split_ts(df, test_size=0.2)
    >>>
    >>> # Last 30 days for testing
    >>> train, test = train_test_split_ts(df, test_size=30)
    >>>
    >>> # With 7-day gap
    >>> train, test = train_test_split_ts(df, test_size=0.2, gap=7)
    """
    n = len(data)

    # Calculate split point
    if isinstance(test_size, float):
        test_n = int(n * test_size)
    else:
        test_n = test_size

    train_n = n - test_n - gap

    if train_n <= 0:
        raise ValueError(f"Not enough data. Need at least {test_n + gap + 1} observations.")

    # Split
    train = data.iloc[:train_n]
    test = data.iloc[train_n + gap:]

    print("[OK] Train/Test Split")
    print(f"  Train: {len(train):,} observations ({len(train)/n*100:.1f}%)")
    if gap > 0:
        print(f"  Gap: {gap} observations")
    print(f"  Test: {len(test):,} observations ({len(test)/n*100:.1f}%)")

    return train, test


# ==================== HELPER FUNCTIONS ====================

def remove_outliers_ts(
    data: Union[pd.Series, pd.DataFrame],
    method: str = 'zscore',
    threshold: float = 3.0,
    replace_with: str = 'interpolate'
) -> Union[pd.Series, pd.DataFrame]:
    """
    Remove or replace outliers in time series.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    method : str
        Outlier detection: 'zscore', 'iqr', or 'mad'.
    threshold : float
        Threshold for outlier detection.
    replace_with : str
        How to handle outliers: 'interpolate', 'mean', 'median', 'remove'.

    Returns:
    --------
    pd.Series or pd.DataFrame
        Data with outliers handled.

    Examples:
    ---------
    >>> # Remove outliers using z-score
    >>> df_clean = remove_outliers_ts(df['sales'], method='zscore', threshold=3)
    >>>
    >>> # Replace with interpolation
    >>> df_clean = remove_outliers_ts(df, method='iqr', replace_with='interpolate')
    """
    result = data.copy()

    if method == 'zscore':
        # Z-score method
        if isinstance(result, pd.Series):
            z_scores = np.abs(stats.zscore(result.dropna()))
            outliers = z_scores > threshold
            outlier_idx = result.dropna().index[outliers]
        else:
            outlier_idx = []
            for col in result.columns:
                z_scores = np.abs(stats.zscore(result[col].dropna()))
                outliers = z_scores > threshold
                outlier_idx.extend(result[col].dropna().index[outliers].tolist())
            outlier_idx = list(set(outlier_idx))

    elif method == 'iqr':
        # IQR method
        if isinstance(result, pd.Series):
            Q1 = result.quantile(0.25)
            Q3 = result.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (result < (Q1 - threshold * IQR)) | (result > (Q3 + threshold * IQR))
            outlier_idx = result[outliers].index
        else:
            outlier_idx = []
            for col in result.columns:
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (result[col] < (Q1 - threshold * IQR)) | (result[col] > (Q3 + threshold * IQR))
                outlier_idx.extend(result[col][outliers].index.tolist())
            outlier_idx = list(set(outlier_idx))

    elif method == 'mad':
        # Median Absolute Deviation method. This is robust when a few extreme
        # spikes would inflate the standard deviation and weaken z-score rules.
        def mad_outlier_index(series):
            clean = series.dropna()
            median = clean.median()
            mad = np.median(np.abs(clean - median))
            if mad == 0:
                return clean.index[clean != median]
            modified_z = 0.6745 * (clean - median) / mad
            return clean.index[np.abs(modified_z) > threshold]

        if isinstance(result, pd.Series):
            outlier_idx = mad_outlier_index(result)
        else:
            outlier_idx = []
            for col in result.columns:
                outlier_idx.extend(mad_outlier_index(result[col]).tolist())
            outlier_idx = list(set(outlier_idx))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle outliers
    if replace_with == 'remove':
        result = result.drop(outlier_idx)
    elif replace_with == 'interpolate':
        result.loc[outlier_idx] = np.nan
        result = result.interpolate()
    elif replace_with == 'mean':
        if isinstance(result, pd.Series):
            result.loc[outlier_idx] = result.mean()
        else:
            for col in result.columns:
                col_outliers = [idx for idx in outlier_idx if idx in result[col].index]
                result.loc[col_outliers, col] = result[col].mean()
    elif replace_with == 'median':
        if isinstance(result, pd.Series):
            result.loc[outlier_idx] = result.median()
        else:
            for col in result.columns:
                col_outliers = [idx for idx in outlier_idx if idx in result[col].index]
                result.loc[col_outliers, col] = result[col].median()

    print(f"[OK] Outlier handling ({method} method)")
    print(f"  Found: {len(outlier_idx)} outliers")
    print(f"  Action: {replace_with}")

    return result

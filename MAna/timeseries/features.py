"""
Time Series Feature Engineering Module

Create powerful features from time series data:
- Lag features (historical values)
- Rolling window statistics (moving averages, std, etc.)
- Date/time features (day of week, month, quarter, holidays)
- Seasonal features (cyclical patterns)
- Fourier features (complex seasonality)
- Calendar effects (holidays, weekends, business days)
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional
import warnings


# ==================== LAG FEATURES ====================

def create_lag_features(
    data: Union[pd.Series, pd.DataFrame],
    lags: Union[int, List[int]],
    columns: Optional[List[str]] = None,
    dropna: bool = False,
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Create lag features (previous time period values).

    Lag features are crucial for time series forecasting as they
    capture temporal dependencies.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data with DatetimeIndex.
    lags : int or list
        Lag periods to create. If int, creates lags 1 to n.
        If list, creates specific lags (e.g., [1, 7, 30]).
    columns : list, optional
        Specific columns to create lags for (DataFrame only).
    dropna : bool
        Drop rows with NaN values created by lagging.
    fill_value : float, optional
        Value to fill NaN created by lagging.

    Returns:
    --------
    pd.DataFrame
        DataFrame with original data and lag features.

    Examples:
    ---------
    >>> # Create lags 1, 2, 3
    >>> df_lagged = create_lag_features(df['sales'], lags=3)
    >>>
    >>> # Create specific lags (1 day, 1 week, 1 month ago)
    >>> df_lagged = create_lag_features(df['sales'], lags=[1, 7, 30])
    >>>
    >>> # Multiple columns
    >>> df_lagged = create_lag_features(df, lags=[1, 7], columns=['sales', 'customers'])
    >>>
    >>> # Fill NaN with 0
    >>> df_lagged = create_lag_features(df['sales'], lags=7, fill_value=0)
    """
    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
        col_name = data.name or 'value'
        df.columns = [col_name]
        columns = [col_name]
    else:
        df = data.copy()
        if columns is None:
            columns = df.columns.tolist()

    # Determine lag list
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = sorted(lags)

    # Create lag features
    for col in columns:
        for lag in lag_list:
            lag_col_name = f'{col}_lag_{lag}'
            df[lag_col_name] = df[col].shift(lag)

    # Handle NaN values
    if fill_value is not None:
        df = df.fillna(fill_value)
    elif dropna:
        df = df.dropna()

    n_features = len(columns) * len(lag_list)
    print(f"✓ Created {n_features} lag features")
    print(f"  Columns: {columns}")
    print(f"  Lags: {lag_list}")

    return df


# ==================== ROLLING WINDOW FEATURES ====================

def create_rolling_features(
    data: Union[pd.Series, pd.DataFrame],
    windows: Union[int, List[int]],
    functions: Union[str, List[str]] = 'mean',
    columns: Optional[List[str]] = None,
    min_periods: Optional[int] = None,
    center: bool = False
) -> pd.DataFrame:
    """
    Create rolling window statistical features.

    Rolling features capture recent trends and volatility.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    windows : int or list
        Window sizes (e.g., [7, 14, 30] for 7-day, 14-day, 30-day windows).
    functions : str or list
        Statistics to compute: 'mean', 'std', 'min', 'max', 'sum', 'median', 'var'.
    columns : list, optional
        Specific columns to create rolling features for.
    min_periods : int, optional
        Minimum observations in window to have a value.
    center : bool
        Center the window (default: False, trailing window).

    Returns:
    --------
    pd.DataFrame
        DataFrame with original data and rolling features.

    Examples:
    ---------
    >>> # 7-day moving average
    >>> df_rolling = create_rolling_features(df['sales'], windows=7, functions='mean')
    >>>
    >>> # Multiple windows and functions
    >>> df_rolling = create_rolling_features(
    ...     df['sales'],
    ...     windows=[7, 14, 30],
    ...     functions=['mean', 'std', 'min', 'max']
    ... )
    >>>
    >>> # Multiple columns
    >>> df_rolling = create_rolling_features(
    ...     df,
    ...     windows=[7, 30],
    ...     functions=['mean', 'std'],
    ...     columns=['sales', 'customers']
    ... )
    """
    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
        col_name = data.name or 'value'
        df.columns = [col_name]
        columns = [col_name]
    else:
        df = data.copy()
        if columns is None:
            columns = df.columns.tolist()

    # Ensure lists
    if isinstance(windows, int):
        window_list = [windows]
    else:
        window_list = windows

    if isinstance(functions, str):
        func_list = [functions]
    else:
        func_list = functions

    # Create rolling features
    for col in columns:
        for window in window_list:
            for func in func_list:
                feature_name = f'{col}_rolling_{window}_{func}'

                if func == 'mean':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).mean()
                elif func == 'std':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).std()
                elif func == 'min':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).min()
                elif func == 'max':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).max()
                elif func == 'sum':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).sum()
                elif func == 'median':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).median()
                elif func == 'var':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).var()
                elif func == 'skew':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).skew()
                elif func == 'kurt':
                    df[feature_name] = df[col].rolling(window=window, min_periods=min_periods, center=center).kurt()
                else:
                    warnings.warn(f"Unknown function: {func}, skipping")

    n_features = len(columns) * len(window_list) * len(func_list)
    print(f"✓ Created {n_features} rolling window features")
    print(f"  Windows: {window_list}")
    print(f"  Functions: {func_list}")

    return df


# ==================== EXPANDING WINDOW FEATURES ====================

def create_expanding_features(
    data: Union[pd.Series, pd.DataFrame],
    functions: Union[str, List[str]] = 'mean',
    columns: Optional[List[str]] = None,
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Create expanding window features (cumulative statistics).

    Expanding windows grow from the start, useful for cumulative metrics.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    functions : str or list
        Statistics: 'mean', 'std', 'min', 'max', 'sum', 'count'.
    columns : list, optional
        Specific columns to process.
    min_periods : int
        Minimum observations required.

    Returns:
    --------
    pd.DataFrame
        DataFrame with expanding window features.

    Examples:
    ---------
    >>> # Cumulative mean
    >>> df_expanding = create_expanding_features(df['sales'], functions='mean')
    >>>
    >>> # Multiple cumulative stats
    >>> df_expanding = create_expanding_features(
    ...     df['sales'],
    ...     functions=['mean', 'std', 'sum', 'count']
    ... )
    """
    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
        col_name = data.name or 'value'
        df.columns = [col_name]
        columns = [col_name]
    else:
        df = data.copy()
        if columns is None:
            columns = df.columns.tolist()

    # Ensure list
    if isinstance(functions, str):
        func_list = [functions]
    else:
        func_list = functions

    # Create expanding features
    for col in columns:
        for func in func_list:
            feature_name = f'{col}_expanding_{func}'

            if func == 'mean':
                df[feature_name] = df[col].expanding(min_periods=min_periods).mean()
            elif func == 'std':
                df[feature_name] = df[col].expanding(min_periods=min_periods).std()
            elif func == 'min':
                df[feature_name] = df[col].expanding(min_periods=min_periods).min()
            elif func == 'max':
                df[feature_name] = df[col].expanding(min_periods=min_periods).max()
            elif func == 'sum':
                df[feature_name] = df[col].expanding(min_periods=min_periods).sum()
            elif func == 'count':
                df[feature_name] = df[col].expanding(min_periods=min_periods).count()
            else:
                warnings.warn(f"Unknown function: {func}, skipping")

    n_features = len(columns) * len(func_list)
    print(f"✓ Created {n_features} expanding window features")

    return df


# ==================== DATE/TIME FEATURES ====================

def create_date_features(
    data: Union[pd.Series, pd.DataFrame],
    features: Optional[List[str]] = None,
    cyclical: bool = False
) -> pd.DataFrame:
    """
    Extract date/time features from DatetimeIndex.

    Captures temporal patterns like day of week, month, seasonality.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex.
    features : list, optional
        Features to extract. If None, creates common features.
        Options: 'year', 'month', 'day', 'dayofweek', 'dayofyear',
                'week', 'quarter', 'hour', 'minute', 'is_weekend',
                'is_month_start', 'is_month_end', 'is_quarter_start',
                'is_quarter_end', 'is_year_start', 'is_year_end'
    cyclical : bool
        Encode cyclical features (month, hour) using sin/cos.

    Returns:
    --------
    pd.DataFrame
        DataFrame with date features added.

    Examples:
    ---------
    >>> # Create common date features
    >>> df_dates = create_date_features(df)
    >>>
    >>> # Specific features
    >>> df_dates = create_date_features(
    ...     df,
    ...     features=['dayofweek', 'month', 'quarter', 'is_weekend']
    ... )
    >>>
    >>> # Cyclical encoding (useful for ML models)
    >>> df_dates = create_date_features(df, cyclical=True)
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    # Default features
    if features is None:
        features = [
            'year', 'month', 'day', 'dayofweek', 'dayofyear',
            'week', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end'
        ]

    # Extract features
    if 'year' in features:
        df['year'] = df.index.year

    if 'month' in features:
        df['month'] = df.index.month
        if cyclical:
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    if 'day' in features:
        df['day'] = df.index.day

    if 'dayofweek' in features:
        df['dayofweek'] = df.index.dayofweek
        if cyclical:
            df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    if 'dayofyear' in features:
        df['dayofyear'] = df.index.dayofyear
        if cyclical:
            df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
            df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    if 'week' in features:
        df['week'] = df.index.isocalendar().week

    if 'quarter' in features:
        df['quarter'] = df.index.quarter
        if cyclical:
            df['quarter_sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df.index.quarter / 4)

    if 'hour' in features:
        df['hour'] = df.index.hour
        if cyclical:
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    if 'minute' in features:
        df['minute'] = df.index.minute

    if 'is_weekend' in features:
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    if 'is_month_start' in features:
        df['is_month_start'] = df.index.is_month_start.astype(int)

    if 'is_month_end' in features:
        df['is_month_end'] = df.index.is_month_end.astype(int)

    if 'is_quarter_start' in features:
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)

    if 'is_quarter_end' in features:
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

    if 'is_year_start' in features:
        df['is_year_start'] = df.index.is_year_start.astype(int)

    if 'is_year_end' in features:
        df['is_year_end'] = df.index.is_year_end.astype(int)

    n_features = len([f for f in features if f in df.columns])
    if cyclical:
        n_features += sum([1 for f in ['month', 'dayofweek', 'quarter', 'hour', 'dayofyear'] if f in features])

    print(f"✓ Created {n_features} date/time features")
    if cyclical:
        print("  (including cyclical sin/cos encodings)")

    return df


# ==================== SEASONAL FEATURES ====================

def create_seasonal_features(
    data: Union[pd.Series, pd.DataFrame],
    periods: Union[int, List[int]] = [7, 30, 365],
    method: str = 'indicator'
) -> pd.DataFrame:
    """
    Create seasonal indicator features.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex.
    periods : int or list
        Seasonal periods (e.g., 7 for weekly, 365 for yearly).
    method : str
        'indicator' for one-hot encoding or 'cyclical' for sin/cos.

    Returns:
    --------
    pd.DataFrame
        DataFrame with seasonal features.

    Examples:
    ---------
    >>> # Weekly seasonality (7 days)
    >>> df_seasonal = create_seasonal_features(df, periods=7)
    >>>
    >>> # Multiple seasonalities
    >>> df_seasonal = create_seasonal_features(df, periods=[7, 30, 365])
    >>>
    >>> # Cyclical encoding
    >>> df_seasonal = create_seasonal_features(df, periods=7, method='cyclical')
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    # Ensure list
    if isinstance(periods, int):
        period_list = [periods]
    else:
        period_list = periods

    # Create seasonal features
    for period in period_list:
        # Calculate position in cycle
        if period == 7:
            # Day of week
            cycle_position = df.index.dayofweek
        elif period == 12:
            # Month of year
            cycle_position = df.index.month - 1
        elif period == 4:
            # Quarter
            cycle_position = df.index.quarter - 1
        else:
            # Generic: position since start
            start_date = df.index.min()
            cycle_position = ((df.index - start_date).days % period)

        if method == 'indicator':
            # One-hot encoding
            for i in range(period):
                df[f'seasonal_{period}_{i}'] = (cycle_position == i).astype(int)

        elif method == 'cyclical':
            # Sin/cos encoding
            df[f'seasonal_{period}_sin'] = np.sin(2 * np.pi * cycle_position / period)
            df[f'seasonal_{period}_cos'] = np.cos(2 * np.pi * cycle_position / period)

    print(f"✓ Created seasonal features for periods: {period_list}")
    print(f"  Method: {method}")

    return df


# ==================== FOURIER FEATURES ====================

def create_fourier_features(
    data: Union[pd.Series, pd.DataFrame],
    period: int,
    order: int = 3
) -> pd.DataFrame:
    """
    Create Fourier features for complex seasonality.

    Fourier terms can capture complex seasonal patterns better than
    simple indicators, especially for long periods.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex.
    period : int
        Seasonal period (e.g., 365 for yearly, 7 for weekly).
    order : int
        Number of Fourier pairs (higher = more flexible).

    Returns:
    --------
    pd.DataFrame
        DataFrame with Fourier features.

    Examples:
    ---------
    >>> # Yearly seasonality with 10 Fourier pairs
    >>> df_fourier = create_fourier_features(df, period=365, order=10)
    >>>
    >>> # Weekly seasonality
    >>> df_fourier = create_fourier_features(df, period=7, order=3)
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    # Calculate time index
    t = np.arange(len(df))

    # Create Fourier terms
    for i in range(1, order + 1):
        df[f'fourier_{period}_sin_{i}'] = np.sin(2 * np.pi * i * t / period)
        df[f'fourier_{period}_cos_{i}'] = np.cos(2 * np.pi * i * t / period)

    n_features = order * 2
    print(f"✓ Created {n_features} Fourier features")
    print(f"  Period: {period}, Order: {order}")

    return df


# ==================== HOLIDAY FEATURES ====================

def create_holiday_features(
    data: Union[pd.Series, pd.DataFrame],
    country: str = 'US',
    include_nearby: bool = True,
    window: int = 1
) -> pd.DataFrame:
    """
    Create holiday indicator features.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex.
    country : str
        Country code for holidays (e.g., 'US', 'UK', 'CA').
    include_nearby : bool
        Include days before/after holiday.
    window : int
        Days before/after holiday to mark (if include_nearby=True).

    Returns:
    --------
    pd.DataFrame
        DataFrame with holiday features.

    Examples:
    ---------
    >>> # US holidays
    >>> df_holidays = create_holiday_features(df, country='US')
    >>>
    >>> # Include 2 days before/after holidays
    >>> df_holidays = create_holiday_features(
    ...     df, country='US', include_nearby=True, window=2
    ... )
    """
    try:
        import holidays
    except ImportError:
        warnings.warn("holidays package not installed. Install with: pip install holidays")
        return data

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    # Get holidays
    years = df.index.year.unique()
    country_holidays = holidays.country_holidays(country, years=years)

    # Create holiday indicator
    df['is_holiday'] = df.index.isin(country_holidays).astype(int)

    # Days before/after holiday
    if include_nearby:
        for i in range(1, window + 1):
            # Days before holiday
            df[f'is_{i}d_before_holiday'] = df.index.isin(
                [d - pd.Timedelta(days=i) for d in country_holidays.keys()]
            ).astype(int)

            # Days after holiday
            df[f'is_{i}d_after_holiday'] = df.index.isin(
                [d + pd.Timedelta(days=i) for d in country_holidays.keys()]
            ).astype(int)

    n_holidays = df['is_holiday'].sum()
    print(f"✓ Created holiday features for {country}")
    print(f"  Holidays found: {n_holidays}")
    if include_nearby:
        print(f"  Window: {window} days before/after")

    return df


# ==================== DIFFERENCE FEATURES ====================

def create_difference_features(
    data: Union[pd.Series, pd.DataFrame],
    periods: Union[int, List[int]] = 1,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create difference features (change from previous period).

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data.
    periods : int or list
        Periods to difference (e.g., 1 for previous day, 7 for same day last week).
    columns : list, optional
        Specific columns to difference.

    Returns:
    --------
    pd.DataFrame
        DataFrame with difference features.

    Examples:
    ---------
    >>> # Day-over-day change
    >>> df_diff = create_difference_features(df['sales'], periods=1)
    >>>
    >>> # Multiple periods (daily, weekly, monthly)
    >>> df_diff = create_difference_features(df['sales'], periods=[1, 7, 30])
    """
    # Convert Series to DataFrame
    if isinstance(data, pd.Series):
        df = data.to_frame()
        col_name = data.name or 'value'
        df.columns = [col_name]
        columns = [col_name]
    else:
        df = data.copy()
        if columns is None:
            columns = df.columns.tolist()

    # Ensure list
    if isinstance(periods, int):
        period_list = [periods]
    else:
        period_list = periods

    # Create difference features
    for col in columns:
        for period in period_list:
            df[f'{col}_diff_{period}'] = df[col].diff(period)
            # Percentage change
            df[f'{col}_pct_change_{period}'] = df[col].pct_change(period)

    n_features = len(columns) * len(period_list) * 2
    print(f"✓ Created {n_features} difference features")
    print(f"  Periods: {period_list}")

    return df


# ==================== AUTO FEATURE ENGINEERING ====================

def auto_create_features(
    data: Union[pd.Series, pd.DataFrame],
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    include_date_features: bool = True,
    include_seasonal: bool = True,
    seasonal_periods: Optional[List[int]] = None,
    include_holidays: bool = False,
    country: str = 'US'
) -> pd.DataFrame:
    """
    Automatically create comprehensive feature set for time series.

    One-stop function to create all common time series features.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex.
    lag_periods : list, optional
        Lag periods (default: [1, 7, 14, 30]).
    rolling_windows : list, optional
        Rolling windows (default: [7, 14, 30]).
    include_date_features : bool
        Create date/time features.
    include_seasonal : bool
        Create seasonal features.
    seasonal_periods : list, optional
        Seasonal periods (default: [7, 30]).
    include_holidays : bool
        Create holiday features.
    country : str
        Country for holidays.

    Returns:
    --------
    pd.DataFrame
        DataFrame with all features.

    Examples:
    ---------
    >>> # Create all features with defaults
    >>> df_features = auto_create_features(df['sales'])
    >>>
    >>> # Custom configuration
    >>> df_features = auto_create_features(
    ...     df['sales'],
    ...     lag_periods=[1, 7, 30, 365],
    ...     rolling_windows=[7, 30, 90],
    ...     include_holidays=True,
    ...     country='US'
    ... )
    """
    print("=" * 70)
    print("AUTO FEATURE ENGINEERING")
    print("=" * 70)

    df = data if isinstance(data, pd.DataFrame) else data.to_frame()

    # Default parameters
    if lag_periods is None:
        lag_periods = [1, 7, 14, 30]
    if rolling_windows is None:
        rolling_windows = [7, 14, 30]
    if seasonal_periods is None:
        seasonal_periods = [7, 30]

    # 1. Lag features
    print("\n1️⃣  Creating lag features...")
    df = create_lag_features(df, lags=lag_periods)

    # 2. Rolling features
    print("\n2️⃣  Creating rolling window features...")
    df = create_rolling_features(
        df,
        windows=rolling_windows,
        functions=['mean', 'std', 'min', 'max']
    )

    # 3. Date features
    if include_date_features:
        print("\n3️⃣  Creating date/time features...")
        df = create_date_features(df, cyclical=True)

    # 4. Seasonal features
    if include_seasonal:
        print("\n4️⃣  Creating seasonal features...")
        df = create_seasonal_features(df, periods=seasonal_periods, method='cyclical')

    # 5. Holiday features
    if include_holidays:
        print("\n5️⃣  Creating holiday features...")
        df = create_holiday_features(df, country=country, include_nearby=True, window=1)

    # 6. Difference features
    print("\n6️⃣  Creating difference features...")
    df = create_difference_features(df, periods=[1, 7])

    print("\n" + "=" * 70)
    print("✓ Feature engineering complete!")
    print(f"  Original features: {1 if isinstance(data, pd.Series) else len(data.columns)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  New features: {len(df.columns) - (1 if isinstance(data, pd.Series) else len(data.columns))}")
    print("=" * 70)

    return df

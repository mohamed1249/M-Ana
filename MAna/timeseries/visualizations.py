"""
Time Series Visualization Module

Simple, focused plots for time series analysis and forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Tuple


# ==================== BASIC TIME SERIES PLOTS ====================

def plot_timeseries(
    data: Union[pd.Series, pd.DataFrame],
    title: str = 'Time Series Plot',
    figsize: Tuple[int, int] = (14, 6),
    color: str = '#3498db'
):
    """
    Simple time series line plot.

    Examples:
    ---------
    >>> plot_timeseries(df['sales'], title='Daily Sales')
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, pd.Series):
        ax.plot(data.index, data.values, color=color, linewidth=2)
        ax.set_ylabel(data.name or 'Value', fontweight='bold')
    else:
        for col in data.columns:
            ax.plot(data.index, data[col], label=col, linewidth=2)
        ax.legend()

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    lower_bound: Optional[pd.Series] = None,
    upper_bound: Optional[pd.Series] = None,
    title: str = 'Forecast',
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot actual vs forecast with optional confidence intervals.

    Examples:
    ---------
    >>> plot_forecast(train_data, predictions)
    >>> plot_forecast(train_data, predictions, lower, upper)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Actual
    ax.plot(actual.index, actual.values, label='Actual',
           color='#3498db', linewidth=2)

    # Forecast
    ax.plot(forecast.index, forecast.values, label='Forecast',
           color='#e74c3c', linewidth=2, linestyle='--')

    # Confidence interval
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(
            forecast.index,
            lower_bound.values,
            upper_bound.values,
            alpha=0.3,
            color='#e74c3c',
            label='95% CI'
        )

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot residuals over time and distribution.

    Examples:
    ---------
    >>> plot_residuals(y_test, predictions)
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Time series plot
    ax1.scatter(range(len(residuals)), residuals, alpha=0.6, color='#9b59b6')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Index', fontweight='bold')
    ax1.set_ylabel('Residual', fontweight='bold')
    ax1.set_title('Residuals Over Time', fontweight='bold')
    ax1.grid(alpha=0.3)

    # Distribution
    ax2.hist(residuals, bins=30, color='#1abc9c', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Residual Distribution', fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== DECOMPOSITION PLOTS ====================

def plot_seasonal_decompose(
    data: pd.Series,
    period: int,
    model: str = 'additive',
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot seasonal decomposition (trend, seasonal, residual).

    Parameters:
    -----------
    data : pd.Series
        Time series data.
    period : int
        Seasonal period (e.g., 7 for weekly, 12 for monthly).
    model : str
        'additive' or 'multiplicative'.

    Examples:
    ---------
    >>> plot_seasonal_decompose(df['sales'], period=7)
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(data, model=model, period=period)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    decomposition.observed.plot(ax=axes[0], color='#3498db')
    axes[0].set_ylabel('Observed', fontweight='bold')
    axes[0].grid(alpha=0.3)

    decomposition.trend.plot(ax=axes[1], color='#e74c3c')
    axes[1].set_ylabel('Trend', fontweight='bold')
    axes[1].grid(alpha=0.3)

    decomposition.seasonal.plot(ax=axes[2], color='#2ecc71')
    axes[2].set_ylabel('Seasonal', fontweight='bold')
    axes[2].grid(alpha=0.3)

    decomposition.resid.plot(ax=axes[3], color='#9b59b6')
    axes[3].set_ylabel('Residual', fontweight='bold')
    axes[3].set_xlabel('Date', fontweight='bold')
    axes[3].grid(alpha=0.3)

    plt.suptitle(f'Seasonal Decomposition ({model.capitalize()})',
                fontweight='bold', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


# ==================== AUTOCORRELATION PLOTS ====================

def plot_acf_pacf(
    data: Union[pd.Series, np.ndarray],
    lags: int = 40,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot ACF and PACF (autocorrelation and partial autocorrelation).

    Helps identify ARIMA parameters.

    Examples:
    ---------
    >>> plot_acf_pacf(df['sales'], lags=30)
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ACF
    plot_acf(data, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation (ACF)', fontweight='bold')

    # PACF
    plot_pacf(data, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation (PACF)', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_lagged_scatter(
    data: pd.Series,
    lag: int = 1,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Scatter plot of value vs lagged value.

    Examples:
    ---------
    >>> plot_lagged_scatter(df['sales'], lag=7)
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(data, data.shift(lag), alpha=0.6, color='#3498db')
    ax.set_xlabel(f'{data.name or "Value"}', fontweight='bold')
    ax.set_ylabel(f'{data.name or "Value"} (lag={lag})', fontweight='bold')
    ax.set_title(f'Lagged Scatter Plot (lag={lag})', fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== COMPARISON PLOTS ====================

def plot_multiple_forecasts(
    actual: pd.Series,
    forecasts: dict,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot multiple model forecasts together.

    Parameters:
    -----------
    actual : pd.Series
        Actual/historical data.
    forecasts : dict
        Dict of {model_name: forecast_series}.

    Examples:
    ---------
    >>> forecasts = {
    ...     'ARIMA': arima_forecast,
    ...     'Prophet': prophet_forecast,
    ...     'Hybrid': hybrid_forecast
    ... }
    >>> plot_multiple_forecasts(train_data, forecasts)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Actual data
    ax.plot(actual.index, actual.values, label='Actual',
           color='#3498db', linewidth=2.5)

    # Forecasts
    colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts)))

    for i, (name, forecast) in enumerate(forecasts.items()):
        ax.plot(forecast.index, forecast.values,
               label=name, color=colors[i], linewidth=2,
               linestyle='--', alpha=0.8)

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== ANOMALY DETECTION PLOTS ====================

def plot_anomalies(
    data: pd.Series,
    anomalies: Union[pd.Series, np.ndarray],
    title: str = 'Anomaly Detection',
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot time series with anomalies highlighted.

    Parameters:
    -----------
    data : pd.Series
        Time series data.
    anomalies : pd.Series or array
        Boolean mask or indices of anomalies.

    Examples:
    ---------
    >>> anomalies = detect_anomalies_zscore(df['sales'])
    >>> plot_anomalies(df['sales'], anomalies)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normal data
    ax.plot(data.index, data.values, color='#3498db',
           linewidth=2, label='Normal')

    # Anomalies
    if isinstance(anomalies, (pd.Series, np.ndarray)):
        if anomalies.dtype == bool:
            anomaly_points = data[anomalies]
        else:
            anomaly_points = data.iloc[anomalies]
    else:
        anomaly_points = data.loc[anomalies]

    ax.scatter(anomaly_points.index, anomaly_points.values,
              color='#e74c3c', s=100, zorder=5, label='Anomaly',
              edgecolors='black', linewidths=1.5)

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== ROLLING STATISTICS ====================

def plot_rolling_stats(
    data: pd.Series,
    window: int = 30,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot rolling mean and standard deviation.

    Examples:
    ---------
    >>> plot_rolling_stats(df['sales'], window=7)
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Original + Rolling Mean
    ax1.plot(data.index, data.values, color='#3498db',
            linewidth=1, alpha=0.5, label='Original')
    ax1.plot(rolling_mean.index, rolling_mean.values,
            color='#e74c3c', linewidth=2, label=f'{window}-period MA')
    ax1.set_ylabel('Value', fontweight='bold')
    ax1.set_title('Rolling Mean', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Rolling Std
    ax2.plot(rolling_std.index, rolling_std.values,
            color='#9b59b6', linewidth=2)
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Std Dev', fontweight='bold')
    ax2.set_title('Rolling Standard Deviation', fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== SEASONAL PATTERNS ====================

def plot_seasonal_pattern(
    data: pd.Series,
    freq: str = 'month',
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot seasonal patterns (boxplot by month, day of week, etc.).

    Parameters:
    -----------
    data : pd.Series
        Time series with DatetimeIndex.
    freq : str
        'month', 'dayofweek', 'quarter', 'hour'.

    Examples:
    ---------
    >>> plot_seasonal_pattern(df['sales'], freq='month')
    >>> plot_seasonal_pattern(df['sales'], freq='dayofweek')
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    df = pd.DataFrame({'value': data})

    if freq == 'month':
        df['period'] = data.index.month
        xlabel = 'Month'
    elif freq == 'dayofweek':
        df['period'] = data.index.dayofweek
        xlabel = 'Day of Week (0=Mon, 6=Sun)'
    elif freq == 'quarter':
        df['period'] = data.index.quarter
        xlabel = 'Quarter'
    elif freq == 'hour':
        df['period'] = data.index.hour
        xlabel = 'Hour'
    else:
        raise ValueError(f"Unknown freq: {freq}")

    fig, ax = plt.subplots(figsize=figsize)

    df.boxplot(column='value', by='period', ax=ax)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(f'Seasonal Pattern by {freq.capitalize()}',
                fontweight='bold', fontsize=14)
    plt.suptitle('')  # Remove auto title
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== SPECTRAL / SEASONAL DIAGNOSTICS ====================

def plot_periodogram(
    data,
    sampling_rate: float = 1.0,
    detrend: str = 'linear',
    ax=None
):
    """Plot spectral power to reveal dominant seasonal periods."""
    from .diagnostics import periodogram_spectrum

    spectrum = periodogram_spectrum(
        data, sampling_rate=sampling_rate, detrend=detrend
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.step(spectrum['frequency'], spectrum['power'], color='#8e44ad')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Spectral Power')
    ax.set_title('Periodogram', fontweight='bold')
    ax.grid(alpha=0.3)
    return ax


def plot_seasonal_lines(
    data: pd.DataFrame,
    value_col: str,
    cycle_col: str,
    position_col: str,
    ax=None
):
    """Plot within-cycle behavior with one line per seasonal cycle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=data,
        x=position_col,
        y=value_col,
        hue=cycle_col,
        palette='husl',
        legend=False,
        ax=ax,
    )
    ax.set_title(f'Seasonal Plot ({cycle_col}/{position_col})', fontweight='bold')
    for line, name in zip(ax.lines, data[cycle_col].drop_duplicates()):
        values = line.get_ydata()
        if len(values):
            ax.annotate(
                str(name),
                xy=(1, values[-1]),
                xytext=(6, 0),
                color=line.get_color(),
                xycoords=ax.get_yaxis_transform(),
                textcoords='offset points',
                va='center',
            )
    return ax


def plot_seasonality_dashboard(
    data: pd.DataFrame,
    date_col: str,
    value_col: str,
    frequency: str = 'D',
    sampling_rate: float = 365.0
):
    """Create weekly, annual, and spectral seasonality views."""
    from .panel import aggregate_panel

    grouped = aggregate_panel(
        data,
        date_col=date_col,
        value_cols=value_col,
        frequency=frequency,
        aggregation='mean',
    ).set_index(date_col)
    features = grouped.copy()
    features['dayofweek'] = features.index.dayofweek
    features['week'] = features.index.isocalendar().week.astype(int)
    features['dayofyear'] = features.index.dayofyear
    features['year'] = features.index.year

    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    plot_seasonal_lines(features, value_col, 'week', 'dayofweek', ax=axes[0])
    plot_seasonal_lines(features, value_col, 'year', 'dayofyear', ax=axes[1])
    plot_periodogram(features[value_col], sampling_rate=sampling_rate, ax=axes[2])
    fig.tight_layout()
    return features, fig


def plot_time_series_diagnostics(
    data,
    lags: int = 30,
    figsize: Tuple[int, int] = (12, 7),
    style: str = 'bmh'
):
    """Plot a series with ACF/PACF and return its ADF result."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from .preprocessing import test_stationarity

    series = pd.Series(data).dropna()
    stationarity = test_stationarity(series, test='adf', verbose=False)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        series_ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        acf_ax = plt.subplot2grid((2, 2), (1, 0))
        pacf_ax = plt.subplot2grid((2, 2), (1, 1))
        series.plot(ax=series_ax)
        series_ax.set_title(
            f"Time Series Analysis\nADF p={stationarity['p_value']:.5f}"
        )
        plot_acf(series, lags=lags, ax=acf_ax)
        plot_pacf(series, lags=lags, ax=pacf_ax)
        fig.tight_layout()
    return fig, stationarity

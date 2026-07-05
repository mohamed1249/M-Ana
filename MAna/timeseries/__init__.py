"""
Time Series Analysis and Forecasting Module

Comprehensive tools for time series analysis:
- Data preprocessing and validation
- Feature engineering (lags, rolling windows, seasonality)
- Decomposition (trend, seasonal, residual)
- Forecasting models (ARIMA, SARIMA, Prophet, LSTM, Exponential Smoothing)
- Anomaly detection
- Forecast evaluation metrics
- Specialized visualizations
"""

from .preprocessing import (
    validate_timeseries,
    create_datetime_index,
    resample_timeseries,
    fill_missing_timestamps,
    detect_frequency,
    test_stationarity,
    difference_series,
    train_test_split_ts,
    remove_outliers_ts,
)

from .features import (
    create_lag_features,
    create_rolling_features,
    create_expanding_features,
    create_date_features,
    create_seasonal_features,
    create_fourier_features,
    create_holiday_features,
    create_difference_features,
    auto_create_features,
)

from .panel import (
    create_grouped_lag_features,
    aggregate_panel,
)

from .splitting import (
    ScaledTimeSeriesSplit,
    chronological_split_and_scale,
)

from .diagnostics import (
    periodogram_spectrum,
    dominant_periods,
)

from .decomposition import (
    seasonal_decompose,
    trend_decompose,
    stl_decompose,
    detrend,
    deseasonalize,
)

from .forecasting import (
    ARIMAForecaster,
    SARIMAForecaster,
    ProphetForecaster,
    LSTMForecaster,
    ExponentialSmoothingForecaster,
    AutoARIMA,
    EnsembleForecaster,
    HybridXGBLinearForecaster,
    NeuralProphetForecaster,
    BaseForecaster,
    ForecastResult,
)

from .evaluation import (
    mae, mse, rmse, mape, smape, r2_score,
    forecast_accuracy,
    forecast_bias,
    direction_accuracy,
    forecast_coverage,
    compare_models,
)

from .anomaly import (
    detect_anomalies_zscore,
    detect_anomalies_iqr,
    detect_anomalies_mad,
    detect_anomalies_isolation_forest,
    detect_anomalies_lof,
    detect_anomalies_prophet,
    detect_anomalies_ensemble,
)

__all__ = [
    # Preprocessing
    'validate_timeseries',
    'create_datetime_index',
    'resample_timeseries',
    'fill_missing_timestamps',
    'detect_frequency',
    'test_stationarity',
    'difference_series',
    'train_test_split_ts',
    'remove_outliers_ts',

    # Features
    'create_lag_features',
    'create_rolling_features',
    'create_expanding_features',
    'create_date_features',
    'create_seasonal_features',
    'create_fourier_features',
    'create_holiday_features',
    'create_difference_features',
    'auto_create_features',

    # Panel Data
    'create_grouped_lag_features',
    'aggregate_panel',

    # Chronological Splitting / Scaling
    'ScaledTimeSeriesSplit',
    'chronological_split_and_scale',

    # Diagnostics
    'periodogram_spectrum',
    'dominant_periods',

    # Decomposition
    'seasonal_decompose',
    'trend_decompose',
    'stl_decompose',
    'detrend',
    'deseasonalize',

    # Forecasting
    'ARIMAForecaster',
    'SARIMAForecaster',
    'ProphetForecaster',
    'LSTMForecaster',
    'ExponentialSmoothingForecaster',
    'AutoARIMA',
    'EnsembleForecaster',
    'HybridXGBLinearForecaster',
    'NeuralProphetForecaster',
    'BaseForecaster',
    'ForecastResult',

    # Evaluation
    'mae', 'mse', 'rmse', 'mape', 'smape', 'r2_score',
    'forecast_accuracy',
    'forecast_bias',
    'direction_accuracy',
    'forecast_coverage',
    'compare_models',

    # Anomaly Detection
    'detect_anomalies_zscore',
    'detect_anomalies_iqr',
    'detect_anomalies_mad',
    'detect_anomalies_isolation_forest',
    'detect_anomalies_lof',
    'detect_anomalies_prophet',
    'detect_anomalies_ensemble',
]

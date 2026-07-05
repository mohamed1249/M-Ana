"""
Time Series Forecast Evaluation Module

Simple, essential metrics for evaluating forecast accuracy.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional
import warnings


# ==================== BASIC ERROR METRICS ====================

def mae(y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Mean Absolute Error.

    Average of absolute differences between actual and predicted.

    Examples:
    ---------
    >>> mae_score = mae(y_test, predictions)
    >>> print(f"MAE: {mae_score:.2f}")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Mean Squared Error.

    Average of squared differences (penalizes large errors more).

    Examples:
    ---------
    >>> mse_score = mse(y_test, predictions)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: Union[pd.Series, np.ndarray],
         y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Root Mean Squared Error.

    Square root of MSE (same units as original data).

    Examples:
    ---------
    >>> rmse_score = rmse(y_test, predictions)
    >>> print(f"RMSE: {rmse_score:.2f}")
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: Union[pd.Series, np.ndarray],
         y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Mean Absolute Percentage Error.

    Average percentage error (in %).

    Examples:
    ---------
    >>> mape_score = mape(y_test, predictions)
    >>> print(f"MAPE: {mape_score:.1f}%")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0

    if not np.any(mask):
        warnings.warn("All true values are zero, MAPE undefined")
        return np.inf

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: Union[pd.Series, np.ndarray],
          y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Better than MAPE when values are close to zero.

    Examples:
    ---------
    >>> smape_score = smape(y_test, predictions)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Avoid division by zero
    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return np.mean(numerator[mask] / denominator[mask]) * 100


def r2_score(y_true: Union[pd.Series, np.ndarray],
             y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    R-squared (coefficient of determination).

    Proportion of variance explained (1.0 = perfect, 0.0 = baseline).

    Examples:
    ---------
    >>> r2 = r2_score(y_test, predictions)
    >>> print(f"R²: {r2:.3f}")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


# ==================== COMPREHENSIVE EVALUATION ====================

def forecast_accuracy(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculate all forecast accuracy metrics at once.

    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    verbose : bool
        Print formatted report.

    Returns:
    --------
    dict
        Dictionary with all metrics.

    Examples:
    ---------
    >>> metrics = forecast_accuracy(y_test, predictions)
    >>> print(f"RMSE: {metrics['rmse']:.2f}")
    >>>
    >>> # With report
    >>> metrics = forecast_accuracy(y_test, predictions, verbose=True)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate all metrics
    metrics = {
        'mae': mae(y_true, y_pred),
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    if verbose:
        print("=" * 50)
        print("FORECAST ACCURACY METRICS")
        print("=" * 50)
        print(f"MAE   : {metrics['mae']:>12.2f}")
        print(f"MSE   : {metrics['mse']:>12.2f}")
        print(f"RMSE  : {metrics['rmse']:>12.2f}")
        print(f"MAPE  : {metrics['mape']:>11.2f}%")
        print(f"sMAPE : {metrics['smape']:>11.2f}%")
        print(f"R²    : {metrics['r2']:>12.3f}")
        print("=" * 50)

    return metrics


# ==================== DIRECTIONAL ACCURACY ====================

def forecast_bias(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Forecast bias (average error).

    Positive = overforecast, Negative = underforecast, 0 = unbiased.

    Examples:
    ---------
    >>> bias = forecast_bias(y_test, predictions)
    >>> if bias > 0:
    ...     print("Model tends to overestimate")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(y_pred - y_true)


def direction_accuracy(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Direction accuracy (% of correct up/down predictions).

    Useful for financial forecasting.

    Examples:
    ---------
    >>> dir_acc = direction_accuracy(y_test, predictions)
    >>> print(f"Direction accuracy: {dir_acc:.1f}%")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) < 2:
        return np.nan

    # Calculate actual and predicted directions
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))

    # Percentage of matching directions
    return np.mean(actual_dir == pred_dir) * 100


# ==================== COVERAGE METRICS ====================

def forecast_coverage(
    y_true: Union[pd.Series, np.ndarray],
    lower_bound: Union[pd.Series, np.ndarray],
    upper_bound: Union[pd.Series, np.ndarray]
) -> float:
    """
    Percentage of actual values within prediction intervals.

    For 95% CI, should be close to 95%.

    Parameters:
    -----------
    y_true : array-like
        Actual values.
    lower_bound : array-like
        Lower confidence bound.
    upper_bound : array-like
        Upper confidence bound.

    Returns:
    --------
    float
        Coverage percentage.

    Examples:
    ---------
    >>> coverage = forecast_coverage(y_test, lower, upper)
    >>> print(f"Coverage: {coverage:.1f}%")
    """
    y_true = np.array(y_true)
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    within_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)

    return np.mean(within_bounds) * 100


# ==================== VISUALIZATION ====================

def plot_forecast_evaluation(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    figsize: tuple = (14, 10)
):
    """
    Create comprehensive forecast evaluation plots.

    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    dates : DatetimeIndex, optional
        Date index for x-axis.
    figsize : tuple
        Figure size.

    Examples:
    ---------
    >>> plot_forecast_evaluation(y_test, predictions, dates=test_dates)
    """
    import matplotlib.pyplot as plt

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if dates is None:
        dates = np.arange(len(y_true))

    # Calculate metrics
    metrics = forecast_accuracy(y_true, y_pred, verbose=False)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.plot(dates, y_true, label='Actual', color='#3498db', linewidth=2)
    ax1.plot(dates, y_pred, label='Predicted', color='#e74c3c', linewidth=2, linestyle='--')
    ax1.set_title('Actual vs Predicted', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Residuals
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.scatter(dates, residuals, alpha=0.6, color='#9b59b6')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Residuals', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Residual')
    ax2.grid(alpha=0.3)

    # 3. Residual Distribution
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, color='#1abc9c', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_title('Residual Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Frequency')
    ax3.grid(alpha=0.3)

    # 4. Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics_text = f"""
    FORECAST ACCURACY METRICS
    {'='*35}

    MAE    : {metrics['mae']:>12.2f}
    RMSE   : {metrics['rmse']:>12.2f}
    MAPE   : {metrics['mape']:>11.2f}%
    sMAPE  : {metrics['smape']:>11.2f}%
    R²     : {metrics['r2']:>12.3f}

    Bias   : {forecast_bias(y_true, y_pred):>12.2f}
    """

    ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.show()


# ==================== SIMPLE COMPARISON ====================

def compare_models(
    y_true: Union[pd.Series, np.ndarray],
    predictions: Dict[str, Union[pd.Series, np.ndarray]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models on same test set.

    Parameters:
    -----------
    y_true : array-like
        Actual values.
    predictions : dict
        Dict of {model_name: predictions}.
    verbose : bool
        Print formatted table.

    Returns:
    --------
    pd.DataFrame
        Comparison table sorted by RMSE.

    Examples:
    ---------
    >>> predictions = {
    ...     'ARIMA': arima_pred,
    ...     'Prophet': prophet_pred,
    ...     'Hybrid': hybrid_pred
    ... }
    >>>
    >>> results = compare_models(y_test, predictions)
    """
    results = []

    for model_name, y_pred in predictions.items():
        metrics = forecast_accuracy(y_true, y_pred, verbose=False)
        metrics['model'] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df[['model', 'mae', 'rmse', 'mape', 'smape', 'r2']]
    df = df.sort_values('rmse')

    if verbose:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON (sorted by RMSE)")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        print(f"\n🏆 Best Model: {df.iloc[0]['model']}")

    return df

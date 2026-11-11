"""
Time Series Forecasting Module

Comprehensive forecasting models for time series:
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)
- Prophet (Facebook's forecasting tool)
- LSTM (Deep Learning with PyTorch/TensorFlow)
- Exponential Smoothing (Holt-Winters)
- Auto ARIMA (automatic parameter selection)
- Ensemble methods (combine multiple models)

All models follow a consistent interface:
- fit(data) - Train the model
- predict(steps) - Generate forecasts
- plot_forecast() - Visualize results
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt


# ==================== BASE FORECASTER CLASS ====================

@dataclass
class ForecastResult:
    """
    Standardized forecast result object.

    Attributes:
    -----------
    forecast : pd.Series
        Forecasted values.
    lower_bound : pd.Series, optional
        Lower confidence interval.
    upper_bound : pd.Series, optional
        Upper confidence interval.
    model_name : str
        Name of the forecasting model.
    fitted_values : pd.Series, optional
        In-sample fitted values.
    residuals : pd.Series, optional
        Residuals (actual - fitted).
    metadata : dict
        Additional model information.
    """
    forecast: pd.Series
    lower_bound: Optional[pd.Series] = None
    upper_bound: Optional[pd.Series] = None
    model_name: str = "Forecaster"
    fitted_values: Optional[pd.Series] = None
    residuals: Optional[pd.Series] = None
    metadata: Dict[str, Any] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to DataFrame."""
        df = pd.DataFrame({
            'forecast': self.forecast
        })

        if self.lower_bound is not None:
            df['lower_bound'] = self.lower_bound
        if self.upper_bound is not None:
            df['upper_bound'] = self.upper_bound

        return df


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    Ensures consistent interface across all models.
    """

    def __init__(self):
        self.is_fitted = False
        self.training_data = None
        self.model = None
        self.model_name = "BaseForecaster"

    @abstractmethod
    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'BaseForecaster':
        """Train the model on data."""
        pass

    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Generate forecasts."""
        pass

    def predict_with_intervals(
        self,
        steps: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """Generate forecasts with confidence intervals."""
        pass

    def plot_forecast(
        self,
        steps: int,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Plot historical data and forecast."""
        pass


# ==================== ARIMA FORECASTER ====================

class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.

    ARIMA is a classic statistical model for time series forecasting.

    Parameters:
    -----------
    order : tuple
        (p, d, q) where:
        - p: number of autoregressive terms
        - d: number of differences (for stationarity)
        - q: number of moving average terms
    seasonal : bool
        Include seasonal component (becomes SARIMA).
    seasonal_order : tuple, optional
        (P, D, Q, s) for seasonal ARIMA.

    Examples:
    ---------
    >>> from MAna.timeseries import ARIMAForecaster
    >>>
    >>> # Simple ARIMA(1,1,1)
    >>> model = ARIMAForecaster(order=(1, 1, 1))
    >>> model.fit(train_data)
    >>> forecast = model.predict(steps=30)
    >>>
    >>> # Plot
    >>> model.plot_forecast(steps=30)
    >>>
    >>> # With confidence intervals
    >>> result = model.predict_with_intervals(steps=30, confidence=0.95)
    >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal: bool = False,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: Optional[str] = None
    ):
        super().__init__()
        self.order = order
        self.seasonal = seasonal
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model_name = "ARIMA" if not seasonal else "SARIMA"

        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.SARIMAX = SARIMAX
        except ImportError:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to data.

        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Training data with DatetimeIndex.

        Returns:
        --------
        self
            Fitted model.
        """
        # Convert DataFrame to Series if needed
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                data = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have single column. Select column first.")

        self.training_data = data.copy()

        # Fit model
        print(f"⏳ Fitting {self.model_name} model...")

        if self.seasonal and self.seasonal_order:
            self.model = self.SARIMAX(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend
            ).fit(disp=False)
        else:
            self.model = self.SARIMAX(
                data,
                order=self.order,
                trend=self.trend
            ).fit(disp=False)

        self.is_fitted = True

        print(f"✓ {self.model_name} model fitted successfully")
        print(f"  Order: {self.order}")
        if self.seasonal_order:
            print(f"  Seasonal Order: {self.seasonal_order}")
        print(f"  AIC: {self.model.aic:.2f}")
        print(f"  BIC: {self.model.bic:.2f}")

        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Generate point forecasts.

        Parameters:
        -----------
        steps : int
            Number of periods to forecast.

        Returns:
        --------
        pd.Series
            Forecasted values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        forecast = self.model.forecast(steps=steps)
        return forecast

    def predict_with_intervals(
        self,
        steps: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Generate forecasts with confidence intervals.

        Parameters:
        -----------
        steps : int
            Number of periods to forecast.
        confidence : float
            Confidence level (e.g., 0.95 for 95% CI).

        Returns:
        --------
        ForecastResult
            Forecast with confidence intervals.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Get forecast
        forecast_result = self.model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean

        # Get confidence intervals
        conf_int = forecast_result.conf_int(alpha=1-confidence)
        lower_bound = conf_int.iloc[:, 0]
        upper_bound = conf_int.iloc[:, 1]

        # Get fitted values and residuals
        fitted_values = self.model.fittedvalues
        residuals = self.model.resid

        return ForecastResult(
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_name=self.model_name,
            fitted_values=fitted_values,
            residuals=residuals,
            metadata={
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': self.model.aic,
                'bic': self.model.bic
            }
        )

    def plot_forecast(
        self,
        steps: int,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (14, 6),
        title: Optional[str] = None
    ):
        """
        Plot historical data and forecast.

        Parameters:
        -----------
        steps : int
            Number of periods to forecast.
        confidence : float
            Confidence level for intervals.
        figsize : tuple
            Figure size.
        title : str, optional
            Plot title.
        """
        result = self.predict_with_intervals(steps, confidence)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot historical data
        ax.plot(self.training_data.index, self.training_data.values,
               label='Historical', color='#3498db', linewidth=2)

        # Plot forecast
        ax.plot(result.forecast.index, result.forecast.values,
               label='Forecast', color='#e74c3c', linewidth=2, linestyle='--')

        # Plot confidence interval
        ax.fill_between(
            result.forecast.index,
            result.lower_bound.values,
            result.upper_bound.values,
            alpha=0.3,
            color='#e74c3c',
            label=f'{int(confidence*100)}% Confidence Interval'
        )

        # Styling
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{self.model_name} Forecast ({steps} steps ahead)',
                        fontsize=14, fontweight='bold')

        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print model summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        print(self.model.summary())


class SARIMAForecaster(ARIMAForecaster):
    """
    SARIMA (Seasonal ARIMA) forecaster.

    Convenience class for ARIMA with seasonality.

    Parameters:
    -----------
    order : tuple
        (p, d, q) - non-seasonal order.
    seasonal_order : tuple
        (P, D, Q, s) where s is the seasonal period.

    Examples:
    ---------
    >>> # Weekly seasonality (s=7)
    >>> model = SARIMAForecaster(
    ...     order=(1, 1, 1),
    ...     seasonal_order=(1, 1, 1, 7)
    ... )
    >>> model.fit(data)
    >>> forecast = model.predict(30)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)
    ):
        super().__init__(
            order=order,
            seasonal=True,
            seasonal_order=seasonal_order
        )


class NeuralProphetForecaster(BaseForecaster):
    """NeuralProphet forecaster with the standard M-Ana fit/predict API."""

    def __init__(self, frequency: Optional[str] = None, **model_kwargs):
        super().__init__()
        self.frequency = frequency
        self.model_kwargs = model_kwargs
        self.model_name = "NeuralProphet"
        self.training_frame = None
        self.training_metrics = None

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'NeuralProphetForecaster':
        try:
            from neuralprophet import NeuralProphet
        except ImportError as exc:
            raise ImportError(
                "NeuralProphet requires the time-series extra: "
                "pip install 'M_Ana_package[timeseries]'"
            ) from exc

        if isinstance(data, pd.DataFrame):
            if len(data.columns) != 1:
                raise ValueError("DataFrame must have a single target column")
            series = data.iloc[:, 0]
        else:
            series = data
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("data must have a DatetimeIndex")

        series = series.dropna().sort_index()
        frequency = self.frequency or pd.infer_freq(series.index)
        if frequency is None:
            raise ValueError("frequency could not be inferred; pass frequency explicitly")

        self.frequency = frequency
        self.training_data = series.copy()
        self.training_frame = pd.DataFrame({'ds': series.index, 'y': series.values})
        self.model = NeuralProphet(**self.model_kwargs)
        self.training_metrics = self.model.fit(self.training_frame, freq=frequency)
        self.is_fitted = True
        return self

    def _forecast_frame(self, steps: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("fit must be called before predict")
        if steps <= 0:
            raise ValueError("steps must be greater than zero")
        future = self.model.make_future_dataframe(
            self.training_frame,
            periods=steps,
            n_historic_predictions=False,
        )
        return self.model.predict(future)

    def predict(self, steps: int) -> pd.Series:
        forecast_frame = self._forecast_frame(steps)
        if 'yhat1' not in forecast_frame.columns:
            raise RuntimeError("NeuralProphet output does not contain 'yhat1'")
        forecast = forecast_frame.dropna(subset=['yhat1']).tail(steps)
        return pd.Series(
            forecast['yhat1'].to_numpy(),
            index=pd.DatetimeIndex(forecast['ds']),
            name='forecast',
        )

    def predict_with_intervals(
        self,
        steps: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        forecast = self.predict(steps)
        return ForecastResult(
            forecast=forecast,
            model_name=self.model_name,
            metadata={'frequency': self.frequency, 'confidence': confidence},
        )


# ==================== AUTO ARIMA ====================

class AutoARIMA(BaseForecaster):
    """
    Automatic ARIMA model selection.

    Automatically finds the best ARIMA parameters using information criteria.

    Parameters:
    -----------
    seasonal : bool
        Search for seasonal models.
    m : int
        Seasonal period (e.g., 12 for monthly, 7 for daily).
    max_p : int
        Maximum p parameter.
    max_q : int
        Maximum q parameter.
    max_d : int
        Maximum d parameter.
    information_criterion : str
        'aic' or 'bic' for model selection.

    Examples:
    ---------
    >>> # Auto-select best ARIMA model
    >>> model = AutoARIMA(seasonal=False)
    >>> model.fit(data)
    >>> print(f"Best order: {model.best_order}")
    >>> forecast = model.predict(30)
    >>>
    >>> # Seasonal Auto ARIMA
    >>> model = AutoARIMA(seasonal=True, m=7)  # Weekly seasonality
    >>> model.fit(data)
    """

    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        max_P: int = 2,
        max_Q: int = 2,
        max_D: int = 1,
        information_criterion: str = 'aic',
        suppress_warnings: bool = True
    ):
        super().__init__()
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_D = max_D
        self.information_criterion = information_criterion
        self.suppress_warnings = suppress_warnings
        self.model_name = "Auto ARIMA"
        self.best_order = None
        self.best_seasonal_order = None

        try:
            from pmdarima import auto_arima
            self.auto_arima = auto_arima
        except ImportError:
            raise ImportError("pmdarima is required. Install with: pip install pmdarima")

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'AutoARIMA':
        """
        Automatically find and fit best ARIMA model.

        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Training data.

        Returns:
        --------
        self
            Fitted model.
        """
        # Convert DataFrame to Series if needed
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                data = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have single column.")

        self.training_data = data.copy()

        print("⏳ Searching for best ARIMA parameters...")

        # Run auto ARIMA
        self.model = self.auto_arima(
            data,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            max_P=self.max_P,
            max_Q=self.max_Q,
            max_D=self.max_D,
            information_criterion=self.information_criterion,
            suppress_warnings=self.suppress_warnings,
            stepwise=True,
            trace=False
        )

        self.best_order = self.model.order
        self.best_seasonal_order = self.model.seasonal_order
        self.is_fitted = True

        print("✓ Best model found!")
        print(f"  Order: {self.best_order}")
        if self.seasonal:
            print(f"  Seasonal Order: {self.best_seasonal_order}")
        print(f"  AIC: {self.model.aic():.2f}")
        print(f"  BIC: {self.model.bic():.2f}")

        return self

    def predict(self, steps: int) -> pd.Series:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        forecast = self.model.predict(n_periods=steps)

        # Create proper index
        last_date = self.training_data.index[-1]
        freq = self.training_data.index.freq

        if freq:
            forecast_index = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=freq
            )[1:]
        else:
            forecast_index = range(len(self.training_data), len(self.training_data) + steps)

        return pd.Series(forecast, index=forecast_index)

    def predict_with_intervals(
        self,
        steps: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """Generate forecasts with confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        forecast, conf_int = self.model.predict(
            n_periods=steps,
            return_conf_int=True,
            alpha=1-confidence
        )

        # Create proper index
        last_date = self.training_data.index[-1]
        freq = self.training_data.index.freq

        if freq:
            forecast_index = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=freq
            )[1:]
        else:
            forecast_index = range(len(self.training_data), len(self.training_data) + steps)

        return ForecastResult(
            forecast=pd.Series(forecast, index=forecast_index),
            lower_bound=pd.Series(conf_int[:, 0], index=forecast_index),
            upper_bound=pd.Series(conf_int[:, 1], index=forecast_index),
            model_name=self.model_name,
            metadata={
                'order': self.best_order,
                'seasonal_order': self.best_seasonal_order,
                'aic': self.model.aic(),
                'bic': self.model.bic()
            }
        )

    def plot_forecast(
        self,
        steps: int,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """Plot forecast."""
        result = self.predict_with_intervals(steps, confidence)

        fig, ax = plt.subplots(figsize=figsize)

        # Historical
        ax.plot(self.training_data.index, self.training_data.values,
               label='Historical', color='#3498db', linewidth=2)

        # Forecast
        ax.plot(result.forecast.index, result.forecast.values,
               label='Forecast', color='#e74c3c', linewidth=2, linestyle='--')

        # Confidence interval
        ax.fill_between(
            result.forecast.index,
            result.lower_bound.values,
            result.upper_bound.values,
            alpha=0.3,
            color='#e74c3c',
            label=f'{int(confidence*100)}% CI'
        )

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Auto ARIMA{self.best_order} Forecast',
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# ==================== PROPHET FORECASTER ====================

class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet forecaster.

    Prophet is designed for business forecasting with strong seasonal effects
    and several seasons of historical data.

    Parameters:
    -----------
    growth : str
        'linear' or 'logistic' growth.
    seasonality_mode : str
        'additive' or 'multiplicative'.
    yearly_seasonality : bool or str
        Model yearly seasonality.
    weekly_seasonality : bool or str
        Model weekly seasonality.
    daily_seasonality : bool or str
        Model daily seasonality.

    Examples:
    ---------
    >>> from MAna.timeseries import ProphetForecaster
    >>>
    >>> # Basic Prophet model
    >>> model = ProphetForecaster()
    >>> model.fit(df, date_col='date', value_col='sales')
    >>> forecast = model.predict(periods=30)
    >>>
    >>> # With custom seasonality
    >>> model = ProphetForecaster(
    ...     seasonality_mode='multiplicative',
    ...     yearly_seasonality=True,
    ...     weekly_seasonality=True
    ... )
    >>> model.fit(df)
    >>> forecast = model.predict(periods=90)
    >>> model.plot_forecast()
    """

    def __init__(
        self,
        growth: str = 'linear',
        seasonality_mode: str = 'additive',
        yearly_seasonality: Union[bool, str] = 'auto',
        weekly_seasonality: Union[bool, str] = 'auto',
        daily_seasonality: Union[bool, str] = 'auto',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.80
    ):
        super().__init__()
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.model_name = "Prophet"

        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            raise ImportError("prophet is required. Install with: pip install prophet")

    def fit(
        self,
        data: Union[pd.Series, pd.DataFrame],
        date_col: Optional[str] = None,
        value_col: Optional[str] = None
    ) -> 'ProphetForecaster':
        """
        Fit Prophet model.

        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Training data. If Series, must have DatetimeIndex.
            If DataFrame, specify date_col and value_col.
        date_col : str, optional
            Date column name (for DataFrame).
        value_col : str, optional
            Value column name (for DataFrame).

        Returns:
        --------
        self
            Fitted model.
        """
        # Prepare data in Prophet format (ds, y)
        if isinstance(data, pd.Series):
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
        else:
            if date_col and value_col:
                df = data[[date_col, value_col]].copy()
                df.columns = ['ds', 'y']
            elif data.index.name or isinstance(data.index, pd.DatetimeIndex):
                if value_col:
                    df = pd.DataFrame({
                        'ds': data.index,
                        'y': data[value_col].values
                    })
                elif len(data.columns) == 1:
                    df = pd.DataFrame({
                        'ds': data.index,
                        'y': data.iloc[:, 0].values
                    })
                else:
                    raise ValueError("Specify value_col for DataFrame with multiple columns")
            else:
                raise ValueError("For DataFrame, specify date_col and value_col")

        self.training_data = df.copy()

        print("⏳ Fitting Prophet model...")

        # Initialize Prophet
        self.model = self.Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)

        self.is_fitted = True

        print("✓ Prophet model fitted successfully")
        print(f"  Growth: {self.growth}")
        print(f"  Seasonality Mode: {self.seasonality_mode}")

        return self

    def predict(self, periods: int, freq: str = 'D') -> pd.Series:
        """
        Generate forecasts.

        Parameters:
        -----------
        periods : int
            Number of periods to forecast.
        freq : str
            Frequency ('D' for daily, 'W' for weekly, etc.).

        Returns:
        --------
        pd.Series
            Forecasted values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Predict
        forecast = self.model.predict(future)

        # Return only future predictions
        forecast_values = forecast.tail(periods).set_index('ds')['yhat']

        return forecast_values

    def predict_with_intervals(
        self,
        periods: int,
        freq: str = 'D'
    ) -> ForecastResult:
        """Generate forecasts with confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Predict
        forecast = self.model.predict(future)

        # Extract forecast period
        forecast_df = forecast.tail(periods).set_index('ds')

        return ForecastResult(
            forecast=forecast_df['yhat'],
            lower_bound=forecast_df['yhat_lower'],
            upper_bound=forecast_df['yhat_upper'],
            model_name=self.model_name,
            metadata={
                'growth': self.growth,
                'seasonality_mode': self.seasonality_mode
            }
        )

    def plot_forecast(
        self,
        periods: Optional[int] = None,
        freq: str = 'D',
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot Prophet forecast with components.

        Parameters:
        -----------
        periods : int, optional
            If provided, generates new forecast. Otherwise uses last prediction.
        freq : str
            Frequency for new forecast.
        figsize : tuple
            Figure size.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        if periods:
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            forecast = self.model.predict(future)
        else:
            # Use training data for forecast
            forecast = self.model.predict(self.training_data)

        # Plot forecast
        self.model.plot(forecast, figsize=figsize)
        plt.title('Prophet Forecast', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Plot components
        self.model.plot_components(forecast, figsize=figsize)
        plt.tight_layout()
        plt.show()


# ==================== EXPONENTIAL SMOOTHING ====================

class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Exponential Smoothing forecaster (Holt-Winters).

    Good for data with trend and/or seasonality.

    Parameters:
    -----------
    trend : str, optional
        Type of trend: 'add', 'mul', or None.
    seasonal : str, optional
        Type of seasonality: 'add', 'mul', or None.
    seasonal_periods : int, optional
        Number of periods in season (e.g., 12 for monthly, 7 for daily).

    Examples:
    ---------
    >>> # Simple exponential smoothing (no trend, no seasonality)
    >>> model = ExponentialSmoothingForecaster()
    >>> model.fit(data)
    >>>
    >>> # Holt's linear trend
    >>> model = ExponentialSmoothingForecaster(trend='add')
    >>> model.fit(data)
    >>>
    >>> # Holt-Winters (trend + seasonality)
    >>> model = ExponentialSmoothingForecaster(
    ...     trend='add',
    ...     seasonal='add',
    ...     seasonal_periods=7
    ... )
    >>> model.fit(data)
    >>> forecast = model.predict(30)
    """

    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False
    ):
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.model_name = "Exponential Smoothing"

        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.ExponentialSmoothing = ExponentialSmoothing
        except ImportError:
            raise ImportError("statsmodels is required.")

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'ExponentialSmoothingForecaster':
        """Fit exponential smoothing model."""
        # Convert DataFrame to Series
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                data = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have single column.")

        self.training_data = data.copy()

        print("⏳ Fitting Exponential Smoothing model...")

        # Fit model
        self.model = self.ExponentialSmoothing(
            data,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend
        ).fit()

        self.is_fitted = True

        print("✓ Model fitted successfully")
        print(f"  Trend: {self.trend or 'None'}")
        print(f"  Seasonal: {self.seasonal or 'None'}")
        if self.seasonal_periods:
            print(f"  Seasonal Periods: {self.seasonal_periods}")

        return self

    def predict(self, steps: int) -> pd.Series:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        forecast = self.model.forecast(steps=steps)
        return forecast

    def predict_with_intervals(
        self,
        steps: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """Generate forecasts with intervals (simulated)."""
        forecast = self.predict(steps)

        # Simulate intervals based on residual standard error
        residuals = self.model.resid
        std_error = np.std(residuals)

        from scipy import stats as sp_stats
        z_score = sp_stats.norm.ppf((1 + confidence) / 2)

        lower = forecast - z_score * std_error
        upper = forecast + z_score * std_error

        return ForecastResult(
            forecast=forecast,
            lower_bound=lower,
            upper_bound=upper,
            model_name=self.model_name,
            fitted_values=self.model.fittedvalues,
            residuals=residuals
        )

    def plot_forecast(
        self,
        steps: int,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """Plot forecast."""
        result = self.predict_with_intervals(steps, confidence)

        fig, ax = plt.subplots(figsize=figsize)

        # Historical
        ax.plot(self.training_data.index, self.training_data.values,
               label='Historical', color='#3498db', linewidth=2)

        # Forecast
        ax.plot(result.forecast.index, result.forecast.values,
               label='Forecast', color='#e74c3c', linewidth=2, linestyle='--')

        # Confidence interval
        ax.fill_between(
            result.forecast.index,
            result.lower_bound.values,
            result.upper_bound.values,
            alpha=0.3,
            color='#e74c3c',
            label=f'{int(confidence*100)}% CI'
        )

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Exponential Smoothing Forecast', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# ==================== LSTM FORECASTER ====================

class LSTMForecaster(BaseForecaster):
    """
    LSTM (Long Short-Term Memory) neural network forecaster.

    Deep learning approach for time series forecasting.
    Requires PyTorch.

    Parameters:
    -----------
    lookback : int
        Number of past time steps to use for prediction.
    hidden_size : int
        Size of LSTM hidden layer.
    num_layers : int
        Number of LSTM layers.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate for optimizer.

    Examples:
    ---------
    >>> # Basic LSTM
    >>> model = LSTMForecaster(lookback=30, hidden_size=64, epochs=100)
    >>> model.fit(data)
    >>> forecast = model.predict(steps=30)
    >>>
    >>> # Deeper network
    >>> model = LSTMForecaster(
    ...     lookback=60,
    ...     hidden_size=128,
    ...     num_layers=2,
    ...     epochs=200
    ... )
    >>> model.fit(data)
    """

    def __init__(
        self,
        lookback: int = 30,
        hidden_size: int = 64,
        num_layers: int = 1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.model_name = "LSTM"

        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, data):
        """Create input sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'LSTMForecaster':
        """Fit LSTM model."""
        from sklearn.preprocessing import MinMaxScaler

        # Convert to Series
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                data = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have single column.")

        self.training_data = data.copy()

        # Scale data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        # Convert to tensors
        X = self.torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        y = self.torch.FloatTensor(y).to(self.device)

        # Build model. Capture the outer torch.nn module before defining the
        # nested class so the inner Module instance does not shadow it.
        nn_module = self.nn

        class LSTMModel(nn_module.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn_module.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.linear = nn_module.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                predictions = self.linear(lstm_out[:, -1, :])
                return predictions

        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Train
        criterion = self.nn.MSELoss()
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"[INFO] Training LSTM model on {self.device}...")

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

        self.is_fitted = True
        print("[OK] LSTM model trained successfully")

        return self

    def predict(self, steps: int) -> pd.Series:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Scale training data
        scaled_data = self.scaler.transform(
            self.training_data.values.reshape(-1, 1)
        ).flatten()

        # Generate forecasts
        self.model.eval()
        forecasts = []

        # Start with last lookback window
        current_sequence = scaled_data[-self.lookback:]

        with self.torch.no_grad():
            for _ in range(steps):
                # Prepare input
                x = self.torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1).to(self.device)

                # Predict
                pred = self.model(x).cpu().numpy()[0, 0]
                forecasts.append(pred)

                # Update sequence
                current_sequence = np.append(current_sequence[1:], pred)

        # Inverse transform
        forecasts = self.scaler.inverse_transform(
            np.array(forecasts).reshape(-1, 1)
        ).flatten()

        # Create index
        last_date = self.training_data.index[-1]
        freq = self.training_data.index.freq

        if freq:
            forecast_index = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=freq
            )[1:]
        else:
            forecast_index = range(len(self.training_data), len(self.training_data) + steps)

        return pd.Series(forecasts, index=forecast_index)

    def plot_forecast(
        self,
        steps: int,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """Plot forecast."""
        forecast = self.predict(steps)

        fig, ax = plt.subplots(figsize=figsize)

        # Historical
        ax.plot(self.training_data.index, self.training_data.values,
               label='Historical', color='#3498db', linewidth=2)

        # Forecast
        ax.plot(forecast.index, forecast.values,
               label='Forecast', color='#e74c3c', linewidth=2, linestyle='--')

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('LSTM Forecast', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# ==================== ENSEMBLE FORECASTER ====================

class EnsembleForecaster(BaseForecaster):
    """
    Ensemble of multiple forecasting models.

    Combines predictions from multiple models using averaging or weighted averaging.

    Parameters:
    -----------
    models : list
        List of fitted forecaster objects.
    weights : list, optional
        Weights for each model (must sum to 1).

    Examples:
    ---------
    >>> # Create individual models
    >>> arima = ARIMAForecaster(order=(1,1,1))
    >>> arima.fit(train_data)
    >>>
    >>> prophet = ProphetForecaster()
    >>> prophet.fit(train_data)
    >>>
    >>> # Ensemble (equal weights)
    >>> ensemble = EnsembleForecaster(models=[arima, prophet])
    >>> forecast = ensemble.predict(30)
    >>>
    >>> # Weighted ensemble
    >>> ensemble = EnsembleForecaster(
    ...     models=[arima, prophet],
    ...     weights=[0.6, 0.4]  # 60% ARIMA, 40% Prophet
    ... )
    >>> forecast = ensemble.predict(30)
    """

    def __init__(
        self,
        models: List[BaseForecaster],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.models = models

        if weights is None:
            self.weights = [1/len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1")
            self.weights = weights

        self.model_name = "Ensemble"
        self.is_fitted = all(m.is_fitted for m in models)

        if not self.is_fitted:
            warnings.warn("Some models are not fitted yet")

    def fit(self, data: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'EnsembleForecaster':
        """Fit any unfitted child models, or validate that all are already fitted."""
        unfitted = [model for model in self.models if not model.is_fitted]
        if unfitted:
            if data is None:
                raise ValueError("Provide data to fit unfitted ensemble members")
            for model in unfitted:
                model.fit(data)
        self.is_fitted = all(model.is_fitted for model in self.models)
        return self

    def predict(self, steps: int) -> pd.Series:
        """Generate ensemble forecast."""
        if not self.is_fitted:
            raise ValueError("All models must be fitted")

        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict(steps)
            predictions.append(pred.values)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        # Create index (assuming all models have same index)
        index = self.models[0].predict(steps).index
        return pd.Series(ensemble_pred, index=index)

    def plot_forecast(
        self,
        steps: int,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """Plot ensemble forecast."""
        forecast = self.predict(steps)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each model's forecast
        for model in self.models:
            model_forecast = model.predict(steps)
            ax.plot(model_forecast.index, model_forecast.values,
                   label=f'{model.model_name} Forecast', alpha=0.5)

        # Plot ensemble forecast
        ax.plot(forecast.index, forecast.values,
               label='Ensemble Forecast', color='#e74c3c', linewidth=2, linestyle='--')

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Ensemble Forecast', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()





# ==================== XGB + LINEAR REGRESSION HYBRID ====================

class HybridXGBLinearForecaster(BaseForecaster):
    """
    Hybrid XGBoost + Linear Regression forecaster with automatic weight optimization.

    Combines the power of gradient boosting (XGBoost) with linear regression,
    automatically finding the optimal blend weight to minimize error.

    This model works by:
    1. Training XGBoost on engineered time series features
    2. Training Linear Regression on the same features
    3. Finding optimal weight w where: prediction = w*XGB + (1-w)*LR
    4. Using validation set to select best weight

    Parameters:
    -----------
    lookback : int
        Number of lag features to create.
    rolling_windows : list
        Rolling window sizes for feature engineering.
    xgb_params : dict, optional
        XGBoost hyperparameters.
    optimize_weight : bool
        Automatically optimize blend weight (default: True).
    weight_resolution : int
        Number of weights to try (default: 101).
    validation_split : float
        Proportion of training data for weight validation.

    Examples:
    ---------
    >>> from MAna.timeseries import HybridXGBLinearForecaster
    >>>
    >>> # Basic usage with defaults
    >>> model = HybridXGBLinearForecaster()
    >>> model.fit(train_data)
    >>> forecast = model.predict(steps=30)
    >>> print(f"Optimal weight: {model.best_weight:.2f}")
    >>>
    >>> # Custom configuration
    >>> model = HybridXGBLinearForecaster(
    ...     lookback=30,
    ...     rolling_windows=[7, 14, 30],
    ...     xgb_params={'max_depth': 5, 'learning_rate': 0.1},
    ...     weight_resolution=51  # Try 51 weights (faster)
    ... )
    >>> model.fit(train_data)
    >>>
    >>> # Plot feature importance
    >>> model.plot_feature_importance()
    >>>
    >>> # Get detailed predictions
    >>> result = model.predict_with_intervals(steps=30)
    >>> print(f"XGB contribution: {model.best_weight*100:.1f}%")
    >>> print(f"LR contribution: {(1-model.best_weight)*100:.1f}%")
    """

    def __init__(
        self,
        lookback: int = 30,
        rolling_windows: List[int] = [7, 14, 30],
        xgb_params: Optional[Dict] = None,
        optimize_weight: bool = True,
        weight_resolution: int = 101,
        validation_split: float = 0.2,
        create_date_features: bool = True,
        create_seasonal_features: bool = True,
        seasonal_periods: List[int] = [7, 30]
    ):
        super().__init__()
        self.lookback = lookback
        self.rolling_windows = rolling_windows
        self.optimize_weight = optimize_weight
        self.weight_resolution = weight_resolution
        self.validation_split = validation_split
        self.create_date_features = create_date_features
        self.create_seasonal_features = create_seasonal_features
        self.seasonal_periods = seasonal_periods
        self.model_name = "Hybrid XGBoost+LinearRegression"

        # Default XGBoost parameters
        if xgb_params is None:
            self.xgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'random_state': 42
            }
        else:
            self.xgb_params = xgb_params

        # Models
        self.xgb_model = None
        self.lr_model = None
        self.best_weight = None
        self.feature_names = None

        # Try imports
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            self.LinearRegression = LinearRegression
            self.mean_squared_error = mean_squared_error
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    def _engineer_features(self, data: pd.Series) -> pd.DataFrame:
        """Create time series features for ML models."""
        from .features import (
            create_lag_features,
            create_rolling_features,
            create_date_features,
            create_seasonal_features
        )

        df = data.to_frame()
        col_name = data.name or 'value'

        # 1. Lag features
        df = create_lag_features(
            df,
            lags=list(range(1, self.lookback + 1)),
            columns=[col_name]
        )

        # 2. Rolling features
        df = create_rolling_features(
            df,
            windows=self.rolling_windows,
            functions=['mean', 'std', 'min', 'max'],
            columns=[col_name]
        )

        # 3. Date features
        if self.create_date_features:
            df = create_date_features(df, cyclical=True)

        # 4. Seasonal features
        if self.create_seasonal_features:
            df = create_seasonal_features(
                df,
                periods=self.seasonal_periods,
                method='cyclical'
            )

        return df

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'HybridXGBLinearForecaster':
        """
        Fit hybrid model with automatic weight optimization.

        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Training data with DatetimeIndex.

        Returns:
        --------
        self
            Fitted model.
        """
        # Convert DataFrame to Series
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                data = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have single column.")

        self.training_data = data.copy()

        print("=" * 70)
        print("HYBRID XGBoost + LinearRegression FORECASTER")
        print("=" * 70)

        # Step 1: Feature Engineering
        print("\n1️⃣  Creating features...")
        df_features = self._engineer_features(data)

        # Remove NaN from lag features
        df_clean = df_features.dropna()

        # Separate features and target
        target_col = data.name or 'value'
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        self.feature_names = X.columns.tolist()

        print(f"  ✓ Created {len(self.feature_names)} features")
        print(f"  ✓ Clean samples: {len(X):,}")

        # Step 2: Split into train/validation for weight optimization
        if self.optimize_weight:
            split_idx = int(len(X) * (1 - self.validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            print("\n2️⃣  Train/Validation split for weight optimization")
            print(f"  Train: {len(X_train):,} samples")
            print(f"  Validation: {len(X_val):,} samples")
        else:
            X_train, y_train = X, y

        # Step 3: Train XGBoost
        print("\n3️⃣  Training XGBoost model...")
        self.xgb_model = self.xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)] if not self.optimize_weight else [(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        print("  ✓ XGBoost trained")

        # Step 4: Train Linear Regression
        print("\n4️⃣  Training Linear Regression model...")
        self.lr_model = self.LinearRegression()
        self.lr_model.fit(X_train, y_train)
        print("  ✓ Linear Regression trained")

        # Step 5: Optimize weight
        if self.optimize_weight:
            print(f"\n5️⃣  Optimizing blend weight (testing {self.weight_resolution} weights)...")

            # Get validation predictions
            pred_xgb_val = self.xgb_model.predict(X_val)
            pred_lr_val = self.lr_model.predict(X_val)

            # Try different weights
            weights = np.linspace(0, 1, self.weight_resolution)
            best_mse = float('inf')
            best_w = 0.5

            mse_history = []

            for w in weights:
                hybrid_pred = w * pred_xgb_val + (1 - w) * pred_lr_val
                mse = self.mean_squared_error(y_val, hybrid_pred)
                mse_history.append(mse)

                if mse < best_mse:
                    best_mse = mse
                    best_w = w

            self.best_weight = best_w

            print("  ✓ Optimal weight found!")
            print(f"    Weight: {self.best_weight:.3f}")
            print(f"    XGBoost contribution: {self.best_weight*100:.1f}%")
            print(f"    LinearReg contribution: {(1-self.best_weight)*100:.1f}%")
            print(f"    Validation MSE: {best_mse:.6f}")

            # Store weight search history
            self.weight_search_history = {
                'weights': weights,
                'mse': mse_history
            }
        else:
            # Equal weights if not optimizing
            self.best_weight = 0.5
            print("\n5️⃣  Using equal weights (no optimization)")

        # Step 6: Retrain on full data
        print("\n6️⃣  Retraining on full dataset...")
        self.xgb_model.fit(X, y, verbose=False)
        self.lr_model.fit(X, y)

        self.is_fitted = True

        print("\n" + "=" * 70)
        print("✓ HYBRID MODEL TRAINING COMPLETE!")
        print("=" * 70)

        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Generate forecasts using hybrid model.

        Parameters:
        -----------
        steps : int
            Number of steps to forecast.

        Returns:
        --------
        pd.Series
            Forecasted values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        forecasts = []
        current_data = self.training_data.copy()

        for step in range(steps):
            # Engineer features for current state
            df_features = self._engineer_features(current_data)

            # Get last row (most recent features)
            X_current = df_features.drop(columns=[current_data.name or 'value']).iloc[[-1]]

            # Predict with both models
            pred_xgb = self.xgb_model.predict(X_current)[0]
            pred_lr = self.lr_model.predict(X_current)[0]

            # Hybrid prediction
            hybrid_pred = self.best_weight * pred_xgb + (1 - self.best_weight) * pred_lr

            forecasts.append(hybrid_pred)

            # Update data with new prediction
            last_date = current_data.index[-1]
            freq = current_data.index.freq

            if freq:
                next_date = last_date + freq
            else:
                # Infer frequency
                from .preprocessing import detect_frequency
                freq = detect_frequency(current_data, return_timedelta=True)
                if freq:
                    next_date = last_date + freq
                else:
                    next_date = last_date + pd.Timedelta(days=1)

            # Append prediction to data
            new_point = pd.Series(
                [hybrid_pred],
                index=[next_date],
                name=current_data.name,
            )
            current_data = pd.concat([current_data, new_point])

        # Create forecast index
        last_date = self.training_data.index[-1]
        freq = self.training_data.index.freq

        if freq:
            forecast_index = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=freq
            )[1:]
        else:
            forecast_index = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq='D'
            )[1:]

        return pd.Series(forecasts, index=forecast_index)

    def predict_with_intervals(
        self,
        steps: int,
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Generate forecasts with confidence intervals (bootstrapped).

        Parameters:
        -----------
        steps : int
            Number of steps to forecast.
        confidence : float
            Confidence level.

        Returns:
        --------
        ForecastResult
            Forecast with confidence intervals.
        """
        forecast = self.predict(steps)

        # Estimate confidence intervals using residual standard error
        # Engineer features for full training data
        df_features = self._engineer_features(self.training_data)
        df_clean = df_features.dropna()

        X = df_clean.drop(columns=[self.training_data.name or 'value'])
        y_true = df_clean[self.training_data.name or 'value']

        # Get predictions
        pred_xgb = self.xgb_model.predict(X)
        pred_lr = self.lr_model.predict(X)
        y_pred = self.best_weight * pred_xgb + (1 - self.best_weight) * pred_lr

        # Calculate residuals
        residuals = y_true - y_pred
        std_error = np.std(residuals)

        # Confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)

        # Increasing uncertainty over time
        uncertainty = std_error * np.sqrt(1 + np.arange(steps) / len(self.training_data))

        lower = forecast - z_score * uncertainty
        upper = forecast + z_score * uncertainty

        return ForecastResult(
            forecast=forecast,
            lower_bound=pd.Series(lower, index=forecast.index),
            upper_bound=pd.Series(upper, index=forecast.index),
            model_name=self.model_name,
            metadata={
                'best_weight': self.best_weight,
                'xgb_contribution': f"{self.best_weight*100:.1f}%",
                'lr_contribution': f"{(1-self.best_weight)*100:.1f}%",
                'n_features': len(self.feature_names)
            }
        )

    def plot_forecast(
        self,
        steps: int,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """Plot forecast with individual model contributions."""
        result = self.predict_with_intervals(steps, confidence)

        fig, ax = plt.subplots(figsize=figsize)

        # Historical data
        ax.plot(
            self.training_data.index,
            self.training_data.values,
            label='Historical',
            color='#3498db',
            linewidth=2
        )

        # Hybrid forecast
        ax.plot(
            result.forecast.index,
            result.forecast.values,
            label=f'Hybrid Forecast (w={self.best_weight:.2f})',
            color='#e74c3c',
            linewidth=2.5,
            linestyle='--'
        )

        # Confidence interval
        ax.fill_between(
            result.forecast.index,
            result.lower_bound.values,
            result.upper_bound.values,
            alpha=0.3,
            color='#e74c3c',
            label=f'{int(confidence*100)}% CI'
        )

        # Styling
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Hybrid XGBoost+LinearRegression Forecast\n'
            f'XGB: {self.best_weight*100:.1f}% | LR: {(1-self.best_weight)*100:.1f}%',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_weight_optimization(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot weight optimization curve showing MSE vs weight.

        Shows how model performance varies with different XGBoost/LR blends.
        """
        if not hasattr(self, 'weight_search_history'):
            print("⚠️  No weight optimization history available")
            return

        weights = self.weight_search_history['weights']
        mse = self.weight_search_history['mse']

        fig, ax = plt.subplots(figsize=figsize)

        # Plot MSE curve
        ax.plot(weights, mse, linewidth=2, color='#3498db')

        # Mark optimal point
        ax.axvline(
            self.best_weight,
            color='#e74c3c',
            linestyle='--',
            linewidth=2,
            label=f'Optimal: {self.best_weight:.3f}'
        )
        ax.scatter(
            [self.best_weight],
            [min(mse)],
            color='#e74c3c',
            s=200,
            zorder=5,
            edgecolors='black',
            linewidths=2
        )

        # Labels
        ax.set_xlabel('Weight (XGBoost contribution)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation MSE', fontsize=12, fontweight='bold')
        ax.set_title('Weight Optimization Curve', fontsize=14, fontweight='bold')

        # Add secondary x-axis for LR contribution
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Weight (LinearRegression contribution)', fontsize=12, fontweight='bold')
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels([f'{1-x:.2f}' for x in ax.get_xticks()])

        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot XGBoost feature importance.

        Parameters:
        -----------
        top_n : int
            Number of top features to show.
        figsize : tuple
            Figure size.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Get feature importance from XGBoost
        importance = self.xgb_model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))

        ax.barh(
            range(len(importance_df)),
            importance_df['importance'].values,
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )

        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'].values)
        ax.invert_yaxis()

        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} XGBoost Features', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_model_components(self) -> Dict[str, Any]:
        """
        Get individual model predictions for analysis.

        Returns:
        --------
        dict
            Dictionary with XGBoost, LinearRegression, and Hybrid predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Engineer features
        df_features = self._engineer_features(self.training_data)
        df_clean = df_features.dropna()

        X = df_clean.drop(columns=[self.training_data.name or 'value'])

        # Get predictions
        pred_xgb = self.xgb_model.predict(X)
        pred_lr = self.lr_model.predict(X)
        pred_hybrid = self.best_weight * pred_xgb + (1 - self.best_weight) * pred_lr

        return {
            'xgboost': pd.Series(pred_xgb, index=df_clean.index),
            'linear_regression': pd.Series(pred_lr, index=df_clean.index),
            'hybrid': pd.Series(pred_hybrid, index=df_clean.index),
            'actual': df_clean[self.training_data.name or 'value']
        }

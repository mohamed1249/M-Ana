import sys
import types
import unittest
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning


class PanelFeatureTests(unittest.TestCase):
    def test_grouped_lags_are_sorted_and_do_not_cross_entities(self):
        from MAna.timeseries import create_grouped_lag_features

        frame = pd.DataFrame(
            {
                "store": ["A", "B", "A", "B"],
                "date": ["2026-01-02", "2026-01-02", "2026-01-01", "2026-01-01"],
                "sales": [20, 200, 10, 100],
            }
        )
        result = create_grouped_lag_features(
            frame,
            group_by="store",
            columns="sales",
            lags=[1],
            time_col="date",
        )

        store_a = result[result["store"] == "A"]
        store_b = result[result["store"] == "B"]
        self.assertTrue(np.isnan(store_a.iloc[0]["sales_lag_1"]))
        self.assertEqual(store_a.iloc[1]["sales_lag_1"], 10)
        self.assertEqual(store_b.iloc[1]["sales_lag_1"], 100)

    def test_panel_aggregation_respects_group_and_frequency(self):
        from MAna.timeseries import aggregate_panel

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-01 01:00"]
                ),
                "store": ["A", "A", "B"],
                "sales": [2, 3, 10],
            }
        )
        result = aggregate_panel(
            frame,
            date_col="date",
            value_cols="sales",
            group_by="store",
            frequency="D",
        )
        self.assertEqual(result.loc[result["store"] == "A", "sales"].iloc[0], 5)
        self.assertEqual(result.loc[result["store"] == "B", "sales"].iloc[0], 10)


class TemporalSplitTests(unittest.TestCase):
    def test_scalers_fit_training_period_only(self):
        from MAna.timeseries import chronological_split_and_scale

        index = pd.date_range("2026-01-01", periods=10, freq="D")
        features = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=index)
        targets = pd.Series(np.arange(10, dtype=float) * 10, index=index)
        split = chronological_split_and_scale(features, targets, test_size=2, gap=1)

        self.assertEqual(split.X_train.shape, (7, 1))
        self.assertEqual(split.X_test.shape, (2, 1))
        self.assertAlmostEqual(float(split.X_train.mean()), 0.0)
        self.assertGreater(float(split.X_test.mean()), 1.0)
        np.testing.assert_allclose(
            split.inverse_transform_targets(split.y_test),
            targets.iloc[-2:].to_numpy(),
        )


class SpectralDiagnosticTests(unittest.TestCase):
    def test_dominant_period_finds_known_cycle(self):
        from MAna.timeseries import dominant_periods

        values = np.sin(2 * np.pi * np.arange(120) / 12)
        result = dominant_periods(values, n=1)
        self.assertAlmostEqual(result.iloc[0]["period"], 12.0, places=6)

    def test_periodogram_plot_returns_configured_axes(self):
        import matplotlib.pyplot as plt
        from MAna.timeseries.visualizations import plot_periodogram

        values = np.sin(2 * np.pi * np.arange(60) / 6)
        ax = plot_periodogram(values)
        try:
            self.assertEqual(ax.get_xscale(), "log")
            self.assertEqual(ax.get_title(), "Periodogram")
        finally:
            plt.close(ax.figure)


class TimeSeriesPreprocessingTests(unittest.TestCase):
    def test_detect_frequency_timedelta_uses_current_pandas_api(self):
        from MAna.timeseries import detect_frequency

        series = pd.Series(
            np.arange(5, dtype=float),
            index=pd.date_range("2026-01-01", periods=5, freq="D"),
        )

        self.assertEqual(detect_frequency(series, return_timedelta=True), pd.Timedelta(days=1))

    def test_remove_outliers_ts_supports_mad_method(self):
        from MAna.timeseries import remove_outliers_ts

        series = pd.Series(
            [1.0, 1.0, 1.0, 99.0, 1.0],
            index=pd.date_range("2026-01-01", periods=5, freq="D"),
        )

        result = remove_outliers_ts(series, method="mad", replace_with="median")

        self.assertEqual(result.loc["2026-01-04"], 1.0)

    def test_kpss_stationarity_suppresses_interpolation_warning(self):
        from MAna.timeseries import test_stationarity

        series = pd.Series(np.random.default_rng(42).normal(size=120))

        with warnings.catch_warnings():
            warnings.simplefilter("error", InterpolationWarning)
            result = test_stationarity(series, test="kpss", verbose=False)

        self.assertEqual(result["test"], "KPSS")
        self.assertIn("p_value", result)


class NeuralProphetTests(unittest.TestCase):
    def test_forecaster_uses_standard_fit_predict_interface(self):
        from MAna.timeseries import NeuralProphetForecaster

        class FakeNeuralProphet:
            def __init__(self, **kwargs):
                self.frequency = None

            def fit(self, frame, freq):
                self.frame = frame
                self.frequency = freq
                return pd.DataFrame({"loss": [1.0]})

            def make_future_dataframe(self, frame, periods, n_historic_predictions):
                offset = pd.tseries.frequencies.to_offset(self.frequency)
                dates = pd.date_range(frame["ds"].max() + offset, periods=periods, freq=offset)
                return pd.DataFrame({"ds": dates})

            def predict(self, frame):
                result = frame.copy()
                result["yhat1"] = np.arange(len(result), dtype=float)
                return result

        original = sys.modules.get("neuralprophet")
        sys.modules["neuralprophet"] = types.SimpleNamespace(
            NeuralProphet=FakeNeuralProphet
        )
        try:
            series = pd.Series(
                np.arange(10, dtype=float),
                index=pd.date_range("2026-01-01", periods=10, freq="D"),
            )
            model = NeuralProphetForecaster().fit(series)
            forecast = model.predict(3)
        finally:
            if original is None:
                sys.modules.pop("neuralprophet", None)
            else:
                sys.modules["neuralprophet"] = original

        self.assertEqual(len(forecast), 3)
        self.assertEqual(forecast.index[0], pd.Timestamp("2026-01-11"))
        self.assertTrue(model.is_fitted)


class LSTMForecasterTests(unittest.TestCase):
    def test_lstm_forecaster_trains_inner_torch_module(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")

        from MAna.timeseries import LSTMForecaster

        series = pd.Series(
            np.sin(np.arange(32, dtype=float) / 3) + 10,
            index=pd.date_range("2026-01-01", periods=32, freq="D"),
        )
        model = LSTMForecaster(
            lookback=4,
            hidden_size=4,
            epochs=1,
            learning_rate=0.01,
        ).fit(series)

        forecast = model.predict(2)

        self.assertEqual(len(forecast), 2)
        self.assertTrue(model.is_fitted)


class HybridForecasterTests(unittest.TestCase):
    def test_hybrid_forecaster_preserves_target_name_during_recursive_predict(self):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost is not installed")

        from MAna.timeseries import HybridXGBLinearForecaster

        series = pd.Series(
            np.linspace(10, 20, 80) + np.sin(np.arange(80) / 4),
            index=pd.date_range("2026-01-01", periods=80, freq="D"),
            name="sales",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model = HybridXGBLinearForecaster(
                lookback=4,
                rolling_windows=[3],
                xgb_params={
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "n_estimators": 5,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                },
                weight_resolution=3,
                seasonal_periods=[7],
            ).fit(series)

            forecast = model.predict(2)

        self.assertEqual(len(forecast), 2)
        self.assertFalse(forecast.isna().any())


class EnsembleForecasterTests(unittest.TestCase):
    def test_ensemble_can_wrap_already_fitted_models(self):
        from MAna.timeseries import EnsembleForecaster

        index = pd.date_range("2026-01-01", periods=2, freq="D")

        class FittedModel:
            is_fitted = True
            model_name = "Fitted"

            def __init__(self, values):
                self.values = values

            def predict(self, steps):
                return pd.Series(self.values[:steps], index=index[:steps])

        ensemble = EnsembleForecaster(
            models=[FittedModel([1.0, 3.0]), FittedModel([3.0, 5.0])],
            weights=[0.25, 0.75],
        )

        forecast = ensemble.predict(2)

        np.testing.assert_allclose(forecast.to_numpy(), [2.5, 4.5])


if __name__ == "__main__":
    unittest.main()

import tempfile
import sys
import types
import unittest
from unittest import mock

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ModelingPreprocessingTests(unittest.TestCase):
    def test_grouped_rolling_average_uses_numeric_value_column(self):
        from MAna.modeling import calculate_rolling_average

        frame = pd.DataFrame(
            {
                "store": ["a", "a", "a", "b", "b"],
                "sales": [10, 12, 14, 20, 24],
            }
        )
        result = calculate_rolling_average(
            frame,
            "store",
            window=2,
            value_column="sales",
        )

        self.assertIn("rolling_2_avg", result.columns)
        self.assertEqual(result.loc[2, "rolling_2_avg"], 13.0)
        self.assertEqual(result.loc[4, "rolling_2_avg"], 22.0)


class ModelingRegistryTests(unittest.TestCase):
    def test_classification_evaluation_and_registry_roundtrip(self):
        from MAna.modeling import ModelRegistry, evaluate_classification

        X, y = make_classification(
            n_samples=80,
            n_features=5,
            n_informative=3,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        model = LogisticRegression(max_iter=500).fit(X_train, y_train)
        metrics, report = evaluate_classification(
            model,
            X_test,
            y_test,
            plot=False,
            return_report=True,
        )

        self.assertIn("accuracy", metrics)
        self.assertEqual(report.model_type, "classification")

        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(base_dir=directory)
            registry.save_model(
                model,
                "demo_classifier",
                version="v1",
                framework="sklearn",
                task_type="classification",
                metrics={"accuracy": float(metrics["accuracy"])},
            )
            loaded = registry.load_model("demo_classifier", version="v1")

        self.assertEqual(loaded.predict(X_test[:3]).shape, (3,))

    def test_auto_classifier_does_not_pass_deprecated_xgboost_label_encoder(self):
        captured_kwargs = {}

        class FakeXGBClassifier(LogisticRegression):
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                super().__init__(max_iter=100)

        fake_xgboost = types.ModuleType("xgboost")
        fake_xgboost.XGBClassifier = FakeXGBClassifier

        X, y = make_classification(
            n_samples=30,
            n_features=4,
            n_informative=2,
            random_state=42,
        )

        from MAna.modeling import auto_classifier

        with mock.patch.dict(sys.modules, {"xgboost": fake_xgboost}):
            auto_classifier(
                X[:20],
                y[:20],
                X[20:],
                y[20:],
                time_limit=0.001,
                n_jobs=1,
                verbose=False,
            )

        self.assertEqual(captured_kwargs.get("eval_metric"), "logloss")
        self.assertNotIn("use_label_encoder", captured_kwargs)


class ModelingVisualizationTests(unittest.TestCase):
    def test_residual_plot_accepts_pandas_series_with_nondefault_index(self):
        import matplotlib

        matplotlib.use("Agg")

        from MAna.modeling import plot_residuals

        y_true = pd.Series([10.0, 12.0, 9.0, 15.0], index=[20, 30, 40, 50])
        y_pred = pd.Series([9.5, 11.5, 10.0, 14.0], index=[20, 30, 40, 50])
        figure = plot_residuals(y_true, y_pred)

        self.assertEqual(len(figure.axes), 3)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest

import numpy as np
import pandas as pd


class StataSmokeTests(unittest.TestCase):
    def test_stata_ab_test(self):
        from MAna.stata import ABTest

        data = pd.DataFrame(
            {
                "variant": ["control"] * 100 + ["treatment"] * 100,
                "converted": [0] * 80 + [1] * 20 + [0] * 65 + [1] * 35,
            }
        )
        result = ABTest(
            data=data,
            variant_col="variant",
            metric_col="converted",
        ).run(verbose=False)

        self.assertEqual(result.metric_type, "binary")
        self.assertGreater(result.treatment_metric, result.control_metric)

    def test_statistics_alias_is_not_exported(self):
        import MAna

        self.assertNotIn("statistics", MAna.__all__)
        self.assertIsNone(importlib.util.find_spec("MAna.statistics"))


class DatabaseSmokeTests(unittest.TestCase):
    def test_query_builder_executes_against_sqlite_connector(self):
        from MAna.database import Query, SQLiteConnector

        db = SQLiteConnector(":memory:")
        try:
            db.execute(
                "CREATE TABLE users "
                "(id INTEGER PRIMARY KEY, age INTEGER, created_at TEXT)"
            )
            db.execute(
                "INSERT INTO users (id, age, created_at) "
                "VALUES (:id, :age, :created_at)",
                {"id": 1, "age": 30, "created_at": "2026-01-01"},
            )
            db.execute(
                "INSERT INTO users (id, age, created_at) "
                "VALUES (:id, :age, :created_at)",
                {"id": 2, "age": 20, "created_at": "2026-01-02"},
            )

            result = (
                Query("users")
                .where(age__gte=25)
                .order_by("-created_at")
                .to_dataframe(db)
            )
            self.assertEqual(result["id"].tolist(), [1])
        finally:
            db.close()

    def test_mongodb_dataframe_helpers_with_collection_protocol(self):
        from MAna.database import dataframe_to_mongo, mongo_to_dataframe

        class FakeCollection:
            def __init__(self):
                self.rows = []

            def delete_many(self, query):
                self.rows.clear()

            def insert_many(self, rows):
                self.rows.extend(rows)

        class FakeMongo:
            def __init__(self):
                self.collection = FakeCollection()

            def get_collection(self, name):
                return self.collection

            def to_dataframe(self, collection_name, query=None, projection=None, limit=None):
                rows = self.collection.rows[:limit]
                return pd.DataFrame(rows)

        connection = FakeMongo()
        source = pd.DataFrame({"name": ["Ada", "Grace"], "score": [10, 20]})
        written = dataframe_to_mongo(
            source, connection, "people", if_exists="replace", batch_size=1
        )
        restored = mongo_to_dataframe(connection, "people")

        self.assertEqual(written, 2)
        pd.testing.assert_frame_equal(restored, source)


class TimeSeriesSmokeTests(unittest.TestCase):
    def test_advertised_public_api(self):
        from MAna.timeseries import (
            HybridXGBLinearForecaster,
            detect_anomalies_ensemble,
            forecast_accuracy,
            seasonal_decompose,
        )

        self.assertTrue(callable(HybridXGBLinearForecaster))

        index = pd.date_range("2026-01-01", periods=42, freq="D")
        values = pd.Series(
            10 + np.sin(np.arange(42) * 2 * np.pi / 7),
            index=index,
            name="value",
        )
        decomposition = seasonal_decompose(values, period=7)
        anomalies = detect_anomalies_ensemble(
            values, methods=["zscore", "iqr"], min_votes=1
        )
        metrics = forecast_accuracy(values.iloc[-7:], values.iloc[-7:], verbose=False)

        self.assertEqual(len(decomposition.observed), len(values))
        self.assertEqual(len(anomalies), len(values))
        self.assertAlmostEqual(metrics["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()

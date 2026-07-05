import importlib.util
import unittest
from datetime import datetime, timezone


PYSPARK_AVAILABLE = importlib.util.find_spec("pyspark") is not None


@unittest.skipUnless(PYSPARK_AVAILABLE, "PySpark is not installed")
class SparkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from MAna.big import create_spark_session

        cls.spark = create_spark_session(
            "MAna integration tests",
            master="local[2]",
            configs={
                "spark.sql.shuffle.partitions": "2",
                "spark.ui.enabled": "false",
            },
            log_level="ERROR",
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_real_dataframe_transform_and_join(self):
        from MAna.big import deduplicate_latest, safe_join, snake_case_columns

        rows = [
            (1, datetime(2026, 1, 1, tzinfo=timezone.utc), 10.0),
            (1, datetime(2026, 1, 2, tzinfo=timezone.utc), 12.0),
            (2, datetime(2026, 1, 1, tzinfo=timezone.utc), 7.0),
        ]
        frame = self.spark.createDataFrame(rows, ["User ID", "Event Time", "Score Value"])
        latest = deduplicate_latest(
            snake_case_columns(frame),
            keys=["user_id"],
            timestamp_column="event_time",
        )
        labels = self.spark.createDataFrame([(1, "gold"), (2, "silver")], ["user_id", "tier"])
        joined = safe_join(latest, labels, on="user_id", how="left")

        collected = {row.user_id: (row.score_value, row.tier) for row in joined.collect()}
        self.assertEqual(collected, {1: (12.0, "gold"), 2: (7.0, "silver")})

    def test_real_feature_pipeline_fit_and_transform(self):
        from MAna.big import build_feature_pipeline

        frame = self.spark.createDataFrame(
            [(20.0, "mobile"), (None, "desktop"), (35.0, "mobile")],
            ["age", "device"],
        )
        pipeline = build_feature_pipeline(
            numeric_columns=["age"],
            categorical_columns=["device"],
            scale_features=True,
        )
        transformed = pipeline.fit(frame).transform(frame)

        self.assertIn("features", transformed.columns)
        self.assertEqual(transformed.select("features").count(), 3)


if __name__ == "__main__":
    unittest.main()

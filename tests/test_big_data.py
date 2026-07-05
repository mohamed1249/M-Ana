import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


class LocalScaleProcessingTests(unittest.TestCase):
    def test_batching_and_threaded_map(self):
        from MAna.big import iter_batches, threaded_map, transform_in_batches

        batches = list(iter_batches([1, 2, 3, 4, 5], batch_size=2))
        transformed = transform_in_batches(
            [1, 2, 3],
            lambda values: [value * 10 for value in values],
            batch_size=2,
        )
        threaded = threaded_map(lambda value: value * value, [1, 2, 3], max_workers=2)

        self.assertEqual(batches, [[1, 2], [3, 4], [5]])
        self.assertEqual(transformed, [10, 20, 30])
        self.assertEqual(threaded, [1, 4, 9])

    def test_chunked_and_multi_file_csv_processing(self):
        from MAna.big import read_csv_directory, stream_csv_chunks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"value": [1, 2, 3]}).to_csv(root / "a.csv", index=False)
            pd.DataFrame({"value": [4, 5]}).to_csv(root / "b.csv", index=False)
            (root / "empty.csv").write_text("", encoding="utf-8")

            chunks = list(stream_csv_chunks(str(root / "a.csv"), chunk_size=2))
            combined, bad = read_csv_directory(
                str(root),
                add_source_column=True,
            )

        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])
        self.assertEqual(len(combined), 5)
        self.assertEqual(set(combined["source_file"]), {"a.csv", "b.csv"})
        self.assertEqual([path.name for path in bad], ["empty.csv"])


class SparkSessionTests(unittest.TestCase):
    def test_session_builder_applies_defaults_and_stops_context(self):
        from MAna.big import spark_session

        class FakeContext:
            def __init__(self):
                self.log_level = None

            def setLogLevel(self, value):
                self.log_level = value

        class FakeSpark:
            def __init__(self):
                self.sparkContext = FakeContext()
                self.stopped = False

            def stop(self):
                self.stopped = True

        class Builder:
            def __init__(self):
                self.values = {}
                self.spark = FakeSpark()

            def appName(self, value):
                self.values["app_name"] = value
                return self

            def master(self, value):
                self.values["master"] = value
                return self

            def config(self, key, value):
                self.values[key] = value
                return self

            def enableHiveSupport(self):
                self.values["hive"] = True
                return self

            def getOrCreate(self):
                return self.spark

        builder = Builder()
        fake_sql = types.ModuleType("pyspark.sql")
        fake_sql.SparkSession = type("SparkSession", (), {"builder": builder})
        fake_pyspark = types.ModuleType("pyspark")

        with mock.patch.dict(
            sys.modules,
            {"pyspark": fake_pyspark, "pyspark.sql": fake_sql},
        ):
            with spark_session(
                "Big Test",
                master="local[*]",
                configs={"spark.test.option": "yes"},
                enable_hive=True,
                log_level="WARN",
            ) as spark:
                self.assertFalse(spark.stopped)

        self.assertTrue(builder.spark.stopped)
        self.assertEqual(builder.values["app_name"], "Big Test")
        self.assertEqual(builder.values["master"], "local[*]")
        self.assertEqual(builder.values["spark.sql.adaptive.enabled"], "true")
        self.assertEqual(builder.values["spark.test.option"], "yes")
        self.assertEqual(builder.spark.sparkContext.log_level, "WARN")


class FakeReaderWriter:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def format(self, value):
        self.calls.append(("format", value))
        return self

    def schema(self, value):
        self.calls.append(("schema", value))
        return self

    def options(self, **values):
        self.calls.append(("options", values))
        return self

    def mode(self, value):
        self.calls.append(("mode", value))
        return self

    def partitionBy(self, *values):
        self.calls.append(("partitionBy", values))
        return self

    def outputMode(self, value):
        self.calls.append(("outputMode", value))
        return self

    def option(self, key, value):
        self.calls.append(("option", key, value))
        return self

    def queryName(self, value):
        self.calls.append(("queryName", value))
        return self

    def trigger(self, **values):
        self.calls.append(("trigger", values))
        return self

    def load(self, value=None):
        self.calls.append(("load", value))
        return self.result

    def save(self, value):
        self.calls.append(("save", value))

    def saveAsTable(self, value):
        self.calls.append(("saveAsTable", value))

    def start(self, value=None):
        self.calls.append(("start", value))
        return self.result


class FakeFrame:
    def __init__(self, columns, dtypes=None):
        self.columns = list(columns)
        self.dtypes = dtypes or [(column, "string") for column in columns]
        self.write = FakeReaderWriter()
        self.writeStream = FakeReaderWriter(result="query")
        self.selected = None
        self.repartition_call = None
        self.coalesce_call = None
        self.join_call = None

    def select(self, *columns):
        self.selected = columns
        return self

    def toDF(self, *columns):
        self.columns = list(columns)
        return self

    def withColumnRenamed(self, source, destination):
        self.columns = [destination if column == source else column for column in self.columns]
        return self

    def repartition(self, count, *columns):
        self.repartition_call = (count, columns)
        return self

    def coalesce(self, count):
        self.coalesce_call = count
        return self

    def join(self, right, on, how):
        self.join_call = (right, on, how)
        return self

    def unionByName(self, other, allowMissingColumns):
        self.union_call = (other, allowMissingColumns)
        return self

    def explain(self, mode):
        print(f"plan:{mode}")


class SparkIOAndTransformTests(unittest.TestCase):
    def test_schema_aware_read_and_partitioned_write(self):
        from MAna.big import read_spark_data, write_spark_data

        frame = FakeFrame(["id", "value"])
        reader = FakeReaderWriter(result=frame)
        spark = type("Spark", (), {"read": reader})()
        loaded = read_spark_data(
            spark,
            ["a.parquet", "b.parquet"],
            schema="id LONG, value STRING",
            options={"mergeSchema": "true"},
            columns=["id"],
        )
        write_spark_data(
            frame,
            "output",
            partition_by=["id"],
            num_partitions=4,
            partition_columns=["id"],
        )

        self.assertIs(loaded, frame)
        self.assertIn(("format", "parquet"), reader.calls)
        self.assertIn(("load", ["a.parquet", "b.parquet"]), reader.calls)
        self.assertEqual(frame.selected, ("id",))
        self.assertEqual(frame.repartition_call, (4, ("id",)))
        self.assertIn(("partitionBy", ("id",)), frame.write.calls)
        self.assertIn(("save", "output"), frame.write.calls)

    def test_column_normalization_safe_join_and_coalesce(self):
        from MAna.big import repartition_dataframe, safe_join, snake_case_columns

        frame = FakeFrame(["User ID", "HTTPStatus"])
        snake_case_columns(frame)
        self.assertEqual(frame.columns, ["user_id", "http_status"])

        left = FakeFrame(["id", "value"])
        right = FakeFrame(["id", "value", "label"])
        result = safe_join(left, right, "id", how="left")
        self.assertIs(result, left)
        self.assertIn("value_left", left.columns)
        self.assertIn("value_right", right.columns)
        self.assertEqual(left.join_call[1:], (["id"], "left"))

        repartition_dataframe(left, 2, strategy="coalesce")
        self.assertEqual(left.coalesce_call, 2)

    def test_schema_validation_and_explain_capture(self):
        from MAna.big import explain_text, validate_schema

        frame = FakeFrame(
            ["id", "value", "extra"],
            dtypes=[("id", "bigint"), ("value", "string"), ("extra", "double")],
        )
        result = validate_schema(
            frame,
            {"id": "bigint", "value": "double", "missing": "string"},
            allow_extra=False,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.missing_columns, ["missing"])
        self.assertIn("value", result.type_mismatches)
        self.assertEqual(result.unexpected_columns, ["extra"])
        self.assertIn("plan:formatted", explain_text(frame))

    def test_checkpointed_stream_writer(self):
        from MAna.big import start_stream

        frame = FakeFrame(["id", "event_time"])
        query = start_stream(
            frame,
            format="parquet",
            checkpoint_location="checkpoint",
            output_path="events",
            partition_by=["event_time"],
            trigger={"processingTime": "30 seconds"},
        )

        self.assertEqual(query, "query")
        self.assertIn(("option", "checkpointLocation", "checkpoint"), frame.writeStream.calls)
        self.assertIn(("trigger", {"processingTime": "30 seconds"}), frame.writeStream.calls)
        self.assertIn(("start", "events"), frame.writeStream.calls)


class SparkMLTests(unittest.TestCase):
    def test_feature_pipeline_builds_expected_stages_without_real_spark(self):
        from MAna.big import build_feature_pipeline

        class Stage:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Pipeline(Stage):
            @property
            def stages(self):
                return self.kwargs["stages"]

        fake_pyspark = types.ModuleType("pyspark")
        fake_ml = types.ModuleType("pyspark.ml")
        fake_feature = types.ModuleType("pyspark.ml.feature")
        fake_ml.Pipeline = Pipeline
        for name in (
            "Imputer",
            "OneHotEncoder",
            "StandardScaler",
            "StringIndexer",
            "VectorAssembler",
        ):
            setattr(fake_feature, name, type(name, (Stage,), {}))

        with mock.patch.dict(
            sys.modules,
            {
                "pyspark": fake_pyspark,
                "pyspark.ml": fake_ml,
                "pyspark.ml.feature": fake_feature,
            },
        ):
            pipeline = build_feature_pipeline(
                numeric_columns=["age"],
                categorical_columns=["country", "device"],
                scale_features=True,
            )

        self.assertEqual(len(pipeline.stages), 6)
        self.assertEqual(pipeline.stages[-1].kwargs["outputCol"], "features")


class BigDataPublicAPITests(unittest.TestCase):
    def test_big_module_is_public_and_imports_without_pyspark(self):
        import MAna
        import MAna.big as big

        expected = {
            "build_feature_pipeline",
            "create_spark_session",
            "read_spark_data",
            "safe_join",
            "start_stream",
            "stream_csv_chunks",
            "validate_schema",
            "write_spark_data",
        }
        self.assertIn("big", MAna.__all__)
        self.assertTrue(expected.issubset(set(big.__all__)))


if __name__ == "__main__":
    unittest.main()

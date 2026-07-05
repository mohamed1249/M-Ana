"""Local batching and optional PySpark workflow helpers."""

from .diagnostics import (
    SchemaValidationResult,
    dataframe_summary,
    duplicate_keys,
    explain_text,
    validate_schema,
)
from .io import (
    infer_data_format,
    read_spark_data,
    read_spark_table,
    write_spark_data,
    write_spark_table,
)
from .local import (
    iter_batches,
    iter_dataframe_batches,
    read_csv_directory,
    read_many_csv_files,
    stream_csv_chunks,
    threaded_map,
    transform_in_batches,
)
from .ml import build_feature_pipeline, random_split_dataframe
from .session import DEFAULT_SPARK_CONFIG, create_spark_session, spark_session
from .streaming import await_termination, read_spark_stream, start_stream
from .transforms import (
    add_ingestion_metadata,
    cast_columns,
    deduplicate_latest,
    fill_missing,
    rename_columns,
    repartition_dataframe,
    require_columns,
    safe_join,
    snake_case_columns,
    union_by_name,
)


__all__ = [
    "DEFAULT_SPARK_CONFIG",
    "SchemaValidationResult",
    "add_ingestion_metadata",
    "await_termination",
    "build_feature_pipeline",
    "cast_columns",
    "create_spark_session",
    "dataframe_summary",
    "deduplicate_latest",
    "duplicate_keys",
    "explain_text",
    "fill_missing",
    "infer_data_format",
    "iter_batches",
    "iter_dataframe_batches",
    "random_split_dataframe",
    "read_csv_directory",
    "read_many_csv_files",
    "read_spark_data",
    "read_spark_stream",
    "read_spark_table",
    "rename_columns",
    "repartition_dataframe",
    "require_columns",
    "safe_join",
    "snake_case_columns",
    "spark_session",
    "start_stream",
    "stream_csv_chunks",
    "threaded_map",
    "transform_in_batches",
    "union_by_name",
    "validate_schema",
    "write_spark_data",
    "write_spark_table",
]

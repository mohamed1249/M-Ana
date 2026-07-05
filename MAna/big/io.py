"""Schema-aware Spark readers and writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .transforms import repartition_dataframe


PathInput = Union[str, Sequence[str]]


def infer_data_format(path: str) -> str:
    """Infer a common Spark data source from a path suffix."""
    suffix = Path(str(path).split("?", 1)[0]).suffix.lower()
    formats = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "json",
        ".parquet": "parquet",
        ".orc": "orc",
        ".avro": "avro",
    }
    if suffix not in formats:
        raise ValueError("format could not be inferred; pass format explicitly")
    return formats[suffix]


def read_spark_data(
    spark: Any,
    paths: PathInput,
    *,
    format: Optional[str] = None,
    schema: Optional[Any] = None,
    options: Optional[Mapping[str, Any]] = None,
    columns: Optional[Sequence[str]] = None,
) -> Any:
    """Read one or more paths with explicit schema and selected columns."""
    path_values = [paths] if isinstance(paths, str) else list(paths)
    if not path_values:
        raise ValueError("paths cannot be empty")
    resolved_format = format or infer_data_format(path_values[0])
    reader = spark.read.format(resolved_format)
    if schema is not None:
        reader = reader.schema(schema)
    if options:
        reader = reader.options(**dict(options))
    target = path_values[0] if len(path_values) == 1 else path_values
    frame = reader.load(target)
    return frame.select(*columns) if columns is not None else frame


def read_spark_table(
    spark: Any,
    table_name: str,
    *,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
) -> Any:
    """Read a catalog table with optional projection and SQL filter."""
    if not table_name.strip():
        raise ValueError("table_name cannot be empty")
    frame = spark.table(table_name)
    if where:
        frame = frame.where(where)
    return frame.select(*columns) if columns is not None else frame


def write_spark_data(
    frame: Any,
    path: str,
    *,
    format: str = "parquet",
    mode: str = "errorifexists",
    partition_by: Sequence[str] = (),
    options: Optional[Mapping[str, Any]] = None,
    num_partitions: Optional[int] = None,
    partition_columns: Sequence[str] = (),
    partition_strategy: str = "repartition",
) -> None:
    """Write a DataFrame with optional compute and storage partitioning."""
    target = frame
    if num_partitions is not None:
        target = repartition_dataframe(
            target,
            num_partitions,
            columns=partition_columns,
            strategy=partition_strategy,
        )
    writer = target.write.format(format).mode(mode)
    if options:
        writer = writer.options(**dict(options))
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)


def write_spark_table(
    frame: Any,
    table_name: str,
    *,
    mode: str = "errorifexists",
    format: Optional[str] = None,
    partition_by: Sequence[str] = (),
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save a DataFrame as a managed or configured catalog table."""
    if not table_name.strip():
        raise ValueError("table_name cannot be empty")
    writer = frame.write.mode(mode)
    if format is not None:
        writer = writer.format(format)
    if options:
        writer = writer.options(**dict(options))
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.saveAsTable(table_name)


__all__ = [
    "infer_data_format",
    "read_spark_data",
    "read_spark_table",
    "write_spark_data",
    "write_spark_table",
]

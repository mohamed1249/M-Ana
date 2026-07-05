"""Reusable Spark DataFrame transformations and join safeguards."""

from __future__ import annotations

import re
from functools import reduce
from typing import Any, Mapping, Optional, Sequence, Tuple


def _spark_functions():
    try:
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise ImportError(
            "Spark transformations require: pip install 'M_Ana_package[big]'"
        ) from exc
    return F


def require_columns(frame: Any, columns: Sequence[str]) -> None:
    """Raise a clear error when required DataFrame columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")


def _snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(value).strip())
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    return value.strip("_").lower()


def snake_case_columns(frame: Any) -> Any:
    """Return a DataFrame with deterministic snake_case column names."""
    names = [_snake_case(column) for column in frame.columns]
    if any(not name for name in names):
        raise ValueError("column names cannot normalize to empty strings")
    if len(set(names)) != len(names):
        raise ValueError("column normalization would create duplicate names")
    return frame.toDF(*names)


def rename_columns(frame: Any, mapping: Mapping[str, str]) -> Any:
    """Rename several columns after validating their existence and uniqueness."""
    require_columns(frame, list(mapping))
    final_names = [mapping.get(column, column) for column in frame.columns]
    if len(set(final_names)) != len(final_names):
        raise ValueError("renaming would create duplicate columns")
    result = frame
    for source, destination in mapping.items():
        result = result.withColumnRenamed(source, destination)
    return result


def cast_columns(frame: Any, casts: Mapping[str, Any]) -> Any:
    """Cast several Spark columns using SQL type strings or DataType objects."""
    require_columns(frame, list(casts))
    F = _spark_functions()
    result = frame
    for column, data_type in casts.items():
        result = result.withColumn(column, F.col(column).cast(data_type))
    return result


def fill_missing(frame: Any, values: Mapping[str, Any]) -> Any:
    """Fill missing values after validating referenced columns."""
    require_columns(frame, list(values))
    return frame.fillna(dict(values))


def union_by_name(frames: Sequence[Any], *, allow_missing_columns: bool = True) -> Any:
    """Union DataFrames by column name rather than physical position."""
    values = list(frames)
    if not values:
        raise ValueError("frames cannot be empty")
    return reduce(
        lambda left, right: left.unionByName(
            right,
            allowMissingColumns=allow_missing_columns,
        ),
        values[1:],
        values[0],
    )


def safe_join(
    left: Any,
    right: Any,
    on: Sequence[str],
    *,
    how: str = "inner",
    broadcast_side: Optional[str] = None,
    suffixes: Tuple[str, str] = ("_left", "_right"),
) -> Any:
    """Join by named keys while preventing ambiguous duplicate output columns."""
    keys = [on] if isinstance(on, str) else list(on)
    if not keys:
        raise ValueError("on cannot be empty")
    require_columns(left, keys)
    require_columns(right, keys)
    if broadcast_side not in {None, "left", "right"}:
        raise ValueError("broadcast_side must be None, 'left', or 'right'")
    if len(suffixes) != 2 or suffixes[0] == suffixes[1]:
        raise ValueError("suffixes must contain two distinct values")

    overlap = (set(left.columns) & set(right.columns)) - set(keys)
    left_prepared = left
    right_prepared = right
    for column in sorted(overlap):
        left_prepared = left_prepared.withColumnRenamed(column, f"{column}{suffixes[0]}")
        right_prepared = right_prepared.withColumnRenamed(column, f"{column}{suffixes[1]}")

    if broadcast_side is not None:
        F = _spark_functions()
        if broadcast_side == "left":
            left_prepared = F.broadcast(left_prepared)
        else:
            right_prepared = F.broadcast(right_prepared)
    return left_prepared.join(right_prepared, on=keys, how=how)


def deduplicate_latest(
    frame: Any,
    keys: Sequence[str],
    timestamp_column: str,
    *,
    tie_breakers: Sequence[str] = (),
) -> Any:
    """Keep the latest row per key using a deterministic Spark window."""
    key_columns = list(keys)
    require_columns(frame, [*key_columns, timestamp_column, *tie_breakers])
    if not key_columns:
        raise ValueError("keys cannot be empty")
    try:
        from pyspark.sql import Window
    except ImportError as exc:
        raise ImportError(
            "Spark transformations require: pip install 'M_Ana_package[big]'"
        ) from exc
    F = _spark_functions()
    ordering = [F.col(timestamp_column).desc_nulls_last()]
    ordering.extend(F.col(column).desc_nulls_last() for column in tie_breakers)
    window = Window.partitionBy(*key_columns).orderBy(*ordering)
    marker = "__mana_row_number__"
    if marker in frame.columns:
        raise ValueError(f"temporary column already exists: {marker}")
    return (
        frame.withColumn(marker, F.row_number().over(window))
        .where(F.col(marker) == 1)
        .drop(marker)
    )


def repartition_dataframe(
    frame: Any,
    num_partitions: int,
    *,
    columns: Sequence[str] = (),
    strategy: str = "repartition",
) -> Any:
    """Change compute partitions explicitly with shuffle or narrow coalesce."""
    if num_partitions <= 0:
        raise ValueError("num_partitions must be greater than zero")
    if strategy not in {"repartition", "coalesce"}:
        raise ValueError("strategy must be 'repartition' or 'coalesce'")
    if columns:
        require_columns(frame, columns)
    if strategy == "coalesce":
        if columns:
            raise ValueError("coalesce does not accept partition columns")
        return frame.coalesce(num_partitions)
    return frame.repartition(num_partitions, *columns)


def add_ingestion_metadata(
    frame: Any,
    *,
    source_column: str = "source_file",
    timestamp_column: str = "ingested_at",
) -> Any:
    """Add source-file and ingestion-timestamp lineage columns."""
    F = _spark_functions()
    return frame.withColumn(source_column, F.input_file_name()).withColumn(
        timestamp_column,
        F.current_timestamp(),
    )


__all__ = [
    "add_ingestion_metadata",
    "cast_columns",
    "deduplicate_latest",
    "fill_missing",
    "rename_columns",
    "repartition_dataframe",
    "require_columns",
    "safe_join",
    "snake_case_columns",
    "union_by_name",
]

"""Spark schema, partition, duplicate-key, and execution-plan diagnostics."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .transforms import _spark_functions, require_columns


@dataclass(frozen=True)
class SchemaValidationResult:
    missing_columns: Sequence[str]
    unexpected_columns: Sequence[str]
    type_mismatches: Mapping[str, Dict[str, str]]

    @property
    def valid(self) -> bool:
        return not self.missing_columns and not self.unexpected_columns and not self.type_mismatches

    def raise_for_errors(self) -> None:
        if not self.valid:
            raise ValueError(
                "Schema validation failed: "
                f"missing={list(self.missing_columns)}, "
                f"unexpected={list(self.unexpected_columns)}, "
                f"type_mismatches={dict(self.type_mismatches)}"
            )


def validate_schema(
    frame: Any,
    expected_types: Mapping[str, str],
    *,
    allow_extra: bool = True,
) -> SchemaValidationResult:
    """Compare Spark ``dtypes`` against expected simple type names."""
    actual = dict(frame.dtypes)
    missing = sorted(set(expected_types) - set(actual))
    unexpected = sorted(set(actual) - set(expected_types)) if not allow_extra else []
    mismatches = {
        column: {"expected": str(expected), "actual": str(actual[column])}
        for column, expected in expected_types.items()
        if column in actual and str(actual[column]).lower() != str(expected).lower()
    }
    return SchemaValidationResult(missing, unexpected, mismatches)


def duplicate_keys(frame: Any, keys: Sequence[str], *, limit: Optional[int] = None) -> Any:
    """Return duplicate key combinations and their row counts as a lazy DataFrame."""
    key_columns = list(keys)
    if not key_columns:
        raise ValueError("keys cannot be empty")
    require_columns(frame, key_columns)
    F = _spark_functions()
    duplicates = (
        frame.groupBy(*key_columns)
        .agg(F.count(F.lit(1)).alias("duplicate_count"))
        .where(F.col("duplicate_count") > 1)
        .orderBy(F.col("duplicate_count").desc())
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        duplicates = duplicates.limit(limit)
    return duplicates


def dataframe_summary(frame: Any, *, include_distinct: bool = False) -> Dict[str, Any]:
    """Run explicit Spark actions for row, null, partition, and optional cardinality counts."""
    F = _spark_functions()
    row_count = int(frame.count())
    null_counts: Dict[str, int] = {}
    if frame.columns:
        expressions = [
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
            for column in frame.columns
        ]
        row = frame.select(*expressions).first()
        values = row.asDict() if hasattr(row, "asDict") else dict(row)
        null_counts = {column: int(values.get(column) or 0) for column in frame.columns}

    try:
        partitions = int(frame.rdd.getNumPartitions())
    except (AttributeError, NotImplementedError):
        partitions = None
    result: Dict[str, Any] = {
        "rows": row_count,
        "columns": len(frame.columns),
        "partitions": partitions,
        "null_counts": null_counts,
        "dtypes": dict(frame.dtypes),
    }
    if include_distinct:
        result["distinct_counts"] = {
            column: int(frame.select(column).distinct().count())
            for column in frame.columns
        }
    return result


def explain_text(frame: Any, *, mode: str = "formatted") -> str:
    """Capture ``DataFrame.explain`` output for logs and notebooks."""
    output = io.StringIO()
    with redirect_stdout(output):
        frame.explain(mode=mode)
    return output.getvalue()


__all__ = [
    "SchemaValidationResult",
    "dataframe_summary",
    "duplicate_keys",
    "explain_text",
    "validate_schema",
]

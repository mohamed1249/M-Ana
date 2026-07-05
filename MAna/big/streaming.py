"""Structured Streaming readers and checkpointed query starters."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def read_spark_stream(
    spark: Any,
    *,
    format: str,
    schema: Optional[Any] = None,
    options: Optional[Mapping[str, Any]] = None,
    path: Optional[str] = None,
) -> Any:
    """Create a streaming DataFrame from a configured Spark source."""
    reader = spark.readStream.format(format)
    if schema is not None:
        reader = reader.schema(schema)
    if options:
        reader = reader.options(**dict(options))
    return reader.load(path) if path is not None else reader.load()


def start_stream(
    frame: Any,
    *,
    format: str,
    checkpoint_location: str,
    output_path: Optional[str] = None,
    output_mode: str = "append",
    query_name: Optional[str] = None,
    partition_by: Sequence[str] = (),
    trigger: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Start a checkpointed Structured Streaming query."""
    if not checkpoint_location.strip():
        raise ValueError("checkpoint_location cannot be empty")
    writer = (
        frame.writeStream.format(format)
        .outputMode(output_mode)
        .option("checkpointLocation", checkpoint_location)
    )
    if options:
        writer = writer.options(**dict(options))
    if query_name:
        writer = writer.queryName(query_name)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    if trigger:
        writer = writer.trigger(**dict(trigger))
    return writer.start(output_path) if output_path is not None else writer.start()


def await_termination(query: Any, *, timeout: Optional[float] = None) -> Any:
    """Wait for a streaming query indefinitely or for a bounded duration."""
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    return query.awaitTermination() if timeout is None else query.awaitTermination(timeout)


__all__ = ["await_termination", "read_spark_stream", "start_stream"]

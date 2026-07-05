"""Spark session creation with optional, lazy PySpark imports."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional


DEFAULT_SPARK_CONFIG = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.session.timeZone": "UTC",
}


def _spark_session_class():
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise ImportError(
            "Spark support requires the big-data extra: "
            "pip install 'M_Ana_package[big]'"
        ) from exc
    return SparkSession


def create_spark_session(
    app_name: str = "MAna",
    *,
    master: Optional[str] = None,
    configs: Optional[Mapping[str, Any]] = None,
    enable_hive: bool = False,
    log_level: Optional[str] = None,
) -> Any:
    """Create or reuse a Spark session with practical SQL defaults.

    ``master`` is intentionally optional so cluster launchers can control it.
    Pass ``master="local[*]"`` for local development.
    """
    if not str(app_name).strip():
        raise ValueError("app_name cannot be empty")
    SparkSession = _spark_session_class()
    builder = SparkSession.builder.appName(str(app_name))
    if master is not None:
        builder = builder.master(master)

    resolved: Dict[str, Any] = dict(DEFAULT_SPARK_CONFIG)
    resolved.update(dict(configs or {}))
    for key, value in resolved.items():
        builder = builder.config(str(key), str(value))
    if enable_hive:
        builder = builder.enableHiveSupport()
    spark = builder.getOrCreate()

    if log_level is not None:
        context = getattr(spark, "sparkContext", None)
        if context is None or not hasattr(context, "setLogLevel"):
            raise RuntimeError("This Spark session does not expose sparkContext.setLogLevel")
        context.setLogLevel(log_level)
    return spark


@contextmanager
def spark_session(
    app_name: str = "MAna",
    *,
    stop: bool = True,
    **kwargs: Any,
) -> Iterator[Any]:
    """Context-manage a Spark session and optionally stop it on exit."""
    spark = create_spark_session(app_name, **kwargs)
    try:
        yield spark
    finally:
        if stop:
            spark.stop()


__all__ = ["DEFAULT_SPARK_CONFIG", "create_spark_session", "spark_session"]

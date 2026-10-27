"""PostgreSQL-first helpers for analytical database workflows.

The generic connectors in :mod:`MAna.database` remain useful across SQL
engines.  This module adds the PostgreSQL operations that data projects tend
to need in practice: environment-based connections, safe batch upserts,
``COPY`` transfers, catalog inspection, query plans, and maintenance.
"""

import csv
import io
import os
import re
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

try:
    from sqlalchemy import text
    from sqlalchemy.engine import URL

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast"}


def quote_postgres_identifier(identifier: str) -> str:
    """Validate and quote one PostgreSQL identifier.

    Identifiers cannot be passed as bound SQL parameters.  Keeping generated
    SQL deliberately conservative makes misspelled or unsafe names fail early.
    """
    if not isinstance(identifier, str) or not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(
            "PostgreSQL identifiers must start with a letter or underscore and "
            "contain only letters, numbers, and underscores"
        )
    return '"{}"'.format(identifier)


def _qualified_table(table_name: str, schema: Optional[str] = None) -> str:
    table = quote_postgres_identifier(table_name)
    if schema:
        return "{}.{}".format(quote_postgres_identifier(schema), table)
    return table


def _require_sqlalchemy() -> None:
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "PostgreSQL helpers require SQLAlchemy. Install with: "
            "pip install 'M_Ana_package[database]'"
        )


def _engine(connection: Any) -> Any:
    return getattr(connection, "engine", connection)


def _read_postgres(
    query: str,
    connection: Any,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    _require_sqlalchemy()
    return pd.read_sql(text(query), _engine(connection), params=dict(params or {}))


def postgres_url(
    database: str,
    username: str = "",
    password: str = "",
    host: str = "localhost",
    port: int = 5432,
    driver: str = "psycopg2",
    sslmode: Optional[str] = None,
    options: Optional[Mapping[str, str]] = None,
) -> str:
    """Build a correctly escaped SQLAlchemy PostgreSQL URL."""
    _require_sqlalchemy()
    query = dict(options or {})
    if sslmode:
        query["sslmode"] = sslmode
    url = URL.create(
        drivername="postgresql+{}".format(driver),
        username=username or None,
        password=password or None,
        host=host,
        port=int(port),
        database=database,
        query=query,
    )
    return url.render_as_string(hide_password=False)


def postgres_url_from_env(prefix: str = "PG", driver: str = "psycopg2") -> str:
    """Build a PostgreSQL URL from standard ``PG*`` environment variables.

    With the default prefix this reads ``PGHOST``, ``PGPORT``, ``PGDATABASE``,
    ``PGUSER``, ``PGPASSWORD``, and ``PGSSLMODE``.  Other prefixes use an
    underscore, for example ``WAREHOUSE_HOST``.
    """
    normalized = prefix.rstrip("_")

    def env_name(suffix: str) -> str:
        if normalized.upper() == "PG":
            return "PG{}".format(suffix)
        return "{}_{}".format(normalized, suffix)

    database = os.getenv(env_name("DATABASE"), "")
    if not database:
        raise ValueError("{} is required".format(env_name("DATABASE")))

    return postgres_url(
        database=database,
        username=os.getenv(env_name("USER"), ""),
        password=os.getenv(env_name("PASSWORD"), ""),
        host=os.getenv(env_name("HOST"), "localhost"),
        port=int(os.getenv(env_name("PORT"), "5432")),
        driver=driver,
        sslmode=os.getenv(env_name("SSLMODE")) or None,
    )


def connect_postgres(
    url: Optional[str] = None,
    *,
    env_prefix: str = "PG",
    driver: str = "psycopg2",
    **engine_options: Any
) -> Any:
    """Create a pooled PostgreSQL connector from a URL or environment."""
    from .connectors import DatabaseConnector

    connection_url = url or postgres_url_from_env(env_prefix, driver=driver)
    return DatabaseConnector(connection_url, **engine_options)


def create_schema_if_not_exists(connection: Any, schema: str) -> None:
    """Create a PostgreSQL schema, safely quoting its name."""
    _require_sqlalchemy()
    statement = "CREATE SCHEMA IF NOT EXISTS {}".format(
        quote_postgres_identifier(schema)
    )
    with _engine(connection).begin() as conn:
        conn.execute(text(statement))


def list_postgres_schemas(
    connection: Any,
    include_system: bool = False,
) -> List[str]:
    """Return visible PostgreSQL schema names."""
    frame = _read_postgres(
        "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name",
        connection,
    )
    schemas = frame["schema_name"].tolist()
    if include_system:
        return schemas
    return [
        schema
        for schema in schemas
        if schema not in _SYSTEM_SCHEMAS and not schema.startswith("pg_")
    ]


def list_postgres_tables(
    connection: Any,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """List PostgreSQL tables with estimated rows and total storage."""
    return _read_postgres(
        """
        SELECT
            n.nspname AS schema_name,
            c.relname AS table_name,
            c.reltuples::bigint AS estimated_rows,
            pg_total_relation_size(c.oid) AS total_bytes,
            pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size
        FROM pg_class AS c
        JOIN pg_namespace AS n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('r', 'p')
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND n.nspname NOT LIKE 'pg_toast%'
          AND (:schema IS NULL OR n.nspname = :schema)
        ORDER BY pg_total_relation_size(c.oid) DESC, n.nspname, c.relname
        """,
        connection,
        {"schema": schema},
    )


def get_postgres_table_sizes(
    connection: Any,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """Return table, index, and combined storage for PostgreSQL tables."""
    return _read_postgres(
        """
        SELECT
            n.nspname AS schema_name,
            c.relname AS table_name,
            pg_relation_size(c.oid) AS table_bytes,
            pg_indexes_size(c.oid) AS index_bytes,
            pg_total_relation_size(c.oid) AS total_bytes,
            pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size
        FROM pg_class AS c
        JOIN pg_namespace AS n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('r', 'p')
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND (:schema IS NULL OR n.nspname = :schema)
        ORDER BY total_bytes DESC
        """,
        connection,
        {"schema": schema},
    )


def get_postgres_indexes(
    connection: Any,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """Inspect index definitions and usage counters."""
    return _read_postgres(
        """
        SELECT
            i.schemaname AS schema_name,
            i.relname AS table_name,
            i.indexrelname AS index_name,
            i.idx_scan,
            i.idx_tup_read,
            i.idx_tup_fetch,
            pg_size_pretty(pg_relation_size(i.indexrelid)) AS index_size,
            x.indexdef
        FROM pg_stat_user_indexes AS i
        JOIN pg_indexes AS x
          ON x.schemaname = i.schemaname
         AND x.tablename = i.relname
         AND x.indexname = i.indexrelname
        WHERE (:schema IS NULL OR i.schemaname = :schema)
          AND (:table_name IS NULL OR i.relname = :table_name)
        ORDER BY i.schemaname, i.relname, i.indexrelname
        """,
        connection,
        {"schema": schema, "table_name": table_name},
    )


def get_postgres_constraints(
    connection: Any,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """Inspect primary-key, unique, foreign-key, and check constraints."""
    return _read_postgres(
        """
        SELECT
            tc.table_schema AS schema_name,
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS columns
        FROM information_schema.table_constraints AS tc
        LEFT JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_catalog = kcu.constraint_catalog
         AND tc.constraint_schema = kcu.constraint_schema
         AND tc.constraint_name = kcu.constraint_name
        WHERE (:schema IS NULL OR tc.table_schema = :schema)
          AND (:table_name IS NULL OR tc.table_name = :table_name)
        GROUP BY tc.table_schema, tc.table_name, tc.constraint_name, tc.constraint_type
        ORDER BY tc.table_schema, tc.table_name, tc.constraint_type, tc.constraint_name
        """,
        connection,
        {"schema": schema, "table_name": table_name},
    )


def postgres_database_size(connection: Any) -> pd.DataFrame:
    """Return the current PostgreSQL database size."""
    return _read_postgres(
        """
        SELECT
            current_database() AS database_name,
            pg_database_size(current_database()) AS total_bytes,
            pg_size_pretty(pg_database_size(current_database())) AS total_size
        """,
        connection,
    )


def get_postgres_table_health(
    connection: Any,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """Return live/dead tuple estimates and vacuum/analyze timestamps."""
    return _read_postgres(
        """
        SELECT
            schemaname AS schema_name,
            relname AS table_name,
            n_live_tup AS estimated_live_rows,
            n_dead_tup AS estimated_dead_rows,
            ROUND(
                100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0),
                2
            ) AS dead_row_percent,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables
        WHERE (:schema IS NULL OR schemaname = :schema)
        ORDER BY n_dead_tup DESC, schemaname, relname
        """,
        connection,
        {"schema": schema},
    )


def get_postgres_slow_queries(
    connection: Any,
    limit: int = 20,
    minimum_calls: int = 1,
) -> pd.DataFrame:
    """Return expensive statements from the ``pg_stat_statements`` extension.

    PostgreSQL must have ``pg_stat_statements`` installed and enabled.  Query
    text is normalized by PostgreSQL, and rows are ordered by total execution
    time rather than a hard-coded definition of "slow".
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if minimum_calls <= 0:
        raise ValueError("minimum_calls must be positive")
    return _read_postgres(
        """
        SELECT
            queryid,
            calls,
            total_exec_time,
            mean_exec_time,
            rows,
            shared_blks_hit,
            shared_blks_read,
            query
        FROM pg_stat_statements
        WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
          AND calls >= :minimum_calls
        ORDER BY total_exec_time DESC
        LIMIT :limit
        """,
        connection,
        {"minimum_calls": minimum_calls, "limit": limit},
    )


def infer_postgres_dtypes(df: pd.DataFrame) -> Dict[str, str]:
    """Infer practical PostgreSQL column types from a DataFrame."""
    inferred = {}
    for column in df.columns:
        series = df[column]
        dtype = series.dtype
        dtype_name = str(dtype).lower()
        sample = series.dropna().head(100)

        if pd.api.types.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"
        elif pd.api.types.is_unsigned_integer_dtype(dtype):
            sql_type = "NUMERIC(20)" if "64" in dtype_name else "BIGINT"
        elif pd.api.types.is_integer_dtype(dtype):
            if "8" in dtype_name or "16" in dtype_name:
                sql_type = "SMALLINT"
            elif "32" in dtype_name:
                sql_type = "INTEGER"
            else:
                sql_type = "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "REAL" if "32" in dtype_name else "DOUBLE PRECISION"
        elif isinstance(dtype, pd.DatetimeTZDtype):
            sql_type = "TIMESTAMPTZ"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "TIMESTAMP"
        elif pd.api.types.is_timedelta64_dtype(dtype):
            sql_type = "INTERVAL"
        elif not sample.empty and sample.map(lambda value: isinstance(value, Decimal)).all():
            sql_type = "NUMERIC"
        elif not sample.empty and sample.map(lambda value: isinstance(value, uuid.UUID)).all():
            sql_type = "UUID"
        elif not sample.empty and sample.map(lambda value: isinstance(value, (dict, list))).all():
            sql_type = "JSONB"
        elif not sample.empty and sample.map(lambda value: isinstance(value, bytes)).all():
            sql_type = "BYTEA"
        else:
            sql_type = "TEXT"
        inferred[str(column)] = sql_type
    return inferred


def _validate_postgres_type(sql_type: str) -> str:
    if not isinstance(sql_type, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_ ]*(?:\(\s*\d+(?:\s*,\s*\d+)?\s*\))?(?:\[\])?",
        sql_type.strip(),
    ):
        raise ValueError("Unsafe or unsupported PostgreSQL type: {!r}".format(sql_type))
    return sql_type.strip().upper()


def generate_postgres_create_table(
    df: pd.DataFrame,
    table_name: str,
    schema: Optional[str] = "public",
    primary_key: Optional[Union[str, Sequence[str]]] = None,
    not_null_columns: Optional[Sequence[str]] = None,
    dtype_overrides: Optional[Mapping[str, str]] = None,
    if_not_exists: bool = True,
) -> str:
    """Generate safe PostgreSQL ``CREATE TABLE`` SQL from a DataFrame schema."""
    if df.columns.empty:
        raise ValueError("DataFrame must have at least one column")
    columns = [str(column) for column in df.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("DataFrame columns must be unique")

    if isinstance(primary_key, str):
        primary_keys = [primary_key]
    else:
        primary_keys = list(primary_key or [])
    required = set(not_null_columns or []) | set(primary_keys)
    overrides = dict(dtype_overrides or {})

    missing = (required | set(overrides)) - set(columns)
    if missing:
        raise ValueError("Columns are missing from DataFrame: {}".format(sorted(missing)))

    dtypes = infer_postgres_dtypes(df)
    dtypes.update({column: _validate_postgres_type(value) for column, value in overrides.items()})
    definitions = []
    for column in columns:
        definition = "{} {}".format(
            quote_postgres_identifier(column),
            _validate_postgres_type(dtypes[column]),
        )
        if column in required:
            definition += " NOT NULL"
        definitions.append(definition)

    if primary_keys:
        definitions.append(
            "PRIMARY KEY ({})".format(
                ", ".join(quote_postgres_identifier(column) for column in primary_keys)
            )
        )
    prefix = "CREATE TABLE IF NOT EXISTS" if if_not_exists else "CREATE TABLE"
    return "{} {} (\n    {}\n)".format(
        prefix,
        _qualified_table(table_name, schema),
        ",\n    ".join(definitions),
    )


def create_postgres_table_from_dataframe(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    schema: Optional[str] = "public",
    primary_key: Optional[Union[str, Sequence[str]]] = None,
    not_null_columns: Optional[Sequence[str]] = None,
    dtype_overrides: Optional[Mapping[str, str]] = None,
    if_not_exists: bool = True,
) -> str:
    """Create a PostgreSQL table inferred from a DataFrame and return its DDL."""
    _require_sqlalchemy()
    statement = generate_postgres_create_table(
        df=df,
        table_name=table_name,
        schema=schema,
        primary_key=primary_key,
        not_null_columns=not_null_columns,
        dtype_overrides=dtype_overrides,
        if_not_exists=if_not_exists,
    )
    with _engine(connection).begin() as conn:
        conn.execute(text(statement))
    return statement


def create_postgres_index(
    connection: Any,
    table_name: str,
    columns: Union[str, Sequence[str]],
    schema: Optional[str] = "public",
    index_name: Optional[str] = None,
    unique: bool = False,
    method: str = "btree",
) -> str:
    """Create a safely quoted PostgreSQL index and return its name."""
    _require_sqlalchemy()
    column_names = [columns] if isinstance(columns, str) else list(columns)
    if not column_names:
        raise ValueError("columns cannot be empty")
    index_method = method.lower()
    if index_method not in {"btree", "hash", "gist", "spgist", "gin", "brin"}:
        raise ValueError("Unsupported PostgreSQL index method: {}".format(method))
    if index_name is None:
        index_name = "idx_{}_{}".format(table_name, "_".join(column_names))[:63]

    statement = "CREATE {}INDEX IF NOT EXISTS {} ON {} USING {} ({})".format(
        "UNIQUE " if unique else "",
        quote_postgres_identifier(index_name),
        _qualified_table(table_name, schema),
        index_method,
        ", ".join(quote_postgres_identifier(column) for column in column_names),
    )
    with _engine(connection).begin() as conn:
        conn.execute(text(statement))
    return index_name


def explain_postgres(
    query: str,
    connection: Any,
    params: Optional[Mapping[str, Any]] = None,
    analyze: bool = False,
    buffers: bool = False,
    format: str = "text",
) -> pd.DataFrame:
    """Run PostgreSQL ``EXPLAIN`` and return the plan as a DataFrame.

    ``analyze=True`` executes the query.  Use it carefully with statements that
    mutate data.  ``buffers`` requires ``analyze`` in PostgreSQL.
    """
    plan_format = format.upper()
    if plan_format not in {"TEXT", "JSON", "YAML", "XML"}:
        raise ValueError("format must be one of: text, json, yaml, xml")
    if buffers and not analyze:
        raise ValueError("buffers=True requires analyze=True")

    options = ["FORMAT {}".format(plan_format)]
    if analyze:
        options.append("ANALYZE TRUE")
    if buffers:
        options.append("BUFFERS TRUE")
    statement = "EXPLAIN ({}) {}".format(", ".join(options), query)
    return _read_postgres(statement, connection, params)


def postgres_upsert_sql(
    table_name: str,
    columns: Sequence[str],
    key_columns: Optional[Union[str, Sequence[str]]] = None,
    schema: Optional[str] = None,
    update_columns: Optional[Sequence[str]] = None,
    conflict_constraint: Optional[str] = None,
    do_nothing: bool = False,
) -> str:
    """Generate a parameterized PostgreSQL ``INSERT ... ON CONFLICT`` statement."""
    column_names = list(columns)
    if not column_names:
        raise ValueError("columns cannot be empty")
    if len(set(column_names)) != len(column_names):
        raise ValueError("columns must be unique")

    if isinstance(key_columns, str):
        keys = [key_columns]
    else:
        keys = list(key_columns or [])
    if keys and conflict_constraint:
        raise ValueError("Use key_columns or conflict_constraint, not both")
    if not keys and not conflict_constraint and not do_nothing:
        raise ValueError("key_columns or conflict_constraint is required for updates")

    missing_keys = set(keys) - set(column_names)
    if missing_keys:
        raise ValueError("Key columns are missing from data: {}".format(sorted(missing_keys)))

    quoted_columns = [quote_postgres_identifier(column) for column in column_names]
    placeholders = [":{}".format(column) for column in column_names]
    statement = "INSERT INTO {} ({}) VALUES ({})".format(
        _qualified_table(table_name, schema),
        ", ".join(quoted_columns),
        ", ".join(placeholders),
    )

    if conflict_constraint:
        conflict = " ON CONFLICT ON CONSTRAINT {}".format(
            quote_postgres_identifier(conflict_constraint)
        )
    elif keys:
        conflict = " ON CONFLICT ({})".format(
            ", ".join(quote_postgres_identifier(column) for column in keys)
        )
    else:
        conflict = " ON CONFLICT"

    if do_nothing:
        return statement + conflict + " DO NOTHING"

    if update_columns is None:
        updates = [column for column in column_names if column not in keys]
    else:
        updates = list(update_columns)
    missing_updates = set(updates) - set(column_names)
    if missing_updates:
        raise ValueError(
            "Update columns are missing from data: {}".format(sorted(missing_updates))
        )
    if not updates:
        return statement + conflict + " DO NOTHING"

    assignments = [
        "{0} = EXCLUDED.{0}".format(quote_postgres_identifier(column))
        for column in updates
    ]
    return statement + conflict + " DO UPDATE SET " + ", ".join(assignments)


def _python_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in record.items():
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_missing = False
        if is_missing:
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def upsert_postgres(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    key_columns: Optional[Union[str, Sequence[str]]] = None,
    schema: Optional[str] = None,
    update_columns: Optional[Sequence[str]] = None,
    conflict_constraint: Optional[str] = None,
    do_nothing: bool = False,
    batch_size: int = 1000,
    show_progress: bool = False,
) -> int:
    """Batch-upsert a DataFrame with PostgreSQL ``ON CONFLICT`` semantics."""
    _require_sqlalchemy()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if df.empty:
        return 0

    statement = text(
        postgres_upsert_sql(
            table_name=table_name,
            columns=list(df.columns),
            key_columns=key_columns,
            schema=schema,
            update_columns=update_columns,
            conflict_constraint=conflict_constraint,
            do_nothing=do_nothing,
        )
    )
    total = len(df)
    engine = _engine(connection)
    with engine.begin() as conn:
        for start in range(0, total, batch_size):
            records = [
                _python_record(record)
                for record in df.iloc[start:start + batch_size].to_dict("records")
            ]
            conn.execute(statement, records)
            if show_progress:
                print("[PostgreSQL] Upserted {:,}/{:,} rows".format(
                    min(start + batch_size, total), total
                ))
    return total


def _raw_connection(connection: Any) -> tuple:
    source = _engine(connection)
    if hasattr(source, "raw_connection"):
        wrapper = source.raw_connection()
        driver_connection = getattr(wrapper, "driver_connection", wrapper)
        return driver_connection, wrapper
    if hasattr(source, "cursor"):
        return source, None
    raise TypeError("connection must be a MAna connector, SQLAlchemy engine, or DBAPI connection")


def _copy_options(delimiter: str, null: str) -> str:
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be one character")
    escaped_delimiter = delimiter.replace("'", "''")
    escaped_null = null.replace("'", "''")
    return "FORMAT CSV, DELIMITER '{}', NULL '{}'".format(
        escaped_delimiter, escaped_null
    )


def copy_from_dataframe(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    schema: Optional[str] = "public",
    columns: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    null: str = r"\N",
) -> int:
    """Load a DataFrame through PostgreSQL ``COPY FROM STDIN``.

    The destination table must already exist.  This path is intended for large
    loads where pandas ``to_sql`` would spend most of its time issuing inserts.
    """
    if df.empty:
        return 0
    selected_columns = list(columns or df.columns)
    missing = set(selected_columns) - set(df.columns)
    if missing:
        raise ValueError("Columns are missing from DataFrame: {}".format(sorted(missing)))

    buffer = io.StringIO()
    df[selected_columns].to_csv(
        buffer,
        index=False,
        header=False,
        sep=delimiter,
        na_rep=null,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    buffer.seek(0)
    copy_sql = "COPY {} ({}) FROM STDIN WITH ({})".format(
        _qualified_table(table_name, schema),
        ", ".join(quote_postgres_identifier(column) for column in selected_columns),
        _copy_options(delimiter, null),
    )

    driver_connection, wrapper = _raw_connection(connection)
    cursor = driver_connection.cursor()
    try:
        if not hasattr(cursor, "copy_expert"):
            raise TypeError("COPY helpers currently require a psycopg2 connection")
        cursor.copy_expert(copy_sql, buffer)
        driver_connection.commit()
    except Exception:
        driver_connection.rollback()
        raise
    finally:
        cursor.close()
        if wrapper is not None:
            wrapper.close()
    return len(df)


def copy_to_dataframe(
    table_name: str,
    connection: Any,
    schema: Optional[str] = "public",
    columns: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    null: str = r"\N",
) -> pd.DataFrame:
    """Export a PostgreSQL table to a DataFrame through ``COPY TO STDOUT``."""
    qualified = _qualified_table(table_name, schema)
    driver_connection, wrapper = _raw_connection(connection)
    cursor = driver_connection.cursor()
    buffer = io.StringIO()
    try:
        if columns is None:
            cursor.execute("SELECT * FROM {} LIMIT 0".format(qualified))
            selected_columns = [description[0] for description in cursor.description]
        else:
            selected_columns = list(columns)
        column_sql = ", ".join(
            quote_postgres_identifier(column) for column in selected_columns
        )
        copy_sql = "COPY (SELECT {} FROM {}) TO STDOUT WITH ({})".format(
            column_sql,
            qualified,
            _copy_options(delimiter, null),
        )
        if not hasattr(cursor, "copy_expert"):
            raise TypeError("COPY helpers currently require a psycopg2 connection")
        cursor.copy_expert(copy_sql, buffer)
    finally:
        cursor.close()
        if wrapper is not None:
            wrapper.close()

    buffer.seek(0)
    if not buffer.getvalue():
        return pd.DataFrame(columns=selected_columns)
    return pd.read_csv(
        buffer,
        sep=delimiter,
        names=selected_columns,
        na_values=[null],
        keep_default_na=True,
    )


def vacuum_analyze_postgres(
    connection: Any,
    table_name: Optional[str] = None,
    schema: Optional[str] = "public",
) -> None:
    """Run ``VACUUM ANALYZE`` outside a transaction block."""
    _require_sqlalchemy()
    statement = "VACUUM ANALYZE"
    if table_name:
        statement += " " + _qualified_table(table_name, schema)

    engine = _engine(connection)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(statement))

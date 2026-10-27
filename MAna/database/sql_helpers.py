"""
SQL Helper Functions for Pandas Integration

Provides enhanced utilities for seamless DataFrame ↔ SQL operations:
- Smart data loading with chunking and progress bars
- Optimized data writing with batching
- Data type optimization
- Bulk operations
- Transaction management
- Table introspection
"""

import pandas as pd
from typing import Optional, Union, List, Dict, Any
import warnings
from contextlib import contextmanager

try:
    from sqlalchemy import inspect, text
    from sqlalchemy.types import (
        BigInteger, Boolean, DateTime, Float, Integer, String, Text,
    )
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    warnings.warn("SQLAlchemy not installed. Install with: pip install sqlalchemy")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ==================== READING DATA ====================

def read_sql(
    query: Union[str, Any],
    connection: Any,
    params: Optional[Dict[str, Any]] = None,
    chunksize: Optional[int] = None,
    show_progress: bool = False,
    index_col: Optional[str] = None,
    parse_dates: Optional[Union[List, Dict]] = None,
    optimize_dtypes: bool = True,
    coerce_float: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Enhanced version of pd.read_sql with chunking and progress tracking.

    Features:
    - Automatic chunking for large datasets
    - Progress bars
    - Data type optimization
    - Memory-efficient processing

    Parameters:
    -----------
    query : str or sqlalchemy.Selectable
        SQL query or table object.
    connection : database connection
        Database connection or engine.
    params : dict, optional
        Query parameters for parameterized queries.
    chunksize : int, optional
        Number of rows per chunk. If provided, returns iterator.
    show_progress : bool
        Show progress bar (requires tqdm).
    index_col : str, optional
        Column to use as DataFrame index.
    parse_dates : list or dict, optional
        Columns to parse as dates.
    optimize_dtypes : bool
        Automatically optimize data types to save memory.
    coerce_float : bool
        Attempt to convert non-numeric to float.
    **kwargs :
        Additional arguments passed to pd.read_sql.

    Returns:
    --------
    pd.DataFrame or iterator
        Query results as DataFrame (or iterator if chunksize specified).

    Examples:
    ---------
    >>> # Basic query
    >>> df = read_sql("SELECT * FROM users", db)
    >>>
    >>> # With parameters (safe from SQL injection)
    >>> df = read_sql(
    ...     "SELECT * FROM users WHERE age > :min_age",
    ...     db,
    ...     params={'min_age': 25}
    ... )
    >>>
    >>> # Large dataset with chunking
    >>> df = read_sql(
    ...     "SELECT * FROM large_table",
    ...     db,
    ...     chunksize=10000,
    ...     show_progress=True
    ... )
    >>>
    >>> # With date parsing
    >>> df = read_sql(
    ...     "SELECT * FROM orders",
    ...     db,
    ...     parse_dates=['order_date', 'ship_date']
    ... )
    """
    sql_connection = getattr(connection, 'engine', connection)

    if isinstance(query, str):
        query_text = text(query) if SQLALCHEMY_AVAILABLE else query
    else:
        query_text = query

    # Read with or without chunking
    if chunksize:
        # Chunked reading
        chunks = pd.read_sql(
            query_text,
            sql_connection,
            params=params,
            chunksize=chunksize,
            index_col=index_col,
            parse_dates=parse_dates,
            coerce_float=coerce_float,
            **kwargs
        )

        if show_progress and TQDM_AVAILABLE:
            # Get total rows for progress bar
            if isinstance(query, str):
                count_query = f"SELECT COUNT(*) FROM ({query}) AS count_query"
                with sql_connection.connect() as conn:
                    total_rows = conn.execute(text(count_query), params or {}).scalar()

                total_chunks = (total_rows + chunksize - 1) // chunksize

                chunks = tqdm(
                    chunks,
                    total=total_chunks,
                    desc="Reading chunks",
                    unit="chunk"
                )

        # Combine chunks
        df_list = []
        for chunk in chunks:
            if optimize_dtypes:
                chunk = optimize_datatypes(chunk)
            df_list.append(chunk)

        df = pd.concat(df_list, ignore_index=True)

    else:
        # Single read
        if show_progress:
            print("⏳ Reading data from database...")

        df = pd.read_sql(
            query_text,
            sql_connection,
            params=params,
            index_col=index_col,
            parse_dates=parse_dates,
            coerce_float=coerce_float,
            **kwargs
        )

        if optimize_dtypes:
            df = optimize_datatypes(df)

        if show_progress:
            print(f"[OK] Loaded {len(df):,} rows, {len(df.columns)} columns")

    return df


def read_sql_table(
    table_name: str,
    connection: Any,
    schema: Optional[str] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    show_progress: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Read entire table into DataFrame.

    Parameters:
    -----------
    table_name : str
        Name of the table.
    connection : database connection
        Database connection.
    schema : str, optional
        Database schema name.
    columns : list, optional
        Specific columns to read.
    chunksize : int, optional
        Chunk size for large tables.
    show_progress : bool
        Show progress bar.
    **kwargs :
        Additional arguments.

    Returns:
    --------
    pd.DataFrame
        Table data.

    Examples:
    ---------
    >>> # Read entire table
    >>> df = read_sql_table('users', db)
    >>>
    >>> # Read specific columns
    >>> df = read_sql_table('users', db, columns=['name', 'email', 'age'])
    >>>
    >>> # Large table with chunking
    >>> df = read_sql_table('big_table', db, chunksize=50000, show_progress=True)
    """
    # Build query
    if columns:
        cols = ', '.join(columns)
        query = f"SELECT {cols} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"

    if schema:
        query = query.replace(table_name, f"{schema}.{table_name}")

    return read_sql(
        query,
        connection,
        chunksize=chunksize,
        show_progress=show_progress,
        **kwargs
    )


# ==================== WRITING DATA ====================

def to_sql(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    if_exists: str = 'append',
    index: bool = False,
    batch_size: int = 1000,
    show_progress: bool = False,
    method: Optional[str] = None,
    dtype: Optional[Dict] = None,
    create_index: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> int:
    """
    Enhanced version of DataFrame.to_sql with batching and progress tracking.

    Features:
    - Automatic batching for large datasets
    - Progress bars
    - Index creation
    - Better error handling

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to write.
    table_name : str
        Name of table to create/append to.
    connection : database connection
        Database connection or engine.
    if_exists : str
        'fail', 'replace', or 'append' (default: 'append').
    index : bool
        Write DataFrame index as column.
    batch_size : int
        Number of rows per batch (default: 1000).
    show_progress : bool
        Show progress bar.
    method : str, optional
        Method for SQL insertion: None, 'multi', or callable.
    dtype : dict, optional
        SQL data types for columns.
    create_index : str or list, optional
        Column(s) to create index on.
    **kwargs :
        Additional arguments passed to DataFrame.to_sql.

    Returns:
    --------
    int
        Number of rows written.

    Examples:
    ---------
    >>> # Basic append
    >>> to_sql(df, 'users', db, if_exists='append')
    >>>
    >>> # Replace table with progress bar
    >>> to_sql(df, 'users', db, if_exists='replace', show_progress=True)
    >>>
    >>> # Large dataset with batching
    >>> to_sql(
    ...     large_df,
    ...     'big_table',
    ...     db,
    ...     batch_size=10000,
    ...     show_progress=True
    ... )
    >>>
    >>> # Create index after insert
    >>> to_sql(
    ...     df,
    ...     'users',
    ...     db,
    ...     if_exists='replace',
    ...     create_index='user_id'
    ... )
    """
    sql_connection = getattr(connection, 'engine', connection)
    total_rows = len(df)

    if show_progress:
        print(f"⏳ Writing {total_rows:,} rows to table '{table_name}'...")

    # Write in batches
    if batch_size and total_rows > batch_size:
        n_batches = (total_rows + batch_size - 1) // batch_size

        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=n_batches, desc="Writing batches", unit="batch")

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size]

            # First batch creates/replaces table
            batch_if_exists = if_exists if i == 0 else 'append'

            batch.to_sql(
                table_name,
                sql_connection,
                if_exists=batch_if_exists,
                index=index,
                method=method,
                dtype=dtype,
                **kwargs
            )

            if show_progress and TQDM_AVAILABLE:
                pbar.update(1)

        if show_progress and TQDM_AVAILABLE:
            pbar.close()

    else:
        # Single batch
        df.to_sql(
            table_name,
            sql_connection,
            if_exists=if_exists,
            index=index,
            method=method,
            dtype=dtype,
            **kwargs
        )

    # Create index if requested
    if create_index and SQLALCHEMY_AVAILABLE:
        create_table_index(connection, table_name, create_index)

    if show_progress:
        print(f"[OK] Successfully wrote {total_rows:,} rows to '{table_name}'")

    return total_rows


def upsert(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    key_columns: Union[str, List[str]],
    batch_size: int = 1000,
    show_progress: bool = False,
    schema: Optional[str] = None,
    update_columns: Optional[List[str]] = None,
    conflict_constraint: Optional[str] = None,
    do_nothing: bool = False,
) -> int:
    """
    Insert or update rows (UPSERT operation).

    Updates existing rows based on key columns, inserts new rows.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to upsert.
    table_name : str
        Target table name.
    connection : database connection
        Database connection.
    key_columns : str or list
        Column(s) to use as primary key for matching.
    batch_size : int
        Batch size for processing.
    show_progress : bool
        Show progress bar.
    schema : str, optional
        PostgreSQL schema name.
    update_columns : list, optional
        PostgreSQL columns to update on conflict. Defaults to all non-key columns.
    conflict_constraint : str, optional
        PostgreSQL unique or primary-key constraint to target instead of columns.
    do_nothing : bool
        Skip conflicting PostgreSQL rows instead of updating them.

    Returns:
    --------
    int
        Number of rows affected.

    Examples:
    ---------
    >>> # Upsert based on user_id
    >>> upsert(df, 'users', db, key_columns='user_id')
    >>>
    >>> # Upsert with composite key
    >>> upsert(df, 'orders', db, key_columns=['order_id', 'line_item'])
    """
    if isinstance(key_columns, str):
        key_columns = [key_columns]

    if show_progress:
        print(f"⏳ Upserting {len(df):,} rows to '{table_name}'...")

    # Get database type
    sql_connection = getattr(connection, 'engine', connection)
    db_type = (
        str(sql_connection.url.drivername)
        if hasattr(sql_connection, 'url')
        else 'unknown'
    )

    rows_affected = 0

    if 'postgresql' in db_type:
        # PostgreSQL - use ON CONFLICT
        rows_affected = _upsert_postgresql(
            df,
            table_name,
            connection,
            key_columns,
            batch_size,
            show_progress,
            schema=schema,
            update_columns=update_columns,
            conflict_constraint=conflict_constraint,
            do_nothing=do_nothing,
        )

    elif 'mysql' in db_type:
        # MySQL - use ON DUPLICATE KEY UPDATE
        rows_affected = _upsert_mysql(
            df, table_name, connection, key_columns, batch_size, show_progress
        )

    else:
        # Fallback - delete then insert
        rows_affected = _upsert_generic(
            df, table_name, connection, key_columns, batch_size, show_progress
        )

    if show_progress:
        print(f"[OK] Upserted {rows_affected:,} rows")

    return rows_affected


def _upsert_postgresql(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    key_columns: List[str],
    batch_size: int,
    show_progress: bool,
    schema: Optional[str] = None,
    update_columns: Optional[List[str]] = None,
    conflict_constraint: Optional[str] = None,
    do_nothing: bool = False,
) -> int:
    """PostgreSQL UPSERT using batched ``ON CONFLICT`` execution."""
    from .postgres import upsert_postgres

    return upsert_postgres(
        df=df,
        table_name=table_name,
        connection=connection,
        key_columns=key_columns,
        schema=schema,
        update_columns=update_columns,
        conflict_constraint=conflict_constraint,
        do_nothing=do_nothing,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def _upsert_mysql(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    key_columns: List[str],
    batch_size: int,
    show_progress: bool
) -> int:
    """MySQL UPSERT using ON DUPLICATE KEY UPDATE."""
    from sqlalchemy import MetaData, Table

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=connection)

    all_columns = [col.name for col in table.columns]
    update_columns = [col for col in all_columns if col not in key_columns]

    columns_str = ', '.join(all_columns)
    placeholders = ', '.join([f':{col}' for col in all_columns])
    updates = ', '.join([f'{col} = VALUES({col})' for col in update_columns])

    query = f"""
    INSERT INTO {table_name} ({columns_str})
    VALUES ({placeholders})
    ON DUPLICATE KEY UPDATE {updates}
    """

    total_rows = 0
    with connection.connect() as conn:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            records = batch.to_dict('records')

            for record in records:
                conn.execute(text(query), record)

            conn.commit()
            total_rows += len(batch)

    return total_rows


def _upsert_generic(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    key_columns: List[str],
    batch_size: int,
    show_progress: bool
) -> int:
    """Generic UPSERT: delete existing then insert."""
    # This is less efficient but works for all databases

    def _python_scalar(value: Any) -> Any:
        return value.item() if hasattr(value, "item") else value

    # First, delete existing rows
    key_values = df[key_columns].drop_duplicates()

    with connection.connect() as conn:
        for _, row in key_values.iterrows():
            where_clause = ' AND '.join([f"{col} = :{col}" for col in key_columns])
            delete_query = f"DELETE FROM {table_name} WHERE {where_clause}"

            params = {col: _python_scalar(row[col]) for col in key_columns}
            conn.execute(text(delete_query), params)

        conn.commit()

    # Then insert all rows
    return to_sql(
        df,
        table_name,
        connection,
        if_exists='append',
        batch_size=batch_size,
        show_progress=show_progress
    )


# ==================== DATA TYPE OPTIMIZATION ====================

def optimize_datatypes(
    df: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Optimize DataFrame data types to reduce memory usage.

    Converts:
    - int64 → int8/int16/int32 (if possible)
    - float64 → float32 (if possible)
    - object → category (for low-cardinality strings)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to optimize.
    verbose : bool
        Print optimization report.

    Returns:
    --------
    pd.DataFrame
        Optimized DataFrame (copy).

    Examples:
    ---------
    >>> df_optimized = optimize_datatypes(df, verbose=True)
    >>>
    >>> # Check memory savings
    >>> original_memory = df.memory_usage(deep=True).sum() / 1024**2
    >>> optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    >>> print(f"Saved {original_memory - optimized_memory:.2f} MB")
    """
    df_optimized = df.copy()

    original_memory = df.memory_usage(deep=True).sum()

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype

        # Optimize integers
        if col_type in ['int64', 'int32', 'int16']:
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()

            if c_min >= -128 and c_max <= 127:
                df_optimized[col] = df_optimized[col].astype('int8')
            elif c_min >= -32768 and c_max <= 32767:
                df_optimized[col] = df_optimized[col].astype('int16')
            elif c_min >= -2147483648 and c_max <= 2147483647:
                df_optimized[col] = df_optimized[col].astype('int32')

        # Optimize floats
        elif col_type == 'float64':
            df_optimized[col] = df_optimized[col].astype('float32')

        # Convert low-cardinality strings to category
        elif col_type == 'object':
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])

            # If less than 50% unique values, convert to category
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')

    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    memory_saved = original_memory - optimized_memory

    if verbose:
        print("[OK] Memory Optimization Report:")
        print(f"  Original: {original_memory / 1024**2:.2f} MB")
        print(f"  Optimized: {optimized_memory / 1024**2:.2f} MB")
        print(f"  Saved: {memory_saved / 1024**2:.2f} MB ({memory_saved / original_memory * 100:.1f}%)")

    return df_optimized


def infer_sql_dtypes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infer optimal SQL data types for DataFrame columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze.

    Returns:
    --------
    dict
        Mapping of column names to SQLAlchemy types.

    Examples:
    ---------
    >>> dtypes = infer_sql_dtypes(df)
    >>> to_sql(df, 'users', db, dtype=dtypes)
    """
    if not SQLALCHEMY_AVAILABLE:
        warnings.warn("SQLAlchemy not available. Returning empty dict.")
        return {}

    dtype_map = {}

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'int8':
            dtype_map[col] = Integer
        elif col_type in ['int16', 'int32']:
            dtype_map[col] = Integer
        elif col_type == 'int64':
            dtype_map[col] = BigInteger
        elif col_type in ['float32', 'float64']:
            dtype_map[col] = Float
        elif col_type == 'bool':
            dtype_map[col] = Boolean
        elif col_type == 'datetime64[ns]':
            dtype_map[col] = DateTime
        elif col_type == 'object':
            # Check max length
            max_len = df[col].astype(str).str.len().max()
            if max_len <= 255:
                dtype_map[col] = String(max_len)
            else:
                dtype_map[col] = Text
        elif col_type == 'category':
            max_len = df[col].astype(str).str.len().max()
            dtype_map[col] = String(max_len)

    return dtype_map


# ==================== TABLE OPERATIONS ====================

def get_table_info(
    table_name: str,
    connection: Any,
    schema: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get comprehensive information about a table.

    Parameters:
    -----------
    table_name : str
        Table name.
    connection : database connection
        Database connection.
    schema : str, optional
        Schema name.

    Returns:
    --------
    dict
        Table information including columns, types, row count, size.

    Examples:
    ---------
    >>> info = get_table_info('users', db)
    >>> print(f"Table: {info['name']}")
    >>> print(f"Columns: {len(info['columns'])}")
    >>> print(f"Rows: {info['row_count']:,}")
    >>> print(f"Size: {info['size_mb']:.2f} MB")
    """
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy required for table introspection")

    inspector = inspect(connection)

    # Get columns
    columns = inspector.get_columns(table_name, schema=schema)

    # Get row count
    full_table_name = f"{schema}.{table_name}" if schema else table_name

    with connection.connect() as conn:
        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {full_table_name}"))
        row_count = count_result.scalar()

    # Get primary keys
    pk = inspector.get_pk_constraint(table_name, schema=schema)

    # Get foreign keys
    fks = inspector.get_foreign_keys(table_name, schema=schema)

    # Get indexes
    indexes = inspector.get_indexes(table_name, schema=schema)

    # Try to get table size (database-specific)
    size_mb = None
    try:
        db_type = str(connection.url.drivername)

        if 'postgresql' in db_type:
            size_query = f"""
            SELECT pg_total_relation_size('{full_table_name}')::bigint / (1024*1024) AS size_mb
            """
        elif 'mysql' in db_type:
            size_query = f"""
            SELECT
                ROUND((data_length + index_length) / 1024 / 1024, 2) AS size_mb
            FROM information_schema.tables
            WHERE table_name = '{table_name}'
            """
        else:
            size_query = None

        if size_query:
            with connection.connect() as conn:
                size_result = conn.execute(text(size_query))
                size_mb = size_result.scalar()
    except Exception:
        pass

    return {
        'name': table_name,
        'schema': schema,
        'columns': columns,
        'column_names': [col['name'] for col in columns],
        'row_count': row_count,
        'size_mb': size_mb,
        'primary_key': pk,
        'foreign_keys': fks,
        'indexes': indexes
    }


def create_table_index(
    connection: Any,
    table_name: str,
    columns: Union[str, List[str]],
    index_name: Optional[str] = None,
    unique: bool = False
):
    """
    Create index on table column(s).

    Parameters:
    -----------
    connection : database connection
        Database connection.
    table_name : str
        Table name.
    columns : str or list
        Column(s) to index.
    index_name : str, optional
        Index name (auto-generated if None).
    unique : bool
        Create unique index.

    Examples:
    ---------
    >>> # Single column index
    >>> create_table_index(db, 'users', 'email')
    >>>
    >>> # Composite index
    >>> create_table_index(db, 'orders', ['user_id', 'order_date'])
    >>>
    >>> # Unique index
    >>> create_table_index(db, 'users', 'email', unique=True)
    """
    if isinstance(columns, str):
        columns = [columns]

    # Generate index name if not provided
    if not index_name:
        col_str = '_'.join(columns)
        index_name = f"idx_{table_name}_{col_str}"

    columns_str = ', '.join(columns)
    unique_str = 'UNIQUE' if unique else ''

    query = f"CREATE {unique_str} INDEX {index_name} ON {table_name} ({columns_str})"

    with connection.connect() as conn:
        conn.execute(text(query))
        conn.commit()

    print(f"[OK] Created index '{index_name}' on {table_name}({columns_str})")


def drop_table_index(
    connection: Any,
    index_name: str,
    table_name: Optional[str] = None
):
    """
    Drop index from table.

    Parameters:
    -----------
    connection : database connection
        Database connection.
    index_name : str
        Index name.
    table_name : str, optional
        Table name (required for some databases).

    Examples:
    ---------
    >>> drop_table_index(db, 'idx_users_email')
    """
    db_type = str(connection.url.drivername) if hasattr(connection, 'url') else 'unknown'

    if 'mysql' in db_type and table_name:
        query = f"DROP INDEX {index_name} ON {table_name}"
    else:
        query = f"DROP INDEX {index_name}"

    with connection.connect() as conn:
        conn.execute(text(query))
        conn.commit()

    print(f"[OK] Dropped index '{index_name}'")


# ==================== BULK OPERATIONS ====================

def bulk_insert(
    df: pd.DataFrame,
    table_name: str,
    connection: Any,
    batch_size: int = 1000,
    show_progress: bool = True
) -> int:
    """
    Fast bulk insert using database-specific methods.

    Much faster than to_sql for large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to insert.
    table_name : str
        Target table.
    connection : database connection
        Database connection.
    batch_size : int
        Rows per batch.
    show_progress : bool
        Show progress bar.

    Returns:
    --------
    int
        Number of rows inserted.

    Examples:
    ---------
    >>> # Fast insert of 1 million rows
    >>> bulk_insert(large_df, 'big_table', db, batch_size=10000)
    """
    return to_sql(
        df,
        table_name,
        connection,
        if_exists='append',
        batch_size=batch_size,
        show_progress=show_progress,
        method='multi'  # Use multi-row INSERT
    )


def execute_query(
    query: str,
    connection: Any,
    params: Optional[Dict[str, Any]] = None,
    return_results: bool = False
) -> Optional[pd.DataFrame]:
    """
    Execute arbitrary SQL query.

    Parameters:
    -----------
    query : str
        SQL query to execute.
    connection : database connection
        Database connection.
    params : dict, optional
        Query parameters.
    return_results : bool
        Return results as DataFrame (for SELECT queries).

    Returns:
    --------
    pd.DataFrame or None
        Query results if return_results=True.

    Examples:
    ---------
    >>> # Execute DDL
    >>> execute_query("CREATE INDEX idx_email ON users(email)", db)
    >>>
    >>> # Execute with results
    >>> df = execute_query(
    ...     "SELECT * FROM users WHERE age > :min_age",
    ...     db,
    ...     params={'min_age': 25},
    ...     return_results=True
    ... )
    """
    if return_results:
        return read_sql(query, connection, params=params)
    else:
        with connection.connect() as conn:
            if params:
                conn.execute(text(query), params)
            else:
                conn.execute(text(query))
            conn.commit()

        print("[OK] Query executed successfully")
        return None


@contextmanager
def transaction(connection: Any):
    """
    Context manager for database transactions.

    Automatically commits on success, rolls back on error.

    Parameters:
    -----------
    connection : database connection
        Database connection.

    Yields:
    -------
    connection
        Database connection within transaction.

    Examples:
    ---------
    >>> with transaction(db) as conn:
    ...     conn.execute("INSERT INTO users (name) VALUES ('Alice')")
    ...     conn.execute("INSERT INTO logs (action) VALUES ('user_created')")
    ...     # Both committed together, or both rolled back on error
    """
    conn = connection.connect()
    trans = conn.begin()

    try:
        yield conn
        trans.commit()
        print("[OK] Transaction committed")
    except Exception as e:
        trans.rollback()
        print(f"[ERROR] Transaction rolled back: {str(e)}")
        raise
    finally:
        conn.close()

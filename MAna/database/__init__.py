"""
Database integration module for seamless data science workflows.

Provides:
- Connection managers for multiple databases (PostgreSQL, MySQL, SQLite, MongoDB)
- Query builder (Pythonic SQL generation)
- Pandas ↔ SQL helpers
- MongoDB utilities
- Connection pooling
- Query optimization
"""

from .connectors import (
    ConnectionConfig,
    DatabaseConnector,
    PostgreSQLConnector,
    MySQLConnector,
    SQLiteConnector,
    connect_to_database
)

from .query_builder import (
    Query,
    select, insert, update, delete
)

from .sql_helpers import (
    read_sql,
    read_sql_table,
    to_sql,
    upsert,
    bulk_insert,
    execute_query,
    get_table_info,
    optimize_datatypes,
    create_table_index,
    drop_table_index,
    transaction,
)

from .mongodb_helpers import (
    MongoDBConnector,
    get_mongo_collection,
    mongo_to_dataframe,
    dataframe_to_mongo,
    upsert_document,
    save_document,
)

from .postgres import (
    connect_postgres,
    copy_from_dataframe,
    copy_to_dataframe,
    create_postgres_index,
    create_postgres_table_from_dataframe,
    create_schema_if_not_exists,
    explain_postgres,
    get_postgres_constraints,
    get_postgres_indexes,
    get_postgres_slow_queries,
    get_postgres_table_health,
    get_postgres_table_sizes,
    list_postgres_schemas,
    list_postgres_tables,
    generate_postgres_create_table,
    infer_postgres_dtypes,
    postgres_database_size,
    postgres_upsert_sql,
    postgres_url,
    postgres_url_from_env,
    quote_postgres_identifier,
    upsert_postgres,
    vacuum_analyze_postgres,
)

upsert_mongo_document = upsert_document
save_mongo_document = save_document

__all__ = [
    # Connectors
    'ConnectionConfig',
    'DatabaseConnector',
    'PostgreSQLConnector',
    'MySQLConnector',
    'SQLiteConnector',
    'MongoDBConnector',
    'connect_to_database',

    # Query Builder
    'Query',
    'select',
    'insert',
    'update',
    'delete',

    # SQL Helpers
    'read_sql',
    'read_sql_table',
    'to_sql',
    'upsert',
    'bulk_insert',
    'execute_query',
    'get_table_info',
    'optimize_datatypes',
    'create_table_index',
    'drop_table_index',
    'transaction',

    # PostgreSQL
    'postgres_url',
    'postgres_url_from_env',
    'connect_postgres',
    'quote_postgres_identifier',
    'create_schema_if_not_exists',
    'infer_postgres_dtypes',
    'generate_postgres_create_table',
    'create_postgres_table_from_dataframe',
    'create_postgres_index',
    'list_postgres_schemas',
    'list_postgres_tables',
    'get_postgres_table_sizes',
    'get_postgres_indexes',
    'get_postgres_constraints',
    'get_postgres_table_health',
    'get_postgres_slow_queries',
    'postgres_database_size',
    'explain_postgres',
    'postgres_upsert_sql',
    'upsert_postgres',
    'copy_from_dataframe',
    'copy_to_dataframe',
    'vacuum_analyze_postgres',

    # MongoDB
    'get_mongo_collection',
    'mongo_to_dataframe',
    'dataframe_to_mongo',
    'upsert_document',
    'save_document',
    'upsert_mongo_document',
    'save_mongo_document',
]

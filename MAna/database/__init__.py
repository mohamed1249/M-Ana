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

    # MongoDB
    'get_mongo_collection',
    'mongo_to_dataframe',
    'dataframe_to_mongo',
    'upsert_document',
    'save_document',
    'upsert_mongo_document',
    'save_mongo_document',
]

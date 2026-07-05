"""
Database Connection Management Module

Provides unified interface for connecting to multiple database types:
- PostgreSQL
- MySQL
- SQLite
- SQL Server
- MongoDB

Features:
- Connection pooling
- Context managers (auto-close)
- Credential management (env vars, config files)
- Retry logic
- Connection health checks
"""

import importlib.util
import os
import re
import warnings
from typing import Optional, Dict, Any, Union, List, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
from urllib.parse import quote_plus

try:
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.engine import Engine
    from sqlalchemy.pool import QueuePool, NullPool, StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    warnings.warn("SQLAlchemy not installed. Install with: pip install sqlalchemy")

PSYCOPG2_AVAILABLE = importlib.util.find_spec("psycopg2") is not None
PYMYSQL_AVAILABLE = importlib.util.find_spec("pymysql") is not None

try:
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
except ImportError:
    pass


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_sql_identifier(identifier: str) -> str:
    """Validate and quote a simple SQL identifier.

    This is intentionally conservative because helper-generated table and
    column names cannot be bound as SQL parameters.
    """
    if not isinstance(identifier, str) or not _SQL_IDENTIFIER_RE.match(identifier):
        raise ValueError(
            "SQL identifiers must start with a letter/underscore and contain "
            "only letters, numbers, and underscores"
        )
    return f'"{identifier}"'


@dataclass
class ConnectionConfig:
    """
    Database connection configuration.

    Attributes:
    -----------
    host : str
        Database host (e.g., 'localhost', 'db.example.com').
    port : int
        Port number.
    database : str
        Database name.
    username : str
        Username for authentication.
    password : str
        Password for authentication.
    db_type : str
        Database type: 'postgresql', 'mysql', 'sqlite', 'mssql', 'mongodb'.
    pool_size : int
        Connection pool size.
    max_overflow : int
        Max connections beyond pool_size.
    pool_recycle : int
        Recycle connections after N seconds.
    echo : bool
        Echo SQL statements (for debugging).
    ssl : bool
        Use SSL/TLS connection.
    options : dict
        Additional connection options.
    """

    host: str = 'localhost'
    port: Optional[int] = None
    database: str = ''
    username: str = ''
    password: str = ''
    db_type: str = 'postgresql'
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 3600
    echo: bool = False
    ssl: bool = False
    options: Dict[str, Any] = None

    def __post_init__(self):
        # Set default ports if not provided
        if self.port is None:
            default_ports = {
                'postgresql': 5432,
                'mysql': 3306,
                'mssql': 1433,
                'mongodb': 27017,
                'sqlite': None
            }
            self.port = default_ports.get(self.db_type)

        if self.options is None:
            self.options = {}

    @classmethod
    def from_env(cls, prefix: str = 'DB') -> 'ConnectionConfig':
        """
        Load configuration from environment variables.

        Expected environment variables:
        - {prefix}_TYPE (e.g., DB_TYPE=postgresql)
        - {prefix}_HOST
        - {prefix}_PORT
        - {prefix}_DATABASE
        - {prefix}_USERNAME
        - {prefix}_PASSWORD

        Parameters:
        -----------
        prefix : str
            Prefix for environment variables (default: 'DB').

        Returns:
        --------
        ConnectionConfig
            Configuration loaded from environment.

        Examples:
        ---------
        >>> # In .env file:
        >>> # DB_TYPE=postgresql
        >>> # DB_HOST=localhost
        >>> # DB_DATABASE=mydb
        >>> # DB_USERNAME=user
        >>> # DB_PASSWORD=pass
        >>>
        >>> config = ConnectionConfig.from_env()
        >>> db = DatabaseConnector(config)
        """
        return cls(
            db_type=os.getenv(f'{prefix}_TYPE', 'postgresql'),
            host=os.getenv(f'{prefix}_HOST', 'localhost'),
            port=int(os.getenv(f'{prefix}_PORT')) if os.getenv(f'{prefix}_PORT') else None,
            database=os.getenv(f'{prefix}_DATABASE', ''),
            username=os.getenv(f'{prefix}_USERNAME', ''),
            password=os.getenv(f'{prefix}_PASSWORD', ''),
            ssl=os.getenv(f'{prefix}_SSL', 'false').lower() == 'true'
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConnectionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'ConnectionConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_connection_string(self) -> str:
        """
        Generate SQLAlchemy connection string.

        Returns:
        --------
        str
            Connection string (e.g., 'postgresql://user:pass@host:port/db').
        """
        if self.db_type == 'sqlite':
            # SQLite uses file path
            return f'sqlite:///{self.database}'

        # URL-encode username and password
        username = quote_plus(self.username) if self.username else ''
        password = quote_plus(self.password) if self.password else ''

        # Build connection string
        if username and password:
            auth = f'{username}:{password}@'
        elif username:
            auth = f'{username}@'
        else:
            auth = ''

        # Database type mapping
        db_type_map = {
            'postgresql': 'postgresql',
            'postgres': 'postgresql',
            'mysql': 'mysql+pymysql',
            'mssql': 'mssql+pyodbc',
            'sqlserver': 'mssql+pyodbc'
        }

        db_type = db_type_map.get(self.db_type, self.db_type)

        # Build URL
        if self.port:
            conn_str = f'{db_type}://{auth}{self.host}:{self.port}/{self.database}'
        else:
            conn_str = f'{db_type}://{auth}{self.host}/{self.database}'

        # Add SSL if needed
        if self.ssl and self.db_type in ['postgresql', 'mysql']:
            conn_str += '?ssl=true'

        return conn_str


class DatabaseConnector:
    """
    Universal database connector with connection pooling and management.

    Supports PostgreSQL, MySQL, SQLite, SQL Server, and more via SQLAlchemy.

    Examples:
    ---------
    >>> # Method 1: From connection string
    >>> db = DatabaseConnector('postgresql://user:pass@localhost/mydb')
    >>>
    >>> # Method 2: From config
    >>> config = ConnectionConfig(
    ...     db_type='postgresql',
    ...     host='localhost',
    ...     database='mydb',
    ...     username='user',
    ...     password='pass'
    ... )
    >>> db = DatabaseConnector(config)
    >>>
    >>> # Method 3: From environment variables
    >>> config = ConnectionConfig.from_env()
    >>> db = DatabaseConnector(config)
    >>>
    >>> # Use connection
    >>> with db.connect() as conn:
    ...     result = conn.execute("SELECT * FROM users LIMIT 5")
    ...     for row in result:
    ...         print(row)
    >>>
    >>> # Get pandas DataFrame
    >>> df = db.query("SELECT * FROM users WHERE age > 25")
    >>>
    >>> # Execute with parameters (safe from SQL injection)
    >>> db.execute("INSERT INTO users (name, age) VALUES (:name, :age)",
    ...            params={'name': 'Alice', 'age': 30})
    """

    def __init__(
        self,
        connection: Union[str, ConnectionConfig, Engine],
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        echo: bool = False,
        **kwargs
    ):
        """
        Initialize database connector.

        Parameters:
        -----------
        connection : str, ConnectionConfig, or Engine
            Connection string, config object, or existing SQLAlchemy engine.
        pool_size : int
            Connection pool size (default: 5).
        max_overflow : int
            Max overflow connections (default: 10).
        pool_recycle : int
            Recycle connections after N seconds (default: 3600).
        echo : bool
            Echo SQL statements (default: False).
        **kwargs :
            Additional engine options.
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required. Install with: pip install sqlalchemy")

        # Handle different input types
        if isinstance(connection, str):
            # Connection string
            self.connection_string = connection
            self.config = None
        elif isinstance(connection, ConnectionConfig):
            # Config object
            self.config = connection
            self.connection_string = connection.to_connection_string()
            pool_size = connection.pool_size
            max_overflow = connection.max_overflow
            pool_recycle = connection.pool_recycle
            echo = connection.echo
        elif isinstance(connection, Engine):
            # Existing engine
            self.engine = connection
            self.connection_string = str(connection.url)
            self.config = None
            print("[OK] Using existing SQLAlchemy engine")
            return
        else:
            raise ValueError(f"Invalid connection type: {type(connection)}")

        # Create engine with connection pooling. Specialized connectors such
        # as SQLite can override the pool class through kwargs.
        poolclass = kwargs.pop('poolclass', QueuePool)
        engine_kwargs = {
            'poolclass': poolclass,
            'pool_recycle': pool_recycle,
            'echo': echo,
            **kwargs,
        }
        if poolclass is QueuePool:
            engine_kwargs.update(
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
        try:
            self.engine = create_engine(
                self.connection_string,
                **engine_kwargs,
            )
            print(f"[OK] Connected to database: {self._get_db_info()}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def _get_db_info(self) -> str:
        """Get database info for display (without exposing credentials)."""
        url = self.engine.url
        if url.database:
            return f"{url.drivername}://{url.host}/{url.database}"
        return f"{url.drivername}://{url.host}"

    @contextmanager
    def connect(self):
        """
        Context manager for database connections.

        Automatically closes connection when done.

        Yields:
        -------
        connection
            Database connection object.

        Examples:
        ---------
        >>> with db.connect() as conn:
        ...     result = conn.execute("SELECT COUNT(*) FROM users")
        ...     count = result.scalar()
        """
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        commit: bool = True
    ) -> Any:
        """
        Execute SQL query (INSERT, UPDATE, DELETE, etc.).

        Uses parameterized queries to prevent SQL injection.

        Parameters:
        -----------
        query : str
            SQL query with parameter placeholders (:param_name).
        params : dict, optional
            Query parameters.
        commit : bool
            Commit transaction (default: True).

        Returns:
        --------
        result
            Query result.

        Examples:
        ---------
        >>> # Insert with parameters
        >>> db.execute(
        ...     "INSERT INTO users (name, age) VALUES (:name, :age)",
        ...     params={'name': 'Bob', 'age': 25}
        ... )
        >>>
        >>> # Update
        >>> db.execute(
        ...     "UPDATE users SET age = :age WHERE name = :name",
        ...     params={'age': 26, 'name': 'Bob'}
        ... )
        >>>
        >>> # Delete
        >>> db.execute(
        ...     "DELETE FROM users WHERE age < :min_age",
        ...     params={'min_age': 18}
        ... )
        """
        with self.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))

            if commit:
                conn.commit()

            return result

    def execute_many(
        self,
        query: str,
        parameter_sets: Sequence[Mapping[str, Any]],
        commit: bool = True
    ) -> Any:
        """
        Execute a parameterized SQL statement for many rows.

        Parameters:
        -----------
        query : str
            SQL query with named placeholders (``:param_name``).
        parameter_sets : sequence of mappings
            One parameter mapping per row.
        commit : bool
            Commit transaction (default: True).

        Returns:
        --------
        result
            SQLAlchemy result object, or ``None`` when there are no rows.

        Examples:
        ---------
        >>> db.execute_many(
        ...     "INSERT INTO users (name, age) VALUES (:name, :age)",
        ...     [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
        ... )
        """
        rows = [dict(params) for params in parameter_sets]
        if not rows:
            return None

        statement = text(query)
        if commit:
            with self.engine.begin() as conn:
                return conn.execute(statement, rows)

        with self.connect() as conn:
            return conn.execute(statement, rows)

    def query(self, sql: str, params: Optional[Dict[str, Any]] = None):
        """
        Execute SELECT query and return pandas DataFrame.

        Parameters:
        -----------
        sql : str
            SQL SELECT query.
        params : dict, optional
            Query parameters.

        Returns:
        --------
        pd.DataFrame
            Query results as DataFrame.

        Examples:
        ---------
        >>> df = db.query("SELECT * FROM users WHERE age > :min_age",
        ...               params={'min_age': 25})
        """
        import pandas as pd

        with self.connect() as conn:
            if params:
                df = pd.read_sql(text(sql), conn, params=params)
            else:
                df = pd.read_sql(text(sql), conn)

        return df

    def get_tables(self) -> List[str]:
        """
        Get list of all tables in database.

        Returns:
        --------
        list
            List of table names.

        Examples:
        ---------
        >>> tables = db.get_tables()
        >>> print(f"Found {len(tables)} tables: {tables}")
        """
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table.

        Parameters:
        -----------
        table_name : str
            Name of the table.

        Returns:
        --------
        list
            List of column info dicts.

        Examples:
        ---------
        >>> columns = db.get_columns('users')
        >>> for col in columns:
        ...     print(f"{col['name']}: {col['type']}")
        """
        inspector = inspect(self.engine)
        return inspector.get_columns(table_name)

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists.

        Parameters:
        -----------
        table_name : str
            Table name.

        Returns:
        --------
        bool
            True if table exists.
        """
        return table_name in self.get_tables()

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive table information.

        Parameters:
        -----------
        table_name : str
            Table name.

        Returns:
        --------
        dict
            Table metadata including columns, row count, size.

        Examples:
        ---------
        >>> info = db.get_table_info('users')
        >>> print(f"Table: {info['name']}")
        >>> print(f"Columns: {len(info['columns'])}")
        >>> print(f"Rows: {info['row_count']:,}")
        """
        if not self.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        inspector = inspect(self.engine)

        # Get columns
        columns = inspector.get_columns(table_name)

        # Get row count
        with self.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()

        # Get primary keys
        pk = inspector.get_pk_constraint(table_name)

        # Get indexes
        indexes = inspector.get_indexes(table_name)

        return {
            'name': table_name,
            'columns': columns,
            'row_count': row_count,
            'primary_key': pk,
            'indexes': indexes
        }

    def test_connection(self) -> bool:
        """
        Test if database connection is alive.

        Returns:
        --------
        bool
            True if connection successful.

        Examples:
        ---------
        >>> if db.test_connection():
        ...     print("Database is accessible")
        ... else:
        ...     print("Database connection failed")
        """
        try:
            with self.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"[ERROR] Connection test failed: {str(e)}")
            return False

    def close(self):
        """Close all connections in the pool."""
        self.engine.dispose()
        print("[OK] Database connections closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self):
        return f"DatabaseConnector({self._get_db_info()})"


# ==================== SPECIALIZED CONNECTORS ====================

class PostgreSQLConnector(DatabaseConnector):
    """
    PostgreSQL-specific connector with additional features.

    Examples:
    ---------
    >>> db = PostgreSQLConnector(
    ...     host='localhost',
    ...     database='mydb',
    ...     username='user',
    ...     password='pass'
    ... )
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = '',
        username: str = '',
        password: str = '',
        **kwargs
    ):
        config = ConnectionConfig(
            db_type='postgresql',
            host=host,
            port=port,
            database=database,
            username=username,
            password=password
        )
        super().__init__(config, **kwargs)

    def get_schemas(self) -> List[str]:
        """Get list of all schemas in PostgreSQL database."""
        query = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
        """
        result = self.query(query)
        return result['schema_name'].tolist()

    def vacuum_analyze(self, table_name: Optional[str] = None):
        """
        Run VACUUM ANALYZE to optimize table.

        Parameters:
        -----------
        table_name : str, optional
            Specific table to vacuum (vacuums all if None).
        """
        if table_name:
            query = f"VACUUM ANALYZE {table_name}"
        else:
            query = "VACUUM ANALYZE"

        self.execute(query, commit=False)
        print("[OK] VACUUM ANALYZE completed" + (f" for {table_name}" if table_name else ""))


class MySQLConnector(DatabaseConnector):
    """
    MySQL-specific connector.

    Examples:
    ---------
    >>> db = MySQLConnector(
    ...     host='localhost',
    ...     database='mydb',
    ...     username='user',
    ...     password='pass'
    ... )
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 3306,
        database: str = '',
        username: str = '',
        password: str = '',
        **kwargs
    ):
        config = ConnectionConfig(
            db_type='mysql',
            host=host,
            port=port,
            database=database,
            username=username,
            password=password
        )
        super().__init__(config, **kwargs)

    def optimize_table(self, table_name: str):
        """
        Optimize MySQL table.

        Parameters:
        -----------
        table_name : str
            Table to optimize.
        """
        self.execute(f"OPTIMIZE TABLE {table_name}", commit=False)
        print(f"[OK] Optimized table: {table_name}")


class SQLiteConnector(DatabaseConnector):
    """
    SQLite-specific connector.

    Examples:
    ---------
    >>> # In-memory database
    >>> db = SQLiteConnector(':memory:')
    >>>
    >>> # File-based database
    >>> db = SQLiteConnector('my_database.db')
    """

    def __init__(self, database: str = ':memory:', **kwargs):
        config = ConnectionConfig(
            db_type='sqlite',
            database=database
        )
        # An in-memory SQLite database must reuse one connection or each
        # operation sees a different empty database.
        if database == ':memory:':
            kwargs['poolclass'] = StaticPool
            kwargs.setdefault('connect_args', {'check_same_thread': False})
        else:
            kwargs['poolclass'] = NullPool
        super().__init__(config, **kwargs)

    def initialize_schema(self, statements: Union[str, Sequence[str]]) -> int:
        """
        Execute one or more SQLite schema statements in a single transaction.

        Parameters:
        -----------
        statements : str or sequence of str
            DDL statements such as ``CREATE TABLE IF NOT EXISTS ...``.

        Returns:
        --------
        int
            Number of non-empty statements executed.
        """
        if isinstance(statements, str):
            schema_statements = [
                statement.strip()
                for statement in statements.split(";")
                if statement.strip()
            ]
        else:
            schema_statements = [
                str(statement).strip().rstrip(";")
                for statement in statements
                if str(statement).strip()
            ]

        if not schema_statements:
            return 0

        with self.engine.begin() as conn:
            for statement in schema_statements:
                conn.execute(text(statement))

        return len(schema_statements)

    def insert_or_ignore(
        self,
        table_name: str,
        rows: Sequence[Mapping[str, Any]]
    ) -> int:
        """
        Insert rows into a SQLite table while ignoring uniqueness conflicts.

        Table and column names are strictly validated, while values are passed
        through bound parameters.

        Parameters:
        -----------
        table_name : str
            Destination table name.
        rows : sequence of mappings
            Records to insert. All rows must share the same columns.

        Returns:
        --------
        int
            Number of rows SQLite reports as inserted.
        """
        normalized_rows = [dict(row) for row in rows]
        if not normalized_rows:
            return 0

        columns = list(normalized_rows[0])
        if not columns:
            raise ValueError("rows must contain at least one column")

        expected_columns = set(columns)
        for row in normalized_rows:
            if set(row) != expected_columns:
                raise ValueError("all rows must contain the same columns")

        quoted_table = _quote_sql_identifier(table_name)
        quoted_columns = ", ".join(_quote_sql_identifier(column) for column in columns)
        placeholders = ", ".join(f":{column}" for column in columns)
        query = (
            f"INSERT OR IGNORE INTO {quoted_table} "
            f"({quoted_columns}) VALUES ({placeholders})"
        )
        result = self.execute_many(query, normalized_rows)
        rowcount = getattr(result, "rowcount", None)
        return int(rowcount) if rowcount is not None and rowcount >= 0 else len(normalized_rows)

    def vacuum(self):
        """Optimize SQLite database file."""
        self.execute("VACUUM", commit=False)
        print("[OK] Database vacuumed")


# ==================== MONGODB CONNECTOR ====================

class MongoDBConnector:
    """
    MongoDB connector for NoSQL operations.

    Examples:
    ---------
    >>> db = MongoDBConnector(
    ...     host='localhost',
    ...     database='mydb',
    ...     username='user',
    ...     password='pass'
    ... )
    >>>
    >>> # Get collection
    >>> users = db.get_collection('users')
    >>>
    >>> # Find documents
    >>> results = users.find({'age': {'$gt': 25}})
    >>> for doc in results:
    ...     print(doc)
    >>>
    >>> # Convert to DataFrame
    >>> df = db.to_dataframe('users', query={'age': {'$gt': 25}})
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 27017,
        database: str = '',
        username: str = '',
        password: str = '',
        auth_source: str = 'admin',
        **kwargs
    ):
        """
        Initialize MongoDB connector.

        Parameters:
        -----------
        host : str
            MongoDB host.
        port : int
            MongoDB port (default: 27017).
        database : str
            Database name.
        username : str
            Username.
        password : str
            Password.
        auth_source : str
            Authentication database (default: 'admin').
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required. Install with: pip install pymongo")

        # Build connection string
        if username and password:
            conn_str = f"mongodb://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}/{database}?authSource={auth_source}"
        else:
            conn_str = f"mongodb://{host}:{port}/{database}"

        try:
            self.client = pymongo.MongoClient(conn_str, **kwargs)
            self.db = self.client[database]

            # Test connection
            self.client.server_info()
            print(f"[OK] Connected to MongoDB: {host}:{port}/{database}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

        self.database_name = database

    @classmethod
    def from_uri(
        cls,
        uri: str,
        database: Optional[str] = None,
        server_selection_timeout_ms: int = 5000,
        ping: bool = True,
        **kwargs
    ) -> "MongoDBConnector":
        """
        Create a MongoDB connector from a full MongoDB URI.

        This preserves URI options such as ``authSource`` and works with both
        ``mongodb://`` and ``mongodb+srv://`` connection strings.
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required. Install with: pip install pymongo")

        from urllib.parse import urlparse

        parsed = urlparse(uri)
        database_name = database or parsed.path.lstrip("/")
        if not database_name:
            raise ValueError(
                "database must be provided when it is not present in the MongoDB URI"
            )

        kwargs.setdefault("serverSelectionTimeoutMS", server_selection_timeout_ms)
        try:
            client = pymongo.MongoClient(uri, **kwargs)
            if ping:
                client.admin.command("ping")

            connector = cls.__new__(cls)
            connector.client = client
            connector.db = client[database_name]
            connector.database_name = database_name
            print(f"[OK] Connected to MongoDB: {database_name}")
            return connector
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

    def get_collection(self, collection_name: str):
        """
        Get MongoDB collection.

        Parameters:
        -----------
        collection_name : str
            Collection name.

        Returns:
        --------
        pymongo.collection.Collection
            MongoDB collection object.
        """
        return self.db[collection_name]

    def list_collections(self) -> List[str]:
        """Get list of all collections in database."""
        return self.db.list_collection_names()

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        return collection_name in self.list_collections()

    def upsert_document(
        self,
        collection_name: str,
        filter_query: Mapping[str, Any],
        document: Mapping[str, Any],
        set_on_insert: Optional[Mapping[str, Any]] = None,
        **kwargs
    ):
        """Atomically insert or update one MongoDB document."""
        if not filter_query:
            raise ValueError("filter_query must not be empty")

        update = {"$set": dict(document)}
        if set_on_insert:
            update["$setOnInsert"] = dict(set_on_insert)

        collection = self.get_collection(collection_name)
        return collection.update_one(dict(filter_query), update, upsert=True, **kwargs)

    def to_dataframe(
        self,
        collection_name: str,
        query: Optional[Dict] = None,
        projection: Optional[Dict] = None,
        limit: Optional[int] = None
    ):
        """
        Convert MongoDB collection to pandas DataFrame.

        Parameters:
        -----------
        collection_name : str
            Collection name.
        query : dict, optional
            MongoDB query filter.
        projection : dict, optional
            Fields to include/exclude.
        limit : int, optional
            Maximum documents to retrieve.

        Returns:
        --------
        pd.DataFrame
            Collection data as DataFrame.

        Examples:
        ---------
        >>> # Get all documents
        >>> df = db.to_dataframe('users')
        >>>
        >>> # With query filter
        >>> df = db.to_dataframe('users', query={'age': {'$gt': 25}})
        >>>
        >>> # With projection (only specific fields)
        >>> df = db.to_dataframe('users', projection={'name': 1, 'age': 1, '_id': 0})
        >>>
        >>> # With limit
        >>> df = db.to_dataframe('users', limit=100)
        """
        import pandas as pd

        collection = self.get_collection(collection_name)

        # Build cursor
        cursor = collection.find(query or {}, projection)

        if limit:
            cursor = cursor.limit(limit)

        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))

        return df

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        print("[OK] MongoDB connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return f"MongoDBConnector(database={self.database_name})"


# ==================== CONVENIENCE FUNCTION ====================

def connect_to_database(
    connection: Union[str, Dict, ConnectionConfig],
    **kwargs
) -> Union[DatabaseConnector, MongoDBConnector]:
    """
    Universal database connection function.

    Auto-detects database type and returns appropriate connector.

    Parameters:
    -----------
    connection : str, dict, or ConnectionConfig
        Connection string, config dict, or config object.
    **kwargs :
        Additional connection options.

    Returns:
    --------
    DatabaseConnector or MongoDBConnector
        Connected database object.

    Examples:
    ---------
    >>> # From connection string
    >>> db = connect_to_database('postgresql://user:pass@localhost/mydb')
    >>>
    >>> # From dict
    >>> db = connect_to_database({
    ...     'db_type': 'mysql',
    ...     'host': 'localhost',
    ...     'database': 'mydb',
    ...     'username': 'user',
    ...     'password': 'pass'
    ... })
    >>>
    >>> # From environment variables
    >>> db = connect_to_database('env')  # Loads from .env file
    >>>
    >>> # MongoDB
    >>> db = connect_to_database('mongodb://localhost/mydb')
    """
    # Handle 'env' shortcut
    if isinstance(connection, str) and connection.lower() == 'env':
        config = ConnectionConfig.from_env()
        return DatabaseConnector(config, **kwargs)

    # Handle connection string
    if isinstance(connection, str):
        # Detect MongoDB
        if connection.startswith('mongodb://') or connection.startswith('mongodb+srv://'):
            return MongoDBConnector.from_uri(connection, **kwargs)
        else:
            # SQL database
            return DatabaseConnector(connection, **kwargs)

    # Handle dict
    if isinstance(connection, dict):
        config = ConnectionConfig.from_dict(connection)

        if config.db_type == 'mongodb':
            return MongoDBConnector(
                host=config.host,
                port=config.port,
                database=config.database,
                username=config.username,
                password=config.password,
                **kwargs
            )
        else:
            return DatabaseConnector(config, **kwargs)

    # Handle ConnectionConfig
    if isinstance(connection, ConnectionConfig):
        if connection.db_type == 'mongodb':
            return MongoDBConnector(
                host=connection.host,
                port=connection.port,
                database=connection.database,
                username=connection.username,
                password=connection.password,
                **kwargs
            )
        else:
            return DatabaseConnector(connection, **kwargs)

    raise ValueError(f"Invalid connection type: {type(connection)}")

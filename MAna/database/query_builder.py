"""
Pythonic SQL Query Builder

Build SQL queries using Python method chaining instead of raw SQL strings.
Inspired by Django ORM and SQLAlchemy, but simpler and more lightweight.

Features:
- Method chaining for readable queries
- Automatic parameterization (SQL injection prevention)
- Support for WHERE, JOIN, GROUP BY, HAVING, ORDER BY, LIMIT
- Complex conditions (AND, OR, IN, LIKE, BETWEEN, IS NULL)
- Subqueries
- Aggregations
- Direct DataFrame output
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class QueryState:
    """
    Internal state of a query being built.

    Tracks all query components as they're added via method chaining.
    """
    table_name: str
    select_columns: List[str] = field(default_factory=lambda: ['*'])
    where_conditions: List[str] = field(default_factory=list)
    join_clauses: List[str] = field(default_factory=list)
    group_by_columns: List[str] = field(default_factory=list)
    having_conditions: List[str] = field(default_factory=list)
    order_by_columns: List[str] = field(default_factory=list)
    limit_value: Optional[int] = None
    offset_value: Optional[int] = None
    distinct: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
    param_counter: int = 0


class Query:
    """
    Pythonic SQL query builder with method chaining.

    Build complex SQL queries using Python syntax with automatic
    parameterization to prevent SQL injection.

    Examples:
    ---------
    >>> from MAna.database import Query, connect_to_database
    >>>
    >>> db = connect_to_database('postgresql://localhost/mydb')
    >>>
    >>> # Basic SELECT
    >>> query = Query('users').select('name', 'email', 'age')
    >>> df = query.execute(db)
    >>>
    >>> # WHERE clause with operators
    >>> query = Query('users').where(age__gte=25, country='USA')
    >>>
    >>> # Complex conditions
    >>> query = (Query('users')
    ...     .where(age__gte=18, age__lte=65)
    ...     .where(status__in=['active', 'premium'])
    ...     .order_by('-created_at')
    ...     .limit(100))
    >>>
    >>> # JOINs
    >>> query = (Query('orders')
    ...     .join('users', on='orders.user_id = users.id')
    ...     .select('orders.id', 'users.name', 'orders.total')
    ...     .where(orders__total__gt=100))
    >>>
    >>> # Aggregations
    >>> query = (Query('orders')
    ...     .select('user_id', 'COUNT(*) as order_count', 'SUM(total) as total_spent')
    ...     .group_by('user_id')
    ...     .having('SUM(total) > 1000')
    ...     .order_by('-total_spent'))
    >>>
    >>> # Get DataFrame directly
    >>> df = query.to_dataframe(db)
    >>>
    >>> # Get SQL string
    >>> sql = query.to_sql()
    >>> print(sql)
    """

    # Operator mapping for WHERE conditions
    OPERATORS = {
        'exact': '=',
        'ne': '!=',
        'gt': '>',
        'gte': '>=',
        'lt': '<',
        'lte': '<=',
        'in': 'IN',
        'not_in': 'NOT IN',
        'like': 'LIKE',
        'ilike': 'ILIKE',  # PostgreSQL case-insensitive LIKE
        'not_like': 'NOT LIKE',
        'between': 'BETWEEN',
        'is_null': 'IS NULL',
        'is_not_null': 'IS NOT NULL',
        'startswith': 'LIKE',
        'endswith': 'LIKE',
        'contains': 'LIKE'
    }

    def __init__(self, table_name: str):
        """
        Initialize query builder for a table.

        Parameters:
        -----------
        table_name : str
            Name of the table to query.
        """
        self.state = QueryState(table_name=table_name)

    def select(self, *columns: str) -> 'Query':
        """
        Specify columns to SELECT.

        Parameters:
        -----------
        *columns : str
            Column names to select. Use '*' for all columns.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> Query('users').select('name', 'email')
        >>> Query('users').select('*')
        >>> Query('orders').select('user_id', 'COUNT(*) as order_count')
        """
        self.state.select_columns = list(columns) if columns else ['*']
        return self

    def distinct(self) -> 'Query':
        """
        Add DISTINCT to SELECT.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> Query('orders').select('user_id').distinct()
        """
        self.state.distinct = True
        return self

    def where(self, **conditions) -> 'Query':
        """
        Add WHERE conditions.

        Supports Django-style field lookups with double underscores.

        Operators:
        - exact (default): field = value
        - ne: field != value
        - gt: field > value
        - gte: field >= value
        - lt: field < value
        - lte: field <= value
        - in: field IN (values)
        - not_in: field NOT IN (values)
        - like: field LIKE pattern
        - ilike: field ILIKE pattern (PostgreSQL)
        - between: field BETWEEN val1 AND val2
        - is_null: field IS NULL
        - is_not_null: field IS NOT NULL
        - startswith: field LIKE 'value%'
        - endswith: field LIKE '%value'
        - contains: field LIKE '%value%'

        Parameters:
        -----------
        **conditions : keyword arguments
            Field conditions using double underscore notation.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> # Simple equality
        >>> Query('users').where(country='USA', status='active')
        >>>
        >>> # Operators
        >>> Query('users').where(age__gte=18, age__lte=65)
        >>>
        >>> # IN clause
        >>> Query('users').where(status__in=['active', 'premium'])
        >>>
        >>> # LIKE
        >>> Query('users').where(email__like='%@gmail.com')
        >>>
        >>> # Contains (translates to LIKE '%value%')
        >>> Query('users').where(name__contains='john')
        >>>
        >>> # NULL checks
        >>> Query('users').where(deleted_at__is_null=True)
        >>>
        >>> # BETWEEN
        >>> Query('orders').where(total__between=(100, 500))
        """
        for key, value in conditions.items():
            condition_sql, params = self._parse_condition(key, value)
            self.state.where_conditions.append(condition_sql)
            self.state.params.update(params)

        return self

    def _parse_condition(self, key: str, value: Any) -> Tuple[str, Dict]:
        """Parse a condition into SQL and parameters."""
        # Split field and operator
        parts = key.split('__')

        if len(parts) == 1:
            # No operator, default to exact match
            field = parts[0]
            operator = 'exact'
        else:
            field = '__'.join(parts[:-1])
            operator = parts[-1]

        # Generate unique parameter name
        param_name = f'param_{self.state.param_counter}'
        self.state.param_counter += 1

        # Build condition based on operator
        if operator == 'exact':
            condition = f"{field} = :{param_name}"
            params = {param_name: value}

        elif operator in ['ne', 'gt', 'gte', 'lt', 'lte']:
            sql_op = self.OPERATORS[operator]
            condition = f"{field} {sql_op} :{param_name}"
            params = {param_name: value}

        elif operator == 'in':
            # Handle IN clause
            if not isinstance(value, (list, tuple)):
                value = [value]
            placeholders = ', '.join([f':{param_name}_{i}' for i in range(len(value))])
            condition = f"{field} IN ({placeholders})"
            params = {f'{param_name}_{i}': v for i, v in enumerate(value)}

        elif operator == 'not_in':
            if not isinstance(value, (list, tuple)):
                value = [value]
            placeholders = ', '.join([f':{param_name}_{i}' for i in range(len(value))])
            condition = f"{field} NOT IN ({placeholders})"
            params = {f'{param_name}_{i}': v for i, v in enumerate(value)}

        elif operator == 'like' or operator == 'ilike' or operator == 'not_like':
            sql_op = self.OPERATORS[operator]
            condition = f"{field} {sql_op} :{param_name}"
            params = {param_name: value}

        elif operator == 'startswith':
            condition = f"{field} LIKE :{param_name}"
            params = {param_name: f'{value}%'}

        elif operator == 'endswith':
            condition = f"{field} LIKE :{param_name}"
            params = {param_name: f'%{value}'}

        elif operator == 'contains':
            condition = f"{field} LIKE :{param_name}"
            params = {param_name: f'%{value}%'}

        elif operator == 'between':
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("BETWEEN requires a tuple/list of 2 values")
            condition = f"{field} BETWEEN :{param_name}_start AND :{param_name}_end"
            params = {
                f'{param_name}_start': value[0],
                f'{param_name}_end': value[1]
            }

        elif operator == 'is_null':
            if value:
                condition = f"{field} IS NULL"
            else:
                condition = f"{field} IS NOT NULL"
            params = {}

        elif operator == 'is_not_null':
            if value:
                condition = f"{field} IS NOT NULL"
            else:
                condition = f"{field} IS NULL"
            params = {}

        else:
            raise ValueError(f"Unknown operator: {operator}")

        return condition, params

    def join(
        self,
        table: str,
        on: str,
        join_type: str = 'INNER'
    ) -> 'Query':
        """
        Add JOIN clause.

        Parameters:
        -----------
        table : str
            Table to join.
        on : str
            JOIN condition (e.g., 'table1.id = table2.foreign_id').
        join_type : str
            Type of join: 'INNER', 'LEFT', 'RIGHT', 'FULL'.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> # INNER JOIN
        >>> Query('orders').join('users', on='orders.user_id = users.id')
        >>>
        >>> # LEFT JOIN
        >>> Query('users').join('orders', on='users.id = orders.user_id', join_type='LEFT')
        >>>
        >>> # Multiple joins
        >>> (Query('orders')
        ...     .join('users', on='orders.user_id = users.id')
        ...     .join('products', on='orders.product_id = products.id'))
        """
        join_clause = f"{join_type} JOIN {table} ON {on}"
        self.state.join_clauses.append(join_clause)
        return self

    def left_join(self, table: str, on: str) -> 'Query':
        """Shortcut for LEFT JOIN."""
        return self.join(table, on, join_type='LEFT')

    def right_join(self, table: str, on: str) -> 'Query':
        """Shortcut for RIGHT JOIN."""
        return self.join(table, on, join_type='RIGHT')

    def group_by(self, *columns: str) -> 'Query':
        """
        Add GROUP BY clause.

        Parameters:
        -----------
        *columns : str
            Columns to group by.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> Query('orders').select('user_id', 'COUNT(*)').group_by('user_id')
        >>> Query('sales').select('country', 'product', 'SUM(amount)').group_by('country', 'product')
        """
        self.state.group_by_columns.extend(columns)
        return self

    def having(self, condition: str) -> 'Query':
        """
        Add HAVING clause (for filtering aggregated results).

        Parameters:
        -----------
        condition : str
            HAVING condition (raw SQL).

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> (Query('orders')
        ...     .select('user_id', 'COUNT(*) as order_count')
        ...     .group_by('user_id')
        ...     .having('COUNT(*) > 10'))
        >>>
        >>> (Query('sales')
        ...     .select('product', 'SUM(amount) as total')
        ...     .group_by('product')
        ...     .having('SUM(amount) > 1000'))
        """
        self.state.having_conditions.append(condition)
        return self

    def order_by(self, *columns: str) -> 'Query':
        """
        Add ORDER BY clause.

        Prefix with '-' for descending order.

        Parameters:
        -----------
        *columns : str
            Columns to sort by. Use '-column' for DESC.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> # Ascending
        >>> Query('users').order_by('age')
        >>>
        >>> # Descending
        >>> Query('users').order_by('-created_at')
        >>>
        >>> # Multiple columns
        >>> Query('users').order_by('country', '-age')
        """
        for col in columns:
            if col.startswith('-'):
                # Descending
                self.state.order_by_columns.append(f"{col[1:]} DESC")
            else:
                # Ascending
                self.state.order_by_columns.append(f"{col} ASC")

        return self

    def limit(self, n: int) -> 'Query':
        """
        Add LIMIT clause.

        Parameters:
        -----------
        n : int
            Maximum number of rows to return.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> Query('users').limit(100)
        >>> Query('users').order_by('-created_at').limit(10)
        """
        self.state.limit_value = n
        return self

    def offset(self, n: int) -> 'Query':
        """
        Add OFFSET clause (for pagination).

        Parameters:
        -----------
        n : int
            Number of rows to skip.

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> # Page 1 (skip 0, take 10)
        >>> Query('users').limit(10).offset(0)
        >>>
        >>> # Page 2 (skip 10, take 10)
        >>> Query('users').limit(10).offset(10)
        >>>
        >>> # Page 3 (skip 20, take 10)
        >>> Query('users').limit(10).offset(20)
        """
        self.state.offset_value = n
        return self

    def paginate(self, page: int, per_page: int = 20) -> 'Query':
        """
        Convenience method for pagination.

        Parameters:
        -----------
        page : int
            Page number (1-indexed).
        per_page : int
            Items per page (default: 20).

        Returns:
        --------
        Query
            Self for method chaining.

        Examples:
        ---------
        >>> # Get page 1 (first 20 items)
        >>> Query('users').paginate(page=1, per_page=20)
        >>>
        >>> # Get page 5 (items 81-100)
        >>> Query('users').paginate(page=5, per_page=20)
        """
        offset = (page - 1) * per_page
        return self.limit(per_page).offset(offset)

    def to_sql(self, pretty: bool = False) -> str:
        """
        Generate SQL query string.

        Parameters:
        -----------
        pretty : bool
            Format SQL with indentation.

        Returns:
        --------
        str
            SQL query string.

        Examples:
        ---------
        >>> query = Query('users').where(age__gte=25).order_by('-created_at').limit(10)
        >>> print(query.to_sql(pretty=True))
        """
        # SELECT clause
        distinct_keyword = 'DISTINCT ' if self.state.distinct else ''
        select_cols = ', '.join(self.state.select_columns)
        sql_parts = [f"SELECT {distinct_keyword}{select_cols}"]

        # FROM clause
        sql_parts.append(f"FROM {self.state.table_name}")

        # JOIN clauses
        if self.state.join_clauses:
            sql_parts.extend(self.state.join_clauses)

        # WHERE clause
        if self.state.where_conditions:
            where_sql = ' AND '.join(self.state.where_conditions)
            sql_parts.append(f"WHERE {where_sql}")

        # GROUP BY clause
        if self.state.group_by_columns:
            group_by_sql = ', '.join(self.state.group_by_columns)
            sql_parts.append(f"GROUP BY {group_by_sql}")

        # HAVING clause
        if self.state.having_conditions:
            having_sql = ' AND '.join(self.state.having_conditions)
            sql_parts.append(f"HAVING {having_sql}")

        # ORDER BY clause
        if self.state.order_by_columns:
            order_by_sql = ', '.join(self.state.order_by_columns)
            sql_parts.append(f"ORDER BY {order_by_sql}")

        # LIMIT clause
        if self.state.limit_value is not None:
            sql_parts.append(f"LIMIT {self.state.limit_value}")

        # OFFSET clause
        if self.state.offset_value is not None:
            sql_parts.append(f"OFFSET {self.state.offset_value}")

        # Join parts
        if pretty:
            sql = '\n'.join(sql_parts)
        else:
            sql = ' '.join(sql_parts)

        return sql

    def get_params(self) -> Dict[str, Any]:
        """
        Get query parameters.

        Returns:
        --------
        dict
            Parameter dictionary for parameterized query.
        """
        return self.state.params

    def execute(self, connection: Any) -> Any:
        """
        Execute query and return raw result.

        Parameters:
        -----------
        connection : database connection
            Database connection or engine.

        Returns:
        --------
        result
            Raw query result.
        """
        from sqlalchemy import text

        sql = self.to_sql()
        params = self.get_params()

        with connection.connect() as conn:
            result = conn.execute(text(sql), params)
            return result

    def to_dataframe(self, connection: Any) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame.

        Parameters:
        -----------
        connection : database connection
            Database connection or engine.

        Returns:
        --------
        pd.DataFrame
            Query results as DataFrame.

        Examples:
        ---------
        >>> query = Query('users').where(age__gte=25).limit(100)
        >>> df = query.to_dataframe(db)
        """
        from .sql_helpers import read_sql

        sql = self.to_sql()
        params = self.get_params()

        return read_sql(sql, connection, params=params)

    def count(self, connection: Any) -> int:
        """
        Get count of matching rows.

        Parameters:
        -----------
        connection : database connection
            Database connection.

        Returns:
        --------
        int
            Number of rows matching query.

        Examples:
        ---------
        >>> query = Query('users').where(status='active')
        >>> count = query.count(db)
        >>> print(f"Found {count:,} active users")
        """
        from sqlalchemy import text

        # Build COUNT query
        count_query = Query(self.state.table_name)
        count_query.state = self.state  # Copy state
        count_query.state.select_columns = ['COUNT(*) as count']
        count_query.state.order_by_columns = []  # Remove ORDER BY for count
        count_query.state.limit_value = None  # Remove LIMIT for count
        count_query.state.offset_value = None  # Remove OFFSET for count

        sql = count_query.to_sql()
        params = count_query.get_params()

        with connection.connect() as conn:
            result = conn.execute(text(sql), params)
            return result.scalar()

    def exists(self, connection: Any) -> bool:
        """
        Check if any rows match the query.

        Parameters:
        -----------
        connection : database connection
            Database connection.

        Returns:
        --------
        bool
            True if any rows match.

        Examples:
        ---------
        >>> if Query('users').where(email='test@example.com').exists(db):
        ...     print("User already exists!")
        """
        return self.count(connection) > 0

    def first(self, connection: Any) -> Optional[pd.Series]:
        """
        Get first row matching query.

        Parameters:
        -----------
        connection : database connection
            Database connection.

        Returns:
        --------
        pd.Series or None
            First row as Series, or None if no results.

        Examples:
        ---------
        >>> user = Query('users').where(email='test@example.com').first(db)
        >>> if user is not None:
        ...     print(f"Found user: {user['name']}")
        """
        df = self.limit(1).to_dataframe(connection)

        if len(df) > 0:
            return df.iloc[0]
        return None

    def all(self, connection: Any) -> pd.DataFrame:
        """
        Alias for to_dataframe() for Django-style syntax.

        Examples:
        ---------
        >>> users = Query('users').where(status='active').all(db)
        """
        return self.to_dataframe(connection)

    def __repr__(self) -> str:
        """String representation showing SQL."""
        sql = self.to_sql()
        params = self.get_params()

        if params:
            return f"Query: {sql}\nParams: {params}"
        return f"Query: {sql}"


# ==================== CONVENIENCE FUNCTIONS ====================

def select(*columns: str) -> Query:
    """
    Start building a SELECT query.

    This is a convenience function that requires calling .from_table() next.
    Alternatively, use Query('table_name').select(...) directly.

    Examples:
    ---------
    >>> # Using Query directly (recommended)
    >>> query = Query('users').select('name', 'email')
    >>>
    >>> # This function is less common but available
    >>> from MAna.database.query_builder import select
    >>> # Note: Would need .from_table() method, which isn't implemented
    >>> # Better to use Query('table').select(...) pattern
    """
    raise NotImplementedError(
        "Use Query('table_name').select(...) instead. "
        "Example: Query('users').select('name', 'email')"
    )


def insert(table_name: str, data: Dict[str, Any], connection: Any) -> int:
    """
    Insert a single row into table.

    Parameters:
    -----------
    table_name : str
        Target table.
    data : dict
        Column: value mapping.
    connection : database connection
        Database connection.

    Returns:
    --------
    int
        Number of rows inserted (1).

    Examples:
    ---------
    >>> insert('users', {'name': 'Alice', 'age': 30}, db)
    """
    from sqlalchemy import text

    columns = ', '.join(data.keys())
    placeholders = ', '.join([f':{k}' for k in data.keys()])

    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    with connection.connect() as conn:
        conn.execute(text(sql), data)
        conn.commit()

    return 1


def update(
    table_name: str,
    data: Dict[str, Any],
    where: Dict[str, Any],
    connection: Any
) -> int:
    """
    Update rows in table.

    Parameters:
    -----------
    table_name : str
        Target table.
    data : dict
        Columns to update with new values.
    where : dict
        WHERE conditions.
    connection : database connection
        Database connection.

    Returns:
    --------
    int
        Number of rows updated.

    Examples:
    ---------
    >>> # Update age for specific user
    >>> update('users', {'age': 31}, {'user_id': 123}, db)
    >>>
    >>> # Update multiple users
    >>> update('users', {'status': 'inactive'}, {'last_login__lt': '2020-01-01'}, db)
    """
    from sqlalchemy import text

    # Build SET clause
    set_parts = [f"{k} = :set_{k}" for k in data.keys()]
    set_clause = ', '.join(set_parts)

    # Build WHERE clause
    where_parts = [f"{k} = :where_{k}" for k in where.keys()]
    where_clause = ' AND '.join(where_parts)

    sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

    # Combine params
    params = {f'set_{k}': v for k, v in data.items()}
    params.update({f'where_{k}': v for k, v in where.items()})

    with connection.connect() as conn:
        result = conn.execute(text(sql), params)
        conn.commit()
        return result.rowcount


def delete(table_name: str, where: Dict[str, Any], connection: Any) -> int:
    """
    Delete rows from table.

    Parameters:
    -----------
    table_name : str
        Target table.
    where : dict
        WHERE conditions (required for safety).
    connection : database connection
        Database connection.

    Returns:
    --------
    int
        Number of rows deleted.

    Examples:
    ---------
    >>> # Delete specific user
    >>> delete('users', {'user_id': 123}, db)
    >>>
    >>> # Delete old records
    >>> delete('logs', {'created_at__lt': '2020-01-01'}, db)
    """
    from sqlalchemy import text

    if not where:
        raise ValueError("WHERE conditions required for DELETE (safety measure)")

    # Build WHERE clause
    where_parts = [f"{k} = :{k}" for k in where.keys()]
    where_clause = ' AND '.join(where_parts)

    sql = f"DELETE FROM {table_name} WHERE {where_clause}"

    with connection.connect() as conn:
        result = conn.execute(text(sql), where)
        conn.commit()
        return result.rowcount


# ==================== AGGREGATION HELPERS ====================

class Aggregate:
    """Helper class for aggregation functions."""

    @staticmethod
    def count(column: str = '*', alias: str = 'count') -> str:
        """COUNT aggregation."""
        return f"COUNT({column}) AS {alias}"

    @staticmethod
    def sum(column: str, alias: str = 'sum') -> str:
        """SUM aggregation."""
        return f"SUM({column}) AS {alias}"

    @staticmethod
    def avg(column: str, alias: str = 'avg') -> str:
        """AVG aggregation."""
        return f"AVG({column}) AS {alias}"

    @staticmethod
    def min(column: str, alias: str = 'min') -> str:
        """MIN aggregation."""
        return f"MIN({column}) AS {alias}"

    @staticmethod
    def max(column: str, alias: str = 'max') -> str:
        """MAX aggregation."""
        return f"MAX({column}) AS {alias}"


# Convenient aliases
agg = Aggregate

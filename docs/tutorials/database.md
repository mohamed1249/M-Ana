# PostgreSQL workflows with `MAna.database`

PostgreSQL is the primary SQL target for `MAna.database`. The module covers the
full path from a pandas DataFrame to a maintained analytical table: secure
connections, schemas, parameterized queries, bulk loading, conflict handling,
catalog inspection, indexes, transactions, and query plans.

SQLite remains useful for portable examples and small local stores. MongoDB is
available for document-oriented metadata. The production SQL examples below
use PostgreSQL.

## Install

```bash
python -m pip install "M_Ana_package[database]"
```

The database extra includes SQLAlchemy, psycopg2, PyMySQL, pymongo,
python-dotenv, and tqdm.

## Connect without putting credentials in code

Set PostgreSQL's standard environment variables in your shell or `.env` file:

```dotenv
PGHOST=localhost
PGPORT=5432
PGDATABASE=analytics
PGUSER=postgres
PGPASSWORD=change-me
PGSSLMODE=prefer
```

Then connect with pooling enabled:

```python
from MAna.database import connect_postgres

db = connect_postgres()
assert db.test_connection()
```

For a hosted database, build an escaped URL explicitly. Passwords containing
`@`, `/`, or spaces are handled correctly.

```python
import os

from MAna.database import connect_postgres, postgres_url

url = postgres_url(
    database="analytics",
    username="app_user",
    password=os.environ["APP_DB_PASSWORD"],
    host="db.example.com",
    sslmode="require",
)
db = connect_postgres(url)
```

## Create a schema and table

Schema and table names are validated and quoted by PostgreSQL-specific helpers.
Values should still be passed as query parameters.

```python
from MAna.database import create_schema_if_not_exists

create_schema_if_not_exists(db, "analytics")

db.execute("""
CREATE TABLE IF NOT EXISTS analytics.customer_features (
    customer_id BIGINT PRIMARY KEY,
    order_count INTEGER NOT NULL,
    lifetime_value DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
""")
```

For a new analytical table, MAna can infer PostgreSQL types and return the DDL
for review before executing it:

```python
from MAna.database import (
    create_postgres_index,
    create_postgres_table_from_dataframe,
    generate_postgres_create_table,
)

ddl = generate_postgres_create_table(
    events,
    "events",
    schema="analytics",
    primary_key="event_id",
    not_null_columns=["event_time"],
    dtype_overrides={"event_type": "VARCHAR(100)"},
)
print(ddl)

create_postgres_table_from_dataframe(
    events,
    "events",
    db,
    schema="analytics",
    primary_key="event_id",
    not_null_columns=["event_time"],
    dtype_overrides={"event_type": "VARCHAR(100)"},
)
create_postgres_index(
    db,
    "events",
    ["customer_id", "event_time"],
    schema="analytics",
)
```

Inference maps integers, floats, booleans, timestamps, timedeltas, UUIDs,
decimal values, bytes, and dict/list values. Object columns default to `TEXT`;
dict/list columns become `JSONB`. Use `dtype_overrides` when domain knowledge is
more precise than inference.

## Load DataFrames

Use `to_sql` for normal loads and table creation. Pass pandas options such as
`schema` through MAna:

```python
from MAna.database import to_sql

to_sql(
    features,
    "customer_features_staging",
    db,
    schema="analytics",
    if_exists="replace",
    batch_size=5_000,
    method="multi",
)
```

For a large DataFrame and an existing destination table, PostgreSQL `COPY` is
usually the faster path:

```python
from MAna.database import copy_from_dataframe, copy_to_dataframe

rows_loaded = copy_from_dataframe(
    features,
    "customer_features",
    db,
    schema="analytics",
)

round_trip = copy_to_dataframe(
    "customer_features",
    db,
    schema="analytics",
    columns=["customer_id", "order_count", "lifetime_value"],
)
```

`COPY` currently uses psycopg2 and preserves nulls with PostgreSQL's `\N`
marker. The destination table must exist before `copy_from_dataframe` is used.

## Batch upserts

The PostgreSQL path executes one parameterized batch at a time rather than one
statement per row. The conflict target must have a primary-key or unique
constraint in PostgreSQL.

```python
from MAna.database import upsert

upsert(
    features,
    "customer_features",
    db,
    key_columns="customer_id",
    schema="analytics",
    update_columns=["order_count", "lifetime_value", "updated_at"],
    batch_size=5_000,
)
```

You can target a named constraint or ignore conflicts:

```python
from MAna.database import upsert_postgres

upsert_postgres(
    events,
    "events",
    db,
    schema="analytics",
    conflict_constraint="events_external_id_key",
    do_nothing=True,
)
```

## Query into pandas

Use bound parameters for values:

```python
from MAna.database import Query, read_sql

active = read_sql(
    """
    SELECT customer_id, lifetime_value
    FROM analytics.customer_features
    WHERE lifetime_value >= :minimum_value
    """,
    db,
    params={"minimum_value": 1_000},
)

top_customers = (
    Query("analytics.customer_features")
    .select("customer_id", "lifetime_value")
    .where(lifetime_value__gte=1_000)
    .order_by("-lifetime_value")
    .limit(100)
    .to_dataframe(db)
)
```

The query builder parameterizes values. Table names, selected expressions, join
clauses, and raw `having` expressions are developer-authored SQL and should not
come from untrusted input.

## Inspect the catalog

```python
from MAna.database import (
    get_postgres_constraints,
    get_postgres_indexes,
    get_postgres_table_sizes,
    get_postgres_table_health,
    list_postgres_schemas,
    list_postgres_tables,
    postgres_database_size,
)

schemas = list_postgres_schemas(db)
tables = list_postgres_tables(db, schema="analytics")
sizes = get_postgres_table_sizes(db, schema="analytics")
health = get_postgres_table_health(db, schema="analytics")
indexes = get_postgres_indexes(db, "customer_features", schema="analytics")
constraints = get_postgres_constraints(db, "customer_features", schema="analytics")
database_size = postgres_database_size(db)
```

Row counts in `list_postgres_tables` are planner estimates, so catalog browsing
does not need to run `COUNT(*)` over every table.

If the `pg_stat_statements` extension is enabled, inspect the most expensive
normalized queries by total execution time:

```python
from MAna.database import get_postgres_slow_queries

slow_queries = get_postgres_slow_queries(db, limit=20, minimum_calls=5)
```

## Read a query plan

Start with `EXPLAIN`. Add `analyze=True` only when it is safe to execute the
query:

```python
from MAna.database import explain_postgres

plan = explain_postgres(
    """
    SELECT *
    FROM analytics.customer_features
    WHERE customer_id = :customer_id
    """,
    db,
    params={"customer_id": 42},
)

measured_plan = explain_postgres(
    "SELECT * FROM analytics.customer_features WHERE lifetime_value > 1000",
    db,
    analyze=True,
    buffers=True,
    format="json",
)
```

`EXPLAIN ANALYZE` executes the statement. Avoid using it casually with
`INSERT`, `UPDATE`, or `DELETE`.

## Transactions and maintenance

```python
from sqlalchemy import text

from MAna.database import transaction, vacuum_analyze_postgres

with transaction(db) as connection:
    connection.execute(
        text("UPDATE analytics.jobs SET status = :status WHERE job_id = :job_id"),
        {"status": "complete", "job_id": 17},
    )

# VACUUM cannot run inside a normal transaction; MAna uses AUTOCOMMIT here.
vacuum_analyze_postgres(db, "customer_features", schema="analytics")
```

## Portable notebook

The companion notebook remains executable without a database server by using
SQLite, and demonstrates the shared MAna connector, pandas, query-builder,
index, transaction, and MongoDB APIs:

[MAna.database notebook](../notebooks/database_module.ipynb)

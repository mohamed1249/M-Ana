import os
import unittest
from unittest.mock import patch

import pandas as pd


class PostgreSQLUrlTests(unittest.TestCase):
    def test_url_escapes_credentials_and_preserves_sslmode(self):
        from sqlalchemy.engine import make_url

        from MAna.database import postgres_url

        url = postgres_url(
            database="analytics",
            username="data user",
            password="p@ss/word",
            host="db.example.com",
            sslmode="require",
        )
        parsed = make_url(url)

        self.assertEqual(parsed.drivername, "postgresql+psycopg2")
        self.assertEqual(parsed.username, "data user")
        self.assertEqual(parsed.password, "p@ss/word")
        self.assertEqual(parsed.database, "analytics")
        self.assertEqual(parsed.query["sslmode"], "require")

    def test_standard_postgres_environment_variables(self):
        from sqlalchemy.engine import make_url

        from MAna.database import postgres_url_from_env

        environment = {
            "PGHOST": "warehouse.internal",
            "PGPORT": "5433",
            "PGDATABASE": "reporting",
            "PGUSER": "analyst",
            "PGPASSWORD": "secret",
            "PGSSLMODE": "verify-full",
        }
        with patch.dict(os.environ, environment, clear=True):
            parsed = make_url(postgres_url_from_env())

        self.assertEqual(parsed.host, "warehouse.internal")
        self.assertEqual(parsed.port, 5433)
        self.assertEqual(parsed.database, "reporting")
        self.assertEqual(parsed.query["sslmode"], "verify-full")

    def test_database_environment_variable_is_required(self):
        from MAna.database import postgres_url_from_env

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PGDATABASE"):
                postgres_url_from_env()


class PostgreSQLStatementTests(unittest.TestCase):
    def test_upsert_sql_supports_schema_and_selected_updates(self):
        from MAna.database import postgres_upsert_sql

        statement = postgres_upsert_sql(
            "customer_features",
            ["customer_id", "score", "updated_at"],
            key_columns="customer_id",
            schema="analytics",
            update_columns=["score"],
        )

        self.assertIn('INSERT INTO "analytics"."customer_features"', statement)
        self.assertIn('ON CONFLICT ("customer_id")', statement)
        self.assertIn('DO UPDATE SET "score" = EXCLUDED."score"', statement)
        self.assertNotIn('"updated_at" = EXCLUDED', statement)

    def test_upsert_sql_supports_constraint_and_do_nothing(self):
        from MAna.database import postgres_upsert_sql

        statement = postgres_upsert_sql(
            "events",
            ["external_id", "payload"],
            conflict_constraint="events_external_id_key",
            do_nothing=True,
        )

        self.assertTrue(
            statement.endswith(
                'ON CONFLICT ON CONSTRAINT "events_external_id_key" DO NOTHING'
            )
        )

    def test_unsafe_identifier_is_rejected(self):
        from MAna.database import postgres_upsert_sql

        with self.assertRaises(ValueError):
            postgres_upsert_sql(
                "events; DROP TABLE events",
                ["event_id"],
                key_columns="event_id",
            )

    def test_explain_rejects_buffers_without_execution(self):
        from MAna.database import explain_postgres

        with self.assertRaisesRegex(ValueError, "requires analyze"):
            explain_postgres("SELECT 1", object(), buffers=True)

    def test_dataframe_schema_generates_postgres_ddl(self):
        from MAna.database import generate_postgres_create_table

        frame = pd.DataFrame(
            {
                "event_id": pd.Series([1, 2], dtype="int64"),
                "active": [True, False],
                "payload": [{"source": "web"}, {"source": "app"}],
                "label": ["new", "returning"],
            }
        )
        statement = generate_postgres_create_table(
            frame,
            "events",
            schema="analytics",
            primary_key="event_id",
            dtype_overrides={"label": "VARCHAR(50)"},
        )

        self.assertIn('"event_id" BIGINT NOT NULL', statement)
        self.assertIn('"active" BOOLEAN', statement)
        self.assertIn('"payload" JSONB', statement)
        self.assertIn('"label" VARCHAR(50)', statement)
        self.assertIn('PRIMARY KEY ("event_id")', statement)

    def test_dataframe_schema_rejects_unsafe_type_override(self):
        from MAna.database import generate_postgres_create_table

        with self.assertRaises(ValueError):
            generate_postgres_create_table(
                pd.DataFrame({"event_id": [1]}),
                "events",
                dtype_overrides={"event_id": "BIGINT; DROP TABLE events"},
            )


class FakeResult:
    rowcount = 0


class FakeTransactionConnection:
    def __init__(self, calls):
        self.calls = calls

    def execute(self, statement, records=None):
        self.calls.append((str(statement), records))
        return FakeResult()


class FakeBegin:
    def __init__(self, calls):
        self.connection = FakeTransactionConnection(calls)

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakePostgresEngine:
    def __init__(self):
        from sqlalchemy.engine import make_url

        self.url = make_url("postgresql+psycopg2://localhost/test")
        self.calls = []

    def begin(self):
        return FakeBegin(self.calls)


class PostgreSQLBatchTests(unittest.TestCase):
    def test_public_upsert_dispatches_to_postgres_and_batches_records(self):
        from MAna.database import upsert

        engine = FakePostgresEngine()
        frame = pd.DataFrame(
            {
                "event_id": [1, 2, 3],
                "score": [1.5, None, 3.5],
                "tags": [["new"], ["returning"], ["vip"]],
            }
        )

        affected = upsert(
            frame,
            "events",
            engine,
            key_columns="event_id",
            schema="analytics",
            batch_size=2,
        )

        self.assertEqual(affected, 3)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(len(engine.calls[0][1]), 2)
        self.assertIsNone(engine.calls[0][1][1]["score"])
        self.assertEqual(engine.calls[0][1][0]["tags"], ["new"])
        self.assertIn('"analytics"."events"', engine.calls[0][0])

    def test_postgres_index_uses_validated_identifiers(self):
        from MAna.database import create_postgres_index

        engine = FakePostgresEngine()
        name = create_postgres_index(
            engine,
            "events",
            ["customer_id", "created_at"],
            schema="analytics",
            method="brin",
        )

        self.assertEqual(name, "idx_events_customer_id_created_at")
        self.assertIn(
            'ON "analytics"."events" USING brin ("customer_id", "created_at")',
            engine.calls[0][0],
        )


class FakeCopyCursor:
    def __init__(self, exported_text=""):
        self.exported_text = exported_text
        self.copy_calls = []
        self.description = []
        self.closed = False

    def copy_expert(self, statement, buffer):
        self.copy_calls.append(statement)
        if "TO STDOUT" in statement:
            buffer.write(self.exported_text)
        else:
            self.input_text = buffer.read()

    def execute(self, statement):
        self.description = [("event_id",), ("score",)]

    def close(self):
        self.closed = True


class FakeRawConnection:
    def __init__(self, exported_text=""):
        self.cursor_instance = FakeCopyCursor(exported_text)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class FakeCopyEngine:
    def __init__(self, exported_text=""):
        self.raw = FakeRawConnection(exported_text)

    def raw_connection(self):
        return self.raw


class PostgreSQLCopyTests(unittest.TestCase):
    def test_copy_from_dataframe_uses_schema_qualified_copy(self):
        from MAna.database import copy_from_dataframe

        engine = FakeCopyEngine()
        frame = pd.DataFrame({"event_id": [1, 2], "score": [0.5, None]})

        written = copy_from_dataframe(frame, "events", engine, schema="analytics")

        self.assertEqual(written, 2)
        self.assertTrue(engine.raw.committed)
        self.assertIn('COPY "analytics"."events"', engine.raw.cursor_instance.copy_calls[0])
        self.assertIn(r"\N", engine.raw.cursor_instance.input_text)

    def test_copy_to_dataframe_reads_exported_csv(self):
        from MAna.database import copy_to_dataframe

        engine = FakeCopyEngine("1,0.5\n2,1.5\n")
        frame = copy_to_dataframe(
            "events",
            engine,
            schema="analytics",
            columns=["event_id", "score"],
        )

        self.assertEqual(frame["event_id"].tolist(), [1, 2])
        self.assertEqual(frame["score"].tolist(), [0.5, 1.5])
        self.assertIn("TO STDOUT", engine.raw.cursor_instance.copy_calls[0])


if __name__ == "__main__":
    unittest.main()

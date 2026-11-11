import unittest

import pandas as pd


class SQLiteUpsertTests(unittest.TestCase):
    def test_generic_upsert_matches_numpy_scalar_keys(self):
        from MAna.database import SQLiteConnector, to_sql, upsert

        db = SQLiteConnector(":memory:")
        try:
            db.initialize_schema(
                """
                CREATE TABLE items (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    value REAL
                )
                """
            )
            to_sql(
                pd.DataFrame(
                    {
                        "id": [1, 2, 3, 4],
                        "name": ["a", "b", "c", "d"],
                        "value": [1.0, 2.0, 3.0, 4.0],
                    }
                ),
                "items",
                db,
                if_exists="append",
            )

            rows = upsert(
                pd.DataFrame(
                    {
                        "id": [4, 6],
                        "name": ["d2", "f"],
                        "value": [8.0, 9.0],
                    }
                ),
                "items",
                db,
                key_columns="id",
            )

            result = db.query("SELECT id, name, value FROM items ORDER BY id")
        finally:
            db.close()

        self.assertEqual(rows, 2)
        self.assertEqual(result.loc[result["id"] == 4, "name"].iloc[0], "d2")
        self.assertIn(6, result["id"].tolist())


if __name__ == "__main__":
    unittest.main()

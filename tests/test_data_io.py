import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class DataIOTests(unittest.TestCase):
    def test_read_data_accepts_path_objects_for_core_formats(self):
        from MAna.data import read_data

        source = pd.DataFrame({"name": ["Ada", "Grace"], "score": [10, 20]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            csv_path = tmpdir / "scores.csv"
            json_path = tmpdir / "scores.json"

            source.to_csv(csv_path, index=False)
            source.to_json(json_path, orient="records")

            pd.testing.assert_frame_equal(read_data(csv_path), source)
            pd.testing.assert_frame_equal(read_data(json_path), source)

    def test_read_from_database_supports_sqlalchemy_two_string_queries(self):
        from MAna.data import read_from_database

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "scores.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                pd.DataFrame({"name": ["Ada", "Grace"], "score": [10, 20]}).to_sql(
                    "scores", conn, index=False, if_exists="replace"
                )
            finally:
                conn.close()

            result = read_from_database(
                "sqlite:///" + db_path.as_posix(),
                "SELECT name, score FROM scores WHERE score >= 20",
            )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            pd.DataFrame({"name": ["Grace"], "score": [20]}),
        )


if __name__ == "__main__":
    unittest.main()

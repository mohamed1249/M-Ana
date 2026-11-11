import unittest

import pandas as pd


class DatabaseDatatypeOptimizationTests(unittest.TestCase):
    def test_unhashable_json_values_remain_objects(self):
        from MAna.database import optimize_datatypes

        plans = [
            [{"Plan": {"Node Type": "Seq Scan"}}],
            [{"Plan": {"Node Type": "Index Scan"}}],
        ]
        frame = pd.DataFrame({"QUERY PLAN": plans})

        optimized = optimize_datatypes(frame)

        self.assertEqual(optimized["QUERY PLAN"].dtype, object)
        self.assertEqual(optimized["QUERY PLAN"].tolist(), plans)

    def test_low_cardinality_strings_still_become_categories(self):
        from MAna.database import optimize_datatypes

        frame = pd.DataFrame(
            {"status": ["paid", "paid", "paid", "shipped", "shipped"]}
        )

        optimized = optimize_datatypes(frame)

        self.assertEqual(str(optimized["status"].dtype), "category")


if __name__ == "__main__":
    unittest.main()

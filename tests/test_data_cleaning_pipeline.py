import unittest

import pandas as pd


class DataCleanerPipelineTests(unittest.TestCase):
    def test_nested_object_columns_support_quality_and_duplicate_checks(self):
        from MAna.data import DataCleaner

        frame = pd.DataFrame(
            {
                "overview": ["A story", "A story", "Another story"],
                "genres": [
                    ["Drama", "History"],
                    ["Drama", "History"],
                    ["Comedy"],
                ],
                "metadata": [
                    {"language": "en"},
                    {"language": "en"},
                    {"language": "fr"},
                ],
                "keywords": [
                    {"period", "war"},
                    {"war", "period"},
                    {"family"},
                ],
            }
        )

        cleaner = DataCleaner(frame, verbose=False)
        self.assertAlmostEqual(
            cleaner.get_report().data_quality_before["uniqueness"],
            2 / 3,
        )

        profile = cleaner.profile_data()
        self.assertEqual(profile["duplicates"], 1)
        self.assertEqual(profile["columns"]["genres"]["unique"], 2)

        cleaned = cleaner.remove_duplicates().get_cleaned_data()
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned.iloc[0]["genres"], ["Drama", "History"])

    def test_structural_preprocessing_methods_are_chainable(self):
        from MAna.data import DataCleaner

        frame = pd.DataFrame(
            {
                "User ID": [1, 1, 2, 3],
                "Event Time": ["2024-01-01", "2024-01-01", "bad", "2024-01-03"],
                "Score": ["1.5", "1.5", "bad", "3.0"],
                "Text": ["first", "first", "   ", "third"],
            }
        )

        cleaner = DataCleaner(frame, id_columns=["User ID"], verbose=False)
        cleaned = (
            cleaner
            .standardize_column_names()
            .rename_columns({"user_id": "account_id"})
            .remove_duplicates(subset=["account_id", "event_time"])
            .parse_dates(columns=["event_time"], extract_features=False)
            .coerce_numeric(columns=["score"])
            .drop_missing_rows(
                subset=["event_time", "score", "text"],
                treat_blank_as_missing=True,
            )
            .get_cleaned_data()
        )

        self.assertEqual(cleaned["account_id"].tolist(), [1, 3])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["event_time"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["score"]))
        self.assertEqual(cleaner.id_columns, ["account_id"])

    def test_drop_columns_method_is_chainable(self):
        from MAna.data import DataCleaner

        frame = pd.DataFrame(
            {
                "Drop Me": [1, 2],
                "Keep Me": [3, 4],
                "Target": [0, 1],
            }
        )

        cleaner = DataCleaner(frame, target_column="Target", verbose=False)
        cleaned = cleaner.standardize_column_names().drop_columns(columns=["drop_me"]).get_cleaned_data()

        self.assertNotIn("drop_me", cleaned.columns)
        self.assertIn("keep_me", cleaned.columns)

    def test_clean_all_updates_column_metadata_after_standardizing_names(self):
        from MAna.data import DataCleaner

        frame = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3, 4],
                "Department Name": ["Dresses", "Tops", None, "Dresses"],
                "Review Text": ["Great", None, "Bad", "Nice"],
                "Positive Feedback Count": [0, 1, 20, 2],
                "Recommended IND": [1, 1, 0, 1],
            }
        )

        cleaner = DataCleaner(
            frame,
            target_column="Recommended IND",
            id_columns=["Customer ID"],
            categorical_columns=["Department Name"],
            numerical_columns=["Positive Feedback Count"],
            text_columns=["Review Text"],
            verbose=False,
        )

        cleaned = cleaner.clean_all(
            missing_strategy="auto",
            outlier_method="iqr",
            encode_method=None,
            scale_method=None,
            standardize_columns=True,
        )

        self.assertIn("recommended_ind", cleaned.columns)
        self.assertIn("customer_id", cleaner.id_columns)
        self.assertEqual(cleaner.target_column, "recommended_ind")


if __name__ == "__main__":
    unittest.main()

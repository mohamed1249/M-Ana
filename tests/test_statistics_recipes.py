import unittest

import numpy as np
import pandas as pd


class DescriptiveStatisticsTests(unittest.TestCase):
    def test_descriptive_statistics_excludes_nonfinite_values(self):
        from MAna.stata import descriptive_statistics

        result = descriptive_statistics([1, 2, 2, 3, np.nan, np.inf])

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["excluded"], 2)
        self.assertEqual(result["mode"], 2.0)
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertLess(result["mean_ci"][0], result["mean"])
        self.assertGreater(result["mean_ci"][1], result["mean"])

    def test_grouped_statistics_and_frequency_table(self):
        from MAna.stata import frequency_table, grouped_descriptive_statistics

        frame = pd.DataFrame(
            {
                "region": ["north", "north", "south", "south", "south"],
                "revenue": [10, 20, 30, 50, np.nan],
            }
        )
        grouped = grouped_descriptive_statistics(frame, "revenue", "region")
        frequencies = frequency_table(["north", "north", "south", None])

        north = grouped.loc[grouped["region"] == "north"].iloc[0]
        self.assertEqual(north["count"], 2)
        self.assertAlmostEqual(north["mean"], 15.0)
        self.assertAlmostEqual(frequencies["proportion"].sum(), 1.0)
        self.assertEqual(frequencies["count"].sum(), 4)


class AssociationTests(unittest.TestCase):
    def test_correlation_and_linear_regression(self):
        from MAna.stata import correlation_test, linear_regression_test

        x = np.arange(1, 11, dtype=float)
        y = 2 * x + 1
        correlation = correlation_test(x, y)
        regression = linear_regression_test(x, y)

        self.assertAlmostEqual(correlation["coefficient"], 1.0)
        self.assertTrue(correlation["significant"])
        self.assertAlmostEqual(regression["slope"], 2.0)
        self.assertAlmostEqual(regression["intercept"], 1.0)
        self.assertAlmostEqual(regression["r_squared"], 1.0)

    def test_partial_correlation_controls_for_confounder(self):
        from MAna.stata import correlation_test, partial_correlation

        rng = np.random.default_rng(7)
        confounder = rng.normal(size=400)
        x = 2 * confounder + rng.normal(scale=0.5, size=400)
        y = 3 * confounder + rng.normal(scale=0.5, size=400)
        frame = pd.DataFrame({"x": x, "y": y, "confounder": confounder})

        unadjusted = correlation_test(x, y)
        adjusted = partial_correlation(frame, "x", "y", "confounder")

        self.assertGreater(unadjusted["coefficient"], 0.8)
        self.assertLess(abs(adjusted["coefficient"]), 0.15)
        self.assertEqual(adjusted["degrees_of_freedom"], 397)

    def test_correlation_matrix_uses_numeric_columns_only(self):
        from MAna.stata import correlation_matrix

        frame = pd.DataFrame(
            {"x": [1, 2, 3], "y": [2, 4, 6], "label": ["a", "b", "c"]}
        )
        result = correlation_matrix(frame)

        self.assertEqual(result.columns.tolist(), ["x", "y"])
        self.assertAlmostEqual(result.loc["x", "y"], 1.0)


class ExperimentPlanningTests(unittest.TestCase):
    def test_bayesian_proportion_test_favors_b(self):
        from MAna.stata import bayesian_proportion_ab_test

        result = bayesian_proportion_ab_test(
            20,
            100,
            40,
            100,
            samples=20_000,
            random_state=7,
        )

        self.assertGreater(result["probability_b_better"], 0.99)
        self.assertGreater(result["expected_difference"], 0)
        self.assertGreater(result["difference_credible_interval"][0], 0)

    def test_power_and_required_sample_size_are_consistent(self):
        from MAna.stata import (
            required_sample_size_two_sample,
            two_sample_ttest_power,
        )

        sample_size = required_sample_size_two_sample(0.5, power=0.80)

        self.assertGreaterEqual(two_sample_ttest_power(0.5, sample_size), 0.80)
        self.assertLess(two_sample_ttest_power(0.5, sample_size - 1), 0.80)

    def test_new_helpers_are_public(self):
        import MAna.stata as stata

        expected = {
            "bayesian_proportion_ab_test",
            "correlation_matrix",
            "correlation_test",
            "descriptive_statistics",
            "frequency_table",
            "grouped_descriptive_statistics",
            "linear_regression_test",
            "partial_correlation",
            "required_sample_size_two_sample",
            "two_sample_ttest_power",
        }
        self.assertTrue(expected.issubset(set(stata.__all__)))


if __name__ == "__main__":
    unittest.main()

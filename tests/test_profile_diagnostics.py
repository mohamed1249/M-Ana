import json
import unittest

import numpy as np
import pandas as pd

from MAna.data import profile_dataset


class DiagnosticTests(unittest.TestCase):
    def test_duplicates_relationships_and_reproducibility(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=70)
        frame = pd.DataFrame({'a': a, 'b': a*2, 'large': a*1000,
                              'other': rng.normal(size=70)})
        frame.loc[:4, ['a', 'b']] = np.nan
        frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
        first = profile_dataset(frame, diagnostic_max_rows=50)
        second = profile_dataset(frame, diagnostic_max_rows=50)
        self.assertEqual(first.health, second.health)
        self.assertEqual(first.health['duplicate_rows'], 1)
        self.assertEqual(first.health['diagnostic_rows'], 50)
        self.assertTrue(any(p['jaccard'] == 1 for p in first.relationships['missingness_pairs']))
        self.assertTrue(first.relationships['correlation_groups']['0.9']['groups'])
        self.assertTrue(any(a['rule'] == 'standard_deviation_ratio' for a in first.advice))
        self.assertTrue(0 <= first.health['hopkins']['value'] <= 1)
        json.dumps(first.to_dict(), allow_nan=False)

    def test_known_rank_and_near_duplicate_definition(self):
        X = pd.DataFrame({'a': [0., 1., 2., 3., 3.], 'b': [0., 2., 4., 6., 6.]})
        report = profile_dataset(X)
        self.assertEqual(report.health['effective_dimensionality']['components'], 1)
        self.assertAlmostEqual(report.health['near_duplicate_row_fraction'], .4)

    def test_limits_and_optional_diagnostics(self):
        X = pd.DataFrame({'a': range(10), 'b': range(10), 'c': range(10)})
        report = profile_dataset(X, diagnostic_max_features=1)
        self.assertEqual(report.health['omitted_numeric_features'], 2)
        self.assertEqual(profile_dataset(X, diagnostics=False).health, {})
        with self.assertRaises(ValueError):
            profile_dataset(X, advice_thresholds={'unknown': 1})

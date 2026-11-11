import json
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from MAna.data import profile_dataset


class DatasetProfilingTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_statistics_missing_values_and_input_preservation(self):
        frame = pd.DataFrame({'x': [0., 1., 1., 2., np.nan],
                              'empty': [np.nan]*5, 'constant': [2.]*5,
                              'text': ['a', 'a', None, 'b', 'c'],
                              'nested': [[1], [1], None, [2], [3]]})
        before = frame.copy(deep=True)
        report = profile_dataset(frame)
        assert_frame_equal(frame, before)
        row = report.fingerprints.loc['x']
        self.assertEqual(row['mean'], 1.)
        self.assertEqual(row['mad'], .5)
        self.assertEqual(row['tie_rate'], .25)
        self.assertEqual(row['zero_rate'], .25)
        self.assertEqual(row['missing_pct'], 20.)
        self.assertEqual(report.fingerprints.loc['empty', 'family'], -1)
        self.assertTrue(report.fingerprints.loc['constant', 'constant'])
        self.assertEqual(report.fingerprints.loc['nested', 'unique_count'], 3)
        json.dumps(report.to_dict(), allow_nan=False)
        self.assertIn('Dataset:', report.summary())

    def test_sampling_is_explicit_reproducible_and_grouping_uses_behavior(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=200)
        frame = pd.DataFrame({'a': values, 'b': values, 'c': np.zeros(200)})
        first = profile_dataset(frame, sample_size=80)
        second = profile_dataset(frame, sample_size=80)
        assert_frame_equal(first.fingerprints, second.fingerprints)
        self.assertTrue(first.metadata['sampled'])
        self.assertEqual(first.metadata['sample_rows'], 80)
        self.assertEqual(first.fingerprints.loc['a', 'family'], first.fingerprints.loc['b', 'family'])
        self.assertEqual(sum(first.families.member_count), 3)
        self.assertEqual(len(first.plot_histograms()), len(first.families))
        self.assertIsNotNone(first.plot_dendrogram())
        self.assertIsNotNone(first.plot_fingerprint_heatmap())

    def test_dbscan_noise_empty_data_infinities_and_html(self):
        report = profile_dataset(pd.DataFrame({'<script>': [1., np.inf, np.nan]}),
                                 family_method='dbscan', min_samples=2)
        self.assertEqual(report.fingerprints.iloc[0]['infinite_count'], 1)
        self.assertEqual(len(report.outliers), 1)
        self.assertIn('&lt;script&gt;', report.to_html())
        self.assertNotIn('<script>', report.to_html())
        for frame in [pd.DataFrame(), pd.DataFrame({'x': []}), pd.DataFrame({'x': [None]})]:
            report = profile_dataset(frame)
            json.dumps(report.to_dict(), allow_nan=False)
            report.summary()

    def test_nullable_values_and_validation(self):
        report = profile_dataset(pd.DataFrame({'x': pd.Series([1, None, 3], dtype='Int64')}))
        self.assertEqual(report.fingerprints.loc['x', 'mean'], 2)
        with self.assertRaises(ValueError):
            profile_dataset(pd.DataFrame({'x': [1, 2]}), sample_size=0)

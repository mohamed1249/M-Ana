import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from MAna.modeling import CorrelationGrouper


class CorrelationGrouperTests(unittest.TestCase):
    def test_groups_positive_features_preserves_constants_and_index(self):
        X = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [2., 4., 6., 8.],
                          'negative': [-1., -2., -3., -4.], 'constant': [7.] * 4},
                         index=[2, 4, 6, 8])
        grouper = CorrelationGrouper()
        result = grouper.fit_transform(X)
        self.assertIn(['a', 'b'], list(grouper.groups_.values()))
        self.assertIn('negative', result)
        self.assertIn('constant', result)
        name = next(k for k, v in grouper.feature_mapping_.items() if v == ['a', 'b'])
        np.testing.assert_allclose(result[name], X[['a', 'b']].mean(axis=1))
        self.assertTrue(result.index.equals(X.index))
        self.assertEqual(result.columns.tolist(), grouper.get_feature_names_out().tolist())
        new = X.iloc[:2].copy()
        new['b'] = [90., 10.]
        before = grouper.correlation_matrix_.copy()
        transformed = grouper.transform(new[new.columns[::-1]])
        np.testing.assert_allclose(transformed[name], new[['a', 'b']].mean(axis=1))
        assert_frame_equal(before, grouper.correlation_matrix_)

    def test_complete_linkage_does_not_chain_weak_pairs(self):
        rng = np.random.default_rng(19)
        a, c = rng.normal(size=(2, 1000))
        X = pd.DataFrame({'a': a, 'b': a + c, 'c': c})
        grouper = CorrelationGrouper(threshold=0.6, correlation='pearson').fit(X)
        for cols in grouper.groups_.values():
            if len(cols) > 1:
                values = grouper.correlation_matrix_.loc[cols, cols].to_numpy()
                self.assertGreaterEqual(values.min(), 0.6 - 1e-12)
        self.assertEqual(len(grouper.get_feature_names_out()), 2)

    def test_constants_single_feature_validation_and_name_collision(self):
        X = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 4., 6.], 'group_1': [0.] * 3})
        result = CorrelationGrouper().fit_transform(X)
        self.assertEqual(result.shape[1], 2)
        self.assertTrue(result.columns.is_unique)
        self.assertEqual(CorrelationGrouper(constant_policy='drop').fit_transform(X).shape[1], 1)
        with self.assertRaises(ValueError):
            CorrelationGrouper(constant_policy='raise').fit(X)
        single = X[['a']]
        assert_frame_equal(CorrelationGrouper().fit_transform(single), single)
        fitted = CorrelationGrouper().fit(X)
        for invalid in [X.drop(columns='a'), X.assign(extra=1.), X.assign(a=np.nan)]:
            with self.assertRaises(ValueError):
                fitted.transform(invalid)

    def test_sklearn_pipeline_and_clone(self):
        X = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 4., 6.]})
        pipeline = Pipeline([('groups', clone(CorrelationGrouper())),
                             ('scale', StandardScaler())])
        result = pipeline.fit_transform(X)
        self.assertEqual(result.shape, (3, 1))
        np.testing.assert_allclose(pipeline.transform(X), result)

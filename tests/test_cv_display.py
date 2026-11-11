import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from MAna.modeling.model_evaluation import cross_validate_model


class CrossValidationDisplayTests(unittest.TestCase):
    def test_extreme_folds_remain_visible_and_raw_scores_unchanged(self):
        raw = {'test_r2': np.array([-3.3e22, .86, -2e18, .86, .86]),
               'test_neg_mean_absolute_error': np.array([-2e9, -.27, -2e7, -.28, -.27])}
        with patch('sklearn.model_selection.cross_validate', return_value=raw), patch.object(plt, 'show'):
            result = cross_validate_model(object(), None, None, scoring=['r2'])
        self.assertIsInstance(result, dict)
        np.testing.assert_array_equal(result['test_r2_scores'], raw['test_r2'])
        np.testing.assert_array_equal(result['test_neg_mean_absolute_error_scores'], raw['test_neg_mean_absolute_error'])
        self.assertTrue((result.fold_scores()['Validation MAE'] > 0).all())
        self.assertIn('fold(s) 1, 3', result._repr_html_())
        self.assertIn('<details>', result._repr_html_())
        self.assertEqual(plt.gcf().axes[0].get_yscale(), 'symlog')
        plt.gcf().canvas.draw()
        plt.close('all')

    def test_single_metric_and_nonfinite_score(self):
        with patch('sklearn.model_selection.cross_validate', return_value={'test_score': np.array([.5, np.nan])}):
            result = cross_validate_model(object(), None, None, scoring='r2', plot=False)
        self.assertIn('Non-finite', repr(result))
        self.assertEqual(len(result.summary()), 1)


if __name__ == '__main__':
    unittest.main()

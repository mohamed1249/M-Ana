import unittest

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from MAna.modeling import silhouette


class SilhouetteLayoutTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_many_clusters_have_bounded_labels_and_legend(self):
        rng = np.random.default_rng(42)
        labels = np.repeat(np.arange(96), 3)
        X = rng.normal(size=(len(labels), 3))
        fig = silhouette(X, labels, show=False)
        ax = fig.axes[0]
        self.assertLessEqual(len(ax.get_yticklabels()), 20)
        self.assertEqual(len(ax.get_legend().get_texts()), 1)
        self.assertIn('\nAverage Silhouette', ax.get_title())
        self.assertNotIn('\\n', ax.get_title())
        self.assertEqual(len(ax.collections), 96)

    def test_sampling_and_explicit_legend_controls(self):
        rng = np.random.default_rng(42)
        labels = np.repeat([0, 1, 2], 30)
        X = rng.normal(size=(90, 2)) + labels[:, None] * 5
        fig = silhouette(
            X, labels, sample_size=40, show_legend=True,
            max_cluster_labels=2, show=False,
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_yticklabels()), 2)
        self.assertEqual(len(ax.get_legend().get_texts()), 4)
        self.assertIn('sample: 40', ax.get_title())

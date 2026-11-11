"""Fitted, inspectable aggregation of positively correlated features."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage as hierarchical_linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CorrelationGrouper(TransformerMixin, BaseEstimator):
    """Average groups of positively correlated numeric DataFrame columns.

    Parameters
    ----------
    threshold : float, default=0.80
        Minimum positive correlation for grouping. With complete linkage, every
        pair within a group meets this threshold (up to numerical precision).
    correlation : {'spearman', 'pearson', 'kendall'}, default='spearman'
        Correlation calculated on training observations only.
    linkage : {'complete', 'average', 'single'}, default='complete'
        Hierarchical linkage. Average/single do not guarantee the pairwise
        threshold for all members. Ward is inappropriate for this dissimilarity.
    constant_policy : {'keep', 'drop', 'raise'}, default='keep'
        Treatment of constant training columns, excluded from correlation fitting.

    Notes
    -----
    Inputs must be nonempty finite numeric DataFrames with unique string columns.
    Scale features separately before fitting if their units differ. No scaling is
    performed here. Transform requires the same column set (order may differ),
    preserves the row index, and never recomputes groups. Negative correlations
    are not treated as similarities. Averaging is lossy: no inverse is provided.
    Feature grouping costs quadratic memory in the number of input features.
    """

    def __init__(self, threshold=0.80, correlation='spearman',
                 linkage='complete', constant_policy='keep'):
        self.threshold = threshold
        self.correlation = correlation
        self.linkage = linkage
        self.constant_policy = constant_policy

    @staticmethod
    def _validate_frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('CorrelationGrouper requires a pandas DataFrame')
        if X.empty:
            raise ValueError('Input must contain rows and columns')
        if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
            raise ValueError('Feature names must be unique strings')
        if any(not pd.api.types.is_numeric_dtype(dtype) or
               pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes):
            raise ValueError('All features must be real numeric columns')
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError('Missing or infinite values must be handled before grouping')

    def fit(self, X, y=None):
        """Learn feature groups and output mappings from training data."""
        self._validate_frame(X)
        if not 0 < self.threshold <= 1:
            raise ValueError('threshold must be in (0, 1]')
        if self.correlation not in ('spearman', 'pearson', 'kendall'):
            raise ValueError('Unsupported correlation method')
        if self.linkage not in ('complete', 'average', 'single'):
            raise ValueError('linkage must be complete, average, or single')
        if self.constant_policy not in ('keep', 'drop', 'raise'):
            raise ValueError('constant_policy must be keep, drop, or raise')

        constants = X.columns[X.nunique() <= 1].tolist()
        if constants and self.constant_policy == 'raise':
            raise ValueError(f'Constant features found: {constants}')
        features = X.drop(columns=constants)
        corr = features.corr(method=self.correlation)
        if not np.isfinite(corr.to_numpy()).all():
            raise ValueError('Some correlations are undefined')
        tree = None
        if len(features.columns) >= 2:
            distance = (1 - corr.to_numpy()).clip(0, 2)
            distance = (distance + distance.T) / 2
            np.fill_diagonal(distance, 0)
            tree = hierarchical_linkage(squareform(distance), method=self.linkage)
            ids = fcluster(tree, t=1 - self.threshold, criterion='distance')
        else:
            ids = np.ones(len(features.columns), dtype=int)

        raw_groups = {}
        for col, group_id in zip(features.columns, ids):
            raw_groups.setdefault(int(group_id), []).append(col)
        # Stable, readable IDs follow input feature order, not dendrogram IDs.
        groups = {i: cols for i, cols in enumerate(raw_groups.values(), start=1)}
        merged = {col for cols in groups.values() if len(cols) > 1 for col in cols}
        mapping = {col: [col] for col in X.columns if col not in merged and
                   not (col in constants and self.constant_policy == 'drop')}
        for group_id, cols in groups.items():
            if len(cols) > 1:
                name = f'group_{group_id}'
                while name in X.columns or name in mapping:
                    name = '_' + name
                mapping[name] = cols
        if not mapping:
            raise ValueError('No features remain after dropping constants')

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.constant_columns_ = constants
        self.correlation_matrix_ = corr
        self.linkage_matrix_ = tree
        self.groups_ = groups
        self.feature_mapping_ = mapping
        return self

    def transform(self, X):
        """Apply training groups, returning a DataFrame of composite features."""
        check_is_fitted(self, 'feature_mapping_')
        self._validate_frame(X)
        if set(X.columns) != set(self.feature_names_in_):
            raise ValueError('Input columns must match the fitted feature names')
        return pd.DataFrame({
            name: X[cols[0]].copy() if len(cols) == 1 else X[cols].mean(axis=1)
            for name, cols in self.feature_mapping_.items()
        }, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Return output names in transform order."""
        check_is_fitted(self, 'feature_mapping_')
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError('input_features must match feature_names_in_')
        return np.asarray(list(self.feature_mapping_), dtype=object)

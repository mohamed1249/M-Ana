"""Bounded, descriptive dataset fingerprints and feature-family reports."""

from dataclasses import dataclass, field
from html import escape
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler

from .data_cleaning import _make_hashable


_FINGERPRINT = [
    'mean', 'median', 'trimmed_mean', 'std', 'iqr', 'mad', 'range',
    'cv', 'skew', 'kurtosis', 'unique_ratio', 'top_value_frequency',
    'zero_rate', 'tie_rate', 'missing_pct', 'outlier_fraction',
]
_LOG_STATS = {'mean', 'median', 'trimmed_mean', 'std', 'iqr', 'mad',
              'range', 'cv', 'skew', 'kurtosis'}
_DEFAULT_THRESHOLDS = {'sparse_rate': 0.8, 'heavy_kurtosis': 3.,
                       'bell_skew': 0.5, 'bell_kurtosis': 1., 'discrete_ties': 0.5}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class DatasetProfile:
    """Sample-based descriptive report. Rates use observed values unless noted.

    Family -1 means DBSCAN noise or an unprofileable/non-numeric feature. Family
    statistics are descriptive, not evidence that original columns correlate.
    """

    fingerprints: pd.DataFrame
    families: pd.DataFrame
    metadata: dict
    transformed_fingerprints: pd.DataFrame
    linkage_matrix: Any = None
    _histograms: dict = field(default_factory=dict, repr=False)
    health: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)
    advice: list = field(default_factory=list)

    @property
    def outliers(self):
        """Features outside numeric families, with explicit exclusion reasons."""
        return self.fingerprints.loc[self.fingerprints['family'] == -1].copy()

    def summary(self):
        lines = [
            f"Dataset: {self.metadata['total_rows']:,} rows x {self.metadata['columns']} features",
            f"Profiled {self.metadata['sample_rows']:,} rows "
            f"({'sample estimates' if self.metadata['sampled'] else 'all rows'}); "
            f"{len(self.families)} numeric feature families.",
        ]
        for family, row in self.families.iterrows():
            ranges = '; '.join(f'{name}={bounds[0]:.3g}..{bounds[1]:.3g}'
                               for name, bounds in row['stat_ranges'].items())
            lines.append(f"Family {family}: {row['member_count']} features; "
                         f"examples: {', '.join(row['examples'])}; {ranges}")
        if len(self.outliers):
            names = self.outliers.index.tolist()
            lines.append(f"Outside families: {len(names)}; examples: {', '.join(names[:5])}. "
                         'See report.outliers for the full list and reasons.')
        lines.append('Ties describe repeated values; they do not imply weak Spearman correlation.')
        if self.health:
            lines.append(f"Health diagnostics: {self.health['diagnostic_rows']} rows, "
                         f"{len(self.health['diagnostic_features'])} numeric features; "
                         f"{self.health['duplicate_rows']} duplicate rows in profile scope.")
        if self.advice:
            lines.append(f'Evidence-based suggestions: {len(self.advice)} (see report.advice).')
        return '\n'.join(lines)

    def to_dict(self):
        """Return JSON-safe values; undefined statistics become null."""
        return _json_safe({
            'metadata': self.metadata,
            'fingerprints': self.fingerprints.reset_index().to_dict('records'),
            'families': self.families.reset_index().to_dict('records'),
            'health': self.health, 'relationships': self.relationships, 'advice': self.advice,
        })

    def to_html(self, filepath=None):
        """Return a self-contained HTML report, optionally writing UTF-8 text."""
        content = ('<!doctype html><html><head><meta charset="utf-8">'
                   '<title>MAna dataset profile</title><style>'
                   'body{font:14px sans-serif;margin:2rem}table{border-collapse:collapse}'
                   'td,th{padding:.4rem;border:1px solid #ddd}pre{white-space:pre-wrap}'
                   '</style></head><body><h1>Dataset profile</h1><pre>'
                   + escape(self.summary()) + '</pre><h2>Feature families</h2>'
                   + self.families.to_html(escape=True)
                   + '<h2>Feature fingerprints</h2>'
                   + self.fingerprints.to_html(escape=True)
                   + '<h2>Health, relationships and advice</h2><pre>'
                   + escape(json.dumps(_json_safe(dict(health=self.health,
                       relationships=self.relationships, advice=self.advice)), indent=2))
                   + '</pre></body></html>')
        if filepath is not None:
            from pathlib import Path
            Path(filepath).write_text(content, encoding='utf-8')
        return content

    def plot_fingerprint_heatmap(self, figsize=(12, 8)):
        """Plot signed-log/robust-scaled fingerprints, ordered by family."""
        import matplotlib.pyplot as plt
        if self.transformed_fingerprints.empty:
            raise ValueError('No numeric fingerprints available')
        frame = self.transformed_fingerprints
        order = self.fingerprints.loc[frame.index].sort_values('family', kind='stable').index
        frame = frame.loc[order]
        fig, ax = plt.subplots(figsize=figsize)
        limit = max(float(np.quantile(np.abs(frame.to_numpy()), .95)), 1.)
        im = ax.imshow(frame, aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
        ax.set_xticks(np.arange(len(frame.columns)))
        ax.set_xticklabels(frame.columns, rotation=90)
        ticks = np.unique(np.linspace(0, len(frame)-1, min(30, len(frame)), dtype=int))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{frame.index[i]} (F{self.fingerprints.loc[frame.index[i], "family"]})'
                            for i in ticks])
        ax.set_title('Feature fingerprints ordered by family (color clipped at 95th percentile)')
        fig.colorbar(im, ax=ax, label='Robust-scaled statistic')
        fig.tight_layout()
        return fig

    def plot_dendrogram(self, figsize=(12, 5), max_leaves=30):
        """Show fitted feature hierarchy, truncated to readable subtree leaves."""
        import matplotlib.pyplot as plt
        if self.linkage_matrix is None:
            raise ValueError('Dendrogram requires hierarchical fitting with at least two features')
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(self.linkage_matrix, labels=self.transformed_fingerprints.index.tolist(),
                   truncate_mode='lastp', p=max_leaves, ax=ax, leaf_rotation=90)
        ax.set_ylabel('Distance between fingerprints')
        ax.set_title('Feature-family hierarchy (parentheses denote collapsed member counts)')
        fig.tight_layout()
        return fig

    def plot_histograms(self, family_ids=None):
        """Return {family_id: figure} for one representative histogram per family."""
        import matplotlib.pyplot as plt
        figures = {}
        for family in self.families.index if family_ids is None else family_ids:
            record = self._histograms[family]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.stairs(record['counts'], record['edges'], fill=True)
            ax.set(title=f"Family {family}: {record['feature']}",
                   xlabel='Value', ylabel='Profiled observation count')
            fig.tight_layout()
            figures[family] = fig
        return figures


def profile_dataset(df, *, sample_size=10000, random_state=42,
                    family_method='hierarchical', n_families=None,
                    distance_threshold=3., eps=2., min_samples=2,
                    trim_fraction=.1, outlier_iqr_factor=1.5,
                    normality_max_samples=5000, personality_thresholds=None,
                    max_family_features=2000, diagnostics=True,
                    diagnostic_max_rows=2000, diagnostic_max_features=200,
                    near_duplicate_tolerance=1e-6,
                    correlation_thresholds=(.8, .9, .95), advice_thresholds=None):
    """Describe feature behavior and group numeric columns by fingerprint.

    Includes fingerprint families, bounded health/relationship diagnostics and
    evidence-based heuristic advice. Samples at most 10,000 rows by default (None uses all rows).
    All reported statistics describe that sample. Missing percentages use all
    sampled rows; numeric moments use finite values. Infinities are counted
    separately. Non-numeric/nested values get quality/discreteness statistics.

    MAD is unscaled median absolute deviation; kurtosis is Fisher excess; CV is
    std/abs(mean) and undefined at zero mean. Tie rate is 1 - unique/observed.
    Shapiro normality tests use at most 5,000 finite values, are descriptive and
    unadjusted for multiple testing. Family labels are configurable heuristics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    if diagnostic_max_rows < 3 or diagnostic_max_features < 1:
        raise ValueError('Diagnostic limits require at least 3 rows and 1 feature')
    if not np.isfinite(near_duplicate_tolerance) or near_duplicate_tolerance < 0:
        raise ValueError('near_duplicate_tolerance must be finite and nonnegative')
    if any(not 0 < t <= 1 for t in correlation_thresholds):
        raise ValueError('Correlation thresholds must be in (0, 1]')
    rules = dict(skew=2., scale_ratio=100., distance_spread=.1,
                 hopkins_half_width=.05, correlation_gap=.2)
    if advice_thresholds:
        if set(advice_thresholds)-set(rules):
            raise ValueError('Unknown advice threshold')
        rules.update(advice_thresholds)
    if any(not np.isfinite(v) or v<0 for v in rules.values()):
        raise ValueError('Advice thresholds must be finite and nonnegative')
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError('Feature names must be unique strings')
    if sample_size is not None and (not isinstance(sample_size, int) or sample_size < 1):
        raise ValueError('sample_size must be a positive integer or None')
    if family_method not in ('hierarchical', 'dbscan'):
        raise ValueError('family_method must be hierarchical or dbscan')
    if not 0 <= trim_fraction < .5 or outlier_iqr_factor <= 0:
        raise ValueError('Invalid trim_fraction or outlier_iqr_factor')
    if not 3 <= normality_max_samples <= 5000:
        raise ValueError('normality_max_samples must be between 3 and 5000')
    if distance_threshold <= 0 or eps <= 0 or min_samples < 1 or max_family_features < 1:
        raise ValueError('Clustering controls must be positive')
    if n_families is not None and (not isinstance(n_families, int) or n_families < 1):
        raise ValueError('n_families must be a positive integer or None')
    thresholds = dict(_DEFAULT_THRESHOLDS)
    if personality_thresholds:
        if set(personality_thresholds) - set(thresholds):
            raise ValueError('Unknown personality threshold')
        thresholds.update(personality_thresholds)
    if any(not np.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError('Personality thresholds must be finite and nonnegative')
    sample = df.sample(n=sample_size, random_state=random_state) if (
        sample_size is not None and len(df) > sample_size) else df
    rng = np.random.default_rng(random_state)
    records, numeric_values = {}, {}
    for col in sample:
        series = sample[col]
        observed = series.dropna()
        numeric = (pd.api.types.is_numeric_dtype(series.dtype)
                   and not pd.api.types.is_complex_dtype(series.dtype))
        values = observed.to_numpy(dtype=float) if numeric else np.array([])
        finite = values[np.isfinite(values)]
        comparable = observed.map(_make_hashable)
        counts = comparable.value_counts(dropna=False)
        n, unique = len(observed), len(counts)
        row = dict.fromkeys(_FINGERPRINT + ['min', 'max', 'normality_statistic',
                                         'normality_pvalue'], np.nan)
        row.update(dtype=str(series.dtype), observed_count=n, finite_count=len(finite),
                   unique_count=unique, unique_ratio=unique/n if n else np.nan,
                   tie_rate=1-unique/n if n else np.nan,
                   top_value_frequency=float(counts.iloc[0]/n) if n else np.nan,
                   missing_pct=float(series.isna().mean()*100) if len(series) else np.nan,
                   infinite_count=int(np.isinf(values).sum()), constant=unique == 1,
                   normality_n=0, normality_test='not tested', family=-1,
                   type_guess='unknown', personality='', exclusion_reason='non-numeric')
        if numeric and len(finite):
            numeric_values[col] = finite
            q1, median, q3 = np.quantile(finite, [.25, .5, .75])
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            iqr = q3-q1
            variable = np.ptp(finite) > 0
            row.update(mean=mean, median=median, trimmed_mean=stats.trim_mean(finite, trim_fraction),
                       std=std, iqr=iqr, mad=np.median(np.abs(finite-median)),
                       min=finite.min(), max=finite.max(), range=np.ptp(finite),
                       cv=std/abs(mean) if mean != 0 else np.nan,
                       zero_rate=np.mean(finite == 0),
                       outlier_fraction=np.mean((finite < q1-outlier_iqr_factor*iqr) |
                                                (finite > q3+outlier_iqr_factor*iqr)),
                       skew=stats.skew(finite, bias=False) if variable and len(finite)>2 else np.nan,
                       kurtosis=stats.kurtosis(finite, bias=False) if variable and len(finite)>3 else np.nan,
                       exclusion_reason='')
            count_like = bool(np.all(finite >= 0) and np.allclose(finite, np.round(finite), rtol=0, atol=1e-8))
            row['type_guess'] = ('constant' if not variable else 'binary' if unique == 2 else
                                 'sparse' if row['zero_rate'] >= thresholds['sparse_rate'] else
                                 'count-like' if count_like else 'continuous')
            if variable and len(finite) >= 3:
                test_values = rng.choice(finite, normality_max_samples, replace=False) if (
                    len(finite) > normality_max_samples) else finite
                test = stats.shapiro(test_values)
                row.update(normality_statistic=test.statistic, normality_pvalue=test.pvalue,
                           normality_n=len(test_values), normality_test='Shapiro-Wilk')
            tags = []
            if row['zero_rate'] >= thresholds['sparse_rate'] and variable:
                tags.append('sparse spiker')
            if row['kurtosis'] > thresholds['heavy_kurtosis']:
                tags.append('heavy-tailed')
            if abs(row['skew']) <= thresholds['bell_skew'] and abs(row['kurtosis']) <= thresholds['bell_kurtosis']:
                tags.append('bell-shaped (moment heuristic)')
            if row['tie_rate'] >= thresholds['discrete_ties'] and variable:
                tags.append('stepwise/discrete')
            row['personality'] = ', '.join(tags)
        elif numeric:
            row['exclusion_reason'] = 'no finite observations'
        else:
            row['type_guess'] = 'constant' if unique == 1 else 'binary' if unique == 2 else 'categorical'
        records[col] = row

    fingerprints = pd.DataFrame.from_dict(records, orient='index')
    fingerprints.index.name = 'feature'
    if fingerprints.empty:
        fingerprints = pd.DataFrame(columns=['family'], index=pd.Index([], name='feature'))
    transformed = pd.DataFrame()
    tree = None
    names = list(numeric_values)
    if len(names) > max_family_features:
        raise ValueError('Too many numeric features for family clustering; increase max_family_features explicitly')
    if names:
        transformed = fingerprints.loc[names, _FINGERPRINT].astype(float).copy()
        for col in _LOG_STATS:
            transformed[col] = np.sign(transformed[col]) * np.log1p(np.abs(transformed[col]))
        transformed = transformed.replace([np.inf, -np.inf], np.nan)
        transformed = transformed.fillna(transformed.median()).fillna(0.)
        transformed[:] = RobustScaler().fit_transform(transformed)
        if len(names) == 1:
            labels = np.array([0 if family_method == 'hierarchical' or min_samples == 1 else -1])
        elif family_method == 'dbscan':
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(transformed)
        else:
            tree = linkage(pdist(transformed), method='complete')
            labels = fcluster(tree, t=n_families if n_families is not None else distance_threshold,
                              criterion='maxclust' if n_families is not None else 'distance') - 1
        fingerprints.loc[names, 'family'] = labels
        for name, label in zip(names, labels):
            if label == -1:
                fingerprints.loc[name, 'exclusion_reason'] = 'DBSCAN noise'
    families, histograms = {}, {}
    for family in sorted(set(fingerprints['family']) - {-1}):
        members = fingerprints.index[fingerprints['family'] == family].tolist()
        block = fingerprints.loc[members]
        ranges = {col: [float(block[col].min()), float(block[col].max())]
                  for col in _FINGERPRINT if block[col].notna().any()}
        center = transformed.loc[members].median().to_numpy()
        representative = members[int(np.argmin(np.linalg.norm(transformed.loc[members].to_numpy()-center, axis=1)))]
        counts, edges = np.histogram(numeric_values[representative], bins=30)
        families[int(family)] = dict(member_count=len(members), members=members,
                                    examples=members[:3], representative=representative,
                                    stat_ranges=ranges)
        histograms[int(family)] = dict(feature=representative, counts=counts, edges=edges)
    family_table = pd.DataFrame.from_dict(families, orient='index')
    family_table.index.name = 'family'
    metadata = dict(total_rows=len(df), columns=df.shape[1], sample_rows=len(sample),
                    sampled=len(sample)<len(df), random_state=random_state,
                    family_method=family_method, thresholds=thresholds,
                    n_families=n_families, distance_threshold=distance_threshold,
                    eps=eps, min_samples=min_samples, trim_fraction=trim_fraction,
                    outlier_iqr_factor=outlier_iqr_factor,
                    normality_max_samples=normality_max_samples,
                    implemented_layers=[0, 1, 2],
                    undefined_family_statistics='median imputation then zero if entirely undefined',
                    family_transform='signed log1p for moments/spreads, then RobustScaler')
    report = DatasetProfile(fingerprints, family_table, metadata, transformed, tree, histograms)
    if diagnostics:
        from .profile_diagnostics import dataset_diagnostics, evidence_advice
        report.health, report.relationships = dataset_diagnostics(
            sample, total_rows=len(df), random_state=random_state,
            max_rows=diagnostic_max_rows, max_features=diagnostic_max_features,
            near_duplicate_tolerance=near_duplicate_tolerance,
            correlation_thresholds=correlation_thresholds)
        report.advice = evidence_advice(fingerprints, report.health, report.relationships, rules)
        metadata['implemented_layers'] = [0, 1, 2, 3, 4, 5]
    metadata['advice_thresholds'] = rules
    # Enforce exportability without retaining the input frame.
    json.dumps(report.to_dict(), allow_nan=False)
    return report

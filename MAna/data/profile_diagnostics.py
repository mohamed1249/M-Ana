"""Bounded exploratory health and relationship diagnostics for profiling."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data_cleaning import _comparable_frame


def dataset_diagnostics(sample, *, total_rows, random_state, max_rows=2000,
                        max_features=200, near_duplicate_tolerance=1e-6,
                        correlation_thresholds=(.8, .9, .95)):
    """Compute health/relationships with explicit sample and feature scope."""
    rng = np.random.default_rng(random_state)
    exact = int(_comparable_frame(sample).duplicated().sum())
    health = {
        'rows_to_features_ratio': total_rows/sample.shape[1] if sample.shape[1] else None,
        'duplicate_rows': exact,
        'duplicate_fraction': exact/len(sample) if len(sample) else None,
        'duplicate_scope_rows': len(sample),
        'duplicate_scope': 'profile sample; duplicate counts exclude the first occurrence',
    }
    work = sample.sample(max_rows, random_state=random_state) if len(sample)>max_rows else sample
    numeric = work.select_dtypes(include=['number', 'bool']).copy()
    numeric = numeric.loc[:, [c for c in numeric if not pd.api.types.is_complex_dtype(numeric[c])]]
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    eligible = [c for c in numeric if numeric[c].nunique() > 1]
    selected = eligible[:max_features]
    numeric = numeric[selected]
    health.update(diagnostic_rows=len(work), diagnostic_features=selected,
                  omitted_numeric_features=len(eligible)-len(selected),
                  geometry_preprocessing='median imputation then standard scaling',
                  near_duplicate_tolerance=near_duplicate_tolerance)
    relationships = {'correlation_pairs': [], 'correlation_groups': {},
                     'missingness_pairs': [], 'scope_rows': len(work),
                     'scope_features': selected,
                     'missingness_scope_features': list(work.columns[:max_features])}
    if selected:
        pearson = numeric.corr(method='pearson', min_periods=3)
        spearman = numeric.corr(method='spearman', min_periods=3)
        for i, a in enumerate(selected):
            for b in selected[i+1:]:
                p, s = pearson.loc[a, b], spearman.loc[a, b]
                if np.isfinite(p) and np.isfinite(s):
                    relationships['correlation_pairs'].append({
                        'feature_1': a, 'feature_2': b, 'pearson': float(p),
                        'spearman': float(s), 'absolute_gap': float(abs(p-s)),
                        'paired_observations': int(numeric[[a, b]].notna().all(axis=1).sum())})
        relationships['correlation_pairs'].sort(key=lambda p: p['absolute_gap'], reverse=True)
        # Undefined pairs prevent grouping, rather than being invented as zero correlations.
        if len(selected)>1 and np.isfinite(spearman.to_numpy()).all():
            distance = (1-spearman.to_numpy()).clip(0, 2)
            distance = (distance+distance.T)/2
            np.fill_diagonal(distance, 0)
            tree = linkage(squareform(distance), method='complete')
            for threshold in correlation_thresholds:
                ids = fcluster(tree, 1-threshold, criterion='distance')
                groups = {}
                for col, group in zip(selected, ids):
                    groups.setdefault(int(group), []).append(col)
                merged = [cols for cols in groups.values() if len(cols)>1]
                relationships['correlation_groups'][str(threshold)] = {
                    'groups': merged,
                    'redundancy_score': sum(len(cols)-1 for cols in merged)/len(selected),
                    'definition': 'fraction removable by keeping one member of each positive-correlation group'}
        else:
            relationships['correlation_group_status'] = 'insufficient features or undefined correlations'

    missing = work.iloc[:, :max_features].isna()
    for i, a in enumerate(missing):
        for b in missing.columns[i+1:]:
            union = int((missing[a] | missing[b]).sum())
            if union:
                joint = int((missing[a] & missing[b]).sum())
                relationships['missingness_pairs'].append({
                    'feature_1': a, 'feature_2': b,
                    'joint_missing_count': joint, 'missing_union_count': union,
                    'jaccard': joint/union})
    relationships['missingness_pairs'].sort(key=lambda p: p['jaccard'], reverse=True)
    if len(work)<3 or not selected:
        health['geometry_status'] = 'requires at least three rows and a variable numeric feature'
        return health, relationships
    filled = numeric.fillna(numeric.median())
    values = StandardScaler().fit_transform(filled)
    pca = PCA(svd_solver='full').fit(values)
    components = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .9)+1)
    health['effective_dimensionality'] = {
        'variance_target': .9, 'components': min(components, len(pca.components_)),
        'available_components': len(pca.components_), 'feature_count': len(selected)}
    nearest = NearestNeighbors(n_neighbors=2).fit(values)
    nearest_distances = nearest.kneighbors(values)[0][:, 1]
    health['near_duplicate_row_fraction'] = float(np.mean(nearest_distances <= near_duplicate_tolerance))
    health['near_duplicate_definition'] = (
        'fraction of diagnostic rows with another row within tolerance in standardized, '
        'median-imputed selected features; includes exact matches and both members of a pair')
    distances = pdist(values)
    q05, q50, q95 = np.quantile(distances, [.05, .5, .95])
    health['distance_concentration'] = {
        'q05': float(q05), 'median': float(q50), 'q95': float(q95),
        'relative_spread': float((q95-q05)/q95) if q95 else None,
        'definition': '(95th - 5th percentile distance) / 95th; smaller means more concentrated'}
    # Explicit distance-based Hopkins convention: higher = departure from uniformity.
    m = min(100, max(1, len(values)//10))
    indices = rng.choice(len(values), m, replace=False)
    artificial = rng.uniform(values.min(axis=0), values.max(axis=0), size=(m, values.shape[1]))
    u = nearest.kneighbors(artificial, n_neighbors=1)[0][:, 0]
    w = nearest_distances[indices]
    denominator = float(u.sum()+w.sum())
    health['hopkins'] = {
        'value': float(u.sum()/denominator) if denominator else None,
        'probe_count': m, 'distance_power': 1,
        'null': 'uniform in the axis-aligned bounding box of standardized selected features',
        'interpretation': 'higher suggests departure from uniformity, not proof of meaningful clusters; '
                          'sensitive to dimension, dependence, duplicates and reference geometry'}
    detector = IsolationForest(n_estimators=100, contamination='auto', random_state=random_state)
    predictions = detector.fit_predict(values)
    health['multivariate_outliers'] = {
        'method': 'IsolationForest', 'contamination': 'auto',
        'count': int((predictions == -1).sum()),
        'fraction': float(np.mean(predictions == -1)),
        'definition': 'model-flagged observations, not verified errors'}
    return health, relationships


def evidence_advice(fingerprints, health, relationships, thresholds):
    """Return structured heuristic suggestions with reproducible numerical evidence."""
    advice = []

    def add(rule, value, threshold, message, **context):
        advice.append(dict(rule=rule, value=float(value), threshold=float(threshold),
                           message=message, **context))

    if 'skew' in fingerprints:
        for feature, row in fingerprints.iterrows():
            if np.isfinite(row['skew']) and abs(row['skew']) > thresholds['skew']:
                add('absolute_skew', abs(row['skew']), thresholds['skew'],
                    'Inspect skew; consider Yeo-Johnson or a domain-appropriate log transform. '
                    'Validate whether transformed distances suit the task.', feature=feature,
                    signed_skew=float(row['skew']), comparison='>')
        spreads = fingerprints['std'].dropna()
        spreads = spreads[spreads>0]
        if len(spreads)>1:
            ratio = float(spreads.max()/spreads.min())
            if ratio>thresholds['scale_ratio']:
                add('standard_deviation_ratio', ratio, thresholds['scale_ratio'],
                    'Consider deliberate feature scaling before distance-based modeling.', comparison='>')
    concentration = health.get('distance_concentration', {}).get('relative_spread')
    if concentration is not None and concentration<thresholds['distance_spread']:
        add('distance_relative_spread', concentration, thresholds['distance_spread'],
            'Distances are concentrated in the diagnostic representation; inspect redundant '
            'dimensions and feature weighting before clustering.', comparison='<')
    hopkins = health.get('hopkins', {}).get('value')
    if hopkins is not None and abs(hopkins-.5)<=thresholds['hopkins_half_width']:
        add('hopkins_distance_from_half', abs(hopkins-.5), thresholds['hopkins_half_width'],
            'This diagnostic gives limited evidence of departure from its uniform reference. '
            'It does not establish that clusters are absent.', hopkins=hopkins, comparison='<=')
    for pair in relationships.get('correlation_pairs', []):
        if pair['absolute_gap']>thresholds['correlation_gap']:
            add('pearson_spearman_gap', pair['absolute_gap'], thresholds['correlation_gap'],
                'Inspect the pair for nonlinear monotonic structure, influential observations '
                'or ties; the gap does not identify the cause.',
                features=[pair['feature_1'], pair['feature_2']], comparison='>')
    return advice

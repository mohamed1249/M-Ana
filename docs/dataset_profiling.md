# Dataset behavior profiling

```python
from MAna.data import profile_dataset

report = profile_dataset(df, sample_size=10000, random_state=42)
print(report.summary())
report.families
report.fingerprints
report.outliers
report.to_html('dataset_profile.html')
payload = report.to_dict()

report.plot_fingerprint_heatmap()
report.plot_dendrogram()
figures = report.plot_histograms()  # dictionary: family ID -> figure
```

The implementation covers safe inspection, fingerprints, feature families,
health, relationships and heuristic advice (Layers 0–5). It never modifies the input. Fingerprint statistics refer
to the same reproducible sample, capped at 10,000 rows by default. Set
`sample_size=None` for full-data statistics. Sample uniqueness and constant flags
are estimates of sample behavior, not guarantees about the entire dataset.

Numeric moments use finite observations. Missing percentages use all sampled
rows; infinities are counted separately. Discreteness uses nonmissing values,
including infinities. Zero and outlier fractions use finite numeric values.
Non-numeric columns receive quality/discreteness statistics but are excluded
from numeric family clustering. Nested lists/dictionaries are normalized for
counting without changing the source data. All-missing columns remain visible.

MAD means unscaled median absolute deviation. Kurtosis is excess kurtosis.
CV is sample standard deviation divided by absolute mean and is undefined when
the mean is zero. Tie rate is `1 - unique_count / observed_count`; it measures
duplicate excess and does **not** imply weak Spearman correlation. Outliers are
outside the usual IQR fences (factor 1.5 by default), including when IQR is zero.
Shapiro–Wilk uses at most 5,000 observations per feature. P-values are descriptive,
not corrected for multiple testing, and are not proof of distributional form.

## Families

Heavy-tailed fingerprint statistics receive signed `log1p` transformations, then
RobustScaler is fitted across features. Undefined entries are median-imputed for
clustering only (zero if the whole statistic is undefined). The fingerprint table
keeps the original undefined values. Complete-linkage hierarchical clustering
uses `distance_threshold=3.0`; `n_families=5` requests at most five families.
Alternatively use `family_method='dbscan', eps=2.0, min_samples=2`.
These defaults are exploratory settings, not statistically optimal thresholds.

Family tables contain member counts, members, examples, a representative, and
minimum/maximum ranges of fingerprint statistics. Representatives are closest to
the coordinate-wise median fingerprint. The report retains histogram bins, not
the source dataset. DBSCAN noise and nonprofileable columns appear in `outliers`
with explicit reasons. Hierarchical mode assigns all profileable numeric columns
to families; isolated features may form singleton families.

Family membership means similar **distributional behavior**, not that features
correlate with each other. For row-wise correlation groups use CorrelationGrouper.
Personality labels are heuristics configurable through `personality_thresholds`:
`sparse_rate`, `heavy_kurtosis`, `bell_skew`, `bell_kurtosis`, `discrete_ties`.
The bell-shaped label only tests skew/kurtosis thresholds, not unimodality.

Plots return figures without calling show. The dendrogram displays the fitted
feature hierarchy, with collapsed subtrees for large numbers of features. Family
clustering is capped at 2,000 numeric columns by default because its resource
requirements can grow quadratically; override `max_family_features` explicitly.

## Health, relationships, and advice

```python
report = profile_dataset(
    df,
    diagnostic_max_rows=2000,
    diagnostic_max_features=200,
    near_duplicate_tolerance=1e-6,
    correlation_thresholds=(0.80, 0.90, 0.95),
    advice_thresholds={'skew': 2., 'scale_ratio': 100.},
)
report.health
report.relationships
report.advice
```

Exact duplicates are counted in the profiling sample, excluding the first
occurrence. Thus a group of three equal rows contributes two duplicates. Use
`sample_size=None` for exact full-data counts. The rows-to-features ratio uses the
original dataset dimensions.

Expensive diagnostics use a reproducible subset of at most 2,000 profiling rows
and the first 200 variable real numeric columns. Column names and omitted counts
are exposed in `health`; these are not full-feature diagnostics when truncated.
Raise the caps explicitly when appropriate. Missingness relationships use the
first 200 original columns, including nonnumeric ones.

PCA, near-duplicate detection, distances, Hopkins, and Isolation Forest use median
imputation followed by standard scaling on that diagnostic subset. Effective
dimensionality counts PCA components needed for 90% variance. Near-duplicate
fraction counts **all rows** with another row within the stated Euclidean
tolerance, including exact matches; its denominator differs from duplicate-row
counts. Distance concentration reports the 5th, median, and 95th distance
percentiles and `(q95-q05)/q95`, not extrema.

Hopkins uses at most 100 probes with `sum(u)/(sum(u)+sum(w))`, ordinary Euclidean
distances (power one), and a uniform axis-aligned bounding box reference. It is
not the dimension-powered variant; values must not be compared indiscriminately
with other definitions. Duplicates, dependent features and dimension can inflate
it. Isolation Forest uses its automatic cutoff; flagged observations are not
verified errors.

Relationships include Pearson/Spearman pair statistics, their absolute gap and
paired observation counts. Complete-linkage groups use positive Spearman
correlations at the requested thresholds. Redundancy is the proportion removable
by keeping one member per group. Undefined correlations prevent grouping and are
reported. Missingness co-occurrence uses Jaccard similarity on missing indicators.

Every advice record contains a rule, measured value, threshold, comparison and
message. Defaults are absolute skew > 2, standard-deviation ratio > 100,
relative distance spread < 0.1, distance of Hopkins from 0.5 <= 0.05, and
Pearson–Spearman gap > 0.2. These are configurable exploratory heuristics.
Advice is scoped to the data used by its underlying diagnostic. Set
`diagnostics=False` to run only Layers 0–2.

A Pearson–Spearman gap is a diagnostic, not proof of a specific cause. Hopkins
depends on its null reference distribution, scaling, and dimension; values near
0.5 must not be presented as proof that meaningful clusters cannot exist.

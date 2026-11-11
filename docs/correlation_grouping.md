# Grouping correlated features

`CorrelationGrouper` learns groups of positively correlated numerical features
and replaces each multi-feature group with a row-wise mean. It is exported from
`MAna.modeling` and `MAna.modeling.data_preprocessing`.

```python
from MAna.modeling import CorrelationGrouper

# Supply a numeric DataFrame; scale separately when feature units differ.
grouper = CorrelationGrouper(
    threshold=0.80,
    correlation='spearman',
    linkage='complete',
    constant_policy='keep',
)
X_reduced = grouper.fit_transform(X_cleaned)

for group_id, columns in grouper.groups_.items():
    if len(columns) > 1:
        print(group_id, columns)

print(grouper.feature_mapping_)
print(grouper.get_feature_names_out())
new_reduced = grouper.transform(new_cleaned)
```

Fit once on training observations. Transform reuses the same groups, even if
relationships differ in new observations. Inputs require the same uniquely named
string columns; reordered columns are accepted. Row indexes are preserved.
Missing/infinite values and nonnumeric features must be handled beforehand.

The default uses Spearman correlation and complete linkage on `1 - correlation`.
The threshold is expressed as correlation, not dendrogram distance. Every pair
in a complete-linkage group meets the threshold. Optional `single` and `average`
linkage relax that guarantee. Negative correlations are not merged. Pearson and
Kendall correlations are also supported.

Constant columns are excluded from correlation calculation and kept separately
by default. Choose `drop` to remove them or `raise` to reject them. Singletons keep
their names. Composite names use `group_N`, with leading underscores added if
necessary to avoid collisions. IDs follow training column order and need not match
IDs from an independently constructed SciPy dendrogram. `groups_` includes
nonconstant singleton groups; `feature_mapping_` describes every output column.

The transformer does not standardize or re-standardize composites. Its effect is
to change feature weighting by collapsing similar columns. High Spearman
correlation does not establish interchangeable measurements, and averaging can
discard meaningful variation. Validate the resulting representation for your
task. No inverse is offered because averaging is lossy.

The fitted transformer supports scikit-learn pipelines, cloning and joblib
persistence. It stores `correlation_matrix_`, `linkage_matrix_`,
`constant_columns_`, and `feature_names_in_` for inspection. Correlation/linkage
memory grows quadratically with feature count; this is not intended for extremely
wide datasets.

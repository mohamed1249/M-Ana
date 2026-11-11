# Fitted preprocessing and inverse transformations

`DataCleaner` retains the fitted objects used by `scale_features()` and
`encode_categorical()`. Each operation records its columns and position in the
transformation chain. Accessors return the most recent object for a column;
scalers and multi-column encoders may be shared between columns.

```python
from MAna.data import DataCleaner

cleaner = DataCleaner(
    training_data,
    numerical_columns=['income'],
    categorical_columns=['region'],
    verbose=False,
)
cleaner.encode_categorical(method='onehot').scale_features(method='robust')

scaler = cleaner.get_scaler('income')
encoder = cleaner.get_encoder('region')
restored_training = cleaner.inverse_transform()

# Uses training parameters and category order without refitting.
transformed_new = cleaner.transform(new_data)
restored_new = cleaner.inverse_transform(transformed_new)
```

Pass DataFrames with the required column names to `transform()` and
`inverse_transform()`. Indexes are preserved. Inverse operations execute in reverse
order and return a copy without modifying the cleaner. Unknown categories raise
an error. Label encoding preserves native categories instead of coercing them to
strings; mixed incomparable category types should be normalized beforehand.

`transform()` currently supports pipelines consisting exclusively of scaling and
encoding. It rejects recorded steps such as duplicate removal, imputation,
renaming, and date parsing. If these were used, prepare new data separately and
use the retrieved fitted objects explicitly. A retrieved multi-column scaler or
encoder expects its full fitted column set in its original order.

Inverse transformation restores the representation before encoding/scaling. It
does not recover dropped observations, deleted columns, pre-imputation values,
or clipped values. It is not a general undo history. Target encoding raises on
inverse transformation because it is not uniquely invertible. Arbitrary edits to
the transformed data, including subsequent destructive cleaning, are not
guaranteed to be reversible. `reset()` restores the original training frame and
clears the fitted objects.

## Persistence

```python
cleaner.save_fitted_pipeline('cleaner.joblib')
loaded = DataCleaner.load_fitted_pipeline('cleaner.joblib')
restored = loaded.inverse_transform()
```

The fitted format uses joblib and saves the entire cleaner, including its original
and current data. Treat files as potentially large and sensitive. Load only trusted
files: joblib/pickle deserialization can execute code. Use compatible Python and
dependency versions when loading persisted estimators.

The existing JSON `save_pipeline()` remains configuration-only; it does not save
learned parameters. Use the fitted format to retain learned scales and categories.

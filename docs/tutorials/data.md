# Data Module Notebook

Use [dataset behavior profiling](../dataset_profiling.md) for feature fingerprints,
compact distribution families, and report exports via `profile_dataset()`.

For stored scalers/encoders, inverse transformations, new-data transformation,
and fitted persistence, see [Fitted preprocessing](../fitted_preprocessing.md).

This notebook documents `MAna.data` through a real review dataset sample from
older project work, then walks through the IO helpers, focused cleaning
functions, data-entry correction scenarios, outlier methods, aggregation,
pivoting, and the `DataCleaner` pipeline.

The other module notebooks deliberately reuse `read_data()` and `DataCleaner`
for source loading and structural preparation. This keeps missing-value,
duplicate, numeric-coercion, and date-parsing policies connected across MAna
instead of rebuilding them with notebook-specific pandas code.

Open it here:

[MAna.data - Reading, Cleaning, and Preparing Real Data](../notebooks/data_module.ipynb)

The bundled sample is intentionally small enough for documentation while still
keeping the behavior real.

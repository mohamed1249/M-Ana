# Data Module Notebook

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

# Time Series Module Notebook

This notebook documents `MAna.timeseries`, the forecasting and time-aware
analysis layer of MAna. It uses real datasets from older project work:
Dubai electricity demand, stock-market closes, gold prices, and investor
historical-price files.

The notebook focuses on the full workflow: validating a time axis, repairing
frequency gaps, resampling, stationarity checks, lag and rolling features,
calendar and Fourier features, panel lags, leakage-safe splitting, spectral
diagnostics, decomposition, forecasting, evaluation, anomaly detection, and
the main time-series visualization helpers.

Open it here:

[MAna.timeseries - Forecasting, Seasonality, Panels, and Anomalies](../notebooks/timeseries_module.ipynb)

The code comments are intentionally detailed because time-series work has many
small leakage traps. The notebook explains why each preparation step happens
before calling the next MAna function.

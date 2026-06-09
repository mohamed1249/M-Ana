# M-Ana

M-Ana is a personal Python data-science toolkit for data loading, cleaning, analysis, visualization, modeling, time-series work, statistics, A/B testing, and database workflows.

The project started as a small collection of helper functions I kept rewriting in notebooks. It has grown into an alpha-stage package that collects reusable tools for the repeated parts of data science work: preparing data, exploring it, testing ideas, training models, evaluating results, and saving useful outputs.

> Status: alpha. The package is useful for personal and experimental workflows, but the public API may still change.

## What It Includes

### Data

Utilities for loading and cleaning data.

- Read CSV, Excel, SAS, Stata, JSON, Parquet, HTML tables, images, and database query results.
- Drop or fill missing values.
- Remove outliers with z-score, IQR, Isolation Forest, Local Outlier Factor, MAD, DBSCAN, and percentile methods.
- Clean data-entry errors and validate expected values.
- Generate cleaning reports and reusable cleaning workflows.

### Analysis and Visualization

Plotting helpers for common exploratory analysis.

- Line, scatter, bar, distribution, box, heatmap, pairplot, area, pie, treemap, sunburst, Sankey, violin, funnel, waterfall, and 3D scatter plots.
- Animated scatter and animated line charts.
- Quick dashboard helpers for exploratory reports.

### Modeling

Reusable helpers for model training, evaluation, persistence, and tracking.

- Classification, regression, and clustering evaluation.
- Feature importance, confusion matrix, ROC, precision-recall, residual, prediction-vs-actual, PCA, elbow, silhouette, and dendrogram plots.
- Model registry and metadata tracking.
- Save/load helpers for scikit-learn, Keras/TensorFlow, and PyTorch models.
- Auto-classifier and auto-regressor experiments.
- PyTorch training helpers, simple classifier/regressor models, device selection, layer freezing, and reproducibility utilities.

### Time Series

Tools for forecasting, anomaly detection, decomposition, feature engineering, and evaluation.

- Datetime validation and indexing.
- Frequency detection, resampling, missing timestamp handling, stationarity tests, differencing, and train/test splitting.
- Lag, rolling, expanding, date, seasonal, Fourier, holiday, and difference features.
- ARIMA, SARIMA, AutoARIMA, Prophet, Exponential Smoothing, LSTM, ensemble, and hybrid forecasting interfaces.
- Z-score, IQR, MAD, Isolation Forest, LOF, Prophet, and ensemble anomaly detection.
- Forecast metrics such as MAE, MSE, RMSE, MAPE, SMAPE, R2, bias, direction accuracy, and coverage.

### Statistics and Experimentation

Statistical utilities for experiments and inference.

- T-tests, proportion tests, chi-square tests, Mann-Whitney, Wilcoxon, Kruskal-Wallis, Levene, and normality tests.
- Effect sizes such as Cohen's d, Cohen's h, Cramer's V, Glass delta, eta squared, and omega squared.
- Confidence intervals, bootstrap tests, multiple-comparison corrections, lift metrics, and standardization helpers.
- A/B testing, multivariate testing, power curves, confidence interval plots, lift analysis, and conversion-funnel visualizations.

### Database

Database helpers for analysis workflows.

- SQL read/write helpers.
- Query builder utilities.
- Connectors for PostgreSQL, MySQL, SQLite, and MongoDB-style workflows.
- Bulk insert, upsert, datatype optimization, table inspection, and index helpers.

## Installation

From source:

```bash
git clone https://github.com/mohamed1249/M-Ana.git
cd M-Ana
pip install -e .
```

From a built wheel in `dist/`:

```bash
pip install dist/M_Ana_package-0.0.6-py3-none-any.whl
```

The package has broad optional functionality. Some modules require heavier libraries such as TensorFlow, PyTorch, Prophet, database drivers, or PyMC. Install the dependencies needed for the parts you use.

## Quick Examples

### Read and Clean Data

```python
from MAna.data import data_io, data_cleaning

df = data_io.read_data("data.csv")
df = data_cleaning.fill_missing_values(df, method="median")
df = data_cleaning.remove_outliers(df, method="iqr")
```

### Visualize a Dataset

```python
from MAna.analysis import visualizations as viz

viz.dist(df, "price")
viz.scatter(df, x_col="age", y_col="income", color_col="segment")
viz.heatmap(df.corr())
```

### Evaluate a Model

```python
from MAna.modeling import model_evaluation

report = model_evaluation.evaluate_classification(model, X_test, y_test)
print(report.metrics)
```

### Forecast a Time Series

```python
from MAna.timeseries.forecasting import ARIMAForecaster

forecaster = ARIMAForecaster(order=(1, 1, 1))
forecaster.fit(series)
forecast = forecaster.predict(steps=30)
```

### Run a Statistical Test

```python
from MAna.stata.hypothesis_tests import t_test

result = t_test(group_a, group_b)
print(result)
```

## Package Structure

```text
MAna/
  analysis/      Visualization and dashboard helpers
  data/          Data loading, cleaning, validation, and preprocessing
  database/      Database connectors, SQL helpers, and query builders
  modeling/      Model training, evaluation, persistence, and visualization
  stata/         Statistical tests, A/B testing, and experiment utilities
  timeseries/    Time-series preprocessing, forecasting, evaluation, and plots
  Test/          Early tests and examples
```

## Roadmap

- Improve public API consistency across modules.
- Add complete documentation pages.
- Expand automated tests.
- Split heavy dependencies into optional extras.
- Add examples for each subpackage.
- Publish a cleaner package release.

## License

This project is released under the MIT License.

## Author

Muhammad Abdulsalam

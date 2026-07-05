# M-Ana

M-Ana is a personal Python data-science toolkit for data loading, cleaning, analysis, visualization, NLP, retrieval-augmented generation, recommender systems, big-data processing, modeling, time-series work, statistical testing, A/B testing, and database workflows.

The project started as a small collection of helper functions I kept rewriting in notebooks. It has grown into a stable package that collects reusable tools for the repeated parts of data science work: preparing data, exploring it, testing ideas, training models, evaluating results, and saving useful outputs.

> Status: stable 1.x. Public APIs follow semantic versioning and compatibility changes are documented.

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

### Natural Language Processing

Text preparation and analysis tools that also provide the foundation for
future RAG and recommender modules.

- Unicode normalization, cleaning, stop-word filtering, and tokenization.
- Overlapping word, character, and sentence chunking with source offsets.
- TF-IDF, count, and hashing vectorization.
- NMF and LDA topic modeling.
- Built-in lexicon sentiment plus optional VADER and transformer backends.
- TF-IDF embeddings, optional sentence-transformer embeddings, and semantic search.
- Leakage-safe text classifiers that combine text, numeric, and categorical features.
- Segmented embeddings and an optional Hugging Face/PyTorch classifier.

### RAG and Retrieval

Reusable retrieval components with explicit metadata and citation handling.

- Persistent semantic indexes with stable document IDs.
- PDF page extraction that preserves source and page metadata.
- FAISS cosine indexes with correct vector normalization.
- Weighted reciprocal-rank fusion for several retrieval signals.
- Grounded context/messages and non-mutating chat-history trimming.
- Provider-neutral RAG orchestration with deduplicated evidence and source records.
- OpenAI-compatible chat and embedding adapters with environment-only credentials.
- Precision, recall, MRR, average precision, and NDCG retrieval evaluation.
- Retried, batched vector-database upserts with deterministic IDs.
- Optional Pinecone adapter for index creation, batched upserts, metadata
  filters, and top-k similarity search.

### Recommender Systems

Reusable recommendation models with a common ranked DataFrame contract.

- Leakage-safe per-user temporal holdout splits and sparse interaction matrices.
- Popularity baselines with Bayesian rating shrinkage and optional time decay.
- Dense or sparse content profiles, similar-item search, and user recommendations.
- Sparse item-KNN collaborative filtering from interaction co-occurrence.
- Weighted-score and reciprocal-rank hybrid recommenders.
- Precision, recall, hit rate, MRR, MAP, NDCG, catalog coverage, and diversity.

### Big-Data Processing

Local bounded-memory helpers plus optional PySpark workflow utilities.

- Spark session creation that respects cluster launchers and supports local mode.
- Schema-aware file/table readers and partition-aware writers.
- Safe joins, deterministic latest-row deduplication, casting, lineage, and unions.
- Schema, duplicate-key, null, partition, and execution-plan diagnostics.
- Mixed numeric/categorical Spark ML feature pipelines.
- Checkpointed Structured Streaming readers and writers.
- Pandas CSV chunking, multi-file ingestion, bounded batching, and threaded I/O.

### Time Series

Tools for forecasting, anomaly detection, decomposition, feature engineering, and evaluation.

- Datetime validation and indexing.
- Frequency detection, resampling, missing timestamp handling, stationarity tests, differencing, and train/test splitting.
- Lag, rolling, expanding, date, seasonal, Fourier, holiday, and difference features.
- Group-aware panel lags and regular panel aggregation without cross-entity leakage.
- Chronological split-before-scale workflows with fitted train-only scalers.
- ARIMA, SARIMA, AutoARIMA, Prophet, NeuralProphet, Exponential Smoothing, LSTM, ensemble, and hybrid forecasting interfaces.
- Z-score, IQR, MAD, Isolation Forest, LOF, Prophet, and ensemble anomaly detection.
- Forecast metrics such as MAE, MSE, RMSE, MAPE, SMAPE, R2, bias, direction accuracy, and coverage.
- Periodograms, dominant-period detection, and seasonal diagnostic dashboards.

### Stata Statistical Utilities and Experimentation

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
- SQLite schema initialization and `INSERT OR IGNORE` helpers for small local stores.
- MongoDB URI connection helpers and atomic document upserts.

## Installation

From source:

```bash
git clone https://github.com/mohamed1249/M-Ana.git
cd M-Ana
pip install -e .
```

From a built wheel in `dist/`:

```bash
pip install dist/m_ana_package-1.0.0-py3-none-any.whl
```

From the GitHub v1.0.0 release:

```bash
pip install "https://github.com/mohamed1249/M-Ana/releases/download/v1.0.0/m_ana_package-1.0.0-py3-none-any.whl"
```

Or install the tagged source directly (requires Git):

```bash
pip install "M_Ana_package @ git+https://github.com/mohamed1249/M-Ana.git@v1.0.0"
```

Install optional feature groups as needed:

```bash
pip install -e ".[stata,database,timeseries,big]"
pip install -e ".[nlp,rag,modeling,visualization]"
pip install -e ".[all]"
```

Heavy libraries such as TensorFlow, PyTorch, Prophet, database drivers, and
PyMC are optional and are no longer installed for every M-Ana user.

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

### Engineer Panel Time-Series Features

```python
from MAna.timeseries import aggregate_panel, create_grouped_lag_features

daily = aggregate_panel(
    transactions,
    date_col="date",
    value_cols=["sales", "customers"],
    group_by="store_id",
    frequency="D",
)
features = create_grouped_lag_features(
    daily,
    group_by="store_id",
    columns="sales",
    lags=[1, 7, 28],
    time_col="date",
)
```

### Split and Scale Without Future Leakage

```python
from MAna.timeseries import chronological_split_and_scale

split = chronological_split_and_scale(X, y, test_size=0.2, gap=7)
model.fit(split.X_train, split.y_train)
predictions = split.inverse_transform_targets(model.predict(split.X_test))
```

### Find Dominant Seasonal Periods

```python
from MAna.timeseries import dominant_periods

periods = dominant_periods(series, n=5, sampling_rate=1.0)
print(periods[["period", "power"]])
```

### Run a Statistical Test

```python
from MAna.stata.hypothesis_tests import t_test

result = t_test(group_a, group_b)
print(result)
```

### Summarize Data and Measure Relationships

```python
from MAna.stata import (
    descriptive_statistics,
    grouped_descriptive_statistics,
    partial_correlation,
)

summary = descriptive_statistics(df["revenue"])
by_region = grouped_descriptive_statistics(df, "revenue", "region")
adjusted = partial_correlation(
    df,
    x="ad_spend",
    y="revenue",
    covariates=["seasonality_index"],
)
```

### Bayesian A/B Testing and Power Planning

```python
from MAna.stata import (
    bayesian_proportion_ab_test,
    required_sample_size_two_sample,
)

comparison = bayesian_proportion_ab_test(
    successes_a=120,
    trials_a=1_000,
    successes_b=145,
    trials_b=1_000,
)
print(comparison["probability_b_better"])

sample_size = required_sample_size_two_sample(effect_size=0.30, power=0.80)
```

### Initialize SQLite and Save MongoDB Documents

```python
from MAna.database import SQLiteConnector, save_document

db = SQLiteConnector("local.db")
db.initialize_schema("""
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
)
""")
db.insert_or_ignore("items", [{"name": "fatigue"}, {"name": "cough"}])

save_document(collection, "report.md", "grounded note content")
```

### Process and Search Text

```python
from MAna.nlp import TextCleaner, TextChunker, semantic_search

cleaner = TextCleaner(stop_words="english", min_token_length=2)
cleaned = cleaner.transform(documents)

chunks = TextChunker(chunk_size=150, overlap=25).transform(cleaned)

matches = semantic_search(
    "customer churn prediction",
    [chunk.text for chunk in chunks],
    top_k=5,
)
```

### Analyze Sentiment and Topics

```python
from MAna.nlp import SentimentAnalyzer, TopicModeler

sentiment = SentimentAnalyzer().to_dataframe(reviews)
topics = TopicModeler(n_topics=5).fit_transform(reviews)
print(topics.topics_frame())
```

### Train a Mixed-Feature Text Classifier

```python
from MAna.nlp import build_text_classifier, evaluate_classifier

model = build_text_classifier(
    "Review Text",
    numeric_columns=["Age"],
    categorical_columns=["Department Name", "Class Name"],
)
model.fit(train, train["sentiment"])
metrics = evaluate_classifier(model, test, test["sentiment"])
```

Fit the pipeline only after splitting the data. Its TF-IDF vocabulary,
imputation, scaling, and category encoding are learned from training data.

### Build a Cited Semantic Index

```python
from MAna.rag import (
    SemanticIndex,
    build_grounded_messages,
    chunk_document_records,
    extract_pdf_pages,
)

records = chunk_document_records(
    extract_pdf_pages("report.pdf"),
    chunk_size=220,
    overlap=40,
)

index = SemanticIndex().fit(
    [record["text"] for record in records],
    [record["metadata"] for record in records],
    [record["id"] for record in records],
)
hits = index.search("How is customer churn predicted?", top_k=5)
messages = build_grounded_messages("How is churn predicted?", hits)
```

### Use Pinecone for Vector Search

```python
from MAna.rag import PineconeVectorStore

# Reads PINECONE_API_KEY from the environment.
store = PineconeVectorStore("mana-docs", dimension=384).ensure_index()
store.upsert_records(records, embedding_model, text_key="text", batch_size=64)
matches = store.search_text("medical symptom retrieval", embedding_model, top_k=5)
```

### Run a Provider-Neutral RAG Pipeline

```python
from MAna.rag import RAGPipeline, chat_completion, create_openai_client

client = create_openai_client()  # Reads OPENAI_API_KEY.
pipeline = RAGPipeline(
    retrieve=lambda query, top_k=5: index.search(query, top_k=top_k),
    generate=lambda messages: chat_completion(
        client,
        messages,
        model="your-chat-model",
    ),
)

result = pipeline.run(
    "How is customer churn predicted?",
    retrieve_kwargs={"top_k": 5},
)
print(result.answer)
print(result.sources)
```

### Evaluate Ranked Retrieval

```python
from MAna.rag import retrieval_metrics

metrics = retrieval_metrics(
    retrieved=[hit.id for hit in hits],
    relevant=["expected-chunk-id"],
    k=5,
)
```

### Fuse Retrieval or Recommendation Rankings

```python
from MAna.rag import reciprocal_rank_fusion

ranking = reciprocal_rank_fusion(
    {
        "semantic": semantic_ids,
        "metadata": metadata_ids,
        "collaborative": collaborative_ids,
    },
    weights={"semantic": 1.5, "metadata": 1.0, "collaborative": 0.8},
)
```

### Train and Evaluate a Hybrid Recommender

```python
from MAna.recommend import (
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    evaluate_recommender,
    temporal_train_test_split,
)

train, test = temporal_train_test_split(
    interactions,
    timestamp_column="timestamp",
)

popularity = PopularityRecommender().fit(train)
neighbors = ItemKNNRecommender(n_neighbors=50).fit(train)
model = HybridRecommender(
    {"popularity": popularity, "neighbors": neighbors},
    weights={"popularity": 0.3, "neighbors": 0.7},
)

recommendations = model.recommend("user-42", top_n=10)
metrics = evaluate_recommender(
    model,
    test,
    k=10,
    catalog_items=interactions["item_id"].unique(),
)
```

### Process Data with Spark

```python
from MAna.big import (
    dataframe_summary,
    read_spark_data,
    snake_case_columns,
    spark_session,
    write_spark_data,
)

with spark_session("MAna ETL", master="local[*]") as spark:
    events = read_spark_data(
        spark,
        "events/*.json",
        format="json",
        schema="user_id STRING, event_time TIMESTAMP, amount DOUBLE",
    )
    events = snake_case_columns(events)
    events = events.withColumn("event_date", events["event_time"].cast("date"))
    print(dataframe_summary(events))

    write_spark_data(
        events,
        "warehouse/events",
        format="parquet",
        mode="overwrite",
        partition_by=["event_date"],
        num_partitions=8,
    )
```

## Package Structure

```text
MAna/
  analysis/      Visualization and dashboard helpers
  big/           Local batching and optional Spark workflows
  data/          Data loading, cleaning, validation, and preprocessing
  database/      Database connectors, SQL helpers, and query builders
  modeling/      Model training, evaluation, persistence, and visualization
  nlp/           Text preprocessing, chunking, vectorization, topics, and embeddings
  rag/           RAG pipelines, ingestion, retrieval, citations, evaluation, and vector stores
  recommend/     Popularity, content, collaborative, hybrid, and evaluation tools
  stata/         Statistical tests, A/B testing, and experiment utilities
  timeseries/    Time-series preprocessing, forecasting, evaluation, and plots
```

## Roadmap

- More RAG reranking and vector database integrations.
- Matrix-factorization, session-based, and online-learning recommenders.
- Spark connectors, lakehouse formats, and richer streaming diagnostics.
- Complete documentation pages and more examples.

## License

This project is released under the MIT License.

## Author

Muhammad Abdulsalam

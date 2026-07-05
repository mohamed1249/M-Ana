# M-Ana Roadmap

## Phase 0: Harden the existing package

- Make Stata statistical utilities, Database, and Time Series installable and importable.
- Stabilize their public APIs and advertised examples.
- Add smoke tests for core workflows.
- Separate required dependencies from optional feature dependencies.

## Phase 1: NLP (completed in 0.2.0)

Shared text infrastructure for later RAG and recommender modules:

- Text cleaning, normalization, tokenization, and chunking
- Classical feature extraction with bag-of-words and TF-IDF
- Sentiment analysis and topic modeling
- Transformer pipelines and embeddings
- Consistent result objects and optional-dependency handling

## Phase 2: RAG and Vector Databases (foundation completed in 0.7.0)

- Document loaders and metadata handling
- Chunking strategies
- Embedding interfaces
- Local FAISS support
- Optional Pinecone and Weaviate adapters
- Semantic search, retrieval, reranking, and RAG evaluation

Completed foundation:

- Persistent semantic index with stable IDs and metadata
- PDF page loader and overlapping citation-ready chunks
- FAISS cosine helpers and generic retried vector upserts
- Pinecone index creation, batched upserts, filtered vector search, and
  environment-variable credential handling
- Grounded messages, cited context, and safe history trimming
- Weighted reciprocal-rank fusion
- Provider-neutral retrieve-augment-generate orchestration
- OpenAI-compatible chat and embedding adapters
- Ranked retrieval evaluation metrics

Remaining:

- Optional Weaviate adapter
- Cross-encoder and model-based reranking

## Phase 3: Recommender Systems (foundation completed in 0.8.0)

- Popularity and baseline recommenders
- Collaborative filtering
- Content-based recommendations using the NLP layer
- Hybrid recommenders
- Ranking metrics and offline evaluation

Completed foundation:

- Shared recommendation result contract
- Validated sparse interactions and temporal holdout splitting
- Popularity and Bayesian-rating cold-start baselines
- Dense or sparse content profiles and similar-item recommendations
- Sparse item-KNN collaborative filtering
- Weighted and reciprocal-rank hybrid models
- Top-k evaluation, catalog coverage, and intra-list diversity

Remaining:

- Explicit and implicit matrix-factorization models
- Session-aware and sequential recommenders
- Online evaluation and contextual bandit helpers

## Phase 4: PySpark (foundation completed in 0.9.0)

- Spark session and data-loading helpers
- Data cleaning and feature engineering
- Distributed SQL workflows
- MLlib training and evaluation wrappers
- Scaling paths for stable M-Ana APIs

Completed foundation:

- Lazy optional PySpark imports and configurable Spark sessions
- Schema-aware batch and table I/O with partition-aware writes
- Safe joins, type casting, missing values, unions, lineage, and deduplication
- Schema, duplicate-key, null, partition, and plan diagnostics
- Mixed-feature Spark ML pipelines and reproducible splitting
- Checkpointed Structured Streaming helpers
- Bounded local batching, chunked CSV ingestion, and threaded I/O

Remaining:

- Kafka-specific source and sink recipes
- Delta Lake and Apache Iceberg adapters
- Watermark, state-store, and streaming-observability helpers
- Cluster sizing and partition-tuning diagnostics

## Definition of done for each module

- Importable from a documented public namespace
- Optional dependencies fail with actionable messages
- Core API has automated smoke tests
- At least one end-to-end README example
- API and dependencies are included in release metadata

## Completed-module enhancements

### Time Series (0.4.0)

- Added grouped panel lags and regular panel aggregation.
- Added chronological split-before-scale workflows with train-only scalers.
- Added periodogram spectra, dominant-period detection, and seasonal dashboards.
- Added a standard `NeuralProphetForecaster` interface.

### Database and RAG (0.5.0)

- Added SQLite schema initialization and `INSERT OR IGNORE` batch helpers.
- Added MongoDB URI connection helpers and atomic document upserts.
- Added an optional Pinecone vector-store adapter backed by the current
  `pinecone` Python SDK.

# RAG Module Notebook

This notebook documents `MAna.rag`, the retrieval-augmented generation layer
for data-science and document-intelligence workflows.

It uses real datasets from the documentation set: TMDB movie overviews,
women's clothing reviews, and YouTube trending-video metadata. The default
examples stay local with a TF-IDF semantic index and fake provider clients, so
the full workflow can be studied without API keys or network calls. Optional
sections show where OpenAI-compatible clients, Pinecone, FAISS, and PDF
extraction fit when those dependencies are available.

Open it here:

[MAna.rag - Local Retrieval, Grounded Prompts, Rank Fusion, and RAG Pipelines](../notebooks/rag_module.ipynb)

The code comments are intentionally detailed because RAG systems fail in small
places: missing metadata, unstable IDs, duplicated chunks, unbounded context,
untested retrieval, or generation code that cannot be swapped without
rewriting the pipeline.

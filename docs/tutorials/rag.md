# Build a small local RAG workflow

M-Ana separates ingestion, retrieval, prompting, and generation. That keeps the
retriever testable without making a network call and lets you replace the model
provider independently.

```python
from MAna.rag import SemanticIndex

index = SemanticIndex(backend="tfidf")
index.fit(
    ["Refunds are accepted for 30 days.", "Digital items are non-refundable."],
    metadata=[{"source": "policy", "section": 1}, {"source": "policy", "section": 2}],
)

matches = index.search("How long is the refund period?", top_k=3)
index.save("./rag-index")
```

Keep source and page metadata on every chunk. Pass secrets through environment
variables or explicit client objects; never store credentials in the index.

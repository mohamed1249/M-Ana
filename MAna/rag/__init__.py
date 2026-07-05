"""Retrieval-augmented generation and vector-search utilities."""

from .loaders import extract_pdf_pages
from .evaluation import retrieval_metrics
from .generation import (
    DOCUMENT_TASK_PROMPTS,
    OpenAIEmbeddingModel,
    build_document_task_messages,
    chat_completion,
    create_azure_openai_client,
    create_openai_client,
    openai_embeddings,
    require_environment_variable,
    run_document_task,
)
from .ingestion import chunk_document_records
from .pipeline import RAGPipeline, RAGResult, rag_answer
from .prompting import (
    build_cited_context,
    build_grounded_messages,
    deduplicate_hits,
    estimate_tokens,
    trim_chat_history,
)
from .ranking import RankedItem, fuse_rankings, reciprocal_rank_fusion
from .retrieval import SearchHit, SemanticIndex, make_stable_id
from .pinecone import (
    PineconeVectorStore,
    create_pinecone_index_if_absent,
    search_pinecone,
)
from .vectorstores import (
    build_faiss_cosine_index,
    normalize_embeddings,
    search_faiss,
    upsert_embedding_batches,
)

__all__ = [
    "SearchHit",
    "SemanticIndex",
    "make_stable_id",
    "chunk_document_records",
    "RAGPipeline",
    "RAGResult",
    "rag_answer",
    "retrieval_metrics",
    "DOCUMENT_TASK_PROMPTS",
    "OpenAIEmbeddingModel",
    "build_document_task_messages",
    "chat_completion",
    "create_openai_client",
    "create_azure_openai_client",
    "openai_embeddings",
    "require_environment_variable",
    "run_document_task",
    "RankedItem",
    "fuse_rankings",
    "reciprocal_rank_fusion",
    "PineconeVectorStore",
    "create_pinecone_index_if_absent",
    "search_pinecone",
    "extract_pdf_pages",
    "build_cited_context",
    "build_grounded_messages",
    "deduplicate_hits",
    "estimate_tokens",
    "trim_chat_history",
    "normalize_embeddings",
    "build_faiss_cosine_index",
    "search_faiss",
    "upsert_embedding_batches",
]

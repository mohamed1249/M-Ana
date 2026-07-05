"""Natural-language processing tools for cleaning, analysis, and retrieval."""

from .chunking import DocumentChunk, TextChunker, chunk_documents, chunk_text
from .classification import (
    TransformerClassifier,
    build_text_classifier,
    evaluate_classifier,
)
from .embeddings import (
    EmbeddingResult,
    TextEmbedder,
    encode_segmented_texts,
    segment_text,
    semantic_search,
)
from .preprocessing import (
    TextCleaner,
    clean_corpus,
    clean_text,
    normalize_unicode,
    strip_accents,
    tokenize,
)
from .sentiment import SentimentAnalyzer, SentimentResult, analyze_sentiment
from .topics import Topic, TopicModeler, TopicModelResult, model_topics
from .vectorization import (
    TextVectorizer,
    VectorizationResult,
    cosine_similarity_matrix,
    vectorize_texts,
)

__all__ = [
    "TextCleaner",
    "clean_text",
    "clean_corpus",
    "normalize_unicode",
    "strip_accents",
    "tokenize",
    "DocumentChunk",
    "TextChunker",
    "chunk_text",
    "chunk_documents",
    "build_text_classifier",
    "evaluate_classifier",
    "TransformerClassifier",
    "TextVectorizer",
    "VectorizationResult",
    "vectorize_texts",
    "cosine_similarity_matrix",
    "SentimentAnalyzer",
    "SentimentResult",
    "analyze_sentiment",
    "Topic",
    "TopicModeler",
    "TopicModelResult",
    "model_topics",
    "EmbeddingResult",
    "TextEmbedder",
    "segment_text",
    "encode_segmented_texts",
    "semantic_search",
]

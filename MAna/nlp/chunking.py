"""Document chunking utilities for NLP and retrieval workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


_WORD_RE = re.compile(r"\S+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)\]]*)\s+")


@dataclass(frozen=True)
class DocumentChunk:
    """One document chunk with source offsets and metadata."""

    text: str
    index: int
    start: int
    end: int
    document_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "document_index": self.document_index,
            "metadata": dict(self.metadata),
        }


def _validate_sizes(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _word_spans(text: str) -> List[tuple[int, int]]:
    return [match.span() for match in _WORD_RE.finditer(text)]


def _sentence_spans(text: str) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_RE.finditer(text):
        end = boundary.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = boundary.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _window_chunks(
    text: str,
    spans: Sequence[tuple[int, int]],
    chunk_size: int,
    overlap: int,
    document_index: int,
    metadata: Dict[str, Any],
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    step = chunk_size - overlap
    for chunk_index, unit_start in enumerate(range(0, len(spans), step)):
        selected = spans[unit_start : unit_start + chunk_size]
        if not selected:
            break
        start, end = selected[0][0], selected[-1][1]
        raw_chunk = text[start:end]
        leading = len(raw_chunk) - len(raw_chunk.lstrip())
        trailing = len(raw_chunk) - len(raw_chunk.rstrip())
        start += leading
        end -= trailing
        chunk_text_value = text[start:end]
        if chunk_text_value:
            chunks.append(
                DocumentChunk(
                    text=chunk_text_value,
                    index=chunk_index,
                    start=start,
                    end=end,
                    document_index=document_index,
                    metadata=dict(metadata),
                )
            )
        if unit_start + chunk_size >= len(spans):
            break
    return chunks


def chunk_text(
    text: object,
    chunk_size: int = 200,
    overlap: int = 20,
    unit: str = "words",
    *,
    document_index: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[DocumentChunk]:
    """Split text into overlapping chunks.

    ``unit`` may be ``"words"``, ``"characters"``, or ``"sentences"``.
    The returned offsets always refer to character positions in the source.
    """
    _validate_sizes(chunk_size, overlap)
    value = "" if text is None else str(text)
    if not value.strip():
        return []

    metadata_value = dict(metadata or {})
    unit = unit.lower()
    if unit == "words":
        spans = _word_spans(value)
    elif unit == "sentences":
        spans = _sentence_spans(value)
    elif unit == "characters":
        spans = [(index, index + 1) for index in range(len(value))]
    else:
        raise ValueError("unit must be 'words', 'characters', or 'sentences'")

    return _window_chunks(
        value,
        spans,
        chunk_size,
        overlap,
        document_index,
        metadata_value,
    )


def chunk_documents(
    documents: Sequence[object],
    chunk_size: int = 200,
    overlap: int = 20,
    unit: str = "words",
    metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> List[DocumentChunk]:
    """Chunk several documents while retaining their source indices."""
    if metadata is not None and len(metadata) != len(documents):
        raise ValueError("metadata must have the same length as documents")

    chunks: List[DocumentChunk] = []
    for document_index, document in enumerate(documents):
        document_metadata = metadata[document_index] if metadata is not None else None
        chunks.extend(
            chunk_text(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
                unit=unit,
                document_index=document_index,
                metadata=document_metadata,
            )
        )
    return chunks


@dataclass
class TextChunker:
    """Reusable chunking configuration."""

    chunk_size: int = 200
    overlap: int = 20
    unit: str = "words"

    def __post_init__(self) -> None:
        _validate_sizes(self.chunk_size, self.overlap)
        if self.unit not in {"words", "characters", "sentences"}:
            raise ValueError("unit must be 'words', 'characters', or 'sentences'")

    def split(
        self,
        text: object,
        *,
        document_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        return chunk_text(
            text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            unit=self.unit,
            document_index=document_index,
            metadata=metadata,
        )

    def transform(
        self,
        documents: Sequence[object],
        metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[DocumentChunk]:
        return chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            unit=self.unit,
            metadata=metadata,
        )

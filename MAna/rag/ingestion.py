"""Prepare citation-ready chunk records for retrieval indexes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ..nlp.chunking import chunk_text
from .retrieval import make_stable_id


def chunk_document_records(
    documents: Sequence[Any],
    *,
    chunk_size: int = 200,
    overlap: int = 20,
    unit: str = "words",
    text_key: str = "text",
    metadata_key: str = "metadata",
    id_prefix: str = "chunk",
) -> List[Dict[str, Any]]:
    """Chunk strings or document mappings into index-ready records.

    Mappings produced by :func:`extract_pdf_pages` are accepted directly. Each
    output contains ``id``, ``text``, and citation metadata with source offsets,
    document index, and chunk index.
    """
    records: List[Dict[str, Any]] = []
    for document_index, document in enumerate(documents):
        if isinstance(document, Mapping):
            text = document.get(text_key, "")
            raw_metadata = document.get(metadata_key, {}) or {}
            if not isinstance(raw_metadata, Mapping):
                raise TypeError(f"{metadata_key!r} must contain a mapping")
            metadata = dict(raw_metadata)
            for key in ("source", "source_path", "page", "page_index"):
                if key in document and key not in metadata:
                    metadata[key] = document[key]
            if document.get("id") is not None:
                metadata.setdefault("document_id", str(document["id"]))
        else:
            text = document
            metadata = {}

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            unit=unit,
            document_index=document_index,
            metadata=metadata,
        )
        for chunk in chunks:
            chunk_metadata = dict(chunk.metadata)
            chunk_metadata.update(
                {
                    "document_index": document_index,
                    "chunk_index": chunk.index,
                    "start": chunk.start,
                    "end": chunk.end,
                }
            )
            records.append(
                {
                    "id": make_stable_id(
                        chunk.text,
                        chunk_metadata,
                        prefix=id_prefix,
                    ),
                    "text": chunk.text,
                    "metadata": chunk_metadata,
                }
            )
    return records


__all__ = ["chunk_document_records"]

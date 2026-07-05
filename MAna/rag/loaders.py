"""Document loaders that retain citation metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_pdf_pages(
    path: str,
    *,
    password: Optional[str] = None,
    min_characters: int = 1,
) -> List[Dict[str, Any]]:
    """Extract PDF text page by page with source metadata."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF extraction requires the RAG extra: "
            "pip install 'M_Ana_package[rag]'"
        ) from exc

    source = Path(path)
    reader = PdfReader(str(source))
    if reader.is_encrypted:
        if password is None or reader.decrypt(password) == 0:
            raise ValueError("the PDF is encrypted and requires a valid password")

    pages: List[Dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < min_characters:
            continue
        pages.append(
            {
                "text": text,
                "metadata": {
                    "source": source.name,
                    "source_path": str(source.resolve()),
                    "page": page_index + 1,
                    "page_index": page_index,
                    "total_pages": len(reader.pages),
                },
            }
        )
    return pages

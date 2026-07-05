"""Provider-neutral retrieve-augment-generate orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .prompting import (
    _hit_parts,
    build_grounded_messages,
    deduplicate_hits,
    trim_chat_history,
)


def _generated_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("answer", "content", "text"):
            if value.get(key) is not None:
                return str(value[key])
    if getattr(value, "content", None) is not None:
        return str(value.content)
    return str(value)


def _source_references(hits: Sequence[Any]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    for number, hit in enumerate(hits, start=1):
        _, metadata, score = _hit_parts(hit)
        item_id = hit.get("id") if isinstance(hit, Mapping) else getattr(hit, "id", None)
        references.append(
            {
                "label": f"Source {number}",
                "id": item_id,
                "source": metadata.get(
                    "source", metadata.get("source_path", "unknown source")
                ),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "score": None if score is None else float(score),
            }
        )
    return references


@dataclass
class RAGResult:
    """Answer, evidence, prompt messages, and normalized source references."""

    query: str
    answer: str
    hits: List[Any]
    messages: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "hits": list(self.hits),
            "messages": [dict(message) for message in self.messages],
            "sources": [dict(source) for source in self.sources],
        }


class RAGPipeline:
    """Combine arbitrary retriever and message-generator callables."""

    def __init__(
        self,
        retrieve: Callable[..., Sequence[Any]],
        generate: Callable[..., Any],
        *,
        system_prompt: Optional[str] = None,
        max_context_characters: Optional[int] = None,
        history_max_tokens: Optional[int] = 2_048,
        deduplicate_context: bool = True,
    ) -> None:
        if not callable(retrieve) or not callable(generate):
            raise TypeError("retrieve and generate must be callable")
        if max_context_characters is not None and max_context_characters < 0:
            raise ValueError("max_context_characters cannot be negative")
        if history_max_tokens is not None and history_max_tokens <= 0:
            raise ValueError("history_max_tokens must be greater than zero")
        self.retrieve = retrieve
        self.generate = generate
        self.system_prompt = system_prompt
        self.max_context_characters = max_context_characters
        self.history_max_tokens = history_max_tokens
        self.deduplicate_context = deduplicate_context

    def run(
        self,
        query: Any,
        *,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
        retrieve_kwargs: Optional[Mapping[str, Any]] = None,
        generation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> RAGResult:
        """Retrieve evidence, build grounded messages, and generate an answer."""
        query_text = str(query).strip()
        if not query_text:
            raise ValueError("query cannot be empty")

        raw_hits = self.retrieve(query_text, **dict(retrieve_kwargs or {}))
        if raw_hits is None:
            raise ValueError("retrieve returned None; return an empty sequence instead")
        hits = list(raw_hits)
        if self.deduplicate_context:
            hits = deduplicate_hits(hits)
        hits = [hit for hit in hits if _hit_parts(hit)[0].strip()]

        grounded = build_grounded_messages(
            query_text,
            hits,
            system_prompt=self.system_prompt,
            max_context_characters=self.max_context_characters,
        )
        history_messages = [
            dict(message)
            for message in (history or [])
            if message.get("role") in {"user", "assistant", "tool"}
            and message.get("content") is not None
        ]
        if self.history_max_tokens is not None:
            history_messages = trim_chat_history(
                history_messages,
                max_tokens=self.history_max_tokens,
                preserve_system=False,
            )
        messages = [grounded[0], *history_messages, grounded[1]]
        generated = self.generate(messages, **dict(generation_kwargs or {}))
        return RAGResult(
            query=query_text,
            answer=_generated_text(generated),
            hits=hits,
            messages=messages,
            sources=_source_references(hits),
        )


def rag_answer(
    query: Any,
    retrieve: Callable[..., Sequence[Any]],
    generate: Callable[..., Any],
    *,
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    retrieve_kwargs: Optional[Mapping[str, Any]] = None,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
    **pipeline_kwargs: Any,
) -> RAGResult:
    """Run a one-off provider-neutral RAG pipeline."""
    return RAGPipeline(retrieve, generate, **pipeline_kwargs).run(
        query,
        history=history,
        retrieve_kwargs=retrieve_kwargs,
        generation_kwargs=generation_kwargs,
    )


__all__ = ["RAGPipeline", "RAGResult", "rag_answer"]

"""Optional model-client adapters and reusable document-generation tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .prompting import build_cited_context
from .vectorstores import normalize_embeddings


Message = Dict[str, str]


def require_environment_variable(name: str) -> str:
    """Return required configuration without placing secrets in source code."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Set the {name} environment variable before continuing.")
    return value


def _import_openai():
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI clients require the RAG extra: "
            "pip install 'M_Ana_package[rag]'"
        ) from exc
    return OpenAI, AzureOpenAI


def create_openai_client(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    **client_kwargs: Any,
) -> Any:
    """Create an OpenAI client using an environment-provided API key."""
    OpenAI, _ = _import_openai()
    return OpenAI(
        api_key=require_environment_variable(api_key_env),
        **client_kwargs,
    )


def create_azure_openai_client(
    *,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
    api_version_env: str = "AZURE_OPENAI_API_VERSION",
    api_key_env: str = "AZURE_OPENAI_API_KEY",
    **client_kwargs: Any,
) -> Any:
    """Create an Azure OpenAI client from explicit or environment configuration."""
    _, AzureOpenAI = _import_openai()
    return AzureOpenAI(
        azure_endpoint=endpoint or require_environment_variable(endpoint_env),
        api_version=api_version or require_environment_variable(api_version_env),
        api_key=require_environment_variable(api_key_env),
        **client_kwargs,
    )


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("answer", "content", "text"):
            if content.get(key) is not None:
                return _content_text(content[key])
        return str(content)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: List[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text", part.get("content"))
            else:
                text = getattr(part, "text", getattr(part, "content", None))
            if text is not None:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def chat_completion(
    client: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    model: str,
    temperature: Optional[float] = 0.2,
    max_tokens: Optional[int] = 800,
    **request_kwargs: Any,
) -> str:
    """Run an OpenAI-compatible chat completion and return response text."""
    if not messages:
        raise ValueError("messages cannot be empty")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        **request_kwargs,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = client.chat.completions.create(**payload)
    choices = response.get("choices", []) if isinstance(response, Mapping) else response.choices
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message", {}) if isinstance(first, Mapping) else first.message
    content = message.get("content") if isinstance(message, Mapping) else message.content
    return _content_text(content)


@dataclass
class OpenAIEmbeddingModel:
    """Adapt an OpenAI-compatible embeddings client to MAna vector stores."""

    client: Any
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    batch_size: int = 100
    normalize_vectors: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

    def encode(
        self,
        texts: Union[str, Sequence[Any]],
        *,
        batch_size: Optional[int] = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: Optional[bool] = None,
        **request_kwargs: Any,
    ) -> Any:
        """Embed texts in ordered batches using the configured endpoint."""
        values = (
            [texts]
            if isinstance(texts, str)
            else ["" if text is None else str(text) for text in texts]
        )
        if not values:
            empty = np.empty((0, self.dimensions or 0), dtype=np.float32)
            return empty if convert_to_numpy else empty.tolist()

        resolved_batch_size = self.batch_size if batch_size is None else batch_size
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        vectors: List[List[float]] = []
        for start in range(0, len(values), resolved_batch_size):
            payload: Dict[str, Any] = {
                "model": self.model,
                "input": values[start : start + resolved_batch_size],
                **request_kwargs,
            }
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**payload)
            data = response.get("data", []) if isinstance(response, Mapping) else response.data
            ordered = sorted(
                data,
                key=lambda item: item.get("index", 0)
                if isinstance(item, Mapping)
                else item.index,
            )
            for item in ordered:
                embedding = item.get("embedding") if isinstance(item, Mapping) else item.embedding
                vectors.append(list(embedding))

        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim != 2 or result.shape[0] != len(values):
            raise ValueError("embedding endpoint returned an unexpected shape")
        should_normalize = (
            self.normalize_vectors
            if normalize_embeddings is None
            else normalize_embeddings
        )
        if should_normalize:
            result = normalize_embeddings_array(result)
        return result if convert_to_numpy else result.tolist()


def normalize_embeddings_array(vectors: Any) -> np.ndarray:
    """Normalize generated embeddings without shadowing API parameters."""
    return normalize_embeddings(vectors)


def openai_embeddings(
    client: Any,
    texts: Union[str, Sequence[Any]],
    *,
    model: str = "text-embedding-3-small",
    dimensions: Optional[int] = None,
    batch_size: int = 100,
    normalize: bool = True,
) -> np.ndarray:
    """Convenience wrapper for ordered OpenAI-compatible embeddings."""
    adapter = OpenAIEmbeddingModel(
        client=client,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
        normalize_vectors=normalize,
    )
    return adapter.encode(texts)


DOCUMENT_TASK_PROMPTS = {
    "summarize": "Summarize the document faithfully and concisely.",
    "classify": "Return only the best document classification label.",
    "tag": "Return only a concise comma-separated list of document tags.",
}


def build_document_task_messages(
    document: Any,
    *,
    task: str,
    custom_instruction: Optional[str] = None,
) -> List[Message]:
    """Build messages for summarization, classification, or document tagging."""
    instruction = custom_instruction or DOCUMENT_TASK_PROMPTS.get(task)
    if instruction is None:
        raise ValueError(f"Unknown task {task!r}; provide custom_instruction.")

    if isinstance(document, str):
        content = document
    elif isinstance(document, Mapping):
        content = build_cited_context([document])
    elif isinstance(document, Sequence):
        content = build_cited_context(document)
    else:
        content = str(document)
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]


def run_document_task(
    document: Any,
    *,
    task: str,
    generate: Any,
    custom_instruction: Optional[str] = None,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Run a provider-neutral document task through a message generator."""
    messages = build_document_task_messages(
        document,
        task=task,
        custom_instruction=custom_instruction,
    )
    return _content_text(generate(messages, **dict(generation_kwargs or {})))


__all__ = [
    "DOCUMENT_TASK_PROMPTS",
    "OpenAIEmbeddingModel",
    "build_document_task_messages",
    "chat_completion",
    "create_azure_openai_client",
    "create_openai_client",
    "openai_embeddings",
    "require_environment_variable",
    "run_document_task",
]

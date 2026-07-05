"""Optional Pinecone adapter for MAna RAG workflows."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .vectorstores import _encode_batch, normalize_embeddings, upsert_embedding_batches


def _import_pinecone(use_grpc: bool = False):
    try:
        if use_grpc:
            from pinecone.grpc import PineconeGRPC as Pinecone
        else:
            from pinecone import Pinecone
        from pinecone import ServerlessSpec
    except ImportError as exc:
        raise ImportError(
            "Pinecone support requires the RAG extra: "
            "pip install 'M_Ana_package[rag]'"
        ) from exc
    return Pinecone, ServerlessSpec


def _index_names(client: Any) -> Sequence[str]:
    indexes = client.list_indexes()
    names = getattr(indexes, "names", None)
    if callable(names):
        return list(names())
    if names is not None:
        return list(names)

    if isinstance(indexes, Mapping):
        indexes = indexes.get("indexes", indexes.values())

    resolved = []
    for item in indexes:
        if isinstance(item, str):
            resolved.append(item)
        elif isinstance(item, Mapping) and item.get("name"):
            resolved.append(str(item["name"]))
        elif getattr(item, "name", None):
            resolved.append(str(item.name))
    return resolved


def _has_index(client: Any, index_name: str) -> bool:
    has_index = getattr(client, "has_index", None)
    if callable(has_index):
        try:
            return bool(has_index(index_name))
        except TypeError:
            return bool(has_index(name=index_name))
    return index_name in _index_names(client)


def _matches_from_response(response: Any) -> Sequence[Any]:
    if isinstance(response, Mapping):
        return response.get("matches", [])
    return getattr(response, "matches", [])


class PineconeVectorStore:
    """Small Pinecone wrapper for semantic search and RAG ingestion.

    API keys are read from the explicit ``api_key`` argument or the
    ``PINECONE_API_KEY`` environment variable. No credentials are stored by
    MAna.
    """

    def __init__(
        self,
        index_name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        namespace: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        index: Optional[Any] = None,
        use_grpc: bool = False,
        **client_kwargs,
    ) -> None:
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.namespace = namespace
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.client = client
        self.index = index
        self.use_grpc = use_grpc
        self.client_kwargs = client_kwargs

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise ValueError(
                "Set PINECONE_API_KEY or pass api_key to use PineconeVectorStore"
            )
        Pinecone, _ = _import_pinecone(self.use_grpc)
        self.client = Pinecone(api_key=self.api_key, **self.client_kwargs)
        return self.client

    def connect(self, **index_kwargs) -> "PineconeVectorStore":
        """Attach the configured Pinecone index and return ``self``."""
        if self.index is None:
            self.index = self._get_client().Index(self.index_name, **index_kwargs)
        return self

    def ensure_index(
        self,
        *,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        spec: Optional[Any] = None,
        wait: bool = True,
        timeout_s: float = 120.0,
        poll_interval_s: float = 2.0,
        deletion_protection: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> "PineconeVectorStore":
        """Create the Pinecone index if it does not exist, then connect it."""
        client = self._get_client()
        if not _has_index(client, self.index_name):
            index_dimension = dimension or self.dimension
            if index_dimension is None:
                raise ValueError("dimension is required when creating a Pinecone index")

            index_metric = metric or self.metric
            if spec is None:
                _, ServerlessSpec = _import_pinecone(self.use_grpc)
                spec = ServerlessSpec(
                    cloud=cloud or self.cloud,
                    region=region or self.region,
                )

            create_kwargs = {
                "name": self.index_name,
                "dimension": int(index_dimension),
                "metric": index_metric,
                "spec": spec,
            }
            if deletion_protection is not None:
                create_kwargs["deletion_protection"] = deletion_protection
            if tags is not None:
                create_kwargs["tags"] = dict(tags)

            client.create_index(**create_kwargs)
            self.dimension = int(index_dimension)
            self.metric = index_metric

            if wait:
                self._wait_until_ready(timeout_s, poll_interval_s)

        return self.connect()

    def _wait_until_ready(self, timeout_s: float, poll_interval_s: float) -> None:
        client = self._get_client()
        describe_index = getattr(client, "describe_index", None)
        if not callable(describe_index):
            return

        deadline = time.monotonic() + timeout_s
        while True:
            description = describe_index(self.index_name)
            status = getattr(description, "status", None)
            if isinstance(description, Mapping):
                status = description.get("status", status)

            ready = None
            if isinstance(status, Mapping):
                ready = status.get("ready")
            elif status is not None:
                ready = getattr(status, "ready", None)

            if ready is True or status is None:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Pinecone index {self.index_name!r} was not ready")
            time.sleep(poll_interval_s)

    def upsert_records(
        self,
        records: Sequence[Mapping[str, Any]],
        embedding_model: Any,
        *,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> int:
        """Encode and upsert records with MAna's deterministic batch writer."""
        self.connect()
        return upsert_embedding_batches(
            self.index,
            [dict(record) for record in records],
            embedding_model,
            namespace=self.namespace if namespace is None else namespace,
            **kwargs,
        )

    def query(
        self,
        vector: Any,
        *,
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_values: bool = False,
        normalize: bool = True,
        **kwargs,
    ) -> Any:
        """Query Pinecone with an embedding vector and return the full response."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.connect()
        values = np.asarray(vector, dtype=np.float32)
        query_vector = (
            normalize_embeddings(values)[0].tolist()
            if normalize
            else values.reshape(-1).tolist()
        )

        query_kwargs = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_values": include_values,
            **kwargs,
        }
        target_namespace = self.namespace if namespace is None else namespace
        if target_namespace is not None:
            query_kwargs["namespace"] = target_namespace
        if metadata_filter is not None:
            query_kwargs["filter"] = dict(metadata_filter)

        return self.index.query(**query_kwargs)

    def search(
        self,
        vector: Any,
        *,
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_values: bool = False,
        normalize: bool = True,
        **kwargs,
    ) -> Sequence[Any]:
        """Query Pinecone and return only the matched records."""
        response = self.query(
            vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            namespace=namespace,
            include_metadata=include_metadata,
            include_values=include_values,
            normalize=normalize,
            **kwargs,
        )
        return _matches_from_response(response)

    def search_text(
        self,
        text: str,
        embedding_model: Any,
        *,
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> Sequence[Any]:
        """Embed text with the provided model, query Pinecone, and return matches."""
        vector = _encode_batch(embedding_model, [text], batch_size=1)[0]
        return self.search(
            vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            namespace=namespace,
            **kwargs,
        )


def create_pinecone_index_if_absent(
    index_name: str,
    dimension: int,
    *,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    wait: bool = True,
    spec: Optional[Any] = None,
    **client_kwargs,
) -> PineconeVectorStore:
    """Create a Pinecone index when missing and return a connected store."""
    store = PineconeVectorStore(
        index_name=index_name,
        dimension=dimension,
        metric=metric,
        cloud=cloud,
        region=region,
        api_key=api_key,
        client=client,
        **client_kwargs,
    )
    return store.ensure_index(spec=spec, wait=wait)


def search_pinecone(
    index: Any,
    embedding: Any,
    *,
    top_k: int = 5,
    metadata_filter: Optional[Mapping[str, Any]] = None,
    namespace: Optional[str] = None,
    include_metadata: bool = True,
    include_values: bool = False,
    normalize: bool = True,
    **kwargs,
) -> Sequence[Any]:
    """Search a Pinecone-compatible index and return matches."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    values = np.asarray(embedding, dtype=np.float32)
    query_vector = (
        normalize_embeddings(values)[0].tolist()
        if normalize
        else values.reshape(-1).tolist()
    )
    query_kwargs = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": include_metadata,
        "include_values": include_values,
        **kwargs,
    }
    if metadata_filter is not None:
        query_kwargs["filter"] = dict(metadata_filter)
    if namespace is not None:
        query_kwargs["namespace"] = namespace

    return _matches_from_response(index.query(**query_kwargs))


__all__ = [
    "PineconeVectorStore",
    "create_pinecone_index_if_absent",
    "search_pinecone",
]

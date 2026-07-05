"""Pandas-friendly MongoDB helper functions."""

from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

from .connectors import MongoDBConnector


def get_mongo_collection(
    uri: str,
    database_name: str,
    collection_name: str,
    *,
    server_selection_timeout_ms: int = 5000,
    **kwargs
) -> Tuple[MongoDBConnector, Any]:
    """Connect from a MongoDB URI and return ``(connector, collection)``.

    The connector owns the client and can be closed with ``connector.close()``.
    """
    connector = MongoDBConnector.from_uri(
        uri,
        database=database_name,
        server_selection_timeout_ms=server_selection_timeout_ms,
        **kwargs,
    )
    return connector, connector.get_collection(collection_name)


def mongo_to_dataframe(
    connection: MongoDBConnector,
    collection_name: str,
    query: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Read MongoDB documents into a pandas DataFrame."""
    return connection.to_dataframe(
        collection_name=collection_name,
        query=query,
        projection=projection,
        limit=limit,
    )


def dataframe_to_mongo(
    df: pd.DataFrame,
    connection: MongoDBConnector,
    collection_name: str,
    if_exists: str = "append",
    batch_size: int = 1000,
) -> int:
    """Write DataFrame rows to a MongoDB collection.

    Parameters
    ----------
    if_exists:
        ``"append"`` keeps existing documents, while ``"replace"`` clears
        the collection before inserting.
    """
    if if_exists not in {"append", "replace"}:
        raise ValueError("if_exists must be 'append' or 'replace'")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    collection = connection.get_collection(collection_name)
    if if_exists == "replace":
        collection.delete_many({})

    records = df.to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if batch:
            collection.insert_many(batch)

    return len(records)


def upsert_document(
    collection: Any,
    filter_query: Mapping[str, Any],
    document: Mapping[str, Any],
    set_on_insert: Optional[Mapping[str, Any]] = None,
    **kwargs
) -> Any:
    """Atomically insert or update a MongoDB document.

    This is the safer version of the common ``find`` then ``replace``/``insert``
    pattern because MongoDB performs the upsert as one operation.
    """
    if not filter_query:
        raise ValueError("filter_query must not be empty")

    update = {"$set": dict(document)}
    if set_on_insert:
        update["$setOnInsert"] = dict(set_on_insert)

    return collection.update_one(dict(filter_query), update, upsert=True, **kwargs)


def save_document(
    collection: Any,
    file_name: str,
    content: Any,
    *,
    filename_field: str = "file_name",
    content_field: str = "content",
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Save document content by filename using an atomic MongoDB upsert."""
    document = {
        filename_field: file_name,
        content_field: content,
    }
    if extra_fields:
        document.update(dict(extra_fields))

    return upsert_document(
        collection,
        {filename_field: file_name},
        document,
    )

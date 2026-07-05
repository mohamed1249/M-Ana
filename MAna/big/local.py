"""Bounded-memory local processing helpers from recurring project patterns."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple, TypeVar, Union

import pandas as pd
from pandas.errors import EmptyDataError


T = TypeVar("T")
R = TypeVar("R")


def iter_batches(items: Sequence[T], *, batch_size: int = 100) -> Iterator[Sequence[T]]:
    """Yield fixed-size slices without copying the complete input."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def iter_dataframe_batches(
    frame: pd.DataFrame,
    *,
    batch_size: int = 100,
) -> Iterator[pd.DataFrame]:
    """Yield bounded DataFrame row slices."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def transform_in_batches(
    items: Sequence[T],
    transform: Callable[[Sequence[T]], Sequence[R]],
    *,
    batch_size: int = 100,
) -> List[R]:
    """Apply a vectorized transform to bounded batches and preserve order."""
    output: List[R] = []
    for batch in iter_batches(items, batch_size=batch_size):
        transformed = list(transform(batch))
        if len(transformed) != len(batch):
            raise ValueError("transform must return one result per input item")
        output.extend(transformed)
    return output


def threaded_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 4,
    preserve_order: bool = True,
) -> List[R]:
    """Run I/O-bound work concurrently with bounded worker count."""
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than zero")
    values = list(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if preserve_order:
            return list(executor.map(function, values))
        futures: List[Future[R]] = [executor.submit(function, value) for value in values]
        return [future.result() for future in as_completed(futures)]


def stream_csv_chunks(
    path: str,
    *,
    chunk_size: int = 100_000,
    transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
    **read_csv_kwargs,
) -> Iterator[pd.DataFrame]:
    """Read and optionally transform a CSV without loading it all at once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    for chunk in pd.read_csv(path, chunksize=chunk_size, **read_csv_kwargs):
        result = transform(chunk) if transform is not None else chunk
        if not isinstance(result, pd.DataFrame):
            raise TypeError("transform must return a pandas DataFrame")
        yield result


def read_many_csv_files(
    paths: Iterable[Union[str, Path]],
    *,
    ignore_bad_files: bool = True,
    add_source_column: bool = False,
    **read_csv_kwargs,
) -> Tuple[pd.DataFrame, List[Path]]:
    """Concatenate CSV extracts and report unreadable or empty files."""
    frames: List[pd.DataFrame] = []
    bad_files: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            frame = pd.read_csv(path, **read_csv_kwargs)
        except (EmptyDataError, OSError, UnicodeError):
            bad_files.append(path)
            if not ignore_bad_files:
                raise
            continue
        if add_source_column:
            frame = frame.copy()
            frame["source_file"] = path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, bad_files


def read_csv_directory(
    directory: str,
    *,
    pattern: str = "*.csv",
    recursive: bool = False,
    **read_csv_kwargs,
) -> Tuple[pd.DataFrame, List[Path]]:
    """Discover and concatenate CSV files under one directory."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    return read_many_csv_files(paths, **read_csv_kwargs)


__all__ = [
    "iter_batches",
    "iter_dataframe_batches",
    "read_csv_directory",
    "read_many_csv_files",
    "stream_csv_chunks",
    "threaded_map",
    "transform_in_batches",
]

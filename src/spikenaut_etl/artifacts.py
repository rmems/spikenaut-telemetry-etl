"""Exclusive artifact staging through pinned, non-symlink directory handles."""

import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import IO, Any

import pyarrow as pa
import pyarrow.parquet as pq

_PINNED_ROOT: ContextVar[tuple[Path, tuple[int, int]] | None] = ContextVar(
    "artifact_root", default=None
)


def _check_pinned_root() -> None:
    pinned = _PINNED_ROOT.get()
    if pinned is not None:
        root, expected = pinned
        actual = root.stat(follow_symlinks=False)
        if (actual.st_dev, actual.st_ino) != expected:
            raise OSError("publication directory changed during audit")


@contextmanager
def pinned_publication(path: Path) -> Iterator[None]:
    ensure_directory(path)
    descriptor = _open_directory(path)
    opened = os.fstat(descriptor)
    token = _PINNED_ROOT.set((path, (opened.st_dev, opened.st_ino)))
    try:
        yield
    finally:
        _PINNED_ROOT.reset(token)
        os.close(descriptor)


def directory_identity(path: Path) -> tuple[int, int]:
    _check_pinned_root()
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"publication path is not a directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _check_directory(path: Path, descriptor: int) -> None:
    _check_pinned_root()
    opened = os.fstat(descriptor)
    current = path.stat(follow_symlinks=False)
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise OSError(f"artifact directory changed during publication: {path}")


def _open_directory(path: Path) -> int:
    """Walk from the filesystem root without following any symlink component."""
    _check_pinned_root()
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            if component == "..":
                raise OSError("parent traversal is not allowed for artifact paths")
            try:
                os.mkdir(component, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        _check_pinned_root()
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _staged_output(path: Path, mode: str) -> Iterator[IO[Any]]:
    directory = _open_directory(path.parent)
    temporary = ".artifact-" + secrets.token_hex(16)
    created = False
    try:
        _check_directory(path.parent, directory)
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        created = True
        with os.fdopen(descriptor, mode) as stream:
            yield stream
        _check_directory(path.parent, directory)
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        _check_directory(path.parent, directory)
    finally:
        try:
            if created:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
        finally:
            os.close(directory)


def write_json(path: Path, value: Any) -> None:
    with _staged_output(path, "w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_parquet(path: Path, table: pa.Table) -> None:
    with _staged_output(path, "wb") as stream:
        pq.write_table(table, stream)


def clean_artifacts(path: Path, names: tuple[str, ...] | None = None) -> None:
    """Delete owned files relative to a pinned directory, never through symlinks."""
    directory = _open_directory(path)
    try:
        _check_directory(path, directory)
        _clean_entries(directory, names)
        _check_directory(path, directory)
    finally:
        os.close(directory)


def _clean_entries(directory: int, names: tuple[str, ...] | None) -> None:
    selected = names if names is not None else tuple(os.listdir(directory))
    for name in selected:
        try:
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if names is None and stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            try:
                _clean_entries(child, None)
            finally:
                os.close(child)
        elif names is None and not name.endswith(".parquet"):
            continue
        elif stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            os.unlink(name, dir_fd=directory)
        elif stat.S_ISDIR(metadata.st_mode) and name.endswith(".tmp"):
            try:
                os.rmdir(name, dir_fd=directory)
            except OSError:
                pass


def ensure_directory(path: Path) -> None:
    descriptor = _open_directory(path)
    try:
        _check_directory(path, descriptor)
    finally:
        os.close(descriptor)

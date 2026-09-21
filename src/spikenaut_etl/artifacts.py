"""Exclusive artifact staging through pinned, non-symlink directory handles."""

import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import pyarrow as pa
import pyarrow.parquet as pq


def directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"publication path is not a directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _check_directory(path: Path, descriptor: int) -> None:
    opened = os.fstat(descriptor)
    current = path.stat(follow_symlinks=False)
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise OSError(f"artifact directory changed during publication: {path}")


def _open_directory(path: Path) -> int:
    """Walk from the filesystem root without following any symlink component."""
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

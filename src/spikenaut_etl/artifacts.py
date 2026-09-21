"""Exclusive artifact staging through pinned, non-symlink directory handles."""

import fcntl
import hashlib
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

_PINNED_ROOT: ContextVar[tuple[Path, tuple[int, int], dict[Path, int]] | None] = (
    ContextVar("artifact_root", default=None)
)
_NO_REPLACE_PUBLICATION: ContextVar[bool] = ContextVar(
    "no_replace_publication", default=False
)


def _check_pinned_root(path: Path) -> None:
    pinned = _PINNED_ROOT.get()
    if pinned is not None:
        root, expected, children = pinned
        actual = root.stat(follow_symlinks=False)
        if (actual.st_dev, actual.st_ino) != expected:
            raise OSError("publication directory changed during audit")
        for child, descriptor in children.items():
            if path.is_relative_to(child):
                opened = os.fstat(descriptor)
                current = child.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    raise OSError("artifact directory changed during publication")


@contextmanager
def pinned_publication(
    path: Path, expected: tuple[int, int] | None = None
) -> Iterator[None]:
    ensure_directory(path)
    descriptor = _open_directory(path)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OSError("another publisher owns the output directory") from exc
        opened = os.fstat(descriptor)
        if expected is not None and (opened.st_dev, opened.st_ino) != expected:
            raise OSError("publication directory changed before pinning")
        children: dict[Path, int] = {}
        token = _PINNED_ROOT.set((path, (opened.st_dev, opened.st_ino), children))
        try:
            yield
        finally:
            _PINNED_ROOT.reset(token)
            for child_descriptor in children.values():
                os.close(child_descriptor)
    finally:
        # Closing the retained descriptor also releases the publication lock.
        os.close(descriptor)


@contextmanager
def no_replace_publication() -> Iterator[None]:
    """Publish new artifact names atomically and fail if a destination appears."""

    token = _NO_REPLACE_PUBLICATION.set(True)
    try:
        yield
    finally:
        _NO_REPLACE_PUBLICATION.reset(token)


def pin_directory(path: Path) -> None:
    """Retain a child identity until the enclosing publication context exits."""
    pinned = _PINNED_ROOT.get()
    if pinned is None or not path.is_relative_to(pinned[0]):
        raise OSError("child directory requires an enclosing publication root")
    children = pinned[2]
    if path not in children:
        children[path] = _open_directory(path)
    _check_pinned_root(path)


def directory_identity(path: Path) -> tuple[int, int]:
    _check_pinned_root(path)
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"publication path is not a directory: {path}")
    return metadata.st_dev, metadata.st_ino


class RetainedArtifact:
    """A regular, single-link artifact held open through a publication boundary."""

    def __init__(self, path: Path, directory: int, descriptor: int) -> None:
        self._path = path
        self._directory = directory
        self._descriptor = descriptor

    def verify(self) -> None:
        """Require the path to still name this exact regular, single-link inode."""
        _check_directory(self._path.parent, self._directory)
        opened = os.fstat(self._descriptor)
        current = os.stat(
            self._path.name,
            dir_fd=self._directory,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise OSError(f"artifact changed during publication: {self._path}")

    def sha256(self) -> str:
        """Hash the retained inode without resolving the artifact path again."""
        self.verify()
        digest = hashlib.sha256()
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        while chunk := os.read(self._descriptor, 1024 * 1024):
            digest.update(chunk)
        self.verify()
        return digest.hexdigest()


@contextmanager
def retain_regular_artifact(path: Path) -> Iterator[RetainedArtifact]:
    """Open an artifact without following links and retain its exact identity."""
    if (
        not path.name
        or path.name in {".", ".."}
        or Path(path.name).name != path.name
        or "\x00" in path.name
    ):
        raise OSError("artifact name must be a single path component")
    directory = _open_directory(path.parent)
    descriptor: int | None = None
    try:
        _check_directory(path.parent, directory)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        retained = RetainedArtifact(path, directory, descriptor)
        retained.verify()
        yield retained
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _check_directory(path: Path, descriptor: int) -> None:
    _check_pinned_root(path)
    opened = os.fstat(descriptor)
    current = path.stat(follow_symlinks=False)
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise OSError(f"artifact directory changed during publication: {path}")


def _open_directory(path: Path) -> int:
    """Walk from the filesystem root without following any symlink component."""
    _check_pinned_root(path)
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            if component == "..":
                raise OSError("parent traversal is not allowed for artifact paths")
            created = False
            try:
                os.mkdir(component, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(descriptor)
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        _check_pinned_root(path)
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
            stream.flush()
            os.fsync(stream.fileno())
        _check_directory(path.parent, directory)
        if _NO_REPLACE_PUBLICATION.get():
            os.link(
                temporary,
                path.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
            os.unlink(temporary, dir_fd=directory)
            created = False
        else:
            os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
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


def clean_artifacts(
    path: Path,
    names: tuple[str, ...] | None = None,
    *,
    protected_identities: set[tuple[int, int]] | None = None,
) -> None:
    """Delete owned files relative to a pinned directory, never through symlinks."""
    if names is not None and any(
        not name or name in {".", ".."} or Path(name).name != name or "\x00" in name
        for name in names
    ):
        raise OSError("artifact cleanup names must be a single path component")
    directory = _open_directory(path)
    try:
        _check_directory(path, directory)
        _clean_entries(directory, names, protected_identities or set())
        _check_directory(path, directory)
    finally:
        os.close(directory)


def _clean_file(
    directory: int,
    name: str,
    protected_identities: set[tuple[int, int]],
) -> None:
    if not protected_identities:
        os.unlink(name, dir_fd=directory)
        return
    temporary = ".artifact-cleanup-" + secrets.token_hex(16) + ".tmp"
    try:
        os.rename(name, temporary, src_dir_fd=directory, dst_dir_fd=directory)
    except FileNotFoundError:
        return
    metadata = os.stat(temporary, dir_fd=directory, follow_symlinks=False)
    if (
        stat.S_ISREG(metadata.st_mode)
        and (
            metadata.st_dev,
            metadata.st_ino,
        )
        in protected_identities
    ):
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise OSError(
                f"protected artifact destination changed during cleanup: {name}"
            ) from exc
    os.unlink(temporary, dir_fd=directory)


def _clean_entries(
    directory: int,
    names: tuple[str, ...] | None,
    protected_identities: set[tuple[int, int]],
) -> None:
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
                _clean_entries(child, None, protected_identities)
            finally:
                os.close(child)
        elif names is None and not name.endswith(".parquet"):
            continue
        elif stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            _clean_file(directory, name, protected_identities)
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

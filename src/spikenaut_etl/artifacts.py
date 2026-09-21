"""Exclusive JSON and Parquet staging for audit and preparation artifacts."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            pq.write_table(table, stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

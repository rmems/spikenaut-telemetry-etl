"""Exclusive JSON staging for audit and preparation artifacts."""

import json
import tempfile
from pathlib import Path
from typing import Any


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

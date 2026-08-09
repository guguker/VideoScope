from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def atomic_write_text(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace ``path`` using a unique sibling temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    payload: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
            allow_nan=False,
        ),
    )

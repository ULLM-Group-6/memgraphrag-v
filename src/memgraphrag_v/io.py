"""JSON Lines I/O with per-line validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    """Write one record per line (LF endings on every OS); returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path, model: type[M]) -> list[M]:
    """Read and validate every non-empty line; errors name the file and line."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return records

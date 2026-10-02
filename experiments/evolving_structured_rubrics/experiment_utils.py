"""JSON and cache identity helpers for the current research method."""
from __future__ import annotations
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def _io_path(path: Path) -> Path:
    """Use an extended-length Windows path for persisted experiment artifacts."""
    if os.name != "nt":
        return path
    absolute = str(path.resolve())
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path = _io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep atomic sibling names short for legacy Windows MAX_PATH.
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:12]}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        for retry in range(20):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if retry == 19:
                    raise
                time.sleep(min(0.05 * (retry + 1), 0.5))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

"""Shared, model-agnostic utilities for structured-rubric experiments."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping

from critiq.structured.telemetry import ModelCallMetrics


def load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""

    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key:
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))


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


def request_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def load_jsonl_dataset(
    path: Path,
    *,
    expected_count: int | None = None,
    require_images: bool = True,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"dataset row {line_number} must be an object")
            for field in ("sample_id", "image_path", "question", "A", "B"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(
                        f"dataset row {line_number} requires non-empty {field}")
            if row.get("answer") not in {"A", "B"}:
                raise ValueError(f"dataset row {line_number} answer must be A/B")
            if require_images and not Path(row["image_path"]).is_file():
                raise ValueError(
                    f"dataset row {line_number} image missing: {row['image_path']}")
            rows.append(row)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"dataset must contain {expected_count} rows, got {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("dataset sample IDs must be unique")
    return tuple(rows)


def make_progress_callback(
    output: Path,
    stage: str,
    total: int,
    pool: Any | None = None,
):
    """Return a thread-safe progress callback used by online stages."""
    started = time.perf_counter()
    completed = 0
    lock = threading.Lock()
    log_path = output / f"logs/{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    initial_line = f"{stage}: 0/{total} started"
    print(initial_line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(initial_line + "\n")
    atomic_write_json(output / "progress.json", {
        "stage": stage,
        "completed": 0,
        "total": total,
        "percent": 0.0,
        "current_sample_id": None,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "rolling_mean_seconds": None,
        "current_sample_api_attempts": 0,
        "current_sample_errors": 0,
        "pool_api_attempts": 0,
        "pool_errors": 0,
        "pool_usage_complete": True,
        "endpoint_call_counts": {},
    })

    def callback(index: int, sample_id: str, metrics: ModelCallMetrics) -> None:
        nonlocal completed
        del index
        with lock:
            completed += 1
            elapsed = time.perf_counter() - started
            mean_seconds = elapsed / completed
            eta = mean_seconds * (total - completed)
            line = (
                f"{stage}: {completed}/{total} sample={sample_id} "
                f"api_attempts={metrics.api_attempts} errors={metrics.error_count} "
                f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            records = pool.records if pool is not None else ()
            try:
                atomic_write_json(
                    output / "progress.json",
                    {
                        "stage": stage,
                        "completed": completed,
                        "total": total,
                        "percent": completed / total * 100,
                        "current_sample_id": sample_id,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                        "rolling_mean_seconds": mean_seconds,
                        "current_sample_api_attempts": metrics.api_attempts,
                        "current_sample_errors": metrics.error_count,
                        "pool_api_attempts": sum(item.api_attempts for item in records),
                        "pool_errors": sum(item.error_count for item in records),
                        "pool_usage_complete": all(
                            item.usage_complete for item in records),
                        "endpoint_call_counts": (
                            dict(pool.records_by_endpoint())
                            if pool is not None
                            else {}
                        ),
                    },
                )
            except OSError as exc:
                warning = f"WARNING: progress.json update skipped: {exc}"
                print(warning, flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(warning + "\n")

    return callback


def validate_manifest_config(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if manifest.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError("config differs from frozen manifest")

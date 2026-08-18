"""Auditable JSON-directory prediction cache for structured execution."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .root_router import RootRouterOutput, RootRouterRequestSpec
from .telemetry import ModelCallMetrics
from .telemetry import combine_model_call_metrics
from .version import (
    STRUCTURED_CACHE_SCHEMA_VERSION,
    STRUCTURED_ROUTER_PARSER_VERSION,
    STRUCTURED_ROUTER_PROMPT_VERSION,
    STRUCTURED_ROUTER_SCHEMA_VERSION,
    STRUCTURED_SEMANTICS_VERSION,
    STRUCTURED_WORKER_PARSER_VERSION,
    STRUCTURED_WORKER_PROMPT_VERSION,
    STRUCTURED_WORKER_SCHEMA_VERSION,
)
from .worker_output import StructuredNodeOutput, StructuredWorkerRequestSpec
from .dual_worker import (
    DualWorkerRequestSpec,
    GateStateOutput,
    PairwiseVoteOutput,
)
from .version import (
    GATE_WORKER_PARSER_VERSION,
    GATE_WORKER_PROMPT_VERSION,
    GATE_WORKER_SCHEMA_VERSION,
    PAIRWISE_WORKER_PARSER_VERSION,
    PAIRWISE_WORKER_PROMPT_VERSION,
    PAIRWISE_WORKER_SCHEMA_VERSION,
)


class CacheMode(str, Enum):
    READ_WRITE = "read_write"
    REFRESH = "refresh"
    DISABLED = "disabled"


class CacheCorruptionError(ValueError):
    """Raised when a cache file exists but cannot be trusted."""


CACHE_FILENAME_HASH_HEX_LENGTH = 32


def _io_path(path: Path) -> Path:
    """Return a Windows extended-length path for filesystem operations.

    Experiment artifacts intentionally retain readable directory names, so a
    deeply nested cache file can exceed the legacy 260-character Windows
    limit even though its filename digest is compact.  Prefixing only the path
    used for I/O keeps the public/cache layout unchanged and avoids treating a
    valid parent directory as missing.
    """
    if os.name != "nt":
        return path
    absolute = str(path.resolve())
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("cache key payload must be JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def node_cache_key_payload(
    *,
    sample_fingerprint: str,
    criterion_name: str,
    criterion_description: str,
    request_spec: StructuredWorkerRequestSpec,
) -> dict[str, object]:
    return {
        "kind": "node",
        "cache_schema_version": STRUCTURED_CACHE_SCHEMA_VERSION,
        "semantics_version": STRUCTURED_SEMANTICS_VERSION,
        "worker_schema_version": STRUCTURED_WORKER_SCHEMA_VERSION,
        "worker_prompt_version": STRUCTURED_WORKER_PROMPT_VERSION,
        "worker_parser_version": STRUCTURED_WORKER_PARSER_VERSION,
        "sample_fingerprint": sample_fingerprint,
        "criterion_name": criterion_name,
        "criterion_description": criterion_description,
        "request_spec": request_spec.to_dict(),
    }


def router_cache_key_payload(
    *,
    sample_fingerprint: str,
    request_spec: RootRouterRequestSpec,
) -> dict[str, object]:
    return {
        "kind": "router",
        "cache_schema_version": STRUCTURED_CACHE_SCHEMA_VERSION,
        "semantics_version": STRUCTURED_SEMANTICS_VERSION,
        "router_schema_version": STRUCTURED_ROUTER_SCHEMA_VERSION,
        "router_prompt_version": STRUCTURED_ROUTER_PROMPT_VERSION,
        "router_parser_version": STRUCTURED_ROUTER_PARSER_VERSION,
        "sample_fingerprint": sample_fingerprint,
        "request_spec": request_spec.to_dict(),
    }


def pairwise_cache_key_payload(*, sample_fingerprint: str, criterion_name: str,
                               criterion_description: str,
                               request_spec: DualWorkerRequestSpec) -> dict[str, object]:
    return {
        "kind": "pairwise",
        "cache_schema_version": STRUCTURED_CACHE_SCHEMA_VERSION,
        "semantics_version": STRUCTURED_SEMANTICS_VERSION,
        "worker_schema_version": PAIRWISE_WORKER_SCHEMA_VERSION,
        "worker_prompt_version": PAIRWISE_WORKER_PROMPT_VERSION,
        "worker_parser_version": PAIRWISE_WORKER_PARSER_VERSION,
        "sample_fingerprint": sample_fingerprint,
        "criterion_name": criterion_name,
        "criterion_description": criterion_description,
        "request_spec": request_spec.to_dict(),
    }


def gate_cache_key_payload(*, sample_fingerprint: str, criterion_name: str,
                           criterion_description: str,
                           request_spec: DualWorkerRequestSpec) -> dict[str, object]:
    return {
        "kind": "gate",
        "cache_schema_version": STRUCTURED_CACHE_SCHEMA_VERSION,
        "semantics_version": STRUCTURED_SEMANTICS_VERSION,
        "worker_schema_version": GATE_WORKER_SCHEMA_VERSION,
        "worker_prompt_version": GATE_WORKER_PROMPT_VERSION,
        "worker_parser_version": GATE_WORKER_PARSER_VERSION,
        "sample_fingerprint": sample_fingerprint,
        "criterion_name": criterion_name,
        "criterion_description": criterion_description,
        "request_spec": request_spec.to_dict(),
    }


@dataclass(frozen=True)
class CachedNodeValue:
    output: StructuredNodeOutput
    generation_metrics: ModelCallMetrics


@dataclass(frozen=True)
class CachedRouterValue:
    output: RootRouterOutput
    generation_metrics: ModelCallMetrics


@dataclass(frozen=True)
class CachedPairwiseValue:
    output: PairwiseVoteOutput
    generation_metrics: ModelCallMetrics


@dataclass(frozen=True)
class CachedGateValue:
    output: GateStateOutput
    generation_metrics: ModelCallMetrics


class JsonPredictionCache:
    """One canonical JSON file per cache key, with atomic same-directory writes."""

    def __init__(self, root: str | Path, mode: CacheMode = CacheMode.READ_WRITE) -> None:
        if not isinstance(mode, CacheMode):
            raise TypeError("mode must be CacheMode")
        self.root = Path(root)
        self.mode = mode
        self._guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        if mode is not CacheMode.DISABLED:
            for kind in ("node", "router", "pairwise", "gate"):
                (self.root / kind).mkdir(parents=True, exist_ok=True)

    def lock_for(self, kind: str, key_sha256: str) -> threading.Lock:
        storage_key = self._storage_key(key_sha256)
        with self._guard:
            return self._locks.setdefault((kind, storage_key), threading.Lock())

    @staticmethod
    def _storage_key(key_sha256: str) -> str:
        """Return the bounded filesystem identifier for a full cache digest.

        The complete SHA-256 remains inside every cache entry and is validated
        on read.  A 128-bit filename keeps deeply nested experiment paths below
        the classic Windows MAX_PATH boundary without changing cache identity.
        Prefix collisions are serialized by ``lock_for`` and rejected rather
        than being treated as cache hits.
        """
        if len(key_sha256) != 64:
            raise ValueError("cache key SHA-256 must contain 64 hex characters")
        try:
            int(key_sha256, 16)
        except ValueError as exc:
            raise ValueError("cache key SHA-256 must be hexadecimal") from exc
        return key_sha256[:CACHE_FILENAME_HASH_HEX_LENGTH]

    def _path(self, kind: str, key_sha256: str) -> Path:
        if kind not in {"node", "router", "pairwise", "gate"}:
            raise ValueError("unsupported cache kind")
        return self.root / kind / f"{self._storage_key(key_sha256)}.json"

    def _legacy_path(self, kind: str, key_sha256: str) -> Path:
        """Return the pre-compact-name path for backward-compatible reads."""
        if kind not in {"node", "router", "pairwise", "gate"}:
            raise ValueError("unsupported cache kind")
        return self.root / kind / f"{key_sha256}.json"

    def _existing_path(self, kind: str, key_sha256: str) -> Path | None:
        path = self._path(kind, key_sha256)
        if _io_path(path).exists():
            return path
        legacy = self._legacy_path(kind, key_sha256)
        return legacy if _io_path(legacy).exists() else None

    def _read(self, kind: str, key_payload: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.mode in {CacheMode.DISABLED, CacheMode.REFRESH}:
            return None
        key_payload = dict(key_payload)
        key_sha256 = canonical_sha256(key_payload)
        path = self._existing_path(kind, key_sha256)
        if path is None:
            return None
        try:
            value = json.loads(_io_path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheCorruptionError(f"failed to read cache entry {path}: {exc}") from exc
        expected = {
            "cache_schema_version", "kind", "key_sha256", "key_payload",
            "output", "generation_metrics",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise CacheCorruptionError("cache entry fields are invalid")
        if value["cache_schema_version"] != STRUCTURED_CACHE_SCHEMA_VERSION:
            raise CacheCorruptionError("cache schema version mismatch")
        if value["kind"] != kind or value["key_sha256"] != key_sha256:
            raise CacheCorruptionError("cache identity mismatch")
        if value["key_payload"] != key_payload:
            raise CacheCorruptionError("cache key payload mismatch")
        return value

    def _write(
        self,
        kind: str,
        key_payload: Mapping[str, Any],
        output: Mapping[str, Any],
        generation_metrics: ModelCallMetrics,
    ) -> None:
        if self.mode is CacheMode.DISABLED:
            return
        key_payload = dict(key_payload)
        key_sha256 = canonical_sha256(key_payload)
        path = self._path(kind, key_sha256)
        io_path = _io_path(path)
        io_path.parent.mkdir(parents=True, exist_ok=True)
        if io_path.exists():
            try:
                existing = json.loads(io_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if self.mode is not CacheMode.REFRESH:
                    raise CacheCorruptionError(
                        f"failed to inspect existing cache entry {path}: {exc}"
                    ) from exc
            else:
                existing_key = existing.get("key_sha256") if isinstance(existing, dict) else None
                if existing_key != key_sha256:
                    raise CacheCorruptionError(
                        "compact cache filename collision: stored full SHA-256 differs"
                    )
        payload = {
            "cache_schema_version": STRUCTURED_CACHE_SCHEMA_VERSION,
            "kind": kind,
            "key_sha256": key_sha256,
            "key_payload": key_payload,
            "output": dict(output),
            "generation_metrics": generation_metrics.to_dict(),
        }
        # Both the durable and temporary names are bounded.  The full digest is
        # stored in the payload above and remains the authoritative identity.
        temporary = io_path.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            for retry in range(20):
                try:
                    os.replace(temporary, io_path)
                    break
                except PermissionError:
                    if retry == 19:
                        raise
                    time.sleep(min(.05 * (retry + 1), .5))
        finally:
            if temporary.exists():
                temporary.unlink()

    def get_node(self, key_payload: Mapping[str, Any]) -> CachedNodeValue | None:
        value = self._read("node", key_payload)
        if value is None:
            return None
        try:
            return CachedNodeValue(
                output=StructuredNodeOutput.from_dict(value["output"]),
                generation_metrics=ModelCallMetrics.from_dict(value["generation_metrics"]),
            )
        except (TypeError, ValueError) as exc:
            raise CacheCorruptionError(f"invalid cached node value: {exc}") from exc

    def put_node(
        self,
        key_payload: Mapping[str, Any],
        output: StructuredNodeOutput,
        generation_metrics: ModelCallMetrics,
    ) -> None:
        self._write("node", key_payload, output.to_dict(), generation_metrics)

    def get_router(self, key_payload: Mapping[str, Any]) -> CachedRouterValue | None:
        value = self._read("router", key_payload)
        if value is None:
            return None
        try:
            return CachedRouterValue(
                output=RootRouterOutput.from_dict(value["output"]),
                generation_metrics=ModelCallMetrics.from_dict(value["generation_metrics"]),
            )
        except (TypeError, ValueError) as exc:
            raise CacheCorruptionError(f"invalid cached router value: {exc}") from exc

    def put_router(
        self,
        key_payload: Mapping[str, Any],
        output: RootRouterOutput,
        generation_metrics: ModelCallMetrics,
    ) -> None:
        self._write("router", key_payload, output.to_dict(), generation_metrics)

    def get_pairwise(self, key_payload: Mapping[str, Any]) -> CachedPairwiseValue | None:
        value = self._read("pairwise", key_payload)
        if value is None:
            return None
        try:
            return CachedPairwiseValue(PairwiseVoteOutput.from_dict(value["output"]),
                                       ModelCallMetrics.from_dict(value["generation_metrics"]))
        except (TypeError, ValueError) as exc:
            raise CacheCorruptionError(f"invalid cached pairwise value: {exc}") from exc

    def put_pairwise(self, key_payload: Mapping[str, Any], output: PairwiseVoteOutput,
                     generation_metrics: ModelCallMetrics) -> None:
        self._write("pairwise", key_payload, output.to_dict(), generation_metrics)

    def get_gate(self, key_payload: Mapping[str, Any]) -> CachedGateValue | None:
        value = self._read("gate", key_payload)
        if value is None:
            return None
        try:
            return CachedGateValue(GateStateOutput.from_dict(value["output"]),
                                   ModelCallMetrics.from_dict(value["generation_metrics"]))
        except (TypeError, ValueError) as exc:
            raise CacheCorruptionError(f"invalid cached gate value: {exc}") from exc

    def put_gate(self, key_payload: Mapping[str, Any], output: GateStateOutput,
                 generation_metrics: ModelCallMetrics) -> None:
        self._write("gate", key_payload, output.to_dict(), generation_metrics)

    def quarantine(self, kind: str, key_payload: Mapping[str, Any]) -> Path:
        """Move one validated cache entry aside so it can be regenerated.

        Quarantine is intentionally recoverable: the previous entry is kept
        outside the active kind directory instead of being deleted.
        """
        if self.mode is CacheMode.DISABLED:
            raise ValueError("cannot quarantine an entry from a disabled cache")
        if kind not in {"node", "router", "pairwise", "gate"}:
            raise ValueError("unsupported cache kind")
        payload = dict(key_payload)
        key_sha256 = canonical_sha256(payload)
        # Validate the entry before moving it. A missing or corrupt cache must
        # not be silently treated as the requested repair target.
        value = self._read(kind, payload)
        if value is None:
            raise FileNotFoundError(f"cache entry does not exist: {kind}/{key_sha256}")
        source = self._existing_path(kind, key_sha256)
        if source is None:
            raise FileNotFoundError(f"cache entry does not exist: {kind}/{key_sha256}")
        directory = self.root / "quarantine" / kind
        directory.mkdir(parents=True, exist_ok=True)
        # The entry itself carries the full digest, so a UUID-only quarantine
        # filename is sufficient and avoids recreating the MAX_PATH problem.
        target = directory / f"{uuid.uuid4().hex}.json"
        for retry in range(20):
            try:
                os.replace(source, target)
                return target
            except PermissionError:
                if retry == 19:
                    raise
                time.sleep(min(.05 * (retry + 1), .5))
        raise RuntimeError("unreachable cache quarantine state")

    def generation_provenance(self, kind: str) -> tuple[ModelCallMetrics, ...]:
        """Return every unique completed generation stored in one namespace.

        This is intentionally independent of sample traces.  It lets a resumed
        cold run count calls that reached the cache before the process stopped
        but whose enclosing sample trace had not yet been committed.
        """
        if kind not in {"node", "router", "pairwise", "gate"}:
            raise ValueError("unsupported cache kind")
        directory = self.root / kind
        if not directory.exists():
            return ()
        metrics_by_key: dict[str, ModelCallMetrics] = {}
        payloads_by_key: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CacheCorruptionError(f"failed to read cache provenance {path}: {exc}") from exc
            expected = {"cache_schema_version", "kind", "key_sha256", "key_payload",
                        "output", "generation_metrics"}
            if not isinstance(value, dict) or set(value) != expected:
                raise CacheCorruptionError(f"cache provenance fields are invalid: {path}")
            if value["cache_schema_version"] != STRUCTURED_CACHE_SCHEMA_VERSION or value["kind"] != kind:
                raise CacheCorruptionError(f"cache provenance version/kind mismatch: {path}")
            expected_key = canonical_sha256(value["key_payload"])
            valid_stems = {expected_key, self._storage_key(expected_key)}
            if value["key_sha256"] != expected_key or path.stem not in valid_stems:
                raise CacheCorruptionError(f"cache provenance identity mismatch: {path}")
            try:
                metric = ModelCallMetrics.from_dict(value["generation_metrics"])
            except (TypeError, ValueError) as exc:
                raise CacheCorruptionError(f"cache provenance metrics are invalid: {path}") from exc
            if metric.cache_hits or metric.cache_misses:
                raise CacheCorruptionError(f"generation provenance contains cache counts: {path}")
            previous = payloads_by_key.get(expected_key)
            if previous is not None and previous != value:
                raise CacheCorruptionError(
                    f"conflicting compact and legacy cache entries: {expected_key}"
                )
            payloads_by_key[expected_key] = value
            metrics_by_key[expected_key] = metric
        return tuple(metrics_by_key[key] for key in sorted(metrics_by_key))

    def cumulative_generation_metrics(self, kinds: tuple[str, ...]) -> ModelCallMetrics:
        """Combine unique cache-entry generation provenance across kinds."""
        return combine_model_call_metrics(
            metric for kind in kinds for metric in self.generation_provenance(kind)
        )

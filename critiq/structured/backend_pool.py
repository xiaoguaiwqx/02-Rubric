"""Thread-safe pool for equivalent OpenAI-compatible inference endpoints."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from critiq.agent import Agent, AgentCallMetrics


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BackendEndpointSpec:
    endpoint_id: str
    base_url: str
    checkpoint_root: str
    max_concurrency: int

    def __post_init__(self) -> None:
        for name in ("endpoint_id", "base_url", "checkpoint_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (isinstance(self.max_concurrency, bool)
                or not isinstance(self.max_concurrency, int)
                or self.max_concurrency < 1):
            raise ValueError("max_concurrency must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {"endpoint_id": self.endpoint_id, "base_url": self.base_url,
                "checkpoint_root": self.checkpoint_root,
                "max_concurrency": self.max_concurrency}


@dataclass(frozen=True)
class BackendPoolSpec:
    pool_id: str
    common_checkpoint_id: str
    global_request_concurrency: int
    endpoints: tuple[BackendEndpointSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pool_id, str) or not self.pool_id.strip():
            raise ValueError("pool_id must be a non-empty string")
        if not isinstance(self.common_checkpoint_id, str) or not self.common_checkpoint_id.strip():
            raise ValueError("common_checkpoint_id must be a non-empty string")
        if (isinstance(self.global_request_concurrency, bool)
                or not isinstance(self.global_request_concurrency, int)
                or self.global_request_concurrency < 1):
            raise ValueError("global_request_concurrency must be a positive integer")
        endpoints = tuple(self.endpoints)
        if not endpoints or any(not isinstance(item, BackendEndpointSpec) for item in endpoints):
            raise ValueError("backend pool requires at least one endpoint spec")
        endpoint_ids = [item.endpoint_id for item in endpoints]
        base_urls = [item.base_url for item in endpoints]
        if len(set(endpoint_ids)) != len(endpoint_ids) or len(set(base_urls)) != len(base_urls):
            raise ValueError("endpoint IDs and base URLs must be unique")
        object.__setattr__(self, "endpoints", endpoints)

    @classmethod
    def from_dict(cls, value: object) -> "BackendPoolSpec":
        if not isinstance(value, dict) or set(value) != {
            "pool_id", "common_checkpoint_id", "global_request_concurrency", "endpoints"
        } or not isinstance(value["endpoints"], list):
            raise ValueError("backend_pool fields are invalid")
        return cls(value["pool_id"], value["common_checkpoint_id"],
                   value["global_request_concurrency"],
                   tuple(BackendEndpointSpec(**item) for item in value["endpoints"]))

    def to_dict(self) -> dict[str, Any]:
        return {"pool_id": self.pool_id,
                "common_checkpoint_id": self.common_checkpoint_id,
                "global_request_concurrency": self.global_request_concurrency,
                "endpoints": [item.to_dict() for item in self.endpoints]}

    @property
    def backend_id(self) -> str:
        identity = {"pool_id": self.pool_id,
                    "common_checkpoint_id": self.common_checkpoint_id,
                    "endpoints": [item.to_dict() for item in self.endpoints]}
        return f"pool:{self.pool_id}:{_canonical_sha256(identity)}"


@dataclass(frozen=True)
class PoolCallRecord:
    sequence: int
    request_type: str
    request_key: str
    structured_attempt: int
    endpoint_id: str
    api_attempts: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_complete: bool
    latency_seconds: float
    error_count: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class AvailableSlotBackendPool:
    """Assign each call to the first available endpoint capacity slot."""

    def __init__(self, spec: BackendPoolSpec) -> None:
        if not isinstance(spec, BackendPoolSpec):
            raise TypeError("spec must be BackendPoolSpec")
        self.spec = spec
        self._endpoints = {item.endpoint_id: item for item in spec.endpoints}
        self._slots: queue.Queue[str] = queue.Queue()
        # Interleave initial slots so a small burst does not all land on the
        # first configured endpoint. Released slots return at completion time,
        # which naturally favors the endpoint that becomes available first.
        for slot_index in range(max(item.max_concurrency for item in spec.endpoints)):
            for endpoint in spec.endpoints:
                if slot_index < endpoint.max_concurrency:
                    self._slots.put(endpoint.endpoint_id)
        self._global = threading.BoundedSemaphore(spec.global_request_concurrency)
        self._lock = threading.Lock()
        self._records: list[PoolCallRecord] = []

    @property
    def backend_id(self) -> str:
        return self.spec.backend_id

    @property
    def records(self) -> tuple[PoolCallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def records_by_endpoint(self) -> Mapping[str, int]:
        counts = {item.endpoint_id: 0 for item in self.spec.endpoints}
        with self._lock:
            for record in self._records:
                counts[record.endpoint_id] += 1
        return MappingProxyType(counts)

    def call(self, prompt: object, *, request_type: str, request_key: str,
             structured_attempt: int, agent_args: Mapping[str, Any]) -> tuple[str | None, AgentCallMetrics]:
        if not isinstance(request_type, str) or not request_type.strip():
            raise ValueError("request_type must be non-empty")
        if not isinstance(request_key, str) or not request_key.strip():
            raise ValueError("request_key must be non-empty")
        self._global.acquire()
        endpoint_id = self._slots.get()
        endpoint = self._endpoints[endpoint_id]
        try:
            kwargs = dict(agent_args)
            kwargs["base_url"] = endpoint.base_url
            agent = Agent(**kwargs)
            started = time.perf_counter()
            raw = agent(prompt, stream=False)
            metrics = agent.last_call_metrics
            elapsed = time.perf_counter() - started
            if not isinstance(metrics, AgentCallMetrics):
                metrics = AgentCallMetrics(api_attempts=1, input_tokens=None,
                    output_tokens=None, total_tokens=None, usage_complete=False,
                    latency_seconds=elapsed)
            with self._lock:
                self._records.append(PoolCallRecord(
                    sequence=len(self._records), request_type=request_type,
                    request_key=request_key, structured_attempt=structured_attempt,
                    endpoint_id=endpoint_id, api_attempts=metrics.api_attempts,
                    input_tokens=metrics.input_tokens, output_tokens=metrics.output_tokens,
                    total_tokens=metrics.total_tokens, usage_complete=metrics.usage_complete,
                    latency_seconds=metrics.latency_seconds,
                    error_count=metrics.error_count))
            return raw, metrics
        finally:
            self._slots.put(endpoint_id)
            self._global.release()

    def provenance_dict(self) -> dict[str, Any]:
        return {"backend_pool": self.spec.to_dict(), "backend_id": self.backend_id,
                "endpoint_call_counts": dict(self.records_by_endpoint()),
                "calls": [item.to_dict() for item in self.records]}

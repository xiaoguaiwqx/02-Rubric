"""Offline artifact and online lazy backends consumed by the executor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Protocol

from critiq.utils import PairData

from .cache import (
    CacheMode,
    JsonPredictionCache,
    canonical_sha256,
    node_cache_key_payload,
    router_cache_key_payload,
)
from .root_router import (
    RootRouterOutput,
    RootRoutingPredictionOutput,
    StructuredRootRouter,
    root_snapshots,
)
from .schema import RubricNode, StructuredRubric
from .telemetry import ModelCallMetrics
from .validation import assert_prediction_compatible
from .worker_output import (
    StructuredNodeOutput,
    StructuredPredictionOutput,
    structured_input_fingerprint,
)


class BackendSource(str, Enum):
    OFFLINE_ARTIFACT = "offline_artifact"
    MODEL = "model"
    CACHE = "cache"


@dataclass(frozen=True)
class NodeBackendResult:
    output: StructuredNodeOutput
    source: BackendSource
    metrics: ModelCallMetrics
    generation_metrics: ModelCallMetrics
    cache_key: str | None = None


@dataclass(frozen=True)
class RouterBackendResult:
    output: RootRouterOutput
    source: BackendSource
    metrics: ModelCallMetrics
    generation_metrics: ModelCallMetrics
    cache_key: str | None = None


class NodeBackend(Protocol):
    def sample_id(self, data: Mapping[str, Any]) -> str: ...
    def sample_fingerprint(self, data: Mapping[str, Any]) -> str: ...
    def evaluate(self, data: PairData, node: RubricNode) -> NodeBackendResult: ...


class RouterBackend(Protocol):
    def sample_id(self, data: Mapping[str, Any]) -> str: ...
    def sample_fingerprint(self, data: Mapping[str, Any]) -> str: ...
    def route(self, data: PairData, rubric: StructuredRubric) -> RouterBackendResult: ...


class OfflineNodeBackend:
    """Read all-node Phase 1 outputs without constructing an Agent."""

    def __init__(self, prediction: StructuredPredictionOutput, rubric: StructuredRubric) -> None:
        assert_prediction_compatible(rubric, prediction)
        self.prediction = prediction
        self._index = {sample_id: index for index, sample_id in enumerate(prediction.sample_ids)}

    def sample_id(self, data: Mapping[str, Any]) -> str:
        value = data.get(self.prediction.request_spec.sample_id_field)
        if not isinstance(value, str) or value not in self._index:
            raise KeyError("sample is not present in offline node artifact")
        return value

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        index = self._index[self.sample_id(data)]
        spec = self.prediction.request_spec
        actual = structured_input_fingerprint(
            data,
            image_field=spec.image_field,
            question_field=spec.question_field,
            sample_id_field=spec.sample_id_field,
            max_data_chars=spec.max_data_chars,
            encode_local_image=spec.encode_local_image,
        )
        if actual != self.prediction.sample_fingerprints[index]:
            raise ValueError("offline node artifact sample fingerprint mismatch")
        return actual

    def evaluate(self, data: PairData, node: RubricNode) -> NodeBackendResult:
        index = self._index[self.sample_id(data)]
        output = self.prediction.node_outputs[index][node.criterion.name]
        zero = ModelCallMetrics()
        return NodeBackendResult(
            output=output,
            source=BackendSource.OFFLINE_ARTIFACT,
            metrics=zero,
            generation_metrics=zero,
        )


class OnlineNodeBackend:
    """Call StructuredMultiModalPairEvaluator only on executor cache misses."""

    def __init__(self, evaluator: Any, cache: JsonPredictionCache | None = None) -> None:
        required = ("infer_one", "sample_fingerprint", "request_spec", "sample_id_field")
        if any(not hasattr(evaluator, name) for name in required):
            raise TypeError("evaluator does not implement structured single-node inference")
        self.evaluator = evaluator
        self.cache = cache

    def sample_id(self, data: Mapping[str, Any]) -> str:
        value = data.get(self.evaluator.sample_id_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("online node sample ID must be non-empty")
        return value

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        return self.evaluator.sample_fingerprint(data)

    def evaluate(self, data: PairData, node: RubricNode) -> NodeBackendResult:
        fingerprint = self.sample_fingerprint(data)
        payload = node_cache_key_payload(
            sample_fingerprint=fingerprint,
            criterion_name=node.criterion.name,
            criterion_description=node.criterion.description,
            request_spec=self.evaluator.request_spec(),
        )
        key = canonical_sha256(payload)
        cache_active = self.cache is not None and self.cache.mode is not CacheMode.DISABLED
        lock = self.cache.lock_for("node", key) if cache_active else _NullLock()
        with lock:
            cached = self.cache.get_node(payload) if cache_active else None
            if cached is not None:
                return NodeBackendResult(
                    output=cached.output,
                    source=BackendSource.CACHE,
                    metrics=ModelCallMetrics.from_agent_calls((), cache_hit=True),
                    generation_metrics=cached.generation_metrics,
                    cache_key=key,
                )
            output, generation_metrics = self.evaluator.infer_one(
                data,
                node.criterion.to_criterion(),
            )
            current_metrics = replace(
                generation_metrics,
                cache_misses=(1 if cache_active else 0),
            )
            if cache_active:
                self.cache.put_node(payload, output, generation_metrics)
            return NodeBackendResult(
                output=output,
                source=BackendSource.MODEL,
                metrics=current_metrics,
                generation_metrics=generation_metrics,
                cache_key=key,
            )


class OfflineRouterBackend:
    def __init__(self, prediction: RootRoutingPredictionOutput, rubric: StructuredRubric) -> None:
        if prediction.rubric_sha256 != rubric.rubric_sha256:
            raise ValueError("offline router artifact rubric mismatch")
        if prediction.roots != root_snapshots(rubric):
            raise ValueError("offline router artifact root snapshots mismatch")
        self.prediction = prediction
        self._index = {sample_id: index for index, sample_id in enumerate(prediction.sample_ids)}

    def sample_id(self, data: Mapping[str, Any]) -> str:
        value = data.get(self.prediction.request_spec.sample_id_field)
        if not isinstance(value, str) or value not in self._index:
            raise KeyError("sample is not present in offline router artifact")
        return value

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        index = self._index[self.sample_id(data)]
        spec = self.prediction.request_spec
        actual = structured_input_fingerprint(
            data,
            image_field=spec.image_field,
            question_field=spec.question_field,
            sample_id_field=spec.sample_id_field,
            max_data_chars=spec.max_data_chars,
            encode_local_image=spec.encode_local_image,
        )
        if actual != self.prediction.sample_fingerprints[index]:
            raise ValueError("offline router artifact sample fingerprint mismatch")
        return actual

    def route(self, data: PairData, rubric: StructuredRubric) -> RouterBackendResult:
        if rubric.rubric_sha256 != self.prediction.rubric_sha256:
            raise ValueError("offline router artifact rubric mismatch")
        sample_id = self.sample_id(data)
        self.sample_fingerprint(data)
        index = self._index[sample_id]
        output = self.prediction.outputs[index]
        return RouterBackendResult(
            output=output,
            source=BackendSource.OFFLINE_ARTIFACT,
            metrics=ModelCallMetrics(),
            generation_metrics=self.prediction.generation_metrics[index],
        )


class OnlineRouterBackend:
    def __init__(self, router: StructuredRootRouter, cache: JsonPredictionCache | None = None) -> None:
        if not isinstance(router, StructuredRootRouter):
            raise TypeError("router must be StructuredRootRouter")
        self.router = router
        self.cache = cache

    def sample_id(self, data: Mapping[str, Any]) -> str:
        value = data.get(self.router.sample_id_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("online router sample ID must be non-empty")
        return value

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        return self.router.sample_fingerprint(data)

    def route(self, data: PairData, rubric: StructuredRubric) -> RouterBackendResult:
        fingerprint = self.sample_fingerprint(data)
        payload = router_cache_key_payload(
            sample_fingerprint=fingerprint,
            request_spec=self.router.request_spec(rubric),
        )
        key = canonical_sha256(payload)
        cache_active = self.cache is not None and self.cache.mode is not CacheMode.DISABLED
        lock = self.cache.lock_for("router", key) if cache_active else _NullLock()
        with lock:
            cached = self.cache.get_router(payload) if cache_active else None
            if cached is not None:
                return RouterBackendResult(
                    output=cached.output,
                    source=BackendSource.CACHE,
                    metrics=ModelCallMetrics.from_agent_calls((), cache_hit=True),
                    generation_metrics=cached.generation_metrics,
                    cache_key=key,
                )
            output, generation_metrics = self.router.route_one_uncached(data, rubric)
            current_metrics = replace(
                generation_metrics,
                cache_misses=(1 if cache_active else 0),
            )
            if cache_active:
                self.cache.put_router(payload, output, generation_metrics)
            return RouterBackendResult(
                output=output,
                source=BackendSource.MODEL,
                metrics=current_metrics,
                generation_metrics=generation_metrics,
                cache_key=key,
            )


class _NullLock:
    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

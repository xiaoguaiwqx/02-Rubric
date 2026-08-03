"""Offline and online backends for the decoupled worker channels."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from critiq.utils import PairData

from .backend import BackendSource
from .cache import (
    CacheMode,
    JsonPredictionCache,
    canonical_sha256,
    gate_cache_key_payload,
    pairwise_cache_key_payload,
)
from .dual_worker import (
    GatePredictionOutput,
    GateStateOutput,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
)
from .schema import RubricNode, StructuredRubric
from .telemetry import ModelCallMetrics
from .worker_output import structured_input_fingerprint


@dataclass(frozen=True)
class PairwiseBackendResult:
    output: PairwiseVoteOutput
    source: BackendSource
    metrics: ModelCallMetrics
    generation_metrics: ModelCallMetrics
    cache_key: str | None = None


@dataclass(frozen=True)
class GateBackendResult:
    output: GateStateOutput
    source: BackendSource
    metrics: ModelCallMetrics
    generation_metrics: ModelCallMetrics
    cache_key: str | None = None


class PairwiseVoteBackend(Protocol):
    def sample_id(self, data: Mapping[str, Any]) -> str: ...
    def sample_fingerprint(self, data: Mapping[str, Any]) -> str: ...
    def evaluate(self, data: PairData, node: RubricNode) -> PairwiseBackendResult: ...


class GateStateBackend(Protocol):
    def sample_id(self, data: Mapping[str, Any]) -> str: ...
    def sample_fingerprint(self, data: Mapping[str, Any]) -> str: ...
    def evaluate(self, data: PairData, node: RubricNode) -> GateBackendResult: ...


def _artifact_fingerprint(prediction: Any, data: Mapping[str, Any], index: int) -> str:
    spec = prediction.request_spec
    actual = structured_input_fingerprint(data, image_field=spec.image_field,
        question_field=spec.question_field, sample_id_field=spec.sample_id_field,
        max_data_chars=spec.max_data_chars, encode_local_image=spec.encode_local_image)
    if actual != prediction.sample_fingerprints[index]:
        raise ValueError("offline dual-worker artifact sample fingerprint mismatch")
    return actual


class OfflinePairwiseVoteBackend:
    def __init__(self, prediction: PairwisePredictionOutput, rubric: StructuredRubric) -> None:
        expected = {node.criterion.name: node.criterion.description for node in rubric.nodes.values()}
        actual = {item.name: item.description for item in prediction.criteria}
        if actual != expected:
            raise ValueError("pairwise artifact criteria do not match rubric")
        self.prediction = prediction
        self._index = {sample_id: index for index, sample_id in enumerate(prediction.sample_ids)}

    def sample_id(self, data: Mapping[str, Any]) -> str:
        value = data.get(self.prediction.request_spec.sample_id_field)
        if not isinstance(value, str) or value not in self._index:
            raise KeyError("sample is not present in pairwise artifact")
        return value

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        return _artifact_fingerprint(self.prediction, data, self._index[self.sample_id(data)])

    def evaluate(self, data: PairData, node: RubricNode) -> PairwiseBackendResult:
        index = self._index[self.sample_id(data)]; self.sample_fingerprint(data)
        return PairwiseBackendResult(self.prediction.node_outputs[index][node.criterion.name],
            BackendSource.OFFLINE_ARTIFACT, ModelCallMetrics(), ModelCallMetrics())


class OfflineGateStateBackend:
    def __init__(self, prediction: GatePredictionOutput, rubric: StructuredRubric) -> None:
        rubric_criteria = {node.criterion.name: node.criterion.description for node in rubric.nodes.values()}
        for item in prediction.criteria:
            if rubric_criteria.get(item.name) != item.description:
                raise ValueError("gate artifact criteria do not match rubric")
        self.prediction = prediction
        self._index = {sample_id: index for index, sample_id in enumerate(prediction.sample_ids)}

    def sample_id(self, data: Mapping[str, Any]) -> str:
        value = data.get(self.prediction.request_spec.sample_id_field)
        if not isinstance(value, str) or value not in self._index:
            raise KeyError("sample is not present in gate artifact")
        return value

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        return _artifact_fingerprint(self.prediction, data, self._index[self.sample_id(data)])

    def evaluate(self, data: PairData, node: RubricNode) -> GateBackendResult:
        index = self._index[self.sample_id(data)]; self.sample_fingerprint(data)
        try:
            output = self.prediction.node_outputs[index][node.criterion.name]
        except KeyError as exc:
            raise KeyError(f"gate artifact lacks criterion {node.criterion.name!r}") from exc
        return GateBackendResult(output, BackendSource.OFFLINE_ARTIFACT,
                                 ModelCallMetrics(), ModelCallMetrics())


class _OnlineDualBackend:
    cache_kind: str

    def __init__(self, evaluator: Any, cache: JsonPredictionCache | None = None) -> None:
        if any(not hasattr(evaluator, name) for name in ("infer_one", "sample_fingerprint", "request_spec", "sample_id_field")):
            raise TypeError("evaluator does not implement dual-worker single-node inference")
        self.evaluator = evaluator; self.cache = cache

    def sample_id(self, data: Mapping[str, Any]) -> str:
        value = data.get(self.evaluator.sample_id_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("dual-worker sample ID must be non-empty")
        return value

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        return self.evaluator.sample_fingerprint(data)


class OnlinePairwiseVoteBackend(_OnlineDualBackend):
    cache_kind = "pairwise"

    def evaluate(self, data: PairData, node: RubricNode) -> PairwiseBackendResult:
        payload = pairwise_cache_key_payload(sample_fingerprint=self.sample_fingerprint(data),
            criterion_name=node.criterion.name, criterion_description=node.criterion.description,
            request_spec=self.evaluator.request_spec())
        key = canonical_sha256(payload); active = self.cache is not None and self.cache.mode is not CacheMode.DISABLED
        lock = self.cache.lock_for("pairwise", key) if active else _NullLock()
        with lock:
            cached = self.cache.get_pairwise(payload) if active else None
            if cached is not None:
                return PairwiseBackendResult(cached.output, BackendSource.CACHE,
                    ModelCallMetrics.from_agent_calls((), cache_hit=True), cached.generation_metrics, key)
            output, generation = self.evaluator.infer_one(data, node.criterion.to_criterion())
            current = replace(generation, cache_misses=1 if active else 0)
            if active: self.cache.put_pairwise(payload, output, generation)
            return PairwiseBackendResult(output, BackendSource.MODEL, current, generation, key)


class OnlineGateStateBackend(_OnlineDualBackend):
    cache_kind = "gate"

    def evaluate(self, data: PairData, node: RubricNode) -> GateBackendResult:
        payload = gate_cache_key_payload(sample_fingerprint=self.sample_fingerprint(data),
            criterion_name=node.criterion.name, criterion_description=node.criterion.description,
            request_spec=self.evaluator.request_spec())
        key = canonical_sha256(payload); active = self.cache is not None and self.cache.mode is not CacheMode.DISABLED
        lock = self.cache.lock_for("gate", key) if active else _NullLock()
        with lock:
            cached = self.cache.get_gate(payload) if active else None
            if cached is not None:
                return GateBackendResult(cached.output, BackendSource.CACHE,
                    ModelCallMetrics.from_agent_calls((), cache_hit=True), cached.generation_metrics, key)
            output, generation = self.evaluator.infer_one(data, node.criterion.to_criterion())
            current = replace(generation, cache_misses=1 if active else 0)
            if active: self.cache.put_gate(payload, output, generation)
            return GateBackendResult(output, BackendSource.MODEL, current, generation, key)


class _NullLock:
    def __enter__(self) -> "_NullLock": return self
    def __exit__(self, *_: object) -> None: return None

"""Versioned, replayable execution traces for structured cascades."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .backend import BackendSource
from .judgement import FinalPreference, OutcomeReason, Vote
from .root_router import RootRouterOutput
from .semantics import (
    ResolvedRootRouting,
    RoutingOutcomeReason,
    RoutingSource,
    TraversalPolicy,
)
from .telemetry import ModelCallMetrics, combine_model_call_metrics
from .version import (
    STRUCTURED_EXECUTION_TRACE_VERSION,
    STRUCTURED_RUBRIC_SCHEMA_VERSION,
    STRUCTURED_SEMANTICS_VERSION,
)
from .worker_output import StructuredNodeOutput


class AggregationMode(str, Enum):
    FLAT = "flat"
    HIERARCHICAL = "hierarchical"


def _exact(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")
    return value


def _validate_backend_metrics(
    source: BackendSource,
    metrics: ModelCallMetrics,
    generation_metrics: ModelCallMetrics,
    *,
    label: str,
) -> None:
    """Validate current-run cost against backend source and provenance."""

    zero = ModelCallMetrics()
    if source is BackendSource.OFFLINE_ARTIFACT:
        if metrics != zero:
            raise ValueError(f"{label} offline metrics must be zero")
        return
    if source is BackendSource.CACHE:
        expected = ModelCallMetrics.from_agent_calls((), cache_hit=True)
        if metrics != expected:
            raise ValueError(f"{label} cache-hit metrics are inconsistent")
        return
    if source is BackendSource.MODEL:
        if generation_metrics.cache_hits or generation_metrics.cache_misses:
            raise ValueError(f"{label} generation provenance must exclude cache counts")
        if metrics.cache_misses not in (0, 1):
            raise ValueError(f"{label} model cache_misses must be zero or one")
        expected = replace(
            generation_metrics,
            cache_misses=metrics.cache_misses,
        )
        if metrics != expected:
            raise ValueError(f"{label} model metrics do not match generation provenance")
        return
    raise TypeError(f"{label} has an unsupported backend source")


@dataclass(frozen=True)
class EdgeExecutionTrace:
    parent_id: str
    child_id: str
    condition: str
    matched: bool
    child_visited: bool
    skip_reason: str | None
    child_subtree_vote: Vote | None

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "condition": self.condition,
            "matched": self.matched,
            "child_visited": self.child_visited,
            "skip_reason": self.skip_reason,
            "child_subtree_vote": None if self.child_subtree_vote is None else self.child_subtree_vote.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EdgeExecutionTrace":
        value = _exact(value, {"parent_id", "child_id", "condition", "matched", "child_visited", "skip_reason", "child_subtree_vote"}, "edge trace")
        vote = value["child_subtree_vote"]
        return cls(
            parent_id=value["parent_id"], child_id=value["child_id"], condition=value["condition"],
            matched=value["matched"], child_visited=value["child_visited"], skip_reason=value["skip_reason"],
            child_subtree_vote=None if vote is None else Vote(vote),
        )


@dataclass(frozen=True)
class NodeExecutionTrace:
    node_id: str
    criterion_name: str
    depth: int
    output: StructuredNodeOutput
    local_vote: Vote
    outcome_reason: OutcomeReason
    backend_source: BackendSource
    metrics: ModelCallMetrics
    generation_metrics: ModelCallMetrics
    cache_key: str | None
    edges: tuple[EdgeExecutionTrace, ...]
    subtree_vote: Vote | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))
        _validate_backend_metrics(
            self.backend_source,
            self.metrics,
            self.generation_metrics,
            label=f"node {self.node_id!r}",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "criterion_name": self.criterion_name,
            "depth": self.depth,
            "output": self.output.to_dict(),
            "local_vote": self.local_vote.value,
            "outcome_reason": self.outcome_reason.value,
            "backend_source": self.backend_source.value,
            "metrics": self.metrics.to_dict(),
            "generation_metrics": self.generation_metrics.to_dict(),
            "cache_key": self.cache_key,
            "edges": [edge.to_dict() for edge in self.edges],
            "subtree_vote": None if self.subtree_vote is None else self.subtree_vote.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NodeExecutionTrace":
        value = _exact(value, {"node_id", "criterion_name", "depth", "output", "local_vote", "outcome_reason", "backend_source", "metrics", "generation_metrics", "cache_key", "edges", "subtree_vote"}, "node trace")
        if not isinstance(value["edges"], list):
            raise ValueError("node trace edges must be a list")
        subtree_vote = value["subtree_vote"]
        return cls(
            node_id=value["node_id"], criterion_name=value["criterion_name"], depth=value["depth"],
            output=StructuredNodeOutput.from_dict(value["output"]), local_vote=Vote(value["local_vote"]),
            outcome_reason=OutcomeReason(value["outcome_reason"]), backend_source=BackendSource(value["backend_source"]),
            metrics=ModelCallMetrics.from_dict(value["metrics"]), generation_metrics=ModelCallMetrics.from_dict(value["generation_metrics"]),
            cache_key=value["cache_key"], edges=tuple(EdgeExecutionTrace.from_dict(edge) for edge in value["edges"]),
            subtree_vote=None if subtree_vote is None else Vote(subtree_vote),
        )


@dataclass(frozen=True)
class RootExecutionTrace:
    root_id: str
    selected: bool
    skip_reason: str | None
    visited_node_ids: tuple[str, ...]
    avoided_node_ids: tuple[str, ...]
    subtree_vote: Vote | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "visited_node_ids", tuple(self.visited_node_ids))
        object.__setattr__(self, "avoided_node_ids", tuple(self.avoided_node_ids))

    def to_dict(self) -> dict[str, object]:
        return {
            "root_id": self.root_id, "selected": self.selected, "skip_reason": self.skip_reason,
            "visited_node_ids": list(self.visited_node_ids), "avoided_node_ids": list(self.avoided_node_ids),
            "subtree_vote": None if self.subtree_vote is None else self.subtree_vote.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RootExecutionTrace":
        value = _exact(value, {"root_id", "selected", "skip_reason", "visited_node_ids", "avoided_node_ids", "subtree_vote"}, "root trace")
        vote = value["subtree_vote"]
        return cls(
            root_id=value["root_id"], selected=value["selected"], skip_reason=value["skip_reason"],
            visited_node_ids=tuple(value["visited_node_ids"]), avoided_node_ids=tuple(value["avoided_node_ids"]),
            subtree_vote=None if vote is None else Vote(vote),
        )


def _resolved_to_dict(value: ResolvedRootRouting) -> dict[str, object]:
    return {
        "selected_root_ids": list(value.selected_root_ids), "source": value.source.value,
        "outcome_reason": value.outcome_reason.value, "consistency_errors": list(value.consistency_errors),
    }


def _resolved_from_dict(value: object) -> ResolvedRootRouting:
    value = _exact(value, {"selected_root_ids", "source", "outcome_reason", "consistency_errors"}, "resolved routing")
    return ResolvedRootRouting(
        selected_root_ids=tuple(value["selected_root_ids"]), source=RoutingSource(value["source"]),
        outcome_reason=RoutingOutcomeReason(value["outcome_reason"]), consistency_errors=tuple(value["consistency_errors"]),
    )


@dataclass(frozen=True)
class RouterExecutionTrace:
    enabled: bool
    output: RootRouterOutput | None
    resolved: ResolvedRootRouting
    backend_source: BackendSource | None
    metrics: ModelCallMetrics
    generation_metrics: ModelCallMetrics
    cache_key: str | None

    def __post_init__(self) -> None:
        zero = ModelCallMetrics()
        if self.backend_source is None:
            if self.output is not None:
                raise ValueError("router without backend source must not contain output")
            if self.metrics != zero or self.generation_metrics != zero:
                raise ValueError("router without backend source must have zero metrics")
            if self.cache_key is not None:
                raise ValueError("router without backend source must not contain cache key")
            return
        if self.output is None:
            raise ValueError("router backend result must contain output")
        _validate_backend_metrics(
            self.backend_source,
            self.metrics,
            self.generation_metrics,
            label="router",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled, "output": None if self.output is None else self.output.to_dict(),
            "resolved": _resolved_to_dict(self.resolved),
            "backend_source": None if self.backend_source is None else self.backend_source.value,
            "metrics": self.metrics.to_dict(), "generation_metrics": self.generation_metrics.to_dict(),
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RouterExecutionTrace":
        value = _exact(value, {"enabled", "output", "resolved", "backend_source", "metrics", "generation_metrics", "cache_key"}, "router trace")
        return cls(
            enabled=value["enabled"], output=None if value["output"] is None else RootRouterOutput.from_dict(value["output"]),
            resolved=_resolved_from_dict(value["resolved"]),
            backend_source=None if value["backend_source"] is None else BackendSource(value["backend_source"]),
            metrics=ModelCallMetrics.from_dict(value["metrics"]), generation_metrics=ModelCallMetrics.from_dict(value["generation_metrics"]),
            cache_key=value["cache_key"],
        )


@dataclass(frozen=True)
class CounterfactualTraversalMetrics:
    total_nodes: int
    visited_nodes: int
    avoided_nodes: int
    matched_edges: int
    skipped_edges: int
    max_visited_depth: int
    selected_roots: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_nodes": self.total_nodes,
            "visited_nodes": self.visited_nodes,
            "avoided_nodes": self.avoided_nodes,
            "matched_edges": self.matched_edges,
            "skipped_edges": self.skipped_edges,
            "max_visited_depth": self.max_visited_depth,
            "selected_roots": self.selected_roots,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CounterfactualTraversalMetrics":
        expected = {"total_nodes", "visited_nodes", "avoided_nodes", "matched_edges", "skipped_edges", "max_visited_depth", "selected_roots"}
        return cls(**_exact(value, expected, "counterfactual metrics"))


@dataclass(frozen=True)
class ExecutionTrace:
    sample_id: str
    sample_fingerprint: str
    rubric_sha256: str
    variant: str
    traversal_policy: TraversalPolicy
    aggregation_mode: AggregationMode
    router: RouterExecutionTrace
    roots: tuple[RootExecutionTrace, ...]
    nodes: tuple[NodeExecutionTrace, ...]
    aggregation_input_votes: Mapping[str, Vote]
    final_preference: FinalPreference
    counterfactual_metrics: CounterfactualTraversalMetrics
    node_metrics: ModelCallMetrics
    wall_latency_seconds: float
    semantics_version: str = STRUCTURED_SEMANTICS_VERSION
    rubric_schema_version: str = STRUCTURED_RUBRIC_SCHEMA_VERSION
    trace_version: str = STRUCTURED_EXECUTION_TRACE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", tuple(self.roots))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(
            self,
            "aggregation_input_votes",
            MappingProxyType(dict(self.aggregation_input_votes)),
        )
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("trace sample_id must be non-empty")
        if len(self.sample_fingerprint) != 64 or len(self.rubric_sha256) != 64:
            raise ValueError("trace fingerprints must be SHA-256 strings")
        if self.semantics_version != STRUCTURED_SEMANTICS_VERSION:
            raise ValueError("trace semantics version mismatch")
        if self.rubric_schema_version != STRUCTURED_RUBRIC_SCHEMA_VERSION:
            raise ValueError("trace rubric schema version mismatch")
        if self.trace_version != STRUCTURED_EXECUTION_TRACE_VERSION:
            raise ValueError("trace version mismatch")
        if isinstance(self.wall_latency_seconds, bool) or not isinstance(self.wall_latency_seconds, (int, float)) or not math.isfinite(self.wall_latency_seconds) or self.wall_latency_seconds < 0:
            raise ValueError("wall_latency_seconds must be finite and non-negative")
        object.__setattr__(self, "wall_latency_seconds", float(self.wall_latency_seconds))
        if len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("trace nodes must be unique")
        expected_node_metrics = combine_model_call_metrics(
            node.metrics for node in self.nodes
        )
        if self.node_metrics != expected_node_metrics:
            raise ValueError("trace node_metrics do not match node detail metrics")

    @property
    def total_metrics(self) -> ModelCallMetrics:
        return combine_model_call_metrics((self.router.metrics, self.node_metrics))

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics_version": self.semantics_version,
            "rubric_schema_version": self.rubric_schema_version,
            "trace_version": self.trace_version,
            "sample_id": self.sample_id,
            "sample_fingerprint": self.sample_fingerprint,
            "rubric_sha256": self.rubric_sha256,
            "variant": self.variant,
            "traversal_policy": self.traversal_policy.value,
            "aggregation_mode": self.aggregation_mode.value,
            "router": self.router.to_dict(),
            "roots": [root.to_dict() for root in self.roots],
            "nodes": [node.to_dict() for node in self.nodes],
            "aggregation_input_votes": {key: vote.value for key, vote in self.aggregation_input_votes.items()},
            "final_preference": self.final_preference.value,
            "counterfactual_metrics": self.counterfactual_metrics.to_dict(),
            "node_metrics": self.node_metrics.to_dict(),
            "wall_latency_seconds": self.wall_latency_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ExecutionTrace":
        expected = {
            "semantics_version", "rubric_schema_version", "trace_version", "sample_id",
            "sample_fingerprint", "rubric_sha256", "variant", "traversal_policy",
            "aggregation_mode", "router", "roots", "nodes", "aggregation_input_votes",
            "final_preference", "counterfactual_metrics", "node_metrics", "wall_latency_seconds",
        }
        value = _exact(value, expected, "execution trace")
        if not isinstance(value["roots"], list) or not isinstance(value["nodes"], list) or not isinstance(value["aggregation_input_votes"], dict):
            raise ValueError("trace roots/nodes/votes have invalid types")
        return cls(
            sample_id=value["sample_id"], sample_fingerprint=value["sample_fingerprint"],
            rubric_sha256=value["rubric_sha256"], variant=value["variant"],
            traversal_policy=TraversalPolicy(value["traversal_policy"]), aggregation_mode=AggregationMode(value["aggregation_mode"]),
            router=RouterExecutionTrace.from_dict(value["router"]),
            roots=tuple(RootExecutionTrace.from_dict(root) for root in value["roots"]),
            nodes=tuple(NodeExecutionTrace.from_dict(node) for node in value["nodes"]),
            aggregation_input_votes={key: Vote(vote) for key, vote in value["aggregation_input_votes"].items()},
            final_preference=FinalPreference(value["final_preference"]),
            counterfactual_metrics=CounterfactualTraversalMetrics.from_dict(value["counterfactual_metrics"]),
            node_metrics=ModelCallMetrics.from_dict(value["node_metrics"]),
            wall_latency_seconds=value["wall_latency_seconds"],
            semantics_version=value["semantics_version"], rubric_schema_version=value["rubric_schema_version"], trace_version=value["trace_version"],
        )


@dataclass(frozen=True)
class StructuredExecutionOutput:
    variant: str
    rubric_sha256: str
    traces: tuple[ExecutionTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))
        if not self.traces:
            raise ValueError("execution output must contain traces")
        if len({trace.sample_id for trace in self.traces}) != len(self.traces):
            raise ValueError("execution output sample IDs must be unique")
        if any(trace.variant != self.variant or trace.rubric_sha256 != self.rubric_sha256 for trace in self.traces):
            raise ValueError("execution output trace identity mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "structured_execution_output",
            "trace_version": STRUCTURED_EXECUTION_TRACE_VERSION,
            "variant": self.variant,
            "rubric_sha256": self.rubric_sha256,
            "traces": [trace.to_dict() for trace in self.traces],
        }

    @classmethod
    def from_dict(cls, value: object) -> "StructuredExecutionOutput":
        value = _exact(value, {"artifact_type", "trace_version", "variant", "rubric_sha256", "traces"}, "execution output")
        if value["artifact_type"] != "structured_execution_output" or value["trace_version"] != STRUCTURED_EXECUTION_TRACE_VERSION:
            raise ValueError("execution output type/version mismatch")
        if not isinstance(value["traces"], list):
            raise ValueError("execution output traces must be a list")
        return cls(variant=value["variant"], rubric_sha256=value["rubric_sha256"], traces=tuple(ExecutionTrace.from_dict(trace) for trace in value["traces"]))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "StructuredExecutionOutput":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load execution output: {exc}") from exc
        return cls.from_dict(value)


@dataclass(frozen=True)
class StructuredExecutionEvaluation:
    prediction: StructuredExecutionOutput
    is_correct: tuple[bool, ...]
    accuracy: float
    coverage: float
    tie_rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_correct", tuple(self.is_correct))
        if not isinstance(self.prediction, StructuredExecutionOutput):
            raise TypeError("prediction must be StructuredExecutionOutput")
        if len(self.is_correct) != len(self.prediction.traces):
            raise ValueError("is_correct length must equal prediction traces")
        if any(not isinstance(value, bool) for value in self.is_correct):
            raise TypeError("is_correct values must be bool")
        for name in ("accuracy", "coverage", "tie_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
            object.__setattr__(self, name, normalized)

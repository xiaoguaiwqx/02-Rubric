"""Trace v2 for the decoupled pairwise-vote and gate-state executor."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .backend import BackendSource
from .dual_worker import GateStateOutput, PairwiseVoteOutput
from .judgement import FinalPreference, Vote
from .semantics import TraversalPolicy
from .telemetry import ModelCallMetrics, combine_model_call_metrics
from .trace import (
    AggregationMode,
    CounterfactualTraversalMetrics,
    RootExecutionTrace,
    RouterExecutionTrace,
)
from .version import DUAL_EXECUTION_TRACE_VERSION, STRUCTURED_RUBRIC_SCHEMA_VERSION, STRUCTURED_SEMANTICS_VERSION


def _exact(value: object, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _validate_channel_metrics(source: BackendSource, metrics: ModelCallMetrics,
                              generation: ModelCallMetrics, label: str) -> None:
    if source is BackendSource.OFFLINE_ARTIFACT:
        if metrics != ModelCallMetrics():
            raise ValueError(f"{label} offline metrics must be zero")
    elif source is BackendSource.CACHE:
        if metrics != ModelCallMetrics.from_agent_calls((), cache_hit=True):
            raise ValueError(f"{label} cache metrics are inconsistent")
    elif source is BackendSource.MODEL:
        if generation.cache_hits or generation.cache_misses:
            raise ValueError(f"{label} generation metrics must exclude cache counts")
        if metrics.cache_misses not in (0, 1):
            raise ValueError(f"{label} cache miss count is invalid")
        if metrics != replace(generation, cache_misses=metrics.cache_misses):
            raise ValueError(f"{label} model metrics differ from generation provenance")
    else:
        raise TypeError(f"{label} has unsupported backend source")


@dataclass(frozen=True)
class DualEdgeExecutionTrace:
    parent_id: str
    child_id: str
    condition: str
    condition_source: str
    matched: bool
    child_visited: bool
    skip_reason: str | None
    child_subtree_vote: Vote | None

    def to_dict(self) -> dict[str, object]:
        return {"parent_id": self.parent_id, "child_id": self.child_id, "condition": self.condition,
                "condition_source": self.condition_source, "matched": self.matched,
                "child_visited": self.child_visited, "skip_reason": self.skip_reason,
                "child_subtree_vote": None if self.child_subtree_vote is None else self.child_subtree_vote.value}

    @classmethod
    def from_dict(cls, value: object) -> "DualEdgeExecutionTrace":
        value = _exact(value, {"parent_id", "child_id", "condition", "condition_source", "matched",
                               "child_visited", "skip_reason", "child_subtree_vote"}, "dual edge trace")
        vote = value["child_subtree_vote"]
        return cls(value["parent_id"], value["child_id"], value["condition"], value["condition_source"],
                   value["matched"], value["child_visited"], value["skip_reason"],
                   None if vote is None else Vote(vote))


@dataclass(frozen=True)
class DualNodeExecutionTrace:
    node_id: str
    criterion_name: str
    depth: int
    pairwise_output: PairwiseVoteOutput
    pairwise_source: BackendSource
    pairwise_metrics: ModelCallMetrics
    pairwise_generation_metrics: ModelCallMetrics
    pairwise_cache_key: str | None
    gate_output: GateStateOutput | None
    gate_source: BackendSource | None
    gate_metrics: ModelCallMetrics
    gate_generation_metrics: ModelCallMetrics
    gate_cache_key: str | None
    local_vote: Vote
    edges: tuple[DualEdgeExecutionTrace, ...]
    subtree_vote: Vote

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))
        if self.local_vote is not self.pairwise_output.vote:
            raise ValueError("dual trace local vote must come from pairwise output")
        _validate_channel_metrics(self.pairwise_source, self.pairwise_metrics,
                                  self.pairwise_generation_metrics, "pairwise")
        if self.gate_output is None:
            if self.gate_source is not None or self.gate_metrics != ModelCallMetrics() or self.gate_generation_metrics != ModelCallMetrics() or self.gate_cache_key is not None:
                raise ValueError("node without gate output must have empty gate provenance")
        elif self.gate_source is None:
            raise ValueError("gate output requires a backend source")
        else:
            _validate_channel_metrics(self.gate_source, self.gate_metrics,
                                      self.gate_generation_metrics, "gate")

    @property
    def metrics(self) -> ModelCallMetrics:
        return combine_model_call_metrics((self.pairwise_metrics, self.gate_metrics))

    def to_dict(self) -> dict[str, object]:
        return {"node_id": self.node_id, "criterion_name": self.criterion_name, "depth": self.depth,
                "pairwise_output": self.pairwise_output.to_dict(), "pairwise_source": self.pairwise_source.value,
                "pairwise_metrics": self.pairwise_metrics.to_dict(), "pairwise_generation_metrics": self.pairwise_generation_metrics.to_dict(),
                "pairwise_cache_key": self.pairwise_cache_key,
                "gate_output": None if self.gate_output is None else self.gate_output.to_dict(),
                "gate_source": None if self.gate_source is None else self.gate_source.value,
                "gate_metrics": self.gate_metrics.to_dict(), "gate_generation_metrics": self.gate_generation_metrics.to_dict(),
                "gate_cache_key": self.gate_cache_key, "local_vote": self.local_vote.value,
                "edges": [edge.to_dict() for edge in self.edges], "subtree_vote": self.subtree_vote.value}

    @classmethod
    def from_dict(cls, value: object) -> "DualNodeExecutionTrace":
        fields = {"node_id", "criterion_name", "depth", "pairwise_output", "pairwise_source",
                  "pairwise_metrics", "pairwise_generation_metrics", "pairwise_cache_key", "gate_output",
                  "gate_source", "gate_metrics", "gate_generation_metrics", "gate_cache_key", "local_vote", "edges", "subtree_vote"}
        value = _exact(value, fields, "dual node trace")
        if not isinstance(value["edges"], list): raise ValueError("dual node edges must be a list")
        return cls(value["node_id"], value["criterion_name"], value["depth"],
            PairwiseVoteOutput.from_dict(value["pairwise_output"]), BackendSource(value["pairwise_source"]),
            ModelCallMetrics.from_dict(value["pairwise_metrics"]), ModelCallMetrics.from_dict(value["pairwise_generation_metrics"]),
            value["pairwise_cache_key"], None if value["gate_output"] is None else GateStateOutput.from_dict(value["gate_output"]),
            None if value["gate_source"] is None else BackendSource(value["gate_source"]),
            ModelCallMetrics.from_dict(value["gate_metrics"]), ModelCallMetrics.from_dict(value["gate_generation_metrics"]),
            value["gate_cache_key"], Vote(value["local_vote"]),
            tuple(DualEdgeExecutionTrace.from_dict(edge) for edge in value["edges"]), Vote(value["subtree_vote"]))


@dataclass(frozen=True)
class DualExecutionTrace:
    sample_id: str
    sample_fingerprint: str
    rubric_sha256: str
    variant: str
    traversal_policy: TraversalPolicy
    aggregation_mode: AggregationMode
    router: RouterExecutionTrace
    roots: tuple[RootExecutionTrace, ...]
    nodes: tuple[DualNodeExecutionTrace, ...]
    aggregation_input_votes: Mapping[str, Vote]
    final_preference: FinalPreference
    counterfactual_metrics: CounterfactualTraversalMetrics
    pairwise_metrics: ModelCallMetrics
    gate_metrics: ModelCallMetrics
    wall_latency_seconds: float
    semantics_version: str = STRUCTURED_SEMANTICS_VERSION
    rubric_schema_version: str = STRUCTURED_RUBRIC_SCHEMA_VERSION
    trace_version: str = DUAL_EXECUTION_TRACE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", tuple(self.roots)); object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "aggregation_input_votes", MappingProxyType(dict(self.aggregation_input_votes)))
        if self.trace_version != DUAL_EXECUTION_TRACE_VERSION or self.semantics_version != STRUCTURED_SEMANTICS_VERSION or self.rubric_schema_version != STRUCTURED_RUBRIC_SCHEMA_VERSION:
            raise ValueError("dual execution trace version mismatch")
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("dual trace sample_id must be non-empty")
        for label, value in (("sample_fingerprint", self.sample_fingerprint),
                             ("rubric_sha256", self.rubric_sha256)):
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError(f"dual trace {label} must be SHA-256 hex")
        if not isinstance(self.traversal_policy, TraversalPolicy):
            raise TypeError("dual trace traversal_policy is invalid")
        if not isinstance(self.aggregation_mode, AggregationMode):
            raise TypeError("dual trace aggregation_mode is invalid")
        if (isinstance(self.wall_latency_seconds, bool)
                or not isinstance(self.wall_latency_seconds, (int, float))
                or not math.isfinite(self.wall_latency_seconds)
                or self.wall_latency_seconds < 0):
            raise ValueError("dual trace wall latency must be finite and non-negative")
        object.__setattr__(self, "wall_latency_seconds", float(self.wall_latency_seconds))
        if len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("dual trace node IDs must be unique")
        if len({root.root_id for root in self.roots}) != len(self.roots):
            raise ValueError("dual trace root IDs must be unique")
        if self.pairwise_metrics != combine_model_call_metrics(node.pairwise_metrics for node in self.nodes):
            raise ValueError("pairwise metric summary does not match node details")
        if self.gate_metrics != combine_model_call_metrics(node.gate_metrics for node in self.nodes):
            raise ValueError("gate metric summary does not match node details")

    @property
    def total_metrics(self) -> ModelCallMetrics:
        return combine_model_call_metrics((self.router.metrics, self.pairwise_metrics, self.gate_metrics))

    def to_dict(self) -> dict[str, object]:
        return {"semantics_version": self.semantics_version, "rubric_schema_version": self.rubric_schema_version,
            "trace_version": self.trace_version, "sample_id": self.sample_id, "sample_fingerprint": self.sample_fingerprint,
            "rubric_sha256": self.rubric_sha256, "variant": self.variant, "traversal_policy": self.traversal_policy.value,
            "aggregation_mode": self.aggregation_mode.value, "router": self.router.to_dict(),
            "roots": [root.to_dict() for root in self.roots], "nodes": [node.to_dict() for node in self.nodes],
            "aggregation_input_votes": {key: vote.value for key, vote in self.aggregation_input_votes.items()},
            "final_preference": self.final_preference.value, "counterfactual_metrics": self.counterfactual_metrics.to_dict(),
            "pairwise_metrics": self.pairwise_metrics.to_dict(), "gate_metrics": self.gate_metrics.to_dict(),
            "wall_latency_seconds": self.wall_latency_seconds}

    @classmethod
    def from_dict(cls, value: object) -> "DualExecutionTrace":
        fields = {"semantics_version", "rubric_schema_version", "trace_version", "sample_id", "sample_fingerprint",
                  "rubric_sha256", "variant", "traversal_policy", "aggregation_mode", "router", "roots", "nodes",
                  "aggregation_input_votes", "final_preference", "counterfactual_metrics", "pairwise_metrics",
                  "gate_metrics", "wall_latency_seconds"}
        value = _exact(value, fields, "dual execution trace")
        return cls(value["sample_id"], value["sample_fingerprint"], value["rubric_sha256"], value["variant"],
            TraversalPolicy(value["traversal_policy"]), AggregationMode(value["aggregation_mode"]),
            RouterExecutionTrace.from_dict(value["router"]), tuple(RootExecutionTrace.from_dict(item) for item in value["roots"]),
            tuple(DualNodeExecutionTrace.from_dict(item) for item in value["nodes"]),
            {key: Vote(vote) for key, vote in value["aggregation_input_votes"].items()}, FinalPreference(value["final_preference"]),
            CounterfactualTraversalMetrics.from_dict(value["counterfactual_metrics"]),
            ModelCallMetrics.from_dict(value["pairwise_metrics"]), ModelCallMetrics.from_dict(value["gate_metrics"]),
            value["wall_latency_seconds"], value["semantics_version"], value["rubric_schema_version"], value["trace_version"])


@dataclass(frozen=True)
class DualExecutionOutput:
    variant: str
    rubric_sha256: str
    traces: tuple[DualExecutionTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))
        if not self.traces or any(trace.variant != self.variant or trace.rubric_sha256 != self.rubric_sha256 for trace in self.traces):
            raise ValueError("dual execution output trace identity mismatch")
        if len({trace.sample_id for trace in self.traces}) != len(self.traces):
            raise ValueError("dual execution output sample IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        return {"artifact_type": "dual_execution_output", "trace_version": DUAL_EXECUTION_TRACE_VERSION,
                "variant": self.variant, "rubric_sha256": self.rubric_sha256,
                "traces": [trace.to_dict() for trace in self.traces]}

    @classmethod
    def from_dict(cls, value: object) -> "DualExecutionOutput":
        value = _exact(value, {"artifact_type", "trace_version", "variant", "rubric_sha256", "traces"}, "dual output")
        if value["artifact_type"] != "dual_execution_output" or value["trace_version"] != DUAL_EXECUTION_TRACE_VERSION:
            raise ValueError("dual execution output type/version mismatch")
        return cls(value["variant"], value["rubric_sha256"], tuple(DualExecutionTrace.from_dict(item) for item in value["traces"]))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "DualExecutionOutput":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

"""Immutable schemas for the Refine v1 operator."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ..telemetry import ModelCallMetrics
from ..version import REFINE_SCHEMA_VERSION
from .specialize_types import SpecializeManagerRequestSpec
from .types import EditCandidate, EvolutionDecision


def _clean(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without outer whitespace")
    return value


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")
    return value


def _versioned(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    value = _exact(value, fields | {"schema_version"}, label)
    if value["schema_version"] != REFINE_SCHEMA_VERSION:
        raise ValueError(f"{label} schema version mismatch")
    return value


def _strings(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(_clean(item, label) for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    return result


def _rate(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return float(value)


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RefineTriggerDecision:
    node_id: str
    triggered: bool
    forced: bool
    reasons: tuple[str, ...]
    accuracy: float
    coverage: float
    support: int
    wrong: int

    def __post_init__(self) -> None:
        _clean(self.node_id, "node_id")
        if not isinstance(self.triggered, bool) or not isinstance(self.forced, bool):
            raise TypeError("triggered and forced must be bool")
        object.__setattr__(self, "reasons", _strings(self.reasons, "reasons", allow_empty=True))
        object.__setattr__(self, "accuracy", _rate(self.accuracy, "accuracy"))
        object.__setattr__(self, "coverage", _rate(self.coverage, "coverage"))
        _count(self.support, "support"); _count(self.wrong, "wrong")
        if self.triggered and self.reasons and not self.forced:
            raise ValueError("a non-forced triggered decision cannot have failure reasons")
        if not self.triggered and not self.reasons:
            raise ValueError("a rejected trigger requires reasons")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": REFINE_SCHEMA_VERSION, "node_id": self.node_id,
                "triggered": self.triggered, "forced": self.forced,
                "reasons": list(self.reasons), "accuracy": self.accuracy,
                "coverage": self.coverage, "support": self.support, "wrong": self.wrong}

    @classmethod
    def from_dict(cls, value: object) -> "RefineTriggerDecision":
        fields = {"node_id", "triggered", "forced", "reasons", "accuracy",
                  "coverage", "support", "wrong"}
        value = _versioned(value, fields, "refine trigger")
        return cls(value["node_id"], value["triggered"], value["forced"],
                   tuple(value["reasons"]), value["accuracy"], value["coverage"],
                   value["support"], value["wrong"])


@dataclass(frozen=True)
class RefineProposal:
    node_id: str
    criterion_name: str
    original_description_sha256: str
    description: str
    failure_analysis: tuple[str, ...]
    rationale: str
    representative_sample_ids: tuple[str, ...]
    raw_response: str
    attempt_count: int
    metrics: ModelCallMetrics
    request_spec: SpecializeManagerRequestSpec

    def __post_init__(self) -> None:
        for field in ("node_id", "criterion_name", "description", "rationale", "raw_response"):
            _clean(getattr(self, field), field)
        if not re.fullmatch(r"[0-9a-f]{64}", self.original_description_sha256):
            raise ValueError("original_description_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "failure_analysis",
                           _strings(self.failure_analysis, "failure_analysis"))
        object.__setattr__(self, "representative_sample_ids",
                           _strings(self.representative_sample_ids,
                                    "representative_sample_ids"))
        if not 1 <= len(self.representative_sample_ids) <= 6:
            raise ValueError("Refine must cite 1-6 representative samples")
        if isinstance(self.attempt_count, bool) or self.attempt_count < 1:
            raise ValueError("attempt_count must be positive")
        if not isinstance(self.metrics, ModelCallMetrics):
            raise TypeError("metrics must be ModelCallMetrics")
        if not isinstance(self.request_spec, SpecializeManagerRequestSpec):
            raise TypeError("request_spec must be SpecializeManagerRequestSpec")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": REFINE_SCHEMA_VERSION, "node_id": self.node_id,
                "criterion_name": self.criterion_name,
                "original_description_sha256": self.original_description_sha256,
                "description": self.description,
                "failure_analysis": list(self.failure_analysis),
                "rationale": self.rationale,
                "representative_sample_ids": list(self.representative_sample_ids),
                "raw_response": self.raw_response, "attempt_count": self.attempt_count,
                "metrics": self.metrics.to_dict(),
                "request_spec": self.request_spec.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "RefineProposal":
        fields = {"node_id", "criterion_name", "original_description_sha256",
                  "description", "failure_analysis", "rationale",
                  "representative_sample_ids", "raw_response", "attempt_count",
                  "metrics", "request_spec"}
        value = _versioned(value, fields, "refine proposal")
        return cls(value["node_id"], value["criterion_name"],
                   value["original_description_sha256"], value["description"],
                   tuple(value["failure_analysis"]), value["rationale"],
                   tuple(value["representative_sample_ids"]), value["raw_response"],
                   value["attempt_count"], ModelCallMetrics.from_dict(value["metrics"]),
                   SpecializeManagerRequestSpec.from_dict(value["request_spec"]))


@dataclass(frozen=True)
class RefineCandidate:
    node_id: str
    edit_candidate: EditCandidate
    proposal: RefineProposal

    def __post_init__(self) -> None:
        _clean(self.node_id, "node_id")
        if not isinstance(self.edit_candidate, EditCandidate):
            raise TypeError("edit_candidate must be EditCandidate")
        if not isinstance(self.proposal, RefineProposal):
            raise TypeError("proposal must be RefineProposal")
        if self.node_id != self.proposal.node_id:
            raise ValueError("candidate node_id does not match proposal")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": REFINE_SCHEMA_VERSION, "node_id": self.node_id,
                "edit_candidate": self.edit_candidate.to_dict(),
                "proposal": self.proposal.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "RefineCandidate":
        value = _versioned(value, {"node_id", "edit_candidate", "proposal"},
                           "refine candidate")
        return cls(value["node_id"], EditCandidate.from_dict(value["edit_candidate"]),
                   RefineProposal.from_dict(value["proposal"]))


@dataclass(frozen=True)
class RefineNodeMetric:
    support: int
    correct: int
    wrong: int
    accuracy: float
    coverage: float
    valid_rate: float

    def __post_init__(self) -> None:
        for field in ("support", "correct", "wrong"):
            _count(getattr(self, field), field)
        if self.correct + self.wrong != self.support:
            raise ValueError("support must equal correct + wrong")
        for field in ("accuracy", "coverage", "valid_rate"):
            object.__setattr__(self, field, _rate(getattr(self, field), field))
        expected = self.correct / self.support if self.support else 0.0
        if not math.isclose(self.accuracy, expected, abs_tol=1e-12):
            raise ValueError("accuracy does not match counts")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": REFINE_SCHEMA_VERSION, **self.__dict__}

    @classmethod
    def from_dict(cls, value: object) -> "RefineNodeMetric":
        fields = {"support", "correct", "wrong", "accuracy", "coverage", "valid_rate"}
        value = _versioned(value, fields, "refine node metric")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True)
class RefineEvaluation:
    node_id: str
    root_node_id: str
    old_node: RefineNodeMetric
    new_node: RefineNodeMetric
    node_accuracy_delta: float
    corrected_sample_ids: tuple[str, ...]
    harmed_sample_ids: tuple[str, ...]
    transition_counts: Mapping[str, int]
    new_scope_expansion: int
    new_scope_expansion_wrong: int
    subtree_scope_support: int
    old_subtree_accuracy: float
    new_subtree_accuracy: float
    subtree_accuracy_delta: float
    old_subtree_coverage: float
    new_subtree_coverage: float
    subtree_corrected_sample_ids: tuple[str, ...]
    subtree_harmed_sample_ids: tuple[str, ...]
    old_m1_accuracy: float
    new_m1_accuracy: float
    m1_accuracy_delta: float
    old_m1_coverage: float
    new_m1_coverage: float
    m1_corrected_sample_ids: tuple[str, ...]
    m1_harmed_sample_ids: tuple[str, ...]
    sibling_joint_decisive: int
    sibling_conflict_count: int
    decision: EvolutionDecision
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _clean(self.node_id, "node_id"); _clean(self.root_node_id, "root_node_id")
        if not isinstance(self.old_node, RefineNodeMetric) or not isinstance(self.new_node, RefineNodeMetric):
            raise TypeError("old_node and new_node must be RefineNodeMetric")
        for field in ("new_scope_expansion", "new_scope_expansion_wrong",
                      "subtree_scope_support", "sibling_joint_decisive",
                      "sibling_conflict_count"):
            _count(getattr(self, field), field)
        for field in ("old_subtree_accuracy", "new_subtree_accuracy",
                      "old_subtree_coverage", "new_subtree_coverage",
                      "old_m1_accuracy", "new_m1_accuracy",
                      "old_m1_coverage", "new_m1_coverage"):
            object.__setattr__(self, field, _rate(getattr(self, field), field))
        for field in ("node_accuracy_delta", "subtree_accuracy_delta", "m1_accuracy_delta"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if not math.isclose(self.node_accuracy_delta,
                            self.new_node.accuracy - self.old_node.accuracy, abs_tol=1e-12):
            raise ValueError("node_accuracy_delta mismatch")
        if not math.isclose(self.subtree_accuracy_delta,
                            self.new_subtree_accuracy - self.old_subtree_accuracy, abs_tol=1e-12):
            raise ValueError("subtree_accuracy_delta mismatch")
        if not math.isclose(self.m1_accuracy_delta,
                            self.new_m1_accuracy - self.old_m1_accuracy, abs_tol=1e-12):
            raise ValueError("m1_accuracy_delta mismatch")
        for field in ("corrected_sample_ids", "harmed_sample_ids",
                      "subtree_corrected_sample_ids", "subtree_harmed_sample_ids",
                      "m1_corrected_sample_ids", "m1_harmed_sample_ids", "reasons"):
            object.__setattr__(self, field,
                               _strings(getattr(self, field), field, allow_empty=True))
        transitions = dict(self.transition_counts)
        if any(not isinstance(key, str) or _count(value, key) != value
               for key, value in transitions.items()):
            raise ValueError("transition_counts must contain non-negative integer values")
        object.__setattr__(self, "transition_counts", MappingProxyType(transitions))
        if not isinstance(self.decision, EvolutionDecision):
            raise TypeError("decision must be EvolutionDecision")
        if self.decision is EvolutionDecision.ACCEPT and self.reasons:
            raise ValueError("accepted Refine cannot have rejection reasons")
        if self.decision is EvolutionDecision.REJECT and not self.reasons:
            raise ValueError("rejected Refine requires reasons")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": REFINE_SCHEMA_VERSION, "node_id": self.node_id,
                "root_node_id": self.root_node_id,
                "old_node": self.old_node.to_dict(), "new_node": self.new_node.to_dict(),
                "node_accuracy_delta": self.node_accuracy_delta,
                "corrected_sample_ids": list(self.corrected_sample_ids),
                "harmed_sample_ids": list(self.harmed_sample_ids),
                "transition_counts": dict(self.transition_counts),
                "new_scope_expansion": self.new_scope_expansion,
                "new_scope_expansion_wrong": self.new_scope_expansion_wrong,
                "subtree_scope_support": self.subtree_scope_support,
                "old_subtree_accuracy": self.old_subtree_accuracy,
                "new_subtree_accuracy": self.new_subtree_accuracy,
                "subtree_accuracy_delta": self.subtree_accuracy_delta,
                "old_subtree_coverage": self.old_subtree_coverage,
                "new_subtree_coverage": self.new_subtree_coverage,
                "subtree_corrected_sample_ids": list(self.subtree_corrected_sample_ids),
                "subtree_harmed_sample_ids": list(self.subtree_harmed_sample_ids),
                "old_m1_accuracy": self.old_m1_accuracy,
                "new_m1_accuracy": self.new_m1_accuracy,
                "m1_accuracy_delta": self.m1_accuracy_delta,
                "old_m1_coverage": self.old_m1_coverage,
                "new_m1_coverage": self.new_m1_coverage,
                "m1_corrected_sample_ids": list(self.m1_corrected_sample_ids),
                "m1_harmed_sample_ids": list(self.m1_harmed_sample_ids),
                "sibling_joint_decisive": self.sibling_joint_decisive,
                "sibling_conflict_count": self.sibling_conflict_count,
                "decision": self.decision.value, "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, value: object) -> "RefineEvaluation":
        fields = set(cls.__dataclass_fields__)
        value = _versioned(value, fields, "refine evaluation")
        kwargs = {field: value[field] for field in fields}
        kwargs["old_node"] = RefineNodeMetric.from_dict(value["old_node"])
        kwargs["new_node"] = RefineNodeMetric.from_dict(value["new_node"])
        kwargs["decision"] = EvolutionDecision(value["decision"])
        for field in ("corrected_sample_ids", "harmed_sample_ids",
                      "subtree_corrected_sample_ids", "subtree_harmed_sample_ids",
                      "m1_corrected_sample_ids", "m1_harmed_sample_ids", "reasons"):
            kwargs[field] = tuple(value[field])
        return cls(**kwargs)

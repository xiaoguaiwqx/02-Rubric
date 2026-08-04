"""Immutable Phase 5 schemas and paired candidate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..judgement import FinalPreference, Vote
from ..schema import RubricEdge, RubricNode, StructuredRubric
from ..version import STRUCTURED_EVOLUTION_SCHEMA_VERSION


def _clean(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without outer whitespace")
    return value


def _sha(value: object, label: str) -> str:
    value = _clean(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return value


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) != fields:
        raise ValueError(f"{label} fields mismatch: {sorted(set(value) ^ fields)}")
    return value


def _versioned(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    value = _exact(value, fields | {"schema_version"}, label)
    if value["schema_version"] != STRUCTURED_EVOLUTION_SCHEMA_VERSION:
        raise ValueError(f"{label} schema version mismatch")
    return value


class OperatorKind(str, Enum):
    MANUAL = "manual"
    REFINE = "refine"
    SPECIALIZE = "specialize"
    CREATE = "create"


class EvolutionDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    INVALID = "invalid"


@dataclass(frozen=True)
class ErrorSampleRef:
    sample_id: str
    criterion_name: str
    vote: Vote
    gold: str
    outcome: str
    thought: str | None

    def __post_init__(self) -> None:
        _clean(self.sample_id, "sample_id")
        _clean(self.criterion_name, "criterion_name")
        if not isinstance(self.vote, Vote):
            raise TypeError("vote must be Vote")
        if self.gold not in {"A", "B"}:
            raise ValueError("gold must be A or B")
        if self.outcome not in {"wrong", "abstain", "answer_invalid", "parse_invalid"}:
            raise ValueError("unsupported error outcome")
        if self.thought is not None and not isinstance(self.thought, str):
            raise TypeError("thought must be str or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "criterion_name": self.criterion_name,
            "vote": self.vote.value,
            "gold": self.gold,
            "outcome": self.outcome,
            "thought": self.thought,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ErrorSampleRef":
        value = _versioned(
            value,
            {"sample_id", "criterion_name", "vote", "gold", "outcome", "thought"},
            "error sample reference",
        )
        return cls(
            sample_id=value["sample_id"],
            criterion_name=value["criterion_name"],
            vote=Vote(value["vote"]),
            gold=value["gold"],
            outcome=value["outcome"],
            thought=value["thought"],
        )


@dataclass(frozen=True)
class NodeFeedback:
    node_id: str
    criterion_name: str
    sample_count: int
    support: int
    correct: int
    wrong: int
    abstain: int
    answer_invalid: int
    parse_invalid: int
    accuracy: float
    coverage: float
    valid_rate: float
    description_length: int
    fitness: float
    errors: tuple[ErrorSampleRef, ...]

    def __post_init__(self) -> None:
        _clean(self.node_id, "node_id")
        _clean(self.criterion_name, "criterion_name")
        counts = (
            self.sample_count, self.support, self.correct, self.wrong,
            self.abstain, self.answer_invalid, self.parse_invalid,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("feedback counts must be non-negative integers")
        if self.sample_count == 0:
            raise ValueError("sample_count must be positive")
        if self.correct + self.wrong != self.support:
            raise ValueError("support must equal correct + wrong")
        if self.support + self.abstain + self.answer_invalid + self.parse_invalid != self.sample_count:
            raise ValueError("node feedback outcomes must partition all samples")
        for value in (self.accuracy, self.coverage, self.valid_rate):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("feedback rates must be finite values in [0, 1]")
        if not math.isfinite(self.fitness):
            raise ValueError("fitness must be finite")
        if isinstance(self.description_length, bool) or not isinstance(self.description_length, int) or self.description_length < 0:
            raise ValueError("description_length must be a non-negative integer")
        expected_accuracy = self.correct / self.support if self.support else 0.0
        if not math.isclose(self.accuracy, expected_accuracy, abs_tol=1e-12):
            raise ValueError("accuracy does not match feedback counts")
        if not math.isclose(self.coverage, self.support / self.sample_count, abs_tol=1e-12):
            raise ValueError("coverage does not match feedback counts")
        expected_valid = (self.support + self.abstain) / self.sample_count
        if not math.isclose(self.valid_rate, expected_valid, abs_tol=1e-12):
            raise ValueError("valid_rate does not match feedback counts")
        object.__setattr__(self, "errors", tuple(self.errors))
        if any(not isinstance(item, ErrorSampleRef) for item in self.errors):
            raise TypeError("errors must contain ErrorSampleRef values")
        outcome_counts = {
            "wrong": self.wrong,
            "abstain": self.abstain,
            "answer_invalid": self.answer_invalid,
            "parse_invalid": self.parse_invalid,
        }
        if any(sum(item.outcome == outcome for item in self.errors) != count for outcome, count in outcome_counts.items()):
            raise ValueError("errors do not match feedback outcome counts")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION,
            "node_id": self.node_id,
            "criterion_name": self.criterion_name,
            "sample_count": self.sample_count,
            "support": self.support,
            "correct": self.correct,
            "wrong": self.wrong,
            "abstain": self.abstain,
            "answer_invalid": self.answer_invalid,
            "parse_invalid": self.parse_invalid,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "valid_rate": self.valid_rate,
            "description_length": self.description_length,
            "fitness": self.fitness,
            "errors": [item.to_dict() for item in self.errors],
        }

    @classmethod
    def from_dict(cls, value: object) -> "NodeFeedback":
        fields = {
            "node_id", "criterion_name", "sample_count", "support", "correct", "wrong",
            "abstain", "answer_invalid", "parse_invalid", "accuracy", "coverage",
            "valid_rate", "description_length", "fitness", "errors",
        }
        value = _versioned(value, fields, "node feedback")
        if not isinstance(value["errors"], list):
            raise ValueError("node feedback errors must be a list")
        kwargs = {key: value[key] for key in fields - {"errors"}}
        return cls(**kwargs, errors=tuple(ErrorSampleRef.from_dict(item) for item in value["errors"]))


@dataclass(frozen=True)
class RubricFeedback:
    rubric_sha256: str
    sample_ids: tuple[str, ...]
    nodes: Mapping[str, NodeFeedback]
    agreement_by_pair: Mapping[str, float]
    rubric_gap_sample_ids: tuple[str, ...]
    cascade_failure_sample_ids: tuple[str, ...]
    aggregation_conflict_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.rubric_sha256, "rubric_sha256")
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("feedback sample IDs must be non-empty and unique")
        copied_nodes = dict(self.nodes)
        if not copied_nodes or any(not isinstance(value, NodeFeedback) for value in copied_nodes.values()):
            raise ValueError("nodes must contain NodeFeedback values")
        if any(key != value.node_id for key, value in copied_nodes.items()):
            raise ValueError("node feedback mapping keys must equal node IDs")
        if any(value.sample_count != len(self.sample_ids) for value in copied_nodes.values()):
            raise ValueError("node feedback sample counts must match RubricFeedback")
        object.__setattr__(self, "nodes", MappingProxyType(copied_nodes))
        agreements = dict(self.agreement_by_pair)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in agreements.values()):
            raise ValueError("agreements must be in [0, 1]")
        object.__setattr__(self, "agreement_by_pair", MappingProxyType(agreements))
        for field in (
            "rubric_gap_sample_ids", "cascade_failure_sample_ids",
            "aggregation_conflict_sample_ids",
        ):
            values = tuple(getattr(self, field))
            if len(set(values)) != len(values) or not set(values).issubset(self.sample_ids):
                raise ValueError(f"{field} must be a unique subset of sample_ids")
            object.__setattr__(self, field, values)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION,
            "rubric_sha256": self.rubric_sha256,
            "sample_ids": list(self.sample_ids),
            "nodes": {key: value.to_dict() for key, value in sorted(self.nodes.items())},
            "agreement_by_pair": dict(sorted(self.agreement_by_pair.items())),
            "rubric_gap_sample_ids": list(self.rubric_gap_sample_ids),
            "cascade_failure_sample_ids": list(self.cascade_failure_sample_ids),
            "aggregation_conflict_sample_ids": list(self.aggregation_conflict_sample_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RubricFeedback":
        fields = {
            "rubric_sha256", "sample_ids", "nodes", "agreement_by_pair",
            "rubric_gap_sample_ids", "cascade_failure_sample_ids",
            "aggregation_conflict_sample_ids",
        }
        value = _versioned(value, fields, "rubric feedback")
        if not isinstance(value["sample_ids"], list) or not isinstance(value["nodes"], Mapping):
            raise ValueError("rubric feedback sample_ids/nodes have invalid types")
        if not isinstance(value["agreement_by_pair"], Mapping):
            raise ValueError("agreement_by_pair must be an object")
        for name in ("rubric_gap_sample_ids", "cascade_failure_sample_ids", "aggregation_conflict_sample_ids"):
            if not isinstance(value[name], list):
                raise ValueError(f"{name} must be a list")
        return cls(
            rubric_sha256=value["rubric_sha256"],
            sample_ids=tuple(value["sample_ids"]),
            nodes={key: NodeFeedback.from_dict(item) for key, item in value["nodes"].items()},
            agreement_by_pair=dict(value["agreement_by_pair"]),
            rubric_gap_sample_ids=tuple(value["rubric_gap_sample_ids"]),
            cascade_failure_sample_ids=tuple(value["cascade_failure_sample_ids"]),
            aggregation_conflict_sample_ids=tuple(value["aggregation_conflict_sample_ids"]),
        )


@dataclass(frozen=True)
class RubricPatch:
    base_rubric_sha256: str
    upsert_nodes: tuple[RubricNode, ...] = ()
    remove_node_ids: tuple[str, ...] = ()
    add_edges: tuple[RubricEdge, ...] = ()
    remove_edges: tuple[RubricEdge, ...] = ()
    root_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _sha(self.base_rubric_sha256, "base_rubric_sha256")
        for field in ("upsert_nodes", "remove_node_ids", "add_edges", "remove_edges"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if any(not isinstance(node, RubricNode) for node in self.upsert_nodes):
            raise TypeError("upsert_nodes must contain RubricNode values")
        if any(not isinstance(edge, RubricEdge) for edge in self.add_edges + self.remove_edges):
            raise TypeError("patch edges must contain RubricEdge values")
        for node_id in self.remove_node_ids:
            _clean(node_id, "remove node ID")
        if len({node.node_id for node in self.upsert_nodes}) != len(self.upsert_nodes):
            raise ValueError("upsert_nodes contains duplicate node IDs")
        if len(set(self.remove_node_ids)) != len(self.remove_node_ids):
            raise ValueError("remove_node_ids contains duplicates")
        if set(self.remove_node_ids) & {node.node_id for node in self.upsert_nodes}:
            raise ValueError("a node cannot be removed and upserted in one patch")
        if self.root_ids is not None:
            object.__setattr__(self, "root_ids", tuple(self.root_ids))
            for root_id in self.root_ids:
                _clean(root_id, "root ID")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION,
            "base_rubric_sha256": self.base_rubric_sha256,
            "upsert_nodes": [node.to_dict() for node in self.upsert_nodes],
            "remove_node_ids": list(self.remove_node_ids),
            "add_edges": [edge.to_dict() for edge in self.add_edges],
            "remove_edges": [edge.to_dict() for edge in self.remove_edges],
            "root_ids": None if self.root_ids is None else list(self.root_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RubricPatch":
        fields = {"base_rubric_sha256", "upsert_nodes", "remove_node_ids", "add_edges", "remove_edges", "root_ids"}
        value = _versioned(value, fields, "rubric patch")
        for name in ("upsert_nodes", "remove_node_ids", "add_edges", "remove_edges"):
            if not isinstance(value[name], list):
                raise ValueError(f"rubric patch {name} must be a list")
        if value["root_ids"] is not None and not isinstance(value["root_ids"], list):
            raise ValueError("rubric patch root_ids must be a list or null")
        return cls(
            base_rubric_sha256=value["base_rubric_sha256"],
            upsert_nodes=tuple(RubricNode.from_dict(item) for item in value["upsert_nodes"]),
            remove_node_ids=tuple(value["remove_node_ids"]),
            add_edges=tuple(RubricEdge.from_dict(item) for item in value["add_edges"]),
            remove_edges=tuple(RubricEdge.from_dict(item) for item in value["remove_edges"]),
            root_ids=None if value["root_ids"] is None else tuple(value["root_ids"]),
        )


@dataclass(frozen=True)
class EditCandidate:
    candidate_id: str
    operator: OperatorKind
    rationale: str
    patch: RubricPatch

    def __post_init__(self) -> None:
        _clean(self.candidate_id, "candidate_id")
        _clean(self.rationale, "rationale")
        if not isinstance(self.operator, OperatorKind):
            raise TypeError("operator must be OperatorKind")
        if not isinstance(self.patch, RubricPatch):
            raise TypeError("patch must be RubricPatch")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION, "candidate_id": self.candidate_id,
                "operator": self.operator.value, "rationale": self.rationale, "patch": self.patch.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "EditCandidate":
        value = _versioned(value, {"candidate_id", "operator", "rationale", "patch"}, "edit candidate")
        return cls(value["candidate_id"], OperatorKind(value["operator"]), value["rationale"], RubricPatch.from_dict(value["patch"]))


@dataclass(frozen=True)
class EvolutionContext:
    rubric: StructuredRubric
    feedback: RubricFeedback

    def __post_init__(self) -> None:
        if self.feedback.rubric_sha256 != self.rubric.rubric_sha256:
            raise ValueError("feedback does not belong to rubric")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION, "rubric": self.rubric.to_dict(), "feedback": self.feedback.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "EvolutionContext":
        value = _versioned(value, {"rubric", "feedback"}, "evolution context")
        return cls(StructuredRubric.from_dict(value["rubric"]), RubricFeedback.from_dict(value["feedback"]))


@dataclass(frozen=True)
class RubricDiff:
    added_node_ids: tuple[str, ...]
    removed_node_ids: tuple[str, ...]
    modified_node_ids: tuple[str, ...]
    added_edges: tuple[RubricEdge, ...]
    removed_edges: tuple[RubricEdge, ...]
    root_ids_changed: bool

    def __post_init__(self) -> None:
        for name in ("added_node_ids", "removed_node_ids", "modified_node_ids", "added_edges", "removed_edges"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.root_ids_changed, bool):
            raise TypeError("root_ids_changed must be bool")

    @property
    def is_noop(self) -> bool:
        return not (
            self.added_node_ids or self.removed_node_ids or self.modified_node_ids
            or self.added_edges or self.removed_edges or self.root_ids_changed
        )

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION,
                "added_node_ids": list(self.added_node_ids), "removed_node_ids": list(self.removed_node_ids),
                "modified_node_ids": list(self.modified_node_ids), "added_edges": [edge.to_dict() for edge in self.added_edges],
                "removed_edges": [edge.to_dict() for edge in self.removed_edges], "root_ids_changed": self.root_ids_changed}

    @classmethod
    def from_dict(cls, value: object) -> "RubricDiff":
        fields = {"added_node_ids", "removed_node_ids", "modified_node_ids", "added_edges", "removed_edges", "root_ids_changed"}
        value = _versioned(value, fields, "rubric diff")
        for name in fields - {"root_ids_changed"}:
            if not isinstance(value[name], list):
                raise ValueError(f"rubric diff {name} must be a list")
        if not isinstance(value["root_ids_changed"], bool):
            raise ValueError("root_ids_changed must be bool")
        return cls(tuple(value["added_node_ids"]), tuple(value["removed_node_ids"]), tuple(value["modified_node_ids"]),
                   tuple(RubricEdge.from_dict(item) for item in value["added_edges"]),
                   tuple(RubricEdge.from_dict(item) for item in value["removed_edges"]), value["root_ids_changed"])


@dataclass(frozen=True)
class ArtifactRefreshPlan:
    refresh_pairwise_node_ids: tuple[str, ...]
    remove_pairwise_criterion_names: tuple[str, ...]
    refresh_gate_node_ids: tuple[str, ...]
    remove_gate_criterion_names: tuple[str, ...]
    router_stale: bool

    def __post_init__(self) -> None:
        for name in (
            "refresh_pairwise_node_ids", "remove_pairwise_criterion_names",
            "refresh_gate_node_ids", "remove_gate_criterion_names",
        ):
            values = tuple(getattr(self, name))
            for value in values:
                _clean(value, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
            object.__setattr__(self, name, values)
        if not isinstance(self.router_stale, bool):
            raise TypeError("router_stale must be bool")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION,
                "refresh_pairwise_node_ids": list(self.refresh_pairwise_node_ids),
                "remove_pairwise_criterion_names": list(self.remove_pairwise_criterion_names),
                "refresh_gate_node_ids": list(self.refresh_gate_node_ids),
                "remove_gate_criterion_names": list(self.remove_gate_criterion_names),
                "router_stale": self.router_stale}

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactRefreshPlan":
        fields = {"refresh_pairwise_node_ids", "remove_pairwise_criterion_names", "refresh_gate_node_ids", "remove_gate_criterion_names", "router_stale"}
        value = _versioned(value, fields, "artifact refresh plan")
        for name in fields - {"router_stale"}:
            if not isinstance(value[name], list):
                raise ValueError(f"artifact refresh plan {name} must be a list")
        if not isinstance(value["router_stale"], bool):
            raise ValueError("router_stale must be bool")
        return cls(tuple(value["refresh_pairwise_node_ids"]), tuple(value["remove_pairwise_criterion_names"]),
                   tuple(value["refresh_gate_node_ids"]), tuple(value["remove_gate_criterion_names"]), value["router_stale"])


@dataclass(frozen=True)
class CandidateAcceptancePolicy:
    min_accuracy_delta: float
    min_valid_rate: float

    def __post_init__(self) -> None:
        for name in ("min_accuracy_delta", "min_valid_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.min_accuracy_delta < 0 or not 0.0 <= self.min_valid_rate <= 1.0:
            raise ValueError("invalid acceptance guardrail")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION, "min_accuracy_delta": self.min_accuracy_delta,
                "min_valid_rate": self.min_valid_rate}

    @classmethod
    def from_dict(cls, value: object) -> "CandidateAcceptancePolicy":
        fields = {"min_accuracy_delta", "min_valid_rate"}
        value = _versioned(value, fields, "candidate acceptance policy")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class CandidateEvaluation:
    before_accuracy: float
    after_accuracy: float
    accuracy_delta: float
    before_coverage: float
    after_coverage: float
    corrected_count: int
    harmed_count: int
    before_valid_rate: float
    final_valid_rate: float
    decision: EvolutionDecision
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "before_accuracy", "after_accuracy", "before_coverage", "after_coverage",
            "before_valid_rate", "final_valid_rate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.accuracy_delta) or not math.isclose(
            self.accuracy_delta, self.after_accuracy - self.before_accuracy, abs_tol=1e-12
        ):
            raise ValueError("accuracy_delta does not match before/after accuracy")
        for name in ("corrected_count", "harmed_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.decision, EvolutionDecision):
            raise TypeError("decision must be EvolutionDecision")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if any(not isinstance(reason, str) or not reason for reason in self.reasons):
            raise ValueError("reasons must contain non-empty strings")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": STRUCTURED_EVOLUTION_SCHEMA_VERSION,
                "before_accuracy": self.before_accuracy, "after_accuracy": self.after_accuracy,
                "accuracy_delta": self.accuracy_delta, "before_coverage": self.before_coverage,
                "after_coverage": self.after_coverage, "corrected_count": self.corrected_count,
                "harmed_count": self.harmed_count, "before_valid_rate": self.before_valid_rate,
                "final_valid_rate": self.final_valid_rate, "decision": self.decision.value, "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, value: object) -> "CandidateEvaluation":
        fields = {"before_accuracy", "after_accuracy", "accuracy_delta", "before_coverage", "after_coverage",
                  "corrected_count", "harmed_count", "before_valid_rate", "final_valid_rate", "decision", "reasons"}
        value = _versioned(value, fields, "candidate evaluation")
        if not isinstance(value["reasons"], list):
            raise ValueError("candidate evaluation reasons must be a list")
        kwargs = {key: value[key] for key in fields - {"decision", "reasons"}}
        return cls(**kwargs, decision=EvolutionDecision(value["decision"]), reasons=tuple(value["reasons"]))


def evaluate_candidate(
    before: Sequence[FinalPreference],
    after: Sequence[FinalPreference],
    gold: Sequence[str],
    *,
    before_valid_rate: float,
    final_valid_rate: float,
    policy: CandidateAcceptancePolicy,
) -> CandidateEvaluation:
    """Apply an explicit, caller-frozen paired M1 acceptance policy."""

    if not before or len(before) != len(after) or len(before) != len(gold):
        raise ValueError("before, after and gold must have the same non-zero length")
    if any(value not in {"A", "B"} for value in gold):
        raise ValueError("gold values must be A or B")
    if not 0.0 <= before_valid_rate <= 1.0 or not 0.0 <= final_valid_rate <= 1.0:
        raise ValueError("valid rates must be in [0, 1]")
    before_correct = [value.value == target for value, target in zip(before, gold)]
    after_correct = [value.value == target for value, target in zip(after, gold)]
    before_decisive = [value in {FinalPreference.A, FinalPreference.B} for value in before]
    after_decisive = [value in {FinalPreference.A, FinalPreference.B} for value in after]
    count = len(gold)
    before_accuracy = sum(before_correct) / count
    after_accuracy = sum(after_correct) / count
    before_coverage = sum(before_decisive) / count
    after_coverage = sum(after_decisive) / count
    corrected = sum(not old and new for old, new in zip(before_correct, after_correct))
    harmed = sum(old and not new for old, new in zip(before_correct, after_correct))
    reasons: list[str] = []
    if after_accuracy - before_accuracy < policy.min_accuracy_delta:
        reasons.append("accuracy_delta_below_threshold")
    if final_valid_rate < policy.min_valid_rate:
        reasons.append("final_valid_rate_below_guardrail")
    return CandidateEvaluation(
        before_accuracy, after_accuracy, after_accuracy - before_accuracy,
        before_coverage, after_coverage, corrected, harmed, before_valid_rate, final_valid_rate,
        EvolutionDecision.REJECT if reasons else EvolutionDecision.ACCEPT,
        tuple(reasons),
    )

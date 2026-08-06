"""Immutable schemas for the Phase 6B Specialize operator."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ..telemetry import ModelCallMetrics
from ..version import SPECIALIZE_SCHEMA_VERSION
from .types import CandidateEvaluation, EditCandidate


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
    if value["schema_version"] != SPECIALIZE_SCHEMA_VERSION:
        raise ValueError(f"{label} schema version mismatch")
    return value


def _string_tuple(values: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(_clean(value, label) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    return result


def _freeze_json(value: Any, label: str = "metadata") -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains non-finite float")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({
            _clean(key, f"{label} key"): _freeze_json(item, f"{label}.{key}")
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{label}[]") for item in value)
    raise ValueError(f"{label} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class SpecializeManagerRequestSpec:
    model: str
    backend_id: str
    prompt_sha256: str
    decoding_config: Mapping[str, Any]
    prompt_version: str
    parser_version: str

    def __post_init__(self) -> None:
        for field in ("model", "backend_id", "prompt_version", "parser_version"):
            _clean(getattr(self, field), field)
        if not re.fullmatch(r"[0-9a-f]{64}", self.prompt_sha256):
            raise ValueError("prompt_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "decoding_config", _freeze_json(self.decoding_config,
                                                                  "decoding_config"))

    @classmethod
    def from_prompt(cls, *, model: str, backend_id: str, prompt: str,
                    decoding_config: Mapping[str, Any], prompt_version: str,
                    parser_version: str) -> "SpecializeManagerRequestSpec":
        return cls(model, backend_id, hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                   decoding_config, prompt_version, parser_version)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION, "model": self.model,
                "backend_id": self.backend_id, "prompt_sha256": self.prompt_sha256,
                "decoding_config": _thaw_json(self.decoding_config),
                "prompt_version": self.prompt_version, "parser_version": self.parser_version}

    @classmethod
    def from_dict(cls, value: object) -> "SpecializeManagerRequestSpec":
        value = _versioned(value, {"model", "backend_id", "prompt_sha256", "decoding_config",
                                   "prompt_version", "parser_version"}, "manager request spec")
        if not isinstance(value["decoding_config"], Mapping):
            raise ValueError("decoding_config must be an object")
        return cls(value["model"], value["backend_id"], value["prompt_sha256"],
                   value["decoding_config"], value["prompt_version"], value["parser_version"])


@dataclass(frozen=True)
class SpecializeTriggerDecision:
    parent_node_id: str
    triggered: bool
    reasons: tuple[str, ...]
    decisive_wrong_sample_ids: tuple[str, ...]
    current_children: int
    remaining_capacity: int

    def __post_init__(self) -> None:
        _clean(self.parent_node_id, "parent_node_id")
        if not isinstance(self.triggered, bool):
            raise TypeError("triggered must be bool")
        object.__setattr__(self, "reasons", _string_tuple(self.reasons, "reasons", allow_empty=True))
        object.__setattr__(self, "decisive_wrong_sample_ids",
                           _string_tuple(self.decisive_wrong_sample_ids, "wrong sample IDs", allow_empty=True))
        for name in ("current_children", "remaining_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.triggered == bool(self.reasons):
            raise ValueError("triggered decision and reasons conflict")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION, "parent_node_id": self.parent_node_id,
                "triggered": self.triggered, "reasons": list(self.reasons),
                "decisive_wrong_sample_ids": list(self.decisive_wrong_sample_ids),
                "current_children": self.current_children,
                "remaining_capacity": self.remaining_capacity}

    @classmethod
    def from_dict(cls, value: object) -> "SpecializeTriggerDecision":
        value = _versioned(value, {"parent_node_id", "triggered", "reasons",
                                   "decisive_wrong_sample_ids", "current_children",
                                   "remaining_capacity"}, "specialize trigger")
        if not isinstance(value["reasons"], list) or not isinstance(value["decisive_wrong_sample_ids"], list):
            raise ValueError("specialize trigger sequence fields must be lists")
        return cls(value["parent_node_id"], value["triggered"], tuple(value["reasons"]),
                   tuple(value["decisive_wrong_sample_ids"]), value["current_children"],
                   value["remaining_capacity"])


@dataclass(frozen=True)
class ErrorSignature:
    sample_id: str
    task_pattern: str
    visual_focus: str
    candidate_difference: str
    parent_failure: str
    suggested_subdomain: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _clean(getattr(self, field), field)

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "ErrorSignature":
        fields = set(cls.__dataclass_fields__)
        value = _exact(value, fields, "error signature")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True)
class ErrorSignatureOutput:
    signature: ErrorSignature | None
    raw_response: str | None
    parse_error: str | None
    attempt_count: int
    metrics: ModelCallMetrics
    request_spec: SpecializeManagerRequestSpec

    def __post_init__(self) -> None:
        if self.signature is not None and not isinstance(self.signature, ErrorSignature):
            raise TypeError("signature must be ErrorSignature or None")
        if self.signature is None and not self.parse_error:
            raise ValueError("failed signature output requires parse_error")
        if self.signature is not None and self.parse_error is not None:
            raise ValueError("valid signature output cannot have parse_error")
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be str or None")
        if isinstance(self.attempt_count, bool) or self.attempt_count < 1:
            raise ValueError("attempt_count must be positive")
        if not isinstance(self.metrics, ModelCallMetrics) or not isinstance(self.request_spec, SpecializeManagerRequestSpec):
            raise TypeError("signature output metrics/request_spec have invalid types")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION,
                "signature": None if self.signature is None else self.signature.to_dict(),
                "raw_response": self.raw_response, "parse_error": self.parse_error,
                "attempt_count": self.attempt_count, "metrics": self.metrics.to_dict(),
                "request_spec": self.request_spec.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "ErrorSignatureOutput":
        value = _versioned(value, {"signature", "raw_response", "parse_error", "attempt_count",
                                   "metrics", "request_spec"}, "error signature output")
        return cls(None if value["signature"] is None else ErrorSignature.from_dict(value["signature"]),
                   value["raw_response"], value["parse_error"], value["attempt_count"],
                   ModelCallMetrics.from_dict(value["metrics"]),
                   SpecializeManagerRequestSpec.from_dict(value["request_spec"]))


@dataclass(frozen=True)
class SemanticCluster:
    cluster_id: str
    label: str
    shared_failure: str
    distinction: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("cluster_id", "label", "shared_failure", "distinction"):
            _clean(getattr(self, field), field)
        object.__setattr__(self, "sample_ids", _string_tuple(self.sample_ids, "cluster sample IDs"))

    def to_dict(self) -> dict[str, Any]:
        return {"cluster_id": self.cluster_id, "label": self.label,
                "shared_failure": self.shared_failure, "distinction": self.distinction,
                "sample_ids": list(self.sample_ids)}

    @classmethod
    def from_dict(cls, value: object) -> "SemanticCluster":
        value = _exact(value, {"cluster_id", "label", "shared_failure", "distinction", "sample_ids"},
                       "semantic cluster")
        if not isinstance(value["sample_ids"], list):
            raise ValueError("semantic cluster sample_ids must be a list")
        return cls(value["cluster_id"], value["label"], value["shared_failure"],
                   value["distinction"], tuple(value["sample_ids"]))


@dataclass(frozen=True)
class ClusterProposal:
    clusters: tuple[SemanticCluster, ...]
    unclustered_sample_ids: tuple[str, ...]
    raw_response: str
    attempt_count: int
    metrics: ModelCallMetrics
    request_spec: SpecializeManagerRequestSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "unclustered_sample_ids",
                           _string_tuple(self.unclustered_sample_ids, "unclustered IDs", allow_empty=True))
        if any(not isinstance(item, SemanticCluster) for item in self.clusters):
            raise TypeError("clusters must contain SemanticCluster values")
        if len({item.cluster_id for item in self.clusters}) != len(self.clusters):
            raise ValueError("cluster IDs must be unique")
        _clean(self.raw_response, "raw_response")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be positive")
        if not isinstance(self.metrics, ModelCallMetrics) or not isinstance(self.request_spec, SpecializeManagerRequestSpec):
            raise TypeError("cluster metrics/request_spec have invalid types")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION,
                "clusters": [item.to_dict() for item in self.clusters],
                "unclustered_sample_ids": list(self.unclustered_sample_ids),
                "raw_response": self.raw_response, "attempt_count": self.attempt_count,
                "metrics": self.metrics.to_dict(), "request_spec": self.request_spec.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "ClusterProposal":
        value = _versioned(value, {"clusters", "unclustered_sample_ids", "raw_response",
                                   "attempt_count", "metrics", "request_spec"}, "cluster proposal")
        if not isinstance(value["clusters"], list) or not isinstance(value["unclustered_sample_ids"], list):
            raise ValueError("cluster proposal sequence fields must be lists")
        return cls(tuple(SemanticCluster.from_dict(item) for item in value["clusters"]),
                   tuple(value["unclustered_sample_ids"]), value["raw_response"],
                   value["attempt_count"], ModelCallMetrics.from_dict(value["metrics"]),
                   SpecializeManagerRequestSpec.from_dict(value["request_spec"]))


@dataclass(frozen=True)
class ChildCriterionProposal:
    cluster_id: str
    criterion_name: str
    description: str
    rationale: str
    representative_sample_ids: tuple[str, ...]
    raw_response: str
    attempt_count: int
    metrics: ModelCallMetrics
    request_spec: SpecializeManagerRequestSpec

    def __post_init__(self) -> None:
        for field in ("cluster_id", "criterion_name", "description", "rationale", "raw_response"):
            _clean(getattr(self, field), field)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.criterion_name):
            raise ValueError("criterion_name must be lower_snake_case")
        object.__setattr__(self, "representative_sample_ids",
                           _string_tuple(self.representative_sample_ids, "representative sample IDs"))
        if not 1 <= len(self.representative_sample_ids) <= 3:
            raise ValueError("child must cite 1-3 representative samples")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be positive")
        if not isinstance(self.metrics, ModelCallMetrics) or not isinstance(self.request_spec, SpecializeManagerRequestSpec):
            raise TypeError("child metrics/request_spec have invalid types")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION, "cluster_id": self.cluster_id,
                "criterion_name": self.criterion_name, "description": self.description,
                "rationale": self.rationale,
                "representative_sample_ids": list(self.representative_sample_ids),
                "raw_response": self.raw_response, "attempt_count": self.attempt_count,
                "metrics": self.metrics.to_dict(), "request_spec": self.request_spec.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "ChildCriterionProposal":
        fields = {"cluster_id", "criterion_name", "description", "rationale",
                  "representative_sample_ids", "raw_response", "attempt_count", "metrics", "request_spec"}
        value = _versioned(value, fields, "child criterion proposal")
        if not isinstance(value["representative_sample_ids"], list):
            raise ValueError("representative_sample_ids must be a list")
        return cls(value["cluster_id"], value["criterion_name"], value["description"],
                   value["rationale"], tuple(value["representative_sample_ids"]),
                   value["raw_response"], value["attempt_count"],
                   ModelCallMetrics.from_dict(value["metrics"]),
                   SpecializeManagerRequestSpec.from_dict(value["request_spec"]))


@dataclass(frozen=True)
class SpecializeCandidate:
    parent_node_id: str
    edit_candidate: EditCandidate
    cluster_proposal: ClusterProposal
    children: tuple[ChildCriterionProposal, ...]
    node_id_by_cluster: Mapping[str, str]

    def __post_init__(self) -> None:
        _clean(self.parent_node_id, "parent_node_id")
        if not isinstance(self.edit_candidate, EditCandidate):
            raise TypeError("edit_candidate must be EditCandidate")
        if not isinstance(self.cluster_proposal, ClusterProposal):
            raise TypeError("cluster_proposal must be ClusterProposal")
        object.__setattr__(self, "children", tuple(self.children))
        if any(not isinstance(item, ChildCriterionProposal) for item in self.children):
            raise TypeError("children must contain ChildCriterionProposal values")
        mapping = dict(self.node_id_by_cluster)
        if set(mapping) != {item.cluster_id for item in self.children}:
            raise ValueError("node_id_by_cluster must cover all children")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("generated node IDs must be unique")
        for key, value in mapping.items():
            _clean(key, "cluster ID"); _clean(value, "node ID")
        object.__setattr__(self, "node_id_by_cluster", MappingProxyType(mapping))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION, "parent_node_id": self.parent_node_id,
                "edit_candidate": self.edit_candidate.to_dict(),
                "cluster_proposal": self.cluster_proposal.to_dict(),
                "children": [item.to_dict() for item in self.children],
                "node_id_by_cluster": dict(self.node_id_by_cluster)}

    @classmethod
    def from_dict(cls, value: object) -> "SpecializeCandidate":
        value = _versioned(value, {"parent_node_id", "edit_candidate", "cluster_proposal",
                                   "children", "node_id_by_cluster"}, "specialize candidate")
        if not isinstance(value["children"], list) or not isinstance(value["node_id_by_cluster"], Mapping):
            raise ValueError("specialize candidate children/mapping have invalid types")
        return cls(value["parent_node_id"], EditCandidate.from_dict(value["edit_candidate"]),
                   ClusterProposal.from_dict(value["cluster_proposal"]),
                   tuple(ChildCriterionProposal.from_dict(item) for item in value["children"]),
                   value["node_id_by_cluster"])


@dataclass(frozen=True)
class ChildDiagnostic:
    node_id: str
    criterion_name: str
    support: int
    accuracy: float
    coverage: float
    valid_rate: float
    fitness: float
    cluster_support: int
    cluster_accuracy: float
    non_target_decisive: int
    non_target_wrong: int
    non_target_abstain: int
    corrected_count: int
    harmed_count: int
    leave_one_out_m1_delta: float

    def __post_init__(self) -> None:
        _clean(self.node_id, "node_id"); _clean(self.criterion_name, "criterion_name")
        for name in ("support", "cluster_support", "non_target_decisive", "non_target_wrong",
                     "non_target_abstain", "corrected_count", "harmed_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("accuracy", "coverage", "valid_rate", "cluster_accuracy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.fitness) or not math.isfinite(self.leave_one_out_m1_delta):
            raise ValueError("diagnostic fitness/delta must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION, **self.__dict__}

    @classmethod
    def from_dict(cls, value: object) -> "ChildDiagnostic":
        fields = set(cls.__dataclass_fields__)
        value = _versioned(value, fields, "child diagnostic")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True)
class SubtreeDiagnostic:
    parent_accuracy: float
    parent_coverage: float
    specialized_accuracy: float
    specialized_coverage: float
    corrected_sample_ids: tuple[str, ...]
    harmed_sample_ids: tuple[str, ...]
    sibling_agreement: float
    sibling_conflict_count: int
    subtree_conflict_count: int
    mechanism_risks: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("parent_accuracy", "parent_coverage", "specialized_accuracy",
                     "specialized_coverage", "sibling_agreement"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("sibling_conflict_count", "subtree_conflict_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("corrected_sample_ids", "harmed_sample_ids", "mechanism_risks"):
            object.__setattr__(self, name, _string_tuple(getattr(self, name), name, allow_empty=True))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION,
                **{key: list(value) if isinstance(value, tuple) else value
                   for key, value in self.__dict__.items()}}

    @classmethod
    def from_dict(cls, value: object) -> "SubtreeDiagnostic":
        fields = set(cls.__dataclass_fields__)
        value = _versioned(value, fields, "subtree diagnostic")
        for field in {"corrected_sample_ids", "harmed_sample_ids", "mechanism_risks"}:
            if not isinstance(value[field], list):
                raise ValueError(f"{field} must be a list")
        kwargs = {field: tuple(value[field]) if field in {"corrected_sample_ids", "harmed_sample_ids", "mechanism_risks"}
                  else value[field] for field in fields}
        return cls(**kwargs)


@dataclass(frozen=True)
class SpecializeEvaluation:
    candidate_evaluation: CandidateEvaluation
    child_diagnostics: tuple[ChildDiagnostic, ...]
    subtree_diagnostic: SubtreeDiagnostic
    new_children_final_valid_rate: float
    parent_fitness: float
    collective_fitness: float
    fitness_delta: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_evaluation, CandidateEvaluation):
            raise TypeError("candidate_evaluation must be CandidateEvaluation")
        object.__setattr__(self, "child_diagnostics", tuple(self.child_diagnostics))
        if not self.child_diagnostics or any(not isinstance(item, ChildDiagnostic) for item in self.child_diagnostics):
            raise ValueError("child_diagnostics must contain ChildDiagnostic values")
        if not isinstance(self.subtree_diagnostic, SubtreeDiagnostic):
            raise TypeError("subtree_diagnostic must be SubtreeDiagnostic")
        if not math.isfinite(self.new_children_final_valid_rate) or not 0 <= self.new_children_final_valid_rate <= 1:
            raise ValueError("new_children_final_valid_rate must be in [0, 1]")
        for name in ("parent_fitness", "collective_fitness", "fitness_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not math.isclose(self.fitness_delta, self.collective_fitness - self.parent_fitness,
                            abs_tol=1e-12):
            raise ValueError("fitness_delta does not match collective-parent fitness")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SPECIALIZE_SCHEMA_VERSION,
                "candidate_evaluation": self.candidate_evaluation.to_dict(),
                "child_diagnostics": [item.to_dict() for item in self.child_diagnostics],
                "subtree_diagnostic": self.subtree_diagnostic.to_dict(),
                "new_children_final_valid_rate": self.new_children_final_valid_rate,
                "parent_fitness": self.parent_fitness,
                "collective_fitness": self.collective_fitness,
                "fitness_delta": self.fitness_delta}

    @classmethod
    def from_dict(cls, value: object) -> "SpecializeEvaluation":
        value = _versioned(value, {"candidate_evaluation", "child_diagnostics",
                                   "subtree_diagnostic", "new_children_final_valid_rate",
                                   "parent_fitness", "collective_fitness", "fitness_delta"},
                           "specialize evaluation")
        if not isinstance(value["child_diagnostics"], list):
            raise ValueError("child_diagnostics must be a list")
        return cls(CandidateEvaluation.from_dict(value["candidate_evaluation"]),
                   tuple(ChildDiagnostic.from_dict(item) for item in value["child_diagnostics"]),
                   SubtreeDiagnostic.from_dict(value["subtree_diagnostic"]),
                   value["new_children_final_valid_rate"], value["parent_fitness"],
                   value["collective_fitness"], value["fitness_delta"])

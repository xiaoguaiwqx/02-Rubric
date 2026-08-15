"""Versioned outputs for decoupled pairwise voting and gate-state judging."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..utils import parse_json
from .aggregation import aggregate_flat_votes
from .judgement import Applicability, CriterionStatus, FinalPreference, Vote
from .version import (
    GATE_WORKER_PARSER_VERSION,
    GATE_WORKER_PROMPT_VERSION,
    GATE_WORKER_PROMPT_V2_1_PILOT_VERSION,
    GATE_WORKER_PROMPT_V2_PILOT_VERSION,
    GATE_WORKER_SCHEMA_VERSION,
    PAIRWISE_WORKER_PARSER_VERSION,
    PAIRWISE_WORKER_PROMPT_VERSION,
    PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    PAIRWISE_WORKER_SCHEMA_VERSION,
    STRUCTURED_SEMANTICS_VERSION,
)
from .worker_output import StructuredCriterionSnapshot


class PairwiseVoteParseError(ValueError):
    """The legacy-compatible response did not contain a usable answer."""


class GateStateParseError(ValueError):
    """The gate response did not satisfy its strict three-field schema."""


def _exact(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _freeze_json(value: Any, label: str = "metadata") -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys must be strings")
            result[key] = _freeze_json(item, f"{label}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{label}[]") for item in value)
    raise ValueError(f"{label} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def worker_prompt_sha256(prompt: str, postfix: str) -> str:
    if not isinstance(prompt, str) or not isinstance(postfix, str):
        raise TypeError("prompt and postfix must be strings")
    return hashlib.sha256((prompt + postfix).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParsedPairwiseVote:
    """The exact legacy parser result before retry/output wrapping."""

    vote: Vote
    answer_valid: bool
    thought: str


@dataclass(frozen=True)
class PairwiseVoteOutput:
    """One exp4-compatible A/B/None judgement with failure provenance."""

    vote: Vote
    parse_ok: bool
    raw_response: str | None
    parse_error: str | None
    attempt_count: int
    thought: str | None = None
    answer_valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.vote, Vote):
            raise TypeError("vote must be Vote")
        if not isinstance(self.parse_ok, bool):
            raise TypeError("parse_ok must be bool")
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be str or None")
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            raise TypeError("parse_error must be str or None")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int) or self.attempt_count < 1:
            raise ValueError("attempt_count must be a positive integer")
        if self.thought is not None and not isinstance(self.thought, str):
            raise TypeError("thought must be str or None")
        if not isinstance(self.answer_valid, bool):
            raise TypeError("answer_valid must be bool")
        if self.parse_ok and self.parse_error is not None:
            raise ValueError("valid pairwise output cannot contain parse_error")
        if not self.parse_ok:
            if self.vote is not Vote.ABSTAIN:
                raise ValueError("pairwise parse failure must project to abstain")
            if not self.parse_error:
                raise ValueError("pairwise parse failure must preserve parse_error")
            if self.answer_valid:
                raise ValueError("pairwise parse failure cannot have a valid answer")
            if self.thought is not None:
                raise ValueError("pairwise parse failure cannot preserve unparsed thought")
        elif self.thought is None:
            raise ValueError("parsed pairwise output must preserve thought")
        if not self.answer_valid and self.vote is not Vote.ABSTAIN:
            raise ValueError("invalid legacy answer must project to abstain")

    @property
    def is_model_abstain(self) -> bool:
        return self.parse_ok and self.answer_valid and self.vote is Vote.ABSTAIN

    def to_dict(self) -> dict[str, Any]:
        return {"vote": self.vote.value, "parse_ok": self.parse_ok, "raw_response": self.raw_response,
                "parse_error": self.parse_error, "attempt_count": self.attempt_count,
                "thought": self.thought, "answer_valid": self.answer_valid}

    @classmethod
    def from_dict(cls, value: object) -> "PairwiseVoteOutput":
        value = _exact(value, {"vote", "parse_ok", "raw_response", "parse_error", "attempt_count",
                               "thought", "answer_valid"}, "pairwise output")
        return cls(vote=Vote(value["vote"]), parse_ok=value["parse_ok"], raw_response=value["raw_response"],
                   parse_error=value["parse_error"], attempt_count=value["attempt_count"],
                   thought=value["thought"], answer_valid=value["answer_valid"])


def parse_pairwise_vote_response(raw_response: object) -> ParsedPairwiseVote:
    """Exactly match exp4 field parsing; invalid answer tokens do not raise."""

    if not isinstance(raw_response, str):
        raise PairwiseVoteParseError("pairwise response must be a string")
    try:
        value = parse_json(raw_response)
        if not isinstance(value, dict):
            raise ValueError("response JSON must be an object")
        answer = value["answer"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")
        token = answer.strip()[0].upper()
        thought = value["thought"]
        if not isinstance(thought, str):
            raise ValueError("thought must be a string")
        thought = thought.strip()
    except Exception as exc:
        raise PairwiseVoteParseError(str(exc)) from exc
    if token == "A":
        return ParsedPairwiseVote(Vote.A, True, thought)
    if token == "B":
        return ParsedPairwiseVote(Vote.B, True, thought)
    if token in {"N", "U"}:
        return ParsedPairwiseVote(Vote.ABSTAIN, True, thought)
    # This mirrors the old evaluator: schema/field parsing succeeded, then an
    # unsupported token became an invalid result without triggering retry.
    return ParsedPairwiseVote(Vote.ABSTAIN, False, thought)


@dataclass(frozen=True)
class GateJudgement:
    applicable: Applicability
    status_a: CriterionStatus
    status_b: CriterionStatus
    parse_ok: bool = True

    @property
    def consistency_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not isinstance(self.parse_ok, bool):
            errors.append("parse_ok must be bool")
        for name, value, kind in (("applicable", self.applicable, Applicability),
                                  ("status_a", self.status_a, CriterionStatus),
                                  ("status_b", self.status_b, CriterionStatus)):
            if not isinstance(value, kind):
                errors.append(f"{name} must be {kind.__name__}")
        if errors:
            return tuple(errors)
        if self.applicable is not Applicability.YES and (
            self.status_a is not CriterionStatus.UNCERTAIN
            or self.status_b is not CriterionStatus.UNCERTAIN
        ):
            errors.append("statuses must be uncertain when applicable is not yes")
        return tuple(errors)

    @property
    def consistency_ok(self) -> bool:
        return not self.consistency_errors

    def to_dict(self) -> dict[str, Any]:
        return {"applicable": self.applicable.value, "status_a": self.status_a.value,
                "status_b": self.status_b.value, "parse_ok": self.parse_ok}

    @classmethod
    def from_dict(cls, value: object) -> "GateJudgement":
        value = _exact(value, {"applicable", "status_a", "status_b", "parse_ok"}, "gate judgement")
        return cls(Applicability(value["applicable"]), CriterionStatus(value["status_a"]),
                   CriterionStatus(value["status_b"]), value["parse_ok"])


@dataclass(frozen=True)
class GateStateOutput:
    judgement: GateJudgement
    raw_response: str | None
    parse_error: str | None
    attempt_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.judgement, GateJudgement):
            raise TypeError("judgement must be GateJudgement")
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be str or None")
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            raise TypeError("parse_error must be str or None")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int) or self.attempt_count < 1:
            raise ValueError("attempt_count must be a positive integer")
        if not self.judgement.parse_ok and not self.parse_error:
            raise ValueError("gate parse failure must preserve parse_error")
        if self.judgement.parse_ok and self.parse_error is not None:
            raise ValueError("parsed gate output cannot contain parse_error")

    @property
    def valid(self) -> bool:
        return self.judgement.parse_ok and self.judgement.consistency_ok

    def to_dict(self) -> dict[str, Any]:
        return {"judgement": self.judgement.to_dict(), "raw_response": self.raw_response,
                "parse_error": self.parse_error, "attempt_count": self.attempt_count}

    @classmethod
    def from_dict(cls, value: object) -> "GateStateOutput":
        value = _exact(value, {"judgement", "raw_response", "parse_error", "attempt_count"}, "gate output")
        return cls(GateJudgement.from_dict(value["judgement"]), value["raw_response"],
                   value["parse_error"], value["attempt_count"])


def parse_gate_state_response(raw_response: object) -> GateJudgement:
    if not isinstance(raw_response, str):
        raise GateStateParseError("gate response must be a string")
    try:
        value = parse_json(raw_response)
    except Exception as exc:
        raise GateStateParseError(str(exc)) from exc
    try:
        value = _exact(value, {"applicable", "status_a", "status_b"}, "gate response")
        if any(not isinstance(value[name], str) for name in value):
            raise ValueError("all gate fields must be strings")
        return GateJudgement(
            Applicability(value["applicable"].strip()),
            CriterionStatus(value["status_a"].strip()),
            CriterionStatus(value["status_b"].strip()),
            True,
        )
    except ValueError as exc:
        raise GateStateParseError(str(exc)) from exc


def make_gate_parse_failure() -> GateJudgement:
    return GateJudgement(Applicability.UNCERTAIN, CriterionStatus.UNCERTAIN,
                         CriterionStatus.UNCERTAIN, False)


@dataclass(frozen=True)
class DualWorkerRequestSpec:
    model: str
    backend_id: str
    prompt_sha256: str
    max_data_chars: int | None
    encode_local_image: bool
    image_field: str
    question_field: str
    sample_id_field: str
    decoding_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("model", "backend_id", "image_field", "question_field", "sample_id_field"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (not isinstance(self.prompt_sha256, str) or len(self.prompt_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.prompt_sha256)):
            raise ValueError("prompt_sha256 must be SHA-256 hex")
        if self.max_data_chars is not None and (isinstance(self.max_data_chars, bool) or not isinstance(self.max_data_chars, int) or self.max_data_chars < 1):
            raise ValueError("max_data_chars must be None or positive")
        if not isinstance(self.encode_local_image, bool):
            raise TypeError("encode_local_image must be bool")
        object.__setattr__(self, "decoding_config", _freeze_json(self.decoding_config, "decoding_config"))

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "backend_id": self.backend_id, "prompt_sha256": self.prompt_sha256,
                "max_data_chars": self.max_data_chars, "encode_local_image": self.encode_local_image,
                "image_field": self.image_field, "question_field": self.question_field,
                "sample_id_field": self.sample_id_field, "decoding_config": _thaw_json(self.decoding_config)}

    @classmethod
    def from_dict(cls, value: object) -> "DualWorkerRequestSpec":
        fields = {"model", "backend_id", "prompt_sha256", "max_data_chars", "encode_local_image",
                  "image_field", "question_field", "sample_id_field", "decoding_config"}
        value = _exact(value, fields, "dual worker request spec")
        return cls(**value)


@dataclass(frozen=True)
class PairwisePredictionOutput:
    sample_ids: tuple[str, ...]
    sample_fingerprints: tuple[str, ...]
    criteria: tuple[StructuredCriterionSnapshot, ...]
    node_outputs: tuple[Mapping[str, PairwiseVoteOutput], ...]
    flat_answers: tuple[FinalPreference, ...]
    request_spec: DualWorkerRequestSpec
    semantics_version: str = STRUCTURED_SEMANTICS_VERSION
    schema_version: str = PAIRWISE_WORKER_SCHEMA_VERSION
    prompt_version: str = PAIRWISE_WORKER_PROMPT_VERSION
    parser_version: str = PAIRWISE_WORKER_PARSER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "sample_fingerprints", tuple(self.sample_fingerprints))
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "node_outputs", tuple(MappingProxyType(dict(row)) for row in self.node_outputs))
        object.__setattr__(self, "flat_answers", tuple(self.flat_answers))
        self._validate()

    def _validate(self) -> None:
        if (self.semantics_version != STRUCTURED_SEMANTICS_VERSION
                or self.schema_version != PAIRWISE_WORKER_SCHEMA_VERSION
                or self.prompt_version not in {
                    PAIRWISE_WORKER_PROMPT_VERSION,
                    PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
                }
                or self.parser_version != PAIRWISE_WORKER_PARSER_VERSION):
            raise ValueError("pairwise artifact version mismatch")
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("pairwise sample IDs must be non-empty and unique")
        if len(self.sample_fingerprints) != len(self.sample_ids) or any(not isinstance(v, str) or len(v) != 64 for v in self.sample_fingerprints):
            raise ValueError("pairwise sample fingerprints are invalid")
        names = tuple(item.name for item in self.criteria)
        if not names or len(set(names)) != len(names):
            raise ValueError("pairwise criteria must be non-empty and unique")
        if len(self.node_outputs) != len(self.sample_ids) or any(set(row) != set(names) for row in self.node_outputs):
            raise ValueError("pairwise output matrix is incomplete")
        if any(not isinstance(item, PairwiseVoteOutput) for row in self.node_outputs for item in row.values()):
            raise TypeError("pairwise output matrix values are invalid")
        if tuple(self.replay_flat_answers()) != self.flat_answers:
            raise ValueError("pairwise flat answers do not match node votes")

    def replay_flat_answers(self) -> tuple[FinalPreference, ...]:
        return tuple(aggregate_flat_votes(output.vote for output in row.values()) for row in self.node_outputs)

    def to_dict(self) -> dict[str, Any]:
        self._validate()
        return {"artifact_type": "pairwise_worker_predictions", "semantics_version": self.semantics_version,
                "schema_version": self.schema_version, "prompt_version": self.prompt_version,
                "parser_version": self.parser_version, "request_spec": self.request_spec.to_dict(),
                "criteria": [item.to_dict() for item in self.criteria],
                "samples": [{"sample_id": sid, "fingerprint": fp,
                             "node_outputs": {name: row[name].to_dict() for name in (c.name for c in self.criteria)},
                             "flat_answer": answer.value}
                            for sid, fp, row, answer in zip(self.sample_ids, self.sample_fingerprints,
                                                           self.node_outputs, self.flat_answers)]}

    @classmethod
    def from_dict(cls, value: object) -> "PairwisePredictionOutput":
        fields = {"artifact_type", "semantics_version", "schema_version", "prompt_version", "parser_version",
                  "request_spec", "criteria", "samples"}
        value = _exact(value, fields, "pairwise artifact")
        if value["artifact_type"] != "pairwise_worker_predictions" or not isinstance(value["criteria"], list) or not isinstance(value["samples"], list):
            raise ValueError("unsupported pairwise artifact")
        criteria = tuple(StructuredCriterionSnapshot.from_dict(item) for item in value["criteria"])
        names = tuple(item.name for item in criteria)
        sample_ids, fps, rows, answers = [], [], [], []
        for sample in value["samples"]:
            sample = _exact(sample, {"sample_id", "fingerprint", "node_outputs", "flat_answer"}, "pairwise sample")
            if not isinstance(sample["node_outputs"], dict) or set(sample["node_outputs"]) != set(names):
                raise ValueError("pairwise sample output matrix is invalid")
            sample_ids.append(sample["sample_id"]); fps.append(sample["fingerprint"])
            rows.append({name: PairwiseVoteOutput.from_dict(sample["node_outputs"][name]) for name in names})
            answers.append(FinalPreference(sample["flat_answer"]))
        return cls(tuple(sample_ids), tuple(fps), criteria, tuple(rows), tuple(answers),
                   DualWorkerRequestSpec.from_dict(value["request_spec"]), value["semantics_version"],
                   value["schema_version"], value["prompt_version"], value["parser_version"])

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "PairwisePredictionOutput":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GatePredictionOutput:
    """Sparse gate outputs: only status-dependent parents need an entry."""

    sample_ids: tuple[str, ...]
    sample_fingerprints: tuple[str, ...]
    criteria: tuple[StructuredCriterionSnapshot, ...]
    node_outputs: tuple[Mapping[str, GateStateOutput], ...]
    request_spec: DualWorkerRequestSpec
    semantics_version: str = STRUCTURED_SEMANTICS_VERSION
    schema_version: str = GATE_WORKER_SCHEMA_VERSION
    prompt_version: str = GATE_WORKER_PROMPT_VERSION
    parser_version: str = GATE_WORKER_PARSER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids)); object.__setattr__(self, "sample_fingerprints", tuple(self.sample_fingerprints))
        object.__setattr__(self, "criteria", tuple(self.criteria)); object.__setattr__(self, "node_outputs", tuple(MappingProxyType(dict(row)) for row in self.node_outputs))
        if (
            self.semantics_version != STRUCTURED_SEMANTICS_VERSION
            or self.schema_version != GATE_WORKER_SCHEMA_VERSION
            or self.prompt_version not in {
                GATE_WORKER_PROMPT_VERSION,
                GATE_WORKER_PROMPT_V2_PILOT_VERSION,
                GATE_WORKER_PROMPT_V2_1_PILOT_VERSION,
            }
            or self.parser_version != GATE_WORKER_PARSER_VERSION
        ):
            raise ValueError("gate artifact version mismatch")
        if (not self.sample_ids or any(not isinstance(value, str) or not value.strip() for value in self.sample_ids)
                or len(set(self.sample_ids)) != len(self.sample_ids)
                or len(self.sample_fingerprints) != len(self.sample_ids)
                or any(not isinstance(value, str) or len(value) != 64 for value in self.sample_fingerprints)
                or len(self.node_outputs) != len(self.sample_ids)):
            raise ValueError("gate artifact sample matrix is invalid")
        names = tuple(item.name for item in self.criteria)
        if not names or len(set(names)) != len(names):
            raise ValueError("gate artifact criteria must be non-empty and unique")
        allowed = set(names)
        if any(not set(row).issubset(allowed) or any(not isinstance(item, GateStateOutput) for item in row.values()) for row in self.node_outputs):
            raise ValueError("gate artifact contains unknown or invalid node outputs")

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_type": "gate_worker_predictions", "semantics_version": self.semantics_version,
                "schema_version": self.schema_version, "prompt_version": self.prompt_version, "parser_version": self.parser_version,
                "request_spec": self.request_spec.to_dict(), "criteria": [item.to_dict() for item in self.criteria],
                "samples": [{"sample_id": sid, "fingerprint": fp, "node_outputs": {k: v.to_dict() for k, v in row.items()}}
                            for sid, fp, row in zip(self.sample_ids, self.sample_fingerprints, self.node_outputs)]}

    @classmethod
    def from_dict(cls, value: object) -> "GatePredictionOutput":
        value = _exact(value, {"artifact_type", "semantics_version", "schema_version", "prompt_version", "parser_version", "request_spec", "criteria", "samples"}, "gate artifact")
        if value["artifact_type"] != "gate_worker_predictions" or not isinstance(value["criteria"], list) or not isinstance(value["samples"], list):
            raise ValueError("unsupported gate artifact")
        criteria = tuple(StructuredCriterionSnapshot.from_dict(item) for item in value["criteria"])
        sample_ids, fps, rows = [], [], []
        for sample in value["samples"]:
            sample = _exact(sample, {"sample_id", "fingerprint", "node_outputs"}, "gate sample")
            if not isinstance(sample["node_outputs"], dict):
                raise ValueError("gate sample outputs must be an object")
            sample_ids.append(sample["sample_id"]); fps.append(sample["fingerprint"])
            rows.append({name: GateStateOutput.from_dict(item) for name, item in sample["node_outputs"].items()})
        return cls(tuple(sample_ids), tuple(fps), criteria, tuple(rows), DualWorkerRequestSpec.from_dict(value["request_spec"]),
                   value["semantics_version"], value["schema_version"], value["prompt_version"], value["parser_version"])

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "GatePredictionOutput":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

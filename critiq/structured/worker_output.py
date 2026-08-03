"""Structured worker parsing, artifacts, replay, and swap diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..utils import parse_json
from .aggregation import aggregate_selected_roots
from .judgement import (
    Applicability,
    CriterionStatus,
    FinalPreference,
    LocalDecision,
    NodeJudgement,
    PairPreference,
    resolve_local_decision,
)
from .version import (
    STRUCTURED_SEMANTICS_VERSION,
    STRUCTURED_WORKER_PARSER_VERSION,
    STRUCTURED_WORKER_PROMPT_VERSION,
    STRUCTURED_WORKER_SCHEMA_VERSION,
)


WORKER_OUTPUT_FIELDS = frozenset(
    {
        "applicable",
        "status_a",
        "status_b",
        "pair_preference",
        "evidence_a",
        "evidence_b",
    }
)


class StructuredOutputParseError(ValueError):
    """Raised when worker text cannot satisfy the structured output schema."""


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 hex string")
    return value


def _freeze_json(value: Any, *, label: str) -> Any:
    """Defensively copy a JSON value into recursively immutable containers."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite floats")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys must be strings")
            frozen[key] = _freeze_json(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{label} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    """Convert recursively frozen JSON containers back to plain JSON values."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise ValueError(f"{label} fields must match exactly: {', '.join(details)}")


def parse_structured_worker_response(raw_response: object) -> NodeJudgement:
    """Parse model text into a judgement without rewriting semantic conflicts."""

    if not isinstance(raw_response, str):
        raise StructuredOutputParseError("worker response must be a string")

    try:
        payload = parse_json(raw_response)
    except Exception as exc:
        raise StructuredOutputParseError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise StructuredOutputParseError("worker response JSON must be an object")

    try:
        _require_exact_keys(payload, WORKER_OUTPUT_FIELDS, label="worker output")
    except ValueError as exc:
        raise StructuredOutputParseError(str(exc)) from exc

    normalized: dict[str, str] = {}
    for field in WORKER_OUTPUT_FIELDS:
        value = payload[field]
        if not isinstance(value, str):
            raise StructuredOutputParseError(f"{field} must be a string")
        normalized[field] = value.strip()

    try:
        return NodeJudgement(
            applicable=Applicability(normalized["applicable"]),
            status_a=CriterionStatus(normalized["status_a"]),
            status_b=CriterionStatus(normalized["status_b"]),
            pair_preference=PairPreference(normalized["pair_preference"]),
            evidence_a=normalized["evidence_a"],
            evidence_b=normalized["evidence_b"],
            parse_ok=True,
        )
    except ValueError as exc:
        raise StructuredOutputParseError(f"invalid enum token: {exc}") from exc


def make_parse_failure_judgement() -> NodeJudgement:
    """Create the only normalized judgement used for exhausted parse failures."""

    return NodeJudgement(
        applicable=Applicability.UNCERTAIN,
        status_a=CriterionStatus.UNCERTAIN,
        status_b=CriterionStatus.UNCERTAIN,
        pair_preference=PairPreference.UNCERTAIN,
        parse_ok=False,
    )


def _judgement_to_dict(judgement: NodeJudgement) -> dict[str, Any]:
    return {
        "applicable": judgement.applicable.value,
        "status_a": judgement.status_a.value,
        "status_b": judgement.status_b.value,
        "pair_preference": judgement.pair_preference.value,
        "evidence_a": judgement.evidence_a,
        "evidence_b": judgement.evidence_b,
        "parse_ok": judgement.parse_ok,
    }


def _judgement_from_dict(value: object) -> NodeJudgement:
    if not isinstance(value, dict):
        raise ValueError("judgement must be an object")
    expected = frozenset((*WORKER_OUTPUT_FIELDS, "parse_ok"))
    _require_exact_keys(value, expected, label="judgement")
    for field in WORKER_OUTPUT_FIELDS:
        if not isinstance(value[field], str):
            raise ValueError(f"judgement.{field} must be a string")
    if not isinstance(value["parse_ok"], bool):
        raise ValueError("judgement.parse_ok must be bool")
    try:
        return NodeJudgement(
            applicable=Applicability(value["applicable"]),
            status_a=CriterionStatus(value["status_a"]),
            status_b=CriterionStatus(value["status_b"]),
            pair_preference=PairPreference(value["pair_preference"]),
            evidence_a=value["evidence_a"],
            evidence_b=value["evidence_b"],
            parse_ok=value["parse_ok"],
        )
    except ValueError as exc:
        raise ValueError(f"invalid serialized judgement: {exc}") from exc


@dataclass(frozen=True)
class StructuredNodeOutput:
    """One model call result with raw provenance and derived local decision."""

    judgement: NodeJudgement
    raw_response: str | None
    parse_error: str | None
    attempt_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.judgement, NodeJudgement):
            raise TypeError("judgement must be NodeJudgement")
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be str or None")
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            raise TypeError("parse_error must be str or None")
        if (
            isinstance(self.attempt_count, bool)
            or not isinstance(self.attempt_count, int)
            or self.attempt_count < 1
        ):
            raise ValueError("attempt_count must be a positive integer")
        if not self.judgement.parse_ok and not self.parse_error:
            raise ValueError("parse failure output must include parse_error")
        if self.judgement.parse_ok and self.parse_error is not None:
            raise ValueError("parsed output must not include parse_error")

    @property
    def local_decision(self) -> LocalDecision:
        return resolve_local_decision(self.judgement)

    def to_dict(self) -> dict[str, Any]:
        return {
            "judgement": _judgement_to_dict(self.judgement),
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "attempt_count": self.attempt_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "StructuredNodeOutput":
        if not isinstance(value, dict):
            raise ValueError("node output must be an object")
        _require_exact_keys(
            value,
            frozenset(
                {"judgement", "raw_response", "parse_error", "attempt_count"}
            ),
            label="node output",
        )
        return cls(
            judgement=_judgement_from_dict(value["judgement"]),
            raw_response=value["raw_response"],
            parse_error=value["parse_error"],
            attempt_count=value["attempt_count"],
        )


@dataclass(frozen=True)
class StructuredCriterionSnapshot:
    """Immutable criterion identity saved with a prediction artifact."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("criterion name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("criterion description must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}

    @classmethod
    def from_dict(cls, value: object) -> "StructuredCriterionSnapshot":
        if not isinstance(value, dict):
            raise ValueError("criterion snapshot must be an object")
        _require_exact_keys(
            value,
            frozenset({"name", "description"}),
            label="criterion snapshot",
        )
        return cls(name=value["name"], description=value["description"])


@dataclass(frozen=True)
class StructuredWorkerRequestSpec:
    """Non-secret identity of the exact worker request configuration."""

    model: str
    worker_backend_id: str
    prompt_sha256: str
    max_data_chars: int | None
    encode_local_image: bool
    image_field: str
    question_field: str
    sample_id_field: str
    decoding_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("model", "worker_backend_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        _require_sha256(self.prompt_sha256, label="prompt_sha256")
        if self.max_data_chars is not None and (
            isinstance(self.max_data_chars, bool)
            or not isinstance(self.max_data_chars, int)
            or self.max_data_chars < 1
        ):
            raise ValueError("max_data_chars must be None or a positive integer")
        if not isinstance(self.encode_local_image, bool):
            raise TypeError("encode_local_image must be bool")
        for name in ("image_field", "question_field", "sample_id_field"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.decoding_config, Mapping):
            raise ValueError("decoding_config must be an object")
        object.__setattr__(
            self,
            "decoding_config",
            _freeze_json(self.decoding_config, label="decoding_config"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "worker_backend_id": self.worker_backend_id,
            "prompt_sha256": self.prompt_sha256,
            "max_data_chars": self.max_data_chars,
            "encode_local_image": self.encode_local_image,
            "image_field": self.image_field,
            "question_field": self.question_field,
            "sample_id_field": self.sample_id_field,
            "decoding_config": _thaw_json(self.decoding_config),
        }

    @classmethod
    def from_dict(cls, value: object) -> "StructuredWorkerRequestSpec":
        if not isinstance(value, dict):
            raise ValueError("request_spec must be an object")
        _require_exact_keys(
            value,
            frozenset(
                {
                    "model",
                    "worker_backend_id",
                    "prompt_sha256",
                    "max_data_chars",
                    "encode_local_image",
                    "image_field",
                    "question_field",
                    "sample_id_field",
                    "decoding_config",
                }
            ),
            label="request_spec",
        )
        return cls(
            model=value["model"],
            worker_backend_id=value["worker_backend_id"],
            prompt_sha256=value["prompt_sha256"],
            max_data_chars=value["max_data_chars"],
            encode_local_image=value["encode_local_image"],
            image_field=value["image_field"],
            question_field=value["question_field"],
            sample_id_field=value["sample_id_field"],
            decoding_config=value["decoding_config"],
        )


@dataclass(frozen=True)
class StructuredPredictionOutput:
    """Versioned all-node outputs reusable by future structured executors."""

    sample_ids: tuple[str, ...]
    sample_fingerprints: tuple[str, ...]
    criteria: tuple[StructuredCriterionSnapshot, ...]
    node_outputs: tuple[Mapping[str, StructuredNodeOutput], ...]
    flat_answers: tuple[FinalPreference, ...]
    request_spec: StructuredWorkerRequestSpec
    semantics_version: str = STRUCTURED_SEMANTICS_VERSION
    schema_version: str = STRUCTURED_WORKER_SCHEMA_VERSION
    prompt_version: str = STRUCTURED_WORKER_PROMPT_VERSION
    parser_version: str = STRUCTURED_WORKER_PARSER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(
            self,
            "sample_fingerprints",
            tuple(self.sample_fingerprints),
        )
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "flat_answers", tuple(self.flat_answers))

        frozen_outputs: list[Mapping[str, StructuredNodeOutput]] = []
        for index, outputs in enumerate(self.node_outputs):
            if not isinstance(outputs, Mapping):
                raise ValueError(f"node_outputs[{index}] must be an object")
            frozen_outputs.append(MappingProxyType(dict(outputs)))
        object.__setattr__(self, "node_outputs", tuple(frozen_outputs))
        self._validate()

    @property
    def model(self) -> str:
        """Compatibility accessor for the model recorded in request_spec."""

        return self.request_spec.model

    @property
    def decoding_config(self) -> Mapping[str, Any]:
        """Read-only compatibility accessor for decoding metadata."""

        return self.request_spec.decoding_config

    def assert_request_compatible(
        self,
        expected: StructuredWorkerRequestSpec,
    ) -> None:
        """Reject replay when the caller's request identity is not identical."""

        if not isinstance(expected, StructuredWorkerRequestSpec):
            raise TypeError("expected must be StructuredWorkerRequestSpec")
        actual_dict = self.request_spec.to_dict()
        expected_dict = expected.to_dict()
        mismatched = sorted(
            key
            for key in actual_dict
            if actual_dict[key] != expected_dict[key]
        )
        if mismatched:
            raise ValueError(
                "structured artifact request mismatch: "
                + ", ".join(mismatched)
            )

    def _validate(self) -> None:
        versions = {
            "semantics_version": (
                self.semantics_version,
                STRUCTURED_SEMANTICS_VERSION,
            ),
            "schema_version": (
                self.schema_version,
                STRUCTURED_WORKER_SCHEMA_VERSION,
            ),
            "prompt_version": (
                self.prompt_version,
                STRUCTURED_WORKER_PROMPT_VERSION,
            ),
            "parser_version": (
                self.parser_version,
                STRUCTURED_WORKER_PARSER_VERSION,
            ),
        }
        for name, (actual, expected) in versions.items():
            if actual != expected:
                raise ValueError(f"{name} mismatch: expected {expected}, got {actual}")

        if not isinstance(self.request_spec, StructuredWorkerRequestSpec):
            raise TypeError("request_spec must be StructuredWorkerRequestSpec")

        if not self.sample_ids:
            raise ValueError("sample_ids must not be empty")
        if any(not isinstance(value, str) or not value.strip() for value in self.sample_ids):
            raise ValueError("every sample ID must be a non-empty string")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must not contain duplicates")
        if len(self.sample_fingerprints) != len(self.sample_ids):
            raise ValueError("sample_fingerprints length must equal sample_ids length")
        for value in self.sample_fingerprints:
            _require_sha256(value, label="sample fingerprint")

        if not self.criteria:
            raise ValueError("criteria must not be empty")
        if any(
            not isinstance(criterion, StructuredCriterionSnapshot)
            for criterion in self.criteria
        ):
            raise ValueError("criteria values must be StructuredCriterionSnapshot")
        criterion_names = tuple(criterion.name for criterion in self.criteria)
        if len(set(criterion_names)) != len(criterion_names):
            raise ValueError("criterion names must not contain duplicates")
        expected_names = set(criterion_names)

        if len(self.node_outputs) != len(self.sample_ids):
            raise ValueError("node_outputs length must equal sample_ids length")
        for index, outputs in enumerate(self.node_outputs):
            if not isinstance(outputs, Mapping):
                raise ValueError(f"node_outputs[{index}] must be an object")
            if set(outputs) != expected_names:
                raise ValueError(
                    f"node_outputs[{index}] keys must equal criterion names"
                )
            if any(
                not isinstance(output, StructuredNodeOutput)
                for output in outputs.values()
            ):
                raise ValueError(
                    f"node_outputs[{index}] values must be StructuredNodeOutput"
                )

        if len(self.flat_answers) != len(self.sample_ids):
            raise ValueError("flat_answers length must equal sample_ids length")
        if any(
            not isinstance(answer, FinalPreference) for answer in self.flat_answers
        ):
            raise ValueError("flat_answers values must be FinalPreference")
        replayed_answers = self.replay_flat_answers()
        if replayed_answers != self.flat_answers:
            raise ValueError("flat_answers do not match replayed node outputs")

    def replay_flat_answers(self) -> tuple[FinalPreference, ...]:
        """Recompute flat-uniform answers without creating an Agent."""

        criterion_names = tuple(criterion.name for criterion in self.criteria)
        return tuple(
            aggregate_selected_roots(
                {
                    name: outputs[name].local_decision.vote
                    for name in criterion_names
                },
                criterion_names,
            )
            for outputs in self.node_outputs
        )

    def to_dict(self) -> dict[str, Any]:
        self._validate()
        return {
            "artifact_type": "structured_worker_predictions",
            "semantics_version": self.semantics_version,
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "parser_version": self.parser_version,
            "request_spec": self.request_spec.to_dict(),
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "samples": [
                {
                    "sample_id": sample_id,
                    "fingerprint": fingerprint,
                    "node_outputs": {
                        criterion.name: outputs[criterion.name].to_dict()
                        for criterion in self.criteria
                    },
                    "flat_answer": flat_answer.value,
                }
                for sample_id, fingerprint, outputs, flat_answer in zip(
                    self.sample_ids,
                    self.sample_fingerprints,
                    self.node_outputs,
                    self.flat_answers,
                )
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> "StructuredPredictionOutput":
        if not isinstance(value, dict):
            raise ValueError("structured prediction artifact must be an object")
        _require_exact_keys(
            value,
            frozenset(
                {
                    "artifact_type",
                    "semantics_version",
                    "schema_version",
                    "prompt_version",
                    "parser_version",
                    "request_spec",
                    "criteria",
                    "samples",
                }
            ),
            label="structured prediction artifact",
        )
        if value["artifact_type"] != "structured_worker_predictions":
            raise ValueError("unsupported artifact_type")
        if not isinstance(value["criteria"], list):
            raise ValueError("criteria must be a list")
        if not isinstance(value["samples"], list):
            raise ValueError("samples must be a list")

        criteria = tuple(
            StructuredCriterionSnapshot.from_dict(item)
            for item in value["criteria"]
        )
        criterion_names = tuple(criterion.name for criterion in criteria)
        sample_ids: list[str] = []
        fingerprints: list[str] = []
        node_outputs: list[dict[str, StructuredNodeOutput]] = []
        flat_answers: list[FinalPreference] = []
        for sample in value["samples"]:
            if not isinstance(sample, dict):
                raise ValueError("each sample artifact must be an object")
            _require_exact_keys(
                sample,
                frozenset(
                    {"sample_id", "fingerprint", "node_outputs", "flat_answer"}
                ),
                label="sample artifact",
            )
            if not isinstance(sample["node_outputs"], dict):
                raise ValueError("sample node_outputs must be an object")
            if set(sample["node_outputs"]) != set(criterion_names):
                raise ValueError(
                    "sample node_outputs keys must equal criterion names"
                )
            sample_ids.append(sample["sample_id"])
            fingerprints.append(sample["fingerprint"])
            node_outputs.append(
                {
                    name: StructuredNodeOutput.from_dict(
                        sample["node_outputs"][name]
                    )
                    for name in criterion_names
                }
            )
            try:
                flat_answers.append(FinalPreference(sample["flat_answer"]))
            except ValueError as exc:
                raise ValueError("invalid flat_answer") from exc

        return cls(
            sample_ids=tuple(sample_ids),
            sample_fingerprints=tuple(fingerprints),
            criteria=criteria,
            node_outputs=tuple(node_outputs),
            flat_answers=tuple(flat_answers),
            request_spec=StructuredWorkerRequestSpec.from_dict(
                value["request_spec"]
            ),
            semantics_version=value["semantics_version"],
            schema_version=value["schema_version"],
            prompt_version=value["prompt_version"],
            parser_version=value["parser_version"],
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "StructuredPredictionOutput":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load structured prediction artifact: {exc}") from exc
        return cls.from_dict(value)


@dataclass(frozen=True)
class StructuredEvaluationOutput:
    """Flat sanity metrics computed from structured local decisions."""

    prediction: StructuredPredictionOutput
    is_correct: tuple[bool, ...]
    accuracy: float
    coverage: float
    per_criterion_accuracy: Mapping[str, float]
    per_criterion_coverage: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_correct", tuple(self.is_correct))
        object.__setattr__(
            self,
            "per_criterion_accuracy",
            MappingProxyType(dict(self.per_criterion_accuracy)),
        )
        object.__setattr__(
            self,
            "per_criterion_coverage",
            MappingProxyType(dict(self.per_criterion_coverage)),
        )


def structured_worker_prompt_sha256(
    worker_prompt: str,
    worker_prompt_postfix: str,
) -> str:
    """Hash the exact prompt template and postfix used by the worker."""

    if not isinstance(worker_prompt, str) or not worker_prompt:
        raise ValueError("worker_prompt must be a non-empty string")
    if not isinstance(worker_prompt_postfix, str):
        raise TypeError("worker_prompt_postfix must be a string")
    return hashlib.sha256(
        f"{worker_prompt}{worker_prompt_postfix}".encode("utf-8")
    ).hexdigest()


def structured_input_fingerprint(
    data: Mapping[str, Any],
    *,
    image_field: str,
    question_field: str,
    sample_id_field: str,
    max_data_chars: int | None = None,
    encode_local_image: bool = True,
) -> str:
    """Hash the actual ordered sample inputs sent to the structured worker."""

    if max_data_chars is not None and (
        isinstance(max_data_chars, bool)
        or not isinstance(max_data_chars, int)
        or max_data_chars < 1
    ):
        raise ValueError("max_data_chars must be None or a positive integer")
    if not isinstance(encode_local_image, bool):
        raise TypeError("encode_local_image must be bool")

    sample_id = data[sample_id_field]
    image_value = data[image_field]
    question = data[question_field]
    candidate_a = data["A"]
    candidate_b = data["B"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample ID must be a non-empty string")
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError("image value must be a non-empty string")
    if not isinstance(candidate_a, str) or not isinstance(candidate_b, str):
        raise ValueError("A/B must be strings")

    if encode_local_image:
        image_path = Path(image_value)
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"failed to fingerprint local image: {exc}") from exc
        mime_type, _ = mimetypes.guess_type(str(image_path))
        image_identity: dict[str, str] = {
            "transport": "embedded_data_url",
            "mime_type": mime_type or "image/jpeg",
            "content_sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
    else:
        image_identity = {
            "transport": "referenced_image_url",
            "value": image_value,
        }

    actual_a = candidate_a[:max_data_chars] if max_data_chars else candidate_a
    actual_b = candidate_b[:max_data_chars] if max_data_chars else candidate_b
    payload = {
        "sample_id": sample_id,
        "image": image_identity,
        "question": "" if question is None else str(question),
        "A": actual_a,
        "B": actual_b,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_ab_swap_consistent(
    original: NodeJudgement,
    swapped: NodeJudgement,
) -> bool:
    """Check structural A/B equivariance while deliberately ignoring evidence."""

    if (
        not original.parse_ok
        or not swapped.parse_ok
        or not original.consistency_ok
        or not swapped.consistency_ok
    ):
        return False
    preference_swap = {
        PairPreference.A: PairPreference.B,
        PairPreference.B: PairPreference.A,
        PairPreference.TIE: PairPreference.TIE,
        PairPreference.UNCERTAIN: PairPreference.UNCERTAIN,
    }
    return (
        original.applicable is swapped.applicable
        and original.status_a is swapped.status_b
        and original.status_b is swapped.status_a
        and preference_swap[original.pair_preference] is swapped.pair_preference
    )

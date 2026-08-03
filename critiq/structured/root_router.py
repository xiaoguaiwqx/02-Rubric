"""Structured joint multi-root routing, artifacts, and online inference."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from critiq.agent import Agent, AgentCallMetrics
from critiq.utils import PairData, parse_json, print_debug

from .root_router_prompts import (
    STRUCTURED_ROOT_ROUTER_POSTFIX,
    STRUCTURED_ROOT_ROUTER_PROMPT,
)
from .schema import StructuredRubric
from .semantics import (
    ResolvedRootRouting,
    RootRoutingDecision,
    resolve_root_routing,
    root_routing_consistency_errors,
)
from .telemetry import ModelCallMetrics, TokenPricing
from .version import (
    STRUCTURED_ROUTER_PARSER_VERSION,
    STRUCTURED_ROUTER_PROMPT_VERSION,
    STRUCTURED_ROUTER_SCHEMA_VERSION,
    STRUCTURED_SEMANTICS_VERSION,
)
from .worker_output import structured_input_fingerprint


class RootRouterOutputParseError(ValueError):
    """Raised when router text does not satisfy the exact JSON schema."""


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(f"{label} fields invalid: missing={sorted(missing)}, extra={sorted(extra)}")


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 hex string")
    return value


def _json_clone(value: object, label: str) -> object:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def _freeze_cloned_json(value: object) -> object:
    """Recursively freeze a value already validated by ``_json_clone``."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_cloned_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_cloned_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    """Return plain JSON containers detached from frozen request metadata."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _prompt_sha256(prompt: str, postfix: str) -> str:
    return hashlib.sha256(f"{prompt}{postfix}".encode("utf-8")).hexdigest()


def parse_structured_root_router_response(raw_response: object) -> RootRoutingDecision:
    """Parse schema only; root-set consistency is validated separately."""

    if not isinstance(raw_response, str):
        raise RootRouterOutputParseError("router response must be a string")
    try:
        payload = parse_json(raw_response)
    except Exception as exc:
        raise RootRouterOutputParseError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise RootRouterOutputParseError("router response JSON must be an object")
    try:
        _require_exact_keys(
            payload,
            {"selected_root_ids", "rationale_by_root"},
            "router output",
        )
    except ValueError as exc:
        raise RootRouterOutputParseError(str(exc)) from exc

    selected = payload["selected_root_ids"]
    rationales = payload["rationale_by_root"]
    if not isinstance(selected, list):
        raise RootRouterOutputParseError("selected_root_ids must be a JSON list")
    if any(not isinstance(root_id, str) for root_id in selected):
        raise RootRouterOutputParseError("selected_root_ids values must be strings")
    if not isinstance(rationales, dict):
        raise RootRouterOutputParseError("rationale_by_root must be a JSON object")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in rationales.items()):
        raise RootRouterOutputParseError("rationale_by_root keys and values must be strings")
    return RootRoutingDecision(
        selected_root_ids=tuple(root_id.strip() for root_id in selected),
        rationale_by_root=MappingProxyType(
            {key.strip(): value.strip() for key, value in rationales.items()}
        ),
        parse_ok=True,
    )


def make_router_parse_failure_decision() -> RootRoutingDecision:
    return RootRoutingDecision(
        selected_root_ids=(),
        rationale_by_root=MappingProxyType({}),
        parse_ok=False,
    )


def _decision_to_dict(decision: RootRoutingDecision) -> dict[str, object]:
    return {
        "selected_root_ids": list(decision.selected_root_ids),
        "rationale_by_root": dict(decision.rationale_by_root),
        "parse_ok": decision.parse_ok,
    }


def _decision_from_dict(value: object) -> RootRoutingDecision:
    if not isinstance(value, dict):
        raise ValueError("router decision must be an object")
    _require_exact_keys(
        value,
        {"selected_root_ids", "rationale_by_root", "parse_ok"},
        "router decision",
    )
    selected = value["selected_root_ids"]
    rationales = value["rationale_by_root"]
    if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        raise ValueError("serialized selected_root_ids must be a string list")
    if not isinstance(rationales, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in rationales.items()
    ):
        raise ValueError("serialized rationale_by_root must map strings to strings")
    if not isinstance(value["parse_ok"], bool):
        raise ValueError("serialized parse_ok must be bool")
    return RootRoutingDecision(
        selected_root_ids=tuple(selected),
        rationale_by_root=MappingProxyType(dict(rationales)),
        parse_ok=value["parse_ok"],
    )


@dataclass(frozen=True)
class RootRouterOutput:
    decision: RootRoutingDecision
    raw_response: str | None
    parse_error: str | None
    attempt_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RootRoutingDecision):
            raise TypeError("decision must be RootRoutingDecision")
        frozen_decision = RootRoutingDecision(
            selected_root_ids=tuple(self.decision.selected_root_ids),
            rationale_by_root=MappingProxyType(dict(self.decision.rationale_by_root)),
            parse_ok=self.decision.parse_ok,
        )
        object.__setattr__(self, "decision", frozen_decision)
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be str or None")
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            raise TypeError("parse_error must be str or None")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int) or self.attempt_count < 1:
            raise ValueError("attempt_count must be a positive integer")
        if self.decision.parse_ok and self.parse_error is not None:
            raise ValueError("parsed router output must not have parse_error")
        if not self.decision.parse_ok and not self.parse_error:
            raise ValueError("parse failure router output must include parse_error")

    def resolve(self, rubric: StructuredRubric) -> ResolvedRootRouting:
        return resolve_root_routing(self.decision, rubric.root_ids, enabled=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": _decision_to_dict(self.decision),
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "attempt_count": self.attempt_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RootRouterOutput":
        if not isinstance(value, dict):
            raise ValueError("router output must be an object")
        _require_exact_keys(value, {"decision", "raw_response", "parse_error", "attempt_count"}, "router output")
        return cls(
            decision=_decision_from_dict(value["decision"]),
            raw_response=value["raw_response"],
            parse_error=value["parse_error"],
            attempt_count=value["attempt_count"],
        )


@dataclass(frozen=True)
class RootCriterionSnapshot:
    node_id: str
    criterion_name: str
    description: str

    def __post_init__(self) -> None:
        for name in ("node_id", "criterion_name", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "node_id": self.node_id,
            "criterion_name": self.criterion_name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RootCriterionSnapshot":
        if not isinstance(value, dict):
            raise ValueError("root snapshot must be an object")
        _require_exact_keys(value, {"node_id", "criterion_name", "description"}, "root snapshot")
        return cls(**value)


def root_snapshots(rubric: StructuredRubric) -> tuple[RootCriterionSnapshot, ...]:
    if not isinstance(rubric, StructuredRubric):
        raise TypeError("rubric must be StructuredRubric")
    return tuple(
        RootCriterionSnapshot(
            node_id=root_id,
            criterion_name=rubric.get_node(root_id).criterion.name,
            description=rubric.get_node(root_id).criterion.description,
        )
        for root_id in rubric.root_ids
    )


def root_set_sha256(rubric: StructuredRubric) -> str:
    payload = [snapshot.to_dict() for snapshot in root_snapshots(rubric)]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RootRouterRequestSpec:
    model: str
    router_backend_id: str
    prompt_sha256: str
    rubric_sha256: str
    roots_sha256: str
    max_data_chars: int | None
    encode_local_image: bool
    image_field: str
    question_field: str
    sample_id_field: str
    decoding_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("model", "router_backend_id", "image_field", "question_field", "sample_id_field"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        _require_sha256(self.prompt_sha256, "prompt_sha256")
        _require_sha256(self.rubric_sha256, "rubric_sha256")
        _require_sha256(self.roots_sha256, "roots_sha256")
        if self.max_data_chars is not None and (
            isinstance(self.max_data_chars, bool)
            or not isinstance(self.max_data_chars, int)
            or self.max_data_chars < 1
        ):
            raise ValueError("max_data_chars must be None or positive")
        if not isinstance(self.encode_local_image, bool):
            raise TypeError("encode_local_image must be bool")
        cloned = _json_clone(self.decoding_config, "decoding_config")
        if not isinstance(cloned, dict):
            raise ValueError("decoding_config must be an object")
        object.__setattr__(self, "decoding_config", _freeze_cloned_json(cloned))

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "router_backend_id": self.router_backend_id,
            "prompt_sha256": self.prompt_sha256,
            "rubric_sha256": self.rubric_sha256,
            "roots_sha256": self.roots_sha256,
            "max_data_chars": self.max_data_chars,
            "encode_local_image": self.encode_local_image,
            "image_field": self.image_field,
            "question_field": self.question_field,
            "sample_id_field": self.sample_id_field,
            "decoding_config": _thaw_json(self.decoding_config),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RootRouterRequestSpec":
        if not isinstance(value, dict):
            raise ValueError("router request spec must be an object")
        expected = {
            "model", "router_backend_id", "prompt_sha256", "rubric_sha256",
            "roots_sha256", "max_data_chars", "encode_local_image", "image_field",
            "question_field", "sample_id_field", "decoding_config",
        }
        _require_exact_keys(value, expected, "router request spec")
        return cls(**value)


@dataclass(frozen=True)
class RootRoutingPredictionOutput:
    sample_ids: tuple[str, ...]
    sample_fingerprints: tuple[str, ...]
    roots: tuple[RootCriterionSnapshot, ...]
    outputs: tuple[RootRouterOutput, ...]
    generation_metrics: tuple[ModelCallMetrics, ...]
    request_spec: RootRouterRequestSpec
    rubric_sha256: str
    semantics_version: str = STRUCTURED_SEMANTICS_VERSION
    schema_version: str = STRUCTURED_ROUTER_SCHEMA_VERSION
    prompt_version: str = STRUCTURED_ROUTER_PROMPT_VERSION
    parser_version: str = STRUCTURED_ROUTER_PARSER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "sample_fingerprints", tuple(self.sample_fingerprints))
        object.__setattr__(self, "roots", tuple(self.roots))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "generation_metrics", tuple(self.generation_metrics))
        self._validate()

    def _validate(self) -> None:
        expected_versions = {
            "semantics_version": STRUCTURED_SEMANTICS_VERSION,
            "schema_version": STRUCTURED_ROUTER_SCHEMA_VERSION,
            "prompt_version": STRUCTURED_ROUTER_PROMPT_VERSION,
            "parser_version": STRUCTURED_ROUTER_PARSER_VERSION,
        }
        for name, expected in expected_versions.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} mismatch: expected {expected}")
        _require_sha256(self.rubric_sha256, "rubric_sha256")
        if not isinstance(self.request_spec, RootRouterRequestSpec):
            raise TypeError("request_spec must be RootRouterRequestSpec")
        if self.request_spec.rubric_sha256 != self.rubric_sha256:
            raise ValueError("request_spec rubric hash mismatch")
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique and non-empty")
        if any(not isinstance(value, str) or not value.strip() for value in self.sample_ids):
            raise ValueError("every sample ID must be non-empty")
        count = len(self.sample_ids)
        if not all(len(values) == count for values in (self.sample_fingerprints, self.outputs, self.generation_metrics)):
            raise ValueError("router artifact sample arrays must have equal length")
        for fingerprint in self.sample_fingerprints:
            _require_sha256(fingerprint, "sample fingerprint")
        if not self.roots or len({root.node_id for root in self.roots}) != len(self.roots):
            raise ValueError("root snapshots must be unique and non-empty")
        if any(not isinstance(value, RootCriterionSnapshot) for value in self.roots):
            raise TypeError("roots must contain RootCriterionSnapshot")
        if any(not isinstance(value, RootRouterOutput) for value in self.outputs):
            raise TypeError("outputs must contain RootRouterOutput")
        if any(not isinstance(value, ModelCallMetrics) for value in self.generation_metrics):
            raise TypeError("generation_metrics must contain ModelCallMetrics")

    def assert_compatible(self, rubric: StructuredRubric, sample_ids: Sequence[str], sample_fingerprints: Sequence[str]) -> None:
        if rubric.rubric_sha256 != self.rubric_sha256:
            raise ValueError("router artifact rubric hash mismatch")
        if root_snapshots(rubric) != self.roots:
            raise ValueError("router artifact root snapshots mismatch")
        if tuple(sample_ids) != self.sample_ids:
            raise ValueError("router artifact sample IDs mismatch")
        if tuple(sample_fingerprints) != self.sample_fingerprints:
            raise ValueError("router artifact sample fingerprints mismatch")

    def to_dict(self) -> dict[str, object]:
        self._validate()
        return {
            "artifact_type": "structured_root_routing_predictions",
            "semantics_version": self.semantics_version,
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "parser_version": self.parser_version,
            "rubric_sha256": self.rubric_sha256,
            "request_spec": self.request_spec.to_dict(),
            "roots": [root.to_dict() for root in self.roots],
            "samples": [
                {
                    "sample_id": sample_id,
                    "fingerprint": fingerprint,
                    "output": output.to_dict(),
                    "generation_metrics": metrics.to_dict(),
                }
                for sample_id, fingerprint, output, metrics in zip(
                    self.sample_ids, self.sample_fingerprints, self.outputs, self.generation_metrics
                )
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> "RootRoutingPredictionOutput":
        if not isinstance(value, dict):
            raise ValueError("router artifact must be an object")
        expected = {
            "artifact_type", "semantics_version", "schema_version", "prompt_version",
            "parser_version", "rubric_sha256", "request_spec", "roots", "samples",
        }
        _require_exact_keys(value, expected, "router artifact")
        if value["artifact_type"] != "structured_root_routing_predictions":
            raise ValueError("router artifact_type mismatch")
        if not isinstance(value["roots"], list) or not isinstance(value["samples"], list):
            raise ValueError("router artifact roots/samples must be lists")
        samples = value["samples"]
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("router artifact sample must be an object")
            _require_exact_keys(sample, {"sample_id", "fingerprint", "output", "generation_metrics"}, "router sample")
        return cls(
            sample_ids=tuple(sample["sample_id"] for sample in samples),
            sample_fingerprints=tuple(sample["fingerprint"] for sample in samples),
            roots=tuple(RootCriterionSnapshot.from_dict(root) for root in value["roots"]),
            outputs=tuple(RootRouterOutput.from_dict(sample["output"]) for sample in samples),
            generation_metrics=tuple(ModelCallMetrics.from_dict(sample["generation_metrics"]) for sample in samples),
            request_spec=RootRouterRequestSpec.from_dict(value["request_spec"]),
            rubric_sha256=value["rubric_sha256"],
            semantics_version=value["semantics_version"],
            schema_version=value["schema_version"],
            prompt_version=value["prompt_version"],
            parser_version=value["parser_version"],
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "RootRoutingPredictionOutput":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load router artifact: {exc}") from exc
        return cls.from_dict(payload)


class StructuredRootRouter:
    """Make exactly one joint multi-label root decision per sample."""

    def __init__(
        self,
        router_args: dict[str, Any],
        router_backend_id: str,
        *,
        max_retries: int = 3,
        router_prompt: str = STRUCTURED_ROOT_ROUTER_PROMPT,
        max_data_chars: int | None = None,
        image_field: str = "image_path",
        question_field: str = "question",
        sample_id_field: str = "sample_id",
        encode_local_image: bool = True,
        pricing: TokenPricing | None = None,
        call_backend: Any | None = None,
    ) -> None:
        if not isinstance(router_args, dict):
            raise TypeError("router_args must be a dict")
        if not isinstance(router_backend_id, str) or not router_backend_id.strip():
            raise ValueError("router_backend_id must be a non-empty string")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not isinstance(router_prompt, str) or not router_prompt:
            raise ValueError("router_prompt must be non-empty")
        required = {"{question}", "{A}", "{B}", "{roots}"}
        missing = {placeholder for placeholder in required if placeholder not in router_prompt}
        if missing:
            raise ValueError(f"router prompt missing placeholders: {sorted(missing)}")
        if max_data_chars is not None and (
            isinstance(max_data_chars, bool)
            or not isinstance(max_data_chars, int)
            or max_data_chars < 1
        ):
            raise ValueError("max_data_chars must be None or positive")
        if not isinstance(encode_local_image, bool):
            raise TypeError("encode_local_image must be bool")
        for name, value in (
            ("image_field", image_field),
            ("question_field", question_field),
            ("sample_id_field", sample_id_field),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if pricing is not None and not isinstance(pricing, TokenPricing):
            raise TypeError("pricing must be TokenPricing or None")
        self.router_args = dict(router_args)
        self.router_backend_id = router_backend_id
        self.max_retries = max_retries
        self.router_prompt = router_prompt
        self.max_data_chars = max_data_chars
        self.image_field = image_field
        self.question_field = question_field
        self.sample_id_field = sample_id_field
        self.encode_local_image = encode_local_image
        self.pricing = pricing
        self.call_backend = call_backend

    @staticmethod
    def _image_path_to_data_url(image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        mime_type, _ = mimetypes.guess_type(str(path))
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type or 'image/jpeg'};base64,{image_b64}"

    def _validate_data(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("router sample must be a mapping")
        for field in (self.sample_id_field, self.image_field, self.question_field):
            value = data.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"router sample requires non-empty {field!r}")
        if not isinstance(data.get("A"), str) or not isinstance(data.get("B"), str):
            raise ValueError("router sample A/B must be strings")

    def sample_fingerprint(self, data: Mapping[str, Any]) -> str:
        self._validate_data(data)
        return structured_input_fingerprint(
            data,
            image_field=self.image_field,
            question_field=self.question_field,
            sample_id_field=self.sample_id_field,
            max_data_chars=self.max_data_chars,
            encode_local_image=self.encode_local_image,
        )

    def request_spec(self, rubric: StructuredRubric) -> RootRouterRequestSpec:
        decoding = _json_clone(self.router_args.get("request_kwargs") or {}, "router request_kwargs")
        if not isinstance(decoding, dict):
            raise ValueError("router request_kwargs must be an object")
        return RootRouterRequestSpec(
            model=str(self.router_args.get("model", "gpt-4o-mini")),
            router_backend_id=self.router_backend_id,
            prompt_sha256=_prompt_sha256(self.router_prompt, STRUCTURED_ROOT_ROUTER_POSTFIX),
            rubric_sha256=rubric.rubric_sha256,
            roots_sha256=root_set_sha256(rubric),
            max_data_chars=self.max_data_chars,
            encode_local_image=self.encode_local_image,
            image_field=self.image_field,
            question_field=self.question_field,
            sample_id_field=self.sample_id_field,
            decoding_config=decoding,
        )

    def _make_prompt(self, data: Mapping[str, Any], rubric: StructuredRubric) -> str:
        self._validate_data(data)
        candidate_a = data["A"][: self.max_data_chars] if self.max_data_chars else data["A"]
        candidate_b = data["B"][: self.max_data_chars] if self.max_data_chars else data["B"]
        roots_text = "\n".join(
            f"- root_id={snapshot.node_id!r}; criterion={snapshot.criterion_name!r}; description={snapshot.description}"
            for snapshot in root_snapshots(rubric)
        )
        return (
            self.router_prompt.replace("{question}", str(data[self.question_field]))
            .replace("{A}", candidate_a)
            .replace("{B}", candidate_b)
            .replace("{roots}", roots_text)
            + STRUCTURED_ROOT_ROUTER_POSTFIX
        )

    def _make_user_content(self, data: Mapping[str, Any], rubric: StructuredRubric) -> list[dict[str, Any]]:
        image_value = str(data[self.image_field])
        image_url = (
            self._image_path_to_data_url(image_value)
            if self.encode_local_image
            else image_value
        )
        return [
            {"type": "text", "text": self._make_prompt(data, rubric)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    @staticmethod
    def _agent_metrics(agent: object) -> AgentCallMetrics:
        metrics = getattr(agent, "last_call_metrics", None)
        if isinstance(metrics, AgentCallMetrics):
            return metrics
        return AgentCallMetrics(
            api_attempts=1,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            usage_complete=False,
        )

    def route_one_uncached(
        self,
        data: PairData,
        rubric: StructuredRubric,
    ) -> tuple[RootRouterOutput, ModelCallMetrics]:
        """Run router retries without consulting a prediction cache."""

        self._validate_data(data)
        last_raw: str | None = None
        last_parse_error = "router did not return a parseable response"
        last_inconsistent: RootRouterOutput | None = None
        call_metrics: list[AgentCallMetrics] = []
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            content = self._make_user_content(data, rubric)
            if self.call_backend is None:
                agent = Agent(**self.router_args)
                raw_response = agent(content, stream=False)
                metrics = self._agent_metrics(agent)
            else:
                raw_response, metrics = self.call_backend.call(
                    content, request_type="router",
                    request_key=str(data[self.sample_id_field]),
                    structured_attempt=attempt, agent_args=self.router_args)
            call_metrics.append(metrics)
            last_raw = raw_response if isinstance(raw_response, str) else None
            try:
                decision = parse_structured_root_router_response(raw_response)
            except RootRouterOutputParseError as exc:
                last_parse_error = str(exc)
                print_debug("Failed to parse structured root router response", attempt, exc)
                continue

            output = RootRouterOutput(
                decision=decision,
                raw_response=raw_response,
                parse_error=None,
                attempt_count=attempt,
            )
            if not root_routing_consistency_errors(decision, rubric.root_ids):
                final = output
                break
            last_inconsistent = output
            print_debug(
                "Structured root router response failed consistency validation",
                root_routing_consistency_errors(decision, rubric.root_ids),
            )
        else:
            if last_inconsistent is not None:
                final = RootRouterOutput(
                    decision=last_inconsistent.decision,
                    raw_response=last_inconsistent.raw_response,
                    parse_error=None,
                    attempt_count=total_attempts,
                )
            else:
                final = RootRouterOutput(
                    decision=make_router_parse_failure_decision(),
                    raw_response=last_raw,
                    parse_error=last_parse_error,
                    attempt_count=total_attempts,
                )

        metrics = ModelCallMetrics.from_agent_calls(
            call_metrics,
            logical_evaluations=1,
            parse_retries=max(0, len(call_metrics) - 1),
            pricing=self.pricing,
        )
        return final, metrics

    def pred(self, dataset: Sequence[PairData], rubric: StructuredRubric) -> RootRoutingPredictionOutput:
        if not dataset:
            raise ValueError("router dataset must not be empty")
        sample_ids: list[str] = []
        fingerprints: list[str] = []
        outputs: list[RootRouterOutput] = []
        metrics: list[ModelCallMetrics] = []
        for data in dataset:
            self._validate_data(data)
            sample_ids.append(str(data[self.sample_id_field]))
            fingerprints.append(self.sample_fingerprint(data))
            output, call_metrics = self.route_one_uncached(data, rubric)
            outputs.append(output)
            metrics.append(call_metrics)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("router dataset sample IDs must be unique")
        return RootRoutingPredictionOutput(
            sample_ids=tuple(sample_ids),
            sample_fingerprints=tuple(fingerprints),
            roots=root_snapshots(rubric),
            outputs=tuple(outputs),
            generation_metrics=tuple(metrics),
            request_spec=self.request_spec(rubric),
            rubric_sha256=rubric.rubric_sha256,
        )


def is_root_selection_swap_consistent(
    original: RootRouterOutput,
    swapped: RootRouterOutput,
    rubric: StructuredRubric,
) -> bool:
    """Compare normalized selected roots; rationales do not participate."""

    original_resolved = original.resolve(rubric)
    swapped_resolved = swapped.resolve(rubric)
    return (
        original_resolved.source == swapped_resolved.source
        and original_resolved.selected_root_ids == swapped_resolved.selected_root_ids
    )

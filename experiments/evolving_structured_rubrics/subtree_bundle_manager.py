"""Manager types and calls for atomic root-subtree evolution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from critiq.subtree_bundle_prompts import (
    BUNDLE_FAILURE_ATTRIBUTION_PROMPT,
    BUNDLE_FAILURE_ATTRIBUTION_PROMPT_VERSION,
    BUNDLE_PARSER_VERSION,
    BUNDLE_REFINE_PROMPT,
    BUNDLE_REFINE_PROMPT_VERSION,
    ROOT_ERROR_SIGNATURE_PROMPT,
    ROOT_ERROR_SIGNATURE_PROMPT_VERSION,
)
from critiq.utils import parse_json
from critiq.structured.cache import canonical_sha256
from critiq.structured.evolution.specialize_types import (
    ErrorSignature,
    SpecializeManagerRequestSpec,
)
from critiq.structured.schema import RubricCriterionSnapshot, RubricNode, StructuredRubric


FAILURE_TYPES = (
    "no_effect",
    "missed_correction",
    "overcorrection_on_harmed",
    "coverage_loss_to_none",
    "sibling_boundary_conflict",
    "evidence_misuse",
    "cluster_or_decomposition_error",
    "bundle_edit_interaction",
    "mixed_or_inconclusive",
)


class BundleManagerFailure(RuntimeError):
    """A structured Manager call exhausted its parser retries."""

    def __init__(self, stage: str, message: str, *, raw_response: str | None,
                 attempt_count: int, metrics: Mapping[str, Any] | None = None) -> None:
        super().__init__(f"{stage} failed after {attempt_count} attempts: {message}")
        self.stage = stage
        self.raw_response = raw_response
        self.attempt_count = attempt_count
        self.metrics = dict(metrics or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": str(self),
            "raw_response": self.raw_response,
            "attempt_count": self.attempt_count,
            "metrics": self.metrics,
        }


def _exact_object(raw: object, fields: set[str], label: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        value = dict(raw)
    else:
        try:
            value = parse_json(raw)
        except Exception as exc:
            raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return dict(value)


def _clean_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a{' possibly empty' if allow_empty else ' non-empty'} list")
    result = tuple(_clean_string(item, label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True)
class RootErrorSignature:
    sample_id: str
    task_pattern: str
    visual_focus: str
    candidate_difference: str
    subtree_failure: str
    suggested_subdomain: str

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def to_legacy(self) -> ErrorSignature:
        return ErrorSignature(
            self.sample_id, self.task_pattern, self.visual_focus,
            self.candidate_difference, self.subtree_failure,
            self.suggested_subdomain)

    @classmethod
    def parse(cls, raw: object, expected_sample_id: str) -> "RootErrorSignature":
        fields = set(cls.__dataclass_fields__)
        value = _exact_object(raw, fields, "root error signature")
        values = {field: _clean_string(value[field], field) for field in fields}
        if values["sample_id"] != expected_sample_id:
            raise ValueError("root error signature sample_id mismatch")
        return cls(**values)


@dataclass(frozen=True)
class BundleEdit:
    node_id: str
    criterion_name: str
    original_description_sha256: str
    description: str
    failure_analysis: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "criterion_name": self.criterion_name,
            "original_description_sha256": self.original_description_sha256,
            "description": self.description,
            "failure_analysis": list(self.failure_analysis),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SubtreeBundleRefineProposal:
    root_id: str
    original_subtree_sha256: str
    edits: tuple[BundleEdit, ...]
    unchanged_node_ids: tuple[str, ...]
    bundle_rationale: str
    representative_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "original_subtree_sha256": self.original_subtree_sha256,
            "edits": [item.to_dict() for item in self.edits],
            "unchanged_node_ids": list(self.unchanged_node_ids),
            "bundle_rationale": self.bundle_rationale,
            "representative_sample_ids": list(self.representative_sample_ids),
        }


def subtree_node_ids(rubric: StructuredRubric, root_id: str) -> tuple[str, ...]:
    return tuple(node_id for node_id in rubric.preorder_node_ids()
                 if rubric.root_id_for(node_id) == root_id)


def subtree_payload(rubric: StructuredRubric, root_id: str) -> dict[str, Any]:
    node_ids = subtree_node_ids(rubric, root_id)
    return {
        "root_id": root_id,
        "nodes": [rubric.get_node(node_id).to_dict() for node_id in node_ids],
        "edges": [edge.to_dict() for edge in rubric.edges
                  if edge.parent_id in node_ids and edge.child_id in node_ids],
    }


def subtree_sha256(rubric: StructuredRubric, root_id: str) -> str:
    return canonical_sha256(subtree_payload(rubric, root_id))


def _description_sha256(description: str) -> str:
    return hashlib.sha256(description.encode("utf-8")).hexdigest()


def bundle_refine_prompt_subtree(
    rubric: StructuredRubric, root_id: str,
) -> str:
    """Render the subtree plus exact hashes the Manager must echo.

    LLMs are not reliable SHA-256 calculators. Supplying hashes computed from
    the current rubric prevents valid proposals being rejected because the
    model guessed or normalized a long description differently. The canonical
    subtree payload and scientific subtree hash remain unchanged.
    """
    payload = subtree_payload(rubric, root_id)
    hashes = {
        node.node_id: _description_sha256(node.criterion.description)
        for node in rubric.children(root_id)
    }
    return (
        json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n\nExact current child description SHA-256 values (copy these values "
        "verbatim into original_description_sha256):\n"
        + json.dumps(hashes, indent=2, ensure_ascii=False)
    )


def _validate_description(value: str, max_chars: int) -> str:
    value = _clean_string(value, "description")
    if len(value) > max_chars:
        raise ValueError("bundle Refine description exceeds max chars")
    headings = (
        "Criterion focus:", "Applicable only when:",
        "Not applicable when:", "Decision rule:",
    )
    positions = [value.find(item) for item in headings]
    if (any(position < 0 for position in positions)
            or positions != sorted(positions)
            or any(value.count(item) != 1 for item in headings)):
        raise ValueError("bundle Refine description lacks ordered required sections")
    return value


def parse_bundle_refine(
    raw: object, *, rubric: StructuredRubric, root_id: str,
    allowed_sample_ids: Sequence[str], max_description_chars: int,
) -> SubtreeBundleRefineProposal:
    fields = {
        "root_id", "original_subtree_sha256", "edits", "unchanged_node_ids",
        "bundle_rationale", "representative_sample_ids",
    }
    value = _exact_object(raw, fields, "bundle Refine proposal")
    if value["root_id"] != root_id:
        raise ValueError("bundle Refine root_id mismatch")
    expected_sha = subtree_sha256(rubric, root_id)
    if value["original_subtree_sha256"] != expected_sha:
        raise ValueError("bundle Refine subtree hash mismatch")
    child_ids = {node.node_id for node in rubric.children(root_id)}
    if not child_ids:
        raise ValueError("bundle Refine requires existing children")
    raw_edits = value["edits"]
    if not isinstance(raw_edits, list) or not raw_edits:
        raise ValueError("bundle Refine must edit at least one child")
    edits = []
    seen = set()
    edit_fields = {
        "node_id", "criterion_name", "original_description_sha256",
        "description", "failure_analysis", "rationale",
    }
    for raw_edit in raw_edits:
        if not isinstance(raw_edit, Mapping) or set(raw_edit) != edit_fields:
            raise ValueError("bundle edit fields are invalid")
        node_id = _clean_string(raw_edit["node_id"], "node_id")
        if node_id not in child_ids or node_id in seen:
            raise ValueError("bundle edit must target each existing child at most once")
        seen.add(node_id)
        node = rubric.get_node(node_id)
        criterion_name = _clean_string(raw_edit["criterion_name"], "criterion_name")
        if criterion_name != node.criterion.name:
            raise ValueError("bundle Refine cannot rename criteria")
        original_sha = _clean_string(
            raw_edit["original_description_sha256"], "original_description_sha256")
        if original_sha != _description_sha256(node.criterion.description):
            raise ValueError("bundle edit original description hash mismatch")
        edits.append(BundleEdit(
            node_id=node_id,
            criterion_name=criterion_name,
            original_description_sha256=original_sha,
            description=_validate_description(
                raw_edit["description"], max_description_chars),
            failure_analysis=_string_list(
                raw_edit["failure_analysis"], "failure_analysis"),
            rationale=_clean_string(raw_edit["rationale"], "rationale"),
        ))
    unchanged = _string_list(
        value["unchanged_node_ids"], "unchanged_node_ids", allow_empty=True)
    if set(unchanged) != child_ids - seen:
        raise ValueError("unchanged_node_ids must exactly cover unedited children")
    representatives = _string_list(
        value["representative_sample_ids"], "representative_sample_ids")
    if not 1 <= len(representatives) <= 6:
        raise ValueError("bundle Refine must cite one to six representative samples")
    if not set(representatives).issubset(set(allowed_sample_ids)):
        raise ValueError("bundle Refine cited a sample outside the allowlist")
    return SubtreeBundleRefineProposal(
        root_id=root_id, original_subtree_sha256=expected_sha,
        edits=tuple(edits), unchanged_node_ids=unchanged,
        bundle_rationale=_clean_string(value["bundle_rationale"], "bundle_rationale"),
        representative_sample_ids=representatives,
    )


def apply_bundle_refine(
    rubric: StructuredRubric, proposal: SubtreeBundleRefineProposal,
) -> StructuredRubric:
    if subtree_sha256(rubric, proposal.root_id) != proposal.original_subtree_sha256:
        raise ValueError("bundle Refine base subtree drift")
    edits = {item.node_id: item for item in proposal.edits}
    nodes = dict(rubric.nodes)
    for node_id, edit in edits.items():
        old = rubric.get_node(node_id)
        nodes[node_id] = RubricNode(
            node_id=node_id,
            criterion=RubricCriterionSnapshot(
                old.criterion.name, edit.description, old.criterion.score),
            examples=old.examples, lineage=old.lineage)
    result = StructuredRubric(nodes, rubric.edges, rubric.root_ids)
    if result.root_ids != rubric.root_ids or result.edges != rubric.edges:
        raise ValueError("bundle Refine changed structure")
    for node_id in rubric.nodes:
        if node_id not in edits and result.get_node(node_id) != rubric.get_node(node_id):
            raise ValueError("bundle Refine changed an unedited node")
    return result


def parse_failure_attribution(
    raw: object, *, operator: str, allowed_child_ids: Sequence[str],
) -> dict[str, Any]:
    fields = {
        "primary_failure_type", "implicated_child_ids", "summary",
        "corrected_harmed_explanation", "preserve_next_time", "change_next_time",
    }
    value = _exact_object(raw, fields, "bundle failure attribution")
    failure_type = _clean_string(value["primary_failure_type"], "primary_failure_type")
    allowed_types = set(FAILURE_TYPES)
    if operator == "split":
        allowed_types.discard("bundle_edit_interaction")
    elif operator == "refine":
        allowed_types.discard("cluster_or_decomposition_error")
    else:
        raise ValueError("unsupported bundle operator")
    if failure_type not in allowed_types:
        raise ValueError("invalid bundle failure type")
    implicated = _string_list(
        value["implicated_child_ids"], "implicated_child_ids", allow_empty=True)
    if not set(implicated).issubset(set(allowed_child_ids)):
        raise ValueError("failure attribution implicated an unknown child")
    return {
        "primary_failure_type": failure_type,
        "implicated_child_ids": list(implicated),
        "summary": _clean_string(value["summary"], "summary"),
        "corrected_harmed_explanation": _clean_string(
            value["corrected_harmed_explanation"],
            "corrected_harmed_explanation"),
        "preserve_next_time": list(_string_list(
            value["preserve_next_time"], "preserve_next_time", allow_empty=True)),
        "change_next_time": list(_string_list(
            value["change_next_time"], "change_next_time")),
    }


class SubtreeBundleManager:
    """Custom Phase21 calls backed by the existing available-slot Managers."""

    def __init__(self, signature_manager: Any, cluster_manager: Any,
                 child_manager: Any, refine_manager: Any) -> None:
        self.signature_manager = signature_manager
        self.cluster_manager = cluster_manager
        self.child_manager = child_manager
        self.refine_manager = refine_manager

    def request_specs(self) -> dict[str, dict[str, Any]]:
        signature = self.signature_manager._spec(
            ROOT_ERROR_SIGNATURE_PROMPT,
            self.signature_manager.analysis_request_kwargs,
            ROOT_ERROR_SIGNATURE_PROMPT_VERSION,
            parser_version=BUNDLE_PARSER_VERSION)
        bundle = SpecializeManagerRequestSpec.from_prompt(
            model=self.refine_manager.model,
            backend_id=self.refine_manager.backend_pool.backend_id,
            prompt=BUNDLE_REFINE_PROMPT,
            decoding_config=self.refine_manager.generation_request_kwargs,
            prompt_version=BUNDLE_REFINE_PROMPT_VERSION,
            parser_version=BUNDLE_PARSER_VERSION)
        attribution = SpecializeManagerRequestSpec.from_prompt(
            model=self.refine_manager.model,
            backend_id=self.refine_manager.backend_pool.backend_id,
            prompt=BUNDLE_FAILURE_ATTRIBUTION_PROMPT,
            decoding_config=self.refine_manager.attribution_request_kwargs,
            prompt_version=BUNDLE_FAILURE_ATTRIBUTION_PROMPT_VERSION,
            parser_version=BUNDLE_PARSER_VERSION)
        return {
            "root_error_signature": signature.to_dict(),
            "semantic_cluster": self.cluster_manager.request_specs()[
                "semantic_cluster"].to_dict(),
            "child_generation": self.child_manager.request_specs()[
                "child_generation"].to_dict(),
            "bundle_refine": bundle.to_dict(),
            "bundle_failure_attribution": attribution.to_dict(),
        }

    @staticmethod
    def _metrics(manager: Any, calls: Sequence[Any]) -> dict[str, Any]:
        return manager._metrics(calls).to_dict()

    def infer_signature(
        self, *, row: Mapping[str, Any], rubric: StructuredRubric, root_id: str,
        root_report: Mapping[str, Any], rubric_memory: Mapping[str, Any],
    ) -> dict[str, Any]:
        sample_id = str(row["sample_id"])
        prompt = ROOT_ERROR_SIGNATURE_PROMPT.format(
            subtree_json=json.dumps(
                subtree_payload(rubric, root_id), indent=2, ensure_ascii=False),
            sample_id=sample_id, question=row["question"], A=row["A"], B=row["B"],
            gold=row["answer"], subtree_answer=root_report["answer"],
            subtree_report_json=json.dumps(root_report, indent=2, ensure_ascii=False),
            rubric_memory_json=json.dumps(rubric_memory, indent=2, ensure_ascii=False),
        )
        manager = self.signature_manager
        calls = []
        last_raw = None
        last_error = "invalid root error signature"
        total = manager.structured_max_retries + 1
        for attempt in range(1, total + 1):
            raw, call_metrics = manager._call(
                manager._multimodal(prompt, (row,)),
                request_type="root_error_signature", request_key=f"{root_id}::{sample_id}",
                structured_attempt=attempt,
                request_kwargs=manager.analysis_request_kwargs)
            calls.append(call_metrics)
            last_raw = raw if isinstance(raw, str) else None
            try:
                signature = RootErrorSignature.parse(raw, sample_id)
                return {
                    "signature": signature.to_dict(), "raw_response": last_raw,
                    "parse_error": None, "attempt_count": attempt,
                    "metrics": self._metrics(manager, calls),
                    "request_spec": self.request_specs()["root_error_signature"],
                }
            except ValueError as exc:
                last_error = str(exc)
        raise BundleManagerFailure(
            "root_error_signature", last_error, raw_response=last_raw,
            attempt_count=total, metrics=self._metrics(manager, calls))

    def generate_bundle_refine(
        self, *, rubric: StructuredRubric, root_id: str,
        signatures: Sequence[RootErrorSignature],
        representative_rows: Sequence[Mapping[str, Any]],
        failure_history: Sequence[Mapping[str, Any]],
        rubric_memory: Mapping[str, Any], max_description_chars: int,
    ) -> dict[str, Any]:
        allowed_ids = [str(row["sample_id"]) for row in representative_rows]
        representative_samples = [
            {
                "sample_id": str(row["sample_id"]),
                "question": row["question"],
                "A": row["A"],
                "B": row["B"],
                "gold": row["answer"],
            }
            for row in representative_rows
        ]
        representative_samples_json = json.dumps(
            representative_samples, indent=2, ensure_ascii=False)
        # Keep this constraint in the rendered prompt (rather than changing
        # the frozen static request template) so paused runs can resume
        # without request-spec identity drift.
        representative_samples_json += (
            "\n\nIMPORTANT ALLOWLIST CONSTRAINT:\n"
            "The `representative_sample_ids` field must contain only IDs from "
            "the following exact allowlist. Do not use IDs from the full error "
            "signatures, rubric memory, or failure history unless they also "
            "appear in this list.\n"
            "Allowed representative_sample_ids (exact):\n"
            + json.dumps(allowed_ids, indent=2, ensure_ascii=False)
        )
        prompt = BUNDLE_REFINE_PROMPT.format(
            root_id=root_id,
            original_subtree_sha256=subtree_sha256(rubric, root_id),
            subtree_json=bundle_refine_prompt_subtree(rubric, root_id),
            signatures_json=json.dumps(
                [item.to_dict() for item in signatures], indent=2, ensure_ascii=False),
            representative_samples_json=representative_samples_json,
            failure_history_json=json.dumps(
                list(failure_history), indent=2, ensure_ascii=False),
            rubric_memory_json=json.dumps(rubric_memory, indent=2, ensure_ascii=False),
            max_description_chars=max_description_chars,
        )
        manager = self.refine_manager
        calls = []
        last_raw = None
        last_error = "invalid bundle Refine proposal"
        total = manager.structured_max_retries + 1
        for attempt in range(1, total + 1):
            raw, call_metrics = manager._call(
                manager._multimodal(prompt, representative_rows),
                request_type="subtree_bundle_refine", request_key=root_id,
                structured_attempt=attempt,
                request_kwargs=manager.generation_request_kwargs)
            calls.append(call_metrics)
            last_raw = raw if isinstance(raw, str) else None
            try:
                proposal = parse_bundle_refine(
                    raw, rubric=rubric, root_id=root_id,
                    allowed_sample_ids=allowed_ids,
                    max_description_chars=max_description_chars)
                return {
                    "proposal": proposal.to_dict(), "raw_response": last_raw,
                    "attempt_count": attempt, "metrics": self._metrics(manager, calls),
                    "request_spec": self.request_specs()["bundle_refine"],
                }
            except ValueError as exc:
                last_error = str(exc)
        raise BundleManagerFailure(
            "bundle_refine", last_error, raw_response=last_raw,
            attempt_count=total, metrics=self._metrics(manager, calls))

    def attribute_failure(
        self, *, operator: str, baseline_rubric: StructuredRubric,
        candidate_rubric: StructuredRubric, root_id: str,
        signatures: Sequence[RootErrorSignature], candidate: Mapping[str, Any],
        paired_evidence: Mapping[str, Any],
        representative_rows: Sequence[Mapping[str, Any]],
        failure_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        child_ids = [node.node_id for node in candidate_rubric.children(root_id)]
        allowed_types = list(FAILURE_TYPES)
        if operator == "split":
            allowed_types.remove("bundle_edit_interaction")
        else:
            allowed_types.remove("cluster_or_decomposition_error")
        prompt = BUNDLE_FAILURE_ATTRIBUTION_PROMPT.format(
            operator=operator,
            baseline_subtree_json=json.dumps(
                subtree_payload(baseline_rubric, root_id), indent=2, ensure_ascii=False),
            candidate_subtree_json=json.dumps(
                subtree_payload(candidate_rubric, root_id), indent=2, ensure_ascii=False),
            signatures_json=json.dumps(
                [item.to_dict() for item in signatures], indent=2, ensure_ascii=False),
            candidate_json=json.dumps(dict(candidate), indent=2, ensure_ascii=False),
            paired_evidence_json=json.dumps(
                dict(paired_evidence), indent=2, ensure_ascii=False),
            failure_history_json=json.dumps(
                list(failure_history), indent=2, ensure_ascii=False),
            allowed_types_json=json.dumps(allowed_types, ensure_ascii=False),
        )
        manager = self.refine_manager
        calls = []
        last_raw = None
        last_error = "invalid bundle failure attribution"
        total = manager.structured_max_retries + 1
        for attempt in range(1, total + 1):
            raw, call_metrics = manager._call(
                manager._multimodal(prompt, representative_rows),
                request_type="subtree_bundle_failure_attribution",
                request_key=f"{operator}::{root_id}", structured_attempt=attempt,
                request_kwargs=manager.attribution_request_kwargs)
            calls.append(call_metrics)
            last_raw = raw if isinstance(raw, str) else None
            try:
                attribution = parse_failure_attribution(
                    raw, operator=operator, allowed_child_ids=child_ids)
                return {
                    "attribution": attribution, "raw_response": last_raw,
                    "attempt_count": attempt, "metrics": self._metrics(manager, calls),
                    "request_spec": self.request_specs()[
                        "bundle_failure_attribution"],
                }
            except ValueError as exc:
                last_error = str(exc)
        raise BundleManagerFailure(
            "bundle_failure_attribution", last_error, raw_response=last_raw,
            attempt_count=total, metrics=self._metrics(manager, calls))


def proposal_from_dict(value: Mapping[str, Any]) -> SubtreeBundleRefineProposal:
    edits = tuple(BundleEdit(
        node_id=item["node_id"], criterion_name=item["criterion_name"],
        original_description_sha256=item["original_description_sha256"],
        description=item["description"],
        failure_analysis=tuple(item["failure_analysis"]), rationale=item["rationale"])
        for item in value["edits"])
    return SubtreeBundleRefineProposal(
        root_id=value["root_id"],
        original_subtree_sha256=value["original_subtree_sha256"], edits=edits,
        unchanged_node_ids=tuple(value["unchanged_node_ids"]),
        bundle_rationale=value["bundle_rationale"],
        representative_sample_ids=tuple(value["representative_sample_ids"]))

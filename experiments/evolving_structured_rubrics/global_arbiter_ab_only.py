"""VL-RewardBench single-prompt Global-Arbiter experiment.

The arbiter is asked to prefer A or B, while A, B, and None are all valid
semantic outputs.  None is an abstention, not a parse failure.  Only transport,
empty-response, malformed-JSON, and invalid-label failures are retried with the
same prompt.  Frozen S0, S3, and S4 artifacts are read only and used as controls.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from critiq.structured.backend_pool import BackendPoolSpec
from critiq.utils import parse_json

from . import _global_arbiter_ab_only_support as support
from . import run_rubric_evolution as base
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_phase10 as vlrb_metrics
from .experiment_utils import atomic_write_json, canonical_sha256, load_json
from .rubric_factory import file_sha256


SCHEMA_VERSION = "1.0.0"
PROTOCOL_VERSION = "global-arbiter-ab-only-v1-vlrb-only"
PROMPT_VERSION = "global-arbiter-evidence-synthesis-ab-only-v1"
EXPERIMENT_DIR = "vl_rewardbench_global_arbiter_ab_only_v1"
CONFIG_KEY = "global_arbiter_ab_only_experiment"
SYSTEM_NAME = "s5_global_arbiter_ab_only"
REQUEST_SCOPE = "s5_global_arbiter_ab_only_all_requests"
REQUEST_KIND = "global_arbiter_ab_only"
FORMAL_AGGREGATION_ORDER = (
    "five_reports_to_ab_only_arbiter_per_replicate_then_k3")
LEGACY_S5_EXPERIMENT_DIR: str | None = None
LEGACY_S5_SYSTEM_NAME = "s5_global_arbiter_ab_only"
REPORT_TITLE = "VL-RewardBench Single-Prompt Global Arbiter v1"
SOURCE_S4_EXPERIMENT_DIR = support.SOURCE_S4_EXPERIMENT
K = support.K
VLRB_COUNT = support.VLRB_COUNT
ENDPOINT_IDS = support.ENDPOINT_IDS

STAGES = (
    "vlrb-global-arbiter-ab-only-freeze",
    "vlrb-global-arbiter-ab-only-audit",
    "vlrb-global-arbiter-ab-only-smoke",
    "vlrb-global-arbiter-ab-only-run",
    "vlrb-global-arbiter-ab-only-retry",
    "vlrb-global-arbiter-ab-only-report",
)

GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT = """## Instruction

You are the final decision arbiter for one structured multimodal preference system. You are given the image, the source instruction or question, two candidate responses, and five subtree assessments produced under complementary rubric dimensions.

Treat the subtree assessments as correlated evidence, not independent votes. Do not decide by counting their A/B labels. Independently verify the image, question, and candidate responses. A subtree assessment may be incorrect or inapplicable.

Prioritize verifiable visual and factual correctness. Consider completeness after factual validity; clarity and creativity may distinguish otherwise acceptable responses but cannot compensate for factual errors.

Return a relative preference for every pair. If both responses are imperfect or the evidence is limited, select the response with stronger support and the less severe error. Do not abstain.

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A using the image and subtree evidence.",
    "analysis_b": "Analyze B using the image and subtree evidence.",
    "thought": "Integrate the evidence into one relative preference.",
    "answer": "A / B"
}
```
"""


def _target(output: Path) -> Path:
    return output.parent / EXPERIMENT_DIR


def _source_s4_target(output: Path) -> Path:
    return output.parent / SOURCE_S4_EXPERIMENT_DIR


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "source_vlrb_experiment": support.SOURCE_S3_EXPERIMENT,
        "source_s4_experiment": SOURCE_S4_EXPERIMENT_DIR,
        "source_rubric_experiment": support.SOURCE_RUBRIC_EXPERIMENT,
        "source_epoch": support.SOURCE_RUBRIC_EPOCH,
        "vl_rewardbench_only": True,
        "k": K,
        "generation_seed_policy": "unset",
        "temperature": 0.5,
        "max_tokens": 2048,
        "max_parse_retries": 10,
        "smoke_sample_count": 20,
        "endpoint_ids": list(ENDPOINT_IDS),
        "scheduler": "sample_bundle_available_slot_affinity",
        "new_request_scope": REQUEST_SCOPE,
        "semantic_answer_space": ["A", "B", "None"],
        "selection_after_benchmark_forbidden": True,
    }
    value = config.get(CONFIG_KEY)
    if value != expected:
        raise RuntimeError(
            f"{CONFIG_KEY} does not match the frozen protocol")
    return dict(value)


def parse_global_arbiter_ab_only_response(raw: object) -> dict[str, str]:
    if not isinstance(raw, str):
        raise ValueError("A/B-only Global-Arbiter response must be text")
    payload = parse_json(raw, allow_invalid_escapes=True)
    if not isinstance(payload, dict):
        raise ValueError("A/B-only Global-Arbiter response must be one JSON object")
    answer = payload.get("answer")
    if answer not in {"A", "B", "None"}:
        raise ValueError("Global-Arbiter answer must be A, B, or None")
    return {"answer": str(answer)}


def _controls(
    output: Path, records: Sequence[Mapping[str, Any]], rubric: Any,
) -> tuple[dict[str, list[list[int | None]]], dict[str, Any]]:
    return support.source_controls(output, records, rubric)


def _manifest(
    config: Mapping[str, Any], output: Path, *, include_endpoints: bool,
) -> dict[str, Any]:
    settings = _settings(config)
    rubric = support.rubric(output)
    records = support.records(output)
    controls, provenance = _controls(output, records, rubric)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in records])
    source_s3 = output.parent / support.SOURCE_S3_EXPERIMENT
    source_s4 = _source_s4_target(output)
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "settings": settings,
        "settings_sha256": canonical_sha256(settings),
        "vl_rewardbench": {
            "count": len(records),
            "semantic_sha256": canonical_sha256([{
                key: record[key]
                for key in ("sample_id", "benchmark_id", "question", "responses",
                            "preferred_original_index", "image_sha256", "group")
            } for record in records]),
        },
        "rubric_sha256": rubric.rubric_sha256,
        "root_ids": list(rubric.root_ids),
        "node_count": len(rubric.nodes),
        "schedule_sha256": canonical_sha256(schedule),
        "system_prompt_sha256": hashlib.sha256(
            GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "source_artifact_sha256": {
            "s3_manifest": file_sha256(source_s3 / "frozen_manifest.json"),
            "s3_prediction": file_sha256(source_s3 / "predictions" / "vlrb_full.json"),
            "s3_report": file_sha256(source_s3 / "final_report.json"),
            "s4_manifest": file_sha256(source_s4 / "frozen_manifest.json"),
            "s4_prediction": file_sha256(
                source_s4 / "predictions" / "arbiter_full.json"),
            "s4_report": file_sha256(source_s4 / "final_report.json"),
        },
        "source_control_provenance": provenance,
        "control_vote_matrices_sha256": canonical_sha256(controls),
        "new_request_scope": REQUEST_SCOPE,
        "formal_aggregation_order": FORMAL_AGGREGATION_ORDER,
        "semantic_answer_space": ["A", "B", "None"],
    }
    if LEGACY_S5_EXPERIMENT_DIR is not None:
        legacy_report = output.parent / LEGACY_S5_EXPERIMENT_DIR / "final_report.json"
        value["source_artifact_sha256"]["legacy_s5_report"] = file_sha256(
            legacy_report)
    if include_endpoints:
        value["endpoint_identities"] = base._inspect_endpoints(
            config, BackendPoolSpec.from_dict(config["backend_pool"]))
    return value


def _manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scientific identity, excluding descriptive provenance shape.

    The compact runner validates frozen control vote matrices directly.  Its
    shorter provenance payload is therefore allowed to differ from the legacy
    multi-runner implementation without invalidating existing model caches.
    """

    normalized = {
        key: item for key, item in value.items()
        if key != "source_control_provenance"
    }
    # Parser-only migration: the model prompt, decoding parameters, data,
    # schedule, and source evidence remain byte-identical.  Accept manifests
    # frozen before None was correctly classified as a semantic abstention.
    settings = dict(normalized.get("settings", {}))
    if settings.get("semantic_answer_space") == ["A", "B"]:
        settings["semantic_answer_space"] = ["A", "B", "None"]
        normalized["settings"] = settings
        normalized["settings_sha256"] = canonical_sha256(settings)
    if normalized.get("semantic_answer_space") == ["A", "B"]:
        normalized["semantic_answer_space"] = ["A", "B", "None"]
    return normalized


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    manifest = _manifest(config, output, include_endpoints=True)
    source_s3_manifest = load_json(
        output.parent / support.SOURCE_S3_EXPERIMENT / "frozen_manifest.json")
    source_s4_manifest = load_json(_source_s4_target(output) / "frozen_manifest.json")
    if (manifest["endpoint_identities"] != source_s3_manifest.get("endpoint_identities")
            or manifest["endpoint_identities"]
            != source_s4_manifest.get("endpoint_identities")):
        raise RuntimeError("A/B-only live endpoints do not match frozen controls")
    records = support.records(output)
    rubric = support.rubric(output)
    _, bundle_manifest = support.source_index(output, records, rubric)
    path = target / "frozen_manifest.json"
    if (path.is_file()
            and _manifest_identity(load_json(path)) != _manifest_identity(manifest)):
        raise RuntimeError("A/B-only Global-Arbiter frozen manifest drift")
    atomic_write_json(path, manifest)
    atomic_write_json(target / "arbiter_bundle_manifest.json", bundle_manifest)
    atomic_write_json(target / "source_reuse_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "source_s3_experiment": support.SOURCE_S3_EXPERIMENT,
        "source_s4_experiment": SOURCE_S4_EXPERIMENT_DIR,
        "source_artifact_sha256": manifest["source_artifact_sha256"],
        "source_root_report_count": bundle_manifest["source_root_report_count"],
        "source_reports_generated": 0,
        "s4_predictions_used_as_model_input": False,
    })
    atomic_write_json(target / "prompt_spec.json", {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        "parser": "answer_A_or_B_or_None",
        "user_prompt_order": ["question", "A", "B", "five_subtree_reports"],
    })
    atomic_write_json(target / "ab_schedule.json", support.source_schedule(
        output, [str(record["sample_id"]) for record in records]))
    details = {
        "record_count": len(records),
        "arbiter_logical_request_count": len(records) * K,
        "reused_subtree_report_count": bundle_manifest["source_root_report_count"],
        "rubric_sha256": manifest["rubric_sha256"],
    }
    support.status(target, STAGES[0], details)
    print(json.dumps(details, indent=2))


def _load_frozen(
    config: Mapping[str, Any], output: Path, *, validate_source_bundles: bool = True,
):
    target = _target(output)
    support.require(target, STAGES[0])
    manifest = load_json(target / "frozen_manifest.json")
    expected = _manifest(config, output, include_endpoints=False)
    for key, value in _manifest_identity(expected).items():
        if manifest.get(key) != value:
            raise RuntimeError(f"A/B-only frozen manifest drift: {key}")
    records = support.records(output)
    rubric = support.rubric(output)
    if validate_source_bundles:
        source_by_id, bundle_manifest = support.source_index(output, records, rubric)
        if load_json(target / "arbiter_bundle_manifest.json") != bundle_manifest:
            raise RuntimeError("A/B-only source report bundle drift")
    else:
        source_by_id = {
            str(sample["sample_id"]): sample
            for sample in support.source_prediction(output)["samples"]
        }
    return target, manifest, records, rubric, source_by_id


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    actual = base._inspect_endpoints(
        config, BackendPoolSpec.from_dict(config["backend_pool"]))
    if actual != manifest.get("endpoint_identities"):
        raise RuntimeError("A/B-only Global-Arbiter endpoint identity drift")


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rubric, source_by_id = _load_frozen(config, output)
    first_record = records[0]
    source = source_by_id[str(first_record["sample_id"])]
    row = support.vlrb_rows((first_record,))[0]
    ordered = support.ordered_row(row, int(source["orders"][0]), 0)
    reports, _, _ = support.source_report_bundle(output, source, 0, rubric)
    prompt = support.global_arbiter_user_prompt(ordered, reports)
    controls, _ = _controls(output, records, rubric)
    checks = {
        "vlrb_only": manifest["settings"]["vl_rewardbench_only"] is True,
        "record_count_1247": len(records) == VLRB_COUNT,
        "five_roots": len(rubric.root_ids) == 5,
        "three_requests_per_sample": K == 3,
        "all_requests_regenerated": manifest["new_request_scope"]
            == REQUEST_SCOPE,
        "semantic_answer_space": _manifest_identity(manifest)[
            "semantic_answer_space"] == ["A", "B", "None"],
        "parser_accepts_a": parse_global_arbiter_ab_only_response(
            '{"answer":"A"}') == {"answer": "A"},
        "parser_accepts_b": parse_global_arbiter_ab_only_response(
            '{"answer":"B"}') == {"answer": "B"},
        "parser_accepts_none": parse_global_arbiter_ab_only_response(
            '{"answer":"None"}') == {"answer": "None"},
        "prompt_schema_ab_only": '"answer": "A / B"'
            in GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        "prompt_requires_relative_choice": "Do not abstain"
            in GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        "source_reports_complete": load_json(
            target / "arbiter_bundle_manifest.json")["source_root_report_count"]
            == len(records) * K * len(rubric.root_ids),
        "source_reports_read_only": load_json(
            target / "source_reuse_manifest.json")["source_reports_generated"] == 0,
        "s4_not_model_input": load_json(
            target / "source_reuse_manifest.json")[
                "s4_predictions_used_as_model_input"] is False,
        "same_replicate_bundle": all(
            item["root_id"] == root_id
            for item, root_id in zip(reports, rubric.root_ids)),
        "sample_prefix_before_reports": prompt.index("## Candidate B")
            < prompt.index("## Subtree Assessments"),
        "no_gold_in_prompt": "gold" not in (
            GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT + prompt).lower(),
        "no_majority_result_in_prompt": "root majority" not in prompt.lower(),
        "controls_exact": set(controls) == {
            "s0_explicit_recursive", "s3_unified_subtree", "s4_global_arbiter"},
        "formal_order_frozen": manifest["formal_aggregation_order"]
            == FORMAL_AGGREGATION_ORDER,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A/B-only Global-Arbiter audit failed: {checks}")
    atomic_write_json(target / "offline_audit.json", {
        "schema_version": SCHEMA_VERSION, "offline_only": True, "checks": checks})
    support.status(target, STAGES[1], checks)
    print(json.dumps(checks, indent=2))


def _run_bundles(
    config: Mapping[str, Any], output: Path, target: Path,
    records: Sequence[Mapping[str, Any]], source_by_id: Mapping[str, Any],
    rubric: Any, *, label: str, total_attempt_limit: int,
) -> dict[str, Any]:
    return support.run_bundles(
        config, output, target, records, source_by_id, rubric,
        label=label,
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=PROMPT_VERSION,
        system_prompt=GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        response_parser=parse_global_arbiter_ab_only_response,
        settings_loader=_settings,
        request_kind=REQUEST_KIND,
    )


def _s5_vote_matrices(result: Mapping[str, Any]) -> list[list[int | None]]:
    votes: list[list[int | None]] = [[] for _ in range(K)]
    for sample in result["samples"]:
        for replicate in range(K):
            call = sample["arbiter"][str(replicate)]
            if not call.get("parse_ok"):
                raise RuntimeError("S5 contains an unresolved technical failure")
            answer = (call.get("parsed") or {}).get("answer")
            if answer not in {"A", "B", "None"}:
                raise RuntimeError("S5 contains an invalid semantic answer")
            original = support.display_to_original(
                answer, int(sample["orders"][replicate]))
            votes[replicate].append(original)
    return votes


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rubric, source_by_id = _load_frozen(config, output)
    support.require(target, STAGES[1])
    _verify_live(config, manifest)
    selected = support.selected_records(records, _settings(config)["smoke_sample_count"])
    result = _run_bundles(
        config, output, target, selected, source_by_id, rubric,
        label="arbiter_ab_only_smoke20",
        total_attempt_limit=1 + _settings(config)["max_parse_retries"])
    telemetry = support.telemetry(result)
    votes = _s5_vote_matrices(result)
    details = {
        "sample_count": len(selected),
        "logical_request_count": telemetry["logical_request_count"],
        "parse_valid_rate": telemetry["parse_valid_rate"],
        "semantic_none_count": sum(
            value is None for replicate in votes for value in replicate),
        "wall_seconds": telemetry["wall_seconds"],
    }
    support.status(target, STAGES[2], details)
    print(json.dumps(details, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rubric, source_by_id = _load_frozen(config, output)
    support.require(target, STAGES[2])
    _verify_live(config, manifest)
    result = _run_bundles(
        config, output, target, records, source_by_id, rubric,
        label="arbiter_ab_only_full", total_attempt_limit=1)
    telemetry = support.telemetry(result)
    failures = support.parse_failures(result)
    votes = _s5_vote_matrices(result) if not failures["failure_count"] else None
    atomic_write_json(target / "parse_failures.json", failures)
    details = {
        "sample_count": len(records),
        "logical_request_count": telemetry["logical_request_count"],
        "parse_valid_rate": telemetry["parse_valid_rate"],
        "unresolved_technical_failures": failures["failure_count"],
        "semantic_none_count": (sum(
            value is None for replicate in votes for value in replicate)
            if votes is not None else None),
        "wall_seconds": telemetry["wall_seconds"],
    }
    support.status(target, STAGES[3], details)
    print(json.dumps(details, indent=2))


def _retry_attempt_limit(target: Path, max_parse_retries: int) -> tuple[int, dict[str, Any]]:
    path = target / "predictions" / "arbiter_ab_only_full.json"
    base_limit = 1 + max_parse_retries
    if not path.is_file():
        return base_limit, {
            "previous_unresolved_technical_failures": None,
            "previous_failed_max_generation_count": 0,
            "target_total_attempt_limit": base_limit,
            "additional_retry_attempts_per_failed_call": max_parse_retries,
        }
    prediction = load_json(path)
    failures = support.parse_failures(prediction)["failures"]
    legacy_rescues = [
        call for sample in prediction.get("samples", ())
        for call in sample.get("arbiter", {}).values()
        if call.get("recovery_mode") == "mandatory_ab_tiebreak"
    ]
    if legacy_rescues:
        previous_limit = max(int(call.get(
            "primary_model_generation_count",
            call.get("model_generation_count", 0))) for call in legacy_rescues)
        target_limit = previous_limit + max_parse_retries
        return target_limit, {
            "previous_unresolved_technical_failures": len(legacy_rescues),
            "previous_failed_max_generation_count": previous_limit,
            "target_total_attempt_limit": target_limit,
            "additional_retry_attempts_per_failed_call": max_parse_retries,
            "legacy_tiebreak_calls_to_remove": len(legacy_rescues),
        }
    if not failures:
        return base_limit, {
            "previous_unresolved_technical_failures": 0,
            "previous_failed_max_generation_count": 0,
            "target_total_attempt_limit": base_limit,
            "additional_retry_attempts_per_failed_call": 0,
            "legacy_tiebreak_calls_to_remove": 0,
        }
    previous_limit = max(int(item.get("model_generation_count", 0))
                         for item in failures)
    target_limit = previous_limit + max_parse_retries
    return target_limit, {
        "previous_unresolved_technical_failures": len(failures),
        "previous_failed_max_generation_count": previous_limit,
        "target_total_attempt_limit": target_limit,
        "additional_retry_attempts_per_failed_call": (
            target_limit - previous_limit),
        "legacy_tiebreak_calls_to_remove": 0,
    }


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rubric, source_by_id = _load_frozen(config, output)
    support.require(target, STAGES[3])
    _verify_live(config, manifest)
    total_attempt_limit, retry_details = _retry_attempt_limit(
        target, _settings(config)["max_parse_retries"])
    result = _run_bundles(
        config, output, target, records, source_by_id, rubric,
        label="arbiter_ab_only_full",
        total_attempt_limit=total_attempt_limit)
    telemetry = support.telemetry(result)
    failures = support.parse_failures(result)
    votes = _s5_vote_matrices(result) if not failures["failure_count"] else None
    atomic_write_json(target / "parse_failures.json", failures)
    details = {
        "unresolved_technical_failures": failures["failure_count"],
        "parse_valid_rate": telemetry["parse_valid_rate"],
        "model_generation_count": telemetry["model_generation_count"],
        "semantic_none_count": (sum(
            value is None for replicate in votes for value in replicate)
            if votes is not None else None),
        **retry_details,
    }
    atomic_write_json(target / "retry_summary.json", {
        "schema_version": SCHEMA_VERSION, **details})
    support.status(target, STAGES[4], details)
    print(json.dumps(details, indent=2))


def _state(prediction: int | None, gold: int) -> str:
    if prediction is None:
        return "none"
    return "correct" if prediction == gold else "wrong"


def _former_s4_none_subset(
    records: Sequence[Mapping[str, Any]], s0: Sequence[int | None],
    s4: Sequence[int | None], s5: Sequence[int | None],
) -> dict[str, Any]:
    indices = [index for index, value in enumerate(s4) if value is None]
    correct = sum(
        s5[index] == int(records[index]["preferred_original_index"])
        for index in indices)
    by_group = {}
    for group in ("general", "hallucination", "reasoning"):
        group_indices = [index for index in indices if records[index]["group"] == group]
        group_correct = sum(
            s5[index] == int(records[index]["preferred_original_index"])
            for index in group_indices)
        by_group[group] = {
            "sample_count": len(group_indices),
            "correct_count": group_correct,
            "accuracy": group_correct / len(group_indices) if group_indices else 0.0,
        }
    s0_correct = sum(
        prediction == int(record["preferred_original_index"])
        for prediction, record in zip(s0, records))
    s4_correct = sum(
        prediction == int(record["preferred_original_index"])
        for prediction, record in zip(s4, records))
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(indices),
        "correct_count": correct,
        "accuracy": correct / len(indices) if indices else 0.0,
        "required_correct_to_match_s0": max(0, s0_correct - s4_correct),
        "required_correct_to_exceed_s0": max(0, s0_correct - s4_correct + 1),
        "by_group": by_group,
        "sample_ids": [str(records[index]["sample_id"]) for index in indices],
    }


def _transition(
    records: Sequence[Mapping[str, Any]], before: Sequence[int | None],
    after: Sequence[int | None],
) -> dict[str, Any]:
    counts = Counter(
        (_state(left, int(record["preferred_original_index"])),
         _state(right, int(record["preferred_original_index"])))
        for record, left, right in zip(records, before, after))
    return {
        "schema_version": SCHEMA_VERSION,
        "changed_prediction_count": sum(left != right for left, right in zip(before, after)),
        "transitions": {
            f"{left}->{right}": value
            for (left, right), value in sorted(counts.items())
        },
    }


def _case_rows(
    records: Sequence[Mapping[str, Any]], before: Sequence[int | None],
    after: Sequence[int | None], sample_ids: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = set(sample_ids)
    return [{
        "sample_id": record["sample_id"],
        "benchmark_id": record["benchmark_id"],
        "group": record["group"],
        "query_source": record["query_source"],
        "question": record["question"],
        "A": record["responses"][0],
        "B": record["responses"][1],
        "gold_original_index": record["preferred_original_index"],
        "s4_original_index": before[index],
        "s5_original_index": after[index],
    } for index, record in enumerate(records)
        if str(record["sample_id"]) in wanted]


def report(config: Mapping[str, Any], output: Path) -> None:
    target, _, records, rubric, _ = _load_frozen(
        config, output, validate_source_bundles=False)
    support.require(target, STAGES[4])
    failures = load_json(target / "parse_failures.json")
    if failures.get("failure_count") != 0:
        raise RuntimeError(
            "A/B-only report requires zero unresolved technical failures; rerun retry")
    result = load_json(target / "predictions" / "arbiter_ab_only_full.json")
    if any(
        call.get("recovery_mode") == "mandatory_ab_tiebreak"
        for sample in result.get("samples", ())
        for call in sample.get("arbiter", {}).values()
    ):
        raise RuntimeError(
            "legacy tie-break outputs are not valid under the single-prompt "
            "protocol; rerun vlrb-global-arbiter-ab-only-retry")
    controls, provenance = _controls(output, records, rubric)
    s5_votes = _s5_vote_matrices(result)
    if LEGACY_S5_EXPERIMENT_DIR is not None:
        legacy = load_json(
            output.parent / LEGACY_S5_EXPERIMENT_DIR / "final_report.json")
        legacy_system = legacy.get("systems", {}).get(LEGACY_S5_SYSTEM_NAME)
        if not isinstance(legacy_system, dict):
            raise RuntimeError("legacy S5 comparison system is missing")
        legacy_votes = legacy_system.get("votes_by_replicate")
        if (not isinstance(legacy_votes, list) or len(legacy_votes) != K
                or any(len(votes) != len(records) for votes in legacy_votes)):
            raise RuntimeError("legacy S5 comparison vote matrix is invalid")
        controls = {**controls, "s5_legacy_ab_only_reference": legacy_votes}
    votes_by_system = {**controls, SYSTEM_NAME: s5_votes}
    systems = {
        name: {
            "votes_by_replicate": votes,
            "metrics": vlrb_metrics._system_metrics(records, votes),
        }
        for name, votes in votes_by_system.items()
    }
    s5_metrics = systems[SYSTEM_NAME]["metrics"]
    comparisons = {
        name: vlrb._paired(
            records, systems[baseline]["metrics"]["original_index_predictions"],
            s5_metrics["original_index_predictions"])
        for name, baseline in (
            ("s5_vs_s4", "s4_global_arbiter"),
            ("s5_vs_s3", "s3_unified_subtree"),
            ("s5_vs_s0", "s0_explicit_recursive"),
        )
    }
    if "s5_legacy_ab_only_reference" in systems:
        comparisons["s5_v2_vs_legacy_s5"] = vlrb._paired(
            records,
            systems["s5_legacy_ab_only_reference"]["metrics"][
                "original_index_predictions"],
            s5_metrics["original_index_predictions"],
        )
    s0_predictions = systems["s0_explicit_recursive"]["metrics"][
        "original_index_predictions"]
    s4_predictions = systems["s4_global_arbiter"]["metrics"][
        "original_index_predictions"]
    s5_predictions = s5_metrics["original_index_predictions"]
    former_none = _former_s4_none_subset(
        records, s0_predictions, s4_predictions, s5_predictions)
    transition = _transition(records, s4_predictions, s5_predictions)
    source_result = support.source_prediction(output)
    override = support.root_majority_override_audit(
        records, source_result, result, rubric)
    category_source = {
        name: {
            "groups": item["metrics"]["groups"],
            "sources": item["metrics"]["source_groups"],
        }
        for name, item in systems.items()
    }
    case_audit = {
        "corrected": _case_rows(
            records, s4_predictions, s5_predictions,
            comparisons["s5_vs_s4"]["corrected_sample_ids"][:10]),
        "harmed": _case_rows(
            records, s4_predictions, s5_predictions,
            comparisons["s5_vs_s4"]["harmed_sample_ids"][:10]),
    }
    telemetry = support.telemetry(result)
    status = load_json(target / "stage_status.json")["stages"]
    run_wall = float(status[STAGES[3]]["details"]["wall_seconds"])
    s0_requests = len(records) * K * len(rubric.nodes)
    full_s5_requests = len(records) * K * (len(rubric.root_ids) + 1)
    efficiency = {
        "s0_logical_request_count": s0_requests,
        "reused_s3_root_report_count": len(records) * K * len(rubric.root_ids),
        "new_s5_arbiter_request_count": len(records) * K,
        "full_s5_system_request_count": full_s5_requests,
        "s5_system_request_reduction_vs_s0": 1.0 - full_s5_requests / s0_requests,
        "full_run_wall_seconds": run_wall,
        "full_run_new_requests_per_minute": len(records) * K / run_wall * 60
        if run_wall else 0.0,
        "final_cache_telemetry": telemetry,
    }
    response_distribution = {
        "final": dict(Counter(
            "None" if value is None else str(value) for value in s5_predictions)),
        "by_replicate": [dict(Counter(str(value) for value in votes))
                         for votes in s5_votes],
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "systems": systems,
        "paired": comparisons,
        "former_s4_none_subset": former_none,
        "s4_to_s5_transition": transition,
        "root_majority_override_audit": override,
        "category_source_analysis": category_source,
        "response_distribution": response_distribution,
        "efficiency": efficiency,
        "control_provenance": provenance,
        "formal_aggregation_order": FORMAL_AGGREGATION_ORDER,
        "new_model_request_scope": REQUEST_SCOPE,
        "selection_after_vlrb_forbidden": True,
    }
    for name, payload in (
        ("paired_comparison.json", comparisons),
        ("former_s4_none_subset.json", former_none),
        ("s4_to_s5_transition.json", transition),
        ("root_majority_override_audit.json", override),
        ("category_source_analysis.json", category_source),
        ("case_audit.json", case_audit),
        ("efficiency.json", efficiency),
        ("response_distribution.json", response_distribution),
        ("final_report.json", value),
    ):
        atomic_write_json(target / name, payload)

    lines = [
        f"# {REPORT_TITLE}", "",
        "The Arbiter is prompted to choose A or B, while A, B, and None are "
        "all parsed as valid semantic outputs. Five same-replicate subtree "
        "reports are synthesized before final K=3.", "",
        "| System | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in systems.items():
        metrics = item["metrics"]
        lines.append(
            f"| {name} | {metrics['strict_accuracy']:.2%} | "
            f"{metrics['overall_acc']:.2%} | {metrics['macro_acc']:.2%} | "
            f"{metrics['coverage']:.2%} | {metrics['correct_count']} |")
    lines.extend([
        "", "## Paired comparisons", "",
        "| Comparison | Corrected | Harmed | Net | Exact McNemar p |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in comparisons.items():
        lines.append(
            f"| {name} | {item['corrected_count']} | {item['harmed_count']} | "
            f"{item['net_corrected']} | {item['mcnemar_exact_two_sided_p']:.6g} |")
    lines.extend([
        "", "## Former S4 None subset", "",
        f"- Samples: {former_none['sample_count']}",
        f"- S5 correct: {former_none['correct_count']}",
        f"- S5 accuracy: {former_none['accuracy']:.2%}",
        f"- Correct needed to match S0 from the frozen S4 count: "
        f"{former_none['required_correct_to_match_s0']}",
        f"- Correct needed to exceed S0 from the frozen S4 count: "
        f"{former_none['required_correct_to_exceed_s0']}",
        "", "## Efficiency", "",
        f"- New S5 Arbiter requests: {efficiency['new_s5_arbiter_request_count']}",
        f"- Full S5 system requests: {efficiency['full_s5_system_request_count']}",
        f"- Request reduction vs S0: {efficiency['s5_system_request_reduction_vs_s0']:.2%}",
        f"- Full-run wall time: {run_wall:.1f}s",
        f"- Full-run throughput: {efficiency['full_run_new_requests_per_minute']:.2f} new requests/min",
        "", "A/B/None are valid semantic outputs. OverallAcc is computed on "
        "covered A/B decisions; Strict ACC counts final None as incorrect.",
    ])
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        name: {
            "strict_accuracy": item["metrics"]["strict_accuracy"],
            "overall_acc": item["metrics"]["overall_acc"],
            "macro_acc": item["metrics"]["macro_acc"],
            "coverage": item["metrics"]["coverage"],
            "correct_count": item["metrics"]["correct_count"],
        }
        for name, item in systems.items()
    }
    summary["former_s4_none_subset"] = {
        "sample_count": former_none["sample_count"],
        "correct_count": former_none["correct_count"],
        "accuracy": former_none["accuracy"],
    }
    support.status(target, STAGES[5], summary)
    print(json.dumps(summary, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze,
        STAGES[1]: audit,
        STAGES[2]: smoke,
        STAGES[3]: run,
        STAGES[4]: retry,
        STAGES[5]: report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported A/B-only Global-Arbiter stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

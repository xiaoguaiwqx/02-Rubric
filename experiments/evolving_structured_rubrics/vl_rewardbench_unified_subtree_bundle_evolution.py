"""VL-RewardBench evaluation for Phase21 root-subtree bundle evolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from critiq.structured.schema import StructuredRubric

from . import aligned_system_runtime as system
from . import _global_arbiter_ab_only_support as support
from . import run_rubric_evolution as base
from . import unified_subtree_bundle_evolution as phase21
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_aligned_evolution as shared
from . import vl_rewardbench_phase10 as vlrb_metrics
from .experiment_utils import atomic_write_json, load_json
from .evolution_protocol import BenchmarkProtocol, EvolutionProtocol
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_unified_subtree_bundle_evolution_v1"
PROTOCOL_VERSION = "vlrb-unified-root-subtree-bundle-evolution-v1"
STAGES = (
    "vlrb-subtree-bundle-freeze",
    "vlrb-subtree-bundle-audit",
    "vlrb-subtree-bundle-smoke",
    "vlrb-subtree-bundle-run",
    "vlrb-subtree-bundle-retry",
    "vlrb-subtree-bundle-report",
)
SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": phase21.EXPERIMENT_DIR,
    "control_experiment": shared.CONTROL_EXPERIMENT,
    "control_system": shared.CONTROL_SYSTEM,
    "dataset_count": 1247,
    "k": 3,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "endpoint_ids": ["vllm-8000", "vllm-8001"],
    "scheduler": "sample_bundle_available_slot_affinity",
    "smoke_sample_count": 20,
    "systems": ["initial_five_root", "phase17_e4_control", "phase21_final"],
    "selection_after_benchmark_forbidden": True,
}
FINAL_LABEL = "phase21_final"
SOURCE_RUBRIC_KEY = "phase21_rubric_sha256"
REPORT_TITLE = "Phase21 Unified Root-Subtree Bundle Evolution"


DEFAULT_PROTOCOL = BenchmarkProtocol(
    identity=EvolutionProtocol.from_settings(
        experiment_dir=EXPERIMENT_DIR, version=PROTOCOL_VERSION,
        config_key="vlrb_unified_subtree_bundle_evolution_v1_experiment",
        stages=STAGES, settings=SETTINGS,
    ),
    source=phase21.DEFAULT_PROTOCOL, final_label=FINAL_LABEL,
    source_rubric_key=SOURCE_RUBRIC_KEY, report_title=REPORT_TITLE,
)


def _target(output: Path, *, protocol: BenchmarkProtocol = DEFAULT_PROTOCOL) -> Path:
    return output.parent / protocol.experiment_dir


def _settings(
    config: Mapping[str, Any], *, protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    return protocol.validate(config)


def _runtime_settings(
    config: Mapping[str, Any], *, protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> system.RuntimeSettings:
    value = _settings(config, protocol=protocol)
    return system.RuntimeSettings(
        temperature=float(value["temperature"]),
        max_tokens=int(value["max_tokens"]),
        max_parse_retries=int(value["max_parse_retries"]),
        generation_seed_policy=str(value["generation_seed_policy"]))


def _rubrics(
    output: Path, *, protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> tuple[StructuredRubric, StructuredRubric]:
    source = output / protocol.source.experiment_dir
    initial = StructuredRubric.load_json(
        source / "epochs" / "epoch_00" / "rubric_initial.json")
    final = StructuredRubric.load_json(source / "final" / "rubric.json")
    return initial, final


def _initial_source(output: Path) -> tuple[Path, Path]:
    target = output.parent / "vl_rewardbench_local_unified_subtree_competition_v1"
    return target / "predictions" / "initial.json", target / "cache" / "initial"


def _request_payload(
    config: Mapping[str, Any], endpoint: Any, *, prompt_version: str,
    system_prompt: str, user_text: str, row: Mapping[str, Any],
    request_key: Mapping[str, Any], image_sha256: str | None = None,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    settings = _runtime_settings(config, protocol=protocol)
    return {
        "schema_version": support.SCHEMA_VERSION,
        "protocol_version": system.PROTOCOL_VERSION,
        "prompt_version": prompt_version,
        "model": config["model"],
        "endpoint_checkpoint": endpoint.checkpoint_root,
        "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
        "image_sha256": image_sha256 or file_sha256(
            Path(str(row["image_path"]))),
        "request": {
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "generation_seed_policy": settings.generation_seed_policy,
        },
        "request_key": dict(request_key),
    }


def _validate_reusable_initial(
    config: Mapping[str, Any], output: Path, rows: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Sequence[int]], rubric: StructuredRubric,
    *, protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    path, cache_root = _initial_source(output)
    if not path.is_file():
        raise RuntimeError(
            "strict Phase21 reuse requires the existing Phase20 Initial artifact")
    value = system.load(path)
    if (value["rubric_sha256"] != rubric.rubric_sha256
            or int(value["k"]) != protocol.settings["k"]
            or [str(item["sample_id"]) for item in value["samples"]]
            != [str(row["sample_id"]) for row in rows]):
        raise RuntimeError("reusable Initial artifact identity drift")
    endpoints = {
        item.endpoint_id: item for item in base.BackendPoolSpec.from_dict(
            config["backend_pool"]).endpoints}
    by_id = {str(row["sample_id"]): row for row in rows}
    verified_calls = 0
    for sample in value["samples"]:
        sample_id = str(sample["sample_id"])
        row = by_id[sample_id]
        image_sha = file_sha256(Path(str(row["image_path"])))
        expected_orders = tuple(int(item) for item in schedule[sample_id])
        if tuple(int(item) for item in sample["orders"]) != expected_orders:
            raise RuntimeError("reusable Initial A/B schedule drift")
        for replicate, order in enumerate(expected_orders):
            displayed = support.ordered_row(row, order, replicate)
            item = sample["replicates"][str(replicate)]
            calls = item["subtrees"]
            for root_id in rubric.root_ids:
                call = calls[root_id]
                parsed = system.unified.parse_subtree_response(call["raw_response"])
                if not call.get("parse_ok") or parsed != call.get("parsed"):
                    raise RuntimeError("reusable Initial subtree parser drift")
                endpoint = endpoints[call["endpoint_id"]]
                user_text = system.unified.subtree_user_prompt(
                    displayed, rubric, root_id)
                payload = _request_payload(
                    config, endpoint,
                    prompt_version=system.unified.SUBTREE_PROMPT_VERSION,
                    system_prompt=system.unified.UNIFIED_SUBTREE_SYSTEM_PROMPT,
                    user_text=user_text, row=displayed, image_sha256=image_sha,
                    request_key={
                        "kind": "aligned_unified_subtree",
                        "split": "vlrb_initial",
                        "sample_id": str(displayed["sample_id"]),
                        "root_id": root_id,
                        "root_subtree_sha256": system._root_subtree_sha256(
                            rubric, root_id),
                        "replicate": replicate, "order": order,
                    }, protocol=protocol)
                key = base.canonical_sha256(payload)
                cache_path = cache_root / "subtree" / key[:2] / f"{key}.json"
                if call.get("cache_key") != key or not cache_path.is_file():
                    raise RuntimeError("reusable Initial subtree cache identity drift")
                cached = load_json(cache_path)
                if cached.get("request") != payload:
                    raise RuntimeError("reusable Initial subtree request drift")
                verified_calls += 1
            reports = system._reports_from_calls(rubric, calls)
            arbiter = item["arbiter"]
            parsed_arbiter = system.arbiter.parse_global_arbiter_ab_only_response(
                arbiter["raw_response"])
            if not arbiter.get("parse_ok") or parsed_arbiter != arbiter.get("parsed"):
                raise RuntimeError("reusable Initial Arbiter parser drift")
            endpoint = endpoints[arbiter["endpoint_id"]]
            user_text = support.global_arbiter_user_prompt(displayed, reports)
            bundle_sha = base.canonical_sha256(reports)
            payload = _request_payload(
                config, endpoint,
                prompt_version=system.unified.ARBITER_PROMPT_VERSION,
                system_prompt=system.arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
                user_text=user_text, row=displayed, image_sha256=image_sha,
                request_key={
                    "kind": "aligned_global_arbiter", "split": "vlrb_initial",
                    "sample_id": str(displayed["sample_id"]),
                    "source_report_bundle_sha256": bundle_sha,
                    "replicate": replicate, "order": order,
                }, protocol=protocol)
            key = base.canonical_sha256(payload)
            cache_path = cache_root / "arbiter" / key[:2] / f"{key}.json"
            if arbiter.get("cache_key") != key or not cache_path.is_file():
                raise RuntimeError("reusable Initial Arbiter cache identity drift")
            cached = load_json(cache_path)
            if cached.get("request") != payload:
                raise RuntimeError("reusable Initial Arbiter request drift")
            verified_calls += 1
    return {
        "path": str(path), "sha256": file_sha256(path),
        "cache_root": str(cache_root), "verified_call_count": verified_calls,
        "strict_request_and_parser_reuse": True,
    }


def freeze(
    config: Mapping[str, Any], output: Path, *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> None:
    settings = _settings(config, protocol=protocol)
    source = output / protocol.source.experiment_dir
    source_report = source / "final" / "discovery_report.json"
    source_final = source / "final_report.json"
    if not source_report.is_file() or not source_final.is_file():
        raise RuntimeError(
            "run subtree-bundle-evolution-final-report before VL-RewardBench")
    records = shared._records(output)
    rows = shared._rows(records)
    initial, final = _rubrics(output, protocol=protocol)
    schedule = shared._schedule(output, rows)
    control = shared._control(output, records)
    initial_reuse = _validate_reusable_initial(
        config, output, rows, schedule, initial, protocol=protocol)
    phase20_path = (output.parent
                    / "vl_rewardbench_local_unified_subtree_competition_v1"
                    / "final_report.json")
    phase20_reference = None
    if phase20_path.is_file():
        phase20_reference = {
            "path": str(phase20_path), "sha256": file_sha256(phase20_path)}
    manifest = {
        "schema_version": "1.0.0",
        "protocol_version": protocol.version,
        "settings": settings,
        "dataset_count": len(records),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "initial_rubric_sha256": initial.rubric_sha256,
        "initial_reuse": initial_reuse,
        protocol.source_rubric_key: final.rubric_sha256,
        "source_discovery_report_sha256": file_sha256(source_report),
        "source_final_report_sha256": file_sha256(source_final),
        "control": {key: value for key, value in control.items()
                    if key not in {"metrics", "votes_by_replicate"}},
        "phase20_diagnostic_reference": phase20_reference,
        "worker_endpoint_identities": base._inspect_endpoints(
            config, base.BackendPoolSpec.from_dict(config["backend_pool"])),
        "unified_runtime_identity": phase21._unified_runtime_identity(config, protocol=protocol.source),
        "schedule_sha256": base.canonical_sha256(schedule),
        "selection_after_benchmark_forbidden": True,
    }
    target = _target(output, protocol=protocol)
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("Phase21 VL-RewardBench manifest drift")
        print("vlrb-subtree-bundle-freeze already completed")
        return
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    atomic_write_json(target / "stage_status.json", {"freeze": {
        "status": "passed", "details": {
            "dataset_count": len(records),
            protocol.source_rubric_key: final.rubric_sha256}}})
    print(json.dumps({"dataset_count": len(records),
                      protocol.source_rubric_key: final.rubric_sha256}, indent=2))


def _load(config: Mapping[str, Any], output: Path, *, protocol: BenchmarkProtocol = DEFAULT_PROTOCOL):
    _settings(config, protocol=protocol)
    target = _target(output, protocol=protocol)
    path = target / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run vlrb-subtree-bundle-freeze first")
    manifest = load_json(path)
    if manifest.get("protocol_version") != protocol.version:
        raise RuntimeError("Phase21 VL-RewardBench manifest protocol drift")
    if manifest.get("settings") != protocol.settings:
        raise RuntimeError("Phase21 VL-RewardBench manifest settings drift")
    records = shared._records(output)
    rows = shared._rows(records)
    schedule = load_json(target / "order_schedule.json")
    initial, final = _rubrics(output, protocol=protocol)
    source = output / protocol.source.experiment_dir
    if ([str(row["sample_id"]) for row in rows] != manifest["sample_ids"]
            or initial.rubric_sha256 != manifest["initial_rubric_sha256"]
            or final.rubric_sha256 != manifest[protocol.source_rubric_key]
            or file_sha256(source / "final_report.json")
            != manifest["source_final_report_sha256"]
            or base.canonical_sha256(schedule) != manifest["schedule_sha256"]):
        raise RuntimeError("Phase21 VL-RewardBench frozen identity drift")
    initial_path = Path(manifest["initial_reuse"]["path"])
    if (not initial_path.is_file()
            or file_sha256(initial_path) != manifest["initial_reuse"]["sha256"]):
        raise RuntimeError("Phase21 reusable Initial identity drift")
    return target, manifest, records, rows, schedule, initial, final


def _verify_live(
    config: Mapping[str, Any], manifest: Mapping[str, Any], *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> None:
    actual = base._inspect_endpoints(
        config, base.BackendPoolSpec.from_dict(config["backend_pool"]))
    if actual != manifest["worker_endpoint_identities"]:
        raise RuntimeError("Phase21 VL-RewardBench endpoint identity drift")
    if (phase21._unified_runtime_identity(config, protocol=protocol.source)
            != manifest["unified_runtime_identity"]):
        raise RuntimeError(
            "Phase21 VL-RewardBench Unified prompt/parser/model identity drift")


def audit(
    config: Mapping[str, Any], output: Path, *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> None:
    target, manifest, records, rows, schedule, initial, final = _load(config, output, protocol=protocol)
    control = shared._control(output, records)
    replayed = vlrb_metrics._system_metrics(records, control["votes_by_replicate"])
    pool = base.BackendPoolSpec.from_dict(config["backend_pool"])
    checks = {
        "dataset_count_1247": len(records) == protocol.settings["dataset_count"],
        "schedule_k3": all(tuple(value) in {(0, 1, 0), (1, 0, 1)}
                           for value in schedule.values()),
        "same_sample_order": manifest["sample_ids"]
            == [str(row["sample_id"]) for row in rows],
        "same_decoding": (
            protocol.settings["temperature"] == 0.5
            and protocol.settings["max_tokens"] == 2048
            and protocol.settings["generation_seed_policy"] == "unset"),
        "same_unified_runtime": phase21._unified_runtime_identity(config, protocol=protocol.source)
            == manifest["unified_runtime_identity"],
        "two_available_slot_endpoints":
            tuple(item.endpoint_id for item in pool.endpoints) == system.ENDPOINT_IDS,
        "initial_five_roots": len(initial.root_ids) == 5,
        "final_five_roots": len(final.root_ids) == 5,
        "phase17_control_reused": control["sha256"] == manifest["control"]["sha256"],
        "phase17_control_exact_replay": (
            replayed["original_index_predictions"]
            == control["metrics"]["original_index_predictions"]
            and replayed["strict_accuracy"]
            == control["metrics"]["strict_accuracy"]),
        "benchmark_selection_forbidden": manifest[
            "selection_after_benchmark_forbidden"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase21 VL-RewardBench audit failed: {checks}")
    atomic_write_json(target / "offline_audit.json", {
        "schema_version": "1.0.0", "offline_only": True, "checks": checks})
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": checks}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(checks, indent=2))


def _evaluate(
    config: Mapping[str, Any], target: Path, label: str,
    rows: Sequence[Mapping[str, Any]], schedule: Mapping[str, Sequence[int]],
    rubric: StructuredRubric, attempts: int,
    *, protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    return system.evaluate(
        config, output_path=target / "predictions" / f"{label}.json",
        cache_dir=target / "cache" / label, split_name=f"vlrb_{label}",
        rows=rows, rubric=rubric, settings=_runtime_settings(config, protocol=protocol),
        orders_by_id=schedule, total_attempt_limit=attempts)


def _failure_count(value: Mapping[str, Any]) -> int:
    return shared._failure_count(value)


def smoke(
    config: Mapping[str, Any], output: Path, *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> None:
    target, manifest, _, rows, schedule, initial, final = _load(config, output, protocol=protocol)
    _verify_live(config, manifest, protocol=protocol)
    if load_json(target / "stage_status.json").get("audit", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-subtree-bundle-audit first")
    selected = rows[:protocol.settings["smoke_sample_count"]]
    selected_schedule = {
        str(row["sample_id"]): schedule[str(row["sample_id"])] for row in selected}
    details = {}
    details["initial_reused"] = {
        "technical_failure_count": 0,
        "strict_request_and_parser_reuse": True,
        "verified_call_count": manifest["initial_reuse"]["verified_call_count"],
    }
    for label, rubric in ((protocol.final_label, final),):
        value = _evaluate(
            config, target, label, selected, selected_schedule, rubric,
            1 + protocol.settings["max_parse_retries"], protocol=protocol)
        details[f"{label}_smoke"] = {
            "sample_count": len(selected),
            "technical_failure_count": _failure_count(value)}
    if any(item["technical_failure_count"] for item in details.values()):
        raise RuntimeError(f"Phase21 VL-RewardBench smoke failed: {details}")
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def run(
    config: Mapping[str, Any], output: Path, *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> None:
    target, manifest, _, rows, schedule, initial, final = _load(config, output, protocol=protocol)
    _verify_live(config, manifest, protocol=protocol)
    if load_json(target / "stage_status.json").get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-subtree-bundle-smoke first")
    details = {}
    details["initial"] = {
        "technical_failure_count": 0, "reused": True,
        "new_model_requests": 0}
    for label, rubric in ((protocol.final_label, final),):
        value = _evaluate(config, target, label, rows, schedule, rubric, 1, protocol=protocol)
        details[label] = {
            "technical_failure_count": _failure_count(value),
            "wall_seconds": value["wall_seconds"]}
    status = load_json(target / "stage_status.json")
    status["run"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def retry(
    config: Mapping[str, Any], output: Path, *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> None:
    target, manifest, _, rows, schedule, initial, final = _load(config, output, protocol=protocol)
    _verify_live(config, manifest, protocol=protocol)
    if load_json(target / "stage_status.json").get("run", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-subtree-bundle-run first")
    details = {}
    attempts = 1 + protocol.settings["max_parse_retries"]
    details["initial"] = {
        "technical_failure_count": 0, "reused": True,
        "new_model_requests": 0}
    for label, rubric in ((protocol.final_label, final),):
        value = _evaluate(config, target, label, rows, schedule, rubric, attempts, protocol=protocol)
        details[label] = {"technical_failure_count": _failure_count(value)}
    status = load_json(target / "stage_status.json")
    status["retry"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def report(
    config: Mapping[str, Any], output: Path, *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL, publish_status: bool = True,
) -> None:
    """Write shared comparisons; wrappers publish success after their own checks."""
    target, manifest, records, _, _, _, _ = _load(config, output, protocol=protocol)
    status = load_json(target / "stage_status.json")
    if status.get("retry", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-subtree-bundle-retry first")
    pending = "running" if publish_status else "pending_wrapper_comparison"
    status["report"] = {"status": pending, "details": {}}
    atomic_write_json(target / "stage_status.json", status)
    initial_value = system.load(Path(manifest["initial_reuse"]["path"]))
    final_value = system.load(target / "predictions" / f"{protocol.final_label}.json")
    failures = {
        "initial": _failure_count(initial_value),
        protocol.final_label: _failure_count(final_value)}
    if any(failures.values()):
        raise RuntimeError(f"unresolved Phase21 VL-RB failures: {failures}")
    initial_metrics = vlrb_metrics._system_metrics(records, shared._votes(initial_value))
    final_metrics = vlrb_metrics._system_metrics(records, shared._votes(final_value))
    control = shared._control(output, records)
    control_metrics = control["metrics"]
    vs_control = vlrb._paired(
        records, control_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    vs_initial = vlrb._paired(
        records, initial_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    vs_control["paired_bootstrap"] = shared._paired_bootstrap_delta_ci(
        records, control_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    vs_initial["paired_bootstrap"] = shared._paired_bootstrap_delta_ci(
        records, initial_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    value = {
        "schema_version": "1.0.0", "protocol_version": protocol.version,
        "systems": {
            "initial_five_root": {"metrics": initial_metrics},
            "phase17_e4_control": {
                "metrics": control_metrics, "reused": True,
                "source_sha256": control["sha256"]},
            protocol.final_label: {"metrics": final_metrics}},
        "paired": {
            f"{protocol.final_label}_vs_phase17_e4": vs_control,
            f"{protocol.final_label}_vs_initial": vs_initial},
        "selection_after_benchmark_forbidden": True,
        "unresolved_technical_failures": 0,
        "manifest_sha256": base.canonical_sha256(manifest),
    }
    phase20_reference = manifest.get("phase20_diagnostic_reference")
    if phase20_reference is not None:
        path = Path(phase20_reference["path"])
        if file_sha256(path) != phase20_reference["sha256"]:
            raise RuntimeError("Phase20 diagnostic reference drift")
        phase20_value = load_json(path)
        value["systems"]["phase20_diagnostic"] = {
            "metrics": phase20_value["systems"]["local_subtree_final"]["metrics"],
            "reused": True,
            "diagnostic_only": True,
            "known_protocol_mixture": True,
            "source_sha256": phase20_reference["sha256"],
        }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        f"# {protocol.report_title} on VL-RewardBench", "",
        "| System | Strict ACC | OverallAcc | MacroAcc | Coverage |",
        "|---|---:|---:|---:|---:|"]
    for name, item in value["systems"].items():
        metric = item["metrics"]
        lines.append(
            f"| {name} | {metric['strict_accuracy']:.4f} | "
            f"{metric.get('overall_acc', metric['covered_accuracy']):.4f} | "
            f"{metric.get('macro_acc', metric['macro_strict_accuracy']):.4f} | "
            f"{metric['coverage']:.4f} |")
    lines.extend(["", f"## Paired {protocol.final_label} vs Phase17 E4", "", "```json",
                  json.dumps(vs_control, indent=2), "```", ""])
    (target / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["report"] = {
        "status": "passed" if publish_status else pending, "details": {
        "control_strict_accuracy": control_metrics["strict_accuracy"],
        f"{protocol.final_label}_strict_accuracy": final_metrics["strict_accuracy"],
        "net_corrected": vs_control["net_corrected"],
        "mcnemar_exact_two_sided_p": vs_control[
            "mcnemar_exact_two_sided_p"]}}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(status["report"]["details"], indent=2))


def run_stage(
    config: Mapping[str, Any], output: Path, stage: str, *,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        protocol.stages[0]: freeze, protocol.stages[1]: audit, protocol.stages[2]: smoke,
        protocol.stages[3]: run, protocol.stages[4]: retry, protocol.stages[5]: report}
    if stage not in actions:
        raise ValueError(f"unsupported Phase21 VL-RewardBench stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output, protocol=protocol)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

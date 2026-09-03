"""VL-RewardBench evaluation for the Phase20 local-subtree evolution rubric."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from critiq.structured.schema import StructuredRubric

from . import aligned_system_runtime as system
from . import local_unified_subtree_evolution as local
from . import run_rubric_evolution as base
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_aligned_evolution as shared
from . import vl_rewardbench_phase10 as vlrb_metrics
from .experiment_utils import atomic_write_json, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_local_unified_subtree_competition_v1"
PROTOCOL_VERSION = "vlrb-local-unified-subtree-competition-evolution-v1"
STAGES = (
    "vlrb-local-subtree-evolution-freeze",
    "vlrb-local-subtree-evolution-audit",
    "vlrb-local-subtree-evolution-smoke",
    "vlrb-local-subtree-evolution-run",
    "vlrb-local-subtree-evolution-retry",
    "vlrb-local-subtree-evolution-report",
)
SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": local.EXPERIMENT_DIR,
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
    "systems": ["initial_five_root", "phase17_e4_control", "local_subtree_final"],
    "selection_after_benchmark_forbidden": True,
}


def _target(output: Path) -> Path:
    return output.parent / EXPERIMENT_DIR


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("vlrb_local_unified_subtree_competition_v1_experiment")
    if value != SETTINGS:
        raise RuntimeError(
            "vlrb_local_unified_subtree_competition_v1_experiment drift")
    return dict(value)


def _runtime_settings(config: Mapping[str, Any]) -> system.RuntimeSettings:
    value = _settings(config)
    return system.RuntimeSettings(
        temperature=float(value["temperature"]),
        max_tokens=int(value["max_tokens"]),
        max_parse_retries=int(value["max_parse_retries"]),
        generation_seed_policy=str(value["generation_seed_policy"]))


def _rubrics(output: Path) -> tuple[StructuredRubric, StructuredRubric]:
    source = output / local.EXPERIMENT_DIR
    initial = StructuredRubric.load_json(
        source / "epochs" / "epoch_00" / "rubric_initial.json")
    final = StructuredRubric.load_json(source / "final" / "rubric.json")
    return initial, final


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    source_report = output / local.EXPERIMENT_DIR / "final" / "discovery_report.json"
    if not source_report.is_file():
        raise RuntimeError("run local-subtree-evolution-report before VL-RewardBench")
    records = shared._records(output)
    rows = shared._rows(records)
    initial, final = _rubrics(output)
    schedule = shared._schedule(output, rows)
    control = shared._control(output, records)
    manifest = {
        "schema_version": "1.0.0", "protocol_version": PROTOCOL_VERSION,
        "settings": settings, "dataset_count": len(records),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "initial_rubric_sha256": initial.rubric_sha256,
        "local_subtree_rubric_sha256": final.rubric_sha256,
        "source_discovery_report_sha256": file_sha256(source_report),
        "control": {key: value for key, value in control.items()
                    if key not in {"metrics", "votes_by_replicate"}},
        "worker_endpoint_identities": base._inspect_endpoints(
            config, base.BackendPoolSpec.from_dict(config["backend_pool"])),
        "schedule_sha256": base.canonical_sha256(schedule),
        "selection_after_benchmark_forbidden": True}
    target = _target(output)
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("local-subtree VL-RewardBench manifest drift")
        print("vlrb-local-subtree-evolution-freeze already completed")
        return
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    atomic_write_json(target / "stage_status.json", {"freeze": {
        "status": "passed", "details": {
            "dataset_count": len(records),
            "local_subtree_rubric_sha256": final.rubric_sha256}}})
    print(json.dumps({"dataset_count": len(records),
                      "local_subtree_rubric_sha256": final.rubric_sha256}, indent=2))


def _load(config: Mapping[str, Any], output: Path):
    _settings(config)
    target = _target(output)
    path = target / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run vlrb-local-subtree-evolution-freeze first")
    manifest = load_json(path)
    records = shared._records(output)
    rows = shared._rows(records)
    schedule = load_json(target / "order_schedule.json")
    initial, final = _rubrics(output)
    if ([str(row["sample_id"]) for row in rows] != manifest["sample_ids"]
            or initial.rubric_sha256 != manifest["initial_rubric_sha256"]
            or final.rubric_sha256 != manifest["local_subtree_rubric_sha256"]
            or base.canonical_sha256(schedule) != manifest["schedule_sha256"]):
        raise RuntimeError("local-subtree VL-RewardBench frozen identity drift")
    return target, manifest, records, rows, schedule, initial, final


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    actual = base._inspect_endpoints(
        config, base.BackendPoolSpec.from_dict(config["backend_pool"]))
    if actual != manifest.get("worker_endpoint_identities"):
        raise RuntimeError("local-subtree VL-RewardBench endpoint identity drift")


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rows, schedule, initial, final = _load(config, output)
    control = shared._control(output, records)
    pool = base.BackendPoolSpec.from_dict(config["backend_pool"])
    replayed = vlrb_metrics._system_metrics(records, control["votes_by_replicate"])
    checks = {
        "dataset_count_1247": len(records) == 1247,
        "schedule_k3": all(tuple(value) in {(0, 1, 0), (1, 0, 1)}
                           for value in schedule.values()),
        "same_sample_order": manifest["sample_ids"]
            == [str(row["sample_id"]) for row in rows],
        "same_decoding": (SETTINGS["temperature"] == 0.5
                          and SETTINGS["max_tokens"] == 2048
                          and SETTINGS["generation_seed_policy"] == "unset"),
        "two_endpoints": tuple(item.endpoint_id for item in pool.endpoints)
            == system.ENDPOINT_IDS,
        "initial_five_roots": len(initial.root_ids) == 5,
        "final_five_roots": len(final.root_ids) == 5,
        "phase17_control_reused": control["sha256"] == manifest["control"]["sha256"],
        "phase17_control_exact_replay": (
            replayed["original_index_predictions"]
            == control["metrics"]["original_index_predictions"]
            and replayed["strict_accuracy"] == control["metrics"]["strict_accuracy"]),
        "benchmark_selection_forbidden": manifest[
            "selection_after_benchmark_forbidden"] is True}
    if not all(checks.values()):
        raise RuntimeError(f"local-subtree VL-RewardBench audit failed: {checks}")
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
) -> dict[str, Any]:
    return system.evaluate(
        config, output_path=target / "predictions" / f"{label}.json",
        cache_dir=target / "cache" / label, split_name=f"vlrb_{label}",
        rows=rows, rubric=rubric, settings=_runtime_settings(config),
        orders_by_id=schedule, total_attempt_limit=attempts)


def _failure_count(value: Mapping[str, Any]) -> int:
    return shared._failure_count(value)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, _, rows, schedule, initial, final = _load(config, output)
    _verify_live(config, manifest)
    if load_json(target / "stage_status.json").get("audit", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-local-subtree-evolution-audit first")
    count = SETTINGS["smoke_sample_count"]
    selected = rows[:count]
    selected_schedule = {str(row["sample_id"]): schedule[str(row["sample_id"])]
                         for row in selected}
    details = {}
    for label, rubric in (("initial", initial), ("local_subtree_final", final)):
        value = _evaluate(config, target, label, selected, selected_schedule,
                          rubric, 1 + SETTINGS["max_parse_retries"])
        details[f"{label}_smoke"] = {
            "sample_count": count,
            "technical_failure_count": _failure_count(value)}
    if any(item["technical_failure_count"] for item in details.values()):
        raise RuntimeError(f"local-subtree VL-RewardBench smoke failed: {details}")
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, _, rows, schedule, initial, final = _load(config, output)
    _verify_live(config, manifest)
    if load_json(target / "stage_status.json").get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-local-subtree-evolution-smoke first")
    details = {}
    for label, rubric in (("initial", initial), ("local_subtree_final", final)):
        value = _evaluate(config, target, label, rows, schedule, rubric, 1)
        details[label] = {"technical_failure_count": _failure_count(value),
                          "wall_seconds": value["wall_seconds"]}
    status = load_json(target / "stage_status.json")
    status["run"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, _, rows, schedule, initial, final = _load(config, output)
    _verify_live(config, manifest)
    if load_json(target / "stage_status.json").get("run", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-local-subtree-evolution-run first")
    details = {}
    attempts = 1 + SETTINGS["max_parse_retries"]
    for label, rubric in (("initial", initial), ("local_subtree_final", final)):
        value = _evaluate(config, target, label, rows, schedule, rubric, attempts)
        details[label] = {"technical_failure_count": _failure_count(value)}
    status = load_json(target / "stage_status.json")
    status["retry"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, _, _, _, _ = _load(config, output)
    if load_json(target / "stage_status.json").get("retry", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-local-subtree-evolution-retry first")
    initial_value = system.load(target / "predictions" / "initial.json")
    final_value = system.load(target / "predictions" / "local_subtree_final.json")
    failures = {"initial": _failure_count(initial_value),
                "local_subtree_final": _failure_count(final_value)}
    if any(failures.values()):
        raise RuntimeError(f"unresolved local-subtree VL-RB failures: {failures}")
    initial_metrics = vlrb_metrics._system_metrics(records, shared._votes(initial_value))
    final_metrics = vlrb_metrics._system_metrics(records, shared._votes(final_value))
    control = shared._control(output, records)
    control_metrics = control["metrics"]
    comparison = vlrb._paired(
        records, control_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    initial_comparison = vlrb._paired(
        records, initial_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    comparison["paired_bootstrap"] = shared._paired_bootstrap_delta_ci(
        records, control_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    initial_comparison["paired_bootstrap"] = shared._paired_bootstrap_delta_ci(
        records, initial_metrics["original_index_predictions"],
        final_metrics["original_index_predictions"])
    value = {
        "schema_version": "1.0.0", "protocol_version": PROTOCOL_VERSION,
        "systems": {
            "initial_five_root": {"metrics": initial_metrics},
            "phase17_e4_control": {"metrics": control_metrics, "reused": True,
                                   "source_sha256": control["sha256"]},
            "local_subtree_final": {"metrics": final_metrics}},
        "paired": {"local_vs_phase17_e4": comparison,
                   "local_vs_initial": initial_comparison},
        "selection_after_benchmark_forbidden": True,
        "unresolved_technical_failures": 0,
        "manifest_sha256": base.canonical_sha256(manifest)}
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# Local Unified-Subtree Competition Evolution on VL-RewardBench", "",
        "| System | Strict ACC | OverallAcc | MacroAcc | Coverage |",
        "|---|---:|---:|---:|---:|"]
    for name, item in value["systems"].items():
        metric = item["metrics"]
        lines.append(
            f"| {name} | {metric['strict_accuracy']:.4f} | "
            f"{metric.get('overall_acc', metric['covered_accuracy']):.4f} | "
            f"{metric.get('macro_acc', metric['macro_strict_accuracy']):.4f} | "
            f"{metric['coverage']:.4f} |")
    lines.extend(["", "## Paired local vs Phase17 E4", "", "```json",
                  json.dumps(comparison, indent=2), "```", ""])
    (target / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "control_strict_accuracy": control_metrics["strict_accuracy"],
        "local_strict_accuracy": final_metrics["strict_accuracy"],
        "net_corrected": comparison["net_corrected"],
        "mcnemar_exact_two_sided_p": comparison["mcnemar_exact_two_sided_p"]}}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(status["report"]["details"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze, STAGES[1]: audit, STAGES[2]: smoke,
        STAGES[3]: run, STAGES[4]: retry, STAGES[5]: report}
    if stage not in actions:
        raise ValueError(f"unsupported local-subtree VL-RewardBench stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

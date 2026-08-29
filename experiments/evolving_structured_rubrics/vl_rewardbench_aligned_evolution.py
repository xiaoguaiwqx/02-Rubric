"""VL-RewardBench evaluation for the Phase19 aligned-evolution rubric."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

from critiq.structured.schema import StructuredRubric

from . import _global_arbiter_ab_only_support as support
from . import aligned_system_runtime as system
from . import run_rubric_evolution as base
from . import unified_subtree_arbiter_evolution as aligned
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_phase10 as vlrb_metrics
from .experiment_utils import atomic_write_json, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_unified_subtree_arbiter_aligned_evolution_v1"
PROTOCOL_VERSION = "vlrb-unified-subtree-global-arbiter-aligned-evolution-v1"
CONTROL_EXPERIMENT = "vl_rewardbench_global_arbiter_ab_preferred_none_v2"
CONTROL_SYSTEM = "s5_v2_global_arbiter_ab_preferred_none_tolerant"
STAGES = (
    "vlrb-aligned-evolution-freeze",
    "vlrb-aligned-evolution-audit",
    "vlrb-aligned-evolution-smoke",
    "vlrb-aligned-evolution-run",
    "vlrb-aligned-evolution-retry",
    "vlrb-aligned-evolution-report",
)
SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": aligned.EXPERIMENT_DIR,
    "control_experiment": CONTROL_EXPERIMENT,
    "control_system": CONTROL_SYSTEM,
    "dataset_count": 1247,
    "k": 3,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "endpoint_ids": ["vllm-8000", "vllm-8001"],
    "scheduler": "sample_bundle_available_slot_affinity",
    "smoke_sample_count": 20,
    "systems": ["initial_five_root", "phase17_e4_control", "aligned_final"],
    "selection_after_benchmark_forbidden": True,
}


def _target(output: Path) -> Path:
    return output.parent / EXPERIMENT_DIR


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("vlrb_unified_subtree_arbiter_aligned_evolution_v1_experiment")
    if value != SETTINGS:
        raise RuntimeError(
            "vlrb_unified_subtree_arbiter_aligned_evolution_v1_experiment drift")
    return dict(value)


def _runtime_settings(config: Mapping[str, Any]) -> system.RuntimeSettings:
    value = _settings(config)
    return system.RuntimeSettings(
        temperature=float(value["temperature"]),
        max_tokens=int(value["max_tokens"]),
        max_parse_retries=int(value["max_parse_retries"]),
        generation_seed_policy=str(value["generation_seed_policy"]),
    )


def _records(output: Path) -> tuple[dict[str, Any], ...]:
    return support.records(output)


def _rows(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return support.vlrb_rows(records)


def _schedule(output: Path, rows: Sequence[Mapping[str, Any]]):
    return support.source_schedule(output, [str(row["sample_id"]) for row in rows])


def _rubrics(output: Path) -> tuple[StructuredRubric, StructuredRubric]:
    source = output / aligned.EXPERIMENT_DIR
    initial = StructuredRubric.load_json(
        source / "epochs" / "epoch_00" / "rubric_initial.json")
    final = StructuredRubric.load_json(source / "final" / "rubric.json")
    return initial, final


def _control(output: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = output.parent / CONTROL_EXPERIMENT / "final_report.json"
    manifest_path = output.parent / CONTROL_EXPERIMENT / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"frozen Phase17 E4 control is missing: {path}")
    if not manifest_path.is_file():
        raise RuntimeError(
            f"frozen Phase17 E4 control manifest is missing: {manifest_path}")
    frozen = load_json(manifest_path)
    frozen_settings = frozen.get("settings", {})
    expected_prompt_sha = hashlib.sha256(
        system.arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    if (frozen.get("protocol_version")
            != "global-arbiter-ab-preferred-none-tolerant-v2-vlrb-only"
            or frozen.get("system_prompt_sha256") != expected_prompt_sha
            or frozen_settings.get("k") != SETTINGS["k"]
            or frozen_settings.get("temperature") != SETTINGS["temperature"]
            or frozen_settings.get("max_tokens") != SETTINGS["max_tokens"]
            or frozen_settings.get("generation_seed_policy")
            != SETTINGS["generation_seed_policy"]):
        raise RuntimeError("frozen Phase17 E4 control protocol identity drift")
    report = load_json(path)
    value = report.get("systems", {}).get(CONTROL_SYSTEM)
    if not isinstance(value, Mapping):
        raise RuntimeError("frozen Phase17 E4 control system is missing")
    metrics = value.get("metrics")
    if (not isinstance(metrics, Mapping)
            or metrics.get("sample_count") != len(records)
            or len(metrics.get("original_index_predictions", [])) != len(records)):
        raise RuntimeError("frozen Phase17 E4 control prediction identity drift")
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "frozen_manifest_sha256": file_sha256(manifest_path),
            "system": CONTROL_SYSTEM, "metrics": dict(metrics),
            "votes_by_replicate": value.get("votes_by_replicate")}


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    source_report = output / aligned.EXPERIMENT_DIR / "final" / "discovery_report.json"
    if not source_report.is_file():
        raise RuntimeError("run aligned-evolution-report before VL-RewardBench")
    records = _records(output)
    rows = _rows(records)
    initial, final = _rubrics(output)
    schedule = _schedule(output, rows)
    control = _control(output, records)
    manifest = {
        "schema_version": "1.0.0", "protocol_version": PROTOCOL_VERSION,
        "settings": settings,
        "dataset_count": len(records),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "initial_rubric_sha256": initial.rubric_sha256,
        "aligned_rubric_sha256": final.rubric_sha256,
        "source_discovery_report_sha256": file_sha256(source_report),
        "control": {key: value for key, value in control.items()
                    if key not in {"metrics", "votes_by_replicate"}},
        "worker_endpoint_identities": base._inspect_endpoints(
            config, base.BackendPoolSpec.from_dict(config["backend_pool"])),
        "schedule_sha256": base.canonical_sha256(schedule),
        "selection_after_benchmark_forbidden": True,
    }
    target = _target(output)
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("aligned VL-RewardBench frozen manifest drift")
        print("vlrb-aligned-evolution-freeze already completed")
        return
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    atomic_write_json(target / "stage_status.json", {
        "freeze": {"status": "passed", "details": {
            "dataset_count": len(records),
            "aligned_rubric_sha256": final.rubric_sha256}}})
    print(json.dumps({"dataset_count": len(records),
                      "aligned_rubric_sha256": final.rubric_sha256}, indent=2))


def _load(config: Mapping[str, Any], output: Path):
    _settings(config)
    target = _target(output)
    path = target / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run vlrb-aligned-evolution-freeze first")
    manifest = load_json(path)
    records = _records(output)
    rows = _rows(records)
    schedule = load_json(target / "order_schedule.json")
    initial, final = _rubrics(output)
    if ([str(row["sample_id"]) for row in rows] != manifest["sample_ids"]
            or initial.rubric_sha256 != manifest["initial_rubric_sha256"]
            or final.rubric_sha256 != manifest["aligned_rubric_sha256"]
            or base.canonical_sha256(schedule) != manifest["schedule_sha256"]):
        raise RuntimeError("aligned VL-RewardBench frozen identity drift")
    return target, manifest, records, rows, schedule, initial, final


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    actual = base._inspect_endpoints(
        config, base.BackendPoolSpec.from_dict(config["backend_pool"]))
    if actual != manifest.get("worker_endpoint_identities"):
        raise RuntimeError("aligned VL-RewardBench Worker endpoint identity drift")


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rows, schedule, initial, final = _load(config, output)
    control = _control(output, records)
    pool = base.BackendPoolSpec.from_dict(config["backend_pool"])
    replayed_control = vlrb_metrics._system_metrics(
        records, control["votes_by_replicate"])
    checks = {
        "dataset_count_1247": len(records) == 1247,
        "schedule_k3": all(tuple(value) in {(0, 1, 0), (1, 0, 1)}
                           for value in schedule.values()),
        "same_sample_order": manifest["sample_ids"]
            == [str(row["sample_id"]) for row in rows],
        "same_model_prompts_and_decoding": (
            SETTINGS["temperature"] == 0.5
            and SETTINGS["max_tokens"] == 2048
            and SETTINGS["generation_seed_policy"] == "unset"),
        "two_endpoints": tuple(item.endpoint_id for item in pool.endpoints)
            == system.ENDPOINT_IDS,
        "initial_five_roots": len(initial.root_ids) == 5,
        "aligned_five_roots": len(final.root_ids) == 5,
        "phase17_control_reused": control["sha256"]
            == manifest["control"]["sha256"],
        "phase17_control_exact_replay": (
            replayed_control["original_index_predictions"]
            == control["metrics"]["original_index_predictions"]
            and replayed_control["strict_accuracy"]
            == control["metrics"]["strict_accuracy"]),
        "benchmark_selection_forbidden": manifest[
            "selection_after_benchmark_forbidden"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"aligned VL-RewardBench audit failed: {checks}")
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
    total = 0
    for sample in value["samples"]:
        for replicate in sample["replicates"].values():
            total += sum(not call.get("parse_ok")
                         for call in replicate["subtrees"].values())
            total += not replicate["arbiter"].get("parse_ok")
    return total


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rows, schedule, initial, final = _load(config, output)
    _verify_live(config, manifest)
    if load_json(target / "stage_status.json").get("audit", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-aligned-evolution-audit first")
    count = SETTINGS["smoke_sample_count"]
    selected = rows[:count]
    selected_schedule = {str(row["sample_id"]): schedule[str(row["sample_id"])]
                         for row in selected}
    details = {}
    # Use the final-run cache namespaces so a successful smoke is reusable by
    # the full run; the prediction artifact is intentionally overwritten by
    # the complete sample set later.
    for label, rubric in (("initial", initial), ("aligned_final", final)):
        value = _evaluate(config, target, label, selected, selected_schedule,
                          rubric, 1 + SETTINGS["max_parse_retries"])
        details[f"{label}_smoke"] = {
            "sample_count": count,
            "technical_failure_count": _failure_count(value),
        }
    if any(item["technical_failure_count"] for item in details.values()):
        raise RuntimeError(f"aligned VL-RewardBench smoke failed: {details}")
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, _, rows, schedule, initial, final = _load(config, output)
    _verify_live(config, manifest)
    if load_json(target / "stage_status.json").get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-aligned-evolution-smoke first")
    details = {}
    for label, rubric in (("initial", initial), ("aligned_final", final)):
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
        raise RuntimeError("run vlrb-aligned-evolution-run first")
    details = {}
    attempts = 1 + SETTINGS["max_parse_retries"]
    for label, rubric in (("initial", initial), ("aligned_final", final)):
        value = _evaluate(config, target, label, rows, schedule, rubric, attempts)
        details[label] = {"technical_failure_count": _failure_count(value)}
    status = load_json(target / "stage_status.json")
    status["retry"] = {"status": "passed", "details": details}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(details, indent=2))


def _votes(value: Mapping[str, Any]) -> list[list[int | None]]:
    result = []
    for predictions in value["metrics"]["predictions_by_replicate"]:
        result.append([0 if item == "A" else 1 if item == "B" else None
                       for item in predictions])
    return result


def _paired_bootstrap_delta_ci(
    records: Sequence[Mapping[str, Any]],
    baseline: Sequence[int | None],
    treatment: Sequence[int | None],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministic paired bootstrap CI for the strict-accuracy delta."""
    deltas = []
    for record, before, after in zip(records, baseline, treatment):
        target = int(record["preferred_original_index"])
        deltas.append(int(after == target) - int(before == target))
    generator = random.Random(seed)
    sample_count = len(deltas)
    values = []
    for _ in range(iterations):
        values.append(sum(deltas[generator.randrange(sample_count)]
                          for _ in range(sample_count)) / sample_count)
    values.sort()
    lower = values[int(0.025 * iterations)]
    upper = values[min(iterations - 1, int(0.975 * iterations))]
    return {
        "metric": "strict_accuracy_delta",
        "estimate": sum(deltas) / sample_count,
        "ci95": [lower, upper],
        "iterations": iterations,
        "seed": seed,
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, rows, _, _, _ = _load(config, output)
    if load_json(target / "stage_status.json").get("retry", {}).get("status") != "passed":
        raise RuntimeError("run vlrb-aligned-evolution-retry first")
    initial_value = system.load(target / "predictions" / "initial.json")
    aligned_value = system.load(target / "predictions" / "aligned_final.json")
    failures = {
        "initial": _failure_count(initial_value),
        "aligned_final": _failure_count(aligned_value),
    }
    if any(failures.values()):
        raise RuntimeError(f"unresolved aligned VL-RewardBench failures: {failures}")
    initial_metrics = vlrb_metrics._system_metrics(records, _votes(initial_value))
    aligned_metrics = vlrb_metrics._system_metrics(records, _votes(aligned_value))
    control = _control(output, records)
    control_metrics = control["metrics"]
    comparison = vlrb._paired(
        records, control_metrics["original_index_predictions"],
        aligned_metrics["original_index_predictions"])
    initial_comparison = vlrb._paired(
        records, initial_metrics["original_index_predictions"],
        aligned_metrics["original_index_predictions"])
    comparison["paired_bootstrap"] = _paired_bootstrap_delta_ci(
        records, control_metrics["original_index_predictions"],
        aligned_metrics["original_index_predictions"])
    initial_comparison["paired_bootstrap"] = _paired_bootstrap_delta_ci(
        records, initial_metrics["original_index_predictions"],
        aligned_metrics["original_index_predictions"])
    value = {
        "schema_version": "1.0.0", "protocol_version": PROTOCOL_VERSION,
        "systems": {
            "initial_five_root": {"metrics": initial_metrics},
            "phase17_e4_control": {"metrics": control_metrics,
                                   "reused": True,
                                   "source_sha256": control["sha256"]},
            "aligned_final": {"metrics": aligned_metrics},
        },
        "paired": {
            "aligned_vs_phase17_e4": comparison,
            "aligned_vs_initial": initial_comparison,
        },
        "selection_after_benchmark_forbidden": True,
        "unresolved_technical_failures": 0,
        "manifest_sha256": base.canonical_sha256(manifest),
    }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# Unified-Subtree + Global-Arbiter Aligned Evolution on VL-RewardBench",
        "", "| System | Strict ACC | OverallAcc | MacroAcc | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in value["systems"].items():
        metric = item["metrics"]
        lines.append(
            f"| {name} | {metric['strict_accuracy']:.4f} | "
            f"{metric.get('overall_acc', metric['covered_accuracy']):.4f} | "
            f"{metric.get('macro_acc', metric['macro_strict_accuracy']):.4f} | "
            f"{metric['coverage']:.4f} |")
    lines.extend(["", "## Paired aligned vs Phase17 E4", "",
                  "```json", json.dumps(comparison, indent=2), "```", ""])
    (target / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "control_strict_accuracy": control_metrics["strict_accuracy"],
        "aligned_strict_accuracy": aligned_metrics["strict_accuracy"],
        "net_corrected": comparison["net_corrected"],
        "mcnemar_exact_two_sided_p": comparison[
            "mcnemar_exact_two_sided_p"]}}
    atomic_write_json(target / "stage_status.json", status)
    print(json.dumps(status["report"]["details"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze, STAGES[1]: audit, STAGES[2]: smoke,
        STAGES[3]: run, STAGES[4]: retry, STAGES[5]: report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported aligned VL-RewardBench stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

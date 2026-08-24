"""VL-RewardBench evaluation for the Qwen2.5-specific Phase18 rubric."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from critiq.structured import PairwisePredictionOutput, StructuredRubric

from . import qwen25_full_evolution as evolution
from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from . import vl_rewardbench_phase10 as phase10
from . import vl_rewardbench_prompt_v2 as prompt_v2
from . import vl_rewardbench_qwen25_transfer as transfer
from .experiment_utils import atomic_write_json, canonical_sha256, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_qwen25_phase18_evolved_v1"
PROTOCOL_VERSION = "vlrb-qwen25-phase18-evolved-v1"
CONFIG_KEY = "vlrb_qwen25_evolved"
WORKER_MODEL = transfer.WORKER_MODEL
SMOKE_COUNT = 20
MAX_RETRY_ATTEMPTS = 10
FINAL_SYSTEM = "qwen25_phase18_specific_final_equal"

STAGES = tuple(f"vlrb-qwen25-evolved-{name}" for name in (
    "freeze", "audit", "smoke", "run", "retry", "report",
))

SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "worker_model": WORKER_MODEL,
    "source_experiment": evolution.EXPERIMENT_DIR,
    "control_experiment": transfer.EXPERIMENT_DIR,
    "transferred_control_epoch": transfer.SOURCE_EPOCH,
    "endpoint_ids": list(transfer.ENDPOINT_IDS),
    "scheduler": "sample_major_available_slot_dynamic",
    "prompt_version": prompt_v2.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    "structured_temperature": 0.5,
    "structured_max_tokens": 2048,
    "smoke_sample_count": SMOKE_COUNT,
    "max_retry_attempts": MAX_RETRY_ATTEMPTS,
    "k": legacy.K,
    "seed": legacy.SEED,
    "native_rerun": False,
    "selection_after_benchmark_forbidden": True,
}


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _source_target(output: Path) -> Path:
    return output / evolution.EXPERIMENT_DIR


def _control_target() -> Path:
    return transfer._target()


def _status_path(target: Path) -> Path:
    return target / "stage_status.json"


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    value[stage] = {"status": "passed", "details": dict(details)}
    atomic_write_json(_status_path(target), value)


def _require(target: Path, stage: str) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get(CONFIG_KEY) != SETTINGS:
        raise RuntimeError(f"{CONFIG_KEY} must match the frozen v1 protocol")
    return dict(SETTINGS)


def _worker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    # Keep exactly the Qwen2.5 transfer request identity.  This makes the
    # control and treatment differ only in criterion descriptions/topology.
    return transfer._worker_config(config)


def _rubric(output: Path) -> StructuredRubric:
    path = _source_target(output) / "final" / "rubric.json"
    report = _source_target(output) / "final_report.json"
    if not path.is_file() or not report.is_file():
        raise RuntimeError("run qwen25-evolution-final-report first")
    rubric = StructuredRubric.load_json(path)
    if load_json(report)["discovery"]["final_rubric_sha256"] != rubric.rubric_sha256:
        raise RuntimeError("Phase18 final rubric/report identity drift")
    return rubric


def _control_artifacts():
    target = _control_target()
    report_path = target / "final_report.json"
    logical_path = target / "retry/structured/combined/logical_votes.json"
    predictions = tuple(prompt_v2._prediction_path(
        target / "retry/structured", replicate) for replicate in range(legacy.K))
    if not report_path.is_file() or not logical_path.is_file() or not all(
            path.is_file() for path in predictions):
        raise RuntimeError("completed Qwen2.5 transferred-E4 control is required")
    return report_path, logical_path, predictions


def _manifest(config: Mapping[str, Any], output: Path, records, schedule,
              *, live: bool) -> dict[str, Any]:
    protocol = _protocol(config)
    rubric = _rubric(output)
    worker = _worker_config(config)
    control_report, control_logical, control_predictions = _control_artifacts()
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol": protocol,
        "exploratory": True,
        "dataset": {
            "count": len(records),
            "record_sha256": canonical_sha256(list(records)),
        },
        "order_schedule_sha256": canonical_sha256(schedule),
        "counterbalance": {"k": legacy.K, "seed": legacy.SEED,
                           "protocol": "balanced_b_1minusb_b"},
        "structured_worker_request_spec": prompt_v2._v2_request_spec(
            worker, records, schedule),
        "source": {
            "rubric_path": str((_source_target(output) / "final/rubric.json").resolve()),
            "rubric_sha256": rubric.rubric_sha256,
            "rubric_file_sha256": file_sha256(
                _source_target(output) / "final/rubric.json"),
            "node_count": len(rubric.nodes),
            "formal_final_report_sha256": file_sha256(
                _source_target(output) / "final_report.json"),
        },
        "control": {
            "experiment": transfer.EXPERIMENT_DIR,
            "report_sha256": file_sha256(control_report),
            "logical_votes_sha256": file_sha256(control_logical),
            "prediction_sha256": [file_sha256(path)
                                  for path in control_predictions],
            "new_model_requests": 0,
        },
        "native_rerun": False,
        "selection_after_benchmark_forbidden": True,
    }
    if live:
        value["endpoint_identities"] = transfer.phase10._inspect_endpoints(worker)
    return value


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target()
    _require(target, STAGES[0])
    records = transfer._records()
    schedule = legacy._order_schedule(records)
    expected = _manifest(config, output, records, schedule, live=False)
    stored = load_json(target / "frozen_manifest.json")
    if {key: value for key, value in stored.items()
            if key != "endpoint_identities"} != expected:
        raise RuntimeError("Qwen2.5 evolved VL-RewardBench manifest drift")
    return target, stored, records, schedule, _rubric(output)


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    live = transfer.phase10._inspect_endpoints(_worker_config(config))
    if live != manifest.get("endpoint_identities"):
        raise RuntimeError("Qwen2.5 evolved endpoint identity drift")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target()
    records = transfer._records()
    schedule = legacy._order_schedule(records)
    manifest = _manifest(config, output, records, schedule, live=True)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        if any(status.get(stage, {}).get("status") == "passed"
               for stage in STAGES[2:]):
            raise RuntimeError("Qwen2.5 evolved manifest drift after inference")
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    details = {
        "dataset_count": len(records),
        "worker_model": WORKER_MODEL,
        "rubric_sha256": manifest["source"]["rubric_sha256"],
        "node_count": manifest["source"]["node_count"],
        "logical_request_count": (
            len(records) * legacy.K * manifest["source"]["node_count"]),
        "native_new_request_count": 0,
    }
    _status(target, STAGES[0], details)
    print(json.dumps(details, indent=2))


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    control_report, control_logical_path, control_predictions = _control_artifacts()
    control_logical = load_json(control_logical_path)
    control_prediction_values = tuple(
        PairwisePredictionOutput.load_json(path) for path in control_predictions)
    checks = {
        "worker_model_qwen25": (
            manifest["structured_worker_request_spec"]["model"] == WORKER_MODEL),
        "prompt_v2": manifest["protocol"]["prompt_version"]
        == prompt_v2.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "k3_schedule_exact_control": (
            canonical_sha256(schedule)
            == load_json(_control_target() / "frozen_manifest.json")
            ["order_schedule_sha256"]),
        "control_complete": (
            len(control_predictions) == legacy.K
            and len(control_logical["sample_ids"]) == len(records)),
        "control_request_identity_exact": all(
            prediction.request_spec.to_dict()
            == manifest["structured_worker_request_spec"]
            for prediction in control_prediction_values),
        "five_initial_roots_preserved": len(rubric.root_ids) == 5,
        "native_rerun_disabled": manifest["native_rerun"] is False,
        "benchmark_labels_in_prompt": False,
    }
    if not all(value for key, value in checks.items()
               if key != "benchmark_labels_in_prompt"):
        raise RuntimeError(f"Qwen2.5 evolved audit failed: {checks}")
    value = {"schema_version": "1.0.0", "offline_only": True,
             "checks": checks, "control_report": str(control_report)}
    atomic_write_json(target / "offline_audit.json", value)
    _status(target, STAGES[1], value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _run_predictions(config: Mapping[str, Any], work: Path,
                     manifest: Mapping[str, Any], records, schedule, rubric):
    return prompt_v2._run_predictions(
        _worker_config(config), work, manifest, records, schedule, rubric)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGES[1])
    _verify_live(config, manifest)
    selected = tuple(records[:SMOKE_COUNT])
    logical, _, summaries = _run_predictions(
        config, target / "smoke", manifest, selected, schedule, rubric)
    calls = {endpoint_id: sum(int(summary["telemetry"]
            ["endpoint_call_counts"].get(endpoint_id, 0)) for summary in summaries)
             for endpoint_id in transfer.ENDPOINT_IDS}
    metrics = prompt_v2._smoke_system_metrics(
        selected, logical["systems"][prompt_v2.EQUAL_SYSTEM]
        ["votes_by_replicate"])
    valid_rate = sum(item["valid_rate"] for item in summaries) / len(summaries)
    if valid_rate < .99 or any(count == 0 for count in calls.values()):
        raise RuntimeError("Qwen2.5 evolved smoke validity/dual endpoint failed")
    details = {"sample_count": len(selected), "valid_rate": valid_rate,
               "endpoint_call_counts": calls, "metrics": metrics}
    _status(target, STAGES[2], details)
    print(json.dumps(details, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGES[2])
    _verify_live(config, manifest)
    _, _, summaries = _run_predictions(
        config, target / "run", manifest, records, schedule, rubric)
    details = {
        "sample_count": len(records), "node_count": len(rubric.nodes),
        "logical_request_count": len(records) * legacy.K * len(rubric.nodes),
        "valid_rates": [item["valid_rate"] for item in summaries],
        "native_new_request_count": 0,
    }
    _status(target, STAGES[3], details)
    print(json.dumps(details, indent=2))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGES[3])
    _verify_live(config, manifest)
    repaired = transfer._retry_structured(
        config, target, manifest, records, schedule, rubric,
        source_work=target / "run")
    details = load_json(target / "retry/structured/report.json")
    _status(target, STAGES[4], details)
    print(json.dumps({key: details[key] for key in (
        "target_count", "recovered_count", "still_failed_count",
        "new_model_requests")}, indent=2))


def _metric_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item[key] for key in (
        "overall_acc", "macro_acc", "coverage", "strict_accuracy")}


def _wrong_overlap(records, left: Mapping[str, Any],
                   right: Mapping[str, Any]) -> dict[str, Any]:
    left_wrong = {str(row["sample_id"]) for row, prediction in zip(
        records, left["original_index_predictions"])
                  if prediction != int(row["preferred_original_index"])}
    right_wrong = {str(row["sample_id"]) for row, prediction in zip(
        records, right["original_index_predictions"])
                   if prediction != int(row["preferred_original_index"])}
    union = left_wrong | right_wrong
    return {
        "left_wrong_count": len(left_wrong),
        "right_wrong_count": len(right_wrong),
        "intersection_count": len(left_wrong & right_wrong),
        "union_count": len(union),
        "jaccard": len(left_wrong & right_wrong) / len(union) if union else 1.0,
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGES[4])
    logical = load_json(target / "retry/structured/combined/logical_votes.json")
    control_logical = load_json(
        _control_target() / "retry/structured/combined/logical_votes.json")
    initial = phase10._system_metrics(
        records, control_logical["systems"][prompt_v2.INITIAL_SYSTEM]
        ["votes_by_replicate"])
    transferred = phase10._system_metrics(
        records, control_logical["systems"][prompt_v2.EQUAL_SYSTEM]
        ["votes_by_replicate"])
    final = phase10._system_metrics(
        records, logical["systems"][prompt_v2.EQUAL_SYSTEM]
        ["votes_by_replicate"])
    paired = {
        "specific_vs_initial": legacy._paired(
            records, initial["original_index_predictions"],
            final["original_index_predictions"]),
        "specific_vs_transferred_e4": legacy._paired(
            records, transferred["original_index_predictions"],
            final["original_index_predictions"]),
    }
    predictions = tuple(PairwisePredictionOutput.load_json(
        prompt_v2._prediction_path(target / "retry/structured", replicate))
        for replicate in range(legacy.K))
    root_metrics = {root_id: phase10._system_metrics(records, votes)
                    for root_id, votes in logical[
                        "root_votes_by_replicate"].items()}
    control_report = load_json(_control_target() / "final_report.json")
    qwen3_initial = control_report["metrics"][
        "qwen3_initial_five_root_prompt_v2"]
    transferred_rubric = transfer._rubric(output)
    shared_node_ids = set(rubric.nodes) & set(transferred_rubric.nodes)
    same_description_ids = [node_id for node_id in shared_node_ids
                            if rubric.get_node(node_id).criterion.description
                            == transferred_rubric.get_node(
                                node_id).criterion.description]
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "worker_model": WORKER_MODEL,
        "rubric_sha256": rubric.rubric_sha256,
        "node_count": len(rubric.nodes),
        "metrics": {
            transfer.INITIAL_SYSTEM: initial,
            transfer.E4_SYSTEM: transferred,
            FINAL_SYSTEM: final,
        },
        "metric_summary": {
            transfer.INITIAL_SYSTEM: _metric_row(initial),
            transfer.E4_SYSTEM: _metric_row(transferred),
            FINAL_SYSTEM: _metric_row(final),
        },
        "paired": paired,
        "root_metrics": root_metrics,
        "node_metrics": prompt_v2._node_metrics(
            records, schedule, rubric, predictions),
        "position_metrics": prompt_v2._position_metrics(
            records, schedule, logical["systems"][prompt_v2.EQUAL_SYSTEM]
            ["votes_by_replicate"]),
        "retry": load_json(target / "retry/structured/report.json"),
        "mechanism_diagnostics": {
            "qwen25_vs_qwen3_initial_wrong_overlap": _wrong_overlap(
                records, initial, qwen3_initial),
            "qwen3_initial_metric_summary": _metric_row(qwen3_initial),
            "rubric_structure": {
                "transferred_e4_node_count": len(transferred_rubric.nodes),
                "qwen25_specific_node_count": len(rubric.nodes),
                "shared_node_id_count": len(shared_node_ids),
                "same_description_node_count": len(same_description_ids),
            },
        },
        "native_rerun": False,
        "selection_after_benchmark_forbidden": True,
    }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# Qwen2.5-specific Full Evolution on VL-RewardBench", "",
        "Exploratory K=3 evaluation; benchmark results never select a rubric.",
        "", "| System | OverallAcc | MacroAcc | Coverage | Strict ACC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in value["metric_summary"].items():
        lines.append(
            f"| {name} | {item['overall_acc']:.4f} | "
            f"{item['macro_acc']:.4f} | {item['coverage']:.4f} | "
            f"{item['strict_accuracy']:.4f} |")
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    details = {
        "qwen25_initial_overall_acc": initial["overall_acc"],
        "qwen25_transferred_e4_overall_acc": transferred["overall_acc"],
        "qwen25_specific_final_overall_acc": final["overall_acc"],
        "specific_vs_initial_net_corrected": paired[
            "specific_vs_initial"]["net_corrected"],
        "specific_vs_transferred_e4_net_corrected": paired[
            "specific_vs_transferred_e4"]["net_corrected"],
        "unresolved_technical_failures": value["retry"]["still_failed_count"],
    }
    _status(target, STAGES[5], details)
    print(json.dumps(details, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = dict(zip(STAGES, (freeze, audit, smoke, run, retry, report)))
    if stage not in actions:
        raise ValueError(f"unsupported Qwen2.5 evolved stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

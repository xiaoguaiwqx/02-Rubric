"""VL-RewardBench evaluation for the combined Phase22 trajectory."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from . import all_sample_adaptive_recluster_evolution as phase22
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_unified_subtree_bundle_evolution as phase21_vlrb
from .experiment_utils import atomic_write_json, load_json
from .evolution_protocol import BenchmarkProtocol, EvolutionProtocol
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_all_sample_adaptive_recluster_evolution_v1"
PROTOCOL_VERSION = "vlrb-all-sample-adaptive-recluster-evolution-v1"
CONFIG_KEY = "vlrb_all_sample_adaptive_recluster_evolution_v1_experiment"
FINAL_LABEL = "phase22_final"
STAGES = (
    "vlrb-all-sample-adaptive-freeze",
    "vlrb-all-sample-adaptive-audit",
    "vlrb-all-sample-adaptive-smoke",
    "vlrb-all-sample-adaptive-run",
    "vlrb-all-sample-adaptive-retry",
    "vlrb-all-sample-adaptive-report",
)
SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": phase22.EXPERIMENT_DIR,
    "phase21_control_experiment": phase21_vlrb.EXPERIMENT_DIR,
    "control_experiment": phase21_vlrb.shared.CONTROL_EXPERIMENT,
    "control_system": phase21_vlrb.shared.CONTROL_SYSTEM,
    "dataset_count": 1247,
    "k": 3,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "endpoint_ids": ["vllm-8000", "vllm-8001"],
    "scheduler": "sample_bundle_available_slot_affinity",
    "smoke_sample_count": 20,
    "systems": [
        "initial_five_root", "phase17_e4_control", "phase21_final",
        FINAL_LABEL,
    ],
    "selection_after_benchmark_forbidden": True,
}


PROTOCOL = BenchmarkProtocol(
    identity=EvolutionProtocol.from_settings(
        experiment_dir=EXPERIMENT_DIR, version=PROTOCOL_VERSION,
        config_key=CONFIG_KEY, stages=STAGES, settings=SETTINGS,
    ),
    source=phase22.PROTOCOL, final_label=FINAL_LABEL,
    source_rubric_key="phase22_rubric_sha256",
    report_title="Phase22 All-Sample Adaptive-Recluster Evolution",
)


def _target(output: Path) -> Path:
    return output.parent / EXPERIMENT_DIR


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return PROTOCOL.validate(config)


def _delegate(name: str, config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    getattr(phase21_vlrb, name)(config, output, protocol=PROTOCOL)


def freeze(config: Mapping[str, Any], output: Path) -> None:
    phase21_target = output.parent / phase21_vlrb.EXPERIMENT_DIR
    phase21_report = phase21_target / "final_report.json"
    phase21_prediction = phase21_target / "predictions" / "phase21_final.json"
    if not phase21_report.is_file() or not phase21_prediction.is_file():
        raise RuntimeError(
            "Phase22 VL-RewardBench requires the completed Phase21 VL-RB control")
    manifest_path = _target(output) / "frozen_manifest.json"
    if manifest_path.is_file():
        phase21_vlrb._load(config, output, protocol=PROTOCOL)
        manifest = load_json(manifest_path)
        reference = manifest.get("phase21_vlrb_control", {})
        if (reference.get("report_sha256") != file_sha256(phase21_report)
                or reference.get("prediction_sha256") !=
                file_sha256(phase21_prediction)):
            raise RuntimeError("Phase21 VL-RB frozen control drift")
        return
    _delegate("freeze", config, output)
    manifest = load_json(manifest_path)
    manifest["phase21_vlrb_control"] = {
        "report_path": str(phase21_report),
        "report_sha256": file_sha256(phase21_report),
        "prediction_path": str(phase21_prediction),
        "prediction_sha256": file_sha256(phase21_prediction),
        "selection_role": "frozen_read_only_control",
    }
    atomic_write_json(manifest_path, manifest)


def audit(config: Mapping[str, Any], output: Path) -> None:
    _delegate("audit", config, output)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    _delegate("smoke", config, output)


def run(config: Mapping[str, Any], output: Path) -> None:
    _delegate("run", config, output)


def retry(config: Mapping[str, Any], output: Path) -> None:
    _delegate("retry", config, output)


def report(config: Mapping[str, Any], output: Path) -> None:
    # A shared report is not a completed Phase22 report: the frozen Phase21
    # comparison must also pass. Never expose a transient successful status.
    phase21_vlrb.report(config, output, protocol=PROTOCOL, publish_status=False)
    target = _target(output)
    status_path = target / "stage_status.json"
    manifest = load_json(target / "frozen_manifest.json")
    reference = manifest["phase21_vlrb_control"]
    prediction_path = Path(reference["prediction_path"])
    if file_sha256(prediction_path) != reference["prediction_sha256"]:
        raise RuntimeError("Phase21 VL-RB prediction control drift")
    records = phase21_vlrb.shared._records(output)
    phase21_value = phase21_vlrb.system.load(prediction_path)
    phase22_value = phase21_vlrb.system.load(
        target / "predictions" / f"{FINAL_LABEL}.json")
    phase21_metrics = phase21_vlrb.vlrb_metrics._system_metrics(
        records, phase21_vlrb.shared._votes(phase21_value))
    phase22_metrics = phase21_vlrb.vlrb_metrics._system_metrics(
        records, phase21_vlrb.shared._votes(phase22_value))
    paired = vlrb._paired(
        records, phase21_metrics["original_index_predictions"],
        phase22_metrics["original_index_predictions"])
    paired["paired_bootstrap"] = (
        phase21_vlrb.shared._paired_bootstrap_delta_ci(
            records, phase21_metrics["original_index_predictions"],
            phase22_metrics["original_index_predictions"]))
    report_path = target / "final_report.json"
    value = load_json(report_path)
    value["systems"]["phase21_final"] = {
        "metrics": phase21_metrics,
        "reused": True,
        "selection_role": "frozen_read_only_control",
        "source_sha256": reference["prediction_sha256"],
    }
    value["paired"]["phase22_final_vs_phase21_final"] = paired
    value["combined_scheme_only"] = True
    value["individual_metric_or_recluster_causal_claim_forbidden"] = True
    atomic_write_json(report_path, value)
    with (target / "final_report.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Paired Phase22 vs Phase21\n\n```json\n")
        stream.write(json.dumps(paired, indent=2))
        stream.write("\n```\n")
    status = load_json(status_path)
    status["report"] = {
        "status": "passed",
        "details": {
            "phase21_control_validated": True,
            "phase22_strict_accuracy": phase22_metrics["strict_accuracy"],
            "phase21_strict_accuracy": phase21_metrics["strict_accuracy"],
            "net_corrected_vs_phase21": paired["net_corrected"],
        },
    }
    atomic_write_json(status_path, status)


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
        raise ValueError(f"unsupported Phase22 VL-RewardBench stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

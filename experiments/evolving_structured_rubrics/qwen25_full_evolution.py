"""Qwen2.5-VL full Split+Refine evolution from the five initial roots.

This module deliberately reuses the frozen Phase17 algorithm while replacing
only the Pairwise Worker identity.  Discovery predictions, ErrorSignatures,
candidate evaluations, and failure histories are therefore generated inside a
new experiment directory and cannot reuse the Qwen3 evolution trajectory.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    PairwisePredictionOutput,
    RubricFeedback,
    StructuredRubric,
    execute_offline_m1,
)

from . import discovery_v2_prompt_v2_evolution as shared
from . import refine_evolution as refine
from . import run_rubric_evolution as base
from . import split_evolution as split
from . import vl_rewardbench_qwen25_transfer as transfer
from .experiment_utils import atomic_write_json, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase18_qwen25_discovery_v2_prompt_v2_split_refine_v1"
PROTOCOL_VERSION = "qwen25-discovery-v2-prompt-v2-locked-split-refine-v1"
CONFIG_KEY = "qwen25_full_evolution"
WORKER_MODEL = transfer.WORKER_MODEL
TRANSFERRED_EPOCH = transfer.SOURCE_EPOCH

SETTINGS = {
    **shared.SETTINGS,
    "protocol_version": PROTOCOL_VERSION,
    "worker_model": WORKER_MODEL,
    "source_experiment": "independent_initial_five_roots",
    "worker_endpoint": "configured_available_slot_pool",
    "worker_endpoints": list(transfer.ENDPOINT_IDS),
    "error_signature_policy": "fresh_qwen25_discovery100_only",
    "qwen3_evolution_artifact_reuse": False,
    "transferred_control_epoch": TRANSFERRED_EPOCH,
}

PROTOCOL = split.EvolutionProtocol(
    EXPERIMENT_DIR,
    "qwen25-evolution",
    rubric_memory_mode="global_rubric_v1",
    control_experiment_dir=None,
    read_only_control_signatures=False,
    pairwise_endpoint="vllm-8000",
    allow_configured_endpoint_pool=True,
    allow_legacy_signature_reuse=False,
)

STAGES = tuple(f"qwen25-evolution-{name}" for name in (
    "freeze", "audit", "smoke", "run", "report", "heldout",
    "final-report",
))


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _worker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create an experiment-local Qwen2.5 view without changing old runs."""

    value = transfer._worker_config(config)
    pool = deepcopy(dict(value["backend_pool"]))
    pool["pool_id"] = "qwen25vl7b-phase18-evolution-pool-v1"
    value["backend_pool"] = pool
    return value


def _adapt_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get(CONFIG_KEY) != SETTINGS:
        raise RuntimeError(f"{CONFIG_KEY} must match the frozen v1 protocol")
    value = _worker_config(config)
    # The shared runner validates this private compatibility key.  It never
    # reads the Phase17 output directory after _activate() changes the globals.
    value["discovery_v2_prompt_v2_evolution"] = SETTINGS
    return value


def _activate() -> None:
    shared.EXPERIMENT_DIR = EXPERIMENT_DIR
    shared.PROTOCOL_VERSION = PROTOCOL_VERSION
    shared.SETTINGS = SETTINGS
    shared.PROTOCOL = PROTOCOL


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _validate_baseline_identity(target: Path) -> PairwisePredictionOutput:
    prediction = PairwisePredictionOutput.load_json(
        split._epoch(target, 0) / "discovery_pairwise.json")
    if prediction.request_spec.model != WORKER_MODEL:
        raise RuntimeError("Phase18 epoch-0 prediction is not Qwen2.5")
    if prediction.prompt_version != shared.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION:
        raise RuntimeError("Phase18 epoch-0 prediction is not Prompt v2")
    return prediction


def _augment_manifest(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    path = target / "frozen_manifest.json"
    manifest = load_json(path)
    worker = _worker_config(config)
    identities = transfer.phase10._inspect_endpoints(worker)
    stored_identities = manifest.get("worker_endpoint_identities")
    if stored_identities is not None and stored_identities != identities:
        status_path = target / "stage_status.json"
        status = load_json(status_path) if status_path.exists() else {}
        if any(status.get(stage, {}).get("status") == "passed"
               for stage in ("smoke", "run", "report", "heldout",
                             "final_report")):
            raise RuntimeError(
                "Qwen2.5 evolution endpoint identity drift after inference")
    manifest.update({
        "worker_model": WORKER_MODEL,
        "worker_endpoint_identities": identities,
        "fresh_evolution_artifact_policy": {
            "initial_predictions": "fresh_qwen25",
            "error_signatures": "fresh_qwen25",
            "clusters": "fresh_manager_from_qwen25_signatures",
            "candidates": "fresh_manager",
            "failure_history": "fresh_trajectory",
            "qwen3_evolution_artifact_reuse_count": 0,
        },
    })
    _write(path, manifest)


def freeze(config: Mapping[str, Any], output: Path) -> None:
    _activate()
    adapted = _adapt_config(config)
    shared.freeze(adapted, output)
    _augment_manifest(config, output)


def audit(config: Mapping[str, Any], output: Path) -> None:
    _activate()
    adapted = _adapt_config(config)
    shared.audit(adapted, output)
    target = _target(output)
    manifest = load_json(target / "frozen_manifest.json")
    checks = {
        "worker_model_qwen25": (
            manifest["pairwise_request_spec"]["model"] == WORKER_MODEL),
        "two_qwen25_endpoints": (
            len(manifest["worker_endpoint_identities"]) == 2
            and all(item["model"] == WORKER_MODEL
                    for item in manifest["worker_endpoint_identities"])),
        "fresh_error_signatures": (
            manifest["error_signature_source"]
            == "fresh_discovery100_only"),
        "read_only_control_signatures_disabled": (
            PROTOCOL.read_only_control_signatures is False),
        "qwen3_evolution_artifact_reuse_zero": (
            manifest["fresh_evolution_artifact_policy"]
            ["qwen3_evolution_artifact_reuse_count"] == 0),
        "heldout_accessed": False,
    }
    result = {
        "schema_version": "1.0.0",
        "offline_only": True,
        "worker_model": WORKER_MODEL,
        "checks": checks,
        "shared_phase17_algorithm_reused": True,
        "phase17_trajectory_artifacts_reused": False,
    }
    if not all(value for key, value in checks.items()
               if key != "heldout_accessed"):
        raise RuntimeError(f"Qwen2.5 evolution audit failed: {checks}")
    _write(target / "qwen25_offline_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def smoke(config: Mapping[str, Any], output: Path) -> None:
    _activate()
    adapted = _adapt_config(config)
    shared.smoke(adapted, output)
    target = _target(output)
    value = load_json(target / "smoke" / "report.json")
    chain_path = target / "smoke" / "fresh_chain_report.json"
    if not chain_path.is_file():
        manifest = load_json(target / "frozen_manifest.json")
        # Materialize a fresh Qwen2.5 epoch-0 snapshot, then exercise one
        # complete signature -> cluster -> proposal -> competition chain in a
        # non-committing smoke directory.
        shared._initialize_baseline(adapted, target, manifest)
        epoch0 = split._epoch(target, 0)
        rubric = StructuredRubric.load_json(epoch0 / "rubric_committed.json")
        prediction = _validate_baseline_identity(target)
        feedback = RubricFeedback.from_dict(load_json(
            epoch0 / "feedback.json"))
        rows = shared._rows(adapted, "discovery")
        root_id = next(iter(manifest.get("initial_eligible_root_ids", ())), None)
        if root_id is None:
            raise RuntimeError("Qwen2.5 smoke has no eligible Split root")
        managers, _, specs, _ = split._managers(adapted, PROTOCOL)
        smoke_dir = target / "smoke" / "fresh_chain"
        memory, memory_hash = split.freeze_rubric_memory(smoke_dir, rubric)
        history = {"root_states": {root_id: {
            "status": "eligible", "attempt_count": 0}}, "attempts": []}
        started = time.monotonic()
        result = split._prepare(
            shared._runtime_config(adapted), target, smoke_dir, root_id, 1,
            rubric, prediction, feedback, rows, history, managers, specs,
            PROTOCOL, memory, memory_hash)
        result = split._evaluate(
            shared._runtime_config(adapted), smoke_dir, rows, rubric,
            prediction, result, adapted["backend_pool"],
            managers["semantic_cluster"], feedback, PROTOCOL)
        chain = {
            "schema_version": "1.0.0",
            "status": "passed",
            "root_id": root_id,
            "decision": result["decision"],
            "parent_accuracy": result["evaluation"].parent_accuracy,
            "specialized_accuracy": result["evaluation"].specialized_accuracy,
            "child_valid_rate": result["child_valid_rate"],
            "child_count": len(result["candidate"].children),
            "pairwise_prompt_version": result[
                "child_prediction"].prompt_version,
            "worker_model": result[
                "child_prediction"].request_spec.model,
            "formal_evolution_mutated": False,
            "heldout_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        if chain["worker_model"] != WORKER_MODEL:
            raise RuntimeError("Qwen2.5 smoke candidate used the wrong Worker")
        _write(chain_path, chain)
    else:
        chain = load_json(chain_path)
    value.update({
        "worker_model": WORKER_MODEL,
        "fresh_signature_candidate_chain": chain,
        "qwen3_evolution_artifact_reuse_count": 0,
    })
    _write(target / "smoke" / "report.json", value)
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": "passed", "details": value}
    _write(target / "stage_status.json", status)


def run(config: Mapping[str, Any], output: Path) -> None:
    _activate()
    adapted = _adapt_config(config)
    target = _target(output)
    manifest = load_json(target / "frozen_manifest.json")
    shared._initialize_baseline(adapted, target, manifest)
    _validate_baseline_identity(target)
    shared.run(adapted, output)
    status = load_json(target / "stage_status.json").get("run", {})
    if status.get("status") != "passed":
        details = status.get("details", {})
        raise RuntimeError(
            "Qwen2.5 evolution did not complete; "
            f"status={status.get('status')!r}, details={details}. "
            "Resume qwen25-evolution-run before starting report/heldout/VL-RewardBench."
        )


def report(config: Mapping[str, Any], output: Path) -> None:
    _activate()
    shared.report(_adapt_config(config), output)
    path = _target(output) / "final" / "discovery_report.json"
    value = load_json(path)
    value.update({
        "worker_model": WORKER_MODEL,
        "fresh_qwen25_trajectory": True,
        "qwen3_evolution_artifact_reuse_count": 0,
    })
    _write(path, value)


def _root_metrics(execution, rubric: StructuredRubric,
                  rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for root_id in rubric.root_ids:
        votes = []
        for trace in execution.traces:
            matches = [item for item in trace.roots if item.root_id == root_id]
            if len(matches) != 1:
                raise RuntimeError(f"incomplete root trace: {root_id}")
            votes.append(matches[0].subtree_vote)
        result[root_id] = base._heldout_vote_metrics(votes, rows)
    return result


def _evaluate_heldout_system(config: Mapping[str, Any], work: Path,
                             rubric: StructuredRubric, rows, label: str):
    prediction = shared._worker_prediction(
        _adapt_config(config), work, rubric, rows, label)
    execution, answers = execute_offline_m1(rubric, prediction, rows)
    return prediction, execution, answers, {
        "m1": base._heldout_vote_metrics(answers, rows),
        "root_subtrees": _root_metrics(execution, rubric, rows),
    }


def heldout(config: Mapping[str, Any], output: Path) -> None:
    _activate()
    adapted = _adapt_config(config)
    target = _target(output)
    shared._load_frozen(adapted, output)
    report_path = target / "final" / "discovery_report.json"
    if not report_path.is_file():
        raise RuntimeError("run qwen25-evolution-report first")
    rows = shared._rows(adapted, "heldout")
    rubrics = {
        "qwen25_initial": base.build_multicrit_open_ended_init_rubric(),
        "qwen25_transferred_phase17_e4": transfer._rubric(output),
        "qwen25_specific_final": StructuredRubric.load_json(
            target / "final" / "rubric.json"),
    }
    heldout_dir = target / "heldout500"
    frozen = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "worker_model": WORKER_MODEL,
        "heldout_dataset_sha256": shared._file_sha(config["heldout_dataset"]),
        "formal_discovery_report_sha256": file_sha256(report_path),
        "rubrics": {name: rubric.rubric_sha256
                    for name, rubric in rubrics.items()},
        "selection_after_heldout_forbidden": True,
    }
    frozen_path = heldout_dir / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != frozen:
        raise RuntimeError("Qwen2.5 heldout manifest drift")
    _write(frozen_path, frozen)
    artifacts = {}
    for name, rubric in rubrics.items():
        prediction, execution, answers, metrics = _evaluate_heldout_system(
            config, heldout_dir / name, rubric, rows, name)
        system_dir = heldout_dir / name
        system_dir.mkdir(parents=True, exist_ok=True)
        prediction.save_json(system_dir / "pairwise.json")
        execution.save_json(system_dir / "m1_execution.json")
        artifacts[name] = (answers, metrics)
    initial_answers = artifacts["qwen25_initial"][0]
    transfer_answers = artifacts["qwen25_transferred_phase17_e4"][0]
    final_answers = artifacts["qwen25_specific_final"][0]
    value = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "worker_model": WORKER_MODEL,
        "systems": {name: item[1] for name, item in artifacts.items()},
        "paired": {
            "specific_vs_initial": base._paired_heldout_comparison(
                initial_answers, final_answers, rows),
            "specific_vs_transferred_e4": base._paired_heldout_comparison(
                transfer_answers, final_answers, rows),
        },
        "selection_after_heldout_forbidden": True,
    }
    _write(heldout_dir / "report.json", value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        name: item[1]["m1"]["accuracy"] for name, item in artifacts.items()}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def final_report(config: Mapping[str, Any], output: Path) -> None:
    _activate()
    adapted = _adapt_config(config)
    target = _target(output)
    shared._load_frozen(adapted, output)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout_value = load_json(target / "heldout500" / "report.json")
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "worker_model": WORKER_MODEL,
        "exploratory": True,
        "discovery": discovery,
        "heldout": heldout_value,
        "vl_rewardbench_required": True,
        "next_required_stage": "vlrb-qwen25-evolved-freeze",
    }
    _write(target / "final_report.json", value)
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed", "details": {
        "final_rubric_sha256": discovery["final_rubric_sha256"],
        "vl_rewardbench_required": True,
    }}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["final_report"]["details"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = dict(zip(STAGES, (
        freeze, audit, smoke, run, report, heldout, final_report,
    )))
    if stage not in actions:
        raise ValueError(f"unsupported Qwen2.5 evolution stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}",
          flush=True)

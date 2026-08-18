"""Prompt-v2-aligned five-root Locked-Split + Role-aware Refine experiment.

The module deliberately reuses the frozen Phase-10 operator implementation
while replacing every Pairwise Worker request in the new trajectory with the
cache-oriented Prompt v2 identity.  Historical Prompt-v1 trajectories remain
read-only and unchanged.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    EvolutionContext,
    PairwisePredictionOutput,
    RubricFeedback,
    StructuredCriterionSnapshot,
    StructuredRubric,
    detect_specialize_trigger,
    execute_offline_m1,
    project_pairwise_prediction,
)
from critiq.structured.version import (
    PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
)

from . import pairwise_cache_ablation as cache_exp
from . import refine_evolution as refine
from . import run_rubric_evolution as base
from . import split_evolution as split
from .experiment_utils import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    load_jsonl_dataset,
)
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase16_prompt_v2_locked_split_refine_v2"
CHECKPOINT_HELDOUT_DIR = "checkpoint_heldout500"
CHECKPOINT_EPOCHS = tuple(range(6))
PROTOCOL_VERSION = "prompt-v2-aligned-locked-split-refine-v2"
PREVIOUS_EXPERIMENT_DIR = "phase16_prompt_v2_locked_split_refine_v1"
SOURCE_PHASE10 = refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR
SOURCE_ABLATION = cache_exp.EXPERIMENT_DIR
SOURCE_DISCOVERY_VARIANT = cache_exp.VARIANT_PROMPT_V2_DYNAMIC
SOURCE_HELDOUT_VARIANT = cache_exp.VARIANT_PROMPT_V2_DYNAMIC
PAIRWISE_PROMPT_MODE = "v2_cache"
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")
SPLIT_TAU = 0.75

SETTINGS = {
    **refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_V1,
    "protocol_version": PROTOCOL_VERSION,
    "worker_endpoint": "configured_available_slot_pool",
    "worker_endpoints": list(ENDPOINT_IDS),
    "pairwise_prompt_mode": PAIRWISE_PROMPT_MODE,
    "pairwise_prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    "initial_prediction_source": (
        "pairwise_worker_cache_prompt_ablation_v1/"
        "discovery90/s3_prompt_v2_dynamic"),
    "error_signature_policy": "fresh_unless_exact_identity",
    "split_trigger": {"tau_split": SPLIT_TAU, "tau_cov_high": 0.80},
    "vl_rewardbench_required": True,
}

PROTOCOL = split.EvolutionProtocol(
    EXPERIMENT_DIR,
    "prompt-v2-evolution",
    rubric_memory_mode="global_rubric_v1",
    control_experiment_dir=None,
    read_only_control_signatures=False,
    pairwise_endpoint="vllm-8000",
    allow_configured_endpoint_pool=True,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _checkpoint_target(output: Path) -> Path:
    return _target(output) / CHECKPOINT_HELDOUT_DIR


def _checkpoint_rubric(target: Path, epoch: int) -> StructuredRubric:
    if epoch not in CHECKPOINT_EPOCHS:
        raise ValueError(f"unsupported checkpoint epoch: {epoch}")
    path = target / "epochs" / f"epoch_{epoch:02d}" / "rubric_committed.json"
    if not path.is_file():
        raise RuntimeError(f"checkpoint Rubric is missing: {path}")
    return StructuredRubric.load_json(path)


def _description_sha256(description: str) -> str:
    return canonical_sha256({"description": description})


def _v2_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["_pairwise_prompt_mode"] = PAIRWISE_PROMPT_MODE
    # Phase16-v2 intentionally broadens only Split scheduling.  The globally
    # frozen Split-v1 configuration and every Refine/acceptance threshold stay
    # unchanged.
    value["evolution_policy"]["trigger_thresholds"]["tau_split"] = SPLIT_TAU
    return value


def _config(config: Mapping[str, Any]) -> dict[str, Any]:
    refine._config(config)
    value = config.get("prompt_v2_aligned_evolution")
    if value != SETTINGS:
        raise ValueError(
            "prompt_v2_aligned_evolution must equal the frozen v1 protocol")
    request = config.get("worker_request_kwargs")
    if (not isinstance(request, Mapping)
            or request.get("temperature") != 0.5
            or request.get("max_tokens") != 2048):
        raise ValueError("Prompt v2 evolution requires temperature=.5/max_tokens=2048")
    pool = base.BackendPoolSpec.from_dict(config["backend_pool"])
    if {item.endpoint_id for item in pool.endpoints} != set(ENDPOINT_IDS):
        raise ValueError("Prompt v2 evolution requires vllm-8000 and vllm-8001")
    if pool.global_request_concurrency != sum(
            item.max_concurrency for item in pool.endpoints):
        raise ValueError("Prompt v2 evolution requires available-slot pool capacity")
    split.validate_policy(config, PROTOCOL)
    return dict(value)


def _source_prediction(output: Path, split_name: str) -> Path:
    if split_name == "discovery":
        return (output / SOURCE_ABLATION / "discovery90"
                / SOURCE_DISCOVERY_VARIANT / "predictions.json")
    if split_name == "heldout":
        return (output / SOURCE_ABLATION / "heldout500"
                / SOURCE_HELDOUT_VARIANT / "predictions.json")
    raise ValueError(f"unsupported split: {split_name}")


def _initial_inputs(config: Mapping[str, Any], output: Path):
    """Load only the frozen data/rubric lineage, not the old Worker identity."""
    base._require(output, "finalize_phase5")
    manifest = load_json(output / "frozen_manifest.json")
    if not manifest.get("phase6_allowed"):
        raise RuntimeError("Phase 5 has not enabled rubric evolution")
    discovery_path = base._path(config["discovery_dataset"])
    if (file_sha256(discovery_path).lower()
            != manifest["discovery_dataset_sha256"].lower()):
        raise RuntimeError("discovery-90 changed before Prompt v2 evolution")
    rows = load_jsonl_dataset(discovery_path, expected_count=90)
    rubric = StructuredRubric.load_json(manifest["rubric_path"])
    if rubric.rubric_sha256 != manifest["rubric_sha256"]:
        raise RuntimeError("Phase 5 initial Rubric changed before Prompt v2 evolution")
    return manifest, rubric, rows


def _load_v2_source(
    config: Mapping[str, Any], output: Path, rubric: StructuredRubric,
    rows: Sequence[Mapping[str, Any]], split_name: str,
) -> tuple[PairwisePredictionOutput, Path]:
    path = _source_prediction(output, split_name)
    if not path.is_file():
        raise RuntimeError(
            f"completed Prompt v2 {split_name} source is required: {path}")
    source = PairwisePredictionOutput.load_json(path)
    projected = project_pairwise_prediction(source, rubric)
    expected = base._expected_pairwise_request_spec(_v2_config(config), rows)
    if (not refine._same_pairwise_scientific_identity(
            projected.request_spec, expected)
            or projected.prompt_version
            != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION):
        raise RuntimeError(f"Prompt v2 {split_name} source identity drift")
    return projected, path


def _manager_contract(config: Mapping[str, Any]):
    managers, profiles, specs, identities = split._managers(config, PROTOCOL)
    refine_manager, refine_profile = refine._manager(config)
    _, retry_specs = refine._integrated_retry_specs(config, PROTOCOL)
    return {
        "managers": managers,
        "profiles": profiles,
        "specs": specs,
        "identities": identities,
        "refine_manager": refine_manager,
        "refine_profile": refine_profile,
        "refine_specs": {
            key: value.to_dict()
            for key, value in refine_manager.request_specs().items()},
        "retry_specs": retry_specs,
    }


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _config(config)
    refine._validate_output(output)
    target = _target(output)
    manifest_path = target / "frozen_manifest.json"
    if manifest_path.exists():
        print("prompt-v2-evolution-freeze already completed")
        return
    phase5, rubric, rows = _initial_inputs(config, output)
    prediction, source_path = _load_v2_source(
        config, output, rubric, rows, "discovery")
    contract = _manager_contract(config)
    roots = tuple(rubric.root_ids)
    if len(roots) != 5:
        raise RuntimeError("Prompt v2 evolution requires exactly five initial roots")

    # Build feedback only from Prompt-v2 votes.  This is the causal treatment.
    provisional = {
        "pairwise_request_spec": prediction.request_spec.to_dict(),
    }
    votes, feedback = split._snapshot(
        split._epoch(target, 0), rubric, prediction, rows, provisional)
    context = EvolutionContext(rubric, feedback)
    runtime_config = _v2_config(config)
    thresholds = runtime_config["evolution_policy"]["trigger_thresholds"]
    triggers = {
        root_id: detect_specialize_trigger(
            context, root_id, thresholds).to_dict()
        for root_id in roots
    }
    eligible = [root_id for root_id in roots if triggers[root_id]["triggered"]]
    memory, memory_hash = split.freeze_rubric_memory(
        split._epoch(target, 0), rubric)
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "exploratory": True,
        "integration_settings": settings,
        "policy": split.POLICY_V1,
        "split_trigger_thresholds": {
            "tau_split": thresholds["tau_split"],
            "tau_cov_high": thresholds["tau_cov_high"],
        },
        "phase5_manifest_sha256": canonical_sha256(phase5),
        "discovery_dataset_sha256": phase5["discovery_dataset_sha256"],
        "initial_rubric_sha256": rubric.rubric_sha256,
        "initial_root_ids": list(roots),
        "initial_triggers": triggers,
        "initial_eligible_root_ids": eligible,
        "initial_eligible_root_count": len(eligible),
        "pairwise_prompt_mode": PAIRWISE_PROMPT_MODE,
        "pairwise_prompt_version": prediction.prompt_version,
        "pairwise_request_spec": prediction.request_spec.to_dict(),
        "pairwise_execution_pool": base.BackendPoolSpec.from_dict(
            config["backend_pool"]).to_dict(),
        "initial_prediction_source": str(source_path.resolve()),
        "initial_prediction_source_sha256": file_sha256(source_path),
        "initial_prediction_projected_sha256": canonical_sha256(
            prediction.to_dict()),
        "manager_profiles": contract["profiles"],
        "manager_request_specs": contract["specs"],
        "manager_endpoint_identities": contract["identities"],
        "refine_protocol": refine.REFINE_V1,
        "refine_manager_profile": contract["refine_profile"],
        "refine_manager_request_specs": contract["refine_specs"],
        "locked_retry_manager_request_specs": contract["retry_specs"],
        "rubric_memory_mode": "global_rubric_v1",
        "epoch_00_rubric_memory_sha256": memory_hash,
        "epoch_00_rubric_memory": memory,
        "error_signature_source": "fresh_unless_exact_identity",
        "heldout_access": "forbidden_until_final_report",
        "vl_rewardbench_required": True,
        "selection_after_heldout_forbidden": True,
    }
    _write(manifest_path, manifest)
    _write(split._epoch(target, 0) / "summary.json", {
        "epoch": 0,
        "m1": base._metrics(votes, rows),
        "triggers": triggers,
        "accepted_roots": [],
        "pairwise_prompt_version": prediction.prompt_version,
    })
    states = {
        root_id: {
            "status": "eligible" if triggers[root_id]["triggered"] else "not_eligible",
            "attempt_count": 0,
            "accepted_epoch": None,
            "children": [],
            "locked_retry": None,
        }
        for root_id in roots
    }
    _write(target / "evolution_history.json", {
        "schema_version": "1.0.0",
        "current_epoch": 0,
        "completed": False,
        "stop_reason": None,
        "root_states": states,
        "attempts": [],
        "refine_states": {},
        "refine_attempts": [],
    })
    _write(target / "stage_status.json", {"freeze": {
        "status": "passed",
        "details": {"eligible": eligible, "initial_m1": base._metrics(votes, rows)},
    }})
    print(json.dumps({
        "target": str(target),
        "eligible_split_roots": eligible,
        "initial_m1_accuracy": base._metrics(votes, rows)["accuracy"],
        "prompt_version": prediction.prompt_version,
    }, indent=2, ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    settings = _config(config)
    target = _target(output)
    manifest = load_json(target / "frozen_manifest.json")
    rubric = StructuredRubric.load_json(
        split._epoch(target, 0) / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(
        split._epoch(target, 0) / "discovery_pairwise.json")
    feedback = RubricFeedback.from_dict(load_json(
        split._epoch(target, 0) / "feedback.json"))
    rows = refine._rows(config, "discovery")
    expected = base._expected_pairwise_request_spec(_v2_config(config), rows)
    contract = _manager_contract(config)
    checks = {
        "prompt_v2_prediction": (
            prediction.prompt_version
            == PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION),
        "pairwise_request_spec_match": refine._same_pairwise_scientific_identity(
            prediction.request_spec, expected),
        "five_initial_roots": len(rubric.root_ids) == 5 and len(rubric.nodes) == 5,
        "all_initial_roots_split_eligible": (
            len(manifest["initial_eligible_root_ids"]) == 5
            and set(manifest["initial_eligible_root_ids"]) == set(rubric.root_ids)),
        "manager_request_specs_match": (
            contract["specs"] == manifest["manager_request_specs"]),
        "refine_request_specs_match": (
            contract["refine_specs"] == manifest["refine_manager_request_specs"]),
        "locked_retry_specs_match": (
            contract["retry_specs"]
            == manifest["locked_retry_manager_request_specs"]),
        "heldout_accessed": False,
    }
    if not all(value for key, value in checks.items() if key != "heldout_accessed"):
        # Never leave a previously-passed audit artifact/status usable after
        # the live configuration drifts from the frozen experiment contract.
        audit_path = target / "offline_audit.json"
        if audit_path.exists():
            audit_path.unlink()
        status = load_json(target / "stage_status.json")
        status["audit"] = {"status": "failed", "details": {"checks": checks}}
        _write(target / "stage_status.json", status)
        raise RuntimeError(f"Prompt v2 evolution audit failed: {checks}")
    split_roots = list(manifest["initial_eligible_root_ids"])
    refine_nodes = refine._integrated_refine_schedule(
        rubric, feedback, config, split_roots)
    value = {
        "schema_version": "1.0.0",
        "offline_only": True,
        "settings": settings,
        "checks": checks,
        "initial_split_roots": split_roots,
        "initial_refine_nodes_after_split_precedence": list(refine_nodes),
        "old_prompt_cache_reuse_allowed": False,
        "error_signatures_generated_or_reused": 0,
    }
    _write(target / "offline_audit.json", value)
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": value}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def smoke(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    status = load_json(target / "stage_status.json")
    if (status.get("audit", {}).get("status") != "passed"
            or not (target / "offline_audit.json").exists()):
        raise RuntimeError("run prompt-v2-evolution-audit first")
    report_path = target / "smoke" / "report.json"
    if report_path.exists():
        print(json.dumps(load_json(report_path), indent=2, ensure_ascii=False))
        return
    manifest = load_json(target / "frozen_manifest.json")
    # The v1 smoke exercised init_01 with the same Prompt-v2 predictions,
    # Manager identities, and request protocol.  Raising tau_split from .70 to
    # .75 only schedules three additional roots; it does not change that smoke
    # attempt.  Reuse it only after an exact frozen-identity audit.
    previous = output / PREVIOUS_EXPERIMENT_DIR
    previous_manifest_path = previous / "frozen_manifest.json"
    previous_report_path = previous / "smoke" / "report.json"
    if previous_manifest_path.exists() and previous_report_path.exists():
        previous_manifest = load_json(previous_manifest_path)
        previous_report = load_json(previous_report_path)
        root_id = next(iter(manifest["initial_eligible_root_ids"]), None)
        identity_keys = (
            "discovery_dataset_sha256", "initial_rubric_sha256",
            "initial_prediction_source_sha256",
            "initial_prediction_projected_sha256", "pairwise_prompt_version",
            "pairwise_request_spec", "manager_profiles",
            "manager_request_specs", "refine_manager_profile",
            "refine_manager_request_specs", "locked_retry_manager_request_specs",
        )
        reusable = (
            root_id is not None
            and all(previous_manifest.get(key) == manifest.get(key)
                    for key in identity_keys)
            and previous_manifest.get("initial_triggers", {}).get(root_id)
            == manifest.get("initial_triggers", {}).get(root_id)
            and previous_report.get("status") == "passed"
            and previous_report.get("root_id") == root_id
            and previous_report.get("pairwise_prompt_version")
            == PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
            and previous_report.get("formal_evolution_mutated") is False
            and previous_report.get("heldout_accessed") is False
        )
        if reusable:
            value = dict(previous_report)
            value.update({
                "reused": True,
                "reuse_reason": "same_init_01_chain_threshold_only_broadened",
                "source_experiment": PREVIOUS_EXPERIMENT_DIR,
                "source_report_sha256": file_sha256(previous_report_path),
                "source_elapsed_seconds": previous_report.get("elapsed_seconds"),
                "elapsed_seconds": 0.0,
            })
            _write(report_path, value)
            status = load_json(target / "stage_status.json")
            status["smoke"] = {"status": "passed", "details": value}
            _write(target / "stage_status.json", status)
            print(json.dumps(value, indent=2, ensure_ascii=False))
            return
    epoch0 = split._epoch(target, 0)
    rubric = StructuredRubric.load_json(epoch0 / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(epoch0 / "discovery_pairwise.json")
    feedback = RubricFeedback.from_dict(load_json(epoch0 / "feedback.json"))
    rows = refine._rows(config, "discovery")
    root_id = next(iter(manifest["initial_eligible_root_ids"]), None)
    if root_id is None:
        raise RuntimeError("Prompt v2 smoke has no eligible root")
    managers, _, specs, _ = split._managers(config, PROTOCOL)
    memory, memory_hash = split.freeze_rubric_memory(target / "smoke", rubric)
    history = {"root_states": {root_id: {
        "status": "eligible", "attempt_count": 0}}, "attempts": []}
    started = time.monotonic()
    result = split._prepare(
        _v2_config(config), target, target / "smoke", root_id, 1,
        rubric, prediction, feedback, rows, history, managers, specs,
        PROTOCOL, memory, memory_hash)
    result = split._evaluate(
        _v2_config(config), target / "smoke", rows, rubric, prediction,
        result, config["backend_pool"], managers["semantic_cluster"],
        feedback, PROTOCOL)
    child = result["child_prediction"]
    if child.prompt_version != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION:
        raise RuntimeError("smoke child prediction did not use Prompt v2")
    value = {
        "schema_version": "1.0.0",
        "status": "passed",
        "root_id": root_id,
        "decision": result["decision"],
        "parent_accuracy": result["evaluation"].parent_accuracy,
        "specialized_accuracy": result["evaluation"].specialized_accuracy,
        "child_valid_rate": result["child_valid_rate"],
        "child_count": len(result["candidate"].children),
        "pairwise_prompt_version": child.prompt_version,
        "formal_evolution_mutated": False,
        "heldout_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write(report_path, value)
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": "passed", "details": value}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def run(config: Mapping[str, Any], output: Path) -> None:
    settings = _config(config)
    target = _target(output)
    if not (target / "offline_audit.json").exists():
        raise RuntimeError("run prompt-v2-evolution-freeze/audit first")
    stage_status = load_json(target / "stage_status.json")
    if stage_status.get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run prompt-v2-evolution-smoke before the full run")
    manifest = load_json(target / "frozen_manifest.json")
    history = load_json(target / "evolution_history.json")
    if history.get("completed"):
        print("prompt-v2-evolution-run already completed")
        return
    refine._run_five_root_locked_split_refine_impl(
        _v2_config(config),
        target=target,
        manifest=manifest,
        history=history,
        settings=settings,
        protocol=PROTOCOL,
        log_prefix="prompt-v2 aligned evolution",
        execution_backend_pool=config["backend_pool"],
    )


def report(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run prompt-v2-evolution-run to completion first")
    final_epoch = split._epoch(target, history["current_epoch"])
    rubric = StructuredRubric.load_json(final_epoch / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(
        final_epoch / "discovery_pairwise.json")
    if prediction.prompt_version != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION:
        raise RuntimeError("final discovery prediction prompt version drift")
    rows = refine._rows(config, "discovery")
    execution, answers = execute_offline_m1(rubric, prediction, rows)
    initial = load_json(split._epoch(target, 0) / "summary.json")["m1"]
    metrics = base._metrics(answers, rows)
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    execution.save_json(final_dir / "m1_execution.json")
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "final_epoch": history["current_epoch"],
        "initial_m1": initial,
        "final_m1": metrics,
        "m1_accuracy_delta": metrics["accuracy"] - initial["accuracy"],
        "final_rubric_sha256": rubric.rubric_sha256,
        "node_count_initial": 5,
        "node_count_final": len(rubric.nodes),
        "pairwise_prompt_version": prediction.prompt_version,
        "split_attempts": history["attempts"],
        "refine_attempts": history["refine_attempts"],
        "accepted_split_count": sum(
            item["decision"] == split.ACCEPTED for item in history["attempts"]),
        "accepted_refine_count": sum(
            item["decision"] == "accepted"
            for item in history["refine_attempts"]),
        "cost": split._cost_summary(target, history),
        "vl_rewardbench_required": True,
        "selection_after_external_evaluation_forbidden": True,
    }
    _write(final_dir / "discovery_report.json", value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_m1_accuracy": metrics["accuracy"],
        "final_rubric_sha256": rubric.rubric_sha256,
    }}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def heldout(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    discovery_path = target / "final" / "discovery_report.json"
    if not discovery_path.exists():
        raise RuntimeError("run prompt-v2-evolution-report first")
    final_rubric = StructuredRubric.load_json(target / "final" / "rubric.json")
    initial_rubric = StructuredRubric.load_json(
        split._epoch(target, 0) / "rubric_committed.json")
    phase10_rubric = StructuredRubric.load_json(
        output / SOURCE_PHASE10 / "final" / "rubric.json")
    rows = refine._rows(config, "heldout")
    control, source_path = _load_v2_source(
        config, output, phase10_rubric, rows, "heldout")
    initial = project_pairwise_prediction(control, initial_rubric)
    heldout_dir = target / "heldout500"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "final_discovery_report_sha256": file_sha256(discovery_path),
        "heldout_dataset_sha256": file_sha256(
            base._path(config["heldout_dataset"])),
        "control_rubric_sha256": phase10_rubric.rubric_sha256,
        "control_prediction_path": str(source_path.resolve()),
        "control_prediction_sha256": file_sha256(source_path),
        "pairwise_request_spec": control.request_spec.to_dict(),
        "pairwise_prompt_version": control.prompt_version,
        "selection_after_heldout_forbidden": True,
        "vl_rewardbench_required": True,
    }
    frozen_path = heldout_dir / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != manifest:
        raise RuntimeError("Prompt v2 evolution heldout manifest drift")
    _write(frozen_path, manifest)

    source_descriptions = {item.name: item.description for item in control.criteria}
    changed_ids = tuple(
        node_id for node_id in final_rubric.preorder_node_ids()
        if source_descriptions.get(final_rubric.get_node(node_id).criterion.name)
        != final_rubric.get_node(node_id).criterion.description)
    generated = None
    generated_path = heldout_dir / "changed_node_predictions.json"
    if changed_ids:
        if generated_path.exists():
            generated = PairwisePredictionOutput.load_json(generated_path)
        else:
            changed_rubric = StructuredRubric(
                {node_id: final_rubric.get_node(node_id)
                 for node_id in changed_ids}, (), changed_ids)
            generated, _, _ = base._generate_pairwise(
                _v2_config(config), heldout_dir, changed_rubric, rows,
                "changed_nodes", execution_backend_pool=config["backend_pool"],
                request_backend_id=control.request_spec.backend_id,
                request_level_progress=True)
            generated.save_json(generated_path)
    combined = refine._combine_heldout_predictions(
        control, generated, final_rubric)
    if combined.prompt_version != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION:
        raise RuntimeError("combined heldout Prompt v2 metadata drift")
    combined.save_json(heldout_dir / "combined_pairwise.json")
    final_execution, final_answers = execute_offline_m1(
        final_rubric, combined, rows)
    final_execution.save_json(heldout_dir / "m1_execution.json")
    initial_execution, initial_answers = execute_offline_m1(
        initial_rubric, initial, rows)
    control_execution, control_answers = execute_offline_m1(
        phase10_rubric, control, rows)

    def root_metrics(execution, rubric):
        values = {}
        for root_id in rubric.root_ids:
            votes = []
            for trace in execution.traces:
                match = [item for item in trace.roots if item.root_id == root_id]
                if len(match) != 1 or match[0].subtree_vote is None:
                    raise RuntimeError(f"incomplete heldout root trace: {root_id}")
                votes.append(match[0].subtree_vote)
            values[root_id] = base._heldout_vote_metrics(votes, rows)
        return values
    value = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "initial_five_root_prompt_v2": base._heldout_vote_metrics(
            initial_answers, rows),
        "phase10_prompt_v2_control": base._heldout_vote_metrics(
            control_answers, rows),
        "prompt_v2_evolved_treatment": base._heldout_vote_metrics(
            final_answers, rows),
        "treatment_vs_phase10_prompt_v2": base._paired_heldout_comparison(
            control_answers, final_answers, rows),
        "treatment_vs_initial_prompt_v2": base._paired_heldout_comparison(
            initial_answers, final_answers, rows),
        "root_subtree_metrics": {
            "initial_five_root_prompt_v2": root_metrics(
                initial_execution, initial_rubric),
            "phase10_prompt_v2_control": root_metrics(
                control_execution, phase10_rubric),
            "prompt_v2_evolved_treatment": root_metrics(
                final_execution, final_rubric),
        },
        "reused_exact_criteria": len(final_rubric.nodes) - len(changed_ids),
        "generated_changed_criteria": len(changed_ids),
        "generated_node_ids": list(changed_ids),
        "selection_after_heldout_forbidden": True,
        "vl_rewardbench_required": True,
    }
    _write(heldout_dir / "report.json", value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        "control_accuracy": value["phase10_prompt_v2_control"]["accuracy"],
        "treatment_accuracy": value["prompt_v2_evolved_treatment"]["accuracy"],
        "generated_changed_criteria": len(changed_ids),
    }}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def final_report(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout_value = load_json(target / "heldout500" / "report.json")
    paired = heldout_value["treatment_vs_phase10_prompt_v2"]
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "discovery": discovery,
        "heldout": heldout_value,
        "primary": {
            "phase10_prompt_v2_control_accuracy": heldout_value[
                "phase10_prompt_v2_control"]["accuracy"],
            "prompt_v2_evolved_accuracy": heldout_value[
                "prompt_v2_evolved_treatment"]["accuracy"],
            "net_corrected": paired["net_corrected"],
            "mcnemar_exact_two_sided_p": paired[
                "mcnemar_exact_two_sided_p"],
        },
        "next_required_stage": "vlrb-prompt-v2-evolved-freeze",
        "vl_rewardbench_required": True,
    }
    _write(target / "final_report.json", value)
    lines = [
        "# Prompt v2 Aligned Evolution v1", "",
        "Exploratory paired evaluation; no post-heldout model selection.", "",
        "| System | heldout-500 ACC | Coverage |", "|---|---:|---:|",
    ]
    for label in ("initial_five_root_prompt_v2", "phase10_prompt_v2_control",
                  "prompt_v2_evolved_treatment"):
        item = heldout_value[label]
        lines.append(f"| {label} | {item['accuracy']:.4f} | {item['coverage']:.4f} |")
    lines.extend(["", "VL-RewardBench evaluation is mandatory and remains pending."])
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed", "details": value["primary"]}
    _write(target / "stage_status.json", status)
    print(json.dumps(value["primary"], indent=2))


def _checkpoint_variant_plan(target: Path) -> tuple[dict[str, Any], ...]:
    """Return one immutable request identity per non-final description.

    Checkpoint replay never re-generates a description already present in the
    final heldout artifact.  Descriptions rather than node IDs are the cache
    identity because a Refine can replace one node's text repeatedly.
    """
    final = _checkpoint_rubric(target, 5)
    final_descriptions = {
        node_id: node.criterion.description
        for node_id, node in final.nodes.items()
    }
    variants: dict[tuple[str, str], dict[str, Any]] = {}
    for epoch in CHECKPOINT_EPOCHS[:-1]:
        rubric = _checkpoint_rubric(target, epoch)
        for node_id in rubric.preorder_node_ids():
            node = rubric.get_node(node_id)
            description = node.criterion.description
            if final_descriptions.get(node_id) == description:
                continue
            key = (node_id, _description_sha256(description))
            item = variants.setdefault(key, {
                "node_id": node_id,
                "criterion_name": node.criterion.name,
                "description": description,
                "description_sha256": key[1],
                "epochs": [],
            })
            item["epochs"].append(epoch)
    return tuple(sorted(variants.values(), key=lambda item: (
        item["node_id"], item["description_sha256"])))


def _checkpoint_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _config(config)
    target = _target(output)
    final_report_path = target / "final_report.json"
    final_heldout_path = target / "heldout500" / "report.json"
    final_prediction_path = target / "heldout500" / "combined_pairwise.json"
    if not all(path.is_file() for path in (
            final_report_path, final_heldout_path, final_prediction_path)):
        raise RuntimeError("completed Phase16 final heldout artifacts are required")
    _, _, rows = _initial_inputs(config, output)
    heldout_rows = load_jsonl_dataset(
        base._path(config["heldout_dataset"]), expected_count=500)
    final_prediction = PairwisePredictionOutput.load_json(final_prediction_path)
    expected = base._expected_pairwise_request_spec(_v2_config(config), heldout_rows)
    if (not refine._same_pairwise_scientific_identity(
            final_prediction.request_spec, expected)
            or final_prediction.prompt_version
            != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION):
        raise RuntimeError("Phase16 final heldout Prompt v2 identity drift")
    checkpoints = []
    for epoch in CHECKPOINT_EPOCHS:
        rubric_path = target / "epochs" / f"epoch_{epoch:02d}" / "rubric_committed.json"
        rubric = _checkpoint_rubric(target, epoch)
        checkpoints.append({
            "epoch": epoch,
            "rubric_path": str(rubric_path.resolve()),
            "rubric_file_sha256": file_sha256(rubric_path),
            "rubric_sha256": rubric.rubric_sha256,
            "node_count": len(rubric.nodes),
        })
    variants = _checkpoint_variant_plan(target)
    return {
        "schema_version": "1.0.0",
        "experiment": "phase16_checkpoint_heldout_diagnostic_v1",
        "exploratory": True,
        "purpose": (
            "Post-hoc epoch trajectory diagnostic only; checkpoint heldout "
            "results must not select or replace the frozen Phase16 final Rubric."),
        "source_experiment": EXPERIMENT_DIR,
        "pairwise_prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "pairwise_request_spec": final_prediction.request_spec.to_dict(),
        "heldout_dataset_path": str(base._path(config["heldout_dataset"]).resolve()),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "heldout_sample_count": len(heldout_rows),
        "phase16_final_report_sha256": file_sha256(final_report_path),
        "phase16_final_heldout_report_sha256": file_sha256(final_heldout_path),
        "final_combined_prediction_path": str(final_prediction_path.resolve()),
        "final_combined_prediction_sha256": file_sha256(final_prediction_path),
        "checkpoints": checkpoints,
        "variants": list(variants),
        "new_pairwise_request_count": len(variants) * len(heldout_rows),
        "selection_after_heldout_forbidden": True,
    }


def checkpoint_heldout_freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _checkpoint_target(output)
    path = target / "frozen_manifest.json"
    value = _checkpoint_manifest(config, output)
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError("Phase16 checkpoint heldout manifest drift")
        print("prompt-v2-evolution-checkpoint-heldout-freeze already completed")
        return
    _write(path, value)
    print(json.dumps({
        "checkpoint_count": len(value["checkpoints"]),
        "unique_historical_descriptions": len(value["variants"]),
        "new_pairwise_request_count": value["new_pairwise_request_count"],
    }, indent=2))


def _load_checkpoint_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target = _checkpoint_target(output)
    path = target / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run prompt-v2-evolution-checkpoint-heldout-freeze first")
    value = load_json(path)
    if value != _checkpoint_manifest(config, output):
        raise RuntimeError("Phase16 checkpoint heldout frozen manifest drift")
    return value


def checkpoint_heldout_audit(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_checkpoint_manifest(config, output)
    target = _checkpoint_target(output)
    final = PairwisePredictionOutput.load_json(
        Path(manifest["final_combined_prediction_path"]))
    expected_names = tuple(
        _checkpoint_rubric(_target(output), 5).get_node(node_id).criterion.name
        for node_id in _checkpoint_rubric(_target(output), 5).preorder_node_ids())
    checks = {
        "six_checkpoints": len(manifest["checkpoints"]) == 6,
        "expected_14_historical_descriptions": len(manifest["variants"]) == 14,
        "request_count_matches_variants": manifest["new_pairwise_request_count"] == 7000,
        "final_prediction_criteria_match": tuple(item.name for item in final.criteria) == expected_names,
        "final_prediction_prompt_v2": (
            final.prompt_version == PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION),
        "heldout_selection_forbidden": manifest["selection_after_heldout_forbidden"],
    }
    value = {"schema_version": "1.0.0", "offline_only": True,
             "checks": checks, "manifest_sha256": file_sha256(
                 target / "frozen_manifest.json")}
    _write(target / "offline_audit.json", value)
    if not all(checks.values()):
        raise RuntimeError(f"Phase16 checkpoint heldout audit failed: {checks}")
    print(json.dumps(value, indent=2))


def _validate_variant_prediction(prediction: PairwisePredictionOutput,
                                item: Mapping[str, Any],
                                reference: PairwisePredictionOutput) -> None:
    if (prediction.sample_ids != reference.sample_ids
            or prediction.sample_fingerprints != reference.sample_fingerprints
            or prediction.request_spec != reference.request_spec
            or prediction.prompt_version != reference.prompt_version):
        raise RuntimeError("checkpoint variant prediction identity drift")
    if (len(prediction.criteria) != 1
            or prediction.criteria[0].name != item["criterion_name"]
            or prediction.criteria[0].description != item["description"]):
        raise RuntimeError("checkpoint variant criterion identity drift")


def _checkpoint_variant_rubric(output: Path, item: Mapping[str, Any]) -> StructuredRubric:
    for epoch in item["epochs"]:
        rubric = _checkpoint_rubric(_target(output), int(epoch))
        node = rubric.get_node(item["node_id"])
        if (node.criterion.name == item["criterion_name"]
                and node.criterion.description == item["description"]):
            return StructuredRubric({node.node_id: node}, (), (node.node_id,))
    raise RuntimeError("checkpoint variant does not match any frozen epoch")


def _variant_path(target: Path, item: Mapping[str, Any]) -> Path:
    # The parent Phase16 output path is already long on Windows.  Keep both
    # the shard directory and artifact name short because _generate_pairwise
    # appends cache/<label>/node/<hash>.json below this location.
    return target / "v" / item["description_sha256"][:12] / "p.json"


def checkpoint_heldout_smoke(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_checkpoint_manifest(config, output)
    target = _checkpoint_target(output)
    audit = target / "offline_audit.json"
    if not audit.is_file() or not all(load_json(audit).get("checks", {}).values()):
        raise RuntimeError("run prompt-v2-evolution-checkpoint-heldout-audit first")
    rows = load_jsonl_dataset(base._path(config["heldout_dataset"]), expected_count=500)
    reference = PairwisePredictionOutput.load_json(
        Path(manifest["final_combined_prediction_path"]))
    item = manifest["variants"][0]
    rubric = _checkpoint_variant_rubric(output, item)
    smoke_dir = target / "smoke"
    prediction, _, _ = base._generate_pairwise(
        _v2_config(config), smoke_dir, rubric, rows[:5], "checkpoint_smoke",
        execution_backend_pool=config["backend_pool"],
        request_backend_id=reference.request_spec.backend_id,
        request_level_progress=True)
    if (len(prediction.criteria) != 1
            or prediction.criteria[0].name != item["criterion_name"]
            or prediction.criteria[0].description != item["description"]
            or prediction.prompt_version != reference.prompt_version):
        raise RuntimeError("checkpoint heldout smoke prediction identity drift")
    value = {"schema_version": "1.0.0", "status": "passed",
             "sample_count": 5, "criterion_name": item["criterion_name"],
             "description_sha256": item["description_sha256"],
             "heldout_selection_forbidden": True}
    _write(smoke_dir / "report.json", value)
    print(json.dumps(value, indent=2))


def checkpoint_heldout_run(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_checkpoint_manifest(config, output)
    target = _checkpoint_target(output)
    smoke = target / "smoke" / "report.json"
    if not smoke.is_file() or load_json(smoke).get("status") != "passed":
        raise RuntimeError("run prompt-v2-evolution-checkpoint-heldout-smoke first")
    rows = load_jsonl_dataset(base._path(config["heldout_dataset"]), expected_count=500)
    reference = PairwisePredictionOutput.load_json(
        Path(manifest["final_combined_prediction_path"]))
    generated = 0
    for index, item in enumerate(manifest["variants"], start=1):
        path = _variant_path(target, item)
        if path.is_file():
            _validate_variant_prediction(
                PairwisePredictionOutput.load_json(path), item, reference)
            continue
        rubric = _checkpoint_variant_rubric(output, item)
        work = path.parent
        prediction, _, _ = base._generate_pairwise(
            _v2_config(config), work, rubric, rows,
            f"v{index:02d}",
            execution_backend_pool=config["backend_pool"],
            request_backend_id=reference.request_spec.backend_id,
            request_level_progress=True)
        _validate_variant_prediction(prediction, item, reference)
        prediction.save_json(path)
        generated += 1
    value = {"schema_version": "1.0.0", "status": "passed",
             "variant_count": len(manifest["variants"]),
             "generated_variant_count": generated,
             "reused_variant_count": len(manifest["variants"]) - generated,
             "new_pairwise_request_count": generated * len(rows),
             "heldout_selection_forbidden": True}
    _write(target / "run_report.json", value)
    print(json.dumps(value, indent=2))


def _assemble_checkpoint_prediction(
    output: Path, epoch: int, reference: PairwisePredictionOutput,
    variants: Mapping[tuple[str, str], PairwisePredictionOutput],
) -> PairwisePredictionOutput:
    rubric = _checkpoint_rubric(_target(output), epoch)
    reference_by_name = {item.name: item for item in reference.criteria}
    reference_outputs = reference.node_outputs
    criteria = []
    rows = []
    selected: list[tuple[str, PairwisePredictionOutput]] = []
    for node_id in rubric.preorder_node_ids():
        node = rubric.get_node(node_id)
        key = (node_id, _description_sha256(node.criterion.description))
        prediction = variants.get(key, reference)
        if node.criterion.name not in {item.name for item in prediction.criteria}:
            raise RuntimeError("checkpoint criterion output is missing")
        selected.append((node.criterion.name, prediction))
        criteria.append(StructuredCriterionSnapshot(
            node.criterion.name, node.criterion.description))
    for sample_index in range(len(reference.sample_ids)):
        row = {}
        for name, prediction in selected:
            row[name] = prediction.node_outputs[sample_index][name]
        rows.append(row)
    # ``node_outputs`` stores PairwiseVoteOutput objects, whereas the flat
    # aggregation helper intentionally accepts only their Vote values.
    # Passing the wrappers through worked nowhere in the executor and caused
    # report-only reconstruction to fail after all heldout requests completed.
    answers = tuple(
        base.aggregate_flat_votes(output.vote for output in row.values())
        for row in rows
    )
    return PairwisePredictionOutput(
        reference.sample_ids, reference.sample_fingerprints, tuple(criteria),
        tuple(rows), answers, reference.request_spec,
        semantics_version=reference.semantics_version,
        schema_version=reference.schema_version,
        prompt_version=reference.prompt_version,
        parser_version=reference.parser_version)


def _checkpoint_root_metrics(execution: Any, rubric: StructuredRubric,
                             rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {}
    for root_id in rubric.root_ids:
        votes = []
        for trace in execution.traces:
            matches = [item for item in trace.roots if item.root_id == root_id]
            if len(matches) != 1 or matches[0].subtree_vote is None:
                raise RuntimeError(f"incomplete checkpoint root trace: {root_id}")
            votes.append(matches[0].subtree_vote)
        values[root_id] = base._heldout_vote_metrics(votes, rows)
    return values


def checkpoint_heldout_report(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_checkpoint_manifest(config, output)
    target = _checkpoint_target(output)
    if not (target / "run_report.json").is_file():
        raise RuntimeError("run prompt-v2-evolution-checkpoint-heldout-run first")
    rows = load_jsonl_dataset(base._path(config["heldout_dataset"]), expected_count=500)
    reference = PairwisePredictionOutput.load_json(
        Path(manifest["final_combined_prediction_path"]))
    variants = {}
    for item in manifest["variants"]:
        prediction = PairwisePredictionOutput.load_json(_variant_path(target, item))
        _validate_variant_prediction(prediction, item, reference)
        variants[(item["node_id"], item["description_sha256"])] = prediction
    checkpoints = {}
    answers_by_epoch = {}
    for entry in manifest["checkpoints"]:
        epoch = int(entry["epoch"])
        rubric = _checkpoint_rubric(_target(output), epoch)
        prediction = _assemble_checkpoint_prediction(output, epoch, reference, variants)
        execution, answers = execute_offline_m1(rubric, prediction, rows)
        checkpoint_dir = target / "checkpoints" / f"epoch_{epoch:02d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prediction.save_json(checkpoint_dir / "combined_pairwise.json")
        execution.save_json(checkpoint_dir / "m1_execution.json")
        checkpoints[f"epoch_{epoch:02d}"] = {
            "rubric_sha256": rubric.rubric_sha256,
            "node_count": len(rubric.nodes),
            "m1": base._heldout_vote_metrics(answers, rows),
            "root_subtree_metrics": _checkpoint_root_metrics(execution, rubric, rows),
        }
        answers_by_epoch[epoch] = answers
    for epoch in CHECKPOINT_EPOCHS:
        entry = checkpoints[f"epoch_{epoch:02d}"]
        entry["vs_epoch_01"] = base._paired_heldout_comparison(
            answers_by_epoch[1], answers_by_epoch[epoch], rows)
        entry["vs_epoch_05"] = base._paired_heldout_comparison(
            answers_by_epoch[5], answers_by_epoch[epoch], rows)
    value = {
        "schema_version": "1.0.0",
        "experiment": manifest["experiment"],
        "exploratory": True,
        "checkpoints": checkpoints,
        "new_pairwise_request_count": manifest["new_pairwise_request_count"],
        "selection_after_heldout_forbidden": True,
    }
    _write(target / "final_report.json", value)
    lines = ["# Phase16 checkpoint heldout-500 diagnostic", "",
             "Exploratory post-hoc trajectory diagnostic; no checkpoint is selected.", "",
             "| Epoch | Nodes | M1 ACC | Coverage | Correct |", "|---|---:|---:|---:|---:|"]
    for epoch in CHECKPOINT_EPOCHS:
        item = checkpoints[f"epoch_{epoch:02d}"]
        metric = item["m1"]
        lines.append(
            f"| {epoch} | {item['node_count']} | {metric['accuracy']:.4f} | "
            f"{metric['coverage']:.4f} | {metric['correct_count']} |")
    (target / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        label: item["m1"]["accuracy"] for label, item in checkpoints.items()
    }, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {
        "prompt-v2-evolution-freeze": freeze,
        "prompt-v2-evolution-audit": audit,
        "prompt-v2-evolution-smoke": smoke,
        "prompt-v2-evolution-run": run,
        "prompt-v2-evolution-report": report,
        "prompt-v2-evolution-heldout": heldout,
        "prompt-v2-evolution-final-report": final_report,
        "prompt-v2-evolution-checkpoint-heldout-freeze": checkpoint_heldout_freeze,
        "prompt-v2-evolution-checkpoint-heldout-audit": checkpoint_heldout_audit,
        "prompt-v2-evolution-checkpoint-heldout-smoke": checkpoint_heldout_smoke,
        "prompt-v2-evolution-checkpoint-heldout-run": checkpoint_heldout_run,
        "prompt-v2-evolution-checkpoint-heldout-report": checkpoint_heldout_report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported Prompt v2 evolution stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

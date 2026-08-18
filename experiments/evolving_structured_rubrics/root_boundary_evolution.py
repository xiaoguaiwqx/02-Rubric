"""Phase 15 Root-boundary pre-Refine followed by frozen Split+Refine."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    BackendPoolSpec,
    DualWorkerRequestSpec,
    EvolutionContext,
    PairwisePredictionOutput,
    RubricFeedback,
    StructuredRubric,
    detect_specialize_trigger,
    execute_offline_m1,
    project_pairwise_prediction,
)
from critiq.structured.judgement import Vote

from . import refine_evolution as refine
from . import run_rubric_evolution as base
from . import split_evolution as split
from .experiment_utils import canonical_sha256, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase15_root_boundary_pre_refine_split_refine_v3"
EVOLUTION_SUBDIR = "evolution"
PAIRWISE_ENDPOINTS = ("vllm-8000", "vllm-8001")
PROTOCOL_V3 = {
    "protocol_version": "root-boundary-pre-refine-split-refine-v3",
    "source": "phase5_initial_five_roots",
    "root_pre_refine_candidate_scope": "all_initial_roots_forced",
    "root_pre_refine_min_epochs": 1,
    "root_pre_refine_max_epochs": 3,
    "accepted_root_locked": True,
    "retry_rejected_roots": True,
    "synchronous_epoch_commit": True,
    "rubric_memory_mode": "global_rubric_v1",
    "downstream_protocol": "five-root-locked-split-role-aware-refine-v1",
    "split_min_epochs": 3,
    "split_max_epochs": 5,
    "worker_pool": "configured_available_slot_pool",
    "worker_endpoints": list(PAIRWISE_ENDPOINTS),
    "worker_max_tokens": 2048,
    "initial_baseline_policy": "regenerate_five_roots_under_capped_identity",
    "gate_worker": False,
    "heldout_access": "final_stage_only",
    "exploratory": True,
}

EVOLUTION_PROTOCOL = split.EvolutionProtocol(
    f"{EXPERIMENT_DIR}/{EVOLUTION_SUBDIR}",
    "root-pre-refine-split-refine",
    rubric_memory_mode="global_rubric_v1",
    control_experiment_dir=None,
    read_only_control_signatures=False,
    allow_configured_endpoint_pool=True,
)


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _root_target(output: Path) -> Path:
    return _target(output) / "root_pre_refine"


def _evolution_target(output: Path) -> Path:
    return _target(output) / EVOLUTION_SUBDIR


def _write(path: Path, value: Any) -> None:
    refine._write(path, value)


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    refine._config(config)
    value = config.get("root_boundary_pre_refine_experiment")
    if value != PROTOCOL_V3:
        raise ValueError(
            "root_boundary_pre_refine_experiment must equal the frozen v3 protocol")
    # The downstream operator is intentionally identical to Phase 10.
    phase10 = config.get("five_root_locked_split_refine_experiment")
    if phase10 != refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_V1:
        raise ValueError("Phase 15 requires the frozen Phase 10 integration protocol")
    split.validate_policy(config, EVOLUTION_PROTOCOL)
    return dict(value)


def _pairwise_pool(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen two-endpoint available-slot execution pool."""
    spec = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoint_ids = tuple(item.endpoint_id for item in spec.endpoints)
    if len(endpoint_ids) != 2 or set(endpoint_ids) != set(PAIRWISE_ENDPOINTS):
        raise RuntimeError(
            "Phase 15 v2 requires exactly vllm-8000 and vllm-8001")
    return spec.to_dict()


def _operator_selection_config(
    config: Mapping[str, Any], output: Path,
) -> dict[str, Any]:
    """Use Phase-10 decoding plus the frozen 2048-token safety cap."""
    value = _source_phase5_config(config, output)
    value["worker_request_kwargs"]["max_tokens"] = PROTOCOL_V3["worker_max_tokens"]
    return value


def _source_phase5_config(
    config: Mapping[str, Any], output: Path,
) -> dict[str, Any]:
    """Replay the original Phase-5 identity solely to load frozen inputs."""
    phase5_manifest = load_json(output / "frozen_manifest.json")
    frozen = DualWorkerRequestSpec.from_dict(
        phase5_manifest["pairwise_request_spec"])
    value = dict(config)
    value["worker_request_kwargs"] = dict(frozen.decoding_config)
    return value


def _root_epoch(target: Path, epoch: int) -> Path:
    return target / "epochs" / f"epoch_{epoch:02d}"


def _decisive(output: Any) -> bool:
    return bool(output.parse_ok and output.answer_valid
                and output.vote in {Vote.A, Vote.B})


def _root_activation_diagnostic(
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prediction = project_pairwise_prediction(prediction, rubric)
    roots = tuple(rubric.root_ids)
    names = {root: rubric.get_node(root).criterion.name for root in roots}
    root_metrics = {}
    active_sets: dict[str, set[int]] = {}
    for root in roots:
        outputs = [row[names[root]] for row in prediction.node_outputs]
        active = {index for index, item in enumerate(outputs) if _decisive(item)}
        correct = sum(outputs[index].vote.value == str(rows[index]["answer"])
                      for index in active)
        active_sets[root] = active
        root_metrics[root] = {
            "criterion_name": names[root],
            "support": len(active),
            "correct": correct,
            "wrong": len(active) - correct,
            "accuracy": correct / len(active) if active else 0.0,
            "coverage": len(active) / len(rows),
        }
    pairs = []
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            joint = active_sets[left] & active_sets[right]
            union = active_sets[left] | active_sets[right]
            conflict = 0
            for row_index in joint:
                left_vote = prediction.node_outputs[row_index][names[left]].vote
                right_vote = prediction.node_outputs[row_index][names[right]].vote
                conflict += left_vote is not right_vote
            pairs.append({
                "left_root_id": left,
                "right_root_id": right,
                "joint_decisive": len(joint),
                "activation_jaccard": len(joint) / len(union) if union else 0.0,
                "conflict_count": conflict,
                "conflict_rate": conflict / len(joint) if joint else 0.0,
            })
    active_counts = [sum(index in active_sets[root] for root in roots)
                     for index in range(len(rows))]
    conflict_samples = 0
    for index in range(len(rows)):
        votes = {prediction.node_outputs[index][names[root]].vote
                 for root in roots if index in active_sets[root]}
        conflict_samples += Vote.A in votes and Vote.B in votes
    _, answers = execute_offline_m1(rubric, prediction, rows)
    return {
        "schema_version": "1.0.0",
        "root_metrics": root_metrics,
        "mean_active_roots": sum(active_counts) / len(active_counts),
        "all_five_active_count": sum(value == len(roots) for value in active_counts),
        "zero_active_count": sum(value == 0 for value in active_counts),
        "conflict_sample_count": conflict_samples,
        "pairwise": pairs,
        "m1": base._metrics(answers, rows),
    }


def _root_change_diagnostic(
    initial_rubric: StructuredRubric,
    initial_prediction: PairwisePredictionOutput,
    final_rubric: StructuredRubric,
    final_prediction: PairwisePredictionOutput,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root_rows = []
    for root_id in initial_rubric.root_ids:
        old_name = initial_rubric.get_node(root_id).criterion.name
        new_name = final_rubric.get_node(root_id).criterion.name
        if old_name != new_name:
            raise RuntimeError("Root Refine changed criterion name")
        suppressed_correct = suppressed_wrong = expanded = 0
        retained = retained_correct = 0
        transitions: dict[str, int] = {}
        corrected = harmed = 0
        for row, old_row, new_row in zip(
                rows, initial_prediction.node_outputs, final_prediction.node_outputs):
            old = old_row[old_name]
            new = new_row[new_name]
            key = f"{old.vote.value}->{new.vote.value}"
            transitions[key] = transitions.get(key, 0) + 1
            old_active = _decisive(old)
            new_active = _decisive(new)
            gold = str(row["answer"])
            if old_active and not new_active:
                if old.vote.value == gold:
                    suppressed_correct += 1
                else:
                    suppressed_wrong += 1
            if not old_active and new_active:
                expanded += 1
            if old_active and new_active:
                retained += 1
                retained_correct += new.vote.value == gold
            old_correct = old_active and old.vote.value == gold
            new_correct = new_active and new.vote.value == gold
            corrected += (not old_correct and new_correct)
            harmed += (old_correct and not new_correct)
        root_rows.append({
            "root_id": root_id,
            "criterion_name": old_name,
            "description_changed": (
                initial_rubric.get_node(root_id).criterion.description
                != final_rubric.get_node(root_id).criterion.description),
            "suppressed_old_correct": suppressed_correct,
            "suppressed_old_wrong": suppressed_wrong,
            "suppression_precision": (
                suppressed_wrong / (suppressed_correct + suppressed_wrong)
                if suppressed_correct + suppressed_wrong else None),
            "expanded_from_none": expanded,
            "retained_support": retained,
            "retained_accuracy": retained_correct / retained if retained else 0.0,
            "corrected": corrected,
            "harmed": harmed,
            "net_corrected": corrected - harmed,
            "transitions": dict(sorted(transitions.items())),
        })
    return {"schema_version": "1.0.0", "roots": root_rows}


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    refine._validate_output(output)
    pairwise_pool = _pairwise_pool(config)
    target = _target(output)
    manifest_path = target / "frozen_manifest.json"
    if manifest_path.exists():
        print("root-pre-refine-freeze already completed")
        return
    source_config = _source_phase5_config(config, output)
    scientific_config = _operator_selection_config(config, output)
    phase5, rubric, rows, prediction, feedback = base._phase6_inputs(
        source_config, output)
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 5:
        raise RuntimeError("Root pre-Refine requires exactly the five initial roots")
    if feedback.rubric_sha256 != rubric.rubric_sha256:
        raise RuntimeError("initial feedback/rubric mismatch")
    manager, manager_profile = refine._manager(config)
    expected = base._expected_pairwise_request_spec(scientific_config, rows)
    memory = split.rubric_memory_snapshot(rubric)
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "settings": settings,
        "phase5_manifest_sha256": canonical_sha256(phase5),
        "discovery_dataset_sha256": phase5["discovery_dataset_sha256"],
        "heldout_dataset_sha256": config["heldout_dataset_sha256"],
        "initial_rubric_sha256": rubric.rubric_sha256,
        "initial_root_ids": list(rubric.root_ids),
        "source_prediction_sha256": canonical_sha256(prediction.to_dict()),
        "source_pairwise_request_spec": prediction.request_spec.to_dict(),
        "pairwise_request_spec": expected.to_dict(),
        "pairwise_execution_pool": pairwise_pool,
        "manager_profile": manager_profile,
        "manager_request_specs": {
            key: value.to_dict() for key, value in manager.request_specs().items()},
        "root_pre_refine_prompt_contract": {
            "mode": "evidence_extension_v1",
            "required_sections": [
                "Criterion focus", "Applicable only when",
                "Not applicable when", "Decision rule"],
            "cross_root_boundary_instruction": True,
        },
        "initial_rubric_memory_sha256": canonical_sha256(memory),
        "phase10_control": refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR,
        "heldout_access": "forbidden_until_final_stage",
    }
    _write(manifest_path, manifest)
    _write(target / "stage_status.json", {"freeze": {"status": "passed"}})
    print(json.dumps({"target": str(target), "roots": list(rubric.root_ids),
                      "heldout_accessed": False}, indent=2))


def baseline(config: Mapping[str, Any], output: Path) -> None:
    """Regenerate the five-root discovery baseline under the capped identity."""
    _settings(config)
    refine._validate_output(output)
    target = _target(output)
    manifest = load_json(target / "frozen_manifest.json")
    status = load_json(target / "stage_status.json")
    if status.get("baseline", {}).get("status") == "passed":
        print("root-pre-refine-baseline already completed")
        return
    source_config = _source_phase5_config(config, output)
    _, rubric, rows, source_prediction, _ = base._phase6_inputs(
        source_config, output)
    if canonical_sha256(source_prediction.to_dict()) != manifest["source_prediction_sha256"]:
        raise RuntimeError("Root pre-Refine source prediction drift")
    scientific_config = _operator_selection_config(config, output)
    expected = DualWorkerRequestSpec.from_dict(manifest["pairwise_request_spec"])
    prediction, _, valid_rate = base._generate_pairwise(
        scientific_config, target / "baseline", rubric, rows,
        "initial_five_roots", execution_backend_pool=_pairwise_pool(config),
        request_backend_id=expected.backend_id, request_level_progress=True)
    if prediction.request_spec != expected:
        raise RuntimeError("Root pre-Refine capped baseline identity drift")
    root_target = _root_target(output)
    votes, _ = split._snapshot(
        _root_epoch(root_target, 0), rubric, prediction, rows, manifest)
    split.freeze_rubric_memory(_root_epoch(root_target, 0), rubric)
    _write(_root_epoch(root_target, 0) / "summary.json", {
        "epoch": 0, "m1": base._metrics(votes, rows),
        "root_diagnostic": _root_activation_diagnostic(rubric, prediction, rows),
        "valid_rate": valid_rate,
    })
    root_states = {
        root_id: {"status": "pending", "attempt_count": 0,
                  "accepted_epoch": None, "last_decision": None}
        for root_id in rubric.root_ids}
    _write(root_target / "evolution_history.json", {
        "schema_version": "1.0.0", "current_epoch": 0,
        "completed": False, "stop_reason": None,
        "root_states": root_states, "refine_states": {},
        "refine_attempts": [],
    })
    status["baseline"] = {"status": "passed", "details": {
        "request_count": len(rows) * len(rubric.nodes),
        "valid_rate": valid_rate,
        "max_tokens": PROTOCOL_V3["worker_max_tokens"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["baseline"]["details"], indent=2))


def audit(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    refine._validate_output(output)
    target = _target(output)
    manifest = load_json(target / "frozen_manifest.json")
    if load_json(target / "stage_status.json").get(
            "baseline", {}).get("status") != "passed":
        raise RuntimeError("run root-pre-refine-baseline first")
    epoch0 = _root_epoch(_root_target(output), 0)
    rubric = StructuredRubric.load_json(epoch0 / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(epoch0 / "discovery_pairwise.json")
    rows = refine._rows(config, "discovery")
    if manifest["pairwise_execution_pool"] != _pairwise_pool(config):
        raise RuntimeError("Root pre-Refine Pairwise execution pool drift")
    if rubric.rubric_sha256 != manifest["initial_rubric_sha256"]:
        raise RuntimeError("Root pre-Refine frozen rubric drift")
    if prediction.request_spec != DualWorkerRequestSpec.from_dict(
            manifest["pairwise_request_spec"]):
        raise RuntimeError("Root pre-Refine frozen prediction identity drift")
    automatic = {}
    feedback = RubricFeedback.from_dict(load_json(epoch0 / "feedback.json"))
    context = EvolutionContext(rubric, feedback)
    for root_id in rubric.root_ids:
        automatic[root_id] = refine.detect_refine_trigger(
            context, root_id, config["evolution_policy"]["trigger_thresholds"]
        ).to_dict()
    value = {
        "schema_version": "1.0.0", "offline_only": True,
        "settings": settings, "forced_root_ids": list(rubric.root_ids),
        "automatic_triggers_for_diagnosis": automatic,
        "baseline": _root_activation_diagnostic(rubric, prediction, rows),
        "heldout_accessed": False,
    }
    _write(target / "offline_audit.json", value)
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": {
        "forced_root_count": len(rubric.root_ids), "heldout_accessed": False}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["audit"]["details"], indent=2))


def smoke(config: Mapping[str, Any], output: Path) -> None:
    """Run one isolated forced Root Refine without mutating formal history."""
    _settings(config)
    refine._validate_output(output)
    target = _target(output)
    if not (target / "offline_audit.json").exists():
        raise RuntimeError("run root-pre-refine-freeze and audit first")
    smoke_dir = target / "smoke"
    summary_path = smoke_dir / "summary.json"
    if summary_path.exists():
        print("root-pre-refine-smoke already completed")
        return
    manifest = load_json(target / "frozen_manifest.json")
    epoch0 = _root_epoch(_root_target(output), 0)
    rubric = StructuredRubric.load_json(epoch0 / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(
        epoch0 / "discovery_pairwise.json")
    feedback = RubricFeedback.from_dict(load_json(epoch0 / "feedback.json"))
    rows = refine._rows(config, "discovery")
    memory, memory_hash = split.freeze_rubric_memory(smoke_dir, rubric)
    manager, _ = refine._manager(config)
    if ({key: value.to_dict() for key, value in manager.request_specs().items()}
            != manifest["manager_request_specs"]):
        raise RuntimeError("Root pre-Refine smoke Manager identity drift")
    node_id = rubric.root_ids[0]
    scientific_config = _operator_selection_config(config, output)
    history = {"refine_states": {}, "refine_attempts": []}
    try:
        result = refine._prepare_refine_attempt(
            config=scientific_config, epoch_dir=smoke_dir, node_id=node_id,
            attempt_no=1, rubric=rubric, prediction=prediction,
            feedback=feedback, rows=rows, history=history,
            manager=manager,
            pool=_pairwise_pool(config),
            rubric_memory=memory, rubric_memory_sha256=memory_hash,
            trigger_mode="uniform_v1", forced=True,
            evidence_extension=_boundary_evidence(rubric, node_id))
    except (refine.RefineTransportPause,
            refine.RefineAttributionInvalid) as exc:
        status = load_json(target / "stage_status.json")
        status["smoke"] = {"status": "paused", "details": {
            "node_id": node_id, "message": str(exc)}}
        _write(target / "stage_status.json", status)
        return
    evaluation = result.get("evaluation")
    value = {
        "schema_version": "1.0.0", "diagnostic_only": True,
        "formal_history_mutated": False, "node_id": node_id,
        "decision": result["decision"],
        "pipeline_valid": result["decision"] in {
            "accepted", "competition_rejected"},
        "node_evaluation": None if evaluation is None else evaluation.to_dict(),
        "rubric_memory_sha256": memory_hash,
    }
    _write(summary_path, value)
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": (
        "passed" if value["pipeline_valid"] else "failed"),
        "details": {"node_id": node_id, "decision": result["decision"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["smoke"]["details"], indent=2))


def _boundary_evidence(rubric: StructuredRubric, node_id: str) -> dict[str, Any]:
    peers = [{
        "node_id": peer_id,
        "criterion_name": rubric.get_node(peer_id).criterion.name,
        "description": rubric.get_node(peer_id).criterion.description,
    } for peer_id in rubric.root_ids if peer_id != node_id]
    return {
        "root_boundary_task": {
            "objective": (
                "Rewrite this root as a distinct soft-routing expert. Preserve its "
                "core preference semantics while making its applicability boundary "
                "non-redundant with the other four roots."),
            "requirements": [
                "Use Applicable only when to state positive activation evidence.",
                "Use Not applicable when to exclude cases owned by peer roots.",
                "Return None when this root has no reliable criterion-specific difference.",
                "Do not lower coverage merely to avoid hard examples.",
                "Do not vote using overall answer quality or another criterion."],
            "peer_roots": peers,
        }
    }


def run(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    refine._validate_output(output)
    target = _target(output)
    if not (target / "offline_audit.json").exists():
        raise RuntimeError("run root-pre-refine-freeze and audit first")
    if load_json(target / "stage_status.json").get(
            "smoke", {}).get("status") != "passed":
        raise RuntimeError("run root-pre-refine-smoke successfully first")
    manifest = load_json(target / "frozen_manifest.json")
    manager, _ = refine._manager(config)
    runtime_specs = {key: value.to_dict()
                     for key, value in manager.request_specs().items()}
    if runtime_specs != manifest["manager_request_specs"]:
        raise RuntimeError("Root pre-Refine Manager request identity drift")
    root_target = _root_target(output)
    history = load_json(root_target / "evolution_history.json")
    if history.get("completed"):
        print("root-pre-refine-run already completed")
        return
    rows = refine._rows(config, "discovery")
    scientific_config = _operator_selection_config(config, output)
    pool = _pairwise_pool(config)
    max_epochs = settings["root_pre_refine_max_epochs"]
    for epoch_no in range(history["current_epoch"] + 1, max_epochs + 1):
        started_epoch = time.monotonic()
        previous = _root_epoch(root_target, epoch_no - 1)
        epoch_dir = _root_epoch(root_target, epoch_no)
        rubric = StructuredRubric.load_json(previous / "rubric_committed.json")
        prediction = PairwisePredictionOutput.load_json(
            previous / "discovery_pairwise.json")
        feedback = RubricFeedback.from_dict(load_json(previous / "feedback.json"))
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        scheduled = tuple(root_id for root_id in rubric.root_ids
                          if history["root_states"][root_id]["status"] in {
                              "pending", "retryable"})
        print(f"root-pre-refine epoch={epoch_no} scheduled={list(scheduled)}",
              flush=True)
        results: dict[str, dict[str, Any]] = {}
        started: dict[str, float] = {}
        try:
            for root_id in scheduled:
                started[root_id] = time.monotonic()
                attempt_no = history["root_states"][root_id]["attempt_count"] + 1
                results[root_id] = refine._prepare_refine_attempt(
                    config=scientific_config, epoch_dir=epoch_dir, node_id=root_id,
                    attempt_no=attempt_no, rubric=rubric,
                    prediction=prediction, feedback=feedback, rows=rows,
                    history=history, manager=manager, pool=pool,
                    rubric_memory=memory, rubric_memory_sha256=memory_hash,
                    trigger_mode="uniform_v1", forced=True,
                    evidence_extension=_boundary_evidence(rubric, root_id))
        except (refine.RefineTransportPause,
                refine.RefineAttributionInvalid) as exc:
            status = load_json(target / "stage_status.json")
            status["root_pre_refine_run"] = {
                "status": ("paused" if isinstance(
                    exc, refine.RefineTransportPause) else "failed"),
                "details": {"epoch": epoch_no, "message": str(exc),
                            "history_appended": False}}
            _write(target / "stage_status.json", status)
            return
        accepted = {root_id: result["candidate"]
                    for root_id, result in results.items()
                    if result["decision"] == "accepted"}
        committed = refine._merge_epoch_rubric(rubric, {}, accepted)
        committed_prediction = refine._merge_epoch_predictions(
            prediction, [], {
                root_id: results[root_id]["candidate_prediction"]
                for root_id in accepted}, committed)
        votes, _ = split._snapshot(
            epoch_dir, committed, committed_prediction, rows, manifest)
        records = []
        for root_id in scheduled:
            result = results[root_id]
            record = refine._record_refine_state(
                history, result, epoch_no,
                time.monotonic() - started[root_id])
            records.append(record)
            state = history["root_states"][root_id]
            state["attempt_count"] += 1
            state["last_decision"] = result["decision"]
            if result["decision"] == "accepted":
                state.update({"status": "accepted", "accepted_epoch": epoch_no})
            elif epoch_no >= max_epochs:
                state["status"] = "exhausted"
            else:
                state["status"] = "retryable"
        history["current_epoch"] = epoch_no
        pending = [root_id for root_id, state in history["root_states"].items()
                   if state["status"] in {"pending", "retryable"}]
        if not pending:
            history.update({"completed": True,
                            "stop_reason": "all_roots_accepted_or_exhausted"})
        elif epoch_no == max_epochs:
            for root_id in pending:
                history["root_states"][root_id]["status"] = "exhausted"
            history.update({"completed": True,
                            "stop_reason": "max_epochs_reached"})
        diagnostics = _root_activation_diagnostic(
            committed, committed_prediction, rows)
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no, "forced": True,
            "scheduled_root_ids": list(scheduled),
            "accepted_root_ids": sorted(accepted),
            "attempts": records,
            "pending_root_ids": pending,
            "m1": base._metrics(votes, rows),
            "root_diagnostic": diagnostics,
            "epoch_wall_seconds": time.monotonic() - started_epoch,
        })
        _write(root_target / "evolution_history.json", history)
        print(f"root-pre-refine epoch={epoch_no} "
              f"accepted={sorted(accepted)} m1_acc={diagnostics['m1']['accuracy']:.4f}",
              flush=True)
        if history["completed"]:
            break
    status = load_json(target / "stage_status.json")
    status["root_pre_refine_run"] = {
        "status": "passed" if history["completed"] else "partial",
        "details": {"current_epoch": history["current_epoch"],
                    "stop_reason": history["stop_reason"]}}
    _write(target / "stage_status.json", status)


def report(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    refine._validate_output(output)
    target = _target(output)
    root_target = _root_target(output)
    history = load_json(root_target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run root-pre-refine-run to completion first")
    rows = refine._rows(config, "discovery")
    initial_epoch = _root_epoch(root_target, 0)
    final_epoch = _root_epoch(root_target, history["current_epoch"])
    initial_rubric = StructuredRubric.load_json(initial_epoch / "rubric_committed.json")
    initial_prediction = PairwisePredictionOutput.load_json(
        initial_epoch / "discovery_pairwise.json")
    final_rubric = StructuredRubric.load_json(final_epoch / "rubric_committed.json")
    final_prediction = PairwisePredictionOutput.load_json(
        final_epoch / "discovery_pairwise.json")
    final_dir = root_target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_rubric.save_json(final_dir / "rubric.json")
    final_prediction.save_json(final_dir / "discovery_pairwise.json")
    feedback_path = final_epoch / "feedback.json"
    _write(final_dir / "feedback.json", load_json(feedback_path))
    initial_diagnostic = _root_activation_diagnostic(
        initial_rubric, initial_prediction, rows)
    final_diagnostic = _root_activation_diagnostic(
        final_rubric, final_prediction, rows)
    value = {
        "schema_version": "1.0.0",
        "experiment": "forced_root_boundary_pre_refine",
        "forced": True,
        "final_epoch": history["current_epoch"],
        "accepted_root_ids": sorted(
            root_id for root_id, state in history["root_states"].items()
            if state["status"] == "accepted"),
        "exhausted_root_ids": sorted(
            root_id for root_id, state in history["root_states"].items()
            if state["status"] == "exhausted"),
        "initial": initial_diagnostic,
        "final": final_diagnostic,
        "m1_accuracy_delta": (final_diagnostic["m1"]["accuracy"]
                              - initial_diagnostic["m1"]["accuracy"]),
        "mean_active_roots_delta": (final_diagnostic["mean_active_roots"]
                                    - initial_diagnostic["mean_active_roots"]),
        "conflict_sample_delta": (final_diagnostic["conflict_sample_count"]
                                  - initial_diagnostic["conflict_sample_count"]),
        "root_changes": _root_change_diagnostic(
            initial_rubric, initial_prediction, final_rubric,
            final_prediction, rows),
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "heldout_accessed": False,
    }
    _write(final_dir / "discovery_report.json", value)
    status = load_json(target / "stage_status.json")
    status["root_pre_refine_report"] = {"status": "passed", "details": {
        "accepted_root_count": len(value["accepted_root_ids"]),
        "m1_accuracy": final_diagnostic["m1"]["accuracy"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["root_pre_refine_report"]["details"], indent=2))


def _initialize_downstream(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    target = _target(output)
    evolution = _evolution_target(output)
    manifest_path = evolution / "frozen_manifest.json"
    if manifest_path.exists():
        return
    root_final = _root_target(output) / "final"
    root_report_path = root_final / "discovery_report.json"
    if not root_report_path.exists():
        raise RuntimeError("run root-pre-refine-report first")
    rubric = StructuredRubric.load_json(root_final / "rubric.json")
    prediction = PairwisePredictionOutput.load_json(
        root_final / "discovery_pairwise.json")
    feedback = RubricFeedback.from_dict(load_json(root_final / "feedback.json"))
    rows = refine._rows(config, "discovery")
    managers, profiles, specs, identities = split._managers(
        config, EVOLUTION_PROTOCOL)
    del managers
    refine_manager, refine_profile = refine._manager(config)
    _, retry_specs = refine._integrated_retry_specs(config)
    context = EvolutionContext(rubric, feedback)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    triggers = {root_id: detect_specialize_trigger(
        context, root_id, thresholds).to_dict() for root_id in rubric.root_ids}
    eligible = [root_id for root_id in rubric.root_ids
                if triggers[root_id]["triggered"]]
    manifest = {
        "schema_version": "1.0.0",
        "experiment": f"{EXPERIMENT_DIR}/{EVOLUTION_SUBDIR}",
        "policy": split.POLICY_V1,
        "manager_seed": 42,
        "source_root_pre_refine_report_sha256": file_sha256(root_report_path),
        "discovery_dataset_sha256": config["discovery_dataset_sha256"],
        "initial_rubric_sha256": rubric.rubric_sha256,
        "initial_root_ids": list(rubric.root_ids),
        "initial_triggers": triggers,
        "initial_eligible_root_ids": eligible,
        "initial_eligible_root_count": len(eligible),
        "pairwise_request_spec": prediction.request_spec.to_dict(),
        "pairwise_execution_pool": _pairwise_pool(config),
        "manager_profiles": profiles,
        "manager_request_specs": specs,
        "manager_endpoint_identities": identities,
        "rubric_memory_mode": "global_rubric_v1",
        "integration_settings": refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_V1,
        "refine_protocol": refine.REFINE_V1,
        "refine_manager_profile": refine_profile,
        "refine_manager_request_specs": {
            key: value.to_dict()
            for key, value in refine_manager.request_specs().items()},
        "locked_retry_manager_request_specs": retry_specs,
        "operator_schedule": (
            "synchronous_epoch_start_split_locked_retry_plus_role_aware_refine"),
        "partial_split_acceptance": False,
        "error_signature_policy": (
            "reuse_only_on_exact_refined_parent_identity_else_generate"),
        "heldout_access": "forbidden_until_heldout_stage",
        "heldout_protocol": {"exploratory": True,
                             "selection_after_heldout_forbidden": True},
    }
    _write(manifest_path, manifest)
    votes, _ = split._snapshot(
        split._epoch(evolution, 0), rubric, prediction, rows, manifest)
    _, memory_hash = split.freeze_rubric_memory(split._epoch(evolution, 0), rubric)
    manifest["epoch_00_rubric_memory_sha256"] = memory_hash
    _write(manifest_path, manifest)
    _write(split._epoch(evolution, 0) / "summary.json", {
        "epoch": 0, "m1": base._metrics(votes, rows),
        "triggers": triggers, "accepted_roots": []})
    states = {root_id: {
        "status": "eligible" if triggers[root_id]["triggered"] else "not_eligible",
        "attempt_count": 0, "accepted_epoch": None, "children": [],
        "locked_retry": None,
    } for root_id in rubric.root_ids}
    _write(evolution / "evolution_history.json", {
        "schema_version": "1.0.0", "current_epoch": 0,
        "completed": False, "stop_reason": None,
        "root_states": states, "attempts": [],
        "refine_states": {}, "refine_attempts": [],
    })
    _write(evolution / "offline_audit.json", {
        "schema_version": "1.0.0", "offline_only": True,
        "split_roots": eligible, "heldout_accessed": False})
    _write(evolution / "stage_status.json", {
        "freeze": {"status": "passed"}, "audit": {"status": "passed"}})
    status = load_json(target / "stage_status.json")
    status["downstream_freeze"] = {"status": "passed", "details": {
        "eligible_split_roots": eligible,
        "initial_rubric_sha256": rubric.rubric_sha256}}
    _write(target / "stage_status.json", status)


def split_refine_run(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    refine._validate_output(output)
    _initialize_downstream(config, output)
    target = _evolution_target(output)
    manifest = load_json(target / "frozen_manifest.json")
    history = load_json(target / "evolution_history.json")
    if history.get("completed"):
        print("root-pre-refine-split-refine-run already completed")
        return
    scientific_config = _operator_selection_config(config, output)
    refine._run_five_root_locked_split_refine_impl(
        scientific_config, target=target, manifest=manifest, history=history,
        settings=refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_V1,
        protocol=EVOLUTION_PROTOCOL,
        log_prefix="root-pre-refine integration",
        execution_backend_pool=_pairwise_pool(config))
    parent_status = load_json(_target(output) / "stage_status.json")
    child_status = load_json(target / "stage_status.json")
    parent_status["split_refine_run"] = child_status["run"]
    _write(_target(output) / "stage_status.json", parent_status)


def split_refine_report(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    refine._validate_output(output)
    target = _evolution_target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run root-pre-refine-split-refine-run to completion first")
    rows = refine._rows(config, "discovery")
    epoch0 = split._epoch(target, 0)
    final_epoch = split._epoch(target, history["current_epoch"])
    initial_rubric = StructuredRubric.load_json(epoch0 / "rubric_committed.json")
    initial_prediction = PairwisePredictionOutput.load_json(
        epoch0 / "discovery_pairwise.json")
    rubric = StructuredRubric.load_json(final_epoch / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(
        final_epoch / "discovery_pairwise.json")
    execution, answers = execute_offline_m1(rubric, prediction, rows)
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    execution.save_json(final_dir / "m1_execution.json")
    initial_diagnostic = _root_activation_diagnostic(
        initial_rubric, initial_prediction, rows)
    final_diagnostic = _root_activation_diagnostic(rubric, prediction, rows)
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "final_epoch": history["current_epoch"],
        "root_pre_refined_initial_m1": initial_diagnostic["m1"],
        "final_m1": base._metrics(answers, rows),
        "m1_accuracy_delta": (base._metrics(answers, rows)["accuracy"]
                              - initial_diagnostic["m1"]["accuracy"]),
        "initial_root_diagnostic": initial_diagnostic,
        "final_root_diagnostic": final_diagnostic,
        "final_rubric_sha256": rubric.rubric_sha256,
        "node_count_initial": len(initial_rubric.nodes),
        "node_count_final": len(rubric.nodes),
        "split_attempts": history["attempts"],
        "refine_attempts": history["refine_attempts"],
        "accepted_split_count": sum(
            item["decision"] == split.ACCEPTED for item in history["attempts"]),
        "accepted_refine_count": sum(
            item["decision"] == "accepted" for item in history["refine_attempts"]),
        "heldout_accessed": False,
    }
    _write(final_dir / "discovery_report.json", value)
    status = load_json(_target(output) / "stage_status.json")
    status["split_refine_report"] = {"status": "passed", "details": {
        "final_epoch": history["current_epoch"],
        "final_m1_accuracy": value["final_m1"]["accuracy"]}}
    _write(_target(output) / "stage_status.json", status)
    print(json.dumps(status["split_refine_report"]["details"], indent=2))


def _heldout_prediction(
    config: Mapping[str, Any],
    *,
    directory: Path,
    label: str,
    rubric: StructuredRubric,
    reusable: PairwisePredictionOutput,
    rows: Sequence[Mapping[str, Any]],
) -> PairwisePredictionOutput:
    reusable_by_name = {item.name: item.description for item in reusable.criteria}
    changed_ids = tuple(node_id for node_id in rubric.preorder_node_ids()
                        if reusable_by_name.get(
                            rubric.get_node(node_id).criterion.name)
                        != rubric.get_node(node_id).criterion.description)
    generated = None
    generated_path = directory / f"{label}_changed_predictions.json"
    if changed_ids:
        if generated_path.exists():
            generated = PairwisePredictionOutput.load_json(generated_path)
        else:
            changed = StructuredRubric(
                {node_id: rubric.get_node(node_id) for node_id in changed_ids},
                (), changed_ids)
            generated, _, _ = base._generate_pairwise(
                config, directory, changed, rows, f"{label}_changed",
                execution_backend_pool=_pairwise_pool(config),
                request_backend_id=reusable.request_spec.backend_id,
                request_level_progress=True)
            generated.save_json(generated_path)
    combined = refine._combine_heldout_predictions(reusable, generated, rubric)
    combined.save_json(directory / f"{label}_combined_pairwise.json")
    return combined


def heldout(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    refine._validate_output(output)
    target = _target(output)
    discovery_path = _evolution_target(output) / "final" / "discovery_report.json"
    if not discovery_path.exists():
        raise RuntimeError("run root-pre-refine-split-refine-report first")
    rows = refine._rows(config, "heldout")
    scientific_config = _operator_selection_config(config, output)
    initial_rubric = StructuredRubric.load_json(
        _root_epoch(_root_target(output), 0) / "rubric_committed.json")
    root_rubric = StructuredRubric.load_json(
        _root_target(output) / "final" / "rubric.json")
    final_rubric = StructuredRubric.load_json(
        _evolution_target(output) / "final" / "rubric.json")
    phase10 = output / refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR
    phase10_rubric = StructuredRubric.load_json(phase10 / "final" / "rubric.json")
    initial_prediction = PairwisePredictionOutput.load_json(
        output / "predictions" / "init_pairwise_p05_heldout500.json")
    phase10_prediction = PairwisePredictionOutput.load_json(
        phase10 / "heldout500" / "combined_pairwise.json")
    expected = base._expected_pairwise_request_spec(scientific_config, rows)
    for item in (initial_prediction, phase10_prediction):
        if not refine._same_pairwise_scientific_identity(item.request_spec, expected):
            raise RuntimeError("Phase 15 heldout Worker scientific identity drift")
    heldout_dir = target / "heldout500"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0.0", "exploratory": True,
        "initial_rubric_sha256": initial_rubric.rubric_sha256,
        "root_pre_refined_rubric_sha256": root_rubric.rubric_sha256,
        "phase10_control_rubric_sha256": phase10_rubric.rubric_sha256,
        "final_treatment_rubric_sha256": final_rubric.rubric_sha256,
        "discovery_report_sha256": file_sha256(discovery_path),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "pairwise_request_spec": expected.to_dict(),
        "selection_after_heldout_forbidden": True,
    }
    manifest_path = heldout_dir / "frozen_manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise RuntimeError("Phase 15 heldout manifest drift")
    _write(manifest_path, manifest)
    root_prediction = _heldout_prediction(
        scientific_config, directory=heldout_dir, label="root_pre_refined",
        rubric=root_rubric, reusable=initial_prediction, rows=rows)
    final_prediction = _heldout_prediction(
        scientific_config, directory=heldout_dir, label="final_treatment",
        rubric=final_rubric, reusable=phase10_prediction, rows=rows)
    systems = {
        "initial_five_roots": (initial_rubric, initial_prediction),
        "root_pre_refined_only": (root_rubric, root_prediction),
        "phase10_control": (phase10_rubric, phase10_prediction),
        "final_treatment": (final_rubric, final_prediction),
    }
    answers = {}
    metrics = {}
    for label, (rubric, prediction) in systems.items():
        projected = project_pairwise_prediction(prediction, rubric)
        _, values = execute_offline_m1(rubric, projected, rows)
        answers[label] = values
        metrics[label] = base._heldout_vote_metrics(values, rows)
    value = {
        "schema_version": "1.0.0", "exploratory": True,
        "systems": metrics,
        "root_pre_refined_vs_initial": base._paired_heldout_comparison(
            answers["initial_five_roots"], answers["root_pre_refined_only"], rows),
        "final_treatment_vs_phase10": base._paired_heldout_comparison(
            answers["phase10_control"], answers["final_treatment"], rows),
        "selection_after_heldout_forbidden": True,
    }
    _write(heldout_dir / "report.json", value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        "root_pre_refined_m1_accuracy": metrics["root_pre_refined_only"]["accuracy"],
        "final_treatment_m1_accuracy": metrics["final_treatment"]["accuracy"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def final_report(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    target = _target(output)
    root = load_json(_root_target(output) / "final" / "discovery_report.json")
    evolution = load_json(
        _evolution_target(output) / "final" / "discovery_report.json")
    heldout_report = load_json(target / "heldout500" / "report.json")
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "root_pre_refine_discovery": root,
        "split_refine_discovery": evolution,
        "heldout": heldout_report,
        "primary": {
            "phase10_control_heldout_accuracy": heldout_report[
                "systems"]["phase10_control"]["accuracy"],
            "root_pre_refined_heldout_accuracy": heldout_report[
                "systems"]["root_pre_refined_only"]["accuracy"],
            "final_treatment_heldout_accuracy": heldout_report[
                "systems"]["final_treatment"]["accuracy"],
            "net_corrected_vs_phase10": heldout_report[
                "final_treatment_vs_phase10"]["net_corrected"],
        },
    }
    _write(target / "final_report.json", value)
    lines = [
        "# Root Boundary Pre-Refine → Split+Refine v3", "",
        "Exploratory heldout comparison; no post-heldout selection.", "",
        f"- Root-pre-refine discovery M1: {root['initial']['m1']['accuracy']:.4f} "
        f"→ {root['final']['m1']['accuracy']:.4f}",
        f"- Final discovery M1: {evolution['final_m1']['accuracy']:.4f}",
        f"- Phase 10 heldout M1: "
        f"{value['primary']['phase10_control_heldout_accuracy']:.4f}",
        f"- Root-only heldout M1: "
        f"{value['primary']['root_pre_refined_heldout_accuracy']:.4f}",
        f"- Final treatment heldout M1: "
        f"{value['primary']['final_treatment_heldout_accuracy']:.4f}",
        f"- Net corrected vs Phase 10: "
        f"{value['primary']['net_corrected_vs_phase10']}",
    ]
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed"}
    _write(target / "stage_status.json", status)
    print(json.dumps(value["primary"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {
        "root-pre-refine-freeze": freeze,
        "root-pre-refine-baseline": baseline,
        "root-pre-refine-audit": audit,
        "root-pre-refine-smoke": smoke,
        "root-pre-refine-run": run,
        "root-pre-refine-report": report,
        "root-pre-refine-split-refine-run": split_refine_run,
        "root-pre-refine-split-refine-report": split_refine_report,
        "root-pre-refine-heldout": heldout,
        "root-pre-refine-final-report": final_report,
    }
    if stage not in actions:
        raise ValueError(f"unknown Root pre-Refine stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

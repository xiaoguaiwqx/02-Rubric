"""Refine v1 forced smoke and Split+Refine evolution experiment stages."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    DualWorkerRequestSpec,
    EvolutionContext,
    EvolutionDecision,
    PairwisePredictionOutput,
    RefineCandidate,
    RefineEvaluation,
    RefineManager,
    RefineManagerFailure,
    RefineProposal,
    RubricFeedback,
    StructuredCriterionSnapshot,
    StructuredRubric,
    apply_rubric_patch,
    assemble_refined_pairwise_prediction,
    build_refine_candidate,
    detect_refine_trigger,
    evaluate_refine_candidate,
    execute_offline_m1,
    extract_rubric_feedback,
    project_pairwise_prediction,
)
from critiq.structured.aggregation import aggregate_flat_votes
from critiq.structured.judgement import Vote

from . import run_rubric_evolution as base
from . import split_evolution as split
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, load_jsonl_dataset
from .rubric_factory import file_sha256
from .split_evolution import rubric_memory_snapshot


EXPERIMENT_DIR = "phase7_refine_operator_v1"
FULL_EXPERIMENT_DIR = "phase7_split_refine_evolution_v1"
ROLE_EXPERIMENT_DIR = "phase8_refine_role_aware_v2"
ROLE_CHECKPOINT_HELDOUT_DIR = "heldout_checkpoint_diagnostic"
ROLE_CHECKPOINT_HELDOUT_V2_DIR = "heldout_checkpoint_diagnostic_v2"
VISUAL_SPLIT_REFINE_EXPERIMENT_DIR = "phase9_visual_split_refine_local_v1"
FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR = (
    "phase10_five_root_locked_split_refine_v1"
)
VISUAL_SPLIT_REFINE_SOURCE_DIR = "phase8_visual_split_retry_locked_v2"
VISUAL_SPLIT_REFINE_ROOT_ID = "init_02_visual_grounding_and_details"
VISUAL_SPLIT_REFINE_ENDPOINT = "vllm-8000"
SOURCE_EXPERIMENT_DIR = "phase6_split_only_evolution_global_memory_v1"
TARGET_NODE_ID = (
    "init_01_completeness_and_coverage__"
    "verified_existence_over_hallucinated_volume__dbf3c657ab"
)
TARGET_CRITERION = "verified_existence_over_hallucinated_volume"
MANAGER_MODEL = "Qwen/Qwen3.5-397B-A17B"
PAIRWISE_ENDPOINT = "vllm-8001"
PHASE5_OUTPUT_DIR = "rubric_evolution_phase5"
REFINE_V1 = {
    "protocol_version": "refine-v1",
    "rubric_memory_mode": "global_rubric_v1",
    "max_description_chars": 1800,
    "wrong_image_count": 3,
    "correct_image_count": 2,
    "abstain_image_count": 1,
    "correct_text_count": 6,
    "abstain_text_count": 6,
    "candidate_count": 1,
    "forced_smoke_target_node_id": TARGET_NODE_ID,
    "heldout_protocol": "exploratory_diagnostic_only",
}

REFINE_ROLE_V2 = {
    "protocol_version": "refine-role-aware-v2",
    "source_experiment": SOURCE_EXPERIMENT_DIR,
    "trigger_mode": "role_aware_v2",
    "candidate_scope": "all_committed_nodes",
    "rubric_memory_mode": "global_rubric_v1",
    "min_epochs": 2,
    "max_epochs": 3,
    "retry_rejected_nodes": True,
    "synchronous_epoch_commit": True,
    "worker_endpoint": PAIRWISE_ENDPOINT,
    "heldout_access": "final_stage_only",
}

VISUAL_SPLIT_REFINE_V1 = {
    "protocol_version": "visual-split-refine-local-v1",
    "source_experiment": VISUAL_SPLIT_REFINE_SOURCE_DIR,
    "root_id": VISUAL_SPLIT_REFINE_ROOT_ID,
    "candidate_scope": "committed_children_only",
    "split_enabled": False,
    "trigger_mode": "role_aware_v2",
    "rubric_memory_mode": "global_rubric_v1",
    "max_epochs": 3,
    "retry_rejected_nodes": True,
    "synchronous_epoch_commit": True,
    "worker_endpoint": VISUAL_SPLIT_REFINE_ENDPOINT,
    "heldout_access": "final_stage_only",
    "exploratory": True,
}

FIVE_ROOT_LOCKED_SPLIT_REFINE_V1 = {
    "protocol_version": "five-root-locked-split-role-aware-refine-v1",
    "candidate_scope": "initial_roots_split_all_committed_nodes_refine",
    "rubric_memory_mode": "global_rubric_v1",
    "min_epochs": 3,
    "max_epochs": 5,
    "split_retry_reuse_clusters": True,
    "split_retry_max_locked_children": 1,
    "strong_child_min_support": 15,
    "strong_child_min_net_corrected": 3,
    "child_refine_trigger_mode": "role_aware_v2",
    "retry_rejected_operations": True,
    "synchronous_epoch_commit": True,
    "worker_endpoint": "vllm-8001",
    "heldout_access": "final_stage_only",
    "partial_split_acceptance": False,
    "exploratory": True,
}

FIVE_ROOT_LOCKED_SPLIT_REFINE_PROTOCOL = split.EvolutionProtocol(
    FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR,
    "five-root-locked-split-refine",
    rubric_memory_mode="global_rubric_v1",
    control_experiment_dir=split.CONTROL_EXPERIMENT_DIR,
    read_only_control_signatures=True,
    allow_configured_endpoint_pool=True,
)

ROLE_CHECKPOINT_HELDOUT_V1 = {
    "protocol_version": "role-aware-refine-checkpoint-heldout-v1",
    # Keep this JSON-native: tuple serialization otherwise makes freeze
    # idempotency falsely report a manifest drift after the first write.
    "checkpoints": ["source", "epoch_01", "epoch_03"],
    "worker_endpoint": PAIRWISE_ENDPOINT,
    "exploratory": True,
    "selection_after_heldout_forbidden": True,
}

ROLE_CHECKPOINT_HELDOUT_V2 = {
    "protocol_version": "role-aware-refine-checkpoint-heldout-v2",
    "checkpoints": ["source", "epoch_01", "epoch_03"],
    "worker_endpoint": PAIRWISE_ENDPOINT,
    "exploratory": True,
    "selection_after_heldout_forbidden": True,
    "repair_of": "role-aware-refine-checkpoint-heldout-v1-node-id-dedup-bug",
}

SPLIT_REFINE_PROTOCOL = split.EvolutionProtocol(
    FULL_EXPERIMENT_DIR,
    "split-refine",
    rubric_memory_mode="global_rubric_v1",
    control_experiment_dir=split.CONTROL_EXPERIMENT_DIR,
    read_only_control_signatures=True,
)


class RefineStageError(RuntimeError):
    """A resumable or scientific Refine stage failed."""


class RefineTransportPause(RefineStageError):
    """A Refine network operation must be resumed without committing history."""


class RefineAttributionInvalid(RefineStageError):
    """A rejection attribution remained schema-invalid and cannot enter history."""


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _role_target(output: Path) -> Path:
    return output / ROLE_EXPERIMENT_DIR


def _visual_split_refine_target(output: Path) -> Path:
    return output / VISUAL_SPLIT_REFINE_EXPERIMENT_DIR


def _five_root_locked_split_refine_target(output: Path) -> Path:
    return output / FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR


def _role_checkpoint_target(output: Path) -> Path:
    return _role_target(output) / ROLE_CHECKPOINT_HELDOUT_DIR


def _role_checkpoint_v2_target(output: Path) -> Path:
    """Keep the repaired checkpoint diagnostic separate from invalid v1 artifacts."""
    return _role_target(output) / ROLE_CHECKPOINT_HELDOUT_V2_DIR


def _source(output: Path) -> Path:
    return output / SOURCE_EXPERIMENT_DIR


def _manifest_path(manifest: Mapping[str, Any], key: str) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"frozen manifest field {key!r} must be a non-empty path")
    return Path(value)


def _validate_output(output: Path) -> None:
    if output.resolve().name != PHASE5_OUTPUT_DIR:
        raise ValueError(f"Refine must run under {PHASE5_OUTPUT_DIR}; got {output}")


def _config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("refine_operator")
    if value != REFINE_V1:
        raise ValueError("refine_operator must equal the frozen Refine v1 protocol")
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    expected = {"tau_acc": .55, "tau_refine": .80, "tau_cov_high": .80,
                "N_min_support": 15}
    if any(thresholds.get(key) != item for key, item in expected.items()):
        raise ValueError(f"Refine thresholds must equal {expected}")
    return dict(value)


def _role_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the independent role-aware Refine ablation protocol."""
    _config(config)
    value = config.get("refine_role_experiment")
    if value != REFINE_ROLE_V2:
        raise ValueError(
            "refine_role_experiment must equal the frozen role-aware v2 protocol")
    return dict(value)


def _visual_split_refine_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed local Split-to-Refine experiment protocol."""
    _config(config)
    value = config.get("visual_split_refine_experiment")
    if value != VISUAL_SPLIT_REFINE_V1:
        raise ValueError(
            "visual_split_refine_experiment must equal the frozen local v1 protocol")
    return dict(value)


def _five_root_locked_split_refine_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen full five-root integration protocol."""
    _config(config)
    value = config.get("five_root_locked_split_refine_experiment")
    if value != FIVE_ROOT_LOCKED_SPLIT_REFINE_V1:
        raise ValueError(
            "five_root_locked_split_refine_experiment must equal the frozen "
            "Phase 10 protocol")
    split.validate_policy(config, FIVE_ROOT_LOCKED_SPLIT_REFINE_PROTOCOL)
    return dict(value)


def _manager(config: Mapping[str, Any]) -> tuple[RefineManager, dict[str, Any]]:
    profile = config.get("refine_manager")
    if not isinstance(profile, Mapping):
        raise ValueError("config must define refine_manager")
    if profile.get("model") != MANAGER_MODEL:
        raise ValueError(f"Refine Manager must use {MANAGER_MODEL}")
    spec = BackendPoolSpec.from_dict(profile["backend_pool"])
    api_key_env = profile.get("api_key_env")
    api_keys = "EMPTY" if api_key_env is None else os.environ.get(api_key_env, "")
    if api_key_env is not None and not api_keys:
        raise RuntimeError(f"environment variable {api_key_env!r} required by Refine is missing")
    generation = profile.get("generation_request_kwargs")
    attribution = profile.get("attribution_request_kwargs")
    if not isinstance(generation, Mapping) or not isinstance(attribution, Mapping):
        raise ValueError("refine_manager must define generation/attribution request kwargs")
    manager = RefineManager(
        model=profile["model"],
        backend_pool=AvailableSlotBackendPool(spec),
        api_keys=api_keys,
        api_retry_attempts=config["api_retry_attempts"],
        structured_max_retries=config["structured_max_retries"],
        generation_request_kwargs=generation,
        attribution_request_kwargs=attribution,
        rubric_memory_mode="global_rubric_v1",
    )
    public = {"model": profile["model"], "backend_pool": spec.to_dict(),
              "api_key_env": api_key_env, "generation_request_kwargs": dict(generation),
              "attribution_request_kwargs": dict(attribution),
              "rubric_memory_mode": "global_rubric_v1"}
    return manager, public


def _source_discovery_paths(source: Path) -> tuple[Path, Path, Path]:
    history = load_json(source / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("Global-memory Split source experiment is incomplete")
    epoch = int(history["current_epoch"])
    epoch_dir = source / "epochs" / f"epoch_{epoch:02d}"
    return (source / "final" / "rubric.json",
            epoch_dir / "discovery_pairwise.json",
            epoch_dir / "feedback.json")


def _rows(config: Mapping[str, Any], split: str):
    count = 90 if split == "discovery" else 500
    return load_jsonl_dataset(base._path(config[f"{split}_dataset"]), expected_count=count)


def _same_pairwise_scientific_identity(
    frozen: DualWorkerRequestSpec,
    runtime: DualWorkerRequestSpec,
) -> bool:
    """Allow only execution-pool routing to differ from a frozen Worker spec."""
    frozen_value = frozen.to_dict(); runtime_value = runtime.to_dict()
    frozen_value.pop("backend_id"); runtime_value.pop("backend_id")
    return frozen_value == runtime_value


def _smoke_go_no_go(
    discovery: Mapping[str, Any],
    heldout_report: Mapping[str, Any] | None,
) -> str:
    """Return a final decision only after the heldout diagnostic exists."""
    if discovery.get("decision") != "accept":
        return "stop_and_diagnose"
    if heldout_report is None:
        return "pending_heldout"
    if (heldout_report.get("heldout_node_accuracy_delta", -1.0) >= 0
            and heldout_report.get("heldout_subtree_delta", -1.0) >= 0):
        return "go"
    return "stop_and_diagnose"


def _case(row: Mapping[str, Any], output, outcome: str) -> dict[str, Any]:
    return {"sample_id": str(row["sample_id"]), "outcome": outcome,
            "question": row["question"], "A": row["A"], "B": row["B"],
            "gold": row["answer"], "current_vote": output.vote.value,
            "worker_thought": output.thought}


def build_refine_evidence(
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    rows: Sequence[Mapping[str, Any]],
    node_id: str,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    node = rubric.get_node(node_id)
    name = node.criterion.name
    wrong, correct, abstain, invalid = [], [], [], []
    for row, outputs in zip(rows, prediction.node_outputs):
        output = outputs[name]
        if not output.parse_ok or not output.answer_valid:
            invalid.append(_case(row, output, "invalid"))
        elif output.vote is Vote.ABSTAIN:
            abstain.append(_case(row, output, "abstain"))
        elif output.vote.value == row["answer"]:
            correct.append(_case(row, output, "correct"))
        else:
            wrong.append(_case(row, output, "wrong"))
    representative_ids = (
        [item["sample_id"] for item in wrong[:settings["wrong_image_count"]]]
        + [item["sample_id"] for item in correct[:settings["correct_image_count"]]]
        + [item["sample_id"] for item in abstain[:settings["abstain_image_count"]]]
    )
    by_id = {str(row["sample_id"]): row for row in rows}
    evidence = {
        "schema_version": "1.0.0",
        "node_id": node_id,
        "criterion_name": name,
        "sample_count": len(rows),
        "support": len(wrong) + len(correct),
        "correct": len(correct),
        "wrong": len(wrong),
        "abstain": len(abstain),
        "invalid": len(invalid),
        "accuracy": len(correct) / (len(correct) + len(wrong)) if correct or wrong else 0.0,
        "coverage": (len(correct) + len(wrong)) / len(rows),
        "wrong_cases": wrong,
        "correct_boundary_cases": correct[:settings["correct_text_count"]],
        "abstain_boundary_cases": abstain[:settings["abstain_text_count"]],
        "invalid_cases": invalid,
        "representative_sample_ids": representative_ids,
    }
    return evidence, tuple(by_id[sample_id] for sample_id in representative_ids)


def _changed_predictions(before, after, rows, criterion_name):
    records = []
    for row, old_row, new_row in zip(rows, before.node_outputs, after.node_outputs):
        old = old_row[criterion_name]; new = new_row[criterion_name]
        if old.vote is not new.vote:
            records.append({"sample_id": str(row["sample_id"]), "gold": row["answer"],
                            "old_vote": old.vote.value, "new_vote": new.vote.value,
                            "old_thought": old.thought, "new_thought": new.thought})
    return records


def _full_target(output: Path) -> Path:
    return output / FULL_EXPERIMENT_DIR


def _refine_shard(node_id: str) -> str:
    return "n" + canonical_sha256(node_id)[:12]


def _refine_attempt(epoch_dir: Path, node_id: str, attempt_no: int) -> Path:
    return epoch_dir / "refine" / _refine_shard(node_id) / f"attempt_{attempt_no:02d}"


def _refine_history_projection(history: Mapping[str, Any], node_id: str) -> list[dict[str, Any]]:
    result = []
    for record in history.get("refine_attempts", []):
        if record["node_id"] != node_id or record["decision"] == "accepted":
            continue
        result.append({
            "epoch": record["epoch"],
            "attempt": record["attempt"],
            "decision": record["decision"],
            "old_description_sha256": record.get("old_description_sha256"),
            "proposed_description": record.get("proposed_description"),
            "node_evaluation": record.get("node_evaluation"),
            "natural_language_attribution": record.get("natural_language_attribution"),
        })
    return result


def _merge_epoch_rubric(
    rubric: StructuredRubric,
    split_candidates: Mapping[str, Any],
    refine_candidates: Mapping[str, RefineCandidate],
) -> StructuredRubric:
    committed = split.merge_accepted_rubrics(rubric, split_candidates)
    nodes = dict(committed.nodes)
    for node_id in sorted(refine_candidates):
        candidate = refine_candidates[node_id]
        patch = candidate.edit_candidate.patch
        if patch.base_rubric_sha256 != rubric.rubric_sha256:
            raise ValueError("Refine candidate base mismatch during synchronous commit")
        if (patch.remove_node_ids or patch.remove_edges or patch.add_edges
                or len(patch.upsert_nodes) != 1
                or patch.upsert_nodes[0].node_id != node_id):
            raise ValueError("Refine patch must replace exactly one existing node")
        if node_id not in rubric.nodes:
            raise ValueError("Refine may not target a same-epoch Split child")
        nodes[node_id] = patch.upsert_nodes[0]
    return StructuredRubric(nodes=nodes, edges=committed.edges,
                            root_ids=committed.root_ids)


def _merge_epoch_predictions(
    baseline: PairwisePredictionOutput,
    split_predictions: Sequence[PairwisePredictionOutput],
    refine_predictions: Mapping[str, PairwisePredictionOutput],
    committed: StructuredRubric,
) -> PairwisePredictionOutput:
    # Split's merger validates criterion names, while Refine preserves names. It is
    # therefore safe to merge additions first and replace refined outputs second.
    merged = split.merge_predictions(baseline, split_predictions, committed)
    rows = [dict(row) for row in merged.node_outputs]
    for node_id, prediction in sorted(refine_predictions.items()):
        criterion = committed.get_node(node_id).criterion
        if (prediction.sample_ids != baseline.sample_ids
                or prediction.sample_fingerprints != baseline.sample_fingerprints
                or prediction.request_spec != baseline.request_spec):
            raise ValueError("Refine prediction identity mismatch during commit")
        if len(prediction.criteria) != 1 or prediction.criteria[0].name != criterion.name:
            raise ValueError("Refine prediction shard must contain the target criterion only")
        for row, shard in zip(rows, prediction.node_outputs):
            row[criterion.name] = shard[criterion.name]
    ordered = tuple(committed.get_node(node_id).criterion
                    for node_id in committed.preorder_node_ids())
    rows = tuple({criterion.name: row[criterion.name] for criterion in ordered}
                 for row in rows)
    answers = tuple(aggregate_flat_votes(output.vote for output in row.values())
                    for row in rows)
    return PairwisePredictionOutput(
        baseline.sample_ids, baseline.sample_fingerprints,
        tuple(StructuredCriterionSnapshot(item.name, item.description)
              for item in ordered), rows, answers, baseline.request_spec)


def _prepare_refine_attempt(
    *,
    config: Mapping[str, Any],
    epoch_dir: Path,
    node_id: str,
    attempt_no: int,
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    feedback: RubricFeedback,
    rows: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    manager: RefineManager,
    pool: AvailableSlotBackendPool,
    rubric_memory: Mapping[str, Any],
    rubric_memory_sha256: str,
    trigger_mode: str = "uniform_v1",
) -> dict[str, Any]:
    attempt_dir = _refine_attempt(epoch_dir, node_id, attempt_no)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    context = EvolutionContext(rubric, feedback)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    trigger = detect_refine_trigger(
        context, node_id, thresholds, trigger_mode=trigger_mode)
    _write(attempt_dir / "trigger.json", trigger.to_dict())
    if not trigger.triggered:
        return {"node_id": node_id, "decision": "not_eligible",
                "attempt_dir": attempt_dir, "trigger": trigger}
    if rubric_memory_sha256 != canonical_sha256(rubric_memory):
        raise RuntimeError("Refine rubric memory hash drift")
    _write(attempt_dir / "rubric_memory_ref.json", {
        "rubric_memory_mode": "global_rubric_v1",
        "rubric_memory_sha256": rubric_memory_sha256,
        "request_spec": manager.request_specs()["refine_generation"].to_dict(),
    })
    evidence, representative_rows = build_refine_evidence(
        rubric, prediction, rows, node_id, REFINE_V1)
    _write(attempt_dir / "evidence.json", evidence)
    prior = _refine_history_projection(history, node_id)
    _write(attempt_dir / "history_projection.json", {
        "schema_version": "1.0.0", "node_id": node_id,
        "attempt": attempt_no, "history": prior,
        "projection_sha256": canonical_sha256(prior),
    })
    proposal_path = attempt_dir / "proposal.json"
    if proposal_path.exists():
        proposal = RefineProposal.from_dict(load_json(proposal_path))
    else:
        try:
            proposal = manager.generate(
                node=rubric.get_node(node_id), evidence=evidence,
                representative_rows=representative_rows,
                prior_failures=prior, rubric_memory=rubric_memory,
                max_description_chars=REFINE_V1["max_description_chars"],
            )
        except RefineManagerFailure as exc:
            _write(attempt_dir / "proposal_failure.json", exc.to_dict())
            if split._manager_failure_kind(exc) == split.TRANSPORT_FAILED:
                raise RefineTransportPause(str(exc)) from exc
            return {"node_id": node_id, "decision": "proposal_invalid",
                    "attempt_dir": attempt_dir, "trigger": trigger,
                    "history_payload": None}
        _write(proposal_path, proposal.to_dict())
    candidate = build_refine_candidate(context, node_id, proposal)
    _write(attempt_dir / "candidate.json", candidate.to_dict())
    after = apply_rubric_patch(rubric, candidate.edit_candidate.patch)
    after.save_json(attempt_dir / "candidate_rubric.json")
    node = after.get_node(node_id)
    node_rubric = StructuredRubric({node_id: node}, (), (node_id,))
    try:
        candidate_prediction, artifact, valid_rate = base._generate_pairwise(
            config, attempt_dir, node_rubric, rows, "candidate_pairwise",
            execution_backend_pool=pool,
            request_backend_id=prediction.request_spec.backend_id,
            request_level_progress=True,
        )
    except RuntimeError as exc:
        if "final-valid rate below" in str(exc):
            raise RefineTransportPause(str(exc)) from exc
        raise
    candidate_prediction.save_json(attempt_dir / "candidate_pairwise.json")
    combined = assemble_refined_pairwise_prediction(
        prediction, candidate_prediction, after, node_id)
    combined.save_json(attempt_dir / "combined_pairwise.json")
    evaluation, old_execution, new_execution = evaluate_refine_candidate(
        before_rubric=rubric, after_rubric=after,
        before_prediction=prediction, combined_prediction=combined,
        candidate_prediction=candidate_prediction, dataset=rows,
        node_id=node_id, min_support=thresholds["N_min_support"],
    )
    _write(attempt_dir / "node_evaluation.json", evaluation.to_dict())
    _write(attempt_dir / "description_diagnostic.json", {
        "old_length": len(rubric.get_node(node_id).criterion.description),
        "new_length": len(node.criterion.description),
        "length_delta": (len(node.criterion.description)
                         - len(rubric.get_node(node_id).criterion.description)),
    })
    _write(attempt_dir / "subtree_diagnostic.json", {
        key: value for key, value in evaluation.to_dict().items()
        if key.startswith("subtree_") or key.startswith("old_subtree_")
        or key.startswith("new_subtree_")
    })
    _write(attempt_dir / "m1_diagnostic.json", {
        key: value for key, value in evaluation.to_dict().items()
        if key.startswith("m1_") or key.startswith("old_m1_")
        or key.startswith("new_m1_")
    })
    old_execution.save_json(attempt_dir / "m1_before.json")
    new_execution.save_json(attempt_dir / "m1_after.json")
    changed = _changed_predictions(
        prediction, combined, rows, node.criterion.name)
    _write(attempt_dir / "changed_predictions.json", changed)
    decision = ("accepted" if evaluation.decision is EvolutionDecision.ACCEPT
                else "competition_rejected")
    attribution = None
    if decision == "competition_rejected":
        attribution_path = attempt_dir / "failure_attribution.json"
        if attribution_path.exists():
            attribution = load_json(attribution_path)
        else:
            try:
                attribution = manager.attribute_failure(
                    original_node=rubric.get_node(node_id), proposal=proposal,
                    evaluation=evaluation, changed_predictions=changed,
                    rubric_memory=rubric_memory,
                )
            except RefineManagerFailure as exc:
                _write(attempt_dir / "failure_attribution_failure.json", exc.to_dict())
                if split._manager_failure_kind(exc) == split.TRANSPORT_FAILED:
                    raise RefineTransportPause(str(exc)) from exc
                _write(attempt_dir / "attribution_invalid.json", {
                    "outcome": "attribution_invalid",
                    "competition_decision_not_committed": True,
                    "history_appended": False,
                    "details": exc.to_dict(),
                })
                raise RefineAttributionInvalid(str(exc)) from exc
            _write(attribution_path, attribution)
        if not isinstance(attribution.get("attribution"), Mapping):
            raise RefineAttributionInvalid("Refine attribution payload is missing")
    history_payload = None if attribution is None else {
        "natural_language_attribution": attribution["attribution"],
        "node_evaluation": evaluation.to_dict(),
        "proposed_description": proposal.description,
    }
    _write(attempt_dir / "history_projection_result.json", {
        "decision": decision, "history_payload": history_payload})
    return {
        "node_id": node_id, "decision": decision, "attempt_dir": attempt_dir,
        "trigger": trigger, "proposal": proposal, "candidate": candidate,
        "after_rubric": after, "candidate_prediction": candidate_prediction,
        "combined": combined, "evaluation": evaluation,
        "pairwise_artifact": str(artifact), "valid_rate": valid_rate,
        "history_payload": history_payload,
    }


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _config(config); _validate_output(output)
    pool_spec = BackendPoolSpec.from_dict(config["backend_pool"])
    if (len(pool_spec.endpoints) != 1
            or pool_spec.endpoints[0].endpoint_id != PAIRWISE_ENDPOINT):
        raise RuntimeError("Refine Pairwise must use only vllm-8001")
    target = _target(output); source = _source(output)
    if (target / "frozen_manifest.json").exists():
        print("refine-freeze already completed")
        return
    rubric_path, prediction_path, feedback_path = _source_discovery_paths(source)
    required = [rubric_path, prediction_path, feedback_path,
                source / "final" / "discovery_report.json",
                source / "heldout500" / "combined_pairwise.json",
                source / "heldout500" / "frozen_manifest.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Refine source artifacts are missing: {missing}")
    rubric = StructuredRubric.load_json(rubric_path)
    prediction = PairwisePredictionOutput.load_json(prediction_path)
    feedback = RubricFeedback.from_dict(load_json(feedback_path))
    if feedback.rubric_sha256 != rubric.rubric_sha256:
        raise RuntimeError("source feedback/rubric mismatch")
    node = rubric.get_node(TARGET_NODE_ID)
    if node.criterion.name != TARGET_CRITERION:
        raise RuntimeError("forced Refine target identity changed")
    discovery_path = base._path(config["discovery_dataset"])
    heldout_path = base._path(config["heldout_dataset"])
    if file_sha256(discovery_path).lower() != config["discovery_dataset_sha256"].lower():
        raise RuntimeError("discovery dataset hash mismatch")
    if file_sha256(heldout_path).lower() != config["heldout_dataset_sha256"].lower():
        raise RuntimeError("heldout dataset hash mismatch")
    manager, profile = _manager(config)
    expected_worker = base._expected_pairwise_request_spec(config, _rows(config, "discovery"))
    if not _same_pairwise_scientific_identity(
            prediction.request_spec, expected_worker):
        raise RuntimeError(
            "source Pairwise scientific identity differs from current config")
    memory = rubric_memory_snapshot(rubric)
    trigger = detect_refine_trigger(
        EvolutionContext(rubric, feedback), TARGET_NODE_ID,
        config["evolution_policy"]["trigger_thresholds"], forced=False)
    manifest = {
        "schema_version": "1.0.0", "stage": "forced_refine_smoke_v1",
        "settings": settings, "source_experiment": SOURCE_EXPERIMENT_DIR,
        "source_rubric_path": str(rubric_path),
        "source_rubric_sha256": rubric.rubric_sha256,
        "source_prediction_path": str(prediction_path),
        "source_prediction_sha256": file_sha256(prediction_path),
        "source_feedback_path": str(feedback_path),
        "target_node_id": TARGET_NODE_ID, "target_criterion_name": TARGET_CRITERION,
        "automatic_trigger": trigger.to_dict(), "forced": True,
        "discovery_dataset_sha256": file_sha256(discovery_path),
        "heldout_dataset_sha256": file_sha256(heldout_path),
        "pairwise_request_spec": expected_worker.to_dict(),
        "manager_profile": profile,
        "manager_request_specs": {key: value.to_dict()
                                  for key, value in manager.request_specs().items()},
        "rubric_memory": memory,
        "rubric_memory_sha256": canonical_sha256(memory),
        "heldout_accessed": False,
    }
    _write(target / "frozen_manifest.json", manifest)
    _write(target / "stage_status.json", {"freeze": {"status": "passed"}})
    print(json.dumps({"status": "passed", "target": str(target),
                      "automatic_trigger": trigger.to_dict()}, indent=2))


def split_refine_freeze(config: Mapping[str, Any], output: Path) -> None:
    """Freeze a fresh five-root Split+Refine trajectory after smoke success."""
    _config(config)
    _validate_output(output)
    smoke_report_path = _target(output) / "report.json"
    if not smoke_report_path.exists():
        raise RuntimeError("run refine-smoke-report before split-refine-freeze")
    if load_json(smoke_report_path).get("go_no_go") != "go":
        raise RuntimeError("Refine smoke did not pass the frozen go/no-go gate")
    manager, profile = _manager(config)
    pool_spec = BackendPoolSpec.from_dict(config["backend_pool"])
    if (len(pool_spec.endpoints) != 1
            or pool_spec.endpoints[0].endpoint_id != PAIRWISE_ENDPOINT):
        raise RuntimeError("Split+Refine Pairwise must use only vllm-8001")
    target = _full_target(output)
    split.freeze(config, output, SPLIT_REFINE_PROTOCOL)
    manifest_path = target / "frozen_manifest.json"
    manifest = load_json(manifest_path)
    manifest.update({
        "refine_protocol": REFINE_V1,
        "refine_manager_profile": profile,
        "refine_manager_request_specs": {
            key: value.to_dict() for key, value in manager.request_specs().items()},
        "refine_smoke_report_sha256": file_sha256(smoke_report_path),
        "operator_schedule": "synchronous_epoch_start_split_plus_refine",
        "heldout_protocol": {
            "decision": "reuse_exact_global_memory_outputs_generate_changed",
            "source_experiment": SOURCE_EXPERIMENT_DIR,
            "confirmatory": False,
            "selection_after_heldout_forbidden": True,
        },
    })
    _write(manifest_path, manifest)
    history_path = target / "evolution_history.json"
    history = load_json(history_path)
    history.setdefault("refine_states", {})
    history.setdefault("refine_attempts", [])
    _write(history_path, history)
    print(json.dumps({"target": str(target), "smoke_gate": "go",
                      "initial_roots": manifest["initial_root_ids"]}, indent=2))


def _record_refine_state(history: dict[str, Any], result: Mapping[str, Any],
                         epoch_no: int, elapsed: float) -> dict[str, Any]:
    node_id = result["node_id"]
    state = history["refine_states"].setdefault(
        node_id, {"attempt_count": 0, "accepted_count": 0,
                  "last_decision": None, "last_epoch": None})
    state["attempt_count"] += 1
    if result["decision"] == "accepted":
        state["accepted_count"] += 1
    state["last_decision"] = result["decision"]
    state["last_epoch"] = epoch_no
    evaluation = result.get("evaluation")
    proposal = result.get("proposal")
    payload = result.get("history_payload")
    record = {
        "epoch": epoch_no, "node_id": node_id,
        "criterion_name": None if proposal is None else proposal.criterion_name,
        "attempt": state["attempt_count"], "decision": result["decision"],
        "attempt_dir": str(result["attempt_dir"]),
        "old_description_sha256": (None if proposal is None
                                    else proposal.original_description_sha256),
        "proposed_description": None if proposal is None else proposal.description,
        "node_evaluation": None if evaluation is None else evaluation.to_dict(),
        "natural_language_attribution": (
            None if payload is None else payload.get("natural_language_attribution")),
        "elapsed_seconds": elapsed,
    }
    if (result["decision"] == "competition_rejected"
            and not isinstance(record["natural_language_attribution"], Mapping)):
        raise RuntimeError("rejected Refine cannot enter history without attribution")
    history["refine_attempts"].append(record)
    return record


def _current_refine_eligible(rubric: StructuredRubric, feedback: RubricFeedback,
                             config: Mapping[str, Any], *,
                             trigger_mode: str = "uniform_v1") -> tuple[str, ...]:
    context = EvolutionContext(rubric, feedback)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    return tuple(node_id for node_id in rubric.preorder_node_ids()
                 if detect_refine_trigger(
                     context, node_id, thresholds,
                     trigger_mode=trigger_mode).triggered)


def _required_attribution(manager, attempt_dir, **kwargs):
    try:
        artifact = manager.attribute_failure(**kwargs)
    except RefineManagerFailure as exc:
        _write(attempt_dir / "failure_attribution_failure.json", exc.to_dict())
        if split._manager_failure_kind(exc) == split.TRANSPORT_FAILED:
            _write(attempt_dir / "transport_failure.json", exc.to_dict())
            raise RefineStageError("Refine failure attribution transport failed") from exc
        _write(attempt_dir / "attribution_invalid.json", {
            "outcome": "attribution_invalid", "details": exc.to_dict(),
            "competition_decision_not_committed": True, "history_appended": False})
        raise RefineStageError("Refine failure attribution remained invalid") from exc
    if not isinstance(artifact.get("attribution"), Mapping):
        raise RefineStageError("Refine rejection lacks natural-language attribution")
    _write(attempt_dir / "failure_attribution.json", artifact)
    return artifact


def _prepare_split_results(
    config, target, epoch_dir, rubric, prediction, feedback, rows, history,
    managers, specs, memory, memory_hash,
):
    results = {}; started = {}
    scheduled = split.retryable_roots(history)
    for root_id in scheduled:
        started[root_id] = time.monotonic()
        attempt_no = history["root_states"][root_id]["attempt_count"] + 1
        try:
            results[root_id] = split._prepare(
                config, target, epoch_dir, root_id, attempt_no, rubric,
                prediction, feedback, rows, history, managers, specs,
                SPLIT_REFINE_PROTOCOL, memory, memory_hash)
        except split.TransportFailed as exc:
            raise RefineTransportPause(str(exc)) from exc
        except split.SpecializeManagerFailure as exc:
            if split._manager_failure_kind(exc) == split.TRANSPORT_FAILED:
                raise RefineTransportPause(str(exc)) from exc
            attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
            payload = split._proposal_failure_payload(
                attempt_dir, exc.stage, exc.to_dict())
            _write(attempt_dir / "proposal_failure.json", payload)
            results[root_id] = {
                "root_id": root_id, "decision": split.PROPOSAL_INVALID,
                "attempt_dir": attempt_dir, "history_payload": payload}
        except split.ProposalInvalid as exc:
            attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
            payload = split._proposal_failure_payload(
                attempt_dir, exc.stage, split._failure_details(exc))
            _write(attempt_dir / "proposal_failure.json", payload)
            results[root_id] = {
                "root_id": root_id, "decision": split.PROPOSAL_INVALID,
                "attempt_dir": attempt_dir, "history_payload": payload}
    prepared = {root: value["candidate"] for root, value in results.items()
                if value["decision"] == "prepared"}
    collisions = split.colliding_roots(prepared)
    collided = {root for owners in collisions.values() for root in owners}
    for root_id in collided:
        results[root_id]["decision"] = "cross_root_collision"
        results[root_id]["history_payload"] = split._failure(
            results[root_id], "cross_root_child_name_collision",
            "synchronous_commit", details={"collisions": collisions})
        _write(results[root_id]["attempt_dir"] / "collision.json",
               results[root_id]["history_payload"])
    for root_id in sorted(set(prepared) - collided):
        try:
            results[root_id] = split._evaluate(
                config, epoch_dir, rows, rubric, prediction, results[root_id],
                split_pool(config), managers["semantic_cluster"], feedback)
        except split.TransportFailed as exc:
            raise RefineTransportPause(str(exc)) from exc
        except split.AttributionInvalid as exc:
            raise RefineAttributionInvalid(str(exc)) from exc
    return tuple(scheduled), results, started, collisions


def split_pool(config):
    """Build the frozen single-8001 pool at the point an epoch needs it."""
    return base._single_endpoint_execution_pool(config, PAIRWISE_ENDPOINT)


def split_refine_run(config: Mapping[str, Any], output: Path) -> None:
    """Run synchronous Split v1 and Refine v1 candidates for 3-5 epochs."""
    _config(config); split.validate_policy(config, SPLIT_REFINE_PROTOCOL)
    _validate_output(output); target = _full_target(output)
    if not (target / "frozen_manifest.json").exists():
        raise RuntimeError("run split-refine-freeze first")
    manifest = load_json(target / "frozen_manifest.json")
    history = load_json(target / "evolution_history.json")
    if history.get("completed"):
        print("split-refine-run already completed"); return
    rows = _rows(config, "discovery")
    split_managers, _, split_specs, _ = split._managers(
        config, SPLIT_REFINE_PROTOCOL)
    split_specs["error_signature"] = manifest["manager_request_specs"]["error_signature"]
    refine_manager, _ = _manager(config)
    worker_pool = split_pool(config)
    max_epochs = config["split_evolution"]["max_epochs"]
    for epoch_no in range(history["current_epoch"] + 1, max_epochs + 1):
        epoch_started = time.monotonic()
        previous = split._epoch(target, epoch_no - 1)
        epoch_dir = split._epoch(target, epoch_no)
        rubric = StructuredRubric.load_json(previous / "rubric_committed.json")
        prediction = PairwisePredictionOutput.load_json(previous / "discovery_pairwise.json")
        feedback = RubricFeedback.from_dict(load_json(previous / "feedback.json"))
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        refine_scheduled = _current_refine_eligible(rubric, feedback, config)
        if set(split.retryable_roots(history)).intersection(refine_scheduled):
            raise RuntimeError("Split and Refine triggers must remain mutually exclusive")
        print(f"split-refine epoch={epoch_no} split={list(split.retryable_roots(history))} "
              f"refine={list(refine_scheduled)}", flush=True)
        try:
            split_scheduled, split_results, split_started, collisions = (
                _prepare_split_results(
                    config, target, epoch_dir, rubric, prediction, feedback,
                    rows, history, split_managers, split_specs, memory, memory_hash))
            refine_results = {}; refine_started = {}
            for node_id in refine_scheduled:
                refine_started[node_id] = time.monotonic()
                attempt_no = history["refine_states"].get(
                    node_id, {"attempt_count": 0})["attempt_count"] + 1
                refine_results[node_id] = _prepare_refine_attempt(
                    config=config, epoch_dir=epoch_dir, node_id=node_id,
                    attempt_no=attempt_no, rubric=rubric,
                    prediction=prediction, feedback=feedback, rows=rows,
                    history=history, manager=refine_manager, pool=worker_pool,
                    rubric_memory=memory, rubric_memory_sha256=memory_hash)
        except RefineTransportPause as exc:
            status = load_json(target / "stage_status.json")
            status["run"] = {"status": "paused", "details": {
                "epoch": epoch_no, "message": str(exc),
                "history_appended": False}}
            _write(target / "stage_status.json", status); return
        except RefineAttributionInvalid as exc:
            status = load_json(target / "stage_status.json")
            status["run"] = {"status": "failed", "details": {
                "epoch": epoch_no, "outcome": "attribution_invalid",
                "message": str(exc), "history_appended": False}}
            _write(target / "stage_status.json", status); return
        accepted_split = {root: result["candidate"]
                          for root, result in split_results.items()
                          if result["decision"] == split.ACCEPTED}
        accepted_refine = {node_id: result["candidate"]
                           for node_id, result in refine_results.items()
                           if result["decision"] == "accepted"}
        committed = _merge_epoch_rubric(rubric, accepted_split, accepted_refine)
        committed_prediction = _merge_epoch_predictions(
            prediction,
            [split_results[root]["child_prediction"] for root in sorted(accepted_split)],
            {node_id: refine_results[node_id]["candidate_prediction"]
             for node_id in accepted_refine}, committed)
        votes, committed_feedback = split._snapshot(
            epoch_dir, committed, committed_prediction, rows, manifest)
        split_records = _commit_split_history(
            history, split_scheduled, split_results, split_started,
            epoch_no, max_epochs)
        refine_records = [_record_refine_state(
            history, refine_results[node_id], epoch_no,
            time.monotonic() - refine_started[node_id])
            for node_id in refine_scheduled]
        history["current_epoch"] = epoch_no
        next_refine = _current_refine_eligible(committed, committed_feedback, config)
        pending = bool(split.retryable_roots(history) or next_refine)
        if epoch_no >= 3 and not pending:
            history.update({"completed": True,
                            "stop_reason": "no_retryable_operations"})
        elif epoch_no == max_epochs:
            history.update({"completed": True, "stop_reason": "max_epochs_reached"})
        metrics = base._metrics(votes, rows)
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no, "split_scheduled": list(split_scheduled),
            "refine_scheduled": list(refine_scheduled),
            "accepted_split_roots": sorted(accepted_split),
            "accepted_refine_nodes": sorted(accepted_refine),
            "split_collisions": collisions, "split_attempts": split_records,
            "refine_attempts": refine_records,
            "next_refine_eligible": list(next_refine),
            "rubric_node_count": len(committed.nodes), "m1": metrics,
            "epoch_wall_seconds": time.monotonic() - epoch_started})
        _write(target / "evolution_history.json", history)
        print(f"split-refine epoch={epoch_no} split_accept={sorted(accepted_split)} "
              f"refine_accept={sorted(accepted_refine)} m1_acc={metrics['accuracy']:.4f}",
              flush=True)
        if history["completed"]: break
    status = load_json(target / "stage_status.json")
    status["run"] = {"status": "passed" if history["completed"] else "partial",
                     "details": {"current_epoch": history["current_epoch"],
                                 "stop_reason": history["stop_reason"]}}
    _write(target / "stage_status.json", status)


def _commit_split_history(history, scheduled, results, started,
                          epoch_no, max_epochs):
    records = []
    for root_id in scheduled:
        result = results[root_id]; state = history["root_states"][root_id]
        state["attempt_count"] += 1; decision = result["decision"]
        if decision == split.ACCEPTED:
            state.update({"status": "accepted_locked", "accepted_epoch": epoch_no,
                          "children": [item.criterion_name
                                       for item in result["candidate"].children]})
        else:
            state["status"] = "exhausted" if epoch_no >= max_epochs else "retryable"
        evaluation = result.get("evaluation")
        record = {
            "epoch": epoch_no, "root_id": root_id,
            "attempt": state["attempt_count"], "decision": decision,
            "competition_completed": decision in split.VALID_COMPETITION_OUTCOMES,
            "attempt_dir": str(result["attempt_dir"]),
            "parent_accuracy": None if evaluation is None else evaluation.parent_accuracy,
            "specialized_accuracy": (None if evaluation is None
                                     else evaluation.specialized_accuracy),
            "accuracy_delta": None if evaluation is None else evaluation.accuracy_delta,
            "elapsed_seconds": time.monotonic() - started[root_id],
            "history_payload": result.get("history_payload")}
        if (decision == split.COMPETITION_REJECTED
                and not isinstance((record["history_payload"] or {}).get(
                    "natural_language_attribution"), Mapping)):
            raise RuntimeError("rejected Split cannot enter history without attribution")
        history["attempts"].append(record); records.append(record)
    return records


def split_refine_report(config: Mapping[str, Any], output: Path) -> None:
    """Freeze the final discovery Rubric and summarize both operator tracks."""
    _config(config); _validate_output(output); target = _full_target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run split-refine-run to completion first")
    final_epoch = split._epoch(target, history["current_epoch"])
    rubric = StructuredRubric.load_json(final_epoch / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(final_epoch / "discovery_pairwise.json")
    rows = _rows(config, "discovery")
    execution, answers = execute_offline_m1(rubric, prediction, rows)
    initial_summary = load_json(split._epoch(target, 0) / "summary.json")
    final_metrics = base._metrics(answers, rows)
    final_dir = target / "final"; final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    execution.save_json(final_dir / "m1_execution.json")
    report_value = {
        "schema_version": "1.0.0",
        "final_epoch": history["current_epoch"],
        "final_rubric_sha256": rubric.rubric_sha256,
        "initial_m1": initial_summary["m1"], "final_m1": final_metrics,
        "m1_accuracy_delta": (final_metrics["accuracy"]
                              - initial_summary["m1"]["accuracy"]),
        "node_count_initial": len(StructuredRubric.load_json(
            split._epoch(target, 0) / "rubric_committed.json").nodes),
        "node_count_final": len(rubric.nodes),
        "split_attempts": history["attempts"],
        "refine_attempts": history["refine_attempts"],
        "accepted_refine_count": sum(
            item["decision"] == "accepted" for item in history["refine_attempts"]),
        "rejected_refine_count": sum(
            item["decision"] == "competition_rejected"
            for item in history["refine_attempts"]),
    }
    _write(final_dir / "discovery_report.json", report_value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_rubric_sha256": rubric.rubric_sha256,
        "final_m1_accuracy": final_metrics["accuracy"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(report_value, indent=2, ensure_ascii=False))


def _combine_heldout_predictions(
    source: PairwisePredictionOutput,
    generated: PairwisePredictionOutput | None,
    final_rubric: StructuredRubric,
) -> PairwisePredictionOutput:
    source_desc = {item.name: item.description for item in source.criteria}
    generated_names = set() if generated is None else {
        item.name for item in generated.criteria}
    if generated is not None and (
            generated.sample_ids != source.sample_ids
            or generated.sample_fingerprints != source.sample_fingerprints
            or generated.request_spec != source.request_spec):
        raise ValueError("heldout generated/source prediction identity mismatch")
    rows = []
    for index, source_row in enumerate(source.node_outputs):
        generated_row = {} if generated is None else generated.node_outputs[index]
        row = {}
        for node_id in final_rubric.preorder_node_ids():
            criterion = final_rubric.get_node(node_id).criterion
            if (criterion.name in source_row
                    and source_desc.get(criterion.name) == criterion.description):
                row[criterion.name] = source_row[criterion.name]
            elif criterion.name in generated_names:
                row[criterion.name] = generated_row[criterion.name]
            else:
                raise ValueError(f"missing heldout output for {criterion.name}")
        rows.append(row)
    ordered = tuple(final_rubric.get_node(node_id).criterion
                    for node_id in final_rubric.preorder_node_ids())
    answers = tuple(aggregate_flat_votes(value.vote for value in row.values())
                    for row in rows)
    return PairwisePredictionOutput(
        source.sample_ids, source.sample_fingerprints,
        tuple(StructuredCriterionSnapshot(item.name, item.description)
              for item in ordered), tuple(rows), answers, source.request_spec)


def split_refine_heldout(config: Mapping[str, Any], output: Path) -> None:
    """Evaluate the frozen final trajectory once on heldout-500."""
    _config(config); _validate_output(output); target = _full_target(output)
    final_report_path = target / "final" / "discovery_report.json"
    if not final_report_path.exists():
        raise RuntimeError("run split-refine-report first")
    final_rubric = StructuredRubric.load_json(target / "final" / "rubric.json")
    source_path = _source(output) / "heldout500" / "combined_pairwise.json"
    source_prediction = PairwisePredictionOutput.load_json(source_path)
    initial_path = output / "predictions" / "init_pairwise_p05_heldout500.json"
    initial_prediction = PairwisePredictionOutput.load_json(initial_path)
    rows = _rows(config, "heldout")
    expected = base._expected_pairwise_request_spec(config, rows)
    if (not _same_pairwise_scientific_identity(source_prediction.request_spec, expected)
            or not _same_pairwise_scientific_identity(
                initial_prediction.request_spec, expected)):
        raise RuntimeError("heldout source/current Worker scientific identity mismatch")
    final_dir = target / "heldout500"; final_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0.0", "exploratory": True,
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "final_discovery_report_sha256": file_sha256(final_report_path),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "source_prediction_sha256": file_sha256(source_path),
        "initial_prediction_sha256": file_sha256(initial_path),
        "pairwise_request_spec": expected.to_dict(),
        "selection_after_heldout_forbidden": True,
    }
    frozen_path = final_dir / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != manifest:
        raise RuntimeError("Split+Refine heldout manifest drift")
    _write(frozen_path, manifest)
    source_desc = {item.name: item.description for item in source_prediction.criteria}
    missing_ids = tuple(node_id for node_id in final_rubric.preorder_node_ids()
                        if source_desc.get(final_rubric.get_node(
                            node_id).criterion.name)
                        != final_rubric.get_node(node_id).criterion.description)
    generated = None
    if missing_ids:
        missing_rubric = StructuredRubric(
            {node_id: final_rubric.get_node(node_id) for node_id in missing_ids},
            (), missing_ids)
        generated, _, _ = base._generate_pairwise(
            config, final_dir, missing_rubric, rows, "changed_nodes",
            execution_backend_pool=split_pool(config),
            request_backend_id=source_prediction.request_spec.backend_id,
            request_level_progress=True)
        generated.save_json(final_dir / "changed_node_predictions.json")
    combined = _combine_heldout_predictions(
        source_prediction, generated, final_rubric)
    combined.save_json(final_dir / "combined_pairwise.json")
    final_execution, final_answers = execute_offline_m1(final_rubric, combined, rows)
    final_execution.save_json(final_dir / "m1_execution.json")
    initial_rubric = StructuredRubric.load_json(
        split._epoch(target, 0) / "rubric_committed.json")
    initial_prediction = project_pairwise_prediction(initial_prediction, initial_rubric)
    initial_execution, initial_answers = execute_offline_m1(
        initial_rubric, initial_prediction, rows)
    initial_execution.save_json(final_dir / "initial_m1_execution.json")
    gold = tuple(str(row["answer"]) for row in rows)
    corrected = [sample_id for sample_id, old, new, answer in zip(
        combined.sample_ids, initial_answers, final_answers, gold)
                 if old.value != answer and new.value == answer]
    harmed = [sample_id for sample_id, old, new, answer in zip(
        combined.sample_ids, initial_answers, final_answers, gold)
              if old.value == answer and new.value != answer]
    initial_metrics = base._heldout_vote_metrics(initial_answers, rows)
    final_metrics = base._heldout_vote_metrics(final_answers, rows)
    frozen_baseline = load_json(output / "reports" / "init_baseline.json")[
        "heldout500"][base.StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]
    if (initial_metrics["accuracy"] != frozen_baseline["accuracy"]
            or initial_metrics["coverage"] != frozen_baseline["coverage"]):
        raise RuntimeError("Init heldout M1 replay differs from frozen baseline")
    paired = base._paired_heldout_comparison(
        initial_answers, final_answers, rows)
    report_value = {
        "schema_version": "1.0.0", "exploratory": True,
        "initial_m1": initial_metrics, "final_m1": final_metrics,
        "m1_accuracy_delta": final_metrics["accuracy"] - initial_metrics["accuracy"],
        "corrected_sample_ids": corrected, "harmed_sample_ids": harmed,
        "corrected": len(corrected), "harmed": len(harmed),
        "net_corrected": len(corrected) - len(harmed),
        "paired_init_to_final": paired,
        "reused_exact_criteria": len(final_rubric.nodes) - len(missing_ids),
        "generated_changed_criteria": len(missing_ids),
        "generated_node_ids": list(missing_ids),
    }
    _write(final_dir / "report.json", report_value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": report_value}
    _write(target / "stage_status.json", status)
    print(json.dumps(report_value, indent=2, ensure_ascii=False))


def split_refine_final_report(config: Mapping[str, Any], output: Path) -> None:
    _config(config); _validate_output(output); target = _full_target(output)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout_report = load_json(target / "heldout500" / "report.json")
    control_report = load_json(
        _source(output) / "heldout500" / "report.json")
    value = {
        "schema_version": "1.0.0", "exploratory": True,
        "discovery": discovery, "heldout": heldout_report,
        "split_only_global_memory_control": control_report,
        "primary": {
            "split_refine_heldout_m1_accuracy": heldout_report["final_m1"]["accuracy"],
            "initial_heldout_m1_accuracy": heldout_report["initial_m1"]["accuracy"],
            "net_corrected": heldout_report["net_corrected"],
        },
    }
    _write(target / "final_report.json", value)
    lines = ["# Split + Refine Evolution v1", "",
             f"- Discovery M1: {discovery['initial_m1']['accuracy']:.4f} -> {discovery['final_m1']['accuracy']:.4f}",
             f"- Heldout M1: {heldout_report['initial_m1']['accuracy']:.4f} -> {heldout_report['final_m1']['accuracy']:.4f}",
             f"- Heldout corrected/harmed: {heldout_report['corrected']}/{heldout_report['harmed']}",
             f"- Accepted Refine attempts: {discovery['accepted_refine_count']}"]
    (target / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed"}
    _write(target / "stage_status.json", status)
    print(json.dumps(value["primary"], indent=2))


def _require_pairwise_endpoint(config: Mapping[str, Any], endpoint_id: str) -> None:
    spec = BackendPoolSpec.from_dict(config["backend_pool"])
    if endpoint_id not in {item.endpoint_id for item in spec.endpoints}:
        raise RuntimeError(f"Pairwise endpoint {endpoint_id!r} is not configured")


def refine_role_freeze(config: Mapping[str, Any], output: Path) -> None:
    """Freeze the Split-only final Rubric without touching heldout artifacts."""
    settings = _role_config(config)
    _validate_output(output)
    _require_pairwise_endpoint(config, PAIRWISE_ENDPOINT)
    target = _role_target(output)
    manifest_path = target / "frozen_manifest.json"
    if manifest_path.exists():
        print("refine-role-freeze already completed")
        return
    source = _source(output)
    rubric_path, prediction_path, feedback_path = _source_discovery_paths(source)
    required = (rubric_path, prediction_path, feedback_path,
                source / "final" / "discovery_report.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"role-aware Refine source artifacts are missing: {missing}")
    rubric = StructuredRubric.load_json(rubric_path)
    prediction = PairwisePredictionOutput.load_json(prediction_path)
    feedback = RubricFeedback.from_dict(load_json(feedback_path))
    if feedback.rubric_sha256 != rubric.rubric_sha256:
        raise RuntimeError("role-aware source feedback/rubric mismatch")
    rows = _rows(config, "discovery")
    discovery_path = base._path(config["discovery_dataset"])
    if file_sha256(discovery_path).lower() != config[
            "discovery_dataset_sha256"].lower():
        raise RuntimeError("discovery dataset hash mismatch")
    expected_worker = base._expected_pairwise_request_spec(config, rows)
    if not _same_pairwise_scientific_identity(
            prediction.request_spec, expected_worker):
        raise RuntimeError("source Pairwise scientific identity differs from current config")
    manager, manager_profile = _manager(config)
    frozen_thresholds = {key: config["evolution_policy"]["trigger_thresholds"][key]
                         for key in ("tau_acc", "tau_refine", "tau_cov_high",
                                     "N_min_support")}
    manifest = {
        "schema_version": "1.0.0",
        "stage": "refine_role_aware_v2",
        "settings": settings,
        "source_experiment": SOURCE_EXPERIMENT_DIR,
        "source_rubric_path": str(rubric_path),
        "source_rubric_sha256": rubric.rubric_sha256,
        "source_prediction_path": str(prediction_path),
        "source_prediction_sha256": file_sha256(prediction_path),
        "source_feedback_path": str(feedback_path),
        "source_feedback_sha256": file_sha256(feedback_path),
        "discovery_dataset_sha256": file_sha256(discovery_path),
        "pairwise_request_spec": prediction.request_spec.to_dict(),
        "manager_profile": manager_profile,
        "trigger_thresholds": frozen_thresholds,
        "child_trigger": {
            "accuracy": "0.5 < ACC < 0.8", "minimum_support": 15,
            "minimum_wrong": 5, "coverage_ceiling": None},
        "acceptance_rule": "new_node_accuracy > old_node_accuracy and new_support >= 15",
        "manager_request_specs": {
            key: value.to_dict() for key, value in manager.request_specs().items()},
        "heldout_accessed": False,
        "heldout_artifacts_frozen": False,
    }
    _write(manifest_path, manifest)
    epoch0 = split._epoch(target, 0)
    votes, snapshot_feedback = split._snapshot(
        epoch0, rubric, prediction, rows, manifest)
    _write(epoch0 / "summary.json", {
        "epoch": 0, "source_experiment": SOURCE_EXPERIMENT_DIR,
        "rubric_node_count": len(rubric.nodes), "m1": base._metrics(votes, rows)})
    if snapshot_feedback.to_dict() != feedback.to_dict():
        raise RuntimeError("role-aware source feedback replay mismatch")
    history = {
        "schema_version": "1.0.0", "current_epoch": 0,
        "completed": False, "stop_reason": None,
        "refine_states": {}, "refine_attempts": [],
    }
    _write(target / "evolution_history.json", history)
    _write(target / "stage_status.json", {"freeze": {"status": "passed"}})
    print(json.dumps({"status": "passed", "target": str(target),
                      "source_nodes": len(rubric.nodes)}, indent=2))


def _trigger_audit_rows(
    rubric: StructuredRubric,
    feedback: RubricFeedback,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    context = EvolutionContext(rubric, feedback)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    rows = []
    for node_id in rubric.preorder_node_ids():
        uniform = detect_refine_trigger(
            context, node_id, thresholds, trigger_mode="uniform_v1")
        role = detect_refine_trigger(
            context, node_id, thresholds, trigger_mode="role_aware_v2")
        node = rubric.get_node(node_id)
        rows.append({
            "node_id": node_id,
            "criterion_name": node.criterion.name,
            "role": "child" if rubric.parent_id(node_id) is not None else "root",
            "accuracy": role.accuracy, "coverage": role.coverage,
            "support": role.support, "wrong": role.wrong,
            "uniform_v1_eligible": uniform.triggered,
            "uniform_v1_reasons": list(uniform.reasons),
            "role_aware_v2_eligible": role.triggered,
            "role_aware_v2_reasons": list(role.reasons),
            "newly_eligible": role.triggered and not uniform.triggered,
        })
    return rows


def refine_role_audit(config: Mapping[str, Any], output: Path) -> None:
    """Offline trigger audit; this stage intentionally has no heldout path."""
    _role_config(config)
    _validate_output(output)
    target = _role_target(output)
    manifest_path = target / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("run refine-role-freeze first")
    manifest = load_json(manifest_path)
    rubric = StructuredRubric.load_json(_manifest_path(manifest, "source_rubric_path"))
    feedback = RubricFeedback.from_dict(load_json(
        _manifest_path(manifest, "source_feedback_path")))
    rows = _trigger_audit_rows(rubric, feedback, config)
    value = {
        "schema_version": "1.0.0", "offline_only": True,
        "node_count": len(rows),
        "uniform_eligible": sum(item["uniform_v1_eligible"] for item in rows),
        "role_aware_eligible": sum(
            item["role_aware_v2_eligible"] for item in rows),
        "newly_eligible_children": [
            item["node_id"] for item in rows
            if item["role"] == "child" and item["newly_eligible"]],
        "nodes": rows,
        "heldout_accessed": False,
    }
    _write(target / "trigger_audit.json", value)
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": {
        "uniform_eligible": value["uniform_eligible"],
        "role_aware_eligible": value["role_aware_eligible"],
        "newly_eligible_children": len(value["newly_eligible_children"])}}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def refine_role_run(config: Mapping[str, Any], output: Path) -> None:
    """Run Refine-only synchronous evolution for two to three epochs."""
    settings = _role_config(config)
    _validate_output(output)
    target = _role_target(output)
    if not (target / "frozen_manifest.json").exists():
        raise RuntimeError("run refine-role-freeze first")
    manifest = load_json(target / "frozen_manifest.json")
    frozen_thresholds = manifest.get("trigger_thresholds")
    current_thresholds = {
        key: config["evolution_policy"]["trigger_thresholds"][key]
        for key in ("tau_acc", "tau_refine", "tau_cov_high", "N_min_support")}
    if frozen_thresholds is not None and frozen_thresholds != current_thresholds:
        raise RuntimeError("role-aware Refine trigger thresholds drifted after freeze")
    history = load_json(target / "evolution_history.json")
    if history.get("completed"):
        print("refine-role-run already completed")
        return
    rows = _rows(config, "discovery")
    manager, _ = _manager(config)
    if ({key: value.to_dict() for key, value in manager.request_specs().items()}
            != manifest["manager_request_specs"]):
        raise RuntimeError("role-aware Refine Manager request identity drift")
    pool = base._single_endpoint_execution_pool(config, PAIRWISE_ENDPOINT)
    min_epochs = settings["min_epochs"]
    max_epochs = settings["max_epochs"]
    for epoch_no in range(history["current_epoch"] + 1, max_epochs + 1):
        epoch_started = time.monotonic()
        previous = split._epoch(target, epoch_no - 1)
        epoch_dir = split._epoch(target, epoch_no)
        rubric = StructuredRubric.load_json(previous / "rubric_committed.json")
        prediction = PairwisePredictionOutput.load_json(
            previous / "discovery_pairwise.json")
        feedback = RubricFeedback.from_dict(load_json(previous / "feedback.json"))
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        scheduled = _current_refine_eligible(
            rubric, feedback, config, trigger_mode=settings["trigger_mode"])
        print(f"refine-role epoch={epoch_no} scheduled={list(scheduled)}", flush=True)
        results: dict[str, Any] = {}
        started: dict[str, float] = {}
        try:
            for node_id in scheduled:
                started[node_id] = time.monotonic()
                attempt_no = history["refine_states"].get(
                    node_id, {"attempt_count": 0})["attempt_count"] + 1
                results[node_id] = _prepare_refine_attempt(
                    config=config, epoch_dir=epoch_dir, node_id=node_id,
                    attempt_no=attempt_no, rubric=rubric,
                    prediction=prediction, feedback=feedback, rows=rows,
                    history=history, manager=manager, pool=pool,
                    rubric_memory=memory, rubric_memory_sha256=memory_hash,
                    trigger_mode=settings["trigger_mode"])
        except RefineTransportPause as exc:
            status = load_json(target / "stage_status.json")
            status["run"] = {"status": "paused", "details": {
                "epoch": epoch_no, "message": str(exc), "history_appended": False}}
            _write(target / "stage_status.json", status)
            return
        except RefineAttributionInvalid as exc:
            status = load_json(target / "stage_status.json")
            status["run"] = {"status": "failed", "details": {
                "epoch": epoch_no, "outcome": "attribution_invalid",
                "message": str(exc), "history_appended": False}}
            _write(target / "stage_status.json", status)
            return
        accepted = {node_id: result["candidate"]
                    for node_id, result in results.items()
                    if result["decision"] == "accepted"}
        committed = _merge_epoch_rubric(rubric, {}, accepted)
        committed_prediction = _merge_epoch_predictions(
            prediction, [], {
                node_id: results[node_id]["candidate_prediction"]
                for node_id in accepted}, committed)
        votes, committed_feedback = split._snapshot(
            epoch_dir, committed, committed_prediction, rows, manifest)
        records = [_record_refine_state(
            history, results[node_id], epoch_no,
            time.monotonic() - started[node_id]) for node_id in scheduled]
        history["current_epoch"] = epoch_no
        next_eligible = _current_refine_eligible(
            committed, committed_feedback, config,
            trigger_mode=settings["trigger_mode"])
        if epoch_no >= min_epochs and not next_eligible:
            history.update({"completed": True,
                            "stop_reason": "no_retryable_refine_nodes"})
        elif epoch_no == max_epochs:
            history.update({"completed": True, "stop_reason": "max_epochs_reached"})
        metrics = base._metrics(votes, rows)
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no, "trigger_mode": settings["trigger_mode"],
            "refine_scheduled": list(scheduled),
            "accepted_refine_nodes": sorted(accepted),
            "refine_attempts": records,
            "next_refine_eligible": list(next_eligible),
            "rubric_node_count": len(committed.nodes), "m1": metrics,
            "epoch_wall_seconds": time.monotonic() - epoch_started})
        _write(target / "evolution_history.json", history)
        print(f"refine-role epoch={epoch_no} accept={sorted(accepted)} "
              f"m1_acc={metrics['accuracy']:.4f}", flush=True)
        if history["completed"]:
            break
    status = load_json(target / "stage_status.json")
    status["run"] = {"status": "passed" if history["completed"] else "partial",
                     "details": {"current_epoch": history["current_epoch"],
                                 "stop_reason": history["stop_reason"]}}
    _write(target / "stage_status.json", status)


def refine_role_report(config: Mapping[str, Any], output: Path) -> None:
    """Freeze the discovery-selected role-aware Refine result."""
    settings = _role_config(config)
    _validate_output(output)
    target = _role_target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run refine-role-run to completion first")
    initial_epoch = split._epoch(target, 0)
    final_epoch = split._epoch(target, history["current_epoch"])
    initial_rubric = StructuredRubric.load_json(initial_epoch / "rubric_committed.json")
    final_rubric = StructuredRubric.load_json(final_epoch / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(final_epoch / "discovery_pairwise.json")
    rows = _rows(config, "discovery")
    execution, answers = execute_offline_m1(final_rubric, prediction, rows)
    initial_metrics = load_json(initial_epoch / "summary.json")["m1"]
    final_metrics = base._metrics(answers, rows)
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    execution.save_json(final_dir / "m1_execution.json")
    report_value = {
        "schema_version": "1.0.0", "trigger_mode": settings["trigger_mode"],
        "final_epoch": history["current_epoch"],
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "initial_rubric_sha256": initial_rubric.rubric_sha256,
        "initial_m1": initial_metrics, "final_m1": final_metrics,
        "m1_accuracy_delta": final_metrics["accuracy"] - initial_metrics["accuracy"],
        "node_count": len(final_rubric.nodes),
        "refine_attempts": history["refine_attempts"],
        "accepted_refine_count": sum(
            item["decision"] == "accepted" for item in history["refine_attempts"]),
        "rejected_refine_count": sum(
            item["decision"] == "competition_rejected"
            for item in history["refine_attempts"]),
        "heldout_accessed": False,
    }
    _write(final_dir / "discovery_report.json", report_value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "final_m1_accuracy": final_metrics["accuracy"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(report_value, indent=2, ensure_ascii=False))


def refine_role_heldout(config: Mapping[str, Any], output: Path) -> None:
    """Access heldout only after the discovery-selected Rubric is frozen."""
    _role_config(config)
    _validate_output(output)
    target = _role_target(output)
    discovery_path = target / "final" / "discovery_report.json"
    if not discovery_path.exists():
        raise RuntimeError("run refine-role-report first")
    discovery = load_json(discovery_path)
    final_rubric = StructuredRubric.load_json(target / "final" / "rubric.json")
    if discovery["final_rubric_sha256"] != final_rubric.rubric_sha256:
        raise RuntimeError("role-aware final Rubric drift")
    source_rubric_path, _, _ = _source_discovery_paths(_source(output))
    source_rubric = StructuredRubric.load_json(source_rubric_path)
    source_path = _source(output) / "heldout500" / "combined_pairwise.json"
    source_prediction = PairwisePredictionOutput.load_json(source_path)
    rows = _rows(config, "heldout")
    expected = base._expected_pairwise_request_spec(config, rows)
    if not _same_pairwise_scientific_identity(source_prediction.request_spec, expected):
        raise RuntimeError("heldout source/current Worker scientific identity mismatch")
    heldout_dir = target / "heldout500"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    heldout_manifest = {
        "schema_version": "1.0.0", "exploratory": True,
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "final_discovery_report_sha256": file_sha256(discovery_path),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "source_prediction_sha256": file_sha256(source_path),
        "pairwise_request_spec": expected.to_dict(),
        "selection_after_heldout_forbidden": True,
    }
    frozen_path = heldout_dir / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != heldout_manifest:
        raise RuntimeError("role-aware Refine heldout manifest drift")
    _write(frozen_path, heldout_manifest)
    source_desc = {item.name: item.description for item in source_prediction.criteria}
    changed_ids = tuple(
        node_id for node_id in final_rubric.preorder_node_ids()
        if source_desc.get(final_rubric.get_node(node_id).criterion.name)
        != final_rubric.get_node(node_id).criterion.description)
    generated = None
    if changed_ids:
        changed_rubric = StructuredRubric(
            {node_id: final_rubric.get_node(node_id) for node_id in changed_ids},
            (), changed_ids)
        generated, _, _ = base._generate_pairwise(
            config, heldout_dir, changed_rubric, rows, "changed_nodes",
            execution_backend_pool=base._single_endpoint_execution_pool(
                config, PAIRWISE_ENDPOINT),
            request_backend_id=source_prediction.request_spec.backend_id,
            request_level_progress=True)
        generated.save_json(heldout_dir / "changed_node_predictions.json")
    combined = _combine_heldout_predictions(
        source_prediction, generated, final_rubric)
    combined.save_json(heldout_dir / "combined_pairwise.json")
    source_projected = project_pairwise_prediction(source_prediction, source_rubric)
    source_execution, source_answers = execute_offline_m1(
        source_rubric, source_projected, rows)
    final_execution, final_answers = execute_offline_m1(final_rubric, combined, rows)
    source_execution.save_json(heldout_dir / "source_m1_execution.json")
    final_execution.save_json(heldout_dir / "m1_execution.json")
    gold = tuple(str(row["answer"]) for row in rows)
    corrected = [sample_id for sample_id, old, new, answer in zip(
        combined.sample_ids, source_answers, final_answers, gold)
                 if old.value != answer and new.value == answer]
    harmed = [sample_id for sample_id, old, new, answer in zip(
        combined.sample_ids, source_answers, final_answers, gold)
              if old.value == answer and new.value != answer]
    source_metrics = base._heldout_vote_metrics(source_answers, rows)
    final_metrics = base._heldout_vote_metrics(final_answers, rows)
    report_value = {
        "schema_version": "1.0.0", "exploratory": True,
        "source_split_only_m1": source_metrics, "final_refined_m1": final_metrics,
        "m1_accuracy_delta": final_metrics["accuracy"] - source_metrics["accuracy"],
        "corrected_sample_ids": corrected, "harmed_sample_ids": harmed,
        "corrected": len(corrected), "harmed": len(harmed),
        "net_corrected": len(corrected) - len(harmed),
        "paired_source_to_final": base._paired_heldout_comparison(
            source_answers, final_answers, rows),
        "reused_exact_criteria": len(final_rubric.nodes) - len(changed_ids),
        "generated_changed_criteria": len(changed_ids),
        "generated_node_ids": list(changed_ids),
        "selection_after_heldout_forbidden": True,
    }
    _write(heldout_dir / "report.json", report_value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": report_value}
    _write(target / "stage_status.json", status)
    print(json.dumps(report_value, indent=2, ensure_ascii=False))


def refine_role_final_report(config: Mapping[str, Any], output: Path) -> None:
    _role_config(config)
    _validate_output(output)
    target = _role_target(output)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout_report = load_json(target / "heldout500" / "report.json")
    audit = load_json(target / "trigger_audit.json")
    value = {
        "schema_version": "1.0.0", "exploratory": True,
        "trigger_audit": audit, "discovery": discovery,
        "heldout": heldout_report,
        "primary": {
            "heldout_m1_accuracy": heldout_report["final_refined_m1"]["accuracy"],
            "split_only_control_accuracy": heldout_report[
                "source_split_only_m1"]["accuracy"],
            "net_corrected": heldout_report["net_corrected"],
        },
    }
    _write(target / "final_report.json", value)
    lines = ["# Role-aware Refine v2", "",
             f"- Newly eligible children: {len(audit['newly_eligible_children'])}",
             f"- Discovery M1: {discovery['initial_m1']['accuracy']:.4f} -> {discovery['final_m1']['accuracy']:.4f}",
             f"- Heldout M1: {heldout_report['source_split_only_m1']['accuracy']:.4f} -> {heldout_report['final_refined_m1']['accuracy']:.4f}",
             f"- Heldout corrected/harmed: {heldout_report['corrected']}/{heldout_report['harmed']}"]
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed"}
    _write(target / "stage_status.json", status)
    print(json.dumps(value["primary"], indent=2))


def _checkpoint_rubrics(output: Path) -> dict[str, StructuredRubric]:
    """Load the three pre-selected Role-aware Refine checkpoints."""
    role = _role_target(output)
    paths = {
        "source": _source(output) / "final" / "rubric.json",
        "epoch_01": role / "epochs" / "epoch_01" / "rubric_committed.json",
        "epoch_03": role / "epochs" / "epoch_03" / "rubric_committed.json",
    }
    rubrics = {name: StructuredRubric.load_json(path) for name, path in paths.items()}
    reference = rubrics["source"]
    for name, rubric in rubrics.items():
        if rubric.preorder_node_ids() != reference.preorder_node_ids():
            raise RuntimeError(f"checkpoint topology drift: {name}")
        for node_id in rubric.preorder_node_ids():
            old = reference.get_node(node_id).criterion
            current = rubric.get_node(node_id).criterion
            if (old.name != current.name or old.score != current.score):
                raise RuntimeError(f"checkpoint immutable criterion drift: {name}/{node_id}")
    return rubrics


def _checkpoint_description_map(
    rubrics: Mapping[str, StructuredRubric],
) -> tuple[dict[str, list[str]], list[str]]:
    """Return changed-node maps and the one shared generated node set."""
    source = rubrics["source"]
    changes: dict[str, list[str]] = {}
    generated: set[str] = set()
    for checkpoint in ("epoch_01", "epoch_03"):
        changed = [
            node_id for node_id in source.preorder_node_ids()
            if (source.get_node(node_id).criterion.description
                != rubrics[checkpoint].get_node(node_id).criterion.description)
        ]
        changes[checkpoint] = changed
        generated.update(changed)
    return changes, [node_id for node_id in source.preorder_node_ids() if node_id in generated]


def _checkpoint_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Build a deterministic, pre-request freeze manifest for the diagnostic."""
    _role_config(config)
    role = _role_target(output)
    status = load_json(role / "stage_status.json")
    if status.get("run", {}).get("status") != "passed":
        raise RuntimeError("Role-aware Refine discovery trajectory is incomplete")
    rubrics = _checkpoint_rubrics(output)
    changes, generated = _checkpoint_description_map(rubrics)
    source_prediction_path = _source(output) / "heldout500" / "combined_pairwise.json"
    source_prediction = PairwisePredictionOutput.load_json(source_prediction_path)
    rows = _rows(config, "heldout")
    expected = base._expected_pairwise_request_spec(config, rows)
    if not _same_pairwise_scientific_identity(source_prediction.request_spec, expected):
        raise RuntimeError("heldout source/current Worker scientific identity mismatch")
    source = rubrics["source"]
    source_descriptions = {
        node_id: source.get_node(node_id).criterion.description
        for node_id in source.preorder_node_ids()
    }
    return {
        "schema_version": "1.0.0",
        **ROLE_CHECKPOINT_HELDOUT_V1,
        "source_prediction_path": str(source_prediction_path),
        "source_prediction_sha256": file_sha256(source_prediction_path),
        "heldout_dataset_path": str(base._path(config["heldout_dataset"])),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "sample_ids": list(source_prediction.sample_ids),
        "sample_fingerprints": list(source_prediction.sample_fingerprints),
        "pairwise_request_spec": expected.to_dict(),
        "checkpoint_rubrics": {
            name: {
                "path": str((
                    _source(output) / "final" / "rubric.json"
                    if name == "source" else role / "epochs" / name / "rubric_committed.json")),
                "rubric_sha256": rubric.rubric_sha256,
                "node_descriptions_sha256": canonical_sha256({
                    node_id: rubric.get_node(node_id).criterion.description
                    for node_id in rubric.preorder_node_ids()}),
            }
            for name, rubric in rubrics.items()
        },
        "source_node_descriptions_sha256": canonical_sha256(source_descriptions),
        "changed_node_ids": changes,
        "generated_node_ids": generated,
        "generated_criterion_names": [
            rubrics["epoch_03"].get_node(node_id).criterion.name
            for node_id in generated
        ],
        "generated_request_count": len(generated) * len(rows),
    }


def refine_role_checkpoints_freeze(config: Mapping[str, Any], output: Path) -> None:
    """Freeze S0/S1/S3 and all heldout request identities before evaluation."""
    _validate_output(output)
    target = _role_checkpoint_target(output)
    target.mkdir(parents=True, exist_ok=True)
    manifest = _checkpoint_manifest(config, output)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        raise RuntimeError("Role-aware checkpoint heldout manifest drift")
    _write(path, manifest)
    _write(target / "stage_status.json", {"freeze": {"status": "passed"}})
    print(json.dumps({"generated_node_count": len(manifest["generated_node_ids"]),
                      "generated_request_count": manifest["generated_request_count"]},
                     indent=2))


def _verify_checkpoint_freeze(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target = _role_checkpoint_target(output)
    path = target / "frozen_manifest.json"
    if not path.exists():
        raise RuntimeError("run refine-role-checkpoints-freeze first")
    frozen = load_json(path)
    current = _checkpoint_manifest(config, output)
    if frozen != current:
        raise RuntimeError("Role-aware checkpoint heldout inputs drifted after freeze")
    return frozen


def _checkpoint_generated_rubric(
    rubrics: Mapping[str, StructuredRubric], node_ids: Sequence[str],
) -> StructuredRubric:
    final = rubrics["epoch_03"]
    nodes = {node_id: final.get_node(node_id) for node_id in node_ids}
    return StructuredRubric(nodes, (), tuple(node_ids))


def refine_role_checkpoints_run(config: Mapping[str, Any], output: Path) -> None:
    """Generate only changed checkpoint descriptions and replay all three systems."""
    _role_config(config); _validate_output(output)
    target = _role_checkpoint_target(output)
    manifest = _verify_checkpoint_freeze(config, output)
    status_path = target / "stage_status.json"
    status = load_json(status_path)
    if status.get("run", {}).get("status") == "passed":
        print("refine-role-checkpoints-run already completed")
        return
    rubrics = _checkpoint_rubrics(output)
    rows = _rows(config, "heldout")
    source_prediction = PairwisePredictionOutput.load_json(
        Path(manifest["source_prediction_path"]))
    generated_path = target / "predictions" / "changed_node_predictions.json"
    generated = (PairwisePredictionOutput.load_json(generated_path)
                 if generated_path.exists() else None)
    node_ids = tuple(manifest["generated_node_ids"])
    if generated is None and node_ids:
        generated, _, _ = base._generate_pairwise(
            config, target, _checkpoint_generated_rubric(rubrics, node_ids), rows,
            "changed_node_predictions",
            execution_backend_pool=base._single_endpoint_execution_pool(
                config, PAIRWISE_ENDPOINT),
            request_backend_id=source_prediction.request_spec.backend_id,
            request_level_progress=True)
    if generated is not None:
        generated.save_json(generated_path)
    evaluations: dict[str, Any] = {}
    answers_by_checkpoint: dict[str, Any] = {}
    for checkpoint, rubric in rubrics.items():
        combined = _combine_heldout_predictions(source_prediction, generated, rubric)
        checkpoint_dir = target / "checkpoints" / checkpoint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        combined.save_json(checkpoint_dir / "combined_pairwise.json")
        execution, answers = execute_offline_m1(rubric, combined, rows)
        execution.save_json(checkpoint_dir / "m1_execution.json")
        evaluations[checkpoint] = base._heldout_vote_metrics(answers, rows)
        answers_by_checkpoint[checkpoint] = answers
    pairs = {
        "epoch_01_vs_source": base._paired_heldout_comparison(
            answers_by_checkpoint["source"], answers_by_checkpoint["epoch_01"], rows),
        "epoch_03_vs_source": base._paired_heldout_comparison(
            answers_by_checkpoint["source"], answers_by_checkpoint["epoch_03"], rows),
        "epoch_03_vs_epoch_01": base._paired_heldout_comparison(
            answers_by_checkpoint["epoch_01"], answers_by_checkpoint["epoch_03"], rows),
    }
    _write(target / "evaluations.json", {"schema_version": "1.0.0",
        "exploratory": True, "metrics": evaluations, "paired": pairs,
        "selection_after_heldout_forbidden": True})
    status["run"] = {"status": "passed", "details": {
        "generated_node_count": len(node_ids),
        "generated_request_count": len(node_ids) * len(rows)}}
    _write(status_path, status)
    print(json.dumps({"metrics": evaluations, "paired": pairs}, indent=2))


def refine_role_checkpoints_report(config: Mapping[str, Any], output: Path) -> None:
    """Write the exploratory paired checkpoint report without selecting a winner."""
    _role_config(config); _validate_output(output)
    target = _role_checkpoint_target(output)
    manifest = _verify_checkpoint_freeze(config, output)
    evaluations_path = target / "evaluations.json"
    if not evaluations_path.exists():
        raise RuntimeError("run refine-role-checkpoints-run first")
    value = load_json(evaluations_path)
    report = {"schema_version": "1.0.0", "experiment":
              "exploratory_role_aware_refine_checkpoint_heldout",
              "frozen_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
              "generated_node_ids": manifest["generated_node_ids"],
              "generated_request_count": manifest["generated_request_count"],
              **value}
    _write(target / "report.json", report)
    metrics = report["metrics"]
    pairs = report["paired"]
    lines = ["# Role-aware Refine Checkpoint Heldout Diagnostic", "",
             "Exploratory paired diagnostic: no checkpoint was selected after heldout.", "",
             "| Checkpoint | M1 ACC | Coverage | Correct |",
             "|---|---:|---:|---:|"]
    for name in ("source", "epoch_01", "epoch_03"):
        item = metrics[name]
        lines.append(f"| {name} | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['correct_count']} |")
    lines.extend(["", "| Comparison | Corrected | Harmed | Net | Exact McNemar p |",
                  "|---|---:|---:|---:|---:|"])
    for name, item in pairs.items():
        lines.append(f"| {name} | {item['corrected_count']} | {item['harmed_count']} | "
                     f"{item['net_corrected']} | {item['mcnemar_exact_two_sided_p']:.6f} |")
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed"}
    _write(target / "stage_status.json", status)
    print(json.dumps(report, indent=2))


def _checkpoint_v2_description_map(
    rubrics: Mapping[str, StructuredRubric],
) -> dict[str, list[str]]:
    """Return changed nodes separately for each checkpoint description version.

    Node IDs cannot be used as a cache identity here: a node can be accepted by
    Refine in epoch 1 and then receive a different description in epoch 3.
    """
    source = rubrics["source"]
    return {
        checkpoint: [
            node_id for node_id in source.preorder_node_ids()
            if (source.get_node(node_id).criterion.description
                != rubrics[checkpoint].get_node(node_id).criterion.description)
        ]
        for checkpoint in ("epoch_01", "epoch_03")
    }


def _checkpoint_nodes_rubric(
    rubric: StructuredRubric,
    node_ids: Sequence[str],
) -> StructuredRubric:
    nodes = {node_id: rubric.get_node(node_id) for node_id in node_ids}
    return StructuredRubric(nodes, (), tuple(node_ids))


def _validate_checkpoint_generated_prediction(
    prediction: PairwisePredictionOutput,
    source_prediction: PairwisePredictionOutput,
    rubric: StructuredRubric,
    node_ids: Sequence[str],
    *,
    label: str,
) -> None:
    """Ensure reused/generated outputs were produced for these exact descriptions."""
    expected = tuple(
        rubric.get_node(node_id).criterion for node_id in node_ids)
    actual = tuple((item.name, item.description) for item in prediction.criteria)
    required = tuple((item.name, item.description) for item in expected)
    if actual != required:
        raise RuntimeError(
            f"{label} criterion descriptions do not match the frozen checkpoint")
    if (prediction.sample_ids != source_prediction.sample_ids
            or prediction.sample_fingerprints != source_prediction.sample_fingerprints
            or prediction.request_spec != source_prediction.request_spec):
        raise RuntimeError(f"{label} prediction identity differs from frozen source")


def _checkpoint_v2_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Freeze corrected request identities and validate reusable epoch-3 output."""
    _role_config(config)
    rubrics = _checkpoint_rubrics(output)
    changes = _checkpoint_v2_description_map(rubrics)
    rows = _rows(config, "heldout")
    source_prediction_path = _source(output) / "heldout500" / "combined_pairwise.json"
    source_prediction = PairwisePredictionOutput.load_json(source_prediction_path)
    expected = base._expected_pairwise_request_spec(config, rows)
    if not _same_pairwise_scientific_identity(source_prediction.request_spec, expected):
        raise RuntimeError("heldout source/current Worker scientific identity mismatch")
    v1_prediction_path = (_role_checkpoint_target(output) / "predictions"
                          / "changed_node_predictions.json")
    if not v1_prediction_path.exists():
        raise RuntimeError("cannot reuse epoch-03 predictions: invalid v1 artifact missing")
    epoch_three_reused = PairwisePredictionOutput.load_json(v1_prediction_path)
    _validate_checkpoint_generated_prediction(
        epoch_three_reused, source_prediction, rubrics["epoch_03"],
        changes["epoch_03"], label="reused epoch_03 v1")
    return {
        "schema_version": "1.0.0",
        **ROLE_CHECKPOINT_HELDOUT_V2,
        "source_prediction_path": str(source_prediction_path),
        "source_prediction_sha256": file_sha256(source_prediction_path),
        "heldout_dataset_path": str(base._path(config["heldout_dataset"])),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "sample_ids": list(source_prediction.sample_ids),
        "sample_fingerprints": list(source_prediction.sample_fingerprints),
        "pairwise_request_spec": expected.to_dict(),
        "checkpoint_rubrics": {
            name: {
                "rubric_sha256": rubric.rubric_sha256,
                "node_descriptions_sha256": canonical_sha256({
                    node_id: rubric.get_node(node_id).criterion.description
                    for node_id in rubric.preorder_node_ids()}),
            }
            for name, rubric in rubrics.items()
        },
        "changed_node_ids": changes,
        "reused_epoch_03_prediction_path": str(v1_prediction_path),
        "reused_epoch_03_prediction_sha256": file_sha256(v1_prediction_path),
        "reused_epoch_03_request_count": len(changes["epoch_03"]) * len(rows),
        "generated_epoch_01_request_count": len(changes["epoch_01"]) * len(rows),
    }


def refine_role_checkpoints_v2_freeze(config: Mapping[str, Any], output: Path) -> None:
    """Freeze the repaired, description-identity checkpoint diagnostic."""
    _validate_output(output)
    target = _role_checkpoint_v2_target(output)
    target.mkdir(parents=True, exist_ok=True)
    manifest = _checkpoint_v2_manifest(config, output)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        raise RuntimeError("Role-aware checkpoint v2 heldout manifest drift")
    _write(path, manifest)
    _write(target / "stage_status.json", {"freeze": {"status": "passed"}})
    print(json.dumps({"reused_epoch_03_request_count": manifest["reused_epoch_03_request_count"],
                      "new_epoch_01_request_count": manifest["generated_epoch_01_request_count"]},
                     indent=2))


def _verify_checkpoint_v2_freeze(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target = _role_checkpoint_v2_target(output)
    path = target / "frozen_manifest.json"
    if not path.exists():
        raise RuntimeError("run refine-role-checkpoints-v2-freeze first")
    frozen = load_json(path)
    if frozen != _checkpoint_v2_manifest(config, output):
        raise RuntimeError("Role-aware checkpoint v2 inputs drifted after freeze")
    return frozen


def refine_role_checkpoints_v2_run(config: Mapping[str, Any], output: Path) -> None:
    """Reuse valid epoch-3 outputs and generate epoch-1's four old descriptions."""
    _role_config(config); _validate_output(output)
    target = _role_checkpoint_v2_target(output)
    manifest = _verify_checkpoint_v2_freeze(config, output)
    status_path = target / "stage_status.json"
    status = load_json(status_path)
    if status.get("run", {}).get("status") == "passed":
        print("refine-role-checkpoints-v2-run already completed")
        return
    rubrics = _checkpoint_rubrics(output)
    changes = manifest["changed_node_ids"]
    rows = _rows(config, "heldout")
    source_prediction = PairwisePredictionOutput.load_json(
        Path(manifest["source_prediction_path"]))
    epoch_three = PairwisePredictionOutput.load_json(
        Path(manifest["reused_epoch_03_prediction_path"]))
    _validate_checkpoint_generated_prediction(
        epoch_three, source_prediction, rubrics["epoch_03"], changes["epoch_03"],
        label="reused epoch_03 v1")
    epoch_one_path = target / "predictions" / "epoch_01_changed_descriptions.json"
    epoch_one = (PairwisePredictionOutput.load_json(epoch_one_path)
                 if epoch_one_path.exists() else None)
    if epoch_one is None:
        epoch_one, _, _ = base._generate_pairwise(
            config, target,
            _checkpoint_nodes_rubric(rubrics["epoch_01"], changes["epoch_01"]),
            rows, "epoch_01_changed_descriptions",
            execution_backend_pool=base._single_endpoint_execution_pool(
                config, PAIRWISE_ENDPOINT),
            request_backend_id=source_prediction.request_spec.backend_id,
            request_level_progress=True)
        epoch_one.save_json(epoch_one_path)
    _validate_checkpoint_generated_prediction(
        epoch_one, source_prediction, rubrics["epoch_01"], changes["epoch_01"],
        label="generated epoch_01")
    generated_by_checkpoint = {"epoch_01": epoch_one, "epoch_03": epoch_three}
    evaluations: dict[str, Any] = {}
    answers_by_checkpoint: dict[str, Any] = {}
    for checkpoint, rubric in rubrics.items():
        generated = None if checkpoint == "source" else generated_by_checkpoint[checkpoint]
        combined = _combine_heldout_predictions(source_prediction, generated, rubric)
        checkpoint_dir = target / "checkpoints" / checkpoint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        combined.save_json(checkpoint_dir / "combined_pairwise.json")
        execution, answers = execute_offline_m1(rubric, combined, rows)
        execution.save_json(checkpoint_dir / "m1_execution.json")
        evaluations[checkpoint] = base._heldout_vote_metrics(answers, rows)
        answers_by_checkpoint[checkpoint] = answers
    pairs = {
        "epoch_01_vs_source": base._paired_heldout_comparison(
            answers_by_checkpoint["source"], answers_by_checkpoint["epoch_01"], rows),
        "epoch_03_vs_source": base._paired_heldout_comparison(
            answers_by_checkpoint["source"], answers_by_checkpoint["epoch_03"], rows),
        "epoch_03_vs_epoch_01": base._paired_heldout_comparison(
            answers_by_checkpoint["epoch_01"], answers_by_checkpoint["epoch_03"], rows),
    }
    _write(target / "evaluations.json", {
        "schema_version": "1.0.0", "exploratory": True,
        "metrics": evaluations, "paired": pairs,
        "selection_after_heldout_forbidden": True,
    })
    status["run"] = {"status": "passed", "details": {
        "reused_epoch_03_request_count": manifest["reused_epoch_03_request_count"],
        "generated_epoch_01_request_count": manifest["generated_epoch_01_request_count"],
    }}
    _write(status_path, status)
    print(json.dumps({"metrics": evaluations, "paired": pairs}, indent=2))


def refine_role_checkpoints_v2_report(config: Mapping[str, Any], output: Path) -> None:
    """Write the repaired exploratory paired report without selecting a checkpoint."""
    _role_config(config); _validate_output(output)
    target = _role_checkpoint_v2_target(output)
    manifest = _verify_checkpoint_v2_freeze(config, output)
    evaluations_path = target / "evaluations.json"
    if not evaluations_path.exists():
        raise RuntimeError("run refine-role-checkpoints-v2-run first")
    value = load_json(evaluations_path)
    report = {"schema_version": "1.0.0", "experiment":
              "exploratory_role_aware_refine_checkpoint_heldout_v2",
              "frozen_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
              "reused_epoch_03_request_count": manifest["reused_epoch_03_request_count"],
              "generated_epoch_01_request_count": manifest["generated_epoch_01_request_count"],
              **value}
    _write(target / "report.json", report)
    lines = ["# Role-aware Refine Checkpoint Heldout Diagnostic v2", "",
             "Exploratory paired diagnostic. Epoch-3 predictions are verified v1 reuse; epoch-1 descriptions were generated separately.", "",
             "| Checkpoint | M1 ACC | Coverage | Correct |",
             "|---|---:|---:|---:|"]
    for name in ("source", "epoch_01", "epoch_03"):
        item = report["metrics"][name]
        lines.append(f"| {name} | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['correct_count']} |")
    lines.extend(["", "| Comparison | Corrected | Harmed | Net | Exact McNemar p |",
                  "|---|---:|---:|---:|---:|"])
    for name, item in report["paired"].items():
        lines.append(f"| {name} | {item['corrected_count']} | {item['harmed_count']} | "
                     f"{item['net_corrected']} | {item['mcnemar_exact_two_sided_p']:.6f} |")
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed"}
    _write(target / "stage_status.json", status)
    print(json.dumps(report, indent=2))


def smoke(config: Mapping[str, Any], output: Path) -> None:
    settings = _config(config); _validate_output(output); target = _target(output)
    manifest_path = target / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("run refine-freeze first")
    manifest = load_json(manifest_path); attempt = target / "smoke" / "attempt_01"
    report_path = attempt / "discovery_report.json"
    if report_path.exists():
        print("refine-smoke already completed")
        print(json.dumps(load_json(report_path), indent=2, ensure_ascii=False)); return
    rows = _rows(config, "discovery")
    rubric = StructuredRubric.load_json(_manifest_path(manifest, "source_rubric_path"))
    prediction = PairwisePredictionOutput.load_json(
        _manifest_path(manifest, "source_prediction_path"))
    feedback = RubricFeedback.from_dict(load_json(
        _manifest_path(manifest, "source_feedback_path")))
    memory = manifest["rubric_memory"]
    if canonical_sha256(memory) != manifest["rubric_memory_sha256"]:
        raise RuntimeError("frozen Refine rubric memory drift")
    manager, _ = _manager(config)
    if ({key: value.to_dict() for key, value in manager.request_specs().items()}
            != manifest["manager_request_specs"]):
        raise RuntimeError("Refine Manager request identity drift")
    trigger = detect_refine_trigger(EvolutionContext(rubric, feedback), TARGET_NODE_ID,
                                    config["evolution_policy"]["trigger_thresholds"],
                                    forced=True)
    _write(attempt / "trigger.json", trigger.to_dict())
    evidence, representative_rows = build_refine_evidence(
        rubric, prediction, rows, TARGET_NODE_ID, settings)
    _write(attempt / "evidence.json", evidence)
    _write(attempt / "rubric_memory_ref.json", {
        "rubric_memory_mode": "global_rubric_v1",
        "rubric_memory_sha256": manifest["rubric_memory_sha256"]})
    proposal_path = attempt / "proposal.json"
    if proposal_path.exists():
        from critiq.structured import RefineProposal
        proposal = RefineProposal.from_dict(load_json(proposal_path))
    else:
        try:
            proposal = manager.generate(
                node=rubric.get_node(TARGET_NODE_ID), evidence=evidence,
                representative_rows=representative_rows, prior_failures=(),
                rubric_memory=memory,
                max_description_chars=settings["max_description_chars"])
        except RefineManagerFailure as exc:
            _write(attempt / "proposal_failure.json", exc.to_dict())
            status = load_json(target / "stage_status.json")
            status["smoke"] = {"status": "paused" if
                               split._manager_failure_kind(exc)
                               == split.TRANSPORT_FAILED else "failed",
                               "details": {"stage": "refine_generation",
                                           "history_appended": False,
                                           **exc.to_dict()}}
            _write(target / "stage_status.json", status)
            raise
        _write(proposal_path, proposal.to_dict())
    candidate = build_refine_candidate(EvolutionContext(rubric, feedback),
                                       TARGET_NODE_ID, proposal)
    _write(attempt / "candidate.json", candidate.to_dict())
    after = apply_rubric_patch(rubric, candidate.edit_candidate.patch)
    after.save_json(attempt / "candidate_rubric.json")
    node = after.get_node(TARGET_NODE_ID)
    node_rubric = StructuredRubric({TARGET_NODE_ID: node}, (), (TARGET_NODE_ID,))
    pool = base._single_endpoint_execution_pool(config, PAIRWISE_ENDPOINT)
    candidate_prediction, _, valid = base._generate_pairwise(
        config, attempt, node_rubric, rows, "candidate_pairwise",
        execution_backend_pool=pool,
        request_backend_id=prediction.request_spec.backend_id,
        request_level_progress=True)
    candidate_prediction.save_json(attempt / "candidate_pairwise.json")
    combined = assemble_refined_pairwise_prediction(
        prediction, candidate_prediction, after, TARGET_NODE_ID)
    combined.save_json(attempt / "combined_pairwise.json")
    evaluation, _, _ = evaluate_refine_candidate(
        before_rubric=rubric, after_rubric=after,
        before_prediction=prediction, combined_prediction=combined,
        candidate_prediction=candidate_prediction, dataset=rows,
        node_id=TARGET_NODE_ID,
        min_support=config["evolution_policy"]["trigger_thresholds"]["N_min_support"])
    _write(attempt / "node_evaluation.json", evaluation.to_dict())
    _write(attempt / "description_diagnostic.json", {
        "old_length": len(rubric.get_node(TARGET_NODE_ID).criterion.description),
        "new_length": len(node.criterion.description),
        "length_delta": (len(node.criterion.description)
                         - len(rubric.get_node(TARGET_NODE_ID).criterion.description)),
    })
    _write(attempt / "subtree_diagnostic.json", {
        "root_node_id": evaluation.root_node_id,
        "scope_support": evaluation.subtree_scope_support,
        "old_accuracy": evaluation.old_subtree_accuracy,
        "new_accuracy": evaluation.new_subtree_accuracy,
        "delta": evaluation.subtree_accuracy_delta,
        "old_coverage": evaluation.old_subtree_coverage,
        "new_coverage": evaluation.new_subtree_coverage,
        "corrected_sample_ids": list(evaluation.subtree_corrected_sample_ids),
        "harmed_sample_ids": list(evaluation.subtree_harmed_sample_ids)})
    _write(attempt / "m1_diagnostic.json", {
        "old_accuracy": evaluation.old_m1_accuracy,
        "new_accuracy": evaluation.new_m1_accuracy,
        "delta": evaluation.m1_accuracy_delta,
        "old_coverage": evaluation.old_m1_coverage,
        "new_coverage": evaluation.new_m1_coverage,
        "corrected_sample_ids": list(evaluation.m1_corrected_sample_ids),
        "harmed_sample_ids": list(evaluation.m1_harmed_sample_ids)})
    changed = _changed_predictions(prediction, combined, rows, TARGET_CRITERION)
    _write(attempt / "changed_predictions.json", changed)
    attribution = None
    if evaluation.decision is EvolutionDecision.REJECT:
        attribution = _required_attribution(
            manager, attempt, original_node=rubric.get_node(TARGET_NODE_ID),
            proposal=proposal, evaluation=evaluation, changed_predictions=changed,
            rubric_memory=memory)
    report = {"schema_version": "1.0.0", "status": "completed",
              "experiment": "forced_refine_smoke", "forced": True,
              "automatic_trigger": manifest["automatic_trigger"],
              "decision": evaluation.decision.value,
              "node_accuracy_old": evaluation.old_node.accuracy,
              "node_accuracy_new": evaluation.new_node.accuracy,
              "node_accuracy_delta": evaluation.node_accuracy_delta,
              "support_old": evaluation.old_node.support,
              "support_new": evaluation.new_node.support,
              "subtree_accuracy_old": evaluation.old_subtree_accuracy,
              "subtree_accuracy_new": evaluation.new_subtree_accuracy,
              "m1_accuracy_old": evaluation.old_m1_accuracy,
              "m1_accuracy_new": evaluation.new_m1_accuracy,
              "candidate_valid_rate": valid,
              "proposal_sha256": canonical_sha256(proposal.to_dict()),
              "candidate_rubric_sha256": after.rubric_sha256,
              "failure_attribution": attribution,
              "heldout_accessed": False, "formal_evolution_mutated": False}
    _write(report_path, report)
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": "passed", "details": report}
    _write(target / "stage_status.json", status)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def heldout(config: Mapping[str, Any], output: Path) -> None:
    _config(config); _validate_output(output); target = _target(output)
    report_path = target / "heldout500" / "report.json"
    if report_path.exists():
        print("refine-smoke-heldout already completed"); return
    smoke_dir = target / "smoke" / "attempt_01"
    if not (smoke_dir / "discovery_report.json").exists():
        raise RuntimeError("run refine-smoke first")
    manifest = load_json(target / "frozen_manifest.json")
    source = _source(output); source_heldout = source / "heldout500" / "combined_pairwise.json"
    source_manifest = load_json(source / "heldout500" / "frozen_manifest.json")
    rows = _rows(config, "heldout")
    if (file_sha256(base._path(config["heldout_dataset"])).lower()
            != manifest["heldout_dataset_sha256"].lower()):
        raise RuntimeError("heldout dataset changed after Refine freeze")
    rubric = StructuredRubric.load_json(_manifest_path(manifest, "source_rubric_path"))
    before = PairwisePredictionOutput.load_json(source_heldout)
    frozen_pairwise = DualWorkerRequestSpec.from_dict(manifest["pairwise_request_spec"])
    if (not _same_pairwise_scientific_identity(before.request_spec, frozen_pairwise)
            or source_manifest["final_rubric_sha256"] != rubric.rubric_sha256):
        raise RuntimeError("source heldout identity mismatch")
    candidate = RefineCandidate.from_dict(load_json(smoke_dir / "candidate.json"))
    after = apply_rubric_patch(rubric, candidate.edit_candidate.patch)
    if after.rubric_sha256 != load_json(smoke_dir / "discovery_report.json")["candidate_rubric_sha256"]:
        raise RuntimeError("Refine candidate changed before heldout")
    heldout_dir = target / "heldout500"
    _write(heldout_dir / "frozen_manifest.json", {
        "schema_version": "1.0.0", "candidate_rubric_sha256": after.rubric_sha256,
        "proposal_sha256": canonical_sha256(candidate.proposal.to_dict()),
        "heldout_dataset_sha256": manifest["heldout_dataset_sha256"],
        "source_pairwise_sha256": file_sha256(source_heldout),
        "pairwise_request_spec": manifest["pairwise_request_spec"],
        "diagnostic_only": True, "selection_after_heldout_forbidden": True})
    node = after.get_node(TARGET_NODE_ID)
    node_rubric = StructuredRubric({TARGET_NODE_ID: node}, (), (TARGET_NODE_ID,))
    pool = base._single_endpoint_execution_pool(config, PAIRWISE_ENDPOINT)
    candidate_prediction, _, valid = base._generate_pairwise(
        config, heldout_dir, node_rubric, rows, "candidate_pairwise",
        execution_backend_pool=pool, request_backend_id=before.request_spec.backend_id,
        request_level_progress=True)
    candidate_prediction.save_json(heldout_dir / "candidate_pairwise.json")
    combined = assemble_refined_pairwise_prediction(before, candidate_prediction,
                                                     after, TARGET_NODE_ID)
    combined.save_json(heldout_dir / "combined_pairwise.json")
    evaluation, _, _ = evaluate_refine_candidate(
        before_rubric=rubric, after_rubric=after, before_prediction=before,
        combined_prediction=combined, candidate_prediction=candidate_prediction,
        dataset=rows, node_id=TARGET_NODE_ID,
        min_support=config["evolution_policy"]["trigger_thresholds"]["N_min_support"])
    _write(heldout_dir / "node_evaluation.json", evaluation.to_dict())
    report = {"schema_version": "1.0.0", "diagnostic_only": True,
              "discovery_decision": load_json(smoke_dir / "discovery_report.json")["decision"],
              "heldout_node_accuracy_old": evaluation.old_node.accuracy,
              "heldout_node_accuracy_new": evaluation.new_node.accuracy,
              "heldout_node_accuracy_delta": evaluation.node_accuracy_delta,
              "heldout_support_old": evaluation.old_node.support,
              "heldout_support_new": evaluation.new_node.support,
              "heldout_subtree_accuracy_old": evaluation.old_subtree_accuracy,
              "heldout_subtree_accuracy_new": evaluation.new_subtree_accuracy,
              "heldout_subtree_delta": evaluation.subtree_accuracy_delta,
              "heldout_m1_accuracy_old": evaluation.old_m1_accuracy,
              "heldout_m1_accuracy_new": evaluation.new_m1_accuracy,
              "heldout_m1_delta": evaluation.m1_accuracy_delta,
              "candidate_valid_rate": valid,
              "candidate_rubric_sha256": after.rubric_sha256,
              "selection_after_heldout_forbidden": True}
    _write(report_path, report)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": report}
    _write(target / "stage_status.json", status)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def report(config: Mapping[str, Any], output: Path) -> None:
    _config(config); target = _target(output)
    discovery_path = target / "smoke" / "attempt_01" / "discovery_report.json"
    if not discovery_path.exists():
        raise RuntimeError("run refine-smoke first")
    discovery = load_json(discovery_path)
    heldout_path = target / "heldout500" / "report.json"
    heldout_report = load_json(heldout_path) if heldout_path.exists() else None
    value = {"schema_version": "1.0.0", "discovery": discovery,
             "heldout": heldout_report,
             "go_no_go": _smoke_go_no_go(discovery, heldout_report)}
    _write(target / "report.json", value)
    lines = ["# Refine v1 Forced Smoke Report", "",
             f"- Discovery decision: `{discovery['decision']}`",
             f"- Node ACC: `{discovery['node_accuracy_old']:.4f}` → `{discovery['node_accuracy_new']:.4f}`",
             f"- Root-subtree ACC: `{discovery['subtree_accuracy_old']:.4f}` → `{discovery['subtree_accuracy_new']:.4f}`",
             f"- Full M1 ACC: `{discovery['m1_accuracy_old']:.4f}` → `{discovery['m1_accuracy_new']:.4f}`",
             f"- Go/no-go: `{value['go_no_go']}`"]
    if heldout_report:
        lines.extend(["", "## Heldout-500 diagnostic", "",
                      f"- Node ACC: `{heldout_report['heldout_node_accuracy_old']:.4f}` → `{heldout_report['heldout_node_accuracy_new']:.4f}`",
                      f"- Root-subtree ACC: `{heldout_report['heldout_subtree_accuracy_old']:.4f}` → `{heldout_report['heldout_subtree_accuracy_new']:.4f}`",
                      f"- Full M1 ACC: `{heldout_report['heldout_m1_accuracy_old']:.4f}` → `{heldout_report['heldout_m1_accuracy_new']:.4f}`"])
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _visual_source_target(output: Path) -> Path:
    return output / VISUAL_SPLIT_REFINE_SOURCE_DIR


def _visual_source_paths(output: Path) -> tuple[Path, Path, Path, Path, Path]:
    source = _visual_source_target(output)
    return (
        source / "final" / "rubric_committed.json",
        source / "final" / "discovery_pairwise.json",
        source / "final" / "feedback.json",
        source / "report.json",
        source / "v2_state.json",
    )


def _visual_child_ids(rubric: StructuredRubric) -> tuple[str, ...]:
    if VISUAL_SPLIT_REFINE_ROOT_ID not in rubric.nodes:
        raise RuntimeError("Visual Grounding root is missing from local Refine source")
    child_ids = tuple(edge.child_id for edge in rubric.child_edges(
        VISUAL_SPLIT_REFINE_ROOT_ID))
    if len(child_ids) != 4 or len(set(child_ids)) != 4:
        raise RuntimeError("local Visual Refine requires exactly four direct children")
    return child_ids


def _visual_subtree_rubric(
    rubric: StructuredRubric,
    child_ids: Sequence[str],
) -> StructuredRubric:
    node_ids = (VISUAL_SPLIT_REFINE_ROOT_ID, *child_ids)
    return StructuredRubric(
        nodes={node_id: rubric.get_node(node_id) for node_id in node_ids},
        edges=tuple(
            edge for edge in rubric.edges
            if edge.parent_id == VISUAL_SPLIT_REFINE_ROOT_ID
            and edge.child_id in child_ids),
        root_ids=(VISUAL_SPLIT_REFINE_ROOT_ID,),
    )


def _visual_variant_rubric(
    base_rubric: StructuredRubric,
    child_source: StructuredRubric,
    child_ids: Sequence[str],
) -> StructuredRubric:
    nodes = dict(base_rubric.nodes)
    nodes.update({node_id: child_source.get_node(node_id) for node_id in child_ids})
    edges = tuple(base_rubric.edges) + tuple(
        edge for edge in child_source.edges
        if edge.parent_id == VISUAL_SPLIT_REFINE_ROOT_ID
        and edge.child_id in child_ids)
    return StructuredRubric(nodes=nodes, edges=edges, root_ids=base_rubric.root_ids)


def _visual_trigger_rows(
    rubric: StructuredRubric,
    feedback: RubricFeedback,
    config: Mapping[str, Any],
    child_ids: Sequence[str],
) -> list[dict[str, Any]]:
    context = EvolutionContext(rubric, feedback)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    result = []
    for node_id in child_ids:
        decision = detect_refine_trigger(
            context, node_id, thresholds, trigger_mode="role_aware_v2")
        result.append({
            "node_id": node_id,
            "criterion_name": rubric.get_node(node_id).criterion.name,
            "triggered": decision.triggered,
            "reasons": list(decision.reasons),
            "accuracy": decision.accuracy,
            "coverage": decision.coverage,
            "support": decision.support,
            "wrong": decision.wrong,
        })
    return result


def _visual_schedule(
    rubric: StructuredRubric,
    feedback: RubricFeedback,
    config: Mapping[str, Any],
    child_ids: Sequence[str],
) -> tuple[str, ...]:
    return tuple(item["node_id"] for item in _visual_trigger_rows(
        rubric, feedback, config, child_ids) if item["triggered"])


def _visual_diagnostic(
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    rows: Sequence[Mapping[str, Any]],
    child_ids: Sequence[str],
) -> dict[str, Any]:
    subtree = _visual_subtree_rubric(rubric, child_ids)
    subtree_prediction = project_pairwise_prediction(prediction, subtree)
    _, subtree_votes = execute_offline_m1(subtree, subtree_prediction, rows)
    parent_name = rubric.get_node(VISUAL_SPLIT_REFINE_ROOT_ID).criterion.name
    parent_outputs = [row[parent_name] for row in prediction.node_outputs]
    parent_scope = [
        output.parse_ok and output.answer_valid and output.vote in {Vote.A, Vote.B}
        for output in parent_outputs
    ]
    indices = [index for index, active in enumerate(parent_scope) if active]
    local_rows = [rows[index] for index in indices]
    names = [rubric.get_node(node_id).criterion.name for node_id in child_ids]
    child_prediction = project_pairwise_prediction(prediction, StructuredRubric(
        nodes={node_id: rubric.get_node(node_id) for node_id in child_ids},
        edges=(), root_ids=tuple(child_ids)))
    return {
        "parent_scope_support": len(indices),
        "subtree_all_discovery": base._heldout_vote_metrics(subtree_votes, rows),
        "subtree_parent_scope": base._heldout_vote_metrics(
            [subtree_votes[index] for index in indices], local_rows),
        "children": [split._v2_child_metrics(
            child_prediction, parent_outputs, rows, name) for name in names],
        "sibling_conflicts": split._v2_conflicts(
            child_prediction, names, parent_scope),
    }


def visual_split_refine_freeze(config: Mapping[str, Any], output: Path) -> None:
    """Freeze the accepted Split v2 Visual subtree without reading heldout data."""
    settings = _visual_split_refine_config(config)
    _validate_output(output)
    _require_pairwise_endpoint(config, VISUAL_SPLIT_REFINE_ENDPOINT)
    target = _visual_split_refine_target(output)
    manifest_path = target / "frozen_manifest.json"
    if manifest_path.exists():
        print("visual-split-refine-freeze already completed")
        return
    rubric_path, prediction_path, feedback_path, report_path, state_path = (
        _visual_source_paths(output))
    required = (rubric_path, prediction_path, feedback_path, report_path, state_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Visual Split-to-Refine source artifacts are missing: {missing}")
    report_value = load_json(report_path)
    state = load_json(state_path)
    if report_value.get("decision") != "accepted" or state.get("decision") != "accepted":
        raise RuntimeError("Visual Split-to-Refine source must be an accepted Split v2 run")
    rubric = StructuredRubric.load_json(rubric_path)
    prediction = PairwisePredictionOutput.load_json(prediction_path)
    feedback = RubricFeedback.from_dict(load_json(feedback_path))
    if feedback.rubric_sha256 != rubric.rubric_sha256:
        raise RuntimeError("Visual source feedback/rubric mismatch")
    child_ids = _visual_child_ids(rubric)
    child_names = [rubric.get_node(node_id).criterion.name for node_id in child_ids]
    locked_names = list(state.get("locked_criterion_names", ()))
    if len(locked_names) != 1 or locked_names[0] not in child_names:
        raise RuntimeError("Visual Split v2 locked-child state is missing or inconsistent")
    rows = _rows(config, "discovery")
    expected_worker = base._expected_pairwise_request_spec(config, rows)
    if not _same_pairwise_scientific_identity(
            prediction.request_spec, expected_worker):
        raise RuntimeError("Visual source Pairwise scientific identity differs from config")
    if file_sha256(base._path(config["discovery_dataset"])) != config[
            "discovery_dataset_sha256"]:
        raise RuntimeError("Visual Split-to-Refine discovery dataset hash mismatch")
    manager, manager_profile = _manager(config)
    trigger_rows = _visual_trigger_rows(rubric, feedback, config, child_ids)
    if not all(item["triggered"] for item in trigger_rows):
        raise RuntimeError("all four source Visual children must meet role-aware Refine trigger")
    manifest = {
        "schema_version": "1.0.0",
        "stage": "visual_split_refine_local_v1",
        "settings": settings,
        "source_rubric_path": str(rubric_path),
        "source_rubric_sha256": rubric.rubric_sha256,
        "source_prediction_path": str(prediction_path),
        "source_prediction_sha256": file_sha256(prediction_path),
        "source_feedback_path": str(feedback_path),
        "source_feedback_sha256": file_sha256(feedback_path),
        "source_split_report_sha256": file_sha256(report_path),
        "source_state_sha256": file_sha256(state_path),
        "root_id": VISUAL_SPLIT_REFINE_ROOT_ID,
        "child_node_ids": list(child_ids),
        "child_criterion_names": child_names,
        "locked_child_criterion_names": locked_names,
        "source_child_descriptions": {
            node_id: rubric.get_node(node_id).criterion.description
            for node_id in child_ids},
        "source_trigger_audit": trigger_rows,
        "discovery_dataset_sha256": file_sha256(base._path(config["discovery_dataset"])),
        "pairwise_request_spec": prediction.request_spec.to_dict(),
        "manager_profile": manager_profile,
        "manager_request_specs": {
            key: value.to_dict() for key, value in manager.request_specs().items()},
        "heldout_accessed": False,
        "split_operations_permitted": False,
    }
    _write(manifest_path, manifest)
    epoch0 = split._epoch(target, 0)
    votes, replayed_feedback = split._snapshot(
        epoch0, rubric, prediction, rows, manifest)
    if replayed_feedback.to_dict() != feedback.to_dict():
        raise RuntimeError("Visual Split-to-Refine source feedback replay mismatch")
    _write(epoch0 / "summary.json", {
        "epoch": 0,
        "source_experiment": VISUAL_SPLIT_REFINE_SOURCE_DIR,
        "refine_scheduled": list(child_ids),
        "m1": base._metrics(votes, rows),
        "visual_diagnostic": _visual_diagnostic(rubric, prediction, rows, child_ids),
    })
    _write(target / "evolution_history.json", {
        "schema_version": "1.0.0",
        "current_epoch": 0,
        "completed": False,
        "stop_reason": None,
        "candidate_scope": list(child_ids),
        "locked_child_criterion_names": locked_names,
        "refine_states": {},
        "refine_attempts": [],
    })
    _write(target / "stage_status.json", {"freeze": {"status": "passed"}})
    print(json.dumps({"target": str(target), "children": child_names,
                      "locked_children": locked_names}, indent=2))


def visual_split_refine_audit(config: Mapping[str, Any], output: Path) -> None:
    """Write the frozen child-only eligibility audit without accessing heldout."""
    _visual_split_refine_config(config)
    _validate_output(output)
    target = _visual_split_refine_target(output)
    manifest_path = target / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("run visual-split-refine-freeze first")
    manifest = load_json(manifest_path)
    rubric = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_committed.json")
    feedback = RubricFeedback.from_dict(load_json(split._epoch(target, 0) / "feedback.json"))
    child_ids = tuple(manifest["child_node_ids"])
    value = {
        "schema_version": "1.0.0",
        "offline_only": True,
        "root_id": VISUAL_SPLIT_REFINE_ROOT_ID,
        "split_enabled": False,
        "children": _visual_trigger_rows(rubric, feedback, config, child_ids),
        "locked_child_criterion_names": manifest["locked_child_criterion_names"],
        "heldout_accessed": False,
    }
    if not all(item["triggered"] for item in value["children"]):
        raise RuntimeError("frozen Visual child eligibility audit drifted")
    _write(target / "eligibility_audit.json", value)
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": {"eligible": 4}}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _verify_visual_split_refine_run(
    config: Mapping[str, Any],
    output: Path,
) -> tuple[Path, Mapping[str, Any], RefineManager]:
    settings = _visual_split_refine_config(config)
    _validate_output(output)
    if settings["worker_endpoint"] != VISUAL_SPLIT_REFINE_ENDPOINT:
        raise RuntimeError("Visual Split-to-Refine must execute Pairwise through vllm-8000")
    target = _visual_split_refine_target(output)
    manifest_path = target / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("run visual-split-refine-freeze first")
    manifest = load_json(manifest_path)
    if manifest.get("settings") != settings:
        raise RuntimeError("Visual Split-to-Refine protocol drifted after freeze")
    rows = _rows(config, "discovery")
    current_worker = base._expected_pairwise_request_spec(config, rows)
    frozen_worker = DualWorkerRequestSpec.from_dict(manifest["pairwise_request_spec"])
    if not _same_pairwise_scientific_identity(frozen_worker, current_worker):
        raise RuntimeError("Visual Split-to-Refine Pairwise scientific identity drift")
    manager, _ = _manager(config)
    if {key: value.to_dict() for key, value in manager.request_specs().items()} != \
            manifest["manager_request_specs"]:
        raise RuntimeError("Visual Split-to-Refine Manager request identity drift")
    return target, manifest, manager


def visual_split_refine_run(config: Mapping[str, Any], output: Path) -> None:
    """Refine only the frozen Visual children for at most three epochs."""
    target, manifest, manager = _verify_visual_split_refine_run(config, output)
    history = load_json(target / "evolution_history.json")
    if history.get("completed"):
        print("visual-split-refine-run already completed")
        return
    rows = _rows(config, "discovery")
    pool = base._single_endpoint_execution_pool(config, VISUAL_SPLIT_REFINE_ENDPOINT)
    child_ids = tuple(manifest["child_node_ids"])
    for epoch_no in range(history["current_epoch"] + 1,
                          manifest["settings"]["max_epochs"] + 1):
        started_epoch = time.monotonic()
        previous = split._epoch(target, epoch_no - 1)
        epoch_dir = split._epoch(target, epoch_no)
        rubric = StructuredRubric.load_json(previous / "rubric_committed.json")
        prediction = PairwisePredictionOutput.load_json(previous / "discovery_pairwise.json")
        feedback = RubricFeedback.from_dict(load_json(previous / "feedback.json"))
        if _visual_child_ids(rubric) != child_ids:
            raise RuntimeError("Visual Refine may not add, remove, or reorder Split children")
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        scheduled = _visual_schedule(rubric, feedback, config, child_ids)
        print(f"visual-split-refine epoch={epoch_no} scheduled={list(scheduled)}",
              flush=True)
        results: dict[str, Any] = {}
        started: dict[str, float] = {}
        try:
            for node_id in scheduled:
                started[node_id] = time.monotonic()
                attempt_no = history["refine_states"].get(
                    node_id, {"attempt_count": 0})["attempt_count"] + 1
                results[node_id] = _prepare_refine_attempt(
                    config=config, epoch_dir=epoch_dir, node_id=node_id,
                    attempt_no=attempt_no, rubric=rubric, prediction=prediction,
                    feedback=feedback, rows=rows, history=history, manager=manager,
                    pool=pool, rubric_memory=memory,
                    rubric_memory_sha256=memory_hash,
                    trigger_mode="role_aware_v2")
        except RefineTransportPause as exc:
            status = load_json(target / "stage_status.json")
            status["run"] = {"status": "paused", "details": {
                "epoch": epoch_no, "message": str(exc), "history_appended": False}}
            _write(target / "stage_status.json", status)
            return
        except RefineAttributionInvalid as exc:
            status = load_json(target / "stage_status.json")
            status["run"] = {"status": "failed", "details": {
                "epoch": epoch_no, "outcome": "attribution_invalid",
                "message": str(exc), "history_appended": False}}
            _write(target / "stage_status.json", status)
            return
        accepted = {
            node_id: result["candidate"] for node_id, result in results.items()
            if result["decision"] == "accepted"}
        committed = _merge_epoch_rubric(rubric, {}, accepted)
        committed_prediction = _merge_epoch_predictions(
            prediction, [], {
                node_id: results[node_id]["candidate_prediction"]
                for node_id in accepted}, committed)
        votes, committed_feedback = split._snapshot(
            epoch_dir, committed, committed_prediction, rows, manifest)
        records = [_record_refine_state(
            history, results[node_id], epoch_no,
            time.monotonic() - started[node_id]) for node_id in scheduled]
        history["current_epoch"] = epoch_no
        next_scheduled = _visual_schedule(
            committed, committed_feedback, config, child_ids)
        if not next_scheduled:
            history.update({"completed": True, "stop_reason": "no_eligible_children"})
        elif epoch_no == manifest["settings"]["max_epochs"]:
            history.update({"completed": True, "stop_reason": "max_epochs_reached"})
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no,
            "split_enabled": False,
            "rubric_memory_sha256": memory_hash,
            "refine_scheduled": list(scheduled),
            "accepted_refine_nodes": sorted(accepted),
            "refine_attempts": records,
            "next_refine_eligible": list(next_scheduled),
            "m1": base._metrics(votes, rows),
            "visual_diagnostic": _visual_diagnostic(
                committed, committed_prediction, rows, child_ids),
            "epoch_wall_seconds": time.monotonic() - started_epoch,
        })
        _write(target / "evolution_history.json", history)
        print(f"visual-split-refine epoch={epoch_no} "
              f"accepted={sorted(accepted)} m1_acc={base._metrics(votes, rows)['accuracy']:.4f}",
              flush=True)
        if history["completed"]:
            break
    status = load_json(target / "stage_status.json")
    status["run"] = {"status": "passed" if history["completed"] else "partial",
                     "details": {"current_epoch": history["current_epoch"],
                                 "stop_reason": history["stop_reason"]}}
    _write(target / "stage_status.json", status)


def visual_split_refine_report(config: Mapping[str, Any], output: Path) -> None:
    """Freeze the discovery-selected child-only Refine result."""
    _visual_split_refine_config(config)
    _validate_output(output)
    target = _visual_split_refine_target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run visual-split-refine-run to completion first")
    manifest = load_json(target / "frozen_manifest.json")
    initial_epoch = split._epoch(target, 0)
    final_epoch = split._epoch(target, history["current_epoch"])
    initial_rubric = StructuredRubric.load_json(initial_epoch / "rubric_committed.json")
    final_rubric = StructuredRubric.load_json(final_epoch / "rubric_committed.json")
    final_prediction = PairwisePredictionOutput.load_json(
        final_epoch / "discovery_pairwise.json")
    rows = _rows(config, "discovery")
    execution, answers = execute_offline_m1(final_rubric, final_prediction, rows)
    initial_summary = load_json(initial_epoch / "summary.json")
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_rubric.save_json(final_dir / "rubric.json")
    final_prediction.save_json(final_dir / "discovery_pairwise.json")
    execution.save_json(final_dir / "m1_execution.json")
    report_value = {
        "schema_version": "1.0.0",
        "experiment": VISUAL_SPLIT_REFINE_EXPERIMENT_DIR,
        "root_id": VISUAL_SPLIT_REFINE_ROOT_ID,
        "split_enabled": False,
        "final_epoch": history["current_epoch"],
        "initial_rubric_sha256": initial_rubric.rubric_sha256,
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "initial_m1": initial_summary["m1"],
        "final_m1": base._metrics(answers, rows),
        "m1_accuracy_delta": base._metrics(answers, rows)["accuracy"]
        - initial_summary["m1"]["accuracy"],
        "locked_child_criterion_names": manifest["locked_child_criterion_names"],
        "refine_attempts": history["refine_attempts"],
        "accepted_refine_count": sum(
            item["decision"] == "accepted" for item in history["refine_attempts"]),
        "rejected_refine_count": sum(
            item["decision"] == "competition_rejected"
            for item in history["refine_attempts"]),
        "initial_visual_diagnostic": initial_summary["visual_diagnostic"],
        "final_visual_diagnostic": _visual_diagnostic(
            final_rubric, final_prediction, rows, manifest["child_node_ids"]),
        "heldout_accessed": False,
    }
    _write(final_dir / "discovery_report.json", report_value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "final_m1_accuracy": report_value["final_m1"]["accuracy"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(report_value, indent=2, ensure_ascii=False))


def _visual_heldout_system(
    target: Path,
    label: str,
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    rows: Sequence[Mapping[str, Any]],
    child_ids: Sequence[str],
) -> tuple[dict[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    system_dir = target / "systems" / label
    system_dir.mkdir(parents=True, exist_ok=True)
    prediction.save_json(system_dir / "combined_pairwise.json")
    execution, votes = execute_offline_m1(rubric, prediction, rows)
    execution.save_json(system_dir / "m1_execution.json")
    subtree = _visual_subtree_rubric(rubric, child_ids)
    subtree_prediction = project_pairwise_prediction(prediction, subtree)
    subtree_execution, subtree_votes = execute_offline_m1(subtree, subtree_prediction, rows)
    subtree_execution.save_json(system_dir / "subtree_execution.json")
    parent_name = rubric.get_node(VISUAL_SPLIT_REFINE_ROOT_ID).criterion.name
    parent_outputs = [row[parent_name] for row in prediction.node_outputs]
    indices = [index for index, output in enumerate(parent_outputs)
               if output.parse_ok and output.answer_valid
               and output.vote in {Vote.A, Vote.B}]
    return ({
        "m1": base._heldout_vote_metrics(votes, rows),
        "subtree_all500": base._heldout_vote_metrics(subtree_votes, rows),
        "subtree_parent_scope": base._heldout_vote_metrics(
            [subtree_votes[index] for index in indices],
            [rows[index] for index in indices]),
    }, tuple(votes), tuple(subtree_votes))


def _validate_visual_changed_prediction(
    generated: PairwisePredictionOutput,
    source: PairwisePredictionOutput,
    final_rubric: StructuredRubric,
    changed_ids: Sequence[str],
) -> None:
    expected = tuple(final_rubric.get_node(node_id).criterion for node_id in changed_ids)
    actual = tuple(generated.criteria)
    if tuple((item.name, item.description) for item in actual) != tuple(
            (item.name, item.description) for item in expected):
        raise RuntimeError("heldout changed-child predictions do not match frozen descriptions")
    if (generated.sample_ids != source.sample_ids
            or generated.sample_fingerprints != source.sample_fingerprints
            or generated.request_spec != source.request_spec):
        raise RuntimeError("heldout changed-child prediction identity drift")


def _merge_visual_changed_predictions(
    outputs: Sequence[PairwisePredictionOutput],
    final_rubric: StructuredRubric,
    changed_ids: Sequence[str],
) -> PairwisePredictionOutput:
    """Merge resumed heldout columns without repeating completed requests."""

    if not outputs:
        raise ValueError("at least one changed-child prediction is required")
    reference = outputs[0]
    names = [final_rubric.get_node(node_id).criterion.name for node_id in changed_ids]
    descriptions = {
        final_rubric.get_node(node_id).criterion.name:
        final_rubric.get_node(node_id).criterion.description
        for node_id in changed_ids}
    columns: dict[str, tuple[Any, ...]] = {}
    for output in outputs:
        if (output.sample_ids != reference.sample_ids
                or output.sample_fingerprints != reference.sample_fingerprints
                or output.request_spec != reference.request_spec):
            raise RuntimeError("resumed changed-child prediction identity drift")
        for criterion in output.criteria:
            if (criterion.name not in descriptions
                    or criterion.description != descriptions[criterion.name]
                    or criterion.name in columns):
                raise RuntimeError("resumed changed-child prediction column drift")
            columns[criterion.name] = tuple(
                row[criterion.name] for row in output.node_outputs)
    if set(columns) != set(names):
        raise RuntimeError("resumed changed-child predictions are incomplete")
    node_outputs = tuple({name: columns[name][index] for name in names}
                         for index in range(len(reference.sample_ids)))
    answers = tuple(aggregate_flat_votes(item.vote for item in row.values())
                    for row in node_outputs)
    return PairwisePredictionOutput(
        reference.sample_ids, reference.sample_fingerprints,
        tuple(StructuredCriterionSnapshot(name, descriptions[name]) for name in names),
        node_outputs, answers, reference.request_spec)


def visual_split_refine_heldout(config: Mapping[str, Any], output: Path) -> None:
    """Run the one final exploratory heldout diagnostic through vllm-8000."""
    settings = _visual_split_refine_config(config)
    _validate_output(output)
    target = _visual_split_refine_target(output)
    discovery_path = target / "final" / "discovery_report.json"
    if not discovery_path.exists():
        raise RuntimeError("run visual-split-refine-report first")
    manifest = load_json(target / "frozen_manifest.json")
    final_rubric = StructuredRubric.load_json(target / "final" / "rubric.json")
    if load_json(discovery_path)["final_rubric_sha256"] != final_rubric.rubric_sha256:
        raise RuntimeError("Visual Split-to-Refine final rubric drift")
    source_rubric = StructuredRubric.load_json(
        _manifest_path(manifest, "source_rubric_path"))
    source_heldout_path = (_visual_source_target(output) / "heldout500_diagnostic"
                           / "systems" / "full_v2" / "combined_pairwise.json")
    if not source_heldout_path.exists():
        raise RuntimeError("Visual Split v2 full-child heldout artifact is missing")
    source_prediction = PairwisePredictionOutput.load_json(source_heldout_path)
    rows = _rows(config, "heldout")
    expected_worker = base._expected_pairwise_request_spec(config, rows)
    if not _same_pairwise_scientific_identity(
            source_prediction.request_spec, expected_worker):
        raise RuntimeError("Visual heldout Pairwise scientific identity drift")
    child_ids = tuple(manifest["child_node_ids"])
    source_descriptions = {
        criterion.name: criterion.description for criterion in source_prediction.criteria}
    changed_ids = tuple(
        node_id for node_id in final_rubric.preorder_node_ids()
        if source_descriptions.get(final_rubric.get_node(node_id).criterion.name)
        != final_rubric.get_node(node_id).criterion.description)
    heldout_dir = target / "heldout500"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    heldout_manifest = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "selection_after_heldout_forbidden": True,
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "source_rubric_sha256": source_rubric.rubric_sha256,
        "source_prediction_sha256": file_sha256(source_heldout_path),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "pairwise_request_spec": source_prediction.request_spec.to_dict(),
        "worker_endpoint": settings["worker_endpoint"],
        "changed_node_ids": list(changed_ids),
        "generated_request_count": len(changed_ids) * len(rows),
    }
    frozen_path = heldout_dir / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != heldout_manifest:
        if (heldout_dir / "final_combined_pairwise.json").exists():
            raise RuntimeError("Visual Split-to-Refine heldout manifest drift after combination")
        _write(heldout_dir / "manifest_repair.json", {
            "reason": "source_prediction_description_delta_was_undercounted",
            "previous": load_json(frozen_path), "replacement": heldout_manifest})
    _write(frozen_path, heldout_manifest)
    generated = None
    generated_path = heldout_dir / "changed_child_predictions.json"
    if changed_ids:
        reusable: list[PairwisePredictionOutput] = []
        if generated_path.exists():
            existing = PairwisePredictionOutput.load_json(generated_path)
            existing_names = {item.name for item in existing.criteria}
            expected_names = {
                final_rubric.get_node(node_id).criterion.name for node_id in changed_ids}
            if not existing_names <= expected_names:
                raise RuntimeError("existing changed-child artifact has unexpected criteria")
            reusable.append(existing)
        present_names = {
            criterion.name for output in reusable for criterion in output.criteria}
        missing_ids = tuple(
            node_id for node_id in changed_ids
            if final_rubric.get_node(node_id).criterion.name not in present_names)
        if missing_ids:
            changed_rubric = StructuredRubric(
                nodes={node_id: final_rubric.get_node(node_id) for node_id in missing_ids},
                edges=(), root_ids=missing_ids)
            additional, _, _ = base._generate_pairwise(
                config, heldout_dir, changed_rubric, rows, "changed_children_missing",
                execution_backend_pool=base._single_endpoint_execution_pool(
                    config, VISUAL_SPLIT_REFINE_ENDPOINT),
                request_backend_id=source_prediction.request_spec.backend_id,
                request_level_progress=True)
            reusable.append(additional)
            additional.save_json(heldout_dir / "changed_child_predictions_additional.json")
        generated = _merge_visual_changed_predictions(
            reusable, final_rubric, changed_ids)
        generated.save_json(generated_path)
        _validate_visual_changed_prediction(
            generated, source_prediction, final_rubric, changed_ids)
    final_prediction = _combine_heldout_predictions(
        source_prediction, generated, final_rubric)
    final_prediction.save_json(heldout_dir / "final_combined_pairwise.json")

    # The Phase 8 heldout artifact may predate a non-Visual description update
    # already present in the committed source rubric.  Reconcile that artifact
    # independently from the final Refine rubric: descriptions unchanged in the
    # source rubric keep their original outputs, while stale source columns reuse
    # the newly generated description-matched outputs.  Without this projection,
    # OfflinePairwiseVoteBackend correctly rejects the source system because its
    # criterion snapshots do not match the source rubric.
    source_prediction_reconciled = _combine_heldout_predictions(
        source_prediction, generated, source_rubric)
    source_prediction_reconciled.save_json(
        heldout_dir / "source_combined_pairwise_reconciled.json")

    base_rubric, base_prediction = split._v2_global_source(output)
    if not _same_pairwise_scientific_identity(
            base_prediction.request_spec, source_prediction.request_spec):
        raise RuntimeError("Visual heldout base/full prediction identity mismatch")
    locked_names = set(manifest["locked_child_criterion_names"])
    locked_ids = tuple(node_id for node_id in child_ids
                       if final_rubric.get_node(node_id).criterion.name in locked_names)
    if len(locked_ids) != 1:
        raise RuntimeError("frozen locked child is missing from final Visual subtree")

    source_locked_rubric = _visual_variant_rubric(base_rubric, source_rubric, locked_ids)
    source_locked_children = project_pairwise_prediction(
        source_prediction, StructuredRubric(
            nodes={node_id: source_rubric.get_node(node_id) for node_id in locked_ids},
            edges=(), root_ids=locked_ids))
    source_locked_prediction = split.merge_predictions(
        base_prediction, [source_locked_children], source_locked_rubric)
    final_locked_rubric = _visual_variant_rubric(base_rubric, final_rubric, locked_ids)
    final_locked_children = project_pairwise_prediction(
        final_prediction, StructuredRubric(
            nodes={node_id: final_rubric.get_node(node_id) for node_id in locked_ids},
            edges=(), root_ids=locked_ids))
    final_locked_prediction = split.merge_predictions(
        base_prediction, [final_locked_children], final_locked_rubric)

    systems = {
        "parent_only": (base_rubric, base_prediction, ()),
        "locked_only_source": (source_locked_rubric, source_locked_prediction, locked_ids),
        "split_v2_full_children": (
            source_rubric, source_prediction_reconciled, child_ids),
        "locked_only_final": (final_locked_rubric, final_locked_prediction, locked_ids),
        "split_v2_refined_children": (final_rubric, final_prediction, child_ids),
    }
    metrics: dict[str, Any] = {}
    votes: dict[str, tuple[Any, ...]] = {}
    subtree_votes: dict[str, tuple[Any, ...]] = {}
    for label, (rubric, prediction, system_children) in systems.items():
        metrics[label], votes[label], subtree_votes[label] = _visual_heldout_system(
            heldout_dir, label, rubric, prediction, rows, system_children)
    parent_scope = [
        output.parse_ok and output.answer_valid and output.vote in {Vote.A, Vote.B}
        for output in [row[base_rubric.get_node(VISUAL_SPLIT_REFINE_ROOT_ID).criterion.name]
                       for row in base_prediction.node_outputs]
    ]
    indices = [index for index, active in enumerate(parent_scope) if active]
    local_rows = [rows[index] for index in indices]
    paired = {}
    for label in ("locked_only_source", "split_v2_full_children",
                  "locked_only_final", "split_v2_refined_children"):
        paired[f"{label}_vs_parent"] = {
            "subtree_parent_scope": base._paired_heldout_comparison(
                [subtree_votes["parent_only"][index] for index in indices],
                [subtree_votes[label][index] for index in indices], local_rows),
            "m1_all500": base._paired_heldout_comparison(
                votes["parent_only"], votes[label], rows),
        }
    paired["refined_vs_source_full"] = {
        "subtree_parent_scope": base._paired_heldout_comparison(
            [subtree_votes["split_v2_full_children"][index] for index in indices],
            [subtree_votes["split_v2_refined_children"][index] for index in indices],
            local_rows),
        "m1_all500": base._paired_heldout_comparison(
            votes["split_v2_full_children"], votes["split_v2_refined_children"], rows),
    }
    paired["refined_vs_locked_source"] = {
        "subtree_parent_scope": base._paired_heldout_comparison(
            [subtree_votes["locked_only_source"][index] for index in indices],
            [subtree_votes["split_v2_refined_children"][index] for index in indices],
            local_rows),
        "m1_all500": base._paired_heldout_comparison(
            votes["locked_only_source"], votes["split_v2_refined_children"], rows),
    }
    source_children = project_pairwise_prediction(
        source_prediction_reconciled, StructuredRubric(
            nodes={node_id: source_rubric.get_node(node_id) for node_id in child_ids},
            edges=(), root_ids=child_ids))
    final_children = project_pairwise_prediction(
        final_prediction, StructuredRubric(
            nodes={node_id: final_rubric.get_node(node_id) for node_id in child_ids},
            edges=(), root_ids=child_ids))
    parent_name = base_rubric.get_node(VISUAL_SPLIT_REFINE_ROOT_ID).criterion.name
    parent_outputs = [row[parent_name] for row in base_prediction.node_outputs]
    names = [final_rubric.get_node(node_id).criterion.name for node_id in child_ids]
    child_metrics = {
        "source": [split._v2_child_metrics(source_children, parent_outputs, rows, name)
                   for name in names],
        "final": [split._v2_child_metrics(final_children, parent_outputs, rows, name)
                  for name in names],
    }
    value = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "metrics": metrics,
        "paired": paired,
        "parent_scope_support": len(indices),
        "child_metrics": child_metrics,
        "source_sibling_conflicts": split._v2_conflicts(
            source_children, names, parent_scope),
        "final_sibling_conflicts": split._v2_conflicts(
            final_children, names, parent_scope),
        "changed_node_ids": list(changed_ids),
        "selection_after_heldout_forbidden": True,
    }
    _write(heldout_dir / "evaluations.json", value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        "generated_request_count": len(changed_ids) * len(rows),
        "changed_node_count": len(changed_ids)}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def visual_split_refine_final_report(config: Mapping[str, Any], output: Path) -> None:
    _visual_split_refine_config(config)
    _validate_output(output)
    target = _visual_split_refine_target(output)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout = load_json(target / "heldout500" / "evaluations.json")
    value = {
        "schema_version": "1.0.0",
        "experiment": VISUAL_SPLIT_REFINE_EXPERIMENT_DIR,
        "exploratory": True,
        "discovery": discovery,
        "heldout": heldout,
        "primary": {
            "source_full_m1_accuracy": heldout["metrics"][
                "split_v2_full_children"]["m1"]["accuracy"],
            "final_refined_m1_accuracy": heldout["metrics"][
                "split_v2_refined_children"]["m1"]["accuracy"],
            "final_vs_source_net_corrected": heldout["paired"][
                "refined_vs_source_full"]["m1_all500"]["net_corrected"],
        },
    }
    _write(target / "final_report.json", value)
    lines = ["# Visual Grounding Split-to-Refine Local v1", "",
             "Exploratory paired heldout diagnostic; no post-heldout selection.", "",
             "| System | Visual subtree ACC (parent scope) | M1 ACC |", "|---|---:|---:|"]
    for label in ("parent_only", "locked_only_source", "split_v2_full_children",
                  "locked_only_final", "split_v2_refined_children"):
        item = heldout["metrics"][label]
        lines.append(
            f"| {label} | {item['subtree_parent_scope']['accuracy']:.4f} | "
            f"{item['m1']['accuracy']:.4f} |")
    lines.extend(["", f"- Final vs source full M1 net corrected: "
                  f"{value['primary']['final_vs_source_net_corrected']}"])
    (target / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed"}
    _write(target / "stage_status.json", status)
    print(json.dumps(value["primary"], indent=2))



# Phase 10: five-root locked Split retry plus role-aware Refine.

def _integrated_retry_specs(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the dedicated locked-retry prompt identities without mutating v1."""
    managers, _, _, _ = split._managers(
        config, FIVE_ROOT_LOCKED_SPLIT_REFINE_PROTOCOL)
    child = managers["child_generation"]
    attribution = managers["semantic_cluster"]
    child.retry_feedback_mode = "locked_sample_v3"
    child.child_input_mode = "multimodal"
    attribution.retry_feedback_mode = "locked_sample_v3"
    return managers, {
        "child_generation": child.request_specs()["child_generation"].to_dict(),
        "split_failure_attribution": (
            attribution.request_specs()["split_failure_attribution"].to_dict()),
    }


def _integrated_select_locked_child(
    diagnostics: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Choose one strong child deterministically from a rejected candidate."""
    eligible = [
        item for item in diagnostics["children"]
        if item["support"] >= settings["strong_child_min_support"]
        and item["net_corrected"] >= settings["strong_child_min_net_corrected"]
    ]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda item: (
            -item["net_corrected"],
            -item["single_child_specialized_accuracy"],
            -item["support"],
            item["criterion_name"],
        ),
    )[0]
    return {
        "criterion_name": selected["criterion_name"],
        "cluster_id": selected["cluster_id"],
        "criterion_sha256": selected["criterion_sha256"],
        "support": selected["support"],
        "net_corrected": selected["net_corrected"],
        "single_child_specialized_accuracy": (
            selected["single_child_specialized_accuracy"]),
        "selection_rule": (
            "max(net_corrected, individual_specialized_accuracy, support, "
            "criterion_name)"),
    }


def _integrated_retry_feedback(
    *,
    lock: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    history: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "mode": "five_root_locked_retry_v1",
        "locked_children": [lock["criterion_name"]],
        "locked_metrics": dict(lock),
        "current_children": diagnostics["children"],
        "sibling_pairs": diagnostics["sibling_pairs"],
        "prior_failure_attributions": [
            record["history_payload"]["natural_language_attribution"]
            for record in history.get("attempts", [])
            if record.get("history_payload", {}).get(
                "natural_language_attribution")
        ],
        "instructions": {
            "locked_children": "lock_exact",
            "other_children": "rewrite_or_replace_in_fixed_cluster",
            "clustering": "forbidden",
            "heldout": "forbidden",
        },
    }


def _integrated_add_lock_diagnostics(
    result: dict[str, Any],
    *,
    rubric: StructuredRubric,
    rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> None:
    """Persist child diagnostics and lock decision for a completed rejection."""
    if result["decision"] != split.COMPETITION_REJECTED:
        return
    candidate = result["candidate"]
    diagnostics = split.build_child_retry_diagnostics(
        candidate=candidate,
        combined=result["combined"],
        rows=rows,
        parent_node_id=result["root_id"],
        parent_name=rubric.get_node(result["root_id"]).criterion.name,
        evaluation=result["evaluation"],
    )
    lock = _integrated_select_locked_child(diagnostics, settings)
    result["child_retry_diagnostics"] = diagnostics
    result["lock"] = lock
    _write(result["attempt_dir"] / "child_retry_diagnostics.json", diagnostics)
    _write(result["attempt_dir"] / "lock_selection.json", {
        "schema_version": "1.0.0",
        "selected": lock,
        "eligible_children": [
            item["criterion_name"] for item in diagnostics["children"]
            if item["support"] >= settings["strong_child_min_support"]
            and item["net_corrected"] >= settings[
                "strong_child_min_net_corrected"]
        ],
    })
    attribution = result.get("history_payload", {}).get(
        "natural_language_attribution")
    result["history_payload"] = split._failure(
        result,
        "specialized_accuracy_below_parent",
        attribution=None if attribution is None else {"attribution": attribution},
    )


def _integrated_locked_retry_prepare(
    *,
    config: Mapping[str, Any],
    target: Path,
    epoch_dir: Path,
    root_id: str,
    attempt_no: int,
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    feedback: RubricFeedback,
    rows: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    managers: Mapping[str, Any],
    settings: Mapping[str, Any],
    rubric_memory: Mapping[str, Any],
    rubric_memory_sha256: str,
    pool: AvailableSlotBackendPool,
) -> dict[str, Any]:
    """Rebuild only unlocked children from an exact prior failed candidate."""
    state = history["root_states"][root_id]
    pending = state.get("locked_retry")
    if not isinstance(pending, Mapping):
        raise RuntimeError("locked retry state is missing")
    source_dir = Path(pending["candidate_attempt_dir"])
    candidate = split.SpecializeCandidate.from_dict(
        load_json(source_dir / "candidate.json"))
    candidate_rubric = StructuredRubric.load_json(
        source_dir / "candidate_rubric.json")
    combined = PairwisePredictionOutput.load_json(
        source_dir / "combined_pairwise.json")
    signature_values = load_json(source_dir / "error_signatures.json")["outputs"]
    signatures = {
        sample_id: split.ErrorSignatureOutput.from_dict(value).signature
        for sample_id, value in signature_values.items()
    }
    if any(value is None for value in signatures.values()):
        raise RuntimeError("locked retry source contains invalid ErrorSignature")
    parent = rubric.get_node(root_id)
    old_parent = candidate_rubric.get_node(root_id)
    if parent.criterion != old_parent.criterion:
        raise RuntimeError("locked retry parent changed before the candidate committed")
    if rubric_memory_sha256 != canonical_sha256(rubric_memory):
        raise RuntimeError("locked retry rubric memory hash drift")

    attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    lock = dict(pending["lock"])
    locked_name = lock["criterion_name"]
    source_children = {
        child.criterion_name: child for child in candidate.children}
    if locked_name not in source_children:
        raise RuntimeError("locked child is absent from source candidate")
    retry_feedback = _integrated_retry_feedback(
        lock=lock,
        diagnostics=load_json(source_dir / "child_retry_diagnostics.json"),
        history=history,
    )
    _write(attempt_dir / "trigger.json", {
        "schema_version": "1.0.0",
        "triggered": True,
        "mode": "locked_retry",
        "root_id": root_id,
        "source_attempt_dir": str(source_dir),
    })
    _write(attempt_dir / "frozen_cluster_proposal.json",
           candidate.cluster_proposal.to_dict())
    _write(attempt_dir / "retry_feedback.json", retry_feedback)
    _write(attempt_dir / "rubric_memory_ref.json", {
        "rubric_memory_mode": "global_rubric_v1",
        "rubric_memory_sha256": rubric_memory_sha256,
        "locked_child": lock,
        "cluster_reused": True,
    })

    rows_by_id = {str(row["sample_id"]): row for row in rows}
    children = []
    fresh_children = []
    packets = []
    retry_manager = managers["child_generation"]
    for cluster in candidate.cluster_proposal.clusters:
        old_child = next(
            item for item in candidate.children
            if item.cluster_id == cluster.cluster_id)
        child_path = attempt_dir / "children" / f"{cluster.cluster_id}.json"
        if old_child.criterion_name == locked_name:
            children.append(old_child)
            _write(child_path, old_child.to_dict())
            continue
        packet = split._v2_packet(
            candidate, combined, rows, parent.criterion.name,
            old_child.criterion_name, signatures)
        packet["requested_action"] = "rewrite_or_replace"
        packets.append(packet)
        if child_path.exists():
            child = split.ChildCriterionProposal.from_dict(load_json(child_path))
        else:
            representative_rows = [
                rows_by_id[sample_id]
                for sample_id in packet["representative_sample_ids"]]
            supplemental_rows = [
                rows_by_id[item["sample_id"]] for item in packet["samples"]
                if item["sample_id"] not in packet["representative_sample_ids"]]
            child = retry_manager.generate_child(
                parent=parent,
                cluster=cluster,
                signatures=[signatures[sample_id]
                            for sample_id in cluster.sample_ids],
                representative_rows=representative_rows,
                supplemental_rows=supplemental_rows,
                siblings=children,
                prior_failures=split._history_projection(
                    history, root_id, include_retry_diagnostics=True),
                rubric_memory=rubric_memory,
                retry_feedback=retry_feedback,
                repair_context=packet,
            )
            _write(child_path, child.to_dict())
        children.append(child)
        fresh_children.append(child)
        print(
            f"five-root locked-retry child {len(fresh_children)} "
            f"root={root_id} cluster={cluster.cluster_id}",
            flush=True,
        )
    _write(attempt_dir / "sample_packets.json", packets)
    try:
        new_candidate = split.build_specialize_candidate(
            EvolutionContext(rubric, feedback), root_id,
            candidate.cluster_proposal, children, rows, signatures)
    except ValueError as exc:
        raise split.ProposalInvalid(
            "locked_retry_child_generation", str(exc)) from exc
    after = apply_rubric_patch(rubric, new_candidate.edit_candidate.patch)
    new_candidate.edit_candidate.patch
    _write(attempt_dir / "candidate.json", new_candidate.to_dict())
    after.save_json(attempt_dir / "candidate_rubric.json")

    locked_cluster = lock["cluster_id"]
    locked_node_id = new_candidate.node_id_by_cluster[locked_cluster]
    old_locked_node_id = candidate.node_id_by_cluster[locked_cluster]
    if locked_node_id != old_locked_node_id:
        raise RuntimeError("locked child node identity drift")
    locked_prediction = project_pairwise_prediction(
        combined,
        StructuredRubric(
            {locked_node_id: candidate_rubric.get_node(locked_node_id)},
            (), (locked_node_id,)),
    )
    predictions = [locked_prediction]
    generated_names = []
    valid_rate = 1.0
    artifact = None
    if fresh_children:
        fresh_ids = tuple(
            new_candidate.node_id_by_cluster[child.cluster_id]
            for child in fresh_children)
        fresh_rubric = StructuredRubric(
            {node_id: after.get_node(node_id) for node_id in fresh_ids},
            (), fresh_ids)
        fresh, artifact, valid_rate = base._generate_pairwise(
            config, attempt_dir, fresh_rubric, rows, "unlocked_children",
            execution_backend_pool=pool,
            request_backend_id=prediction.request_spec.backend_id,
            request_level_progress=True,
        )
        predictions.append(fresh)
        generated_names = [child.criterion_name for child in fresh_children]
    _write(attempt_dir / "pairwise_reuse.json", {
        "locked_prediction_reused": True,
        "locked_criterion_name": locked_name,
        "locked_prediction_source_sha256": canonical_sha256(
            locked_prediction.to_dict()),
        "generated_criterion_names": generated_names,
        "generated_artifact": None if artifact is None else str(artifact),
        "generated_valid_rate": valid_rate,
    })
    new_combined = split.merge_predictions(prediction, predictions, after)
    new_combined.save_json(attempt_dir / "combined_pairwise.json")
    child_ids = tuple(new_candidate.node_id_by_cluster.values())
    child_prediction = project_pairwise_prediction(
        new_combined,
        StructuredRubric(
            {node_id: after.get_node(node_id) for node_id in child_ids},
            (), child_ids))
    evaluation, _, after_execution = split.evaluate_specialize_candidate(
        before_rubric=rubric, after_rubric=after,
        combined_prediction=new_combined, child_prediction=child_prediction,
        dataset=rows, parent_node_id=root_id,
        cluster_proposal=new_candidate.cluster_proposal,
        candidate=new_candidate, policy=split.runtime_acceptance_policy(config))
    records = base._split_parent_scope_predictions(
        context=EvolutionContext(rubric, feedback), candidate=new_candidate,
        combined_prediction=new_combined, after_execution=after_execution,
        rows=rows)
    _write(attempt_dir / "evaluation.json", evaluation.to_dict())
    _write(attempt_dir / "specialized_predictions.json", records)
    diagnostics = split.build_child_retry_diagnostics(
        candidate=new_candidate, combined=new_combined, rows=rows,
        parent_node_id=root_id, parent_name=parent.criterion.name,
        evaluation=evaluation)
    _write(attempt_dir / "child_retry_diagnostics.json", diagnostics)
    decision = (split.ACCEPTED if evaluation.specialized_accuracy
                >= evaluation.parent_accuracy else split.COMPETITION_REJECTED)
    result = {
        "root_id": root_id, "decision": decision, "attempt_dir": attempt_dir,
        "candidate": new_candidate, "after_rubric": after,
        "signatures": signatures, "child_prediction": child_prediction,
        "combined": new_combined, "evaluation": evaluation,
        "changed_predictions": records, "child_retry_diagnostics": diagnostics,
        "lock": lock,
    }
    if decision == split.COMPETITION_REJECTED:
        attribution = split._required_failure_attribution(
            managers["semantic_cluster"], attempt_dir, parent=parent,
            signatures=tuple(signatures.values()),
            cluster_proposal=new_candidate.cluster_proposal.to_dict(),
            children=new_candidate.children,
            local_metrics=evaluation.to_dict(),
            changed_predictions=records,
            retry_feedback=retry_feedback)
        result["history_payload"] = split._failure(
            result, "specialized_accuracy_below_parent",
            attribution=attribution)
    return result


def _integrated_refine_schedule(
    rubric: StructuredRubric,
    feedback: RubricFeedback,
    config: Mapping[str, Any],
    split_roots: Sequence[str],
) -> tuple[str, ...]:
    scheduled = _current_refine_eligible(
        rubric, feedback, config, trigger_mode="role_aware_v2")
    blocked = set(split_roots)
    return tuple(node_id for node_id in scheduled if node_id not in blocked)


def _integrated_record_split_history(
    history: dict[str, Any],
    scheduled: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    started: Mapping[str, float],
    *,
    epoch_no: int,
    max_epochs: int,
) -> list[dict[str, Any]]:
    records = []
    for root_id in scheduled:
        result = results[root_id]
        state = history["root_states"][root_id]
        decision = result["decision"]
        state["attempt_count"] += 1
        evaluation = result.get("evaluation")
        if decision == split.ACCEPTED:
            state.update({
                "status": "accepted_locked",
                "accepted_epoch": epoch_no,
                "children": [
                    item.criterion_name for item in result["candidate"].children],
            })
            state.pop("locked_retry", None)
        else:
            state["status"] = (
                "exhausted" if state["attempt_count"] >= max_epochs
                else "retryable")
            lock = result.get("lock")
            if lock is not None:
                state["locked_retry"] = {
                    "candidate_attempt_dir": str(result["attempt_dir"]),
                    "lock": lock,
                }
        record = {
            "epoch": epoch_no,
            "root_id": root_id,
            "attempt": state["attempt_count"],
            "decision": decision,
            "competition_completed": decision in split.VALID_COMPETITION_OUTCOMES,
            "attempt_dir": str(result["attempt_dir"]),
            "parent_accuracy": None if evaluation is None else evaluation.parent_accuracy,
            "specialized_accuracy": (
                None if evaluation is None else evaluation.specialized_accuracy),
            "accuracy_delta": None if evaluation is None else evaluation.accuracy_delta,
            "locked_child": result.get("lock"),
            "elapsed_seconds": time.monotonic() - started[root_id],
            "history_payload": result.get("history_payload"),
        }
        if (decision == split.COMPETITION_REJECTED
                and not isinstance((record["history_payload"] or {}).get(
                    "natural_language_attribution"), Mapping)):
            raise RuntimeError(
                "rejected locked Split cannot enter history without attribution")
        history["attempts"].append(record)
        records.append(record)
    return records


def five_root_locked_split_refine_freeze(
    config: Mapping[str, Any], output: Path,
) -> None:
    """Freeze the independent Phase 10 five-root integration experiment."""
    settings = _five_root_locked_split_refine_config(config)
    _validate_output(output)
    _require_pairwise_endpoint(config, PAIRWISE_ENDPOINT)
    target = _five_root_locked_split_refine_target(output)
    manifest_path = target / "frozen_manifest.json"
    if manifest_path.exists():
        print("five-root-locked-split-refine-freeze already completed")
        return
    split.freeze(config, output, FIVE_ROOT_LOCKED_SPLIT_REFINE_PROTOCOL)
    manifest = load_json(manifest_path)
    refine_manager, refine_profile = _manager(config)
    _, retry_specs = _integrated_retry_specs(config)
    manifest.update({
        "integration_settings": settings,
        "refine_protocol": REFINE_V1,
        "refine_manager_profile": refine_profile,
        "refine_manager_request_specs": {
            key: value.to_dict()
            for key, value in refine_manager.request_specs().items()},
        "locked_retry_manager_request_specs": retry_specs,
        "operator_schedule": (
            "synchronous_epoch_start_split_locked_retry_plus_role_aware_refine"),
        "partial_split_acceptance": False,
        "heldout_protocol": {
            "exploratory": True,
            "selection_after_heldout_forbidden": True,
            "source_experiment": SOURCE_EXPERIMENT_DIR,
        },
    })
    _write(manifest_path, manifest)
    history = load_json(target / "evolution_history.json")
    history.setdefault("refine_states", {})
    history.setdefault("refine_attempts", [])
    for state in history["root_states"].values():
        state.setdefault("locked_retry", None)
    _write(target / "evolution_history.json", history)
    status = load_json(target / "stage_status.json")
    status["freeze"] = {"status": "passed", "details": {
        "settings": settings, "initial_root_ids": manifest["initial_root_ids"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps({
        "target": str(target),
        "eligible_split_roots": manifest["initial_eligible_root_ids"],
        "worker_endpoint": PAIRWISE_ENDPOINT,
    }, indent=2, ensure_ascii=False))


def five_root_locked_split_refine_audit(
    config: Mapping[str, Any], output: Path,
) -> None:
    """Audit frozen scheduling inputs without opening heldout artifacts."""
    settings = _five_root_locked_split_refine_config(config)
    _validate_output(output)
    target = _five_root_locked_split_refine_target(output)
    manifest = load_json(target / "frozen_manifest.json")
    rubric = StructuredRubric.load_json(
        split._epoch(target, 0) / "rubric_committed.json")
    feedback = RubricFeedback.from_dict(load_json(
        split._epoch(target, 0) / "feedback.json"))
    split_roots = list(manifest["initial_eligible_root_ids"])
    refine_nodes = _integrated_refine_schedule(
        rubric, feedback, config, split_roots)
    value = {
        "schema_version": "1.0.0",
        "offline_only": True,
        "settings": settings,
        "split_roots": split_roots,
        "refine_nodes_after_split_precedence": list(refine_nodes),
        "heldout_accessed": False,
    }
    _write(target / "offline_audit.json", value)
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": {
        "eligible_split_roots": len(split_roots),
        "eligible_refine_nodes": len(refine_nodes)}}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def five_root_locked_split_refine_run(
    config: Mapping[str, Any], output: Path,
) -> None:
    """Run synchronous five-root Split/locked-retry/role-aware-Refine evolution."""
    settings = _five_root_locked_split_refine_config(config)
    _validate_output(output)
    target = _five_root_locked_split_refine_target(output)
    if not (target / "offline_audit.json").exists():
        raise RuntimeError(
            "run five-root-locked-split-refine-freeze and "
            "five-root-locked-split-refine-audit first")
    manifest = load_json(target / "frozen_manifest.json")
    history = load_json(target / "evolution_history.json")
    if history.get("completed"):
        print("five-root-locked-split-refine-run already completed")
        return
    rows = _rows(config, "discovery")
    managers, _, specs, _ = split._managers(
        config, FIVE_ROOT_LOCKED_SPLIT_REFINE_PROTOCOL)
    specs["error_signature"] = manifest["manager_request_specs"]["error_signature"]
    for stage in ("semantic_cluster", "child_generation"):
        if specs[stage] != manifest["manager_request_specs"][stage]:
            raise RuntimeError(f"Phase 10 Manager request identity drift: {stage}")
    retry_managers, retry_specs = _integrated_retry_specs(config)
    if retry_specs != manifest["locked_retry_manager_request_specs"]:
        raise RuntimeError("Phase 10 locked-retry Manager request identity drift")
    refine_manager, _ = _manager(config)
    expected_refine_specs = {
        key: value.to_dict()
        for key, value in refine_manager.request_specs().items()}
    if expected_refine_specs != manifest["refine_manager_request_specs"]:
        raise RuntimeError("Phase 10 Refine Manager request identity drift")
    pool = split_pool(config)
    max_epochs = settings["max_epochs"]

    for epoch_no in range(history["current_epoch"] + 1, max_epochs + 1):
        started_epoch = time.monotonic()
        epoch_dir = split._epoch(target, epoch_no)
        previous = split._epoch(target, epoch_no - 1)
        rubric = StructuredRubric.load_json(previous / "rubric_committed.json")
        prediction = PairwisePredictionOutput.load_json(
            previous / "discovery_pairwise.json")
        feedback = RubricFeedback.from_dict(load_json(previous / "feedback.json"))
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        split_scheduled = split.retryable_roots(history)
        refine_scheduled = _integrated_refine_schedule(
            rubric, feedback, config, split_scheduled)
        print(
            f"five-root integration epoch={epoch_no} "
            f"split={list(split_scheduled)} refine={list(refine_scheduled)}",
            flush=True,
        )
        split_results: dict[str, dict[str, Any]] = {}
        split_started: dict[str, float] = {}
        # Candidate construction is isolated per root.  A proposal/schema
        # failure is a scientific attempt for that root, not a reason to skip
        # every later root in this synchronous epoch.
        for root_id in split_scheduled:
            split_started[root_id] = time.monotonic()
            attempt_no = history["root_states"][root_id]["attempt_count"] + 1
            state = history["root_states"][root_id]
            try:
                if state.get("locked_retry"):
                    result = _integrated_locked_retry_prepare(
                        config=config, target=target, epoch_dir=epoch_dir,
                        root_id=root_id, attempt_no=attempt_no, rubric=rubric,
                        prediction=prediction, feedback=feedback, rows=rows,
                        history=history, managers=retry_managers,
                        settings=settings, rubric_memory=memory,
                        rubric_memory_sha256=memory_hash, pool=pool)
                else:
                    result = split._prepare(
                        config, target, epoch_dir, root_id, attempt_no, rubric,
                        prediction, feedback, rows, history, managers, specs,
                        FIVE_ROOT_LOCKED_SPLIT_REFINE_PROTOCOL,
                        memory, memory_hash)
                split_results[root_id] = result
            except split.TransportFailed as exc:
                split._pause_transport(
                    target, epoch_no, root_id, attempt_no, exc.stage,
                    split._failure_details(exc))
                return
            except split.AttributionInvalid as exc:
                status = load_json(target / "stage_status.json")
                status["run"] = {"status": "failed", "details": {
                    "epoch": epoch_no, "root_id": root_id,
                    "outcome": "attribution_invalid", "details": exc.details}}
                _write(target / "stage_status.json", status)
                return
            except split.SpecializeManagerFailure as exc:
                attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
                _write(attempt_dir / "manager_failure.json", exc.to_dict())
                if split._manager_failure_kind(exc) == split.TRANSPORT_FAILED:
                    split._pause_transport(
                        target, epoch_no, root_id, attempt_no, exc.stage,
                        exc.to_dict())
                    return
                payload = split._proposal_failure_payload(
                    attempt_dir, exc.stage, exc.to_dict())
                _write(attempt_dir / "proposal_failure.json", payload)
                split_results[root_id] = {
                    "root_id": root_id, "decision": split.PROPOSAL_INVALID,
                    "attempt_dir": attempt_dir, "history_payload": payload,
                    "lock": (state.get("locked_retry") or {}).get("lock")}
            except split.ProposalInvalid as exc:
                attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
                payload = split._proposal_failure_payload(
                    attempt_dir, exc.stage, split._failure_details(exc))
                _write(attempt_dir / "proposal_failure.json", payload)
                split_results[root_id] = {
                    "root_id": root_id, "decision": split.PROPOSAL_INVALID,
                    "attempt_dir": attempt_dir, "history_payload": payload,
                    "lock": (state.get("locked_retry") or {}).get("lock")}
            except Exception as exc:
                split._abort_program(
                    target, epoch_no, root_id, attempt_no,
                    "candidate_construction", exc)
                raise

        prepared = {
            root_id: value["candidate"] for root_id, value in split_results.items()
            if value.get("candidate") is not None}
        collisions = split.colliding_roots(prepared)
        collided = {
            root_id for owners in collisions.values() for root_id in owners}
        for root_id in collided:
            split_results[root_id]["decision"] = "cross_root_collision"
            split_results[root_id]["history_payload"] = split._failure(
                split_results[root_id], "cross_root_child_name_collision",
                "synchronous_commit", details={"collisions": collisions})
        try:
            for root_id in sorted(set(prepared) - collided):
                if split_results[root_id]["decision"] != "prepared":
                    continue
                result = split._evaluate(
                    config, epoch_dir, rows, rubric, prediction,
                    split_results[root_id], pool, managers["semantic_cluster"],
                    feedback, FIVE_ROOT_LOCKED_SPLIT_REFINE_PROTOCOL)
                _integrated_add_lock_diagnostics(
                    result, rubric=rubric, rows=rows, settings=settings)
                split_results[root_id] = result

            refine_results: dict[str, dict[str, Any]] = {}
            refine_started: dict[str, float] = {}
            for node_id in refine_scheduled:
                refine_started[node_id] = time.monotonic()
                attempt_no = history["refine_states"].get(
                    node_id, {"attempt_count": 0})["attempt_count"] + 1
                refine_results[node_id] = _prepare_refine_attempt(
                    config=config, epoch_dir=epoch_dir, node_id=node_id,
                    attempt_no=attempt_no, rubric=rubric,
                    prediction=prediction, feedback=feedback, rows=rows,
                    history=history, manager=refine_manager, pool=pool,
                    rubric_memory=memory, rubric_memory_sha256=memory_hash,
                    trigger_mode="role_aware_v2")
        except split.TransportFailed as exc:
            split._pause_transport(
                target, epoch_no, root_id, attempt_no, exc.stage,
                split._failure_details(exc))
            return
        except (RefineTransportPause, RefineAttributionInvalid) as exc:
            status = load_json(target / "stage_status.json")
            status["run"] = {"status": (
                "paused" if isinstance(exc, RefineTransportPause) else "failed"),
                "details": {"epoch": epoch_no, "message": str(exc),
                            "history_appended": False}}
            _write(target / "stage_status.json", status)
            return

        accepted_split = {
            root_id: result["candidate"] for root_id, result
            in split_results.items() if result["decision"] == split.ACCEPTED}
        accepted_refine = {
            node_id: result["candidate"] for node_id, result
            in refine_results.items() if result["decision"] == "accepted"}
        committed = _merge_epoch_rubric(
            rubric, accepted_split, accepted_refine)
        committed_prediction = _merge_epoch_predictions(
            prediction,
            [split_results[root_id]["child_prediction"]
             for root_id in sorted(accepted_split)],
            {node_id: refine_results[node_id]["candidate_prediction"]
             for node_id in accepted_refine},
            committed)
        votes, committed_feedback = split._snapshot(
            epoch_dir, committed, committed_prediction, rows, manifest)
        split_records = _integrated_record_split_history(
            history, split_scheduled, split_results, split_started,
            epoch_no=epoch_no, max_epochs=max_epochs)
        refine_records = [
            _record_refine_state(
                history, refine_results[node_id], epoch_no,
                time.monotonic() - refine_started[node_id])
            for node_id in refine_scheduled]
        history["current_epoch"] = epoch_no
        next_split = split.retryable_roots(history)
        next_refine = _integrated_refine_schedule(
            committed, committed_feedback, config, next_split)
        if epoch_no >= settings["min_epochs"] and not (next_split or next_refine):
            history.update({"completed": True,
                            "stop_reason": "no_retryable_operations"})
        elif epoch_no == max_epochs:
            history.update({"completed": True, "stop_reason": "max_epochs_reached"})
        metrics = base._metrics(votes, rows)
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no,
            "split_scheduled": list(split_scheduled),
            "refine_scheduled": list(refine_scheduled),
            "accepted_split_roots": sorted(accepted_split),
            "accepted_refine_nodes": sorted(accepted_refine),
            "split_collisions": collisions,
            "split_attempts": split_records,
            "refine_attempts": refine_records,
            "next_split_retryable": list(next_split),
            "next_refine_eligible": list(next_refine),
            "rubric_node_count": len(committed.nodes),
            "m1": metrics,
            "epoch_wall_seconds": time.monotonic() - started_epoch})
        _write(target / "evolution_history.json", history)
        print(
            f"five-root integration epoch={epoch_no} "
            f"split_accept={sorted(accepted_split)} "
            f"refine_accept={sorted(accepted_refine)} "
            f"m1_acc={metrics['accuracy']:.4f}",
            flush=True)
        if history["completed"]:
            break

    status = load_json(target / "stage_status.json")
    status["run"] = {
        "status": "passed" if history["completed"] else "partial",
        "details": {"current_epoch": history["current_epoch"],
                    "stop_reason": history["stop_reason"]},
    }
    _write(target / "stage_status.json", status)


def five_root_locked_split_refine_report(
    config: Mapping[str, Any], output: Path,
) -> None:
    _five_root_locked_split_refine_config(config)
    _validate_output(output)
    target = _five_root_locked_split_refine_target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run five-root-locked-split-refine-run to completion first")
    final_epoch = split._epoch(target, history["current_epoch"])
    rubric = StructuredRubric.load_json(final_epoch / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(
        final_epoch / "discovery_pairwise.json")
    rows = _rows(config, "discovery")
    execution, answers = execute_offline_m1(rubric, prediction, rows)
    initial_summary = load_json(split._epoch(target, 0) / "summary.json")
    metrics = base._metrics(answers, rows)
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    execution.save_json(final_dir / "m1_execution.json")
    value = {
        "schema_version": "1.0.0",
        "experiment": FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR,
        "exploratory": True,
        "final_epoch": history["current_epoch"],
        "initial_m1": initial_summary["m1"],
        "final_m1": metrics,
        "m1_accuracy_delta": metrics["accuracy"] - initial_summary["m1"]["accuracy"],
        "final_rubric_sha256": rubric.rubric_sha256,
        "node_count_initial": len(StructuredRubric.load_json(
            split._epoch(target, 0) / "rubric_committed.json").nodes),
        "node_count_final": len(rubric.nodes),
        "split_attempts": history["attempts"],
        "refine_attempts": history["refine_attempts"],
        "locked_retry_count": sum(
            bool(item.get("locked_child")) for item in history["attempts"]),
        "accepted_split_count": sum(
            item["decision"] == split.ACCEPTED for item in history["attempts"]),
        "accepted_refine_count": sum(
            item["decision"] == "accepted"
            for item in history["refine_attempts"]),
    }
    _write(final_dir / "discovery_report.json", value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_epoch": history["current_epoch"],
        "final_m1_accuracy": metrics["accuracy"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))




def five_root_locked_split_refine_heldout(
    config: Mapping[str, Any], output: Path,
) -> None:
    """Evaluate the frozen Phase 10 final rubric once on heldout-500."""
    _five_root_locked_split_refine_config(config)
    _validate_output(output)
    target = _five_root_locked_split_refine_target(output)
    discovery_path = target / "final" / "discovery_report.json"
    if not discovery_path.exists():
        raise RuntimeError("run five-root-locked-split-refine-report first")
    final_rubric = StructuredRubric.load_json(target / "final" / "rubric.json")
    source = _source(output)
    source_prediction_path = source / "heldout500" / "combined_pairwise.json"
    initial_prediction_path = (
        output / "predictions" / "init_pairwise_p05_heldout500.json")
    source_prediction = PairwisePredictionOutput.load_json(source_prediction_path)
    initial_prediction = PairwisePredictionOutput.load_json(initial_prediction_path)
    rows = _rows(config, "heldout")
    expected = base._expected_pairwise_request_spec(config, rows)
    if (not _same_pairwise_scientific_identity(
            source_prediction.request_spec, expected)
            or not _same_pairwise_scientific_identity(
                initial_prediction.request_spec, expected)):
        raise RuntimeError("Phase 10 heldout Worker scientific identity drift")

    heldout_dir = target / "heldout500"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "final_discovery_report_sha256": file_sha256(discovery_path),
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "source_prediction_sha256": file_sha256(source_prediction_path),
        "initial_prediction_sha256": file_sha256(initial_prediction_path),
        "pairwise_request_spec": expected.to_dict(),
        "selection_after_heldout_forbidden": True,
    }
    frozen_path = heldout_dir / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != manifest:
        raise RuntimeError("Phase 10 heldout manifest drift")
    _write(frozen_path, manifest)

    source_descriptions = {
        item.name: item.description for item in source_prediction.criteria}
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
                config, heldout_dir, changed_rubric, rows, "changed_nodes",
                execution_backend_pool=split_pool(config),
                request_backend_id=source_prediction.request_spec.backend_id,
                request_level_progress=True)
            generated.save_json(generated_path)
    combined = _combine_heldout_predictions(
        source_prediction, generated, final_rubric)
    combined.save_json(heldout_dir / "combined_pairwise.json")
    final_execution, final_answers = execute_offline_m1(
        final_rubric, combined, rows)
    final_execution.save_json(heldout_dir / "m1_execution.json")

    initial_rubric = StructuredRubric.load_json(
        split._epoch(target, 0) / "rubric_committed.json")
    initial_projected = project_pairwise_prediction(
        initial_prediction, initial_rubric)
    _, initial_answers = execute_offline_m1(
        initial_rubric, initial_projected, rows)
    source_rubric = StructuredRubric.load_json(source / "final" / "rubric.json")
    _, source_answers = execute_offline_m1(source_rubric, source_prediction, rows)
    initial_metrics = base._heldout_vote_metrics(initial_answers, rows)
    source_metrics = base._heldout_vote_metrics(source_answers, rows)
    final_metrics = base._heldout_vote_metrics(final_answers, rows)
    initial_frozen = load_json(output / "reports" / "init_baseline.json")[
        "heldout500"][base.StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]
    if (initial_metrics["accuracy"] != initial_frozen["accuracy"]
            or initial_metrics["coverage"] != initial_frozen["coverage"]):
        raise RuntimeError("Phase 10 Init heldout replay differs from frozen baseline")

    value = {
        "schema_version": "1.0.0",
        "exploratory": True,
        "initial_m1": initial_metrics,
        "split_only_global_memory_m1": source_metrics,
        "final_m1": final_metrics,
        "final_vs_initial": base._paired_heldout_comparison(
            initial_answers, final_answers, rows),
        "final_vs_split_only_global_memory": base._paired_heldout_comparison(
            source_answers, final_answers, rows),
        "reused_exact_criteria": len(final_rubric.nodes) - len(changed_ids),
        "generated_changed_criteria": len(changed_ids),
        "generated_node_ids": list(changed_ids),
        "selection_after_heldout_forbidden": True,
    }
    _write(heldout_dir / "report.json", value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        "generated_changed_criteria": len(changed_ids),
        "final_m1_accuracy": final_metrics["accuracy"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def five_root_locked_split_refine_final_report(
    config: Mapping[str, Any], output: Path,
) -> None:
    _five_root_locked_split_refine_config(config)
    _validate_output(output)
    target = _five_root_locked_split_refine_target(output)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout = load_json(target / "heldout500" / "report.json")
    final_vs_control = heldout["final_vs_split_only_global_memory"]
    value = {
        "schema_version": "1.0.0",
        "experiment": FIVE_ROOT_LOCKED_SPLIT_REFINE_EXPERIMENT_DIR,
        "exploratory": True,
        "discovery": discovery,
        "heldout": heldout,
        "primary": {
            "initial_m1_accuracy": heldout["initial_m1"]["accuracy"],
            "split_only_global_memory_m1_accuracy": (
                heldout["split_only_global_memory_m1"]["accuracy"]),
            "integrated_final_m1_accuracy": heldout["final_m1"]["accuracy"],
            "net_corrected_vs_split_only_global_memory": (
                final_vs_control["net_corrected"]),
            "exact_mcnemar_p_vs_split_only_global_memory": (
                final_vs_control["mcnemar_exact_two_sided_p"]),
        },
    }
    _write(target / "final_report.json", value)
    lines = [
        "# Five-root Locked-Split + Role-aware Refine v1",
        "",
        "Exploratory heldout comparison; no post-heldout selection.",
        "",
        f"- Discovery M1: {discovery['initial_m1']['accuracy']:.4f} -> "
        f"{discovery['final_m1']['accuracy']:.4f}",
        f"- Heldout M1: {heldout['initial_m1']['accuracy']:.4f} -> "
        f"{heldout['final_m1']['accuracy']:.4f}",
        f"- Split-only + Global Memory M1: "
        f"{heldout['split_only_global_memory_m1']['accuracy']:.4f}",
        f"- Net corrected vs Split-only + Global Memory: "
        f"{final_vs_control['net_corrected']}",
    ]
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed"}
    _write(target / "stage_status.json", status)
    print(json.dumps(value["primary"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {"refine-freeze": freeze, "refine-smoke": smoke,
               "refine-smoke-heldout": heldout, "refine-smoke-report": report,
               "split-refine-freeze": split_refine_freeze,
               "split-refine-run": split_refine_run,
               "split-refine-report": split_refine_report,
               "split-refine-heldout": split_refine_heldout,
               "split-refine-final-report": split_refine_final_report,
               "refine-role-freeze": refine_role_freeze,
               "refine-role-audit": refine_role_audit,
               "refine-role-run": refine_role_run,
               "refine-role-report": refine_role_report,
               "refine-role-heldout": refine_role_heldout,
               "refine-role-final-report": refine_role_final_report,
               "refine-role-checkpoints-freeze": refine_role_checkpoints_freeze,
               "refine-role-checkpoints-run": refine_role_checkpoints_run,
               "refine-role-checkpoints-report": refine_role_checkpoints_report,
               "refine-role-checkpoints-v2-freeze": refine_role_checkpoints_v2_freeze,
               "refine-role-checkpoints-v2-run": refine_role_checkpoints_v2_run,
               "refine-role-checkpoints-v2-report": refine_role_checkpoints_v2_report,
               "visual-split-refine-freeze": visual_split_refine_freeze,
               "visual-split-refine-audit": visual_split_refine_audit,
               "visual-split-refine-run": visual_split_refine_run,
               "visual-split-refine-report": visual_split_refine_report,
               "visual-split-refine-heldout": visual_split_refine_heldout,
               "visual-split-refine-final-report": visual_split_refine_final_report,
               "five-root-locked-split-refine-freeze": (
                   five_root_locked_split_refine_freeze),
               "five-root-locked-split-refine-audit": (
                   five_root_locked_split_refine_audit),
               "five-root-locked-split-refine-run": (
                   five_root_locked_split_refine_run),
               "five-root-locked-split-refine-report": (
                   five_root_locked_split_refine_report),
               "five-root-locked-split-refine-heldout": (
                   five_root_locked_split_refine_heldout),
               "five-root-locked-split-refine-final-report": (
                   five_root_locked_split_refine_final_report)}
    if stage not in actions:
        raise ValueError(f"unsupported Refine stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

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
) -> dict[str, Any]:
    attempt_dir = _refine_attempt(epoch_dir, node_id, attempt_no)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    context = EvolutionContext(rubric, feedback)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    trigger = detect_refine_trigger(context, node_id, thresholds)
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
                             config: Mapping[str, Any]) -> tuple[str, ...]:
    context = EvolutionContext(rubric, feedback)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    return tuple(node_id for node_id in rubric.preorder_node_ids()
                 if detect_refine_trigger(context, node_id, thresholds).triggered)


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


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {"refine-freeze": freeze, "refine-smoke": smoke,
               "refine-smoke-heldout": heldout, "refine-smoke-report": report,
               "split-refine-freeze": split_refine_freeze,
               "split-refine-run": split_refine_run,
               "split-refine-report": split_refine_report,
               "split-refine-heldout": split_refine_heldout,
               "split-refine-final-report": split_refine_final_report}
    if stage not in actions:
        raise ValueError(f"unsupported Refine stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

"""Phase19: evolve with the deployed Unified-Subtree + Global-Arbiter semantics."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from critiq.structured.cache import canonical_sha256
from critiq.structured.evolution import EvolutionContext
from critiq.structured.evolution.refine import detect_refine_trigger
from critiq.structured.evolution.specialize import detect_specialize_trigger
from critiq.structured.evolution.types import RubricFeedback
from critiq.structured.schema import RubricCriterionSnapshot, RubricNode, StructuredRubric
from critiq.structured.dual_worker import PairwisePredictionOutput

from . import aligned_system_runtime as system
from . import discovery_v2_prompt_v2_evolution as phase17
from . import refine_evolution as refine
from . import run_rubric_evolution as base
from . import split_evolution as split
from .experiment_utils import atomic_write_json, load_json


EXPERIMENT_DIR = "phase19_unified_subtree_arbiter_aligned_evolution_v1"
PROTOCOL_VERSION = "unified-subtree-global-arbiter-aligned-evolution-v1"
STAGES = (
    "aligned-evolution-freeze",
    "aligned-evolution-audit",
    "aligned-evolution-smoke",
    "aligned-evolution-run",
    "aligned-evolution-report",
    "aligned-evolution-heldout",
)
SETTINGS = {
    **refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_V1,
    "protocol_version": PROTOCOL_VERSION,
    "source_protocol": phase17.PROTOCOL_VERSION,
    "candidate_scope": "phase17_locked_split_role_aware_refine",
    "acceptance_metric": "system_strict_accuracy",
    "none_is_wrong": True,
    "tie_policy": "reject",
    "same_root_winner": "strict_delta_corrected_negative_harmed_stable_id",
    "joint_commit": "all_winners_if_strict_improvement_else_best_single",
    "subset_search": False,
    "min_epochs": 3,
    "max_epochs": 5,
    "internal_k": 1,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "pairwise_prompt_mode": phase17.PAIRWISE_PROMPT_MODE,
    "pairwise_prompt_version": phase17.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    "worker_endpoint": "configured_available_slot_pool",
    "dev_used_for_selection": False,
    "heldout_access": "final_only_exploratory",
    "vl_rewardbench_required": True,
}


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _write(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("unified_subtree_arbiter_aligned_evolution_v1_experiment")
    if value != SETTINGS:
        raise RuntimeError(
            "unified_subtree_arbiter_aligned_evolution_v1_experiment drift")
    return dict(value)


def _runtime_settings(config: Mapping[str, Any]) -> system.RuntimeSettings:
    value = _settings(config)
    return system.RuntimeSettings(
        temperature=float(value["temperature"]),
        max_tokens=int(value["max_tokens"]),
        max_parse_retries=int(value["max_parse_retries"]),
        generation_seed_policy=str(value["generation_seed_policy"]),
    )


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return phase17._runtime_config(config)


def _rows(config: Mapping[str, Any], split_name: str):
    return phase17._rows(config, split_name)


def _load_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _settings(config)
    path = _target(output) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run aligned-evolution-freeze first")
    value = load_json(path)
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("aligned evolution frozen manifest identity drift")
    return value


def _system_eval(
    config: Mapping[str, Any], *, target: Path, output_path: Path,
    split_name: str, rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
    baseline: Mapping[str, Any] | None = None,
    changed_roots: Sequence[str] | None = None,
) -> dict[str, Any]:
    return system.evaluate(
        config, output_path=output_path, cache_dir=target / "system_cache",
        split_name=split_name, rows=rows, rubric=rubric,
        settings=_runtime_settings(config), baseline=baseline,
        changed_root_ids=changed_roots)


def _endpoint_identities(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return base._inspect_endpoints(
        config, base.BackendPoolSpec.from_dict(config["backend_pool"]))


def _pairwise_execution_pool(config: Mapping[str, Any]) -> dict[str, Any]:
    """Use the frozen two-endpoint Worker pool for all Phase19 candidates.

    The legacy Split/Refine helpers default to a single ``vllm-8001`` route.
    Phase19 is explicitly frozen with the configured 8000+8001 pool, so using
    that legacy route here would silently violate the experiment contract.
    """
    spec = base.BackendPoolSpec.from_dict(config["backend_pool"])
    endpoint_ids = tuple(item.endpoint_id for item in spec.endpoints)
    if endpoint_ids != system.ENDPOINT_IDS:
        raise RuntimeError(
            "aligned evolution requires the configured vllm-8000 + vllm-8001 pool")
    return spec.to_dict()


def _verify_live_endpoints(
    config: Mapping[str, Any], manifest: Mapping[str, Any],
) -> None:
    if manifest.get("worker_endpoint_identities") != _endpoint_identities(config):
        raise RuntimeError("aligned evolution live Worker endpoint identity drift")


def _require_complete_system(value: Mapping[str, Any], label: str) -> None:
    failures = int(value["metrics"]["technical_failure_count"])
    if failures:
        raise RuntimeError(
            f"{label} has {failures} unresolved system calls; "
            "the evolution state was not advanced")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    refine._validate_output(output)
    target = _target(output)
    discovery_rows = _rows(config, "discovery")
    dev_rows = _rows(config, "dev")
    rubric = base.build_multicrit_open_ended_init_rubric()
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 5:
        raise RuntimeError("aligned evolution requires five initial roots")
    runtime = _runtime_config(config)
    contract = phase17._manager_contract(config)
    expected_spec = base._expected_pairwise_request_spec(runtime, discovery_rows)
    epoch0 = split._epoch(target, 0)
    memory, memory_hash = split.freeze_rubric_memory(epoch0, rubric)
    rubric.save_json(epoch0 / "rubric_initial.json")
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "settings": settings,
        "initial_rubric_sha256": rubric.rubric_sha256,
        "initial_root_ids": list(rubric.root_ids),
        "datasets": {
            "discovery": {"path": phase17.DISCOVERY_PATH,
                          "count": len(discovery_rows),
                          "sha256": phase17._file_sha(phase17.DISCOVERY_PATH)},
            "dev": {"path": phase17.DEV_PATH, "count": len(dev_rows),
                    "sha256": phase17._file_sha(phase17.DEV_PATH)},
            "heldout": {"path": str(config["heldout_dataset"]),
                        "count": 500,
                        "sha256": phase17._file_sha(config["heldout_dataset"])},
        },
        "pairwise_request_spec": expected_spec.to_dict(),
        "pairwise_execution_pool": base.BackendPoolSpec.from_dict(
            config["backend_pool"]).to_dict(),
        "worker_endpoint_identities": _endpoint_identities(config),
        "manager_profiles": contract["profiles"],
        "manager_request_specs": contract["specs"],
        "manager_endpoint_identities": contract["identities"],
        "refine_protocol": refine.REFINE_V1,
        "refine_manager_profile": contract["refine_profile"],
        "refine_manager_request_specs": contract["refine_specs"],
        "locked_retry_manager_request_specs": contract["retry_specs"],
        "epoch_00_rubric_memory": memory,
        "epoch_00_rubric_memory_sha256": memory_hash,
        "system_prompts": {
            "unified_subtree": system.unified.SUBTREE_PROMPT_VERSION,
            "global_arbiter": system.unified.ARBITER_PROMPT_VERSION,
        },
        "node_worker_role": "diagnostic_trigger_and_proposal_only",
        "dev_visible_to_manager": False,
        "dev_affects_acceptance": False,
        "dev_affects_early_stop": False,
        "heldout_accessed": False,
    }
    path = target / "frozen_manifest.json"
    if path.is_file():
        existing = load_json(path)
        status = (load_json(target / "stage_status.json")
                  if (target / "stage_status.json").is_file() else {})
        if ("worker_endpoint_identities" not in existing
                and status.get("smoke", {}).get("status") != "passed"):
            _write(path, manifest)
            print("aligned-evolution-freeze upgraded pre-smoke endpoint identity")
            return
        if existing != manifest:
            raise RuntimeError("aligned evolution frozen manifest drift")
        print("aligned-evolution-freeze already completed")
        return
    _write(path, manifest)
    _write(target / "stage_status.json", {
        "freeze": {"status": "passed", "details": {
            "discovery_count": len(discovery_rows),
            "dev_count": len(dev_rows),
            "heldout_accessed": False,
        }}})
    print(json.dumps(_target(output).as_posix(), ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_manifest(config, output)
    target = _target(output)
    rubric = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    contract = phase17._manager_contract(config)
    pool = base.BackendPoolSpec.from_dict(config["backend_pool"])
    checks = {
        "five_initial_roots": len(rubric.root_ids) == 5 and len(rubric.nodes) == 5,
        "discovery_count_100": len(_rows(config, "discovery")) == 100,
        "dev_count_150": len(_rows(config, "dev")) == 150,
        "pairwise_prompt_v2": manifest["settings"]["pairwise_prompt_version"]
            == phase17.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "manager_specs_match": contract["specs"] == manifest["manager_request_specs"],
        "refine_specs_match": contract["refine_specs"]
            == manifest["refine_manager_request_specs"],
        "two_worker_endpoints": tuple(item.endpoint_id for item in pool.endpoints)
            == system.ENDPOINT_IDS,
        "worker_endpoint_identities_frozen": (
            len(manifest.get("worker_endpoint_identities", [])) == 2),
        "strict_acceptance": manifest["settings"]["acceptance_metric"]
            == "system_strict_accuracy",
        "dev_selection_forbidden": manifest["dev_affects_acceptance"] is False,
        "heldout_not_accessed": manifest["heldout_accessed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"aligned evolution audit failed: {checks}")
    _write(target / "offline_audit.json", {
        "schema_version": "1.0.0", "offline_only": True, "checks": checks})
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": checks}
    _write(target / "stage_status.json", status)
    print(json.dumps(checks, indent=2))


def _smoke_rubric(rubric: StructuredRubric) -> StructuredRubric:
    root_id = rubric.root_ids[0]
    old = rubric.get_node(root_id)
    replacement = RubricNode(
        node_id=old.node_id,
        criterion=RubricCriterionSnapshot(
            old.criterion.name,
            old.criterion.description + "\n\nFor this smoke only, state the decisive evidence explicitly.",
            old.criterion.score),
        examples=old.examples,
        lineage=old.lineage,
    )
    nodes = dict(rubric.nodes)
    nodes[root_id] = replacement
    return StructuredRubric(nodes, rubric.edges, rubric.root_ids)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_manifest(config, output)
    _verify_live_endpoints(config, manifest)
    target = _target(output)
    status = load_json(target / "stage_status.json")
    if status.get("audit", {}).get("status") != "passed":
        raise RuntimeError("run aligned-evolution-audit first")
    report_path = target / "smoke" / "report.json"
    if report_path.is_file():
        print(json.dumps(load_json(report_path), indent=2))
        return
    rows = _rows(config, "discovery")[:20]
    rubric = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    before = _system_eval(
        config, target=target, output_path=target / "smoke" / "before.json",
        split_name="aligned_smoke_before", rows=rows, rubric=rubric)
    candidate = _smoke_rubric(rubric)
    changed_root = rubric.root_ids[0]
    after = _system_eval(
        config, target=target, output_path=target / "smoke" / "after.json",
        split_name="aligned_smoke_candidate", rows=rows, rubric=candidate,
        baseline=before, changed_roots=(changed_root,))
    reused = [
        call.get("incremental_reuse") is True
        for sample in after["samples"]
        for root_id, call in sample["replicates"]["0"]["subtrees"].items()
        if root_id != changed_root]
    regenerated = [
        sample["replicates"]["0"]["regenerated_root_ids"]
        for sample in after["samples"]]
    paired_value = system.paired(before["metrics"], after["metrics"], rows)
    attribution = _failure_mapping("smoke_diagnostic", paired_value)
    value = {
        "schema_version": "1.0.0", "status": "passed",
        "sample_count": 20,
        "before_parse_valid": before["metrics"]["technical_failure_rate"] == 0,
        "after_parse_valid": after["metrics"]["technical_failure_rate"] == 0,
        "unchanged_root_reports_reused": bool(reused) and all(reused),
        "only_target_root_regenerated": all(items == [changed_root]
                                             for items in regenerated),
        "paired": paired_value,
        "failure_attribution_schema_valid": all(
            key in attribution for key in (
                "summary", "failure_categories", "details", "avoid_next_time")),
        "formal_evolution_mutated": False,
        "heldout_accessed": False,
    }
    if not all(value[key] for key in (
            "before_parse_valid", "after_parse_valid",
            "unchanged_root_reports_reused", "only_target_root_regenerated",
            "failure_attribution_schema_valid")):
        raise RuntimeError(f"aligned evolution smoke failed: {value}")
    _write(report_path, value)
    status["smoke"] = {"status": "passed", "details": value}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2))


def _initialize(config: Mapping[str, Any], target: Path,
                manifest: dict[str, Any]) -> None:
    epoch0 = split._epoch(target, 0)
    required = (
        epoch0 / "discovery_pairwise.json",
        epoch0 / "rubric_committed.json",
        epoch0 / "feedback.json",
        epoch0 / "system" / "discovery.json",
        epoch0 / "summary.json",
        target / "evolution_history.json",
    )
    if all(path.is_file() for path in required):
        return
    rows = _rows(config, "discovery")
    rubric = StructuredRubric.load_json(epoch0 / "rubric_initial.json")
    prediction = phase17._worker_prediction(
        config, target / "baseline", rubric, rows, "discovery_initial")
    _, feedback = split._snapshot(epoch0, rubric, prediction, rows, manifest)
    system_value = _system_eval(
        config, target=target,
        output_path=epoch0 / "system" / "discovery.json",
        split_name="aligned_epoch00_discovery", rows=rows, rubric=rubric)
    _require_complete_system(system_value, "aligned epoch-0 discovery baseline")
    thresholds = _runtime_config(config)["evolution_policy"]["trigger_thresholds"]
    context = EvolutionContext(rubric, feedback)
    triggers = {root_id: detect_specialize_trigger(
        context, root_id, thresholds).to_dict() for root_id in rubric.root_ids}
    eligible = [root_id for root_id in rubric.root_ids
                if triggers[root_id]["triggered"]]
    manifest.update({
        "initial_triggers": triggers,
        "initial_eligible_root_ids": eligible,
        "initial_eligible_root_count": len(eligible),
        "initial_prediction_sha256": canonical_sha256(prediction.to_dict()),
    })
    _write(target / "frozen_manifest.json", manifest)
    _write(epoch0 / "arbiter_aligned_feedback.json",
           _aligned_feedback(system_value, rows, rubric))
    _write(epoch0 / "summary.json", {
        "epoch": 0, "system": system_value["metrics"],
        "diagnostic_m1": load_json(epoch0 / "m1_execution.json").get(
            "metrics", None),
        "triggers": triggers, "accepted_roots": []})
    _write(target / "evolution_history.json", {
        "schema_version": "1.0.0", "current_epoch": 0,
        "completed": False, "stop_reason": None,
        "root_states": {root_id: {
            "status": "eligible" if triggers[root_id]["triggered"] else "not_eligible",
            "attempt_count": 0, "accepted_epoch": None,
            "children": [], "locked_retry": None,
        } for root_id in rubric.root_ids},
        "attempts": [], "refine_states": {}, "refine_attempts": [],
        "joint_commits": [],
    })


def _aligned_feedback(value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                      rubric: StructuredRubric) -> dict[str, Any]:
    attributed = system.attributed_error_ids(value, rows)
    predictions = value["metrics"]["predictions"]
    errors = []
    for row, prediction in zip(rows, predictions):
        if prediction != row["answer"]:
            sample = next(item for item in value["samples"]
                          if item["sample_id"] == row["sample_id"])
            reports = sample["replicates"]["0"]["subtrees"]
            correct_roots = [root_id for root_id, call in reports.items()
                             if (call.get("parsed") or {}).get("answer")
                             == row["answer"]]
            errors.append({
                "sample_id": str(row["sample_id"]),
                "gold": row["answer"], "prediction": prediction,
                "attributed_root_ids": [root_id for root_id in rubric.root_ids
                                        if str(row["sample_id"])
                                        in attributed.get(root_id, ())],
                "correct_root_ids": correct_roots,
                "failure_type": ("arbiter_synthesis_failure" if correct_roots
                                 else "subtree_evidence_regression"),
            })
    return {
        "schema_version": "1.0.0",
        "rubric_sha256": rubric.rubric_sha256,
        "metrics": value["metrics"],
        "attributed_error_ids_by_root": {
            key: list(items) for key, items in attributed.items()},
        "errors": errors,
    }


def _failure_mapping(kind: str, paired_value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary": (
            f"System-level candidate rejected as {kind}: corrected "
            f"{paired_value['corrected']} and harmed {paired_value['harmed']} samples."),
        "failure_categories": [kind],
        "details": [
            f"Strict accuracy delta={paired_value['strict_accuracy_delta']:.6f}.",
            "Acceptance is determined only by the Unified-Subtree plus Global-Arbiter system."],
        "avoid_next_time": [
            "Preserve evidence on harmed samples and target the attributed root errors."],
    }


def _set_system_outcome(result: dict[str, Any], *, accepted: bool,
                        paired_value: Mapping[str, Any], kind: str) -> None:
    result["system_evaluation"] = {
        **dict(paired_value), "accepted": accepted, "failure_type": kind,
    }
    if "root_id" in result:
        result["decision"] = split.ACCEPTED if accepted else split.COMPETITION_REJECTED
        if not accepted:
            attribution = _failure_mapping(kind, paired_value)
            result["history_payload"] = split._failure(
                result, kind, "system_competition", attribution=attribution,
                details={"system_evaluation": dict(paired_value)})
            _write(Path(result["attempt_dir"]) / "aligned_failure_attribution.json",
                   attribution)
    else:
        result["decision"] = "accepted" if accepted else "competition_rejected"
        if not accepted:
            attribution = _failure_mapping(kind, paired_value)
            result["history_payload"] = {
                "natural_language_attribution": attribution,
                "structured_failure": {
                    "code": kind,
                    "stage": "system_competition",
                },
                "node_evaluation": (None if result.get("evaluation") is None
                                    else result["evaluation"].to_dict()),
                "system_evaluation": dict(paired_value),
                "proposed_description": (None if result.get("proposal") is None
                                         else result["proposal"].description),
            }
            _write(Path(result["attempt_dir"]) / "aligned_failure_attribution.json",
                   attribution)


def _candidate_key(item: tuple[str, str, dict[str, Any]]) -> tuple[Any, ...]:
    kind, item_id, result = item
    value = result["system_evaluation"]
    return (
        value["strict_accuracy_delta"], value["corrected"], -value["harmed"],
        f"{kind}:{item_id}",
    )


def _candidate_root(rubric: StructuredRubric, kind: str, item_id: str) -> str:
    return item_id if kind == "split" else rubric.root_id_for(item_id)


def _merge_selected(
    rubric: StructuredRubric, prediction: PairwisePredictionOutput,
    selected: Sequence[tuple[str, str, dict[str, Any]]],
) -> tuple[StructuredRubric, PairwisePredictionOutput]:
    split_candidates = {item_id: result["candidate"]
                        for kind, item_id, result in selected if kind == "split"}
    refine_candidates = {item_id: result["candidate"]
                         for kind, item_id, result in selected if kind == "refine"}
    committed = refine._merge_epoch_rubric(
        rubric, split_candidates, refine_candidates)
    merged = refine._merge_epoch_predictions(
        prediction,
        [result["child_prediction"] for kind, _, result in selected
         if kind == "split"],
        {item_id: result["candidate_prediction"]
         for kind, item_id, result in selected if kind == "refine"},
        committed)
    return committed, merged


def _compose_candidate_reports(
    baseline: Mapping[str, Any], rubric: StructuredRubric,
    selected: Sequence[tuple[str, str, dict[str, Any]]],
) -> dict[str, Any]:
    """Compose frozen winner reports without resampling any subtree.

    Each independently evaluated candidate already contains exactly one newly
    generated root report and four baseline reports.  A joint competition must
    combine those frozen root reports and rerun only the Arbiter; otherwise
    sampling noise at temperature 0.5 becomes a hidden treatment variable.
    """
    result = copy.deepcopy(dict(baseline))
    baseline_samples = {
        str(item["sample_id"]): item for item in result["samples"]}
    for kind, item_id, candidate in selected:
        root_id = _candidate_root(rubric, kind, item_id)
        artifact = candidate["aligned_system_artifact"]
        candidate_samples = {
            str(item["sample_id"]): item for item in artifact["samples"]}
        if set(candidate_samples) != set(baseline_samples):
            raise RuntimeError("candidate system sample identity drift")
        for sample_id, sample in baseline_samples.items():
            source = candidate_samples[sample_id]
            if set(source["replicates"]) != set(sample["replicates"]):
                raise RuntimeError("candidate system replicate identity drift")
            for replicate, target_item in sample["replicates"].items():
                source_call = source["replicates"][replicate]["subtrees"].get(root_id)
                if not isinstance(source_call, Mapping) or not source_call.get("parse_ok"):
                    raise RuntimeError(
                        f"candidate root report is unresolved: {root_id}")
                target_item["subtrees"][root_id] = copy.deepcopy(dict(source_call))
    result["composed_candidate_root_ids"] = [
        _candidate_root(rubric, kind, item_id)
        for kind, item_id, _ in selected]
    return result


def _evaluate_candidates(
    config: Mapping[str, Any], target: Path, epoch_dir: Path,
    rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
    baseline: Mapping[str, Any], split_results: Mapping[str, dict[str, Any]],
    refine_results: Mapping[str, dict[str, Any]],
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates = ([
        ("split", item_id, result) for item_id, result in split_results.items()
        if result.get("candidate") is not None
    ] + [
        ("refine", item_id, result) for item_id, result in refine_results.items()
        if result.get("candidate") is not None
    ])
    positive = []
    baseline_metrics = baseline["metrics"]
    for kind, item_id, result in candidates:
        root_id = _candidate_root(rubric, kind, item_id)
        after = result["after_rubric"]
        artifact = _system_eval(
            config, target=target,
            output_path=Path(result["attempt_dir"]) / "aligned_system.json",
            split_name=f"epoch{epoch_dir.name}_{kind}_{canonical_sha256(item_id)[:8]}",
            rows=rows, rubric=after, baseline=baseline,
            changed_roots=(root_id,))
        paired_value = system.paired(baseline_metrics, artifact["metrics"], rows)
        result["aligned_system_artifact"] = artifact
        valid = artifact["metrics"]["technical_failure_rate"] == 0
        accepted = valid and paired_value["net_corrected"] > 0
        _set_system_outcome(
            result, accepted=accepted, paired_value=paired_value,
            kind=("independent_system_improvement" if accepted
                  else "no_system_effect" if paired_value["net_corrected"] == 0
                  else "subtree_evidence_regression"))
        if kind == "split" and not accepted:
            failure_kind = ("no_system_effect" if paired_value["net_corrected"] == 0
                            else "subtree_evidence_regression")
            refine._integrated_add_lock_diagnostics(
                result, rubric=rubric, rows=rows, settings=SETTINGS)
            _set_system_outcome(
                result, accepted=False, paired_value=paired_value,
                kind=failure_kind)
        _write(Path(result["attempt_dir"]) / "system_competition.json", {
            "schema_version": "1.0.0", "candidate_kind": kind,
            "candidate_id": item_id, "root_id": root_id,
            "before": baseline_metrics, "after": artifact["metrics"],
            "paired": paired_value, "accepted_independently": accepted,
            "local_decision_is_diagnostic_only": True,
        })
        if accepted:
            positive.append((kind, item_id, result))
    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for item in positive:
        grouped[_candidate_root(rubric, item[0], item[1])].append(item)
    winners = []
    for root_id, items in grouped.items():
        winner = max(items, key=_candidate_key)
        winners.append(winner)
        for item in items:
            if item is winner:
                continue
            paired_value = item[2]["system_evaluation"]
            _set_system_outcome(
                item[2], accepted=False, paired_value=paired_value,
                kind="same_root_candidate_superseded")
    return sorted(winners, key=lambda item: _candidate_root(rubric, item[0], item[1]))


def _joint_commit(
    config: Mapping[str, Any], target: Path, epoch_dir: Path,
    rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
    prediction: PairwisePredictionOutput, baseline: Mapping[str, Any],
    winners: Sequence[tuple[str, str, dict[str, Any]]],
) -> tuple[StructuredRubric, PairwisePredictionOutput, dict[str, Any],
           list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    if not winners:
        return rubric, prediction, dict(baseline), [], {
            "decision": "no_positive_independent_candidate", "winner_count": 0}
    if len(winners) == 1:
        committed, merged = _merge_selected(rubric, prediction, winners)
        return committed, merged, winners[0][2]["aligned_system_artifact"], list(winners), {
            "decision": "single_candidate_commit", "winner_count": 1,
            "committed": [f"{winners[0][0]}:{winners[0][1]}"],
        }
    joint_rubric, joint_prediction = _merge_selected(rubric, prediction, winners)
    roots = tuple(_candidate_root(rubric, item[0], item[1]) for item in winners)
    frozen_reports = _compose_candidate_reports(baseline, rubric, winners)
    joint = _system_eval(
        config, target=target,
        output_path=epoch_dir / "joint" / "aligned_system.json",
        split_name=f"{epoch_dir.name}_joint", rows=rows, rubric=joint_rubric,
        baseline=frozen_reports, changed_roots=())
    joint["composed_candidate_root_ids"] = list(roots)
    paired_value = system.paired(baseline["metrics"], joint["metrics"], rows)
    if (joint["metrics"]["technical_failure_rate"] == 0
            and paired_value["net_corrected"] > 0):
        return joint_rubric, joint_prediction, joint, list(winners), {
            "decision": "joint_commit", "winner_count": len(winners),
            "paired": paired_value,
            "committed": [f"{kind}:{item_id}" for kind, item_id, _ in winners],
        }
    best = max(winners, key=_candidate_key)
    committed, merged = _merge_selected(rubric, prediction, (best,))
    diagnostics = []
    for omitted in winners:
        subset = tuple(item for item in winners if item is not omitted)
        subset_rubric, _ = _merge_selected(rubric, prediction, subset)
        subset_roots = tuple(_candidate_root(rubric, item[0], item[1])
                             for item in subset)
        subset_reports = _compose_candidate_reports(baseline, rubric, subset)
        artifact = _system_eval(
            config, target=target,
            output_path=epoch_dir / "joint" / (
                f"leave_out_{canonical_sha256(omitted[0] + omitted[1])[:8]}.json"),
            split_name=f"{epoch_dir.name}_joint_leave_one_out",
            rows=rows, rubric=subset_rubric, baseline=subset_reports,
            changed_roots=())
        artifact["composed_candidate_root_ids"] = list(subset_roots)
        diagnostics.append({
            "omitted": f"{omitted[0]}:{omitted[1]}",
            "paired": system.paired(baseline["metrics"], artifact["metrics"], rows),
        })
    for item in winners:
        if item is best:
            continue
        _set_system_outcome(
            item[2], accepted=False,
            paired_value=item[2]["system_evaluation"],
            kind="joint_interaction_regression")
    return committed, merged, best[2]["aligned_system_artifact"], [best], {
        "decision": "joint_rejected_best_single_commit",
        "winner_count": len(winners), "joint_paired": paired_value,
        "committed": [f"{best[0]}:{best[1]}"],
        "leave_one_root_out_diagnostics": diagnostics,
    }


def _run_impl(config: Mapping[str, Any], target: Path, manifest: Mapping[str, Any],
              history: dict[str, Any]) -> None:
    runtime_config = _runtime_config(config)
    rows = _rows(config, "discovery")
    managers, _, specs, _ = split._managers(runtime_config, phase17.PROTOCOL)
    for stage in ("error_signature", "semantic_cluster", "child_generation"):
        if specs[stage] != manifest["manager_request_specs"][stage]:
            raise RuntimeError(f"aligned Manager request identity drift: {stage}")
    retry_managers, retry_specs = refine._integrated_retry_specs(
        runtime_config, phase17.PROTOCOL)
    if retry_specs != manifest["locked_retry_manager_request_specs"]:
        raise RuntimeError("aligned locked-retry Manager request identity drift")
    refine_manager, _ = refine._manager(runtime_config)
    pool = _pairwise_execution_pool(runtime_config)
    max_epochs = SETTINGS["max_epochs"]
    for epoch_no in range(int(history["current_epoch"]) + 1, max_epochs + 1):
        started_epoch = time.monotonic()
        epoch_dir = split._epoch(target, epoch_no)
        previous = split._epoch(target, epoch_no - 1)
        rubric = StructuredRubric.load_json(previous / "rubric_committed.json")
        prediction = PairwisePredictionOutput.load_json(
            previous / "discovery_pairwise.json")
        feedback = RubricFeedback.from_dict(load_json(previous / "feedback.json"))
        baseline = system.load(previous / "system" / "discovery.json")
        _require_complete_system(
            baseline, f"aligned epoch-{epoch_no - 1} discovery baseline")
        attributed = system.attributed_error_ids(baseline, rows)
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        split_scheduled = split.retryable_roots(history)
        refine_scheduled = refine._integrated_refine_schedule(
            rubric, feedback, runtime_config, split_scheduled)
        print(
            f"aligned evolution epoch={epoch_no} split={list(split_scheduled)} "
            f"refine={list(refine_scheduled)}", flush=True)
        split_results: dict[str, dict[str, Any]] = {}
        split_started: dict[str, float] = {}
        for root_id in split_scheduled:
            split_started[root_id] = time.monotonic()
            state = history["root_states"][root_id]
            attempt_no = int(state["attempt_count"]) + 1
            try:
                if state.get("locked_retry"):
                    result = refine._integrated_locked_retry_prepare(
                        config=runtime_config, target=target, epoch_dir=epoch_dir,
                        root_id=root_id, attempt_no=attempt_no, rubric=rubric,
                        prediction=prediction, feedback=feedback, rows=rows,
                        history=history, managers=retry_managers,
                        settings=SETTINGS, rubric_memory=memory,
                        rubric_memory_sha256=memory_hash, pool=pool,
                        defer_rejection_attribution=True)
                else:
                    result = split._prepare(
                        runtime_config, target, epoch_dir, root_id, attempt_no,
                        rubric, prediction, feedback, rows, history, managers,
                        specs, phase17.PROTOCOL, memory, memory_hash,
                        decisive_sample_allowlist=attributed.get(root_id, ()))
                split_results[root_id] = result
            except (split.TransportFailed, split.SpecializeManagerFailure) as exc:
                split._pause_transport(
                    target, epoch_no, root_id, attempt_no,
                    getattr(exc, "stage", "split_candidate"),
                    split._failure_details(exc))
                return
            except split.ProposalInvalid as exc:
                attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
                payload = split._proposal_failure_payload(
                    attempt_dir, exc.stage, split._failure_details(exc))
                _write(attempt_dir / "proposal_failure.json", payload)
                split_results[root_id] = {
                    "root_id": root_id, "decision": split.PROPOSAL_INVALID,
                    "attempt_dir": attempt_dir, "history_payload": payload,
                    "lock": (state.get("locked_retry") or {}).get("lock")}

        prepared = {root_id: value["candidate"]
                    for root_id, value in split_results.items()
                    if value.get("candidate") is not None}
        collisions = split.colliding_roots(prepared)
        collided = {root_id for owners in collisions.values() for root_id in owners}
        for root_id in collided:
            split_results[root_id]["decision"] = "cross_root_collision"
            split_results[root_id]["history_payload"] = split._failure(
                split_results[root_id], "cross_root_child_name_collision",
                "synchronous_commit", details={"collisions": collisions})
        for root_id in sorted(set(prepared) - collided):
            if split_results[root_id]["decision"] != "prepared":
                continue
            result = split._evaluate(
                runtime_config, epoch_dir, rows, rubric, prediction,
                split_results[root_id], pool, managers["semantic_cluster"],
                feedback, phase17.PROTOCOL,
                defer_rejection_attribution=True)
            refine._integrated_add_lock_diagnostics(
                result, rubric=rubric, rows=rows, settings=SETTINGS)
            split_results[root_id] = result

        refine_results: dict[str, dict[str, Any]] = {}
        refine_started: dict[str, float] = {}
        for node_id in refine_scheduled:
            refine_started[node_id] = time.monotonic()
            attempt_no = int(history["refine_states"].get(
                node_id, {"attempt_count": 0})["attempt_count"]) + 1
            root_id = rubric.root_id_for(node_id)
            try:
                refine_results[node_id] = refine._prepare_refine_attempt(
                    config=runtime_config, epoch_dir=epoch_dir,
                    node_id=node_id, attempt_no=attempt_no, rubric=rubric,
                    prediction=prediction, feedback=feedback, rows=rows,
                    history=history, manager=refine_manager, pool=pool,
                    rubric_memory=memory, rubric_memory_sha256=memory_hash,
                    trigger_mode="role_aware_v2",
                    evidence_sample_allowlist=attributed.get(root_id, ()),
                    defer_rejection_attribution=True)
            except (refine.RefineTransportPause,
                    refine.RefineAttributionInvalid) as exc:
                status = load_json(target / "stage_status.json")
                status["run"] = {"status": "paused", "details": {
                    "epoch": epoch_no, "node_id": node_id,
                    "message": str(exc), "history_appended": False}}
                _write(target / "stage_status.json", status)
                return

        winners = _evaluate_candidates(
            config, target, epoch_dir, rows, rubric, baseline,
            split_results, refine_results)
        committed, committed_prediction, committed_system, committed_items, joint = (
            _joint_commit(
                config, target, epoch_dir, rows, rubric, prediction,
                baseline, winners))
        committed_ids = {(kind, item_id) for kind, item_id, _ in committed_items}
        for kind, item_id, result in winners:
            if (kind, item_id) not in committed_ids:
                _set_system_outcome(
                    result, accepted=False,
                    paired_value=result["system_evaluation"],
                    kind="joint_interaction_regression")
        split._snapshot(epoch_dir, committed, committed_prediction, rows, manifest)
        committed_system_path = epoch_dir / "system" / "discovery.json"
        _write(committed_system_path, committed_system)
        _require_complete_system(
            committed_system, f"aligned epoch-{epoch_no} committed system")
        _write(epoch_dir / "arbiter_aligned_feedback.json",
               _aligned_feedback(committed_system, rows, committed))
        split_records = refine._integrated_record_split_history(
            history, split_scheduled, split_results, split_started,
            epoch_no=epoch_no, max_epochs=max_epochs)
        refine_records = [refine._record_refine_state(
            history, refine_results[node_id], epoch_no,
            time.monotonic() - refine_started[node_id])
            for node_id in refine_scheduled]
        history["current_epoch"] = epoch_no
        history["joint_commits"].append({"epoch": epoch_no, **joint})
        committed_feedback = RubricFeedback.from_dict(
            load_json(epoch_dir / "feedback.json"))
        next_split = split.retryable_roots(history)
        next_refine = refine._integrated_refine_schedule(
            committed, committed_feedback, runtime_config, next_split)
        if epoch_no >= SETTINGS["min_epochs"] and not (next_split or next_refine):
            history.update({"completed": True,
                            "stop_reason": "no_retryable_operations"})
        elif epoch_no == max_epochs:
            history.update({"completed": True,
                            "stop_reason": "max_epochs_reached"})
        accepted_split = sorted(item_id for kind, item_id, _ in committed_items
                                if kind == "split")
        accepted_refine = sorted(item_id for kind, item_id, _ in committed_items
                                 if kind == "refine")
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no,
            "split_scheduled": list(split_scheduled),
            "refine_scheduled": list(refine_scheduled),
            "accepted_split_roots": accepted_split,
            "accepted_refine_nodes": accepted_refine,
            "split_attempts": split_records,
            "refine_attempts": refine_records,
            "joint_commit": joint,
            "system": committed_system["metrics"],
            "next_split_retryable": list(next_split),
            "next_refine_eligible": list(next_refine),
            "rubric_node_count": len(committed.nodes),
            "epoch_wall_seconds": time.monotonic() - started_epoch,
        })
        _write(target / "evolution_history.json", history)
        print(
            f"aligned evolution epoch={epoch_no} split_accept={accepted_split} "
            f"refine_accept={accepted_refine} "
            f"strict_acc={committed_system['metrics']['strict_accuracy']:.4f}",
            flush=True)
        if history["completed"]:
            break
    status = load_json(target / "stage_status.json")
    status["run"] = {"status": "passed" if history["completed"] else "partial",
                     "details": {"current_epoch": history["current_epoch"],
                                 "stop_reason": history["stop_reason"]}}
    _write(target / "stage_status.json", status)


def _dev_epoch(config: Mapping[str, Any], target: Path, epoch_no: int) -> dict[str, Any]:
    epoch_dir = split._epoch(target, epoch_no)
    report_path = epoch_dir / "dev150" / "system_report.json"
    rubric = StructuredRubric.load_json(epoch_dir / "rubric_committed.json")
    if report_path.is_file():
        value = load_json(report_path)
        if value["rubric_sha256"] != rubric.rubric_sha256:
            raise RuntimeError("aligned Dev report rubric drift")
        return value
    rows = _rows(config, "dev")
    artifact = _system_eval(
        config, target=target,
        output_path=epoch_dir / "dev150" / "system.json",
        split_name=f"aligned_dev_epoch_{epoch_no:02d}", rows=rows, rubric=rubric)
    value = {
        "schema_version": "1.0.0", "epoch": epoch_no,
        "rubric_sha256": rubric.rubric_sha256,
        "node_count": len(rubric.nodes), "system": artifact["metrics"],
        "diagnostic_only": True, "manager_visible": False,
        "selection_forbidden": True,
    }
    _write(report_path, value)
    return value


def _ensure_dev(config: Mapping[str, Any], target: Path) -> None:
    history = load_json(target / "evolution_history.json")
    reports = [_dev_epoch(config, target, epoch)
               for epoch in range(int(history["current_epoch"]) + 1)]
    initial = reports[0]["system"]["predictions"]
    rows = _rows(config, "dev")
    trajectory = []
    for report in reports:
        paired_value = system.paired(
            {**reports[0]["system"], "predictions": initial},
            report["system"], rows)
        trajectory.append({
            "epoch": report["epoch"], "rubric_sha256": report["rubric_sha256"],
            "strict_accuracy": report["system"]["strict_accuracy"],
            "coverage": report["system"]["coverage"],
            "corrected_vs_epoch0": paired_value["corrected"],
            "harmed_vs_epoch0": paired_value["harmed"],
            "net_corrected_vs_epoch0": paired_value["net_corrected"],
        })
    _write(target / "dev150_trajectory.json", {
        "schema_version": "1.0.0", "diagnostic_only": True,
        "selection_forbidden": True, "epochs": trajectory})


def run(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    target = _target(output)
    status = load_json(target / "stage_status.json")
    if status.get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run aligned-evolution-smoke first")
    manifest = _load_manifest(config, output)
    _verify_live_endpoints(config, manifest)
    _initialize(config, target, manifest)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        _run_impl(config, target, manifest, history)
    _ensure_dev(config, target)


def report(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    target = _target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run aligned-evolution-run to completion first")
    _ensure_dev(config, target)
    final_epoch = int(history["current_epoch"])
    epoch_dir = split._epoch(target, final_epoch)
    rubric = StructuredRubric.load_json(epoch_dir / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(
        epoch_dir / "discovery_pairwise.json")
    final_system = system.load(epoch_dir / "system" / "discovery.json")
    initial_system = system.load(split._epoch(target, 0) / "system" / "discovery.json")
    rows = _rows(config, "discovery")
    comparison = system.paired(initial_system["metrics"],
                               final_system["metrics"], rows)
    all_attempts = tuple(history["attempts"]) + tuple(history["refine_attempts"])
    failure_type_counts: dict[str, int] = {}
    transition_counts = {"accepted": 0, "rejected": 0, "tie_rejected": 0}
    system_deltas = []
    for attempt in all_attempts:
        evaluation = attempt.get("system_evaluation")
        if isinstance(evaluation, Mapping):
            system_deltas.append({
                "epoch": attempt["epoch"],
                "operator": "split" if "root_id" in attempt else "refine",
                "target_id": attempt.get("root_id", attempt.get("node_id")),
                **dict(evaluation),
            })
            if evaluation.get("accepted"):
                transition_counts["accepted"] += 1
            else:
                transition_counts["rejected"] += 1
                if evaluation.get("corrected") == evaluation.get("harmed"):
                    transition_counts["tie_rejected"] += 1
        payload = attempt.get("history_payload")
        if not isinstance(payload, Mapping):
            continue
        failure = payload.get("structured_failure")
        if isinstance(failure, Mapping):
            code = str(failure.get("code", "unspecified"))
            failure_type_counts[code] = failure_type_counts.get(code, 0) + 1
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    _write(final_dir / "system_discovery.json", final_system)
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "final_epoch": final_epoch,
        "initial_system": initial_system["metrics"],
        "final_system": final_system["metrics"],
        "paired_vs_initial": comparison,
        "final_rubric_sha256": rubric.rubric_sha256,
        "node_count_initial": 5, "node_count_final": len(rubric.nodes),
        "split_attempts": history["attempts"],
        "refine_attempts": history["refine_attempts"],
        "joint_commits": history["joint_commits"],
        "operator_system_transitions": system_deltas,
        "operator_transition_counts": transition_counts,
        "failure_type_counts": failure_type_counts,
        "dev_trajectory": load_json(target / "dev150_trajectory.json"),
        "dev_used_for_selection": False,
        "heldout_accessed": False,
        "vl_rewardbench_required": True,
    }
    _write(final_dir / "discovery_report.json", value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_epoch": final_epoch,
        "strict_accuracy": final_system["metrics"]["strict_accuracy"],
        "final_rubric_sha256": rubric.rubric_sha256}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["report"]["details"], indent=2))


def heldout(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_manifest(config, output)
    _verify_live_endpoints(config, manifest)
    target = _target(output)
    if not (target / "final" / "discovery_report.json").is_file():
        raise RuntimeError("run aligned-evolution-report first")
    rows = _rows(config, "heldout")
    initial = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    final = StructuredRubric.load_json(target / "final" / "rubric.json")
    initial_artifact = _system_eval(
        config, target=target,
        output_path=target / "heldout500" / "initial_system.json",
        split_name="aligned_heldout_initial", rows=rows, rubric=initial)
    final_artifact = _system_eval(
        config, target=target,
        output_path=target / "heldout500" / "final_system.json",
        split_name="aligned_heldout_final", rows=rows, rubric=final)
    comparison = system.paired(
        initial_artifact["metrics"], final_artifact["metrics"], rows)
    value = {
        "schema_version": "1.0.0", "exploratory": True,
        "selection_after_heldout_forbidden": True,
        "initial": initial_artifact["metrics"],
        "aligned_final": final_artifact["metrics"],
        "paired_vs_initial": comparison,
    }
    _write(target / "heldout500" / "report.json", value)
    manifest["heldout_accessed"] = True
    _write(target / "frozen_manifest.json", manifest)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        "initial_strict_accuracy": initial_artifact["metrics"]["strict_accuracy"],
        "final_strict_accuracy": final_artifact["metrics"]["strict_accuracy"],
        "net_corrected": comparison["net_corrected"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze,
        STAGES[1]: audit,
        STAGES[2]: smoke,
        STAGES[3]: run,
        STAGES[4]: report,
        STAGES[5]: heldout,
    }
    if stage not in actions:
        raise ValueError(f"unsupported aligned evolution stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

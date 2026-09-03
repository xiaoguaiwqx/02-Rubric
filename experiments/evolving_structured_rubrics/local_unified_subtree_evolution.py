"""Phase20: Phase17 operators with root-local Unified-Subtree competition.

Split candidates are selected by paired root-scope Strict ACC.  Refine keeps
the Phase17 node gate and adds a non-regression guard on the containing root.
The Global Arbiter is rerun only after synchronous epoch commit and never
participates in candidate acceptance or error-sample selection.
"""

from __future__ import annotations

from collections import Counter
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


EXPERIMENT_DIR = "phase20_local_unified_subtree_competition_v1"
PROTOCOL_VERSION = "local-unified-subtree-competition-evolution-v1"
STAGES = (
    "local-subtree-evolution-freeze",
    "local-subtree-evolution-audit",
    "local-subtree-evolution-smoke",
    "local-subtree-evolution-run",
    "local-subtree-evolution-report",
    "local-subtree-evolution-heldout",
)
SETTINGS = {
    **refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_V1,
    "protocol_version": PROTOCOL_VERSION,
    "source_protocol": phase17.PROTOCOL_VERSION,
    "candidate_scope": "phase17_locked_split_role_aware_refine",
    "split_acceptance": "frozen_root_scope_unified_subtree_net_corrected_gt_0",
    "refine_acceptance": "phase17_node_gate_and_root_scope_non_regression",
    "root_scope": "epoch_start_root_pairwise_decisive_samples",
    "none_is_wrong": True,
    "technical_failure_policy": "pause_without_history_commit",
    "candidate_commit": "all_independently_passing_candidates_synchronous",
    "same_root_winner": False,
    "subset_search": False,
    "global_arbiter_role": "post_commit_diagnostic_only",
    "error_signature_source": "root_or_node_local_errors_without_arbiter_filter",
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


class RootEvaluationTechnicalFailure(RuntimeError):
    """A root-only candidate has unresolved model or parse failures."""


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _write(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("local_unified_subtree_competition_v1_experiment")
    if value != SETTINGS:
        raise RuntimeError("local_unified_subtree_competition_v1_experiment drift")
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


def _endpoint_identities(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return base._inspect_endpoints(
        config, base.BackendPoolSpec.from_dict(config["backend_pool"]))


def _pairwise_execution_pool(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = base.BackendPoolSpec.from_dict(config["backend_pool"])
    if tuple(item.endpoint_id for item in spec.endpoints) != system.ENDPOINT_IDS:
        raise RuntimeError("local subtree evolution requires vllm-8000 + vllm-8001")
    return spec.to_dict()


def _verify_live_endpoints(
    config: Mapping[str, Any], manifest: Mapping[str, Any],
) -> None:
    if manifest.get("worker_endpoint_identities") != _endpoint_identities(config):
        raise RuntimeError("local subtree evolution Worker endpoint identity drift")


def _load_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _settings(config)
    path = _target(output) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run local-subtree-evolution-freeze first")
    value = load_json(path)
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("local subtree evolution manifest identity drift")
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


def _require_complete_system(value: Mapping[str, Any], label: str) -> None:
    failures = int(value["metrics"]["technical_failure_count"])
    if failures:
        raise RuntimeError(f"{label} has {failures} unresolved system calls")


def _root_scope_ids(
    prediction: PairwisePredictionOutput, rubric: StructuredRubric,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    if len(prediction.node_outputs) != len(rows):
        raise ValueError("root scope prediction length mismatch")
    for root_id in rubric.root_ids:
        criterion = rubric.get_node(root_id).criterion.name
        sample_ids = []
        for row, outputs in zip(rows, prediction.node_outputs):
            item = outputs.get(criterion)
            if item is None:
                raise ValueError(f"root scope prediction is missing {criterion}")
            if item.vote.value in {"A", "B"}:
                sample_ids.append(str(row["sample_id"]))
        values[root_id] = tuple(sample_ids)
    return values


def _freeze_scopes(
    epoch_dir: Path, prediction: PairwisePredictionOutput,
    rubric: StructuredRubric, rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    scopes = _root_scope_ids(prediction, rubric, rows)
    value = {
        "schema_version": "1.0.0",
        "rubric_sha256": rubric.rubric_sha256,
        "prediction_sha256": canonical_sha256(prediction.to_dict()),
        "definition": "epoch_start_root_pairwise_vote_in_A_or_B",
        "roots": {root_id: {
            "sample_ids": list(sample_ids), "support": len(sample_ids),
            "scope_sha256": canonical_sha256(list(sample_ids)),
        } for root_id, sample_ids in scopes.items()},
    }
    path = epoch_dir / "root_scopes.json"
    if path.is_file() and load_json(path) != value:
        raise RuntimeError(f"epoch root scope drift: {path}")
    _write(path, value)
    return scopes


def _root_baselines(
    epoch_dir: Path, system_value: Mapping[str, Any],
    rubric: StructuredRubric, rows: Sequence[Mapping[str, Any]],
    scopes: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    return {root_id: system.root_from_system(
        system_value, rows, root_id, scopes[root_id],
        output_path=epoch_dir / "root_baselines" / f"{root_id}.json")
        for root_id in rubric.root_ids}


def _root_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any],
    paired_value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_row = {str(row["sample_id"]): row for row in rows}
    before_calls = {str(item["sample_id"]): item["call"] for item in before["samples"]}
    after_calls = {str(item["sample_id"]): item["call"] for item in after["samples"]}

    def cases(sample_ids: Sequence[str], limit: int) -> list[dict[str, Any]]:
        result = []
        for sample_id in tuple(sample_ids)[:limit]:
            row = by_row[sample_id]
            result.append({
                "sample_id": sample_id, "image_path": row["image_path"],
                "question": row["question"],
                "A": row["A"], "B": row["B"], "gold": row["answer"],
                "before": before_calls[sample_id].get("parsed"),
                "after": after_calls[sample_id].get("parsed"),
            })
        return result

    return {
        "schema_version": "1.0.0",
        "root_scope_evaluation": dict(paired_value),
        "harmed_cases": cases(
            paired_value["root_scope_harmed_sample_ids"], 6),
        "corrected_cases": cases(
            paired_value["root_scope_corrected_sample_ids"], 3),
    }


def _evaluate_root_candidate(
    config: Mapping[str, Any], *, target: Path, epoch_dir: Path,
    rows: Sequence[Mapping[str, Any]], root_id: str,
    result: dict[str, Any], baseline: Mapping[str, Any],
    scope_sample_ids: Sequence[str], operator: str,
) -> dict[str, Any]:
    artifact = system.evaluate_root(
        config,
        output_path=Path(result["attempt_dir"]) / "root_unified_subtree.json",
        cache_dir=target / "root_candidate_cache",
        split_name=(f"{epoch_dir.name}_{operator}_"
                    f"{canonical_sha256(str(result['attempt_dir']))[:8]}"),
        rows=rows, rubric=result["after_rubric"], root_id=root_id,
        scope_sample_ids=scope_sample_ids,
        settings=_runtime_settings(config))
    comparison = system.paired_root(baseline, artifact, rows)
    if comparison["technical_failure_count"]:
        raise RootEvaluationTechnicalFailure(
            f"{operator} {root_id} has unresolved root report calls")
    result["root_scope_artifact"] = artifact
    result["root_scope_evaluation"] = comparison
    result["system_evaluation"] = comparison
    _write(Path(result["attempt_dir"]) / "root_scope_competition.json", {
        "schema_version": "1.0.0", "operator": operator,
        "root_id": root_id, "paired": comparison,
        "frozen_scope_sha256": canonical_sha256(list(scope_sample_ids)),
        "global_arbiter_used_for_acceptance": False,
    })
    root_evidence = _root_evidence(baseline, artifact, comparison, rows)
    result["root_scope_evidence"] = root_evidence
    _write(Path(result["attempt_dir"]) / "root_scope_evidence.json",
           root_evidence)
    return result


def _semantic_reason(
    before: Mapping[str, Any], after: Mapping[str, Any],
    paired_value: Mapping[str, Any],
) -> str:
    if (after["metrics"]["root_scope_none_count"]
            > before["metrics"]["root_scope_none_count"]):
        return "excessive_abstention"
    if paired_value["root_scope_harmed"] and not paired_value["root_scope_corrected"]:
        return "decision_rule_regression"
    return "subtree_reasoning_interference"


def _split_passes_root_guard(paired_value: Mapping[str, Any]) -> bool:
    return (paired_value["technical_failure_count"] == 0
            and paired_value["root_scope_net_corrected"] > 0)


def _refine_passes_gates(
    node_decision: str, paired_value: Mapping[str, Any],
) -> bool:
    return (node_decision == "accepted"
            and paired_value["technical_failure_count"] == 0
            and paired_value["root_scope_net_corrected"] >= 0)


def _augment_attribution(
    attribution: Mapping[str, Any], paired_value: Mapping[str, Any],
    semantic_reason: str,
    root_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach root-level evidence to the existing Manager attribution schema."""
    value = dict(attribution)
    body = dict(value["attribution"])
    categories = list(body.get("failure_categories", []))
    if semantic_reason not in categories:
        categories.append(semantic_reason)
    details = list(body.get("details", []))
    details.append(
        "Unified-Subtree root guard: corrected "
        f"{paired_value['root_scope_corrected']} / harmed "
        f"{paired_value['root_scope_harmed']} on frozen support "
        f"{paired_value['root_scope_support']}.")
    if root_evidence is not None:
        def compact_cases(name: str) -> list[dict[str, Any]]:
            return [
                {
                    "sample_id": item["sample_id"],
                    "gold": item["gold"],
                    "before": item["before"],
                    "after": item["after"],
                }
                for item in root_evidence.get(name, [])
            ]

        details.append(
            "Representative frozen-scope before/after cases: "
            f"harmed={json.dumps(compact_cases('harmed_cases'), ensure_ascii=False)}; "
            f"corrected={json.dumps(compact_cases('corrected_cases'), ensure_ascii=False)}.")
    avoid = list(body.get("avoid_next_time", []))
    avoid.append(
        "Preserve the root-level decisions on the recorded harmed samples while "
        "retaining the node-local improvement.")
    body.update({
        "summary": (str(body.get("summary", ""))
                    + " The candidate failed the frozen root Unified-Subtree guard.").strip(),
        "failure_categories": categories,
        "details": details,
        "avoid_next_time": avoid,
    })
    value["attribution"] = body
    return value


def _split_rejection(
    result: dict[str, Any], *, manager: Any, rubric: StructuredRubric,
    failure_stage: str, baseline: Mapping[str, Any],
) -> None:
    paired_value = result["root_scope_evaluation"]
    local_metrics = result["evaluation"].to_dict()
    local_metrics["root_scope_evaluation"] = dict(paired_value)
    semantic_reason = _semantic_reason(
        baseline, result["root_scope_artifact"], paired_value)
    root_evidence = result["root_scope_evidence"]
    attribution = split._required_failure_attribution(
        manager, Path(result["attempt_dir"]),
        parent=rubric.get_node(result["root_id"]),
        signatures=tuple(result["signatures"].values()),
        cluster_proposal=result["candidate"].cluster_proposal.to_dict(),
        children=result["candidate"].children,
        local_metrics=local_metrics,
        changed_predictions=result["changed_predictions"],
        retry_feedback=result.get("child_retry_diagnostics"))
    attribution = _augment_attribution(
        attribution, paired_value, semantic_reason, root_evidence)
    _write(Path(result["attempt_dir"]) / "failure_attribution.json", attribution)
    payload = split._failure(
        result, failure_stage, "root_scope_competition",
        attribution=attribution, details={
            "root_scope_evaluation": dict(paired_value),
            "semantic_reason": semantic_reason,
        })
    payload["root_scope_evaluation"] = dict(paired_value)
    payload["root_scope_evidence"] = root_evidence
    result["history_payload"] = payload
    result["decision"] = split.COMPETITION_REJECTED


def _refine_rejection(
    result: dict[str, Any], *, manager: Any, rubric: StructuredRubric,
    memory: Mapping[str, Any], failure_stage: str,
    baseline: Mapping[str, Any] | None = None,
) -> None:
    attribution = refine._required_attribution(
        manager, Path(result["attempt_dir"]),
        original_node=rubric.get_node(result["node_id"]),
        proposal=result["proposal"], evaluation=result["evaluation"],
        changed_predictions=load_json(
            Path(result["attempt_dir"]) / "changed_predictions.json"),
        rubric_memory=memory)
    paired_value = result.get("root_scope_evaluation")
    root_evidence = result.get("root_scope_evidence")
    semantic_reason = None
    if paired_value is not None and baseline is not None:
        semantic_reason = _semantic_reason(
            baseline, result["root_scope_artifact"], paired_value)
        attribution = _augment_attribution(
            attribution, paired_value, semantic_reason, root_evidence)
        _write(Path(result["attempt_dir"]) / "failure_attribution.json", attribution)
    payload = {
        "natural_language_attribution": attribution["attribution"],
        "structured_failure": {
            "code": failure_stage,
            "stage": ("node_competition" if paired_value is None
                      else "root_scope_competition"),
            "details": ({"root_scope_evaluation": dict(paired_value)}
                        if paired_value is not None else {}),
        },
        "node_evaluation": result["evaluation"].to_dict(),
        "root_scope_evaluation": paired_value,
        "root_scope_evidence": root_evidence,
        "proposed_description": result["proposal"].description,
    }
    if semantic_reason is not None:
        payload["semantic_reason"] = semantic_reason
        payload["evidence_sample_ids"] = {
            "harmed": list(paired_value["root_scope_harmed_sample_ids"]),
            "corrected": list(paired_value["root_scope_corrected_sample_ids"]),
        }
        payload["recommended_revision"] = (
            "Keep the node-local gain but preserve the root decisions on harmed samples.")
    result["history_payload"] = payload
    result["decision"] = "competition_rejected"


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    refine._validate_output(output)
    target = _target(output)
    discovery_rows = _rows(config, "discovery")
    dev_rows = _rows(config, "dev")
    rubric = base.build_multicrit_open_ended_init_rubric()
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 5:
        raise RuntimeError("local subtree evolution requires five initial roots")
    runtime = _runtime_config(config)
    contract = phase17._manager_contract(config)
    expected_spec = base._expected_pairwise_request_spec(runtime, discovery_rows)
    epoch0 = split._epoch(target, 0)
    memory, memory_hash = split.freeze_rubric_memory(epoch0, rubric)
    rubric.save_json(epoch0 / "rubric_initial.json")
    manifest = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION, "settings": settings,
        "initial_rubric_sha256": rubric.rubric_sha256,
        "initial_root_ids": list(rubric.root_ids),
        "datasets": {
            "discovery": {"path": phase17.DISCOVERY_PATH,
                          "count": len(discovery_rows),
                          "sha256": phase17._file_sha(phase17.DISCOVERY_PATH)},
            "dev": {"path": phase17.DEV_PATH, "count": len(dev_rows),
                    "sha256": phase17._file_sha(phase17.DEV_PATH)},
            "heldout": {"path": str(config["heldout_dataset"]), "count": 500,
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
        "global_arbiter_used_for_acceptance": False,
        "dev_visible_to_manager": False,
        "dev_affects_acceptance": False,
        "dev_affects_early_stop": False,
        "heldout_accessed": False,
    }
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("local subtree evolution frozen manifest drift")
        print("local-subtree-evolution-freeze already completed")
        return
    _write(path, manifest)
    _write(target / "stage_status.json", {"freeze": {
        "status": "passed", "details": {
            "discovery_count": len(discovery_rows),
            "dev_count": len(dev_rows), "heldout_accessed": False}}})
    print(json.dumps({"target": str(target), "discovery_count": len(discovery_rows),
                      "dev_count": len(dev_rows)}, indent=2))


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
        "split_root_scope_acceptance": manifest["settings"]["split_acceptance"]
            == "frozen_root_scope_unified_subtree_net_corrected_gt_0",
        "refine_dual_gate": manifest["settings"]["refine_acceptance"]
            == "phase17_node_gate_and_root_scope_non_regression",
        "arbiter_not_in_acceptance": manifest["global_arbiter_used_for_acceptance"] is False,
        "dev_selection_forbidden": manifest["dev_affects_acceptance"] is False,
        "heldout_not_accessed": manifest["heldout_accessed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"local subtree evolution audit failed: {checks}")
    _write(target / "offline_audit.json", {
        "schema_version": "1.0.0", "offline_only": True, "checks": checks})
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": checks}
    _write(target / "stage_status.json", status)
    print(json.dumps(checks, indent=2))


def _replace_root_description(
    rubric: StructuredRubric, root_id: str, suffix: str,
) -> StructuredRubric:
    old = rubric.get_node(root_id)
    replacement = RubricNode(
        node_id=old.node_id,
        criterion=RubricCriterionSnapshot(
            old.criterion.name, old.criterion.description + suffix,
            old.criterion.score),
        examples=old.examples, lineage=old.lineage)
    nodes = dict(rubric.nodes)
    nodes[root_id] = replacement
    return StructuredRubric(nodes, rubric.edges, rubric.root_ids)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_manifest(config, output)
    _verify_live_endpoints(config, manifest)
    target = _target(output)
    status = load_json(target / "stage_status.json")
    if status.get("audit", {}).get("status") != "passed":
        raise RuntimeError("run local-subtree-evolution-audit first")
    report_path = target / "smoke" / "report.json"
    if report_path.is_file():
        print(json.dumps(load_json(report_path), indent=2))
        return
    rows = _rows(config, "discovery")[:20]
    rubric = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    prediction = phase17._worker_prediction(
        config, target / "smoke" / "pairwise", rubric, rows, "smoke_pairwise")
    scopes = _root_scope_ids(prediction, rubric, rows)
    baseline_system = _system_eval(
        config, target=target, output_path=target / "smoke" / "system_before.json",
        split_name="local_subtree_smoke_before", rows=rows, rubric=rubric)
    _require_complete_system(baseline_system, "local subtree smoke baseline")
    root_id = rubric.root_ids[0]
    baseline = system.root_from_system(
        baseline_system, rows, root_id, scopes[root_id],
        output_path=target / "smoke" / "root_before.json")
    split_like = _replace_root_description(
        rubric, root_id, "\n\nSmoke Split candidate: synthesize the subtree evidence explicitly.")
    split_after = system.evaluate_root(
        config, output_path=target / "smoke" / "split_after.json",
        cache_dir=target / "smoke" / "cache", split_name="local_split_smoke",
        rows=rows, rubric=split_like, root_id=root_id,
        scope_sample_ids=scopes[root_id], settings=_runtime_settings(config))
    refine_like = _replace_root_description(
        split_like, root_id, "\nSmoke Refine candidate: abstain only when evidence is insufficient.")
    refine_after = system.evaluate_root(
        config, output_path=target / "smoke" / "refine_after.json",
        cache_dir=target / "smoke" / "cache", split_name="local_refine_smoke",
        rows=rows, rubric=refine_like, root_id=root_id,
        scope_sample_ids=scopes[root_id], settings=_runtime_settings(config))
    split_paired = system.paired_root(baseline, split_after, rows)
    refine_paired = system.paired_root(split_after, refine_after, rows)
    value = {
        "schema_version": "1.0.0", "status": "passed", "sample_count": 20,
        "root_scope_support": len(scopes[root_id]),
        "scope_candidate_independent": (
            split_after["metrics"]["root_scope_sample_ids"]
            == refine_after["metrics"]["root_scope_sample_ids"]
            == list(scopes[root_id])),
        "split_root_parse_valid": split_paired["technical_failure_count"] == 0,
        "refine_root_parse_valid": refine_paired["technical_failure_count"] == 0,
        "split_gate_computed": "root_scope_net_corrected" in split_paired,
        "refine_gate_computed": "root_scope_net_corrected" in refine_paired,
        "global_arbiter_candidate_calls": 0,
        "formal_evolution_mutated": False, "heldout_accessed": False,
    }
    required = (
        "scope_candidate_independent", "split_root_parse_valid",
        "refine_root_parse_valid", "split_gate_computed", "refine_gate_computed")
    if not all(value[key] for key in required):
        raise RuntimeError(f"local subtree evolution smoke failed: {value}")
    _write(report_path, value)
    status["smoke"] = {"status": "passed", "details": value}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2))


def _initialize(
    config: Mapping[str, Any], target: Path, manifest: dict[str, Any],
) -> None:
    epoch0 = split._epoch(target, 0)
    required = (
        epoch0 / "discovery_pairwise.json", epoch0 / "rubric_committed.json",
        epoch0 / "feedback.json", epoch0 / "system" / "discovery.json",
        epoch0 / "root_scopes.json", epoch0 / "summary.json",
        target / "evolution_history.json")
    if all(path.is_file() for path in required):
        return
    rows = _rows(config, "discovery")
    rubric = StructuredRubric.load_json(epoch0 / "rubric_initial.json")
    prediction = phase17._worker_prediction(
        config, target / "baseline", rubric, rows, "discovery_initial")
    _, feedback = split._snapshot(epoch0, rubric, prediction, rows, manifest)
    system_value = _system_eval(
        config, target=target, output_path=epoch0 / "system" / "discovery.json",
        split_name="local_epoch00_discovery", rows=rows, rubric=rubric)
    _require_complete_system(system_value, "local epoch-0 discovery baseline")
    scopes = _freeze_scopes(epoch0, prediction, rubric, rows)
    _root_baselines(epoch0, system_value, rubric, rows, scopes)
    thresholds = _runtime_config(config)["evolution_policy"]["trigger_thresholds"]
    context = EvolutionContext(rubric, feedback)
    triggers = {root_id: detect_specialize_trigger(
        context, root_id, thresholds).to_dict() for root_id in rubric.root_ids}
    eligible = [root_id for root_id in rubric.root_ids
                if triggers[root_id]["triggered"]]
    manifest.update({
        "initial_triggers": triggers, "initial_eligible_root_ids": eligible,
        "initial_eligible_root_count": len(eligible),
        "initial_prediction_sha256": canonical_sha256(prediction.to_dict())})
    _write(target / "frozen_manifest.json", manifest)
    _write(epoch0 / "summary.json", {
        "epoch": 0, "system": system_value["metrics"], "triggers": triggers,
        "root_scope_support": {key: len(value) for key, value in scopes.items()},
        "accepted_roots": []})
    _write(target / "evolution_history.json", {
        "schema_version": "1.0.0", "current_epoch": 0, "completed": False,
        "stop_reason": None,
        "root_states": {root_id: {
            "status": "eligible" if triggers[root_id]["triggered"] else "not_eligible",
            "attempt_count": 0, "accepted_epoch": None, "children": [],
            "locked_retry": None} for root_id in rubric.root_ids},
        "attempts": [], "refine_states": {}, "refine_attempts": [],
        "synchronous_commits": []})


def _record_interactions(
    epoch_dir: Path, baseline_roots: Mapping[str, Mapping[str, Any]],
    committed_system: Mapping[str, Any], rubric: StructuredRubric,
    rows: Sequence[Mapping[str, Any]], scopes: Mapping[str, Sequence[str]],
    changed_roots: Sequence[str],
) -> dict[str, Any]:
    values = {}
    for root_id in changed_roots:
        after = system.root_from_system(
            committed_system, rows, root_id, scopes[root_id],
            output_path=epoch_dir / "post_commit_roots" / f"{root_id}.json")
        values[root_id] = system.paired_root(baseline_roots[root_id], after, rows)
    result = {
        "schema_version": "1.0.0", "changed_roots": list(changed_roots),
        "root_interactions": values,
        "diagnostic_only": True, "rollback_forbidden": True,
    }
    _write(epoch_dir / "post_commit_interactions.json", result)
    return result


def _run_impl(
    config: Mapping[str, Any], target: Path, manifest: Mapping[str, Any],
    history: dict[str, Any],
) -> None:
    runtime_config = _runtime_config(config)
    rows = _rows(config, "discovery")
    managers, _, specs, _ = split._managers(runtime_config, phase17.PROTOCOL)
    retry_managers, retry_specs = refine._integrated_retry_specs(
        runtime_config, phase17.PROTOCOL)
    if retry_specs != manifest["locked_retry_manager_request_specs"]:
        raise RuntimeError("local subtree locked-retry Manager identity drift")
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
        baseline_system = system.load(previous / "system" / "discovery.json")
        _require_complete_system(
            baseline_system, f"local epoch-{epoch_no - 1} discovery baseline")
        scopes = _freeze_scopes(epoch_dir, prediction, rubric, rows)
        baseline_roots = _root_baselines(
            epoch_dir, baseline_system, rubric, rows, scopes)
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        split_scheduled = split.retryable_roots(history)
        refine_scheduled = refine._integrated_refine_schedule(
            rubric, feedback, runtime_config, split_scheduled)
        print(
            f"local subtree evolution epoch={epoch_no} "
            f"split={list(split_scheduled)} refine={list(refine_scheduled)}",
            flush=True)
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
                        specs, phase17.PROTOCOL, memory, memory_hash)
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
            try:
                result = split_results[root_id]
                # A locked retry has already rebuilt and evaluated its child
                # bundle in _integrated_locked_retry_prepare.  Reuse that
                # result, but do not skip the Phase20 root-level guard.  A
                # normal candidate still needs the regular child evaluation.
                if result.get("decision") == "prepared":
                    result = split._evaluate(
                        runtime_config, epoch_dir, rows, rubric, prediction,
                        result, pool, managers["semantic_cluster"],
                        feedback, phase17.PROTOCOL,
                        defer_rejection_attribution=True)
                legacy_decision = result["decision"]
                result = _evaluate_root_candidate(
                    config, target=target, epoch_dir=epoch_dir, rows=rows,
                    root_id=root_id, result=result,
                    baseline=baseline_roots[root_id],
                    scope_sample_ids=scopes[root_id], operator="split")
                result["legacy_specialized_decision"] = legacy_decision
                accepted = _split_passes_root_guard(
                    result["root_scope_evaluation"])
                if accepted:
                    result["decision"] = split.ACCEPTED
                    result["history_payload"] = None
                else:
                    # Strong-child locking is a retry policy for candidates
                    # rejected by the final Phase20 root guard.  Root-accepted
                    # candidates must not acquire a legacy retry lock.
                    result["decision"] = split.COMPETITION_REJECTED
                    refine._integrated_add_lock_diagnostics(
                        result, rubric=rubric, rows=rows, settings=SETTINGS)
                    _split_rejection(
                        result, manager=managers["semantic_cluster"],
                        rubric=rubric,
                        failure_stage="root_unified_subtree_regression",
                        baseline=baseline_roots[root_id])
                split_results[root_id] = result
            except RootEvaluationTechnicalFailure as exc:
                split._pause_transport(
                    target, epoch_no, root_id,
                    int(history["root_states"][root_id]["attempt_count"]) + 1,
                    "root_unified_subtree", {"message": str(exc)})
                return
            except (split.TransportFailed, split.AttributionInvalid) as exc:
                split._pause_transport(
                    target, epoch_no, root_id,
                    int(history["root_states"][root_id]["attempt_count"]) + 1,
                    getattr(exc, "stage", "split_failure_attribution"),
                    split._failure_details(exc))
                return

        refine_results: dict[str, dict[str, Any]] = {}
        refine_started: dict[str, float] = {}
        for node_id in refine_scheduled:
            refine_started[node_id] = time.monotonic()
            attempt_no = int(history["refine_states"].get(
                node_id, {"attempt_count": 0})["attempt_count"]) + 1
            root_id = rubric.root_id_for(node_id)
            try:
                result = refine._prepare_refine_attempt(
                    config=runtime_config, epoch_dir=epoch_dir,
                    node_id=node_id, attempt_no=attempt_no, rubric=rubric,
                    prediction=prediction, feedback=feedback, rows=rows,
                    history=history, manager=refine_manager, pool=pool,
                    rubric_memory=memory, rubric_memory_sha256=memory_hash,
                    trigger_mode="role_aware_v2",
                    defer_rejection_attribution=True)
                if result.get("candidate") is None:
                    refine_results[node_id] = result
                    continue
                node_decision = result["decision"]
                result["legacy_node_decision"] = node_decision
                if node_decision != "accepted":
                    _refine_rejection(
                        result, manager=refine_manager, rubric=rubric,
                        memory=memory, failure_stage="node_local_regression")
                    refine_results[node_id] = result
                    continue
                result = _evaluate_root_candidate(
                    config, target=target, epoch_dir=epoch_dir, rows=rows,
                    root_id=root_id, result=result,
                    baseline=baseline_roots[root_id],
                    scope_sample_ids=scopes[root_id], operator="refine")
                accepted = _refine_passes_gates(
                    node_decision, result["root_scope_evaluation"])
                if accepted:
                    result["decision"] = "accepted"
                    result["history_payload"] = None
                else:
                    _refine_rejection(
                        result, manager=refine_manager, rubric=rubric,
                        memory=memory,
                        failure_stage="root_unified_subtree_regression",
                        baseline=baseline_roots[root_id])
                refine_results[node_id] = result
            except (refine.RefineTransportPause, refine.RefineAttributionInvalid,
                    refine.RefineStageError, RootEvaluationTechnicalFailure) as exc:
                status = load_json(target / "stage_status.json")
                status["run"] = {"status": "paused", "details": {
                    "epoch": epoch_no, "node_id": node_id,
                    "stage": ("technical_failure" if isinstance(
                        exc, RootEvaluationTechnicalFailure)
                        else "refine_failure_attribution"),
                    "message": str(exc), "history_appended": False}}
                _write(target / "stage_status.json", status)
                return

        accepted_split = {
            root_id: result["candidate"] for root_id, result
            in split_results.items() if result["decision"] == split.ACCEPTED}
        accepted_refine = {
            node_id: result["candidate"] for node_id, result
            in refine_results.items() if result["decision"] == "accepted"}
        committed = refine._merge_epoch_rubric(
            rubric, accepted_split, accepted_refine)
        committed_prediction = refine._merge_epoch_predictions(
            prediction,
            [split_results[root_id]["child_prediction"]
             for root_id in sorted(accepted_split)],
            {node_id: refine_results[node_id]["candidate_prediction"]
             for node_id in accepted_refine}, committed)
        changed_roots = sorted(
            set(accepted_split)
            | {rubric.root_id_for(node_id) for node_id in accepted_refine})
        if changed_roots:
            committed_system = _system_eval(
                config, target=target,
                output_path=epoch_dir / "system" / "discovery.json",
                split_name=f"local_epoch_{epoch_no:02d}_committed",
                rows=rows, rubric=committed, baseline=baseline_system,
                changed_roots=changed_roots)
        else:
            committed_system = dict(baseline_system)
            _write(epoch_dir / "system" / "discovery.json", committed_system)
        _require_complete_system(
            committed_system, f"local epoch-{epoch_no} committed system")
        interactions = _record_interactions(
            epoch_dir, baseline_roots, committed_system, committed, rows,
            scopes, changed_roots)
        split._snapshot(epoch_dir, committed, committed_prediction, rows, manifest)
        split_records = refine._integrated_record_split_history(
            history, split_scheduled, split_results, split_started,
            epoch_no=epoch_no, max_epochs=max_epochs)
        for record in split_records:
            source = split_results[record["root_id"]]
            record["legacy_specialized_decision"] = source.get(
                "legacy_specialized_decision")
            record["root_scope_evaluation"] = source.get(
                "root_scope_evaluation")
            record["root_scope_evidence"] = source.get(
                "root_scope_evidence")
        refine_records = [refine._record_refine_state(
            history, refine_results[node_id], epoch_no,
            time.monotonic() - refine_started[node_id])
            for node_id in refine_scheduled]
        for record in refine_records:
            source = refine_results[record["node_id"]]
            record["legacy_node_decision"] = source.get("legacy_node_decision")
            record["root_scope_evaluation"] = source.get(
                "root_scope_evaluation")
            record["root_scope_evidence"] = source.get(
                "root_scope_evidence")
        history["current_epoch"] = epoch_no
        history["synchronous_commits"].append({
            "epoch": epoch_no,
            "accepted_split_roots": sorted(accepted_split),
            "accepted_refine_nodes": sorted(accepted_refine),
            "changed_roots": changed_roots,
            "post_commit_interactions": interactions})
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
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no, "split_scheduled": list(split_scheduled),
            "refine_scheduled": list(refine_scheduled),
            "accepted_split_roots": sorted(accepted_split),
            "accepted_refine_nodes": sorted(accepted_refine),
            "split_attempts": split_records, "refine_attempts": refine_records,
            "root_scope_support": {key: len(value) for key, value in scopes.items()},
            "post_commit_interactions": interactions,
            "global_arbiter_diagnostic": committed_system["metrics"],
            "next_split_retryable": list(next_split),
            "next_refine_eligible": list(next_refine),
            "rubric_node_count": len(committed.nodes),
            "epoch_wall_seconds": time.monotonic() - started_epoch})
        _write(target / "evolution_history.json", history)
        print(
            f"local subtree evolution epoch={epoch_no} "
            f"split_accept={sorted(accepted_split)} "
            f"refine_accept={sorted(accepted_refine)} "
            f"diagnostic_strict_acc={committed_system['metrics']['strict_accuracy']:.4f}",
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
            raise RuntimeError("local subtree Dev report rubric drift")
        return value
    rows = _rows(config, "dev")
    artifact = _system_eval(
        config, target=target,
        output_path=epoch_dir / "dev150" / "system.json",
        split_name=f"local_dev_epoch_{epoch_no:02d}", rows=rows, rubric=rubric)
    value = {
        "schema_version": "1.0.0", "epoch": epoch_no,
        "rubric_sha256": rubric.rubric_sha256, "node_count": len(rubric.nodes),
        "system": artifact["metrics"], "diagnostic_only": True,
        "manager_visible": False, "selection_forbidden": True}
    _write(report_path, value)
    return value


def _ensure_dev(config: Mapping[str, Any], target: Path) -> None:
    history = load_json(target / "evolution_history.json")
    reports = [_dev_epoch(config, target, epoch)
               for epoch in range(int(history["current_epoch"]) + 1)]
    rows = _rows(config, "dev")
    trajectory = []
    for report in reports:
        paired_value = system.paired(reports[0]["system"], report["system"], rows)
        trajectory.append({
            "epoch": report["epoch"], "rubric_sha256": report["rubric_sha256"],
            "strict_accuracy": report["system"]["strict_accuracy"],
            "coverage": report["system"]["coverage"],
            "corrected_vs_epoch0": paired_value["corrected"],
            "harmed_vs_epoch0": paired_value["harmed"],
            "net_corrected_vs_epoch0": paired_value["net_corrected"]})
    _write(target / "dev150_trajectory.json", {
        "schema_version": "1.0.0", "diagnostic_only": True,
        "selection_forbidden": True, "epochs": trajectory})


def run(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    target = _target(output)
    status = load_json(target / "stage_status.json")
    if status.get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run local-subtree-evolution-smoke first")
    manifest = _load_manifest(config, output)
    _verify_live_endpoints(config, manifest)
    _initialize(config, target, manifest)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        _run_impl(config, target, manifest, history)
    if load_json(target / "evolution_history.json").get("completed"):
        _ensure_dev(config, target)


def report(config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    target = _target(output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run local-subtree-evolution-run to completion first")
    _ensure_dev(config, target)
    final_epoch = int(history["current_epoch"])
    epoch_dir = split._epoch(target, final_epoch)
    rubric = StructuredRubric.load_json(epoch_dir / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(
        epoch_dir / "discovery_pairwise.json")
    final_system = system.load(epoch_dir / "system" / "discovery.json")
    initial_system = system.load(split._epoch(target, 0) / "system" / "discovery.json")
    rows = _rows(config, "discovery")
    comparison = system.paired(initial_system["metrics"], final_system["metrics"], rows)
    attempts = tuple(history["attempts"]) + tuple(history["refine_attempts"])
    disagreements = Counter()
    root_transitions = []
    for item in attempts:
        evaluation = item.get("system_evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        local_decision = item.get("legacy_specialized_decision", item.get(
            "legacy_node_decision"))
        accepted = item["decision"] == "accepted"
        disagreements[f"legacy={local_decision}|root_accepted={accepted}"] += 1
        root_transitions.append({
            "epoch": item["epoch"],
            "operator": "split" if "root_id" in item else "refine",
            "target_id": item.get("root_id", item.get("node_id")),
            "legacy_decision": local_decision, "accepted": accepted,
            **dict(evaluation)})
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    _write(final_dir / "system_discovery.json", final_system)
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION, "final_epoch": final_epoch,
        "initial_system": initial_system["metrics"],
        "final_system": final_system["metrics"],
        "paired_vs_initial": comparison,
        "final_rubric_sha256": rubric.rubric_sha256,
        "node_count_initial": 5, "node_count_final": len(rubric.nodes),
        "split_attempts": history["attempts"],
        "refine_attempts": history["refine_attempts"],
        "synchronous_commits": history["synchronous_commits"],
        "root_scope_transitions": root_transitions,
        "legacy_vs_root_scope_decision_counts": dict(disagreements),
        "dev_trajectory": load_json(target / "dev150_trajectory.json"),
        "global_arbiter_used_for_acceptance": False,
        "dev_used_for_selection": False, "heldout_accessed": False,
        "vl_rewardbench_required": True}
    _write(final_dir / "discovery_report.json", value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_epoch": final_epoch,
        "diagnostic_strict_accuracy": final_system["metrics"]["strict_accuracy"],
        "final_rubric_sha256": rubric.rubric_sha256}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["report"]["details"], indent=2))


def heldout(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_manifest(config, output)
    _verify_live_endpoints(config, manifest)
    target = _target(output)
    if not (target / "final" / "discovery_report.json").is_file():
        raise RuntimeError("run local-subtree-evolution-report first")
    rows = _rows(config, "heldout")
    initial = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    final = StructuredRubric.load_json(target / "final" / "rubric.json")
    initial_artifact = _system_eval(
        config, target=target,
        output_path=target / "heldout500" / "initial_system.json",
        split_name="local_heldout_initial", rows=rows, rubric=initial)
    final_artifact = _system_eval(
        config, target=target,
        output_path=target / "heldout500" / "final_system.json",
        split_name="local_heldout_final", rows=rows, rubric=final)
    comparison = system.paired(
        initial_artifact["metrics"], final_artifact["metrics"], rows)
    value = {
        "schema_version": "1.0.0", "exploratory": True,
        "selection_after_heldout_forbidden": True,
        "initial": initial_artifact["metrics"],
        "local_final": final_artifact["metrics"],
        "paired_vs_initial": comparison}
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
        STAGES[0]: freeze, STAGES[1]: audit, STAGES[2]: smoke,
        STAGES[3]: run, STAGES[4]: report, STAGES[5]: heldout}
    if stage not in actions:
        raise ValueError(f"unsupported local subtree evolution stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

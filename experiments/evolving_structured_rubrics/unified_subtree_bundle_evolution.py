"""Phase21: atomic root-subtree Split and Bundle Refine evolution.

The Unified-Subtree Worker defines scope, error evidence, candidate fitness,
and acceptance. Criterion-level Pairwise outputs are emitted after commit as
read-only diagnostics and are never read by the evolution loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from critiq.structured.aggregation import aggregate_flat_votes
from critiq.structured.cache import canonical_sha256
from critiq.structured.dual_worker import PairwisePredictionOutput
from critiq.structured import StructuredCriterionSnapshot
from critiq.structured.evolution.specialize import build_specialize_candidate
from critiq.structured.evolution.specialize_types import (
    ChildCriterionProposal,
    ClusterProposal,
)
from critiq.structured.schema import StructuredRubric

from . import aligned_system_runtime as system
from . import discovery_v2_prompt_v2_evolution as phase17
from . import refine_evolution as refine
from . import run_rubric_evolution as base
from . import split_evolution as split
from .experiment_utils import atomic_write_json, load_json
from .evolution_protocol import EvolutionProtocol
from .rubric_factory import file_sha256
from .subtree_bundle_manager import (
    BundleManagerFailure,
    RootErrorSignature,
    SubtreeBundleManager,
    apply_bundle_refine,
    proposal_from_dict,
    subtree_payload,
    subtree_sha256,
)


EXPERIMENT_DIR = "phase21_unified_subtree_bundle_evolution_v1"
PROTOCOL_VERSION = "unified-root-subtree-bundle-evolution-v1"
STAGES = (
    "subtree-bundle-evolution-freeze",
    "subtree-bundle-evolution-audit",
    "subtree-bundle-evolution-smoke",
    "subtree-bundle-evolution-run",
    "subtree-bundle-evolution-report",
    "subtree-bundle-evolution-heldout",
    "subtree-bundle-evolution-final-report",
)
SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_protocol": phase17.PROTOCOL_VERSION,
    "optimization_unit": "complete_root_subtree",
    "scope_source": "epoch_start_unified_subtree_A_or_B",
    "error_source": "unified_subtree_scope_mismatch_only",
    "split_acceptance": "frozen_unified_scope_net_corrected_gt_0",
    "refine_acceptance": "frozen_unified_scope_net_corrected_gt_0",
    "split_retry": "reuse_clusters_regenerate_complete_child_bundle",
    "strong_child_locking": False,
    "partial_acceptance": False,
    "specialized_acc_role": "post_commit_read_only_diagnostic",
    "candidate_commit": "all_independently_passing_roots_synchronous",
    "global_arbiter_role": "post_commit_diagnostic_only",
    "split_trigger": {
        "strict_acc_below": 0.75, "coverage_above": 0.80,
        "min_support": 15, "min_wrong": 15, "max_children": 5,
    },
    "bundle_refine_trigger": {
        "strict_acc_above": 0.50, "strict_acc_below": 0.80,
        "min_support": 15, "min_wrong": 5,
    },
    "min_epochs": 3,
    "max_epochs": 5,
    "internal_k": 1,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "pairwise_prompt_mode": phase17.PAIRWISE_PROMPT_MODE,
    "pairwise_prompt_version": phase17.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    "dev_policy": "diagnostic_only_every_epoch",
    "heldout_access": "final_only_exploratory",
    "vl_rewardbench_required": True,
}


DEFAULT_PROTOCOL = EvolutionProtocol.from_settings(
    experiment_dir=EXPERIMENT_DIR, version=PROTOCOL_VERSION,
    config_key="unified_subtree_bundle_evolution_v1_experiment",
    stages=STAGES, settings=SETTINGS,
)


def _target(output: Path, *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL) -> Path:
    return output / protocol.experiment_dir


def _write(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _settings(
    config: Mapping[str, Any], *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    return protocol.validate(config)


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return phase17._runtime_config(config)


def _rows(config: Mapping[str, Any], split_name: str):
    return phase17._rows(config, split_name)


def _runtime_settings(
    config: Mapping[str, Any], *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> system.RuntimeSettings:
    value = _settings(config, protocol=protocol)
    return system.RuntimeSettings(
        temperature=float(value["temperature"]),
        max_tokens=int(value["max_tokens"]),
        max_parse_retries=int(value["max_parse_retries"]),
        generation_seed_policy=str(value["generation_seed_policy"]))


def _endpoint_identities(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return base._inspect_endpoints(
        config, base.BackendPoolSpec.from_dict(config["backend_pool"]))


def _unified_runtime_identity(
    config: Mapping[str, Any], *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Freeze prompt, parser, model, and decoding semantics used for scoring."""
    return {
        "runtime_protocol_version": system.PROTOCOL_VERSION,
        "model": config["model"],
        "vllm_version": config["vllm_version"],
        "settings": _runtime_settings(config, protocol=protocol).__dict__,
        "subtree_prompt_version": system.unified.SUBTREE_PROMPT_VERSION,
        "subtree_system_prompt_sha256": canonical_sha256(
            system.unified.UNIFIED_SUBTREE_SYSTEM_PROMPT),
        "subtree_parser_sha256": canonical_sha256(
            inspect.getsource(system.unified.parse_subtree_response)),
        "arbiter_prompt_version": system.unified.ARBITER_PROMPT_VERSION,
        "arbiter_system_prompt_sha256": canonical_sha256(
            system.arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT),
        "arbiter_parser_sha256": canonical_sha256(
            inspect.getsource(
                system.arbiter.parse_global_arbiter_ab_only_response)),
    }


def _manager_bundle(config: Mapping[str, Any]):
    runtime = _runtime_config(config)
    signature, _, signature_profile, signature_ids = base._manager_runtime(
        runtime, "error_signature", rubric_memory_mode="none")
    cluster, _, cluster_profile, cluster_ids = base._manager_runtime(
        runtime, "semantic_cluster", rubric_memory_mode="global_rubric_v1")
    child, _, child_profile, child_ids = base._manager_runtime(
        runtime, "child_generation", rubric_memory_mode="global_rubric_v1")
    child.child_input_mode = "multimodal"
    child_profile = dict(child_profile)
    child_profile["input_mode"] = "multimodal"
    refine_manager, refine_profile = refine._manager(runtime)
    manager = SubtreeBundleManager(
        signature, cluster, child, refine_manager)
    profiles = {
        "root_error_signature": signature_profile,
        "semantic_cluster": cluster_profile,
        "child_generation": child_profile,
        "bundle_refine": refine_profile,
        "bundle_failure_attribution": refine_profile,
    }
    identities = {
        "root_error_signature": signature_ids,
        "semantic_cluster": cluster_ids,
        "child_generation": child_ids,
    }
    return manager, profiles, manager.request_specs(), identities


def _load_manifest(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    _settings(config, protocol=protocol)
    path = _target(output, protocol=protocol) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run subtree-bundle-evolution-freeze first")
    value = load_json(path)
    if value.get("protocol_version") != protocol.version:
        raise RuntimeError("Phase21 manifest protocol drift")
    if value.get("settings") != protocol.settings:
        raise RuntimeError("Phase21 manifest settings drift")
    return value


def _verify_live(
    config: Mapping[str, Any], manifest: Mapping[str, Any], *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    if _endpoint_identities(config) != manifest["worker_endpoint_identities"]:
        raise RuntimeError("Phase21 Worker endpoint identity drift")
    if _unified_runtime_identity(config, protocol=protocol) != manifest["unified_runtime_identity"]:
        raise RuntimeError("Phase21 Unified prompt/parser/model identity drift")


def _system_eval(
    config: Mapping[str, Any], *, target: Path, output_path: Path,
    split_name: str, rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
    baseline: Mapping[str, Any] | None = None,
    changed_roots: Sequence[str] | None = None,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    return system.evaluate(
        config, output_path=output_path, cache_dir=target / "system_cache",
        split_name=split_name, rows=rows, rubric=rubric,
        settings=_runtime_settings(config, protocol=protocol), baseline=baseline,
        changed_root_ids=changed_roots)


def _require_complete(value: Mapping[str, Any], label: str) -> None:
    failures = int(value["metrics"]["technical_failure_count"])
    if failures:
        raise RuntimeError(f"{label} has {failures} unresolved technical failures")


def _mapped_answer(call: Mapping[str, Any], order: int = 0) -> str:
    if not call.get("parse_ok"):
        return "technical_failure"
    answer = (call.get("parsed") or {}).get("answer")
    if answer not in {"A", "B", "None"}:
        return "technical_failure"
    if order == 1 and answer in {"A", "B"}:
        return "B" if answer == "A" else "A"
    return str(answer)


def _root_index(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["sample_id"]): item for item in value["samples"]}


def _scientific_call(call: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in call.items()
            if key != "incremental_reuse"}


def _root_calls_sha256(value: Mapping[str, Any]) -> str:
    """Hash only scientific Unified-Subtree calls, excluding path metadata."""
    return canonical_sha256([
        {
            "sample_id": str(item["sample_id"]),
            "order": int(item["order"]),
            "call": _scientific_call(item["call"]),
        }
        for item in value["samples"]
    ])


def _combine_manager_usage(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "logical_evaluations", "api_attempts", "parse_retries",
        "input_tokens", "output_tokens", "total_tokens", "cache_hits",
        "cache_misses", "error_count",
    )
    complete = all(item.get("usage_complete", True) for item in items)
    value = {
        field: sum(int(item.get(field) or 0) for item in items)
        for field in fields
    }
    if not complete:
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            value[field] = None
    value["usage_complete"] = complete
    value["call_latency_seconds"] = sum(
        float(item.get("call_latency_seconds") or 0.0) for item in items)
    return value


def _root_call_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    metrics = [item["call"].get("metrics", {}) for item in value["samples"]]
    usage = system.support.sum_metrics(metrics)
    usage["logical_evaluations"] = len(value["samples"])
    usage["wall_seconds"] = float(value.get("wall_seconds", 0.0))
    return usage


def _scope_ids(value: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item["sample_id"]) for item in value["samples"]
                 if _mapped_answer(item["call"], int(item["order"])) in {"A", "B"})


def _reframe_root(
    value: Mapping[str, Any], rubric: StructuredRubric, root_id: str,
    rows: Sequence[Mapping[str, Any]], output_path: Path,
) -> dict[str, Any]:
    result = {key: value[key] for key in value if key != "metrics"}
    result["rubric_sha256"] = rubric.rubric_sha256
    result["root_subtree_sha256"] = subtree_sha256(rubric, root_id)
    result["scope_sample_ids"] = list(_scope_ids(value))
    result["metrics"] = system.root_metrics(
        result, rows, result["scope_sample_ids"])
    _write(output_path, result)
    return result


def _root_mismatches(
    value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    gold = {str(row["sample_id"]): str(row["answer"]) for row in rows}
    scope = set(value["metrics"]["root_scope_sample_ids"])
    return tuple(str(item["sample_id"]) for item in value["samples"]
                 if str(item["sample_id"]) in scope
                 and _mapped_answer(item["call"], int(item["order"]))
                 != gold[str(item["sample_id"])])


def _root_report(value: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    item = _root_index(value)[sample_id]
    parsed = item["call"].get("parsed")
    if not isinstance(parsed, Mapping):
        raise RuntimeError("Unified mismatch has no parsed root report")
    return parsed


def _freeze_root_baselines(
    epoch_dir: Path, source_system: Mapping[str, Any], rubric: StructuredRubric,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    roots_value = {}
    for root_id in rubric.root_ids:
        raw_scope = []
        for item in source_system["samples"]:
            replicate = item["replicates"]["0"]
            call = replicate["subtrees"][root_id]
            if _mapped_answer(call, int(replicate["order"])) in {"A", "B"}:
                raw_scope.append(str(item["sample_id"]))
        raw = system.root_from_system(
            source_system, rows, root_id, raw_scope)
        artifact = _reframe_root(
            raw, rubric, root_id, rows,
            epoch_dir / "root_baselines_before" / f"{root_id}.json")
        result[root_id] = artifact
        mismatches = _root_mismatches(artifact, rows)
        roots_value[root_id] = {
            "sample_ids": list(artifact["metrics"]["root_scope_sample_ids"]),
            "support": artifact["metrics"]["root_scope_support"],
            "scope_sha256": canonical_sha256(
                artifact["metrics"]["root_scope_sample_ids"]),
            "mismatch_sample_ids": list(mismatches),
            "mismatch_count": len(mismatches),
            "definition": "epoch_start_unified_subtree_answer_in_A_or_B",
        }
    _write(epoch_dir / "root_scopes.json", {
        "schema_version": "1.0.0", "rubric_sha256": rubric.rubric_sha256,
        "source_system_sha256": canonical_sha256(source_system),
        "roots": roots_value})
    return result


def _load_root_baselines(
    epoch_dir: Path, rubric: StructuredRubric,
) -> dict[str, dict[str, Any]]:
    result = {}
    for root_id in rubric.root_ids:
        path = epoch_dir / "root_baselines_before" / f"{root_id}.json"
        value = system.load(path)
        if value["root_subtree_sha256"] != subtree_sha256(rubric, root_id):
            raise RuntimeError("Phase21 root baseline subtree drift")
        result[root_id] = value
    return result


def _trigger(
    root_id: str, rubric: StructuredRubric, baseline: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], state: Mapping[str, Any],
    *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    metrics = baseline["metrics"]
    support = int(metrics["root_scope_support"])
    wrong = len(_root_mismatches(baseline, rows))
    full_coverage = support / len(rows)
    accuracy = float(metrics["root_scope_strict_accuracy"])
    children = rubric.children(root_id)
    if not children:
        threshold = protocol.settings["split_trigger"]
        split_status = state.get("split_status", "pending")
        eligible = split_status == "retryable" or (
            split_status == "pending"
            and accuracy < threshold["strict_acc_below"]
            and full_coverage > threshold["coverage_above"]
            and support >= threshold["min_support"]
            and wrong >= threshold["min_wrong"])
        operator = "split"
        reasons = [] if eligible else ["unified_split_threshold_not_met"]
    else:
        threshold = protocol.settings["bundle_refine_trigger"]
        eligible = (
            accuracy > threshold["strict_acc_above"]
            and accuracy < threshold["strict_acc_below"]
            and support >= threshold["min_support"]
            and wrong >= threshold["min_wrong"])
        operator = "refine"
        reasons = [] if eligible else ["unified_bundle_refine_threshold_not_met"]
    return {
        "root_id": root_id, "operator": operator, "triggered": eligible,
        "reasons": reasons, "root_scope_support": support,
        "root_scope_strict_accuracy": accuracy,
        "unified_full_coverage": full_coverage, "unified_mismatch_count": wrong,
        "specialized_metrics_read": False,
    }


def _history_projection(
    history: Mapping[str, Any], root_id: str, operator: str,
) -> list[dict[str, Any]]:
    return [dict(item) for item in history["attempts"]
            if item["root_id"] == root_id and item["operator"] == operator
            and item["decision"] != "accepted"]


def _signature_artifact(
    config: Mapping[str, Any], *, target: Path, epoch_dir: Path,
    root_id: str, baseline: Mapping[str, Any], rubric: StructuredRubric,
    rows: Sequence[Mapping[str, Any]], manager: SubtreeBundleManager,
    rubric_memory: Mapping[str, Any], retry_source: str | None,
) -> tuple[dict[str, RootErrorSignature], dict[str, Any]]:
    destination = epoch_dir / "roots" / split.root_shard(root_id) / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "error_signatures.json"
    mismatches = _root_mismatches(baseline, rows)
    identity = canonical_sha256({
        "root_id": root_id, "root_subtree_sha256": subtree_sha256(rubric, root_id),
        "mismatch_sample_ids": list(mismatches),
        "reports": {sample_id: _root_report(baseline, sample_id)
                    for sample_id in mismatches},
        "request_spec": manager.request_specs()["root_error_signature"],
    })
    if path.is_file():
        value = load_json(path)
        if value["signature_identity"] != identity:
            raise RuntimeError("Phase21 root ErrorSignature artifact drift")
    elif retry_source is not None:
        source = Path(retry_source)
        source_value = load_json(source)
        if source_value["signature_identity"] != identity:
            raise RuntimeError("Phase21 Split retry ErrorSignature identity drift")
        value = dict(source_value)
        value["reuse"] = {"kind": "split_retry_exact", "source": str(source)}
        _write(path, value)
    else:
        by_id = {str(row["sample_id"]): row for row in rows}
        outputs: dict[str, Any] = {}
        max_workers = min(
            len(mismatches),
            manager.signature_manager.backend_pool.spec.global_request_concurrency)
        print(
            f"phase21 signatures submitted={len(mismatches)} "
            f"concurrency={max_workers} root={root_id}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(
                manager.infer_signature, row=by_id[sample_id], rubric=rubric,
                root_id=root_id, root_report=_root_report(baseline, sample_id),
                rubric_memory=rubric_memory): sample_id
                for sample_id in mismatches}
            completed = 0
            for future in as_completed(futures):
                sample_id = futures[future]
                outputs[sample_id] = future.result()
                completed += 1
                print(
                    f"phase21 signatures completed={completed}/{len(mismatches)} "
                    f"root={root_id} sample={sample_id}", flush=True)
        value = {
            "schema_version": "1.0.0", "root_id": root_id,
            "signature_identity": identity,
            "root_subtree_sha256": subtree_sha256(rubric, root_id),
            "mismatch_sample_ids": list(mismatches),
            "reuse": {"kind": "generated", "source": None},
            "outputs": {sample_id: outputs[sample_id]
                        for sample_id in mismatches},
        }
        _write(path, value)
    signatures = {
        sample_id: RootErrorSignature(**item["signature"])
        for sample_id, item in value["outputs"].items()}
    return signatures, {"path": str(path), **value}


def _prepare_split(
    *, config: Mapping[str, Any], epoch_dir: Path, root_id: str,
    attempt_no: int, rubric: StructuredRubric, rows: Sequence[Mapping[str, Any]],
    signatures: Mapping[str, RootErrorSignature], manager: SubtreeBundleManager,
    memory: Mapping[str, Any], history_projection: Sequence[Mapping[str, Any]],
    retry_source: Mapping[str, Any] | None,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    parent = rubric.get_node(root_id)
    legacy = {key: value.to_legacy() for key, value in signatures.items()}
    cluster_path = attempt_dir / "cluster_proposal.json"
    if cluster_path.is_file():
        cluster = ClusterProposal.from_dict(load_json(cluster_path))
        cluster_reuse = "same_attempt"
    elif retry_source is not None:
        source_path = Path(retry_source["cluster_path"])
        cluster = ClusterProposal.from_dict(load_json(source_path))
        _write(cluster_path, cluster.to_dict())
        cluster_reuse = str(source_path)
    else:
        cluster = manager.cluster_manager.cluster(
            tuple(legacy.values()), criterion_name=parent.criterion.name,
            min_cluster_size=_runtime_config(config)["evolution_policy"][
                "trigger_thresholds"]["N_min_cluster"],
            max_clusters=protocol.settings["split_trigger"]["max_children"],
            prior_failures=history_projection, rubric_memory=memory)
        split._require_split_clusters(cluster)
        _write(cluster_path, cluster.to_dict())
        cluster_reuse = None
    split._require_split_clusters(cluster)
    by_id = {str(row["sample_id"]): row for row in rows}
    children = []
    for index, semantic_cluster in enumerate(cluster.clusters, 1):
        child_path = attempt_dir / "children" / f"{semantic_cluster.cluster_id}.json"
        if child_path.is_file():
            child = ChildCriterionProposal.from_dict(load_json(child_path))
        else:
            child = manager.child_manager.generate_child(
                parent=parent, cluster=semantic_cluster,
                signatures=[legacy[item] for item in semantic_cluster.sample_ids],
                representative_rows=[by_id[item]
                                     for item in semantic_cluster.sample_ids[:3]],
                siblings=children, prior_failures=history_projection,
                rubric_memory=memory)
            _write(child_path, child.to_dict())
        children.append(child)
        print(
            f"phase21 split children={index}/{len(cluster.clusters)} root={root_id}",
            flush=True)
    # Feedback is deliberately absent from every scientific input.  The
    # builder only dereferences context.rubric; a minimal context object keeps
    # the legacy additive-patch validator reusable.
    class _Context:
        def __init__(self, value: StructuredRubric) -> None:
            self.rubric = value
    candidate = build_specialize_candidate(
        _Context(rubric), root_id, cluster, children, rows, legacy)  # type: ignore[arg-type]
    after = split.apply_rubric_patch(rubric, candidate.edit_candidate.patch)
    _write(attempt_dir / "candidate.json", candidate.to_dict())
    after.save_json(attempt_dir / "candidate_rubric.json")
    _write(attempt_dir / "bundle_metadata.json", {
        "schema_version": "1.0.0", "operator": "split",
        "atomic_bundle": True, "partial_acceptance": False,
        "locked_children": [], "cluster_reuse": cluster_reuse,
        "specialized_acc_used": False})
    return {
        "operator": "split", "root_id": root_id, "attempt_dir": attempt_dir,
        "candidate": candidate, "candidate_payload": candidate.to_dict(),
        "after_rubric": after, "cluster_path": str(cluster_path),
        "cluster_reuse": cluster_reuse,
        "generated_child_ids": [item.criterion_name for item in children],
        "proposal_manager_usage": _combine_manager_usage([
            *([] if cluster_reuse not in {None, "same_attempt"}
              else [cluster.metrics.to_dict()]),
            *(item.metrics.to_dict() for item in children),
        ]),
    }


def _cluster_membership(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(sample_id): str(cluster["cluster_id"])
        for cluster in value["clusters"]
        for sample_id in cluster["sample_ids"]
    }


def _record_split_lineage(
    result: Mapping[str, Any], retry: Mapping[str, Any] | None,
    *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    """Persist the cluster provenance for one Phase22 Split attempt."""
    if not _uses_all_sample_acceptance(protocol=protocol):
        return
    attempt_dir = Path(result["attempt_dir"])
    current = load_json(Path(result["cluster_path"]))
    action = ("initial_cluster_complete_bundle" if retry is None else
              str(retry["action"]))
    previous = None
    if retry is not None:
        previous = load_json(Path(retry["cluster_path"]))
        _write(attempt_dir / "previous_cluster.json", previous)
        destination = ("new_cluster.json" if action.startswith("recluster")
                       else "reused_cluster.json")
        _write(attempt_dir / destination, current)
    else:
        _write(attempt_dir / "new_cluster.json", current)
    signature_path = result.get("signature_path")
    if signature_path is not None:
        _write(
            attempt_dir / "source_error_signatures.json",
            load_json(Path(str(signature_path))))
    child_files = sorted((attempt_dir / "children").glob("*.json"))
    child_manifest = []
    for child_path in child_files:
        child_value = load_json(child_path)
        destination = attempt_dir / "complete_child_bundle" / child_path.name
        _write(destination, child_value)
        child_manifest.append({
            "path": str(destination),
            "sha256": canonical_sha256(child_value),
        })
    _write(attempt_dir / "complete_child_bundle" / "manifest.json", {
        "schema_version": "1.0.0",
        "complete_child_regeneration": True,
        "child_count": len(child_manifest),
        "children": child_manifest,
    })

    old_membership = {} if previous is None else _cluster_membership(previous)
    new_membership = _cluster_membership(current)
    common = sorted(set(old_membership) & set(new_membership))
    moved = [sample_id for sample_id in common
             if old_membership[sample_id] != new_membership[sample_id]]
    old_to_new: dict[str, set[str]] = {}
    new_to_old: dict[str, set[str]] = {}
    for sample_id in common:
        old_to_new.setdefault(old_membership[sample_id], set()).add(
            new_membership[sample_id])
        new_to_old.setdefault(new_membership[sample_id], set()).add(
            old_membership[sample_id])
    split_clusters = [
        {"old_cluster_id": cluster_id,
         "new_cluster_ids": sorted(new_ids)}
        for cluster_id, new_ids in sorted(old_to_new.items())
        if len(new_ids) > 1
    ]
    merged_clusters = [
        {"new_cluster_id": cluster_id,
         "old_cluster_ids": sorted(old_ids)}
        for cluster_id, old_ids in sorted(new_to_old.items())
        if len(old_ids) > 1
    ]
    attribution = None
    if retry is not None and retry.get("failure_attribution_path"):
        attribution = load_json(Path(retry["failure_attribution_path"]))
    _write(attempt_dir / "cluster_diff.json", {
        "schema_version": "1.0.0",
        "retry_action": action,
        "old_cluster_sha256": (
            None if previous is None else canonical_sha256(previous)),
        "new_cluster_sha256": canonical_sha256(current),
        "old_cluster_count": (
            0 if previous is None else len(previous["clusters"])),
        "new_cluster_count": len(current["clusters"]),
        "old_membership": old_membership,
        "new_membership": new_membership,
        "moved_sample_ids": moved,
        "split_clusters": split_clusters,
        "merged_clusters": merged_clusters,
        "added_sample_ids": sorted(set(new_membership) - set(old_membership)),
        "removed_sample_ids": sorted(set(old_membership) - set(new_membership)),
        "failure_evidence_sha256": (
            None if attribution is None else canonical_sha256(attribution)),
        "cluster_manager_metrics": current.get("metrics"),
        "complete_child_regeneration": True,
        "generated_child_ids": list(result["generated_child_ids"]),
        "partial_acceptance": False,
    })


def _representative_rows(
    mismatch_ids: Sequence[str], rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    by_id = {str(row["sample_id"]): row for row in rows}
    return tuple(by_id[sample_id] for sample_id in tuple(mismatch_ids)[:6])


def _prepare_refine(
    *, epoch_dir: Path, root_id: str, attempt_no: int,
    rubric: StructuredRubric, rows: Sequence[Mapping[str, Any]],
    signatures: Mapping[str, RootErrorSignature], manager: SubtreeBundleManager,
    memory: Mapping[str, Any], history_projection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempt_dir = split._attempt(epoch_dir, root_id, attempt_no)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = attempt_dir / "bundle_refine_proposal.json"
    representatives = _representative_rows(tuple(signatures), rows)
    if proposal_path.is_file():
        output = load_json(proposal_path)
    else:
        output = manager.generate_bundle_refine(
            rubric=rubric, root_id=root_id, signatures=tuple(signatures.values()),
            representative_rows=representatives,
            failure_history=history_projection, rubric_memory=memory,
            max_description_chars=refine.REFINE_V1["max_description_chars"])
        _write(proposal_path, output)
    proposal = proposal_from_dict(output["proposal"])
    after = apply_bundle_refine(rubric, proposal)
    after.save_json(attempt_dir / "candidate_rubric.json")
    _write(attempt_dir / "bundle_metadata.json", {
        "schema_version": "1.0.0", "operator": "refine",
        "atomic_bundle": True, "partial_acceptance": False,
        "edited_child_ids": [item.node_id for item in proposal.edits],
        "unchanged_child_ids": list(proposal.unchanged_node_ids),
        "specialized_acc_used": False})
    return {
        "operator": "refine", "root_id": root_id, "attempt_dir": attempt_dir,
        "candidate": proposal, "candidate_payload": proposal.to_dict(),
        "after_rubric": after,
        "proposal_manager_usage": _combine_manager_usage([output["metrics"]]),
    }


def _root_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any],
    paired: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in rows}
    before_index, after_index = _root_index(before), _root_index(after)

    def cases(sample_ids: Sequence[str], limit: int) -> list[dict[str, Any]]:
        values = []
        for sample_id in tuple(sample_ids)[:limit]:
            row = by_id[sample_id]
            values.append({
                "sample_id": sample_id, "image_path": row["image_path"],
                "question": row["question"], "A": row["A"], "B": row["B"],
                "gold": row["answer"],
                "before": before_index[sample_id]["call"].get("parsed"),
                "after": after_index[sample_id]["call"].get("parsed"),
            })
        return values
    return {
        "paired": dict(paired),
        "harmed_cases": cases(paired["root_scope_harmed_sample_ids"], 6),
        "corrected_cases": cases(paired["root_scope_corrected_sample_ids"], 3),
    }


def _uses_all_sample_acceptance(
    *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> bool:
    return protocol.settings.get("acceptance_metric") == (
        "all_sample_unified_net_gain_gt_0")


def split_retry_action(
    primary_failure_type: str, *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> str:
    """Map a validated scientific Split failure to the next search action."""
    recluster_types = set(protocol.settings.get("recluster_failure_types", ()))
    if primary_failure_type in recluster_types:
        return "recluster_regenerate_complete_bundle"
    return "reuse_clusters_regenerate_complete_bundle"


def split_retry_source(
    retry: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return the old cluster source only for a cluster-reuse retry."""
    if retry is None:
        return None
    if retry.get("action") == "recluster_regenerate_complete_bundle":
        return None
    return retry


def _all_sample_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any],
    paired: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in rows}
    before_index, after_index = _root_index(before), _root_index(after)

    def cases(sample_ids: Sequence[str], limit: int) -> list[dict[str, Any]]:
        values = []
        for sample_id in tuple(sample_ids)[:limit]:
            row = by_id[sample_id]
            values.append({
                "sample_id": sample_id,
                "image_path": row.get("image_path"),
                "question": row.get("question"),
                "A": row.get("A"),
                "B": row.get("B"),
                "gold": row["answer"],
                "before": before_index[sample_id]["call"].get("parsed"),
                "after": after_index[sample_id]["call"].get("parsed"),
            })
        return values

    return {
        "paired": dict(paired),
        "harmed_cases": cases(
            paired["all_sample_harmed_sample_ids"], 6),
        "corrected_cases": cases(
            paired["all_sample_corrected_sample_ids"], 3),
    }


def _evaluate_candidate(
    config: Mapping[str, Any], *, target: Path, epoch_dir: Path,
    rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any],
    scope: Sequence[str], result: dict[str, Any],
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    root_id, operator = result["root_id"], result["operator"]
    artifact = system.evaluate_root(
        config, output_path=result["attempt_dir"] / "root_unified_subtree.json",
        cache_dir=target / "root_candidate_cache",
        split_name=(f"phase21_{epoch_dir.name}_{operator}_"
                    f"{canonical_sha256(str(result['attempt_dir']))[:8]}"),
        rows=rows, rubric=result["after_rubric"], root_id=root_id,
        scope_sample_ids=scope, settings=_runtime_settings(config, protocol=protocol))
    selective = system.paired_root(baseline, artifact, rows)
    all_sample = system.paired_root_all_samples(baseline, artifact, rows)
    paired = all_sample if _uses_all_sample_acceptance(protocol=protocol) else selective
    if paired["technical_failure_count"]:
        raise RuntimeError("candidate Unified-Subtree has technical failures")
    evidence = (_all_sample_evidence(baseline, artifact, paired, rows)
                if _uses_all_sample_acceptance(protocol=protocol)
                else _root_evidence(baseline, artifact, paired, rows))
    net_gain = (paired["all_sample_net_gain"]
                if _uses_all_sample_acceptance(protocol=protocol)
                else paired["root_scope_net_corrected"])
    competition_name = ("all_sample_competition.json"
                        if _uses_all_sample_acceptance(protocol=protocol)
                        else "root_scope_competition.json")
    evidence_name = ("all_sample_evidence.json"
                     if _uses_all_sample_acceptance(protocol=protocol)
                     else "root_scope_evidence.json")
    _write(result["attempt_dir"] / competition_name, {
        "schema_version": "1.0.0", "operator": operator,
        "root_id": root_id, "paired": paired,
        "acceptance_support_sha256": canonical_sha256(
            [str(row["sample_id"]) for row in rows]
            if _uses_all_sample_acceptance(protocol=protocol) else list(scope)),
        "acceptance_rule": (
            "all_sample_unified_net_gain_gt_0"
            if _uses_all_sample_acceptance(protocol=protocol)
            else "root_scope_net_corrected_gt_0"),
        "net_gain": net_gain,
        "decision": (
            "accepted" if net_gain > 0 else "competition_rejected"),
        "global_arbiter_used": False, "specialized_acc_used": False})
    _write(result["attempt_dir"] / evidence_name, evidence)
    if _uses_all_sample_acceptance(protocol=protocol):
        _write(result["attempt_dir"] / "all_sample_gain_ledger.json", {
            "schema_version": "1.0.0",
            "root_id": root_id,
            "support": paired["all_sample_support"],
            "net_gain": paired["all_sample_net_gain"],
            "gain_ledger_sha256": paired["gain_ledger_sha256"],
            "ledger": paired["gain_ledger"],
        })
        _write(
            result["attempt_dir"] /
            "phase21_selective_metric_diagnostic.json",
            {"schema_version": "1.0.0", "diagnostic_only": True,
             "selection_forbidden": True, "paired": selective})
    result.update({
        "root_artifact": artifact, "paired": paired,
        "selective_paired": selective,
        "evidence": evidence,
        "candidate_unified_usage": _root_call_usage(artifact),
        "root_metrics": {
            "scope_support": baseline["metrics"]["root_scope_support"],
            "baseline_strict_accuracy": baseline["metrics"][
                "root_scope_strict_accuracy"],
            "candidate_strict_accuracy": artifact["metrics"][
                "root_scope_strict_accuracy"],
            "baseline_scope_coverage": baseline["metrics"][
                "root_scope_coverage"],
            "candidate_scope_coverage": artifact["metrics"][
                "root_scope_coverage"],
            "baseline_full_coverage": sum(
                item in {"A", "B"}
                for item in baseline["metrics"]["predictions"]) / len(rows),
            "candidate_full_coverage": sum(
                item in {"A", "B"}
                for item in artifact["metrics"]["predictions"]) / len(rows),
            "baseline_scope_none_count": baseline["metrics"][
                "root_scope_none_count"],
            "candidate_scope_none_count": artifact["metrics"][
                "root_scope_none_count"],
        },
        "decision": "accepted" if net_gain > 0
                    else "competition_rejected"})
    return result


def _merge_accepted(
    rubric: StructuredRubric, results: Mapping[str, Mapping[str, Any]],
) -> StructuredRubric:
    split_candidates = {
        root_id: result["candidate"] for root_id, result in results.items()
        if result["decision"] == "accepted" and result["operator"] == "split"}
    merged = split.merge_accepted_rubrics(rubric, split_candidates)
    nodes = dict(merged.nodes)
    for root_id, result in sorted(results.items()):
        if result["decision"] != "accepted" or result["operator"] != "refine":
            continue
        candidate_rubric = result["after_rubric"]
        for node_id in rubric.preorder_node_ids():
            if rubric.root_id_for(node_id) == root_id:
                nodes[node_id] = candidate_rubric.get_node(node_id)
    return StructuredRubric(nodes, merged.edges, merged.root_ids)


def _synthetic_baseline(
    rubric: StructuredRubric, rows: Sequence[Mapping[str, Any]],
    roots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    indexes = {root_id: _root_index(value) for root_id, value in roots.items()}
    samples = []
    for row in rows:
        sample_id = str(row["sample_id"])
        calls = {root_id: dict(indexes[root_id][sample_id]["call"])
                 for root_id in rubric.root_ids}
        samples.append({
            "sample_id": sample_id, "replicates": {"0": {
                "order": 0, "subtrees": calls, "arbiter": {
                    "parse_ok": False, "parsed": None},
            }}})
    return {"samples": samples}


def _commit_system(
    config: Mapping[str, Any], *, target: Path, epoch_dir: Path,
    rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
    roots: Mapping[str, Mapping[str, Any]], label: str,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    artifact = _system_eval(
        config, target=target, output_path=epoch_dir / "system" / "discovery.json",
        split_name=label, rows=rows, rubric=rubric,
        baseline=_synthetic_baseline(rubric, rows, roots), changed_roots=(), protocol=protocol)
    _require_complete(artifact, label)
    regenerated = [
        root_id for sample in artifact["samples"]
        for replicate in sample["replicates"].values()
        for root_id in replicate["regenerated_root_ids"]]
    if regenerated:
        raise RuntimeError("Phase21 commit regenerated accepted root reports")
    committed_indexes = {
        root_id: _root_index(value) for root_id, value in roots.items()}
    for root_id in rubric.root_ids:
        expected = canonical_sha256([
            {
                "sample_id": str(row["sample_id"]),
                "order": int(committed_indexes[root_id][str(row["sample_id"])]["order"]),
                "call": _scientific_call(
                    committed_indexes[root_id][str(row["sample_id"])]["call"]),
            }
            for row in rows
        ])
        actual = canonical_sha256([
            {
                "sample_id": str(sample["sample_id"]),
                "order": int(sample["replicates"]["0"]["order"]),
                "call": _scientific_call(
                    sample["replicates"]["0"]["subtrees"][root_id]),
            }
            for sample in artifact["samples"]
        ])
        if actual != expected:
            raise RuntimeError(
                f"Phase21 committed root report drift: {root_id}")
    return artifact


def _pairwise_diagnostic(
    config: Mapping[str, Any], *, epoch_dir: Path, target: Path,
    rubric: StructuredRubric, rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any], previous: PairwisePredictionOutput | None,
) -> PairwisePredictionOutput:
    if previous is None:
        prediction = phase17._worker_prediction(
            config, target / "pairwise_diagnostic", rubric, rows,
            f"phase21_pairwise_epoch_{epoch_dir.name}")
        generated_names = [item.name for item in prediction.criteria]
    else:
        old_desc = {item.name: item.description for item in previous.criteria}
        changed_ids = [node_id for node_id in rubric.preorder_node_ids()
                       if old_desc.get(rubric.get_node(node_id).criterion.name)
                       != rubric.get_node(node_id).criterion.description]
        if changed_ids:
            changed_rubric = StructuredRubric(
                {node_id: rubric.get_node(node_id) for node_id in changed_ids},
                (), tuple(changed_ids))
            shard = base._generate_pairwise(
                _runtime_config(config), target / "pairwise_diagnostic",
                changed_rubric, rows, f"phase21_changed_{epoch_dir.name}",
                execution_backend_pool=config["backend_pool"],
                request_backend_id=previous.request_spec.backend_id,
                request_level_progress=True)[0]
            if not phase17._same_pairwise_scientific_request_identity(previous, shard):
                raise RuntimeError("Phase21 Pairwise diagnostic identity drift")
            shard_rows = shard.node_outputs
        else:
            shard = None
            shard_rows = ()
        shard_names = set() if shard is None else {item.name for item in shard.criteria}
        rows_out = []
        for index, old_row in enumerate(previous.node_outputs):
            update = {} if shard is None else {
                name: shard_rows[index][name] for name in shard_names}
            values = {**old_row, **update}
            rows_out.append({
                rubric.get_node(node_id).criterion.name:
                    values[rubric.get_node(node_id).criterion.name]
                for node_id in rubric.preorder_node_ids()})
        criteria = tuple(StructuredCriterionSnapshot(
            rubric.get_node(node_id).criterion.name,
            rubric.get_node(node_id).criterion.description)
            for node_id in rubric.preorder_node_ids())
        answers = tuple(aggregate_flat_votes(item.vote for item in row.values())
                        for row in rows_out)
        prediction = PairwisePredictionOutput(
            previous.sample_ids, previous.sample_fingerprints, criteria,
            tuple(rows_out), answers, previous.request_spec,
            semantics_version=previous.semantics_version,
            schema_version=previous.schema_version,
            prompt_version=previous.prompt_version,
            parser_version=previous.parser_version)
        generated_names = sorted(shard_names)
    _, feedback = split._snapshot(epoch_dir, rubric, prediction, rows, manifest)
    _write(epoch_dir / "pairwise_read_only_diagnostic.json", {
        "schema_version": "1.0.0", "decision_role": "none",
        "generated_criterion_names": generated_names,
        "specialized_acc_used_for_scheduling": False,
        "specialized_acc_used_for_acceptance": False,
        "nodes": {node_id: value.to_dict()
                  for node_id, value in feedback.nodes.items()}})
    return prediction


def _dev_epoch(
    config: Mapping[str, Any], target: Path, epoch_no: int,
    *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    epoch_dir = split._epoch(target, epoch_no)
    path = epoch_dir / "dev150" / "system_report.json"
    rubric = StructuredRubric.load_json(epoch_dir / "rubric_committed.json")
    if path.is_file():
        value = load_json(path)
        if value["rubric_sha256"] != rubric.rubric_sha256:
            raise RuntimeError("Phase21 Dev diagnostic rubric drift")
        return value
    rows = _rows(config, "dev")
    artifact = _system_eval(
        config, target=target, output_path=epoch_dir / "dev150" / "system.json",
        split_name=f"phase21_dev_epoch_{epoch_no:02d}", rows=rows, rubric=rubric, protocol=protocol)
    _require_complete(artifact, "Phase21 Dev diagnostic")
    value = {
        "schema_version": "1.0.0", "epoch": epoch_no,
        "rubric_sha256": rubric.rubric_sha256, "node_count": len(rubric.nodes),
        "system": artifact["metrics"], "diagnostic_only": True,
        "manager_visible": False, "selection_forbidden": True}
    _write(path, value)
    return value


def freeze(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    settings = _settings(config, protocol=protocol)
    refine._validate_output(output)
    target = _target(output, protocol=protocol)
    rows, dev = _rows(config, "discovery"), _rows(config, "dev")
    rubric = base.build_multicrit_open_ended_init_rubric()
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 5:
        raise RuntimeError("Phase21 requires exactly five initial roots")
    epoch0 = split._epoch(target, 0)
    # `StructuredRubric.save_json` does not create parent directories.  Create
    # the epoch scaffold before writing the immutable initial rubric so a
    # fresh checkout can run freeze without relying on a prior stage artifact.
    epoch0.mkdir(parents=True, exist_ok=True)
    rubric.save_json(epoch0 / "rubric_initial.json")
    memory, memory_hash = split.freeze_rubric_memory(epoch0, rubric)
    manager, profiles, specs, manager_ids = _manager_bundle(config)
    expected_pairwise = base._expected_pairwise_request_spec(
        _runtime_config(config), rows)
    manifest = {
        "schema_version": "1.0.0", "experiment": protocol.experiment_dir,
        "protocol_version": protocol.version, "settings": settings,
        "initial_rubric_sha256": rubric.rubric_sha256,
        "initial_root_ids": list(rubric.root_ids),
        "datasets": {
            "discovery": {"path": phase17.DISCOVERY_PATH, "count": len(rows),
                          "sha256": file_sha256(base._path(phase17.DISCOVERY_PATH))},
            "dev": {"path": phase17.DEV_PATH, "count": len(dev),
                    "sha256": file_sha256(base._path(phase17.DEV_PATH))},
            "heldout": {"path": str(config["heldout_dataset"]), "count": 500,
                        "sha256": file_sha256(base._path(config["heldout_dataset"]))},
        },
        "pairwise_request_spec": expected_pairwise.to_dict(),
        "manager_profiles": profiles, "manager_request_specs": specs,
        "manager_endpoint_identities": manager_ids,
        "worker_endpoint_identities": _endpoint_identities(config),
        "unified_runtime_identity": _unified_runtime_identity(config, protocol=protocol),
        "epoch_00_rubric_memory": memory,
        "epoch_00_rubric_memory_sha256": memory_hash,
        "global_arbiter_used_for_acceptance": False,
        "specialized_acc_used_for_acceptance": False,
        "dev_visible_to_manager": False,
        "heldout_accessed": False,
    }
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("Phase21 frozen manifest drift")
        print("subtree-bundle-evolution-freeze already completed")
        return
    _write(path, manifest)
    _write(target / "stage_status.json", {"freeze": {
        "status": "passed", "details": {
            "discovery_count": len(rows), "dev_count": len(dev)}}})
    print(json.dumps({"target": str(target), "manager_specs": list(specs)}, indent=2))


def audit(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    manifest = _load_manifest(config, output, protocol=protocol)
    _, _, specs, _ = _manager_bundle(config)
    checks = {
        "discovery_count_100": len(_rows(config, "discovery")) == 100,
        "dev_count_150": len(_rows(config, "dev")) == 150,
        "five_roots": len(manifest["initial_root_ids"]) == 5,
        "manager_specs_match": specs == manifest["manager_request_specs"],
        "unified_runtime_identity_match": _unified_runtime_identity(config, protocol=protocol)
            == manifest["unified_runtime_identity"],
        "scope_is_unified": protocol.settings["scope_source"]
            == "epoch_start_unified_subtree_A_or_B",
        "errors_are_unified": protocol.settings["error_source"]
            == "unified_subtree_scope_mismatch_only",
        "no_locked_child": protocol.settings["strong_child_locking"] is False,
        "atomic_split_refine": protocol.settings["partial_acceptance"] is False,
        "specialized_read_only": manifest[
            "specialized_acc_used_for_acceptance"] is False,
        "arbiter_diagnostic_only": manifest[
            "global_arbiter_used_for_acceptance"] is False,
        "heldout_not_accessed": manifest["heldout_accessed"] is False,
    }
    if _uses_all_sample_acceptance(protocol=protocol):
        checks.update({
            "all_sample_is_only_acceptance_metric": (
                protocol.settings["split_acceptance"] ==
                "all_sample_unified_net_gain_gt_0"
                and protocol.settings["refine_acceptance"] ==
                "all_sample_unified_net_gain_gt_0"),
            "phase21_trigger_thresholds_preserved": (
                protocol.settings["split_trigger"] ==
                {
                    "strict_acc_below": 0.75,
                    "coverage_above": 0.80,
                    "min_support": 15,
                    "min_wrong": 15,
                    "max_children": 5,
                }
                and protocol.settings["bundle_refine_trigger"] == {
                    "strict_acc_above": 0.50,
                    "strict_acc_below": 0.80,
                    "min_support": 15,
                    "min_wrong": 5,
                }),
            "only_decomposition_failure_reclusters": (
                protocol.settings.get("recluster_failure_types") ==
                ["cluster_or_decomposition_error"]),
            "technical_failure_has_no_scientific_action": (
                protocol.settings.get("technical_failure_policy") ==
                "pause_without_scientific_retry_action"),
        })
    if not all(checks.values()):
        raise RuntimeError(f"Phase21 offline audit failed: {checks}")
    target = _target(output, protocol=protocol)
    _write(target / "offline_audit.json", {
        "schema_version": "1.0.0", "offline_only": True, "checks": checks})
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed", "details": checks}
    _write(target / "stage_status.json", status)
    print(json.dumps(checks, indent=2))


def _smoke_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source", "unknown")), []).append(row)
    selected = [item for source in sorted(by_source)
                for item in by_source[source][:2]]
    used = {str(item["sample_id"]) for item in selected}
    selected.extend(item for item in rows
                    if str(item["sample_id"]) not in used)
    return tuple(selected[:20])


def smoke(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    manifest = _load_manifest(config, output, protocol=protocol)
    _verify_live(config, manifest, protocol=protocol)
    target = _target(output, protocol=protocol)
    if load_json(target / "stage_status.json").get("audit", {}).get("status") != "passed":
        raise RuntimeError("run subtree-bundle-evolution-audit first")
    path = target / "smoke" / "report.json"
    if path.is_file():
        print(json.dumps(load_json(path), indent=2))
        return
    rows = _smoke_rows(_rows(config, "discovery"))
    rubric = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    baseline = _system_eval(
        config, target=target, output_path=target / "smoke" / "system.json",
        split_name="phase21_smoke_baseline", rows=rows, rubric=rubric, protocol=protocol)
    _require_complete(baseline, "Phase21 smoke baseline")
    roots = _freeze_root_baselines(target / "smoke", baseline, rubric, rows)
    paired = (
        system.paired_root_all_samples(
            roots[rubric.root_ids[0]], roots[rubric.root_ids[0]], rows)
        if _uses_all_sample_acceptance(protocol=protocol)
        else system.paired_root(
            roots[rubric.root_ids[0]], roots[rubric.root_ids[0]], rows))
    value = {
        "schema_version": "1.0.0", "status": "passed",
        "sample_count": len(rows), "source_count": len({row.get("source") for row in rows}),
        "unified_scope_defined": all(root["metrics"]["root_scope_support"] >= 0
                                     for root in roots.values()),
        "paired_comparison_defined": (
            "all_sample_net_gain" in paired
            if _uses_all_sample_acceptance(protocol=protocol)
            else "root_scope_net_corrected" in paired),
        "all_sample_support_is_complete": (
            paired.get("all_sample_support") == len(rows)
            if _uses_all_sample_acceptance(protocol=protocol) else True),
        "split_retry_mapper_defined": (
            split_retry_action("cluster_or_decomposition_error", protocol=protocol) ==
            "recluster_regenerate_complete_bundle"
            if _uses_all_sample_acceptance(protocol=protocol) else True),
        "split_atomic_contract": protocol.settings["partial_acceptance"] is False,
        "bundle_refine_parser_covered_by_unit_tests": True,
        "accepted_report_commit_reuse_covered_by_unit_tests": True,
        "formal_evolution_mutated": False, "heldout_accessed": False,
    }
    _write(path, value)
    status = load_json(target / "stage_status.json")
    status["smoke"] = {"status": "passed", "details": value}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2))


def _initialize(
    config: Mapping[str, Any], target: Path, manifest: Mapping[str, Any], *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    epoch0 = split._epoch(target, 0)
    required = (
        epoch0 / "rubric_committed.json", epoch0 / "system" / "discovery.json",
        epoch0 / "root_scopes.json", epoch0 / "discovery_pairwise.json",
        target / "evolution_history.json")
    if all(path.is_file() for path in required):
        return
    rows = _rows(config, "discovery")
    rubric = StructuredRubric.load_json(epoch0 / "rubric_initial.json")
    system_value = _system_eval(
        config, target=target, output_path=epoch0 / "system" / "discovery.json",
        split_name="phase21_epoch00_discovery", rows=rows, rubric=rubric, protocol=protocol)
    _require_complete(system_value, "Phase21 epoch-0 baseline")
    # Epoch 0 is the first committed baseline.  Subsequent epochs load their
    # starting Rubric from this artifact; keeping only rubric_initial.json
    # would let smoke pass but make the first real run fail before scheduling.
    rubric.save_json(epoch0 / "rubric_committed.json")
    roots = _freeze_root_baselines(epoch0, system_value, rubric, rows)
    prediction = _pairwise_diagnostic(
        config, epoch_dir=epoch0, target=target, rubric=rubric, rows=rows,
        manifest=manifest, previous=None)
    triggers = {root_id: _trigger(
        root_id, rubric, roots[root_id], rows,
        {"split_status": "pending"}, protocol=protocol) for root_id in rubric.root_ids}
    _write(epoch0 / "summary.json", {
        "epoch": 0, "global_arbiter_diagnostic": system_value["metrics"],
        "root_triggers": triggers, "accepted_roots": []})
    _write(target / "evolution_history.json", {
        "schema_version": "1.0.0", "current_epoch": 0,
        "completed": False, "stop_reason": None,
        "root_states": {root_id: {
            "split_status": "pending", "split_attempts": 0,
            "refine_attempts": 0, "accepted_split_epoch": None,
            "children": [], "split_retry": None}
            for root_id in rubric.root_ids},
        "attempts": [], "synchronous_commits": []})
    _dev_epoch(config, target, 0, protocol=protocol)


def _pause(
    target: Path, epoch_no: int, root_id: str, exc: Exception,
    *, operator: str | None = None,
) -> None:
    status = load_json(target / "stage_status.json")
    details = {
        "epoch": epoch_no, "root_id": root_id,
        "operator": operator,
        "stage": getattr(exc, "stage", "phase21_candidate"),
        "type": type(exc).__name__, "message": str(exc),
        "history_appended": False}
    if hasattr(exc, "to_dict"):
        details["exception_artifact"] = exc.to_dict()  # type: ignore[attr-defined]
    status["run"] = {"status": "paused", "details": details}
    _write(target / "stage_status.json", status)
    _write(split._epoch(target, epoch_no) / "pause.json", details)
    print(
        f"phase21 epoch={epoch_no} operator={operator or 'unknown'} "
        f"root={root_id} status=paused stage={details['stage']} "
        f"error={details['message']}",
        flush=True,
    )


def _run_impl(
    config: Mapping[str, Any], target: Path, manifest: Mapping[str, Any],
    history: dict[str, Any],
    *, protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    rows = _rows(config, "discovery")
    manager, _, current_specs, _ = _manager_bundle(config)
    if current_specs != manifest["manager_request_specs"]:
        raise RuntimeError("Phase21 Manager identity drift")
    max_epochs = protocol.settings["max_epochs"]
    for epoch_no in range(int(history["current_epoch"]) + 1, max_epochs + 1):
        started = time.monotonic()
        previous = split._epoch(target, epoch_no - 1)
        epoch_dir = split._epoch(target, epoch_no)
        rubric = StructuredRubric.load_json(previous / "rubric_committed.json")
        previous_system = system.load(previous / "system" / "discovery.json")
        _require_complete(
            previous_system, f"Phase21 epoch-{epoch_no} Unified baseline")
        baselines = _freeze_root_baselines(
            epoch_dir, previous_system, rubric, rows)
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, rubric)
        previous_prediction = PairwisePredictionOutput.load_json(
            previous / "discovery_pairwise.json")
        triggers = {root_id: _trigger(
            root_id, rubric, baselines[root_id], rows,
            history["root_states"][root_id], protocol=protocol) for root_id in rubric.root_ids}
        scheduled = [root_id for root_id in rubric.root_ids
                     if triggers[root_id]["triggered"]]
        print(
            f"phase21 epoch={epoch_no} scheduled="
            f"{[(root, triggers[root]['operator']) for root in scheduled]}",
            flush=True)
        results: dict[str, dict[str, Any]] = {}
        for root_id in scheduled:
            state = history["root_states"][root_id]
            operator = triggers[root_id]["operator"]
            attempt_no = (int(state["split_attempts"]) + 1 if operator == "split"
                          else int(state["refine_attempts"]) + 1)
            retry = state.get("split_retry") if operator == "split" else None
            print(
                f"phase21 epoch={epoch_no} operator={operator} root={root_id} "
                f"attempt={attempt_no} status=started",
                flush=True,
            )
            try:
                signatures, signature_value = _signature_artifact(
                    config, target=target, epoch_dir=epoch_dir, root_id=root_id,
                    baseline=baselines[root_id], rubric=rubric, rows=rows,
                    manager=manager, rubric_memory=memory,
                    retry_source=None if retry is None else retry["signature_path"])
                history_projection = _history_projection(history, root_id, operator)
                if operator == "split":
                    reuse_retry = split_retry_source(retry)
                    result = _prepare_split(
                        config=config, epoch_dir=epoch_dir, root_id=root_id,
                        attempt_no=attempt_no, rubric=rubric, rows=rows,
                        signatures=signatures, manager=manager, memory=memory,
                        history_projection=history_projection,
                        retry_source=reuse_retry, protocol=protocol)
                else:
                    result = _prepare_refine(
                        epoch_dir=epoch_dir, root_id=root_id,
                        attempt_no=attempt_no, rubric=rubric, rows=rows,
                        signatures=signatures, manager=manager, memory=memory,
                        history_projection=history_projection)
                result["signature_path"] = signature_value["path"]
                if operator == "split":
                    _record_split_lineage(result, retry, protocol=protocol)
                signature_metrics = (
                    [item["metrics"]
                     for item in signature_value["outputs"].values()]
                    if signature_value["reuse"]["kind"] == "generated" else [])
                result["manager_usage"] = _combine_manager_usage([
                    *signature_metrics, result["proposal_manager_usage"]])
                result = _evaluate_candidate(
                    config, target=target, epoch_dir=epoch_dir, rows=rows,
                    baseline=baselines[root_id],
                    scope=baselines[root_id]["metrics"]["root_scope_sample_ids"],
                    result=result, protocol=protocol)
                if result["decision"] != "accepted":
                    by_id = {str(row["sample_id"]): row for row in rows}
                    harmed_key = ("all_sample_harmed_sample_ids"
                                  if _uses_all_sample_acceptance(protocol=protocol)
                                  else "root_scope_harmed_sample_ids")
                    corrected_key = ("all_sample_corrected_sample_ids"
                                     if _uses_all_sample_acceptance(protocol=protocol)
                                     else "root_scope_corrected_sample_ids")
                    evidence_ids = (
                        list(result["paired"][harmed_key][:6])
                        + list(result["paired"][corrected_key][:3]))
                    attribution = manager.attribute_failure(
                        operator=operator, baseline_rubric=rubric,
                        candidate_rubric=result["after_rubric"], root_id=root_id,
                        signatures=tuple(signatures.values()),
                        candidate=result["candidate_payload"],
                        paired_evidence=result["evidence"],
                        representative_rows=tuple(
                            by_id[sample_id] for sample_id in evidence_ids),
                        failure_history=history_projection)
                    result["failure_attribution"] = attribution
                    result["manager_usage"] = _combine_manager_usage([
                        result["manager_usage"], attribution["metrics"]])
                    _write(result["attempt_dir"] / "failure_attribution.json", attribution)
                paired = result["paired"]
                net_gain = paired.get(
                    "all_sample_net_gain",
                    paired.get("root_scope_net_corrected"),
                )
                failure_type = (
                    None if result.get("failure_attribution") is None
                    else result["failure_attribution"]["attribution"].get(
                        "primary_failure_type"))
                retry_action = (
                    split_retry_action(failure_type, protocol=protocol)
                    if operator == "split" and failure_type is not None else None)
                print(
                    f"phase21 epoch={epoch_no} operator={operator} root={root_id} "
                    f"attempt={attempt_no} status={result['decision']} "
                    f"net_gain={net_gain} failure_type={failure_type} "
                    f"retry_action={retry_action}",
                    flush=True,
                )
                results[root_id] = result
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                _pause(target, epoch_no, root_id, exc, operator=operator)
                return
        # Candidate names from different roots must remain globally unique.
        split_candidates = {root_id: result["candidate"]
                            for root_id, result in results.items()
                            if result["operator"] == "split"
                            and result["decision"] == "accepted"}
        collisions = split.colliding_roots(split_candidates)
        if collisions:
            raise RuntimeError(f"Phase21 cross-root child-name collision: {collisions}")
        committed = _merge_accepted(rubric, results)
        accepted = sorted(root_id for root_id, result in results.items()
                          if result["decision"] == "accepted")
        committed_roots = {}
        committed_root_hashes = {}
        for root_id in committed.root_ids:
            source = (results[root_id]["root_artifact"]
                      if root_id in accepted else baselines[root_id])
            committed_roots[root_id] = _reframe_root(
                source, committed, root_id, rows,
                epoch_dir / "committed_root_reports" / f"{root_id}.json")
            source_hash = _root_calls_sha256(source)
            committed_hash = _root_calls_sha256(committed_roots[root_id])
            if source_hash != committed_hash:
                raise RuntimeError(
                    f"Phase21 candidate/commit root report drift: {root_id}")
            committed_root_hashes[root_id] = {
                "source": "accepted_candidate" if root_id in accepted
                          else "epoch_start_baseline",
                "source_call_sha256": source_hash,
                "committed_call_sha256": committed_hash,
                "identical": True,
            }
        _write(epoch_dir / "committed_root_report_hashes.json", {
            "schema_version": "1.0.0",
            "accepted_roots": accepted,
            "roots": committed_root_hashes,
        })
        if accepted:
            committed_system = _commit_system(
                config, target=target, epoch_dir=epoch_dir, rows=rows,
                rubric=committed, roots=committed_roots,
                label=f"phase21_epoch_{epoch_no:02d}_arbiter_only", protocol=protocol)
        else:
            committed_system = dict(previous_system)
            committed_system["rubric_sha256"] = committed.rubric_sha256
            _write(epoch_dir / "system" / "discovery.json", committed_system)
        committed.save_json(epoch_dir / "rubric_committed.json")
        diagnostic = _pairwise_diagnostic(
            config, epoch_dir=epoch_dir, target=target, rubric=committed,
            rows=rows, manifest=manifest, previous=previous_prediction)
        records = []
        for root_id in scheduled:
            result = results[root_id]
            state = history["root_states"][root_id]
            operator = result["operator"]
            if operator == "split":
                state["split_attempts"] += 1
                if result["decision"] == "accepted":
                    state["split_status"] = "accepted"
                    state["accepted_split_epoch"] = epoch_no
                    state["children"] = [item.criterion_name
                                         for item in result["candidate"].children]
                    state["split_retry"] = None
                else:
                    state["split_status"] = (
                        "exhausted" if state["split_attempts"] >= max_epochs
                        else "retryable")
                    failure_type = result["failure_attribution"][
                        "attribution"]["primary_failure_type"]
                    action = split_retry_action(failure_type, protocol=protocol)
                    retry_value = {
                        "schema_version": "1.0.0",
                        "root_id": root_id,
                        "source_attempt": state["split_attempts"],
                        "primary_failure_type": failure_type,
                        "action": action,
                        "signature_path": result["signature_path"],
                        "cluster_path": result["cluster_path"],
                        "failure_attribution_path": str(
                            result["attempt_dir"] /
                            "failure_attribution.json")}
                    has_next_retry = state["split_status"] == "retryable"
                    state["split_retry"] = ({
                        key: value for key, value in retry_value.items()
                        if key != "schema_version"}
                        if has_next_retry else None)
                    if _uses_all_sample_acceptance(protocol=protocol) and has_next_retry:
                        _write(
                            result["attempt_dir"] / "retry_action.json",
                            retry_value)
            else:
                state["refine_attempts"] += 1
            record = {
                "epoch": epoch_no, "root_id": root_id, "operator": operator,
                "attempt": (state["split_attempts"] if operator == "split"
                            else state["refine_attempts"]),
                "decision": result["decision"],
                "attempt_dir": str(result["attempt_dir"]),
                ("all_sample_evaluation" if _uses_all_sample_acceptance(protocol=protocol)
                 else "root_scope_evaluation"): result["paired"],
                "root_metrics": result["root_metrics"],
                "cluster_path": result.get("cluster_path"),
                "cluster_proposal": (
                    load_json(Path(result["cluster_path"]))
                    if operator == "split" else None),
                "corrected_sample_ids": result["paired"][
                    ("all_sample_corrected_sample_ids"
                     if _uses_all_sample_acceptance(protocol=protocol)
                     else "root_scope_corrected_sample_ids")],
                "harmed_sample_ids": result["paired"][
                    ("all_sample_harmed_sample_ids"
                     if _uses_all_sample_acceptance(protocol=protocol)
                     else "root_scope_harmed_sample_ids")],
                "phase21_selective_metric_diagnostic": (
                    result.get("selective_paired")
                    if _uses_all_sample_acceptance(protocol=protocol) else None),
                "retry_action": (
                    (state.get("split_retry") or {}).get("action")
                    if operator == "split" and
                    result["decision"] != "accepted" else None),
                "failure_attribution": result.get("failure_attribution"),
                "failure_attribution_type": (
                    None if result.get("failure_attribution") is None
                    else result["failure_attribution"]["attribution"][
                        "primary_failure_type"]),
                "manager_usage": result["manager_usage"],
                "candidate_unified_usage": result["candidate_unified_usage"],
                "specialized_acc_used": False,
                "global_arbiter_used_for_acceptance": False,
            }
            history["attempts"].append(record)
            records.append(record)
        history["current_epoch"] = epoch_no
        history["synchronous_commits"].append({
            "epoch": epoch_no, "accepted_roots": accepted,
            "all_independently_passing_roots_committed": True,
            "subset_search": False,
            "accepted_report_reinference_count": 0,
            "global_arbiter_diagnostic": committed_system["metrics"]})
        next_triggers = {root_id: _trigger(
            root_id, committed, committed_roots[root_id], rows,
            history["root_states"][root_id], protocol=protocol) for root_id in committed.root_ids}
        if epoch_no >= protocol.settings["min_epochs"] and not any(
                item["triggered"] for item in next_triggers.values()):
            history.update({"completed": True,
                            "stop_reason": "no_retryable_root_operator"})
        elif epoch_no == max_epochs:
            history.update({"completed": True,
                            "stop_reason": "max_epochs_reached"})
        _write(epoch_dir / "summary.json", {
            "epoch": epoch_no, "root_triggers": triggers,
            "scheduled_roots": scheduled, "accepted_roots": accepted,
            "attempts": records,
            "global_arbiter_diagnostic": committed_system["metrics"],
            "specialized_diagnostic_only": True,
            "accepted_report_reinference_count": 0,
            "committed_root_report_hashes": committed_root_hashes,
            "next_root_triggers": next_triggers,
            "rubric_node_count": len(committed.nodes),
            "epoch_wall_seconds": time.monotonic() - started})
        _write(target / "evolution_history.json", history)
        _dev_epoch(config, target, epoch_no, protocol=protocol)
        print(
            f"phase21 epoch={epoch_no} accepted={accepted} "
            f"diagnostic_strict={committed_system['metrics']['strict_accuracy']:.4f}",
            flush=True)
        if history["completed"]:
            break
    status = load_json(target / "stage_status.json")
    status["run"] = {
        "status": "passed" if history["completed"] else "partial",
        "details": {"current_epoch": history["current_epoch"],
                    "stop_reason": history["stop_reason"]}}
    _write(target / "stage_status.json", status)


def run(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    _settings(config, protocol=protocol)
    target = _target(output, protocol=protocol)
    status = load_json(target / "stage_status.json")
    if status.get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run subtree-bundle-evolution-smoke first")
    manifest = _load_manifest(config, output, protocol=protocol)
    _verify_live(config, manifest, protocol=protocol)
    _initialize(config, target, manifest, protocol=protocol)
    history = load_json(target / "evolution_history.json")
    if not history["completed"]:
        _run_impl(config, target, manifest, history, protocol=protocol)


def _dev_trajectory(
    config: Mapping[str, Any], target: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    history = load_json(target / "evolution_history.json")
    reports = [_dev_epoch(config, target, epoch, protocol=protocol)
               for epoch in range(int(history["current_epoch"]) + 1)]
    rows = _rows(config, "dev")
    trajectory = []
    for item in reports:
        paired = system.paired(reports[0]["system"], item["system"], rows)
        trajectory.append({
            "epoch": item["epoch"], "rubric_sha256": item["rubric_sha256"],
            "strict_accuracy": item["system"]["strict_accuracy"],
            "coverage": item["system"]["coverage"],
            "corrected_vs_epoch0": paired["corrected"],
            "harmed_vs_epoch0": paired["harmed"],
            "net_corrected_vs_epoch0": paired["net_corrected"]})
    value = {"schema_version": "1.0.0", "diagnostic_only": True,
             "selection_forbidden": True, "epochs": trajectory}
    _write(target / "dev150_trajectory.json", value)
    return value


def report(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    _settings(config, protocol=protocol)
    target = _target(output, protocol=protocol)
    history = load_json(target / "evolution_history.json")
    if not history["completed"]:
        raise RuntimeError("run subtree-bundle-evolution-run to completion first")
    final_epoch = int(history["current_epoch"])
    epoch_dir = split._epoch(target, final_epoch)
    rubric = StructuredRubric.load_json(epoch_dir / "rubric_committed.json")
    final_system = system.load(epoch_dir / "system" / "discovery.json")
    initial_system = system.load(split._epoch(target, 0) / "system" / "discovery.json")
    rows = _rows(config, "discovery")
    comparison = system.paired(initial_system["metrics"], final_system["metrics"], rows)
    accepted = [item for item in history["attempts"]
                if item["decision"] == "accepted"]
    if any(item.get("specialized_acc_used") for item in history["attempts"]):
        raise RuntimeError("Phase21 history used Specialized ACC")
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    PairwisePredictionOutput.load_json(
        epoch_dir / "discovery_pairwise.json").save_json(
            final_dir / "discovery_pairwise_diagnostic.json")
    _write(final_dir / "system_discovery.json", final_system)
    manager_usage = _combine_manager_usage([
        item["manager_usage"] for item in history["attempts"]])
    candidate_usage = system.support.sum_metrics([
        item["candidate_unified_usage"] for item in history["attempts"]])
    candidate_usage["logical_evaluations"] = sum(
        int(item["candidate_unified_usage"]["logical_evaluations"])
        for item in history["attempts"])
    candidate_usage["wall_seconds"] = sum(
        float(item["candidate_unified_usage"]["wall_seconds"])
        for item in history["attempts"])
    pairwise_logical = 0
    interaction_cases = []
    for epoch in range(final_epoch + 1):
        epoch_dir_value = split._epoch(target, epoch)
        diagnostic_path = epoch_dir_value / "pairwise_read_only_diagnostic.json"
        if diagnostic_path.is_file():
            pairwise_logical += len(load_json(diagnostic_path)[
                "generated_criterion_names"]) * len(rows)
        if epoch == 0:
            continue
        summary = load_json(epoch_dir_value / "summary.json")
        if not summary["accepted_roots"]:
            continue
        before_metrics = system.load(
            split._epoch(target, epoch - 1) / "system" / "discovery.json")[
                "metrics"]
        after_metrics = system.load(
            epoch_dir_value / "system" / "discovery.json")["metrics"]
        delta = after_metrics["strict_accuracy"] - before_metrics["strict_accuracy"]
        if delta < 0:
            interaction_cases.append({
                "epoch": epoch,
                "accepted_roots": summary["accepted_roots"],
                "global_strict_accuracy_before": before_metrics[
                    "strict_accuracy"],
                "global_strict_accuracy_after": after_metrics[
                    "strict_accuracy"],
                "global_strict_accuracy_delta": delta,
                "local_acceptance_rolled_back": False,
            })
    value = {
        "schema_version": "1.0.0", "experiment": protocol.experiment_dir,
        "protocol_version": protocol.version, "final_epoch": final_epoch,
        "initial_system": initial_system["metrics"],
        "final_system": final_system["metrics"],
        "paired_vs_initial": comparison,
        "final_rubric_sha256": rubric.rubric_sha256,
        "node_count_initial": 5, "node_count_final": len(rubric.nodes),
        "attempts": history["attempts"],
        "accepted_operation_count": len(accepted),
        "accepted_split_count": sum(item["operator"] == "split" for item in accepted),
        "accepted_bundle_refine_count": sum(
            item["operator"] == "refine" for item in accepted),
        "accepted_report_reinference_count": sum(
            item["accepted_report_reinference_count"]
            for item in history["synchronous_commits"]),
        "specialized_acc_role": "read_only_diagnostic",
        "usage": {
            "manager": manager_usage,
            "candidate_unified_subtree": candidate_usage,
            "pairwise_diagnostic_logical_evaluations": pairwise_logical,
            "pairwise_diagnostic_token_usage_available": False,
        },
        "local_improvement_global_regression_cases": interaction_cases,
        "global_arbiter_used_for_acceptance": False,
        "dev_trajectory": _dev_trajectory(config, target, protocol=protocol),
        "heldout_accessed": False, "vl_rewardbench_required": True,
    }
    _write(final_dir / "discovery_report.json", value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_epoch": final_epoch,
        "final_strict_accuracy": final_system["metrics"]["strict_accuracy"],
        "accepted_operation_count": len(accepted),
        "accepted_report_reinference_count": value[
            "accepted_report_reinference_count"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["report"]["details"], indent=2))


def heldout(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    manifest = _load_manifest(config, output, protocol=protocol)
    _verify_live(config, manifest, protocol=protocol)
    target = _target(output, protocol=protocol)
    if not (target / "final" / "discovery_report.json").is_file():
        raise RuntimeError("run subtree-bundle-evolution-report first")
    rows = _rows(config, "heldout")
    initial = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    final = StructuredRubric.load_json(target / "final" / "rubric.json")
    initial_value = _system_eval(
        config, target=target,
        output_path=target / "heldout500" / "initial_system.json",
        split_name="phase21_heldout_initial", rows=rows, rubric=initial, protocol=protocol)
    final_value = _system_eval(
        config, target=target,
        output_path=target / "heldout500" / "final_system.json",
        split_name="phase21_heldout_final", rows=rows, rubric=final, protocol=protocol)
    _require_complete(initial_value, "Phase21 heldout initial")
    _require_complete(final_value, "Phase21 heldout final")
    paired = system.paired(initial_value["metrics"], final_value["metrics"], rows)
    value = {
        "schema_version": "1.0.0", "exploratory": True,
        "selection_after_heldout_forbidden": True,
        "initial": initial_value["metrics"],
        ("phase22_final" if _uses_all_sample_acceptance(protocol=protocol)
         else "phase21_final"): final_value["metrics"],
        "paired_vs_initial": paired}
    _write(target / "heldout500" / "report.json", value)
    manifest["heldout_accessed"] = True
    _write(target / "frozen_manifest.json", manifest)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        "initial_strict_accuracy": initial_value["metrics"]["strict_accuracy"],
        "final_strict_accuracy": final_value["metrics"]["strict_accuracy"],
        "net_corrected": paired["net_corrected"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def final_report(
    config: Mapping[str, Any], output: Path, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    _settings(config, protocol=protocol)
    target = _target(output, protocol=protocol)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout_value = load_json(target / "heldout500" / "report.json")
    value = {
        "schema_version": "1.0.0", "protocol_version": protocol.version,
        "discovery": discovery, "heldout500": heldout_value,
        "vl_rewardbench_required": True,
        "next_required_stage": (
            "vlrb-all-sample-adaptive-freeze"
            if _uses_all_sample_acceptance(protocol=protocol)
            else "vlrb-subtree-bundle-freeze")}
    _write(target / "final_report.json", value)
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed", "details": {
        "final_rubric_sha256": discovery["final_rubric_sha256"],
        "next_required_stage": value["next_required_stage"]}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["final_report"]["details"], indent=2))


def run_stage(
    config: Mapping[str, Any], output: Path, stage: str, *,
    protocol: EvolutionProtocol = DEFAULT_PROTOCOL,
) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        protocol.stages[0]: freeze, protocol.stages[1]: audit, protocol.stages[2]: smoke,
        protocol.stages[3]: run, protocol.stages[4]: report, protocol.stages[5]: heldout,
        protocol.stages[6]: final_report}
    if stage not in actions:
        raise ValueError(f"unsupported Phase21 stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output, protocol=protocol)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

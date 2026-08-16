"""Full-rubric five-root Child-Gate experiment.

All five Phase10 roots remain active.  One sibling Gate independently selects
the direct children of each root; the frozen Pairwise votes and M1 aggregation
semantics are never changed.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    ModelCallMetrics,
    PairwisePredictionOutput,
    StructuredRubric,
    aggregate_child_subtrees,
    aggregate_selected_roots,
)
from critiq.structured.telemetry import combine_model_call_metrics

from . import run_rubric_evolution as base
from . import visual_gate as vg
from .experiment_utils import atomic_write_json, load_json, make_progress_callback
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase14_full_rubric_child_gate_v1"
PROTOCOL_VERSION = "full-rubric-child-gate-v1"
PROMPT_VERSION = "sibling-gate-prompt-json-v4"
SCHEMA_VERSION = "1.0.0"
VISUAL_GATE_EXPERIMENT = "phase13_visual_grounding_gate_only_v3"


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "source_rubric_experiment": vg.SOURCE_RUBRIC_EXPERIMENT,
        "source_pairwise_experiment": vg.SOURCE_PAIRWISE_EXPERIMENT,
        "source_pairwise_variant": vg.SOURCE_PAIRWISE_VARIANT,
        "root_policy": "all_roots_always_active",
        "routing_scope": "per_root_direct_children_only",
        "gate_endpoints": ["vllm-8000", "vllm-8001"],
        "gate_temperature": 0.2,
        "gate_max_tokens": 2048,
        "gate_seed": 42,
        "gate_max_retries": 5,
        "always_format_reminder": True,
        "smoke_sample_count": 20,
        "position_audit_count": 20,
        "uncertain_policy": "activate",
        "invalid_policy": "all_children_for_failed_root",
        "heldout_access": "final_stage_only",
    }
    value = config.get("full_child_gate_experiment")
    if value != expected:
        raise RuntimeError("full_child_gate_experiment does not match the frozen v1 protocol")
    return dict(value)


def _rubric(output: Path) -> StructuredRubric:
    return vg._rubric(output)


def _rows(config: Mapping[str, Any], split: str):
    return vg._rows(config, split)


def _prediction(output: Path, rubric: StructuredRubric, rows,
                split: str) -> PairwisePredictionOutput:
    return vg._load_source_prediction(output, rubric, rows, split)


def _contracts(rubric: StructuredRubric) -> dict[str, dict[str, Any]]:
    contracts = {
        root_id: vg.build_routing_contract(
            rubric, parent_node_id=root_id, expected_child_names=None)
        for root_id in rubric.root_ids
    }
    if len(rubric.nodes) != 22 or len(rubric.edges) != 17:
        raise RuntimeError("Full Child-Gate requires the frozen 5-root/17-child Phase10 tree")
    if any(not contract["children"] for contract in contracts.values()):
        raise RuntimeError("Every Phase10 root must have direct children")
    child_ids = [
        child["node_id"] for contract in contracts.values()
        for child in contract["children"]
    ]
    if len(child_ids) != 17 or len(set(child_ids)) != 17:
        raise RuntimeError("Full Child-Gate direct-child identity drift")
    return contracts


def _request_specs(config: Mapping[str, Any], protocol: Mapping[str, Any],
                   contracts: Mapping[str, Mapping[str, Any]], spec) -> dict[str, Any]:
    request_protocol = dict(protocol)
    request_protocol["prompt_version"] = PROMPT_VERSION
    return {
        root_id: vg._request_spec(config, request_protocol, contract, spec)
        for root_id, contract in contracts.items()
    }


def _make_manifest(config: Mapping[str, Any], output: Path,
                   protocol: Mapping[str, Any], rubric: StructuredRubric,
                   rows, prediction: PairwisePredictionOutput,
                   contracts: Mapping[str, Mapping[str, Any]], spec, *,
                   inspect_live: bool) -> dict[str, Any]:
    prior = _target(output) / "frozen_manifest.json"
    endpoint_identities = vg._inspect_endpoints(config, spec) if inspect_live else None
    if endpoint_identities is None and prior.is_file():
        endpoint_identities = load_json(prior)["gate_endpoint_identities"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_DIR,
        "protocol": dict(protocol),
        "source": {
            "rubric_path": str(vg._rubric_path(output).resolve()),
            "rubric_file_sha256": file_sha256(vg._rubric_path(output)),
            "rubric_sha256": rubric.rubric_sha256,
            "node_count": len(rubric.nodes),
            "edge_count": len(rubric.edges),
            "root_ids": list(rubric.root_ids),
            "discovery_prediction_path": str(vg._prediction_path(output, "discovery90").resolve()),
            "discovery_prediction_sha256": file_sha256(vg._prediction_path(output, "discovery90")),
            "heldout_prediction_path": str(vg._prediction_path(output, "heldout500").resolve()),
        },
        "discovery": {
            "dataset_path": str(base._path(config["discovery_dataset"]).resolve()),
            "dataset_sha256": config["discovery_dataset_sha256"],
            "sample_count": len(rows),
            "sample_ids": [str(row["sample_id"]) for row in rows],
        },
        "heldout": {
            "dataset_path": str(base._path(config["heldout_dataset"]).resolve()),
            "dataset_sha256": config["heldout_dataset_sha256"],
            "sample_count": 500,
            "accessed_during_freeze": False,
        },
        "routing_contracts": {
            root_id: {
                "contract_sha256": contract["contract_sha256"],
                "child_ids": [item["node_id"] for item in contract["children"]],
            }
            for root_id, contract in contracts.items()
        },
        "gate_request_specs": _request_specs(config, protocol, contracts, spec),
        "gate_endpoint_identities": endpoint_identities,
        "source_pairwise_request_spec": prediction.request_spec.to_dict(),
        "heldout_access": "final_stage_only",
    }


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target(output)
    vg._require(target, "full-child-gate-freeze")
    manifest = load_json(target / "frozen_manifest.json")
    protocol = _protocol(config)
    rubric = _rubric(output)
    rows = _rows(config, "discovery90")
    prediction = _prediction(output, rubric, rows, "discovery90")
    contracts = _contracts(rubric)
    spec = vg._gate_pool_spec(config, protocol["gate_endpoints"])
    expected = _make_manifest(
        config, output, protocol, rubric, rows, prediction, contracts, spec,
        inspect_live=False)
    if manifest != expected:
        raise RuntimeError("Full Child-Gate frozen manifest drift")
    for root_id, contract in contracts.items():
        if load_json(target / "routing_contracts" / f"{root_id}.json") != contract:
            raise RuntimeError(f"Full Child-Gate contract drift: {root_id}")
    return target, manifest, protocol, rubric, rows, prediction, contracts, spec


def _summarize_gate(samples, metrics: Sequence[ModelCallMetrics]) -> dict[str, Any]:
    combined = combine_model_call_metrics(metrics)
    failures = [item for item in samples if not item["parse_ok"]]
    return {
        "sample_count": len(samples),
        "parse_valid_count": len(samples) - len(failures),
        "parse_valid_rate": (len(samples) - len(failures)) / len(samples),
        "parse_failure_count": len(failures),
        "parse_failure_sample_ids": [item["sample_id"] for item in failures],
        "fallback_all_children_count": sum(item["fallback_all_children"] for item in samples),
        "cache_hit_count": sum(item["cache_hit"] for item in samples),
        "current_run_metrics": combined.to_dict(),
    }


def _run_all_roots(config: Mapping[str, Any], target: Path,
                   manifest: Mapping[str, Any], protocol: Mapping[str, Any],
                   contracts: Mapping[str, Mapping[str, Any]], spec, rows,
                   work: Path, label: str, *, cache_root: Path | None = None,
                   experiment_name: str = EXPERIMENT_DIR) -> dict[str, Any]:
    if vg._inspect_endpoints(config, spec) != manifest["gate_endpoint_identities"]:
        raise RuntimeError("Full Child-Gate endpoint identity drift")
    pool = AvailableSlotBackendPool(spec)
    callback = make_progress_callback(work, label, len(rows) * len(contracts), pool)
    root_artifacts: dict[str, Any] = {}
    all_metrics: list[ModelCallMetrics] = []
    started = time.perf_counter()
    cache_root = cache_root or (target / "cache" / "gate")
    for root_id, contract in contracts.items():
        request_spec = manifest["gate_request_specs"][root_id]
        results: list[dict[str, Any] | None] = [None] * len(rows)
        metrics: list[ModelCallMetrics | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=spec.global_request_concurrency) as executor:
            futures = {
                executor.submit(
                    vg._route_one, config, protocol, contract, request_spec,
                    row, pool, cache_root / root_id,
                ): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                index = futures[future]
                result, call_metrics = future.result()
                results[index] = result
                metrics[index] = call_metrics
                callback(index, f"{rows[index]['sample_id']}::{root_id}", call_metrics)
        completed = [item for item in results if item is not None]
        completed_metrics = [item for item in metrics if item is not None]
        all_metrics.extend(completed_metrics)
        root_artifacts[root_id] = {
            "request_spec": request_spec,
            "samples": completed,
            "summary": _summarize_gate(completed, completed_metrics),
        }
    elapsed = time.perf_counter() - started
    all_samples = [
        item for root in root_artifacts.values() for item in root["samples"]]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment_name,
        "prompt_version": PROMPT_VERSION,
        "sample_count": len(rows),
        "root_ids": list(contracts),
        "roots": root_artifacts,
        "summary": {
            **_summarize_gate(all_samples, all_metrics),
            "logical_route_count": len(all_samples),
            "wall_seconds": elapsed,
            "requests_per_minute": (
                len(all_samples) / elapsed * 60 if elapsed else 0.0),
            "endpoint_call_counts": dict(pool.records_by_endpoint()),
        },
    }
    work.mkdir(parents=True, exist_ok=True)
    atomic_write_json(work / "gate_predictions.json", artifact)
    atomic_write_json(work / "provenance.json", pool.provenance_dict())
    return artifact


def _load_gate(path: Path, manifest: Mapping[str, Any], rows) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Full Child-Gate artifact missing: {path}")
    value = load_json(path)
    if (value.get("sample_count") != len(rows)
            or value.get("root_ids") != manifest["source"]["root_ids"]):
        raise RuntimeError("Full Child-Gate artifact identity drift")
    expected_ids = [str(row["sample_id"]) for row in rows]
    for root_id in manifest["source"]["root_ids"]:
        root = value["roots"][root_id]
        if (root["request_spec"] != manifest["gate_request_specs"][root_id]
                or [item["sample_id"] for item in root["samples"]] != expected_ids):
            raise RuntimeError(f"Full Child-Gate root artifact drift: {root_id}")
    return value


def _root_vote(rubric: StructuredRubric, outputs: Mapping[str, Any], root_id: str,
               selected: Sequence[str]):
    selected_set = set(selected)
    child_votes = [
        vg._node_vote(rubric, outputs, child.node_id)
        for child in rubric.children(root_id) if child.node_id in selected_set
    ]
    return aggregate_child_subtrees(vg._node_vote(rubric, outputs, root_id), child_votes)


def _selection_labels(rubric: StructuredRubric, prediction: PairwisePredictionOutput,
                      selections: Sequence[Mapping[str, Sequence[str]]]):
    if len(selections) != len(prediction.node_outputs):
        raise ValueError("Full Child-Gate selection matrix length mismatch")
    root_labels = {root_id: [] for root_id in rubric.root_ids}
    full_labels = []
    for outputs, selected in zip(prediction.node_outputs, selections):
        root_votes = {
            root_id: _root_vote(rubric, outputs, root_id, selected[root_id])
            for root_id in rubric.root_ids
        }
        for root_id, vote in root_votes.items():
            root_labels[root_id].append(vg._vote_label(vote))
        full_labels.append(aggregate_selected_roots(root_votes, rubric.root_ids).value)
    return root_labels, full_labels


def _selection_from_gate(gate: Mapping[str, Any], root_ids: Sequence[str]):
    count = gate["sample_count"]
    return [
        {
            root_id: tuple(gate["roots"][root_id]["samples"][index]["active_child_ids"])
            for root_id in root_ids
        }
        for index in range(count)
    ]


def _constant_selection(rubric: StructuredRubric, count: int, *, all_children: bool):
    value = {
        root_id: tuple(child.node_id for child in rubric.children(root_id))
        if all_children else tuple()
        for root_id in rubric.root_ids
    }
    return [dict(value) for _ in range(count)]


def _oracle_selection(rubric: StructuredRubric, prediction: PairwisePredictionOutput, rows):
    matrix = []
    for outputs, row in zip(prediction.node_outputs, rows):
        choices_by_root = []
        for root_id in rubric.root_ids:
            child_ids = tuple(child.node_id for child in rubric.children(root_id))
            representatives = {}
            for subset in vg._all_subsets(child_ids):
                vote = _root_vote(rubric, outputs, root_id, subset)
                representatives.setdefault(vote, subset)
            choices_by_root.append([
                (root_id, vote, subset)
                for vote, subset in representatives.items()
            ])
        all_children = {
            root_id: tuple(child.node_id for child in rubric.children(root_id))
            for root_id in rubric.root_ids
        }
        candidates = []
        for combination in product(*choices_by_root):
            root_votes = {root_id: vote for root_id, vote, _subset in combination}
            if aggregate_selected_roots(root_votes, rubric.root_ids).value == row["answer"]:
                selected = {root_id: subset for root_id, _vote, subset in combination}
                candidates.append(selected)
        matrix.append(min(
            candidates,
            key=lambda item: (sum(len(value) for value in item.values()), tuple(item.items())),
            default=all_children,
        ))
    return matrix


def _visual_only_selection(output: Path, rubric: StructuredRubric, rows, split: str):
    path = (output / VISUAL_GATE_EXPERIMENT /
            ("discovery90/primary/gate_predictions.json"
             if split == "discovery90" else "heldout500/gate/gate_predictions.json"))
    value = load_json(path)
    if [item["sample_id"] for item in value["samples"]] != [str(row["sample_id"]) for row in rows]:
        raise RuntimeError("Visual-only Gate sample identity drift")
    visual_id = vg.PARENT_NODE_ID
    all_children = _constant_selection(rubric, len(rows), all_children=True)
    for index, item in enumerate(value["samples"]):
        all_children[index][visual_id] = tuple(item["active_child_ids"])
    return all_children


def _system_metrics(rubric: StructuredRubric, prediction: PairwisePredictionOutput,
                    rows, selections):
    roots, full = _selection_labels(rubric, prediction, selections)
    full_metrics = vg._metrics(full, rows)
    full_metrics["accuracy_ci95_wilson"] = vg._wilson(
        full_metrics["correct_count"], len(rows))
    root_metrics = {}
    for root_id, labels in roots.items():
        value = vg._metrics(labels, rows)
        value["accuracy_ci95_wilson"] = vg._wilson(value["correct_count"], len(rows))
        root_metrics[root_id] = value
    return {"full_m1": full_metrics, "roots": root_metrics}


def _routing_diagnostics(rubric: StructuredRubric, prediction: PairwisePredictionOutput,
                         rows, gate: Mapping[str, Any], selections):
    by_root = {}
    total_active = 0
    for root_id in rubric.root_ids:
        child_ids = tuple(child.node_id for child in rubric.children(root_id))
        per_child = {
            child_id: {"active": 0, "decisive": 0, "correct": 0}
            for child_id in child_ids
        }
        all_conflicts = gated_conflicts = empty = 0
        active_counts: dict[str, int] = {}
        for index, (outputs, row) in enumerate(zip(prediction.node_outputs, rows)):
            selected = set(selections[index][root_id])
            total_active += len(selected)
            empty += int(not selected)
            active_counts[str(len(selected))] = active_counts.get(str(len(selected)), 0) + 1
            all_votes = [vg._node_vote(rubric, outputs, child_id) for child_id in child_ids]
            gated_votes = [vg._node_vote(rubric, outputs, child_id) for child_id in selected]
            all_conflicts += int(vg.Vote.A in all_votes and vg.Vote.B in all_votes)
            gated_conflicts += int(vg.Vote.A in gated_votes and vg.Vote.B in gated_votes)
            for child_id in selected:
                vote = vg._node_vote(rubric, outputs, child_id)
                item = per_child[child_id]
                item["active"] += 1
                if vote in {vg.Vote.A, vg.Vote.B}:
                    item["decisive"] += 1
                    item["correct"] += int(vote.value == row["answer"])
        for item in per_child.values():
            item["accuracy_on_active_decisive"] = (
                item["correct"] / item["decisive"] if item["decisive"] else 0.0)
        by_root[root_id] = {
            "child_count": len(child_ids),
            "active_children_distribution": active_counts,
            "mean_active_children": sum(
                len(item[root_id]) for item in selections) / len(rows),
            "empty_route_count": empty,
            "all_children_sibling_conflict_count": all_conflicts,
            "gated_sibling_conflict_count": gated_conflicts,
            "sibling_conflict_reduction": all_conflicts - gated_conflicts,
            "per_child": per_child,
            "gate_summary": gate["roots"][root_id]["summary"],
        }
    return {
        "fixed_child_count": 17,
        "mean_active_children_total": total_active / len(rows),
        "activation_reduction_rate": 1.0 - total_active / (len(rows) * 17),
        "roots": by_root,
    }


def _position_report(primary: Mapping[str, Any], swapped: Mapping[str, Any],
                     manifest: Mapping[str, Any]):
    roots = {}
    agreements = []
    for root_id in manifest["source"]["root_ids"]:
        child_ids = manifest["routing_contracts"][root_id]["child_ids"]
        value = vg._position_audit(
            primary["roots"][root_id], swapped["roots"][root_id], child_ids)
        roots[root_id] = value
        agreements.extend(value["per_child_status_agreement"].values())
    return {
        "roots": roots,
        "mean_root_exact_status_match_rate": statistics.fmean(
            item["exact_status_match_rate"] for item in roots.values()),
        "mean_per_child_status_agreement": statistics.fmean(agreements),
    }


def _build_report(output: Path, rubric: StructuredRubric,
                  prediction: PairwisePredictionOutput, rows, gate, split: str):
    parent = _constant_selection(rubric, len(rows), all_children=False)
    all_children = _constant_selection(rubric, len(rows), all_children=True)
    gated = _selection_from_gate(gate, rubric.root_ids)
    visual_only = _visual_only_selection(output, rubric, rows, split)
    oracle = _oracle_selection(rubric, prediction, rows)
    selections = {
        "parent_only": parent,
        "all_children": all_children,
        "visual_only_gate": visual_only,
        "full_child_gate": gated,
        "oracle_child_routing_upper_bound": oracle,
    }
    systems = {
        name: _system_metrics(rubric, prediction, rows, selected)
        for name, selected in selections.items()
    }
    return {
        "systems": systems,
        "paired": {
            "full_gate_vs_all_children": vg._paired(
                rows, systems["all_children"]["full_m1"]["predictions"],
                systems["full_child_gate"]["full_m1"]["predictions"]),
            "full_gate_vs_visual_only": vg._paired(
                rows, systems["visual_only_gate"]["full_m1"]["predictions"],
                systems["full_child_gate"]["full_m1"]["predictions"]),
        },
        "routing_diagnostics": _routing_diagnostics(
            rubric, prediction, rows, gate, gated),
    }


def freeze(config: Mapping[str, Any], output: Path) -> None:
    protocol = _protocol(config)
    target = _target(output)
    rubric = _rubric(output)
    rows = _rows(config, "discovery90")
    prediction = _prediction(output, rubric, rows, "discovery90")
    contracts = _contracts(rubric)
    spec = vg._gate_pool_spec(config, protocol["gate_endpoints"])
    manifest = _make_manifest(
        config, output, protocol, rubric, rows, prediction, contracts, spec,
        inspect_live=True)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        status = load_json(target / "stage_status.json") if (target / "stage_status.json").is_file() else {}
        if any(item.get("status") == "passed" for key, item in status.items()
               if key != "full-child-gate-freeze"):
            raise RuntimeError("Full Child-Gate manifest drift after downstream execution")
    atomic_write_json(path, manifest)
    for root_id, contract in contracts.items():
        atomic_write_json(target / "routing_contracts" / f"{root_id}.json", contract)
    parent = _system_metrics(
        rubric, prediction, rows,
        _constant_selection(rubric, len(rows), all_children=False))
    all_children = _system_metrics(
        rubric, prediction, rows,
        _constant_selection(rubric, len(rows), all_children=True))
    audit = {
        "schema_version": SCHEMA_VERSION,
        "rubric_sha256": rubric.rubric_sha256,
        "root_count": len(rubric.root_ids),
        "child_count": sum(len(rubric.children(root_id)) for root_id in rubric.root_ids),
        "node_count": len(rubric.nodes),
        "parent_only_discovery_m1": parent["full_m1"],
        "all_children_discovery_m1": all_children["full_m1"],
        "gate_contract_excludes_decision_rule": all(
            "decision_rule" not in child for contract in contracts.values()
            for child in contract["children"]),
        "heldout_accessed": False,
    }
    atomic_write_json(target / "offline_audit.json", audit)
    details = {
        "root_count": 5, "child_count": 17, "node_count": 22,
        "all_children_discovery_m1": all_children["full_m1"]["accuracy"],
        "heldout_accessed": False,
    }
    vg._set_status(target, "full-child-gate-freeze", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, protocol, _rubric_value, rows, _pred, contracts, spec = _load_frozen(config, output)
    selected = rows[:int(protocol["smoke_sample_count"])]
    artifact = _run_all_roots(
        config, target, manifest, protocol, contracts, spec, selected,
        target / "smoke", "full_child_gate_smoke")
    if artifact["summary"]["parse_valid_count"] != len(selected) * len(contracts):
        raise RuntimeError("Full Child-Gate smoke requires 100% parse-valid routes")
    details = {
        "sample_count": len(selected),
        "logical_route_count": artifact["summary"]["logical_route_count"],
        "parse_valid_rate": artifact["summary"]["parse_valid_rate"],
        "endpoint_call_counts": artifact["summary"]["endpoint_call_counts"],
    }
    vg._set_status(target, "full-child-gate-smoke", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def discovery(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, protocol, _rubric_value, rows, _pred, contracts, spec = _load_frozen(config, output)
    vg._require(target, "full-child-gate-smoke")
    primary = _run_all_roots(
        config, target, manifest, protocol, contracts, spec, rows,
        target / "discovery90" / "primary", "full_child_gate_discovery90")
    swapped = []
    for row in rows[:int(protocol["position_audit_count"])]:
        item = dict(row)
        item["sample_id"] = f"{row['sample_id']}::ab_swapped"
        item["A"], item["B"] = row["B"], row["A"]
        item["answer"] = "B" if row["answer"] == "A" else "A"
        swapped.append(item)
    position = _run_all_roots(
        config, target, manifest, protocol, contracts, spec, swapped,
        target / "discovery90" / "position_swapped",
        "full_child_gate_position_swapped")
    details = {
        "primary_route_count": primary["summary"]["logical_route_count"],
        "position_route_count": position["summary"]["logical_route_count"],
        "primary_valid_rate": primary["summary"]["parse_valid_rate"],
        "position_valid_rate": position["summary"]["parse_valid_rate"],
    }
    vg._set_status(target, "full-child-gate-discovery", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, protocol, rubric, rows, prediction, _contracts_value, _spec = _load_frozen(config, output)
    vg._require(target, "full-child-gate-discovery")
    primary = _load_gate(
        target / "discovery90/primary/gate_predictions.json", manifest, rows)
    swapped_rows = []
    for row in rows[:int(protocol["position_audit_count"])]:
        item = dict(row)
        item["sample_id"] = f"{row['sample_id']}::ab_swapped"
        item["A"], item["B"] = row["B"], row["A"]
        item["answer"] = "B" if row["answer"] == "A" else "A"
        swapped_rows.append(item)
    swapped = _load_gate(
        target / "discovery90/position_swapped/gate_predictions.json",
        manifest, swapped_rows)
    systems = _build_report(output, rubric, prediction, rows, primary, "discovery90")
    position = _position_report(primary, swapped, manifest)
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_DIR,
        "split": "discovery90",
        "rubric_sha256": rubric.rubric_sha256,
        "gate_summary": primary["summary"],
        "position_audit": position,
        **systems,
        "heldout_accessed": False,
        "selection_after_heldout_forbidden": True,
    }
    atomic_write_json(target / "discovery90/report.json", value)
    details = {
        "gate_valid_rate": primary["summary"]["parse_valid_rate"],
        "mean_active_children": systems["routing_diagnostics"]["mean_active_children_total"],
        "m1_accuracy": {
            name: item["full_m1"]["accuracy"]
            for name, item in systems["systems"].items()
        },
        "full_gate_vs_all_net_corrected": systems["paired"][
            "full_gate_vs_all_children"]["net_corrected"],
    }
    vg._set_status(target, "full-child-gate-report", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _heldout_manifest(config: Mapping[str, Any], output: Path, target: Path,
                      manifest: Mapping[str, Any], rubric: StructuredRubric, rows):
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "frozen_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
        "discovery_report_sha256": file_sha256(target / "discovery90/report.json"),
        "rubric_sha256": rubric.rubric_sha256,
        "heldout_dataset_sha256": file_sha256(base._path(config["heldout_dataset"])),
        "heldout_prediction_sha256": file_sha256(vg._prediction_path(output, "heldout500")),
        "sample_count": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "gate_request_specs": manifest["gate_request_specs"],
        "exploratory_reused_heldout": True,
        "selection_after_heldout_forbidden": True,
    }


def heldout(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, protocol, rubric, _drows, _dpred, contracts, spec = _load_frozen(config, output)
    vg._require(target, "full-child-gate-report")
    rows = _rows(config, "heldout500")
    prediction = _prediction(output, rubric, rows, "heldout500")
    frozen = _heldout_manifest(config, output, target, manifest, rubric, rows)
    path = target / "heldout500/frozen_manifest.json"
    if path.is_file() and load_json(path) != frozen:
        raise RuntimeError("Full Child-Gate heldout manifest drift")
    atomic_write_json(path, frozen)
    gate = _run_all_roots(
        config, target, manifest, protocol, contracts, spec, rows,
        target / "heldout500/gate", "full_child_gate_heldout500")
    systems = _build_report(output, rubric, prediction, rows, gate, "heldout500")
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_DIR,
        "split": "heldout500",
        "rubric_sha256": rubric.rubric_sha256,
        "gate_summary": gate["summary"],
        **systems,
        "exploratory_reused_heldout": True,
        "selection_after_heldout_forbidden": True,
    }
    atomic_write_json(target / "heldout500/report.json", value)
    details = {
        "gate_valid_rate": gate["summary"]["parse_valid_rate"],
        "mean_active_children": systems["routing_diagnostics"]["mean_active_children_total"],
        "m1_accuracy": {
            name: item["full_m1"]["accuracy"]
            for name, item in systems["systems"].items()
        },
        "full_gate_vs_all_net_corrected": systems["paired"][
            "full_gate_vs_all_children"]["net_corrected"],
    }
    vg._set_status(target, "full-child-gate-heldout-exploratory", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def final_report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, _protocol_value, rubric, _rows_value, _pred, _contracts_value, _spec = _load_frozen(config, output)
    vg._require(target, "full-child-gate-heldout-exploratory")
    discovery_value = load_json(target / "discovery90/report.json")
    heldout_value = load_json(target / "heldout500/report.json")
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_DIR,
        "rubric_sha256": rubric.rubric_sha256,
        "gate_request_specs": manifest["gate_request_specs"],
        "discovery": discovery_value,
        "heldout": heldout_value,
        "exploratory_reused_heldout": True,
    }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# Full-Rubric Child-Gate v1",
        "",
        "All five roots remain active; only their direct-child activation changes.",
        "",
        "| Split | System | Full M1 ACC | Coverage | Correct |",
        "|---|---|---:|---:|---:|",
    ]
    order = ("parent_only", "all_children", "visual_only_gate",
             "full_child_gate", "oracle_child_routing_upper_bound")
    for split_name, report_value in (("Discovery-90", discovery_value),
                                     ("Heldout-500", heldout_value)):
        for name in order:
            item = report_value["systems"][name]["full_m1"]
            lines.append(
                f"| {split_name} | {name} | {item['accuracy']:.2%} | "
                f"{item['coverage']:.2%} | {item['correct_count']} |")
    lines.extend([
        "",
        "Heldout-500 is a reused exploratory paired diagnostic.",
    ])
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    details = {
        "final_report": str(target / "final_report.json"),
        "heldout_full_gate_m1": heldout_value["systems"]["full_child_gate"]["full_m1"]["accuracy"],
        "heldout_all_children_m1": heldout_value["systems"]["all_children"]["full_m1"]["accuracy"],
        "heldout_net_corrected": heldout_value["paired"][
            "full_gate_vs_all_children"]["net_corrected"],
    }
    vg._set_status(target, "full-child-gate-final-report", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {
        "full-child-gate-freeze": freeze,
        "full-child-gate-smoke": smoke,
        "full-child-gate-discovery": discovery,
        "full-child-gate-report": report,
        "full-child-gate-heldout-exploratory": heldout,
        "full-child-gate-final-report": final_report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported Full Child-Gate stage: {stage}")
    started = time.perf_counter()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.perf_counter() - started:.1f}")

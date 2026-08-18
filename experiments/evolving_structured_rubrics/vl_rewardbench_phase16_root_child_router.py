"""Exploratory Root-Router + Child-Gate transfer evaluation for Phase16 E5.

Pairwise Prompt-v2 K=3 predictions are immutable inputs.  This module adds
only routing calls, so differences from the E5 all-children system can be
attributed to routing rather than another Worker generation pass.
"""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    CacheMode,
    JsonPredictionCache,
    ModelCallMetrics,
    PairwisePredictionOutput,
    StructuredRootRouter,
    StructuredRubric,
    aggregate_selected_roots,
    combine_model_call_metrics,
)
from critiq.structured.root_router import (
    RootRoutingPredictionOutput,
    root_routing_consistency_errors,
)

from . import full_child_gate as child_gate
from . import prompt_v2_aligned_evolution as evolution
from . import run_rubric_evolution as base
from . import run_shared_output_pool as shared
from . import visual_gate as gate_utils
from . import vl_rewardbench as legacy
from . import vl_rewardbench_phase10 as phase10
from . import vl_rewardbench_prompt_v2 as prompt_v2
from . import vl_rewardbench_prompt_v2_evolved as evolved
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, make_progress_callback
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_phase16_e5_root_child_router_v1"
PROTOCOL_VERSION = "vlrb-phase16-e5-root-child-router-v1"
SOURCE_STAGE = "retry"
# ``_logical_from_predictions`` retains the historical Phase10 system key even
# when its node outputs are the Phase16 E5 treatment predictions.  The source
# report labels the same system ``prompt_v2_evolved_final_equal``.
SOURCE_SYSTEM = "phase10_final_equal"
REPLICATE_SEEDS = (42, 43, 44)
STAGES = (
    "vlrb-phase16-router-freeze", "vlrb-phase16-router-audit",
    "vlrb-phase16-router-smoke", "vlrb-phase16-router-run",
    "vlrb-phase16-router-retry", "vlrb-phase16-router-report",
)


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {}
    value[stage] = {"status": "passed", "details": dict(details)}
    atomic_write_json(path, value)


def _require(target: Path, stage: str) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment": evolved.EXPERIMENT_DIR,
        "source_stage": SOURCE_STAGE,
        "source_system": SOURCE_SYSTEM,
        "source_rubric_experiment": evolution.EXPERIMENT_DIR,
        "endpoint_ids": ["vllm-8000", "vllm-8001"],
        "scheduler": "available_slot_dynamic",
        "root_router_temperature": 0.2,
        "child_gate_temperature": 0.2,
        "max_tokens": 2048,
        "max_retries": 5,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "smoke_sample_count": 20,
        "k": 3,
        "root_fallback": "all_roots",
        "child_fallback": "all_children_for_failed_root",
        "selection_after_benchmark_forbidden": True,
        "exploratory": True,
    }
    value = config.get("vlrb_phase16_root_child_router")
    if value != expected:
        raise RuntimeError("vlrb_phase16_root_child_router does not match the frozen v1 protocol")
    return dict(value)


def _rubric(output: Path) -> StructuredRubric:
    rubric = StructuredRubric.load_json(output / evolution.EXPERIMENT_DIR / "final" / "rubric.json")
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 23:
        raise RuntimeError("Phase16 E5 rubric identity drift")
    return rubric


def _records():
    return prompt_v2._records()


def _schedule(records):
    return legacy._order_schedule(records)


def _prediction_path(replicate: int) -> Path:
    return evolved._target() / SOURCE_STAGE / "structured" / "prompt_v2" / f"replicate_{replicate + 1:02d}" / "predictions.json"


def _source_predictions(rubric: StructuredRubric, records, schedule):
    result = []
    for replicate in range(legacy.K):
        path = _prediction_path(replicate)
        if not path.is_file():
            raise RuntimeError(f"Phase16 E5 prediction missing: {path}")
        prediction = PairwisePredictionOutput.load_json(path)
        rows = legacy._ordered_rows(records, schedule, replicate)
        gate_utils._validate_prediction(prediction, rubric, rows)
        result.append(prediction)
    return tuple(result)


def _contracts(rubric: StructuredRubric) -> dict[str, dict[str, Any]]:
    return {
        root_id: gate_utils.build_routing_contract(
            rubric, parent_node_id=root_id, expected_child_names=None)
        for root_id in rubric.root_ids
    }


def _gate_protocol(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    protocol = _protocol(config)
    return {
        "gate_endpoints": protocol["endpoint_ids"],
        "gate_temperature": protocol["child_gate_temperature"],
        "gate_max_tokens": protocol["max_tokens"],
        "gate_seed": seed,
        "gate_max_retries": protocol["max_retries"],
        "always_format_reminder": True,
        "prompt_version": gate_utils.PROMPT_VERSION,
    }


def _router(config: Mapping[str, Any], spec, seed: int) -> StructuredRootRouter:
    protocol = _protocol(config)
    return StructuredRootRouter(
        router_args={
            "model": config["model"], "api_keys": "EMPTY",
            "system": (
                "You are a multimodal rubric router. Select applicable roots only; "
                "never decide which answer is better. Return JSON only."),
            "request_kwargs": {"temperature": protocol["root_router_temperature"],
                               "max_tokens": protocol["max_tokens"], "seed": seed},
            "api_retry_attempts": config["api_retry_attempts"],
        },
        router_backend_id=spec.backend_id, max_retries=protocol["max_retries"],
        max_data_chars=None, encode_local_image=True,
    )


def _router_request_spec(config, rubric, spec, seed: int) -> dict[str, Any]:
    return _router(config, spec, seed).request_spec(rubric).to_dict()


def _manifest(config: Mapping[str, Any], output: Path, *, inspect_live: bool):
    protocol = _protocol(config)
    rubric = _rubric(output)
    records = _records(); schedule = _schedule(records)
    predictions = _source_predictions(rubric, records, schedule)
    contracts = _contracts(rubric)
    spec = gate_utils._gate_pool_spec(config, protocol["endpoint_ids"])
    identities = gate_utils._inspect_endpoints(config, spec) if inspect_live else None
    prior = _target() / "frozen_manifest.json"
    if identities is None and prior.is_file():
        identities = load_json(prior)["endpoint_identities"]
    source_report = evolved._target() / "final_report.json"
    if not source_report.is_file():
        raise RuntimeError("Phase16 E5 VL-RewardBench report is required")
    return {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "protocol": protocol,
        "dataset": {"sample_count": len(records), "record_sha256": canonical_sha256(list(records)),
                    "sample_ids": [str(row["sample_id"]) for row in records],
                    "order_schedule_sha256": canonical_sha256(schedule)},
        "source": {"rubric_path": str((output / evolution.EXPERIMENT_DIR / "final" / "rubric.json").resolve()),
                   "rubric_sha256": rubric.rubric_sha256, "node_count": len(rubric.nodes),
                   "prediction_paths": [str(_prediction_path(i).resolve()) for i in range(legacy.K)],
                   "prediction_sha256": [file_sha256(_prediction_path(i)) for i in range(legacy.K)],
                   "source_report_sha256": file_sha256(source_report)},
        "routing_contracts": {root_id: {"contract_sha256": contract["contract_sha256"],
                                        "child_ids": [child["node_id"] for child in contract["children"]]}
                              for root_id, contract in contracts.items()},
        "child_gate_request_specs": {f"replicate_{i + 1:02d}": {
            root_id: gate_utils._request_spec(config, _gate_protocol(config, seed), contract, spec)
            for root_id, contract in contracts.items()}
            for i, seed in enumerate(REPLICATE_SEEDS)},
        "root_router_request_specs": {f"replicate_{i + 1:02d}": _router_request_spec(config, rubric, spec, seed)
                                      for i, seed in enumerate(REPLICATE_SEEDS)},
        "endpoint_identities": identities,
    }


def _load_frozen(config, output):
    target = _target(); _require(target, STAGES[0])
    stored = load_json(target / "frozen_manifest.json")
    expected = _manifest(config, output, inspect_live=False)
    if stored != expected:
        raise RuntimeError("Phase16 root+child router frozen manifest drift")
    rubric = _rubric(output); records = _records(); schedule = _schedule(records)
    return target, stored, rubric, records, schedule, _contracts(rubric), gate_utils._gate_pool_spec(config, stored["protocol"]["endpoint_ids"])


def _verify_live(config, manifest, spec) -> None:
    if gate_utils._inspect_endpoints(config, spec) != manifest["endpoint_identities"]:
        raise RuntimeError("Phase16 root+child router endpoint identity drift")


def freeze(config, output) -> None:
    target = _target(); value = _manifest(config, output, inspect_live=True)
    path = target / "frozen_manifest.json"
    if path.is_file() and load_json(path) != value:
        status = load_json(target / "stage_status.json") if (target / "stage_status.json").is_file() else {}
        if any(item.get("status") == "passed" for key, item in status.items() if key != STAGES[0]):
            raise RuntimeError("Phase16 router manifest drift after inference")
    atomic_write_json(path, value)
    details = {"sample_count": value["dataset"]["sample_count"], "node_count": value["source"]["node_count"],
               "logical_root_router_requests": value["dataset"]["sample_count"] * legacy.K,
               "logical_child_gate_requests": value["dataset"]["sample_count"] * legacy.K * 5}
    _status(target, STAGES[0], details); print(json.dumps(details, indent=2))


def audit(config, output) -> None:
    target, manifest, rubric, records, schedule, _contracts_value, _spec = _load_frozen(config, output)
    predictions = _source_predictions(rubric, records, schedule)
    logical = load_json(evolved._target() / SOURCE_STAGE / "combined" / "logical_votes.json")
    replay = []
    for replicate, prediction in enumerate(predictions):
        all_children = child_gate._constant_selection(rubric, len(records), all_children=True)
        _roots, labels = child_gate._selection_labels(rubric, prediction, all_children)
        replay.append([legacy._original_index(label, int(schedule[str(row["sample_id"])][replicate]))
                       for label, row in zip(labels, records)])
    source = logical["systems"][SOURCE_SYSTEM]["votes_by_replicate"]
    if replay != source:
        raise RuntimeError("all-children E5 routing replay differs from frozen source")
    metrics = phase10._system_metrics(records, replay)
    details = {"all_children_replay_exact_match": True, "overall_acc": metrics["overall_acc"],
               "macro_acc": metrics["macro_acc"], "heldout_labels_used_for_routing": False}
    atomic_write_json(target / "offline_audit.json", details); _status(target, STAGES[1], details)
    print(json.dumps(details, indent=2))


def _run_router(config, target, manifest, rubric, rows, spec, seed, work: Path):
    pool = AvailableSlotBackendPool(spec)
    router = _router(config, spec, seed)
    expected = manifest["root_router_request_specs"][f"replicate_{REPLICATE_SEEDS.index(seed) + 1:02d}"]
    if router.request_spec(rubric).to_dict() != expected:
        raise RuntimeError("Root Router request identity drift")
    cache = JsonPredictionCache(work / "cache/router", CacheMode.READ_WRITE)
    callback = make_progress_callback(work, f"phase16_root_router_r{REPLICATE_SEEDS.index(seed) + 1}", len(rows), pool)
    router.call_backend = pool
    prediction, cache_hits = shared._router_pred_parallel(router, rows, rubric, spec.global_request_concurrency, callback, cache)
    prediction.save_json(work / "root_router_predictions.json")
    summary = {"parse_valid_rate": sum(item.parse_error is None for item in prediction.outputs) / len(rows),
               "cache_hit_count": cache_hits, "endpoint_call_counts": dict(pool.records_by_endpoint()),
               "request_spec": expected}
    atomic_write_json(work / "root_router_summary.json", summary)
    return prediction, summary


def _run_replicates(config, output, target, manifest, rubric, records, schedule, contracts, spec, *, work_name, count=None):
    selected = tuple(records[:count] if count is not None else records)
    gates = []; routers = []; summaries = []
    for replicate, seed in enumerate(REPLICATE_SEEDS):
        rows = legacy._ordered_rows(selected, schedule, replicate)
        gate_protocol = _gate_protocol(config, seed)
        expected_gate = manifest["child_gate_request_specs"][f"replicate_{replicate + 1:02d}"]
        actual_gate = {root_id: gate_utils._request_spec(config, gate_protocol, contract, spec)
                       for root_id, contract in contracts.items()}
        if actual_gate != expected_gate:
            raise RuntimeError("Child Gate request identity drift")
        work = target / work_name / f"replicate_{replicate + 1:02d}"
        gate_artifact = child_gate._run_all_roots(
            config, work, {"gate_endpoint_identities": manifest["endpoint_identities"],
                            "gate_request_specs": expected_gate}, gate_protocol, contracts, spec, rows, work,
            f"phase16_child_gate_r{replicate + 1}", cache_root=work / "cache/gate",
            experiment_name=EXPERIMENT_DIR)
        router_prediction, router_summary = _run_router(config, target, manifest, rubric, rows, spec, seed, work)
        gates.append(gate_artifact); routers.append(router_prediction); summaries.append(router_summary)
    return tuple(gates), tuple(routers), tuple(summaries)


def _artifact_failures(gates, routers):
    failures = []
    for replicate, gate in enumerate(gates):
        failures.extend({"kind": "child_gate", "replicate": replicate + 1, "root_id": root_id,
                         "sample_id": item["sample_id"], "parse_error": item["parse_error"]}
                        for root_id, root in gate["roots"].items() for item in root["samples"] if not item["parse_ok"])
    for replicate, router in enumerate(routers):
        failures.extend({"kind": "root_router", "replicate": replicate + 1,
                         "sample_id": sample_id, "parse_error": output.parse_error}
                        for sample_id, output in zip(router.sample_ids, router.outputs) if output.parse_error is not None)
    return failures


def smoke(config, output) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(config, output)
    _require(target, STAGES[1]); _verify_live(config, manifest, spec)
    gates, routers, summaries = _run_replicates(config, output, target, manifest, rubric, records, schedule, contracts, spec,
                                                  work_name="smoke", count=manifest["protocol"]["smoke_sample_count"])
    failures = _artifact_failures(gates, routers)
    if failures: raise RuntimeError("Root+Child routing smoke parse check failed")
    calls = {endpoint: sum(int(gate["summary"]["endpoint_call_counts"].get(endpoint, 0)) +
                           int(summary["endpoint_call_counts"].get(endpoint, 0))
                           for gate, summary in zip(gates, summaries))
             for endpoint in manifest["protocol"]["endpoint_ids"]}
    if any(value == 0 for value in calls.values()): raise RuntimeError("smoke did not use both endpoints")
    details = {"sample_count": manifest["protocol"]["smoke_sample_count"], "parse_valid_rate": 1.0,
               "endpoint_call_counts": calls, "cache_hit_count": sum(item["cache_hit_count"] for item in summaries)}
    _status(target, STAGES[2], details); atomic_write_json(target / "smoke/summary.json", details); print(json.dumps(details, indent=2))


def run(config, output) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(config, output)
    _require(target, STAGES[2]); _verify_live(config, manifest, spec)
    started = time.perf_counter()
    gates, routers, summaries = _run_replicates(config, output, target, manifest, rubric, records, schedule, contracts, spec, work_name="run")
    failures = _artifact_failures(gates, routers)
    details = {"sample_count": len(records), "parse_failure_count": len(failures),
               "wall_seconds": time.perf_counter() - started,
               "logical_requests": len(records) * legacy.K * 6}
    atomic_write_json(target / "run/failure_manifest.json", failures); _status(target, STAGES[3], details); print(json.dumps(details, indent=2))


def retry(config, output) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(config, output)
    _require(target, STAGES[3]); _verify_live(config, manifest, spec)
    # Gate cache knows to regenerate failed shards.  Router retries use a fresh
    # namespace, avoiding reuse of a cached invalid root decision.
    for replicate in range(legacy.K):
        source = target / "run" / f"replicate_{replicate + 1:02d}" / "cache/gate"
        destination = target / "retry" / f"replicate_{replicate + 1:02d}" / "cache/gate"
        if source.exists() and not destination.exists(): shutil.copytree(source, destination)
    gates, routers, _summaries = _run_replicates(config, output, target, manifest, rubric, records, schedule, contracts, spec, work_name="retry")
    failures = _artifact_failures(gates, routers)
    total = len(records) * legacy.K * 6
    details = {"still_failed_count": len(failures), "parse_valid_rate": 1 - len(failures) / total,
               "failed_items": failures}
    atomic_write_json(target / "retry/report.json", details); _status(target, STAGES[4], details); print(json.dumps(details, indent=2))


def _load_artifacts(target, rubric, records, schedule):
    gates = []; routers = []
    for replicate in range(legacy.K):
        rows = legacy._ordered_rows(records, schedule, replicate)
        work = target / "retry" / f"replicate_{replicate + 1:02d}"
        gates.append(load_json(work / "gate_predictions.json"))
        routers.append(RootRoutingPredictionOutput.load_json(work / "root_router_predictions.json"))
        if routers[-1].sample_ids != tuple(str(row["sample_id"]) for row in rows):
            raise RuntimeError("Root Router sample identity drift")
    return tuple(gates), tuple(routers)


def _labels(rubric, prediction, selections, roots):
    labels = []
    for outputs, selected_children, selected_roots in zip(prediction.node_outputs, selections, roots):
        root_votes = {root_id: child_gate._root_vote(rubric, outputs, root_id, selected_children[root_id])
                      for root_id in rubric.root_ids}
        labels.append(aggregate_selected_roots(root_votes, selected_roots).value)
    return labels


def _votes(rubric, records, schedule, predictions, gates, routers):
    systems = {"all_roots_all_children": [], "root_router_only": [], "child_gate_only": [], "root_router_child_gate": []}
    root_counts = []; child_counts = []
    for replicate, (prediction, gate, router) in enumerate(zip(predictions, gates, routers)):
        rows = legacy._ordered_rows(records, schedule, replicate)
        all_children = child_gate._constant_selection(rubric, len(rows), all_children=True)
        gated = child_gate._selection_from_gate(gate, rubric.root_ids)
        all_roots = [tuple(rubric.root_ids) for _ in rows]
        routed = [item.resolve(rubric).selected_root_ids for item in router.outputs]
        variants = {"all_roots_all_children": (all_children, all_roots), "root_router_only": (all_children, routed),
                    "child_gate_only": (gated, all_roots), "root_router_child_gate": (gated, routed)}
        for name, (selections, roots) in variants.items():
            labels = _labels(rubric, prediction, selections, roots)
            systems[name].append([legacy._original_index(label, int(schedule[str(row["sample_id"])][replicate]))
                                  for label, row in zip(labels, records)])
        root_counts.extend(len(item) for item in routed)
        child_counts.extend(sum(len(value) for value in selection.values()) for selection in gated)
    return systems, {"mean_active_roots": sum(root_counts) / len(root_counts),
                     "root_activation_reduction_rate": 1 - sum(root_counts) / (len(root_counts) * 5),
                     "mean_active_children": sum(child_counts) / len(child_counts),
                     "child_activation_reduction_rate": 1 - sum(child_counts) / (len(child_counts) * (len(rubric.nodes) - len(rubric.root_ids)))}


def report(config, output) -> None:
    target, manifest, rubric, records, schedule, _contracts_value, _spec = _load_frozen(config, output)
    _require(target, STAGES[4])
    predictions = _source_predictions(rubric, records, schedule)
    gates, routers = _load_artifacts(target, rubric, records, schedule)
    votes, diagnostics = _votes(rubric, records, schedule, predictions, gates, routers)
    metrics = {name: phase10._system_metrics(records, matrix) for name, matrix in votes.items()}
    baseline = metrics["all_roots_all_children"]["original_index_predictions"]
    paired = {name: legacy._paired(records, baseline, item["original_index_predictions"])
              for name, item in metrics.items() if name != "all_roots_all_children"}
    failures = load_json(target / "retry/report.json")
    value = {"schema_version": "1.0.0", "experiment": EXPERIMENT_DIR, "exploratory": True,
             "source_system": SOURCE_SYSTEM, "rubric_sha256": rubric.rubric_sha256,
             "metrics": metrics, "paired_vs_all_roots_all_children": paired,
             "routing_diagnostics": diagnostics, "parse": failures,
             "note": "K=3 root and child routing was generated without benchmark labels; no oracle routing is reported."}
    atomic_write_json(target / "final_report.json", value)
    lines = ["# Phase16 E5 Root Router + Child Gate", "",
             "Exploratory K=3 VL-RewardBench routing evaluation. Pairwise Prompt-v2 predictions are reused; only Router/Gate calls are new.", "",
             "| System | OverallAcc | MacroAcc | Coverage |", "|---|---:|---:|---:|"]
    for name, item in metrics.items():
        lines.append(f"| {name} | {item['overall_acc']:.2%} | {item['macro_acc']:.2%} | {item['coverage']:.2%} |")
    lines.extend(["", f"Root activation reduction: {diagnostics['root_activation_reduction_rate']:.2%}",
                  f"Child activation reduction: {diagnostics['child_activation_reduction_rate']:.2%}",
                  f"Parse valid rate: {failures['parse_valid_rate']:.2%}"])
    (target / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    details = {name: {"overall_acc": item["overall_acc"], "macro_acc": item["macro_acc"]} for name, item in metrics.items()}
    _status(target, STAGES[5], details); print(json.dumps(details, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {STAGES[0]: freeze, STAGES[1]: audit, STAGES[2]: smoke, STAGES[3]: run, STAGES[4]: retry, STAGES[5]: report}
    if stage not in actions: raise ValueError(f"unknown Phase16 root+child router stage: {stage}")
    started = time.monotonic(); actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

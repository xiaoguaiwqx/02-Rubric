"""K=3 counterfactual coalition audit over cached Phase21 root reports.

The audit never regenerates a Unified-Subtree report.  It first upgrades the
frozen Phase21 baseline and all 25 one-root counterfactual systems from K=1 to
K=3 by reusing the old call as replicate zero.  It then enumerates all
multi-root coalitions for epochs 1 and 5, using the five candidate root reports
from the corresponding epoch and the common epoch-start baseline reports.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import itertools
import json
import math
from pathlib import Path
import queue
import random
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from critiq.structured.backend_pool import BackendEndpointSpec, BackendPoolSpec
from critiq.structured.cache import canonical_sha256
from critiq.structured.schema import StructuredRubric

from . import _global_arbiter_ab_only_support as support
from . import aligned_system_runtime as runtime
from . import global_arbiter_ab_only as arbiter
from . import rejected_candidate_counterfactual_arbiter_audit as source_audit
from . import unified_subtree_bundle_evolution as phase21
from .experiment_utils import atomic_write_json, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase21_counterfactual_coalition_audit_k3_v1"
PROTOCOL_VERSION = "phase21-counterfactual-coalition-audit-k3-v1"
CONFIG_KEY = "phase21_counterfactual_coalition_audit_k3_v1_experiment"
COALITION_EPOCHS = (1, 5)
K = 3
STAGES = (
    "counterfactual-coalition-freeze",
    "counterfactual-coalition-audit",
    "counterfactual-coalition-smoke",
    "counterfactual-coalition-singletons",
    "counterfactual-coalition-coalitions",
    "counterfactual-coalition-retry",
    "counterfactual-coalition-report",
)

SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": phase21.EXPERIMENT_DIR,
    "source_k1_audit": source_audit.EXPERIMENT_DIR,
    "sample_count": 100,
    "candidate_count": 25,
    "coalition_epochs": [1, 5],
    "root_candidate_count_per_epoch": 5,
    "k": 3,
    "reuse_existing_k1_as_replicate_zero": True,
    "new_singleton_replicates": [1, 2],
    "enumerate_all_non_singleton_coalitions": True,
    "subtree_regeneration_count": 0,
    "ab_swap": False,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "system_metric": "strict_accuracy_after_k3_majority",
    "coalition_value": "correct_count_delta_vs_common_baseline",
    "exact_subset_selection": True,
    "shapley_attribution": True,
    "pairwise_interaction_audit": True,
    "bootstrap_seed": 42,
    "bootstrap_repetitions": 5000,
    "selection_after_results_forbidden": True,
}


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _phase21_source(output: Path) -> Path:
    return output / phase21.EXPERIMENT_DIR


def _k1_source(output: Path) -> Path:
    return output / source_audit.EXPERIMENT_DIR


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get(CONFIG_KEY)
    if value != SETTINGS:
        raise RuntimeError(f"{CONFIG_KEY} drift")
    return dict(value)


def _rows(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(phase21._rows(config, "discovery"))


def _runtime_settings(config: Mapping[str, Any]) -> runtime.RuntimeSettings:
    settings = _settings(config)
    return runtime.RuntimeSettings(
        temperature=float(settings["temperature"]),
        max_tokens=int(settings["max_tokens"]),
        max_parse_retries=int(settings["max_parse_retries"]),
        generation_seed_policy=str(settings["generation_seed_policy"]),
    )


def _status(target: Path, name: str, details: Mapping[str, Any]) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {
        "schema_version": "1.0.0", "stages": {}}
    value.setdefault("stages", {})[name] = {
        "status": "completed", "details": dict(details)}
    _write(path, value)


def _require_stage(target: Path, name: str) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {}
    if value.get("stages", {}).get(name, {}).get("status") != "completed":
        raise RuntimeError(f"run {name} first")


def _status_details(target: Path, name: str) -> dict[str, Any]:
    return load_json(target / "stage_status.json")["stages"][name]["details"]


def _source_manifest(output: Path) -> dict[str, Any]:
    path = _k1_source(output) / "frozen_manifest.json"
    report = _k1_source(output) / "final_report.json"
    if not path.is_file() or not report.is_file():
        raise RuntimeError("complete the K=1 counterfactual audit first")
    value = load_json(path)
    if value.get("protocol_version") != source_audit.PROTOCOL_VERSION:
        raise RuntimeError("K=1 counterfactual source protocol drift")
    if len(value.get("candidates", [])) != 25:
        raise RuntimeError("K=1 counterfactual source candidate count drift")
    return value


def _entries_by_epoch(manifest: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest["candidates"]:
        groups.setdefault(int(entry["epoch"]), []).append(dict(entry))
    for values in groups.values():
        values.sort(key=lambda item: int(item["root_index"]))
    return groups


def _system_id(epoch: int, mask: int) -> str:
    return "baseline" if mask == 0 else f"e{epoch:02d}_coalition_{mask:05b}"


def _coalition_spec(entries: Sequence[Mapping[str, Any]], mask: int) -> dict[str, Any]:
    selected = [dict(entry) for index, entry in enumerate(entries)
                if mask & (1 << index)]
    return {
        "epoch": int(entries[0]["epoch"]),
        "mask": mask,
        "system_id": _system_id(int(entries[0]["epoch"]), mask),
        "root_ids": [str(entry["root_id"]) for entry in selected],
        "candidate_ids": [str(entry["candidate_id"]) for entry in selected],
        "size": len(selected),
    }


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    target = _target(output)
    source_manifest = _source_manifest(output)
    rows = _rows(config)
    groups = _entries_by_epoch(source_manifest)
    if any(len(groups.get(epoch, [])) != 5 for epoch in COALITION_EPOCHS):
        raise RuntimeError("coalition epochs must each contain five root candidates")
    k1_report_path = _k1_source(output) / "final_report.json"
    phase21_control = (_phase21_source(output) / "epochs" / "epoch_00"
                       / "system" / "discovery.json")
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "settings": settings,
        "dataset": {
            "count": len(rows),
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "sha256": source_manifest["dataset"]["sha256"],
        },
        "source": {
            "k1_manifest_path": str(
                (_k1_source(output) / "frozen_manifest.json").resolve()),
            "k1_manifest_sha256": file_sha256(
                _k1_source(output) / "frozen_manifest.json"),
            "k1_report_path": str(k1_report_path.resolve()),
            "k1_report_sha256": file_sha256(k1_report_path),
            "phase21_control_path": str(phase21_control.resolve()),
            "phase21_control_sha256": file_sha256(phase21_control),
            "worker_endpoint_identities": source_manifest["source"][
                "worker_endpoint_identities"],
            "unified_runtime_identity": source_manifest["source"][
                "unified_runtime_identity"],
        },
        "candidates": source_manifest["candidates"],
        "coalition_epochs": {
            str(epoch): [_coalition_spec(groups[epoch], mask)
                         for mask in range(1 << len(groups[epoch]))]
            for epoch in COALITION_EPOCHS
        },
        "reused_k1_arbiter_request_count": 26 * len(rows),
        "planned_new_singleton_arbiter_request_count": 26 * len(rows) * 2,
        "planned_new_coalition_arbiter_request_count": (
            len(COALITION_EPOCHS) * 26 * len(rows) * K),
        "new_subtree_request_count": 0,
    }
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("counterfactual coalition manifest drift")
        print("counterfactual-coalition-freeze already completed")
        return
    _write(path, manifest)
    details = {
        "candidate_count": len(manifest["candidates"]),
        "coalition_epochs": list(COALITION_EPOCHS),
        "coalitions_per_epoch": 32,
        "reused_k1_requests": manifest["reused_k1_arbiter_request_count"],
        "planned_new_arbiter_requests": (
            manifest["planned_new_singleton_arbiter_request_count"]
            + manifest["planned_new_coalition_arbiter_request_count"]),
        "new_subtree_requests": 0,
    }
    _status(target, STAGES[0], details)
    print(json.dumps(details, indent=2))


def _load_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _settings(config)
    path = _target(output) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"run {STAGES[0]} first")
    value = load_json(path)
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("counterfactual coalition protocol drift")
    return value


def audit(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    manifest = _load_manifest(config, output)
    current = _source_manifest(output)
    rows = _rows(config)
    sample_ids = [str(row["sample_id"]) for row in rows]
    k1_report = load_json(_k1_source(output) / "final_report.json")
    checks = {
        "sample_count_100": len(rows) == 100,
        "sample_order_match": sample_ids == manifest["dataset"]["sample_ids"],
        "candidate_count_25": len(current["candidates"]) == 25,
        "candidate_manifest_match": current["candidates"] == manifest["candidates"],
        "k1_report_candidate_count_25": k1_report.get("candidate_count") == 25,
        "k1_report_has_zero_new_subtrees": (
            k1_report.get("new_subtree_request_count") == 0),
        "all_k1_candidates_technical_valid": all(
            item.get("system_prediction_distribution", {}).get(
                "technical_failure") == 0
            for item in k1_report.get("candidate_results", [])),
        "epoch_1_has_five_candidates": len(
            _entries_by_epoch(current).get(1, [])) == 5,
        "epoch_5_has_five_candidates": len(
            _entries_by_epoch(current).get(5, [])) == 5,
        "subtree_regeneration_zero": SETTINGS["subtree_regeneration_count"] == 0,
        "k_is_three": SETTINGS["k"] == 3,
    }
    for entry in current["candidates"]:
        prediction = (_k1_source(output) / "predictions"
                      / f"{entry['candidate_id']}.json")
        checks[f"k1_prediction_{entry['candidate_id']}"] = prediction.is_file()
    if all(checks.values()):
        groups = _entries_by_epoch(current)
        sample_ids = [str(row["sample_id"]) for row in rows]
        baseline_bundles = _bundle_index(output, groups[1], 0, sample_ids)
        baseline_k1 = _existing_replicate_zero(output, groups[1], 0)
        checks["baseline_k1_report_bundles_match"] = all(
            baseline_k1[sample_id]["source_report_bundle_sha256"]
            == baseline_bundles[sample_id][1]
            for sample_id in sample_ids)
        singleton_match = True
        for entries in groups.values():
            for index in range(len(entries)):
                mask = 1 << index
                bundles = _bundle_index(output, entries, mask, sample_ids)
                k1 = _existing_replicate_zero(output, entries, mask)
                singleton_match = singleton_match and all(
                    k1[sample_id]["source_report_bundle_sha256"]
                    == bundles[sample_id][1]
                    for sample_id in sample_ids)
        checks["all_singleton_k1_report_bundles_match"] = singleton_match
    if not all(checks.values()):
        raise RuntimeError(f"counterfactual coalition offline audit failed: {checks}")
    value = {"schema_version": "1.0.0", "offline_only": True, "checks": checks}
    _write(target / "offline_audit.json", value)
    _status(target, STAGES[1], checks)
    print(json.dumps(checks, indent=2))


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if source_audit._endpoint_identities_with_retry(config) != manifest["source"][
            "worker_endpoint_identities"]:
        raise RuntimeError("counterfactual coalition endpoint identity drift")
    if phase21._unified_runtime_identity(config) != manifest["source"][
            "unified_runtime_identity"]:
        raise RuntimeError("counterfactual coalition prompt/parser/model drift")


def _root_order(output: Path) -> tuple[str, ...]:
    rubric = StructuredRubric.load_json(
        _phase21_source(output) / "epochs" / "epoch_00" / "rubric_committed.json")
    return tuple(rubric.root_ids)


def _bundle_index(
    output: Path,
    entries: Sequence[Mapping[str, Any]],
    mask: int,
    sample_ids: Sequence[str],
) -> dict[str, tuple[list[dict[str, Any]], str]]:
    source = _phase21_source(output)
    by_root = {str(entry["root_id"]): entry for entry in entries}
    if len(by_root) != 5:
        raise RuntimeError("coalition requires exactly one candidate per root")
    baseline_paths = dict(entries[0]["baseline_root_paths"])
    root_order = _root_order(output)
    baseline_rubric = StructuredRubric.load_json(
        source / "epochs" / "epoch_00" / "rubric_committed.json")
    indexes = {}
    for index, root_id in enumerate(root_order):
        entry = by_root[root_id]
        relative = (entry["candidate_root_path"] if mask & (1 << index)
                    else baseline_paths[root_id])
        indexes[root_id] = source_audit._sample_index(
            source_audit._artifact(source, str(relative)))
    bundles = {}
    for sample_id in sample_ids:
        reports = []
        for root_id in root_order:
            call = indexes[root_id][sample_id]["call"]
            if not call.get("parse_ok"):
                raise RuntimeError(f"unresolved cached subtree report: {sample_id}")
            reports.append({
                "root_id": root_id,
                "criterion_name": baseline_rubric.get_node(root_id).criterion.name,
                "report": call["parsed"],
            })
        bundles[sample_id] = (reports, canonical_sha256(reports))
    return bundles


def _existing_replicate_zero(
    output: Path, entries: Sequence[Mapping[str, Any]], mask: int,
) -> dict[str, dict[str, Any]]:
    if mask == 0:
        value = runtime.load(
            _phase21_source(output) / "epochs" / "epoch_00"
            / "system" / "discovery.json")
        return {
            str(item["sample_id"]): {
                "source_report_bundle_sha256": item["replicates"]["0"][
                    "report_bundle_sha256"],
                "arbiter": support.compact_call(
                    item["replicates"]["0"]["arbiter"]),
            }
            for item in value["samples"]
        }
    if mask.bit_count() == 1:
        index = next(index for index in range(5) if mask & (1 << index))
        entry = entries[index]
        value = load_json(
            _k1_source(output) / "predictions" / f"{entry['candidate_id']}.json")
        return {
            str(item["sample_id"]): {
                "source_report_bundle_sha256": item[
                    "source_report_bundle_sha256"],
                "arbiter": item["arbiter"],
            }
            for item in value["samples"]
        }
    return {}


def _majority(votes: Sequence[str]) -> str:
    if votes.count("A") >= 2:
        return "A"
    if votes.count("B") >= 2:
        return "B"
    return "None"


def _system_metrics(predictions: Sequence[str],
                    rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold = [str(row["answer"]) for row in rows]
    correct = [prediction == target for prediction, target in zip(predictions, gold)]
    decisive = [prediction in {"A", "B"} for prediction in predictions]
    coverage = sum(decisive)
    return {
        "sample_count": len(rows),
        "strict_accuracy": sum(correct) / len(rows),
        "correct_count": sum(correct),
        "coverage": coverage / len(rows),
        "covered_accuracy": (
            sum(ok and active for ok, active in zip(correct, decisive)) / coverage
            if coverage else 0.0),
        "none_rate": predictions.count("None") / len(rows),
        "prediction_distribution": {
            "A": predictions.count("A"),
            "B": predictions.count("B"),
            "None": predictions.count("None"),
        },
        "predictions": list(predictions),
    }


def _arbiter_call(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, target: Path,
    system_id: str, row: Mapping[str, Any], reports: Sequence[Mapping[str, Any]],
    digest: str, replicate: int, total_attempt_limit: int, namespace: str,
) -> dict[str, Any]:
    settings = _runtime_settings(config)
    call = support.call_one(
        config, endpoint, target / "cache" / namespace,
        user_text=support.global_arbiter_user_prompt(row, reports), row=row,
        request_key={
            "kind": "phase21_counterfactual_coalition_arbiter",
            "system_id": system_id,
            "sample_id": str(row["sample_id"]),
            "source_report_bundle_sha256": digest,
            "replicate": replicate,
            "order": 0,
        },
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=runtime.unified.ARBITER_PROMPT_VERSION,
        system_prompt=arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        response_parser=arbiter.parse_global_arbiter_ab_only_response,
        settings_loader=settings.as_loader(),
    )
    return support.compact_call(call)


def _run_system(
    config: Mapping[str, Any], output: Path, entries: Sequence[Mapping[str, Any]],
    mask: int, rows: Sequence[Mapping[str, Any]], path: Path,
    *, total_attempt_limit: int, namespace: str,
) -> dict[str, Any]:
    if path.is_file():
        old = load_json(path)
        if old.get("technical_failure_call_count") == 0 and old.get("k") == K:
            return old
    target = _target(output)
    epoch = int(entries[0]["epoch"])
    system_id = _system_id(epoch, mask)
    sample_ids = [str(row["sample_id"]) for row in rows]
    bundles = _bundle_index(output, entries, mask, sample_ids)
    replicate_zero = _existing_replicate_zero(output, entries, mask)
    values: dict[str, dict[str, Any]] = {
        sample_id: {"replicates": {}} for sample_id in sample_ids}
    for sample_id in sample_ids:
        item = replicate_zero.get(sample_id)
        if item is None:
            continue
        if item["source_report_bundle_sha256"] != bundles[sample_id][1]:
            raise RuntimeError(f"reused K=1 report bundle drift: {system_id}::{sample_id}")
        values[sample_id]["replicates"]["0"] = item["arbiter"]

    tasks = []
    for row in rows:
        sample_id = str(row["sample_id"])
        for replicate in range(K):
            if str(replicate) not in values[sample_id]["replicates"]:
                tasks.append((row, replicate))
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoints = tuple(item for item in pool.endpoints
                      if item.endpoint_id in runtime.ENDPOINT_IDS)
    if tuple(item.endpoint_id for item in endpoints) != runtime.ENDPOINT_IDS:
        raise RuntimeError("coalition audit requires vllm-8000 and vllm-8001")
    assignment_path = target / "endpoint_assignment.json"
    assignments = load_json(assignment_path) if assignment_path.is_file() else {}
    pending: queue.Queue[tuple[Mapping[str, Any], int]] = queue.Queue()
    assigned = {endpoint.endpoint_id: queue.Queue() for endpoint in endpoints}
    for task in tasks:
        row, replicate = task
        key = f"{system_id}::{row['sample_id']}::r{replicate}"
        endpoint_id = assignments.get(key)
        (assigned[endpoint_id] if endpoint_id in assigned else pending).put(task)
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()
    print(f"coalition_{system_id}: 0/{len(tasks)} new Arbiter calls started", flush=True)

    def worker(endpoint: BackendEndpointSpec) -> None:
        nonlocal completed
        while True:
            try:
                row, replicate = assigned[endpoint.endpoint_id].get_nowait()
            except queue.Empty:
                try:
                    row, replicate = pending.get_nowait()
                except queue.Empty:
                    return
            sample_id = str(row["sample_id"])
            key = f"{system_id}::{sample_id}::r{replicate}"
            with lock:
                if key not in assignments:
                    assignments[key] = endpoint.endpoint_id
                    _write(assignment_path, assignments)
            reports, digest = bundles[sample_id]
            call = _arbiter_call(
                config, endpoint, target, system_id, row, reports, digest,
                replicate, total_attempt_limit, namespace)
            with lock:
                values[sample_id]["replicates"][str(replicate)] = call
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(tasks) - completed)
                print(
                    f"coalition_{system_id}: {completed}/{len(tasks)} "
                    f"sample={sample_id} replicate={replicate} "
                    f"endpoint={endpoint.endpoint_id} elapsed={elapsed/60:.1f}m "
                    f"ETA={eta/60:.1f}m", flush=True)

    workers = runtime._interleaved_workers(endpoints)
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))

    samples = []
    technical = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        replicates = values[sample_id]["replicates"]
        if set(replicates) != {"0", "1", "2"}:
            raise RuntimeError(f"incomplete K=3 system: {system_id}::{sample_id}")
        votes = [runtime._answer(replicates[str(index)], 0) for index in range(K)]
        technical += votes.count("technical_failure")
        samples.append({
            "sample_id": sample_id,
            "order": 0,
            "source_report_bundle_sha256": bundles[sample_id][1],
            "replicates": replicates,
            "votes": votes,
            "majority_answer": _majority(votes),
        })
    predictions = [item["majority_answer"] for item in samples]
    result = {
        "schema_version": "1.0.0",
        "protocol_version": PROTOCOL_VERSION,
        "system_id": system_id,
        "epoch": epoch,
        "mask": mask,
        "root_ids": _coalition_spec(entries, mask)["root_ids"],
        "candidate_ids": _coalition_spec(entries, mask)["candidate_ids"],
        "k": K,
        "reused_k1_request_count": sum(
            "0" in item["replicates"] for item in values.values()),
        "new_arbiter_request_count": len(tasks),
        "new_subtree_request_count": 0,
        "technical_failure_call_count": technical,
        "samples": samples,
        "metrics": _system_metrics(predictions, rows),
        "wall_seconds": time.perf_counter() - started,
    }
    _write(path, result)
    _write(assignment_path, assignments)
    return result


def _smoke_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("source", "unknown")), []).append(row)
    selected = [items[0] for _, items in sorted(groups.items()) if items]
    used = {str(row["sample_id"]) for row in selected}
    selected.extend(row for row in rows if str(row["sample_id"]) not in used)
    return tuple(selected[:12])


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require_stage(target, STAGES[1])
    manifest = _load_manifest(config, output)
    _verify_live(config, manifest)
    groups = _entries_by_epoch(manifest)
    rows = _smoke_rows(_rows(config))
    epoch_entries = groups[1]
    specs = (
        (0, "baseline", "singletons"),
        (1, "singleton", "singletons"),
        (3, "pair", "coalitions_epoch_01"),
    )
    outputs = []
    for mask, label, namespace in specs:
        outputs.append(_run_system(
            config, output, epoch_entries, mask, rows,
            target / "smoke" / f"{label}.json",
            total_attempt_limit=1 + SETTINGS["max_parse_retries"],
            namespace=namespace))
    technical = sum(item["technical_failure_call_count"] for item in outputs)
    value = {
        "status": "passed" if technical == 0 else "failed",
        "sample_count": len(rows),
        "systems": [item["system_id"] for item in outputs],
        "k": K,
        "technical_failure_call_count": technical,
        "new_subtree_request_count": 0,
    }
    _write(target / "smoke" / "report.json", value)
    if technical:
        raise RuntimeError(f"counterfactual coalition smoke has {technical} failures")
    _status(target, STAGES[2], value)
    print(json.dumps(value, indent=2))


def singletons(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require_stage(target, STAGES[2])
    manifest = _load_manifest(config, output)
    _verify_live(config, manifest)
    groups = _entries_by_epoch(manifest)
    rows = _rows(config)
    values = []
    baseline_entries = groups[1]
    values.append(_run_system(
        config, output, baseline_entries, 0, rows,
        target / "singletons" / "baseline.json",
        total_attempt_limit=1 + SETTINGS["max_parse_retries"],
        namespace="singletons"))
    for epoch in sorted(groups):
        entries = groups[epoch]
        for index, entry in enumerate(entries):
            values.append(_run_system(
                config, output, entries, 1 << index, rows,
                target / "singletons" / f"{entry['candidate_id']}.json",
                total_attempt_limit=1 + SETTINGS["max_parse_retries"],
                namespace="singletons"))
    technical = sum(item["technical_failure_call_count"] for item in values)
    details = {
        "system_count": len(values),
        "logical_k3_arbiter_requests": len(values) * len(rows) * K,
        "reused_k1_arbiter_requests": sum(
            item["reused_k1_request_count"] for item in values),
        "new_arbiter_requests": sum(item["new_arbiter_request_count"] for item in values),
        "technical_failure_call_count": technical,
        "new_subtree_request_count": 0,
    }
    _status(target, STAGES[3], details)
    print(json.dumps(details, indent=2))


def coalitions(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require_stage(target, STAGES[3])
    manifest = _load_manifest(config, output)
    _verify_live(config, manifest)
    groups = _entries_by_epoch(manifest)
    rows = _rows(config)
    values = []
    for epoch in COALITION_EPOCHS:
        entries = groups[epoch]
        for mask in range(1, 1 << len(entries)):
            if mask.bit_count() <= 1:
                continue
            values.append(_run_system(
                config, output, entries, mask, rows,
                target / "coalitions" / f"epoch_{epoch:02d}"
                / f"mask_{mask:05b}.json",
                total_attempt_limit=1 + SETTINGS["max_parse_retries"],
                namespace=f"coalitions_epoch_{epoch:02d}"))
    technical = sum(item["technical_failure_call_count"] for item in values)
    details = {
        "epoch_count": len(COALITION_EPOCHS),
        "new_non_singleton_system_count": len(values),
        "logical_arbiter_requests": len(values) * len(rows) * K,
        "new_arbiter_requests": sum(item["new_arbiter_request_count"] for item in values),
        "technical_failure_call_count": technical,
        "new_subtree_request_count": 0,
    }
    _status(target, STAGES[4], details)
    print(json.dumps(details, indent=2))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require_stage(target, STAGES[4])
    manifest = _load_manifest(config, output)
    _verify_live(config, manifest)
    groups = _entries_by_epoch(manifest)
    rows = _rows(config)
    values = [_run_system(
        config, output, groups[1], 0, rows,
        target / "singletons" / "baseline.json",
        total_attempt_limit=1 + 2 * SETTINGS["max_parse_retries"],
        namespace="singletons")]
    for epoch in sorted(groups):
        for index, entry in enumerate(groups[epoch]):
            values.append(_run_system(
                config, output, groups[epoch], 1 << index, rows,
                target / "singletons" / f"{entry['candidate_id']}.json",
                total_attempt_limit=1 + 2 * SETTINGS["max_parse_retries"],
                namespace="singletons"))
    for epoch in COALITION_EPOCHS:
        for mask in range(1, 32):
            if mask.bit_count() <= 1:
                continue
            values.append(_run_system(
                config, output, groups[epoch], mask, rows,
                target / "coalitions" / f"epoch_{epoch:02d}"
                / f"mask_{mask:05b}.json",
                total_attempt_limit=1 + 2 * SETTINGS["max_parse_retries"],
                namespace=f"coalitions_epoch_{epoch:02d}"))
    technical = sum(item["technical_failure_call_count"] for item in values)
    details = {
        "system_count": len(values),
        "unresolved_technical_failure_calls": technical,
        "target_total_attempt_limit": 21,
    }
    _status(target, STAGES[5], details)
    print(json.dumps(details, indent=2))


def _paired(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return runtime.paired(baseline["metrics"], candidate["metrics"], rows)


def _sign(value: int | float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _exact_best(values: Mapping[int, int]) -> int:
    return min(values, key=lambda mask: (-values[mask], mask.bit_count(), mask))


def _shapley(values: Mapping[int, float], n: int) -> list[float]:
    result = []
    denominator = math.factorial(n)
    for player in range(n):
        total = 0.0
        bit = 1 << player
        for mask in range(1 << n):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (math.factorial(size) * math.factorial(n - size - 1)
                      / denominator)
            total += weight * (values[mask | bit] - values[mask])
        result.append(total)
    return result


def _interaction(values: Mapping[int, float], left: int, right: int) -> float:
    return (values[(1 << left) | (1 << right)]
            - values[1 << left] - values[1 << right] + values[0])


def _bootstrap_best(
    systems: Mapping[int, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
    *, repetitions: int, seed: int,
) -> dict[str, Any]:
    gold = [str(row["answer"]) for row in rows]
    correct = {
        mask: [prediction == target for prediction, target in zip(
            value["metrics"]["predictions"], gold)]
        for mask, value in systems.items()
    }
    generator = random.Random(seed)
    counts: Counter[int] = Counter()
    for _ in range(repetitions):
        indices = [generator.randrange(len(rows)) for _ in rows]
        baseline_count = sum(correct[0][index] for index in indices)
        utilities = {
            mask: sum(items[index] for index in indices) - baseline_count
            for mask, items in correct.items()
        }
        counts[_exact_best(utilities)] += 1
    return {
        "repetitions": repetitions,
        "best_mask_frequency": {
            f"{mask:05b}": count for mask, count in counts.most_common()},
        "modal_best_mask": f"{counts.most_common(1)[0][0]:05b}",
        "modal_best_rate": counts.most_common(1)[0][1] / repetitions,
    }


def _load_singleton(target: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    return load_json(target / "singletons" / f"{entry['candidate_id']}.json")


def _telemetry(value: Mapping[str, Any]) -> dict[str, Any]:
    calls = [call for sample in value["samples"]
             for call in sample["replicates"].values()]
    return {
        "logical_arbiter_requests": len(calls),
        "model_generations": sum(int(call.get("model_generation_count", 0))
                                 for call in calls),
        "api_attempts": sum(int(call.get("metrics", {}).get("api_attempts") or 0)
                            for call in calls),
        "input_tokens": sum(int(call.get("metrics", {}).get("input_tokens") or 0)
                            for call in calls),
        "output_tokens": sum(int(call.get("metrics", {}).get("output_tokens") or 0)
                             for call in calls),
        "errors": sum(int(call.get("metrics", {}).get("error_count") or 0)
                      for call in calls),
        "wall_seconds": float(value.get("wall_seconds") or 0.0),
    }


def _sum_telemetry(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("logical_arbiter_requests", "model_generations", "api_attempts",
              "input_tokens", "output_tokens", "errors", "wall_seconds")
    items = [_telemetry(value) for value in values]
    return {field: sum(item[field] for item in items) for field in fields}


def _epoch_systems(
    target: Path, entries: Sequence[Mapping[str, Any]], epoch: int,
) -> dict[int, dict[str, Any]]:
    systems = {0: load_json(target / "singletons" / "baseline.json")}
    for index, entry in enumerate(entries):
        systems[1 << index] = _load_singleton(target, entry)
    for mask in range(1, 32):
        if mask.bit_count() <= 1:
            continue
        systems[mask] = load_json(
            target / "coalitions" / f"epoch_{epoch:02d}" / f"mask_{mask:05b}.json")
    return systems


def _write_epoch_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def report(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require_stage(target, STAGES[4])
    manifest = _load_manifest(config, output)
    rows = _rows(config)
    groups = _entries_by_epoch(manifest)
    baseline = load_json(target / "singletons" / "baseline.json")
    if baseline["technical_failure_call_count"]:
        raise RuntimeError("K=3 baseline has unresolved technical failures")
    k1_report = load_json(_k1_source(output) / "final_report.json")
    k1_by_id = {str(item["candidate_id"]): item
                for item in k1_report["candidate_results"]}
    singleton_records = []
    for epoch in sorted(groups):
        for entry in groups[epoch]:
            system = _load_singleton(target, entry)
            if system["technical_failure_call_count"]:
                raise RuntimeError(f"K=3 singleton failure: {entry['candidate_id']}")
            paired = _paired(baseline, system, rows)
            old = k1_by_id[str(entry["candidate_id"])]
            singleton_records.append({
                "candidate_id": entry["candidate_id"],
                "epoch": epoch,
                "root_id": entry["root_id"],
                "k1_system_net": int(old["system_net_corrected"]),
                "k3_system_net": int(paired["net_corrected"]),
                "k1_system_strict_accuracy": float(old["system_strict_accuracy"]),
                "k3_system_strict_accuracy": system["metrics"]["strict_accuracy"],
                "k1_k3_sign_match": (_sign(old["system_net_corrected"])
                                      == _sign(paired["net_corrected"])),
                "k3_corrected": paired["corrected"],
                "k3_harmed": paired["harmed"],
                "k3_none_rate": system["metrics"]["none_rate"],
                "scope_selective_utility_delta": float(
                    old["scope_selective_utility_delta"]),
                "formal_net_corrected_normalized": float(
                    old["formal_net_corrected_normalized"]),
                "covered_accuracy_delta": float(old["covered_accuracy_delta"]),
                "coverage_delta": float(old["coverage_delta"]),
                "full_selective_utility_delta": float(
                    old["full_selective_utility_delta"]),
                "k3_system_strict_accuracy_delta": (
                    system["metrics"]["strict_accuracy"]
                    - baseline["metrics"]["strict_accuracy"]),
            })
    _write_epoch_csv(target / "k3_singleton_results.csv", singleton_records)
    k1_values = [float(item["k1_system_net"]) for item in singleton_records]
    k3_values = [float(item["k3_system_net"]) for item in singleton_records]
    stability = {
        "candidate_count": len(singleton_records),
        "baseline_k1_strict_accuracy": k1_report["common_control"]["strict_accuracy"],
        "baseline_k3_strict_accuracy": baseline["metrics"]["strict_accuracy"],
        "sign_match_count": sum(item["k1_k3_sign_match"] for item in singleton_records),
        "sign_match_rate": sum(item["k1_k3_sign_match"] for item in singleton_records)
        / len(singleton_records),
        "k1_k3_pearson": source_audit._pearson(k1_values, k3_values),
        "k1_k3_spearman": source_audit._spearman(k1_values, k3_values),
        "records": singleton_records,
    }
    _write(target / "k1_k3_stability_report.json", stability)
    outcome = "k3_system_strict_accuracy_delta"
    local_predictor_correlations = {
        "primary_selective_utility": source_audit._correlations(
            singleton_records, "scope_selective_utility_delta", outcome),
        "formal_strict_net": source_audit._correlations(
            singleton_records, "formal_net_corrected_normalized", outcome),
        "covered_accuracy_delta": source_audit._correlations(
            singleton_records, "covered_accuracy_delta", outcome),
        "coverage_delta": source_audit._correlations(
            singleton_records, "coverage_delta", outcome),
        "full_data_selective_utility": source_audit._correlations(
            singleton_records, "full_selective_utility_delta", outcome),
    }
    _write(target / "k3_local_predictor_correlations.json",
           local_predictor_correlations)

    epoch_reports = {}
    for epoch in COALITION_EPOCHS:
        entries = groups[epoch]
        systems = _epoch_systems(target, entries, epoch)
        if any(value["technical_failure_call_count"] for value in systems.values()):
            raise RuntimeError(f"epoch {epoch} coalition has technical failures")
        values = {mask: int(_paired(baseline, system, rows)["net_corrected"])
                  for mask, system in systems.items()}
        best_mask = _exact_best(values)
        singleton_positive_mask = sum(
            1 << index for index in range(5) if values[1 << index] > 0)
        records = []
        for mask in range(32):
            paired = _paired(baseline, systems[mask], rows)
            records.append({
                "mask": f"{mask:05b}",
                "size": mask.bit_count(),
                "root_ids": "|".join(_coalition_spec(entries, mask)["root_ids"]),
                "strict_accuracy": systems[mask]["metrics"]["strict_accuracy"],
                "coverage": systems[mask]["metrics"]["coverage"],
                "corrected": paired["corrected"],
                "harmed": paired["harmed"],
                "net_corrected": paired["net_corrected"],
                "additivity_residual": values[mask] - sum(
                    values[1 << index] for index in range(5)
                    if mask & (1 << index)),
                "is_exact_best": mask == best_mask,
            })
        epoch_dir = target / f"epoch_{epoch:02d}"
        _write_epoch_csv(epoch_dir / "coalition_results.csv", records)
        shapley_values = _shapley(values, 5)
        shapley = {
            entries[index]["root_id"]: shapley_values[index]
            for index in range(5)}
        _write(epoch_dir / "shapley_values.json", shapley)
        interaction_rows = []
        for left, right in itertools.combinations(range(5), 2):
            interaction_rows.append({
                "left_root_id": entries[left]["root_id"],
                "right_root_id": entries[right]["root_id"],
                "interaction": _interaction(values, left, right),
            })
        _write_epoch_csv(epoch_dir / "pairwise_interactions.csv", interaction_rows)
        bootstrap = _bootstrap_best(
            systems, rows, repetitions=SETTINGS["bootstrap_repetitions"],
            seed=SETTINGS["bootstrap_seed"] + epoch)
        report_value = {
            "epoch": epoch,
            "baseline_strict_accuracy": baseline["metrics"]["strict_accuracy"],
            "exact_best_mask": f"{best_mask:05b}",
            "exact_best_root_ids": _coalition_spec(entries, best_mask)["root_ids"],
            "exact_best_net_corrected": values[best_mask],
            "exact_best_strict_accuracy": systems[best_mask]["metrics"][
                "strict_accuracy"],
            "all_candidate_mask": "11111",
            "all_candidate_net_corrected": values[31],
            "singleton_positive_union_mask": f"{singleton_positive_mask:05b}",
            "singleton_positive_union_net_corrected": values[
                singleton_positive_mask],
            "shapley_values": shapley,
            "pairwise_interactions": interaction_rows,
            "bootstrap_best_subset": bootstrap,
        }
        _write(epoch_dir / "report.json", report_value)
        epoch_reports[str(epoch)] = report_value

    final = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "claim_scope": (
            "Discovery100 cached-report mechanism audit only; no Rubric or "
            "checkpoint is selected after observing these results."),
        "k1_k3_stability": stability,
        "k3_local_predictor_correlations": local_predictor_correlations,
        "epochs": epoch_reports,
        "logical_k3_system_count": 26 + 52,
        "logical_k3_arbiter_request_count": (26 + 52) * len(rows) * K,
        "reused_k1_arbiter_request_count": 26 * len(rows),
        "new_arbiter_request_count": 26 * len(rows) * 2 + 52 * len(rows) * K,
        "new_subtree_request_count": 0,
        "selection_after_results_forbidden": True,
    }
    all_systems = [baseline]
    all_systems.extend(
        _load_singleton(target, entry)
        for entries in groups.values() for entry in entries)
    all_systems.extend(
        load_json(target / "coalitions" / f"epoch_{epoch:02d}"
                  / f"mask_{mask:05b}.json")
        for epoch in COALITION_EPOCHS for mask in range(1, 32)
        if mask.bit_count() > 1)
    final["usage"] = _sum_telemetry(all_systems)
    _write(target / "final_report.json", final)
    markdown = [
        "# Phase21 K=3 Counterfactual Coalition Audit", "",
        f"- Baseline K=3 Strict ACC: {baseline['metrics']['strict_accuracy']:.2%}",
        f"- K=1/K=3 singleton sign match: {stability['sign_match_rate']:.2%}",
        f"- K=1/K=3 singleton Spearman: {stability['k1_k3_spearman']}",
    ]
    for epoch in COALITION_EPOCHS:
        value = epoch_reports[str(epoch)]
        markdown.extend([
            "",
            f"## Epoch {epoch}", "",
            f"- Exact best mask: `{value['exact_best_mask']}`",
            f"- Exact best roots: {', '.join(value['exact_best_root_ids']) or 'none'}",
            f"- Exact best net corrected: {value['exact_best_net_corrected']}",
            f"- All-candidate net corrected: {value['all_candidate_net_corrected']}",
            ("- Positive-singleton union net corrected: "
             f"{value['singleton_positive_union_net_corrected']}"),
        ])
    markdown.extend(["", "This is an in-sample Discovery100 mechanism audit."])
    (target / "final_report.md").write_text("\n".join(markdown), encoding="utf-8")
    details = {
        "singleton_sign_match_rate": stability["sign_match_rate"],
        "singleton_spearman": stability["k1_k3_spearman"],
        "epoch_1_best_mask": epoch_reports["1"]["exact_best_mask"],
        "epoch_5_best_mask": epoch_reports["5"]["exact_best_mask"],
        "new_arbiter_request_count": final["new_arbiter_request_count"],
        "new_subtree_request_count": 0,
    }
    _status(target, STAGES[6], details)
    print(json.dumps(details, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze,
        STAGES[1]: audit,
        STAGES[2]: smoke,
        STAGES[3]: singletons,
        STAGES[4]: coalitions,
        STAGES[5]: retry,
        STAGES[6]: report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported counterfactual coalition stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

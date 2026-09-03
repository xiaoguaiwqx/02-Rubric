"""Frozen-candidate Dev150 K=3 counterfactual transfer audit.

The system list is frozen from the Discovery100 K=3 coalition audit.  Dev150
is used only for paired evaluation: no subset search, Rubric update, Manager
feedback, checkpoint choice, or early stopping is permitted.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
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
from . import counterfactual_coalition_audit as discovery_audit
from . import global_arbiter_ab_only as arbiter
from . import rejected_candidate_counterfactual_arbiter_audit as candidate_audit
from . import unified_subtree_bundle_evolution as phase21
from .experiment_utils import atomic_write_json, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase21_counterfactual_dev150_transfer_k3_v1"
PROTOCOL_VERSION = "phase21-counterfactual-dev150-transfer-k3-v1"
CONFIG_KEY = "phase21_counterfactual_dev150_transfer_k3_v1_experiment"
K = 3
STAGES = (
    "counterfactual-dev-transfer-freeze",
    "counterfactual-dev-transfer-audit",
    "counterfactual-dev-transfer-smoke",
    "counterfactual-dev-transfer-root-reports",
    "counterfactual-dev-transfer-run",
    "counterfactual-dev-transfer-retry",
    "counterfactual-dev-transfer-report",
)

# These masks are frozen from the completed Discovery100 audit.  Bit order is
# the source manifest's root_index order (Completeness through Clarity).
FROZEN_SYSTEMS = (
    {"system_id": "baseline", "epoch": None, "mask": 0,
     "discovery_role": "common_control"},
    {"system_id": "e1_completeness", "epoch": 1, "mask": 0b00001,
     "discovery_role": "exact_best_singleton"},
    {"system_id": "e1_cvf", "epoch": 1, "mask": 0b00111,
     "discovery_role": "tied_best_coalition"},
    {"system_id": "e1_positive_union", "epoch": 1, "mask": 0b10011,
     "discovery_role": "positive_singleton_union"},
    {"system_id": "e1_all", "epoch": 1, "mask": 0b11111,
     "discovery_role": "all_candidates"},
    {"system_id": "e5_completeness", "epoch": 5, "mask": 0b00001,
     "discovery_role": "tied_best_singleton"},
    {"system_id": "e5_clarity", "epoch": 5, "mask": 0b10000,
     "discovery_role": "tied_best_singleton"},
    {"system_id": "e5_positive_union", "epoch": 5, "mask": 0b10101,
     "discovery_role": "positive_singleton_union"},
    {"system_id": "e5_all", "epoch": 5, "mask": 0b11111,
     "discovery_role": "all_candidates"},
)

SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": phase21.EXPERIMENT_DIR,
    "source_candidate_audit": candidate_audit.EXPERIMENT_DIR,
    "source_discovery_k3_audit": discovery_audit.EXPERIMENT_DIR,
    "dataset_role": "dev150_evaluation_only",
    "sample_count": 150,
    "frozen_system_count": 9,
    "unique_candidate_root_count": 10,
    "k": 3,
    "reuse_phase21_dev_k1_baseline_as_replicate_zero": True,
    "ab_swap": False,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "bootstrap_seed": 42,
    "bootstrap_repetitions": 5000,
    "selection_after_dev_forbidden": True,
}


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _phase21_source(output: Path) -> Path:
    return output / phase21.EXPERIMENT_DIR


def _candidate_source(output: Path) -> Path:
    return output / candidate_audit.EXPERIMENT_DIR


def _discovery_source(output: Path) -> Path:
    return output / discovery_audit.EXPERIMENT_DIR


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get(CONFIG_KEY)
    if value != SETTINGS:
        raise RuntimeError(f"{CONFIG_KEY} drift")
    return dict(value)


def _rows(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(phase21._rows(config, "dev"))


def _runtime_settings(config: Mapping[str, Any]) -> runtime.RuntimeSettings:
    value = _settings(config)
    return runtime.RuntimeSettings(
        temperature=float(value["temperature"]),
        max_tokens=int(value["max_tokens"]),
        max_parse_retries=int(value["max_parse_retries"]),
        generation_seed_policy=str(value["generation_seed_policy"]),
    )


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {
        "schema_version": "1.0.0", "stages": {}}
    value.setdefault("stages", {})[stage] = {
        "status": "completed", "details": dict(details)}
    _write(path, value)


def _require(target: Path, stage: str) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {}
    if value.get("stages", {}).get(stage, {}).get("status") != "completed":
        raise RuntimeError(f"run {stage} first")


def _candidate_manifest(output: Path) -> dict[str, Any]:
    path = _candidate_source(output) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("complete the Phase21 K=1 candidate audit first")
    value = load_json(path)
    if value.get("protocol_version") != candidate_audit.PROTOCOL_VERSION:
        raise RuntimeError("candidate source protocol drift")
    return value


def _entries_by_epoch(manifest: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for value in manifest["candidates"]:
        epoch = int(value["epoch"])
        if epoch in {1, 5}:
            groups.setdefault(epoch, []).append(dict(value))
    for values in groups.values():
        values.sort(key=lambda item: int(item["root_index"]))
    return groups


def _root_order(output: Path) -> tuple[str, ...]:
    rubric = StructuredRubric.load_json(
        _phase21_source(output) / "epochs" / "epoch_00" / "rubric_committed.json")
    return tuple(rubric.root_ids)


def _baseline_dev_path(output: Path) -> Path:
    return (_phase21_source(output) / "epochs" / "epoch_00" / "dev150"
            / "system.json")


def _discovery_system_path(output: Path, epoch: int | None, mask: int) -> Path:
    target = _discovery_source(output)
    if mask == 0:
        return target / "singletons" / "baseline.json"
    if mask.bit_count() == 1:
        entries = _entries_by_epoch(_candidate_manifest(output))[int(epoch)]
        index = next(index for index in range(5) if mask & (1 << index))
        return target / "singletons" / f"{entries[index]['candidate_id']}.json"
    return (target / "coalitions" / f"epoch_{int(epoch):02d}"
            / f"mask_{mask:05b}.json")


def _frozen_specs(output: Path) -> list[dict[str, Any]]:
    groups = _entries_by_epoch(_candidate_manifest(output))
    specs = []
    for frozen in FROZEN_SYSTEMS:
        value = dict(frozen)
        epoch, mask = value["epoch"], int(value["mask"])
        selected = [] if not mask else [
            groups[int(epoch)][index] for index in range(5)
            if mask & (1 << index)]
        discovery_path = _discovery_system_path(output, epoch, mask)
        if not discovery_path.is_file():
            raise RuntimeError(f"missing frozen Discovery system: {discovery_path}")
        discovery = load_json(discovery_path)
        value.update({
            "mask_bits": f"{mask:05b}",
            "root_ids": [str(item["root_id"]) for item in selected],
            "candidate_ids": [str(item["candidate_id"]) for item in selected],
            "discovery_source_path": str(discovery_path.resolve()),
            "discovery_source_sha256": file_sha256(discovery_path),
            "discovery_metrics": {
                key: discovery["metrics"][key] for key in (
                    "strict_accuracy", "coverage", "covered_accuracy", "none_rate")},
        })
        specs.append(value)
    return specs


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    settings = _settings(config)
    rows = _rows(config)
    source = _candidate_manifest(output)
    groups = _entries_by_epoch(source)
    if any(len(groups.get(epoch, ())) != 5 for epoch in (1, 5)):
        raise RuntimeError("epochs 1 and 5 must each contain five candidates")
    baseline_path = _baseline_dev_path(output)
    phase21_manifest_path = _phase21_source(output) / "frozen_manifest.json"
    discovery_report_path = _discovery_source(output) / "final_report.json"
    if not all(path.is_file() for path in (
            baseline_path, phase21_manifest_path, discovery_report_path)):
        raise RuntimeError("Phase21 and Discovery K=3 source artifacts are incomplete")
    unique_entries = groups[1] + groups[5]
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "settings": settings,
        "dataset": {
            "role": "dev150_evaluation_only",
            "count": len(rows),
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "sha256": canonical_sha256(rows),
        },
        "source": {
            "phase21_manifest_path": str(phase21_manifest_path.resolve()),
            "phase21_manifest_sha256": file_sha256(phase21_manifest_path),
            "baseline_dev_path": str(baseline_path.resolve()),
            "baseline_dev_sha256": file_sha256(baseline_path),
            "candidate_manifest_path": str(
                (_candidate_source(output) / "frozen_manifest.json").resolve()),
            "candidate_manifest_sha256": file_sha256(
                _candidate_source(output) / "frozen_manifest.json"),
            "discovery_k3_report_path": str(discovery_report_path.resolve()),
            "discovery_k3_report_sha256": file_sha256(discovery_report_path),
            "worker_endpoint_identities": source["source"][
                "worker_endpoint_identities"],
            "unified_runtime_identity": source["source"][
                "unified_runtime_identity"],
        },
        "candidate_roots": [{
            **{key: entry[key] for key in (
                "candidate_id", "epoch", "root_index", "root_id",
                "candidate_root_path", "candidate_rubric_path")},
            "candidate_rubric_sha256": entry["source_hashes"]["candidate_rubric"],
            "candidate_root_sha256": entry["source_hashes"]["candidate_root"],
        } for entry in unique_entries],
        "systems": _frozen_specs(output),
        "planned_new_root_report_requests": 10 * len(rows),
        "logical_k3_arbiter_requests": len(FROZEN_SYSTEMS) * len(rows) * K,
        "reused_baseline_replicate_zero_requests": len(rows),
        "planned_new_arbiter_requests": (
            len(FROZEN_SYSTEMS) * len(rows) * K - len(rows)),
        "selection_after_dev_forbidden": True,
    }
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("Dev150 transfer manifest drift")
        print(f"{STAGES[0]} already completed")
        return
    _write(path, manifest)
    details = {
        "sample_count": len(rows), "system_count": len(FROZEN_SYSTEMS),
        "unique_candidate_roots": len(unique_entries),
        "planned_new_root_reports": manifest["planned_new_root_report_requests"],
        "planned_new_arbiter_requests": manifest["planned_new_arbiter_requests"],
    }
    _status(target, STAGES[0], details)
    print(json.dumps(details, indent=2))


def _manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _settings(config)
    path = _target(output) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"run {STAGES[0]} first")
    value = load_json(path)
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Dev150 transfer protocol drift")
    return value


def _baseline_index(output: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    value = runtime.load(_baseline_dev_path(output))
    return ({str(item["sample_id"]): item for item in value["samples"]}, value)


def audit(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    manifest = _manifest(config, output)
    rows = _rows(config)
    baseline_by_id, baseline = _baseline_index(output)
    current_candidates = _candidate_manifest(output)
    groups = _entries_by_epoch(current_candidates)
    sample_ids = [str(row["sample_id"]) for row in rows]
    checks = {
        "dev_count_150": len(rows) == 150,
        "dev_order_frozen": sample_ids == manifest["dataset"]["sample_ids"],
        "dev_hash_frozen": canonical_sha256(rows) == manifest["dataset"]["sha256"],
        "baseline_k1": int(baseline.get("k", 0)) == 1,
        "baseline_sample_identity": list(baseline_by_id) == sample_ids,
        "baseline_zero_technical_failures": (
            baseline["metrics"]["technical_failure_count"] == 0),
        "exactly_nine_frozen_systems": len(manifest["systems"]) == 9,
        "system_ids_unique": len({item["system_id"] for item in manifest["systems"]}) == 9,
        "exactly_ten_candidate_roots": len(manifest["candidate_roots"]) == 10,
        "epoch_1_five_roots": len(groups.get(1, ())) == 5,
        "epoch_5_five_roots": len(groups.get(5, ())) == 5,
        "selection_after_dev_forbidden": manifest["selection_after_dev_forbidden"],
    }
    root_order = _root_order(output)
    checks["candidate_root_order_complete"] = all(
        tuple(str(item["root_id"]) for item in groups[epoch]) == root_order
        for epoch in (1, 5))
    source_root = _phase21_source(output)
    # Discovery counterfactual bundles use each epoch's baseline_root_paths for
    # unmodified roots.  Reusing the epoch-0 Dev reports is valid only while
    # those exact subtree identities are equal, not merely because their root
    # IDs match.
    baseline_hashes = dict(baseline["root_subtree_sha256"])
    for epoch in (1, 5):
        paths = groups[epoch][0]["baseline_root_paths"]
        checks[f"epoch_{epoch}_baseline_roots_equal_epoch0"] = all(
            runtime.load(source_root / str(paths[root_id]))[
                "root_subtree_sha256"] == baseline_hashes[root_id]
            for root_id in root_order)
    for entry in manifest["candidate_roots"]:
        rubric_path = source_root / str(entry["candidate_rubric_path"])
        root_path = source_root / str(entry["candidate_root_path"])
        checks[f"rubric_{entry['candidate_id']}"] = (
            rubric_path.is_file()
            and file_sha256(rubric_path) == entry["candidate_rubric_sha256"])
        checks[f"root_{entry['candidate_id']}"] = root_path.is_file()
        checks[f"root_hash_{entry['candidate_id']}"] = (
            root_path.is_file()
            and file_sha256(root_path) == entry["candidate_root_sha256"])
    baseline_calls_valid = True
    for item in baseline["samples"]:
        replicate = item["replicates"]["0"]
        baseline_calls_valid = baseline_calls_valid and all(
            call.get("parse_ok") for call in replicate["subtrees"].values())
        baseline_calls_valid = baseline_calls_valid and replicate["arbiter"].get("parse_ok")
    checks["baseline_all_calls_parse_valid"] = baseline_calls_valid
    if not all(checks.values()):
        raise RuntimeError(f"Dev150 transfer offline audit failed: {checks}")
    value = {"schema_version": "1.0.0", "offline_only": True, "checks": checks}
    _write(target / "offline_audit.json", value)
    _status(target, STAGES[1], checks)
    print(json.dumps(checks, indent=2))


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if candidate_audit._endpoint_identities_with_retry(config) != manifest["source"][
            "worker_endpoint_identities"]:
        raise RuntimeError("Dev150 transfer endpoint identity drift")
    if phase21._unified_runtime_identity(config) != manifest["source"][
            "unified_runtime_identity"]:
        raise RuntimeError("Dev150 transfer prompt/parser/model drift")


def _candidate_entry(manifest: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    return next(dict(item) for item in manifest["candidate_roots"]
                if item["candidate_id"] == candidate_id)


def _evaluate_candidate_root(
    config: Mapping[str, Any], output: Path, manifest: Mapping[str, Any],
    candidate_id: str, rows: Sequence[Mapping[str, Any]], output_path: Path,
    *, total_attempt_limit: int,
) -> dict[str, Any]:
    if output_path.is_file():
        value = runtime.load(output_path)
        if ([str(item["sample_id"]) for item in value["samples"]]
                == [str(row["sample_id"]) for row in rows]
                and value["metrics"]["technical_failure_count"] == 0):
            return value
    entry = _candidate_entry(manifest, candidate_id)
    rubric_path = _phase21_source(output) / str(entry["candidate_rubric_path"])
    rubric = StructuredRubric.load_json(rubric_path)
    return runtime.evaluate_root(
        config, output_path=output_path, cache_dir=_target(output) / "cache" / "roots",
        split_name=f"dev150_transfer_root_{candidate_id}", rows=rows, rubric=rubric,
        root_id=str(entry["root_id"]),
        scope_sample_ids=[str(row["sample_id"]) for row in rows],
        settings=_runtime_settings(config), total_attempt_limit=total_attempt_limit)


def _root_report_indexes(
    output: Path, manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    *, smoke: bool,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    directory = "smoke_candidate_roots" if smoke else "candidate_roots"
    indexes = {}
    for entry in manifest["candidate_roots"]:
        path = _target(output) / directory / f"{entry['candidate_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"run candidate root materialization first: {path}")
        value = runtime.load(path)
        indexes[str(entry["candidate_id"])] = {
            str(item["sample_id"]): item for item in value["samples"]}
    return indexes


def _bundle_index(
    output: Path, manifest: Mapping[str, Any], spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], *, smoke: bool,
) -> dict[str, tuple[list[dict[str, Any]], str]]:
    baseline_by_id, _ = _baseline_index(output)
    candidate_indexes = _root_report_indexes(output, manifest, rows, smoke=smoke)
    root_order = _root_order(output)
    baseline_rubric = StructuredRubric.load_json(
        _phase21_source(output) / "epochs" / "epoch_00" / "rubric_committed.json")
    replacement = dict(zip(spec["root_ids"], spec["candidate_ids"]))
    bundles = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        reports = []
        for root_id in root_order:
            if root_id in replacement:
                call = candidate_indexes[replacement[root_id]][sample_id]["call"]
            else:
                call = baseline_by_id[sample_id]["replicates"]["0"]["subtrees"][root_id]
            if not call.get("parse_ok"):
                raise RuntimeError(f"unresolved root report: {sample_id}::{root_id}")
            reports.append({
                "root_id": root_id,
                "criterion_name": baseline_rubric.get_node(root_id).criterion.name,
                "report": call["parsed"],
            })
        bundles[sample_id] = (reports, canonical_sha256(reports))
    return bundles


def _baseline_replicate_zero(output: Path) -> dict[str, dict[str, Any]]:
    by_id, _ = _baseline_index(output)
    return {sample_id: {
        "source_report_bundle_sha256": item["replicates"]["0"][
            "report_bundle_sha256"],
        "arbiter": support.compact_call(item["replicates"]["0"]["arbiter"]),
    } for sample_id, item in by_id.items()}


def _arbiter_call(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, output: Path,
    system_id: str, row: Mapping[str, Any], reports: Sequence[Mapping[str, Any]],
    digest: str, replicate: int, total_attempt_limit: int,
) -> dict[str, Any]:
    call = support.call_one(
        config, endpoint, _target(output) / "cache" / "arbiter",
        user_text=support.global_arbiter_user_prompt(row, reports), row=row,
        request_key={
            "kind": "phase21_counterfactual_dev150_transfer_arbiter",
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
        settings_loader=_runtime_settings(config).as_loader(),
    )
    return support.compact_call(call)


def _majority(votes: Sequence[str]) -> str:
    return discovery_audit._majority(votes)


def _system_metrics(predictions: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return discovery_audit._system_metrics(predictions, rows)


def _run_system(
    config: Mapping[str, Any], output: Path, manifest: Mapping[str, Any],
    spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], path: Path,
    *, smoke: bool, total_attempt_limit: int,
) -> dict[str, Any]:
    if path.is_file():
        old = load_json(path)
        if (old.get("technical_failure_call_count") == 0 and old.get("k") == K
                and [item["sample_id"] for item in old["samples"]]
                == [str(row["sample_id"]) for row in rows]):
            return old
    bundles = _bundle_index(output, manifest, spec, rows, smoke=smoke)
    values = {str(row["sample_id"]): {"replicates": {}} for row in rows}
    if spec["system_id"] == "baseline":
        reused = _baseline_replicate_zero(output)
        for row in rows:
            sample_id = str(row["sample_id"])
            if reused[sample_id]["source_report_bundle_sha256"] != bundles[sample_id][1]:
                raise RuntimeError(f"baseline replicate-zero bundle drift: {sample_id}")
            values[sample_id]["replicates"]["0"] = reused[sample_id]["arbiter"]
    tasks = [(row, replicate) for row in rows for replicate in range(K)
             if str(replicate) not in values[str(row["sample_id"])]["replicates"]]
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoints = tuple(item for item in pool.endpoints
                      if item.endpoint_id in runtime.ENDPOINT_IDS)
    if tuple(item.endpoint_id for item in endpoints) != runtime.ENDPOINT_IDS:
        raise RuntimeError("Dev150 transfer requires vllm-8000 and vllm-8001")
    pending: queue.Queue[tuple[Mapping[str, Any], int]] = queue.Queue()
    for task in tasks:
        pending.put(task)
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()
    print(f"dev_transfer_{spec['system_id']}: 0/{len(tasks)} Arbiter calls started", flush=True)

    def worker(endpoint: BackendEndpointSpec) -> None:
        nonlocal completed
        while True:
            try:
                row, replicate = pending.get_nowait()
            except queue.Empty:
                return
            sample_id = str(row["sample_id"])
            reports, digest = bundles[sample_id]
            call = _arbiter_call(
                config, endpoint, output, str(spec["system_id"]), row, reports,
                digest, replicate, total_attempt_limit)
            with lock:
                values[sample_id]["replicates"][str(replicate)] = call
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(tasks) - completed)
                print(
                    f"dev_transfer_{spec['system_id']}: {completed}/{len(tasks)} "
                    f"sample={sample_id} replicate={replicate} "
                    f"endpoint={endpoint.endpoint_id} elapsed={elapsed/60:.1f}m "
                    f"ETA={eta/60:.1f}m", flush=True)

    workers = runtime._interleaved_workers(endpoints)
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    samples, technical = [], 0
    for row in rows:
        sample_id = str(row["sample_id"])
        replicates = values[sample_id]["replicates"]
        votes = [runtime._answer(replicates[str(index)], 0) for index in range(K)]
        technical += votes.count("technical_failure")
        samples.append({
            "sample_id": sample_id,
            "source_report_bundle_sha256": bundles[sample_id][1],
            "replicates": replicates,
            "votes": votes,
            "majority_answer": _majority(votes),
        })
    result = {
        "schema_version": "1.0.0", "protocol_version": PROTOCOL_VERSION,
        "system_id": spec["system_id"], "epoch": spec["epoch"],
        "mask": spec["mask"], "mask_bits": spec["mask_bits"],
        "root_ids": spec["root_ids"], "candidate_ids": spec["candidate_ids"],
        "k": K,
        "reused_baseline_replicate_zero_count": len(rows) if spec["system_id"] == "baseline" else 0,
        "new_arbiter_request_count": len(tasks),
        "technical_failure_call_count": technical,
        "samples": samples,
        "metrics": _system_metrics([item["majority_answer"] for item in samples], rows),
        "wall_seconds": time.perf_counter() - started,
    }
    _write(path, result)
    return result


def _smoke_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("source", "unknown")), []).append(row)
    selected = [values[0] for _, values in sorted(groups.items())]
    used = {str(row["sample_id"]) for row in selected}
    selected.extend(row for row in rows if str(row["sample_id"]) not in used)
    return tuple(selected[:12])


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require(target, STAGES[1])
    manifest = _manifest(config, output)
    _verify_live(config, manifest)
    rows = _smoke_rows(_rows(config))
    specs = [manifest["systems"][0], manifest["systems"][1], manifest["systems"][-1]]
    candidate_ids = sorted({candidate_id for spec in specs
                            for candidate_id in spec["candidate_ids"]})
    for candidate_id in candidate_ids:
        _evaluate_candidate_root(
            config, output, manifest, candidate_id, rows,
            target / "smoke_candidate_roots" / f"{candidate_id}.json",
            total_attempt_limit=1 + SETTINGS["max_parse_retries"])
    # Bundle construction expects all ten indexes.  Materialize the unused
    # candidate roots on the smoke rows too; this also validates every source.
    for entry in manifest["candidate_roots"]:
        if entry["candidate_id"] in candidate_ids:
            continue
        _evaluate_candidate_root(
            config, output, manifest, entry["candidate_id"], rows,
            target / "smoke_candidate_roots" / f"{entry['candidate_id']}.json",
            total_attempt_limit=1 + SETTINGS["max_parse_retries"])
    outputs = [_run_system(
        config, output, manifest, spec, rows,
        target / "smoke" / f"{spec['system_id']}.json", smoke=True,
        total_attempt_limit=1 + SETTINGS["max_parse_retries"])
        for spec in specs]
    technical = sum(item["technical_failure_call_count"] for item in outputs)
    value = {
        "status": "passed" if technical == 0 else "failed",
        "sample_count": len(rows), "system_ids": [item["system_id"] for item in outputs],
        "candidate_root_count": len(manifest["candidate_roots"]),
        "technical_failure_call_count": technical,
    }
    _write(target / "smoke" / "report.json", value)
    if technical:
        raise RuntimeError(f"Dev150 transfer smoke has {technical} failures")
    _status(target, STAGES[2], value)
    print(json.dumps(value, indent=2))


def root_reports(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require(target, STAGES[2])
    manifest = _manifest(config, output)
    _verify_live(config, manifest)
    rows = _rows(config)
    values = [_evaluate_candidate_root(
        config, output, manifest, str(entry["candidate_id"]), rows,
        target / "candidate_roots" / f"{entry['candidate_id']}.json",
        total_attempt_limit=1 + SETTINGS["max_parse_retries"])
        for entry in manifest["candidate_roots"]]
    technical = sum(item["metrics"]["technical_failure_count"] for item in values)
    details = {
        "candidate_root_count": len(values),
        "logical_root_report_requests": len(values) * len(rows),
        "technical_failure_count": technical,
    }
    if technical:
        raise RuntimeError(f"candidate Dev root reports have {technical} failures")
    _status(target, STAGES[3], details)
    print(json.dumps(details, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require(target, STAGES[3])
    manifest = _manifest(config, output)
    _verify_live(config, manifest)
    rows = _rows(config)
    values = [_run_system(
        config, output, manifest, spec, rows,
        target / "systems" / f"{spec['system_id']}.json", smoke=False,
        total_attempt_limit=1 + SETTINGS["max_parse_retries"])
        for spec in manifest["systems"]]
    details = {
        "system_count": len(values),
        "logical_k3_arbiter_requests": len(values) * len(rows) * K,
        "reused_baseline_replicate_zero_requests": len(rows),
        "new_arbiter_requests": sum(item["new_arbiter_request_count"] for item in values),
        "technical_failure_call_count": sum(
            item["technical_failure_call_count"] for item in values),
    }
    _status(target, STAGES[4], details)
    print(json.dumps(details, indent=2))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require(target, STAGES[4])
    manifest = _manifest(config, output)
    _verify_live(config, manifest)
    rows = _rows(config)
    values = [_run_system(
        config, output, manifest, spec, rows,
        target / "systems" / f"{spec['system_id']}.json", smoke=False,
        total_attempt_limit=1 + 2 * SETTINGS["max_parse_retries"])
        for spec in manifest["systems"]]
    details = {
        "system_count": len(values),
        "unresolved_technical_failure_calls": sum(
            item["technical_failure_call_count"] for item in values),
        "target_total_attempt_limit": 21,
    }
    _status(target, STAGES[5], details)
    print(json.dumps(details, indent=2))


def _paired_ci(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], *, seed: int,
) -> list[float]:
    gold = [str(row["answer"]) for row in rows]
    before = baseline["metrics"]["predictions"]
    after = candidate["metrics"]["predictions"]
    deltas = [int(right == target) - int(left == target)
              for left, right, target in zip(before, after, gold)]
    rng = random.Random(seed)
    values = sorted(sum(deltas[rng.randrange(len(deltas))]
                        for _ in deltas) / len(deltas)
                    for _ in range(SETTINGS["bootstrap_repetitions"]))
    count = len(values)
    return [values[int(0.025 * count)], values[min(count - 1, int(0.975 * count))]]


def _mcnemar(corrected: int, harmed: int) -> float:
    discordant = corrected + harmed
    if discordant == 0:
        return 1.0
    smaller = min(corrected, harmed)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _group_metrics(
    value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], field: str,
) -> dict[str, Any]:
    predictions = value["metrics"]["predictions"]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row.get(field, "unknown")), []).append(index)
    result = {}
    for name, indices in sorted(groups.items()):
        decisive = [predictions[index] in {"A", "B"} for index in indices]
        correct = [predictions[index] == str(rows[index]["answer"]) for index in indices]
        result[name] = {
            "sample_count": len(indices),
            "strict_accuracy": sum(correct) / len(indices),
            "coverage": sum(decisive) / len(indices),
        }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    _require(target, STAGES[4])
    manifest = _manifest(config, output)
    rows = _rows(config)
    systems = {spec["system_id"]: load_json(
        target / "systems" / f"{spec['system_id']}.json")
        for spec in manifest["systems"]}
    if any(value["technical_failure_call_count"] for value in systems.values()):
        raise RuntimeError("Dev150 transfer report requires zero technical failures; run retry")
    baseline = systems["baseline"]
    records = []
    details = {}
    for index, spec in enumerate(manifest["systems"]):
        value = systems[spec["system_id"]]
        if spec["system_id"] == "baseline":
            paired = {"corrected": 0, "harmed": 0, "net_corrected": 0,
                      "strict_accuracy_delta": 0.0}
        else:
            paired = runtime.paired(baseline["metrics"], value["metrics"], rows)
        discovery_delta = (spec["discovery_metrics"]["strict_accuracy"]
                           - manifest["systems"][0]["discovery_metrics"]["strict_accuracy"])
        dev_delta = paired["strict_accuracy_delta"]
        comparison = {
            **paired,
            "paired_bootstrap_ci95": [0.0, 0.0] if spec["system_id"] == "baseline" else _paired_ci(
                baseline, value, rows, seed=SETTINGS["bootstrap_seed"] + index),
            "mcnemar_exact_two_sided_p": 1.0 if spec["system_id"] == "baseline" else _mcnemar(
                paired["corrected"], paired["harmed"]),
        }
        details[spec["system_id"]] = {
            "spec": spec,
            "dev_metrics": {key: value["metrics"][key] for key in (
                "strict_accuracy", "coverage", "covered_accuracy", "none_rate",
                "prediction_distribution")},
            "paired_vs_dev_baseline": comparison,
            "discovery_strict_accuracy_delta": discovery_delta,
            "dev_strict_accuracy_delta": dev_delta,
            "gain_sign_transfers": ((discovery_delta > 0) == (dev_delta > 0))
            if discovery_delta != 0 and dev_delta != 0 else discovery_delta == dev_delta,
            "by_source": _group_metrics(value, rows, "source"),
            "by_domain": _group_metrics(value, rows, "domain"),
        }
        records.append({
            "system_id": spec["system_id"], "epoch": spec["epoch"],
            "mask": spec["mask_bits"], "discovery_role": spec["discovery_role"],
            "discovery_strict_accuracy": spec["discovery_metrics"]["strict_accuracy"],
            "discovery_delta": discovery_delta,
            "dev_strict_accuracy": value["metrics"]["strict_accuracy"],
            "dev_coverage": value["metrics"]["coverage"],
            "dev_none_rate": value["metrics"]["none_rate"],
            "dev_delta": dev_delta,
            "corrected": paired["corrected"], "harmed": paired["harmed"],
            "net_corrected": paired["net_corrected"],
            "sign_transfers": details[spec["system_id"]]["gain_sign_transfers"],
        })
    treatment_records = records[1:]
    discovery_deltas = [float(item["discovery_delta"]) for item in treatment_records]
    dev_deltas = [float(item["dev_delta"]) for item in treatment_records]
    correlation = {
        "system_count": len(treatment_records),
        "pearson": candidate_audit._pearson(discovery_deltas, dev_deltas),
        "spearman": candidate_audit._spearman(discovery_deltas, dev_deltas),
        "gain_sign_transfer_count": sum(bool(item["sign_transfers"])
                                        for item in treatment_records),
        "gain_sign_transfer_rate": sum(bool(item["sign_transfers"])
                                       for item in treatment_records) / len(treatment_records),
        "descriptive_only_small_n": True,
    }
    _write_csv(target / "dev150_transfer_results.csv", records)
    final = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "claim_scope": (
            "Frozen Discovery100 systems evaluated on independent Dev150; "
            "no Dev subset search or checkpoint selection."),
        "systems": details,
        "discovery_to_dev_correlation": correlation,
        "logical_k3_system_count": len(systems),
        "logical_k3_arbiter_request_count": len(systems) * len(rows) * K,
        "reused_baseline_replicate_zero_count": len(rows),
        "new_candidate_root_request_count": 10 * len(rows),
        "new_arbiter_request_count": sum(
            value["new_arbiter_request_count"] for value in systems.values()),
        "selection_after_dev_forbidden": True,
    }
    _write(target / "final_report.json", final)
    markdown = [
        "# Frozen Candidate Dev150 K=3 Transfer Audit", "",
        "Dev150 is evaluation-only. The nine systems were frozen before this run.", "",
        "| System | Discovery Strict | Dev Strict | Dev Coverage | Dev net vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in records:
        markdown.append(
            f"| {item['system_id']} | {item['discovery_strict_accuracy']:.2%} | "
            f"{item['dev_strict_accuracy']:.2%} | {item['dev_coverage']:.2%} | "
            f"{item['net_corrected']:+d} |")
    markdown.extend([
        "", f"- Gain-sign transfer: {correlation['gain_sign_transfer_count']}/8",
        f"- Discovery/Dev delta Spearman (descriptive, n=8): {correlation['spearman']}",
        "- No Dev150 result was used to select a Rubric, checkpoint, or coalition.",
    ])
    (target / "final_report.md").write_text("\n".join(markdown), encoding="utf-8")
    summary = {
        "baseline_dev_strict_accuracy": baseline["metrics"]["strict_accuracy"],
        "best_observed_dev_system": max(records, key=lambda item: item["dev_strict_accuracy"])["system_id"],
        "best_observed_dev_strict_accuracy": max(item["dev_strict_accuracy"] for item in records),
        "gain_sign_transfer_rate": correlation["gain_sign_transfer_rate"],
        "selection_after_dev_forbidden": True,
    }
    _status(target, STAGES[6], summary)
    print(json.dumps(summary, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze, STAGES[1]: audit, STAGES[2]: smoke,
        STAGES[3]: root_reports, STAGES[4]: run, STAGES[5]: retry,
        STAGES[6]: report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported Dev150 transfer stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

"""Qwen2.5-VL transfer of the frozen Clean S5-v2 aggregation protocol.

The experiment keeps the Phase17 epoch-4 rubric, Unified-Subtree prompt,
Global-Arbiter prompt, parsers, decoding parameters, scheduling policy, and
evaluation data fixed.  Only the worker checkpoint changes from Qwen3-VL-8B
to Qwen2.5-VL-7B-Instruct.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from critiq.structured import StructuredRubric
from critiq.structured.backend_pool import BackendEndpointSpec

from . import _global_arbiter_ab_only_support as support
from . import full_rubric_unified_worker as shared
from . import global_arbiter_ab_only as arbiter
from . import internal_global_arbiter_k1 as internal
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_phase10 as vlrb_metrics
from . import vl_rewardbench_qwen25_transfer as qwen25
from .experiment_utils import atomic_write_json, canonical_sha256, load_json
from .rubric_factory import file_sha256


SCHEMA_VERSION = "1.0.0"
PROTOCOL_VERSION = "qwen25-clean-s5-v2-transfer-v1"
CONFIG_KEY = "qwen25_clean_s5_transfer_experiment"
EXPERIMENT_DIR = "qwen25_clean_s5_v2_transfer_v1"
SYSTEM_NAME = "qwen25_clean_s5_v2"
WORKER_MODEL = qwen25.WORKER_MODEL
SOURCE_RUBRIC_SHA256 = shared.SOURCE_RUBRIC_SHA256
INTERNAL_COUNTS = dict(internal.SPLIT_COUNTS)
VLRB_COUNT = support.VLRB_COUNT
VLRB_K = support.K
ENDPOINT_IDS = support.ENDPOINT_IDS

QWEN3_CLEAN_EXPERIMENT = shared.CLEAN_S5_EXPERIMENT
QWEN3_CLEAN_SYSTEM = shared.CLEAN_S5_SYSTEM
QWEN25_EXPLICIT_EXPERIMENT = qwen25.EXPERIMENT_DIR
QWEN25_EXPLICIT_SYSTEM = qwen25.E4_SYSTEM

STAGES = (
    "qwen25-clean-s5-freeze",
    "qwen25-clean-s5-audit",
    "qwen25-clean-s5-smoke",
    "qwen25-clean-s5-internal-run",
    "qwen25-clean-s5-vlrb-run",
    "qwen25-clean-s5-retry",
    "qwen25-clean-s5-report",
)


def _target(output: Path) -> Path:
    return output.parent / EXPERIMENT_DIR


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "worker_model": WORKER_MODEL,
        "source_experiment": shared.SOURCE_EXPERIMENT,
        "source_epoch": shared.SOURCE_EPOCH,
        "rubric_sha256": SOURCE_RUBRIC_SHA256,
        "internal_datasets": dict(INTERNAL_COUNTS),
        "vl_rewardbench_count": VLRB_COUNT,
        "internal_k": 1,
        "vl_rewardbench_k": VLRB_K,
        "internal_ab_swap": False,
        "generation_seed_policy": "unset",
        "temperature": 0.5,
        "max_tokens": 2048,
        "max_parse_retries": 10,
        "smoke_internal_samples_per_split": 2,
        "smoke_vlrb_sample_count": 20,
        "endpoint_ids": list(ENDPOINT_IDS),
        "scheduler": "sample_bundle_available_slot_affinity",
        "semantic_answer_space": ["A", "B", "None"],
        "subtree_calls_per_replicate": 5,
        "arbiter_calls_per_replicate": 1,
        "same_prompt_retry_only": True,
        "selection_after_diagnostics_forbidden": True,
    }
    value = config.get(CONFIG_KEY)
    if value != expected:
        raise RuntimeError(f"{CONFIG_KEY} does not match the frozen protocol")
    return dict(value)


def _worker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = qwen25._worker_config(config)
    _settings(value)
    return value


def _pool(config: Mapping[str, Any]):
    return qwen25._pool_spec(_worker_config(config))


def _rubric(output: Path) -> StructuredRubric:
    rubric = shared._rubric(output)
    if rubric.rubric_sha256 != SOURCE_RUBRIC_SHA256:
        raise RuntimeError("Qwen2.5 Clean S5 source rubric drift")
    return rubric


def _internal_rows(config: Mapping[str, Any], split: str):
    return shared._internal_rows(config, split)


def _vlrb_records(output: Path):
    return shared._vlrb_records(output)


def _source_paths(output: Path) -> dict[str, Path]:
    return {
        "qwen3_clean_s5": output.parent / QWEN3_CLEAN_EXPERIMENT / "final_report.json",
        "qwen3_clean_s5_manifest": output.parent / QWEN3_CLEAN_EXPERIMENT
            / "frozen_manifest.json",
        "qwen3_unified_subtree_manifest": output.parent
            / support.SOURCE_S3_EXPERIMENT / "frozen_manifest.json",
        "qwen25_explicit_e4": output.parent / QWEN25_EXPLICIT_EXPERIMENT / "final_report.json",
        "qwen3_internal_clean_s5": output.parent / internal.EXPERIMENT_DIR / "final_report.json",
        "vlrb_schedule": output.parent / support.SOURCE_SCHEDULE_EXPERIMENT / "order_schedule.json",
    }


def _qwen25_explicit_report(output: Path) -> Mapping[str, Any]:
    path = _source_paths(output)["qwen25_explicit_e4"]
    report = load_json(path)
    metrics = report.get("metrics", {}).get(QWEN25_EXPLICIT_SYSTEM)
    if not isinstance(metrics, dict):
        raise RuntimeError("Qwen2.5 explicit E4 control is missing")
    predictions = metrics.get("original_index_predictions")
    if not isinstance(predictions, list) or len(predictions) != VLRB_COUNT:
        raise RuntimeError("Qwen2.5 explicit E4 control predictions are incomplete")
    return report


def _manifest(config: Mapping[str, Any], output: Path, *, include_endpoints: bool) -> dict[str, Any]:
    rubric = _rubric(output)
    records = _vlrb_records(output)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in records])
    paths = _source_paths(output)
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"required frozen control is missing: {path}")
    _qwen25_explicit_report(output)
    datasets = {
        split: {
            "count": INTERNAL_COUNTS[split],
            "path": str(shared._internal_dataset_path(config, split).resolve()),
            "sha256": file_sha256(shared._internal_dataset_path(config, split)),
            "semantic_sha256": canonical_sha256([{
                key: row.get(key)
                for key in ("sample_id", "question", "A", "B", "answer")
            } for row in _internal_rows(config, split)]),
        }
        for split in INTERNAL_COUNTS
    }
    datasets["vl_rewardbench"] = {
        "count": len(records),
        "semantic_sha256": canonical_sha256([{
            key: record[key]
            for key in ("sample_id", "benchmark_id", "question", "responses",
                        "preferred_original_index", "image_sha256", "group")
        } for record in records]),
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "settings": _settings(config),
        "settings_sha256": canonical_sha256(_settings(config)),
        "worker_model": WORKER_MODEL,
        "rubric_sha256": rubric.rubric_sha256,
        "root_ids": list(rubric.root_ids),
        "node_count": len(rubric.nodes),
        "datasets": datasets,
        "subtree_prompt_version": internal.SUBTREE_PROMPT_VERSION,
        "subtree_system_prompt_sha256": hashlib.sha256(
            internal.UNIFIED_SUBTREE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "arbiter_prompt_version": internal.ARBITER_PROMPT_VERSION,
        "arbiter_system_prompt_sha256": hashlib.sha256(
            arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "vlrb_schedule_sha256": canonical_sha256(schedule),
        "source_artifacts": {name: file_sha256(path) for name, path in paths.items()},
        "logical_request_budget": {
            "internal": sum(INTERNAL_COUNTS.values()) * 6,
            "vl_rewardbench": len(records) * VLRB_K * 6,
            "total": sum(INTERNAL_COUNTS.values()) * 6 + len(records) * VLRB_K * 6,
        },
        "single_changed_variable": "worker_model",
    }
    if include_endpoints:
        value["endpoint_identities"] = qwen25.phase10._inspect_endpoints(
            _worker_config(config))
    return value


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    support.status(target, stage, details)


def _require(target: Path, stage: str) -> None:
    support.require(target, stage)


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    manifest = _manifest(config, output, include_endpoints=True)
    path = target / "frozen_manifest.json"
    if path.is_file() and load_json(path) != manifest:
        raise RuntimeError("Qwen2.5 Clean S5 frozen manifest drift")
    atomic_write_json(path, manifest)
    _rubric(output).save_json(target / "rubric_snapshot.json")
    atomic_write_json(target / "prompt_spec.json", {
        "schema_version": SCHEMA_VERSION,
        "subtree_prompt_version": internal.SUBTREE_PROMPT_VERSION,
        "subtree_system_prompt": internal.UNIFIED_SUBTREE_SYSTEM_PROMPT,
        "arbiter_prompt_version": internal.ARBITER_PROMPT_VERSION,
        "arbiter_system_prompt": arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        "subtree_user_prompt_builder": "internal_global_arbiter_k1.subtree_user_prompt",
        "arbiter_user_prompt_builder": "_global_arbiter_ab_only_support.global_arbiter_user_prompt",
        "parsers": {"subtree": "A/B/None", "arbiter": "A/B/None"},
        "retry": "same_prompt_technical_failures_only",
    })
    atomic_write_json(target / "schedules" / "internal_k1.json", {
        split: {str(row["sample_id"]): [0] for row in _internal_rows(config, split)}
        for split in INTERNAL_COUNTS
    })
    records = _vlrb_records(output)
    atomic_write_json(target / "schedules" / "vlrb_k3.json", support.source_schedule(
        output, [str(record["sample_id"]) for record in records]))
    details = {
        "worker_model": WORKER_MODEL,
        "rubric_sha256": manifest["rubric_sha256"],
        "internal_logical_requests": manifest["logical_request_budget"]["internal"],
        "vlrb_logical_requests": manifest["logical_request_budget"]["vl_rewardbench"],
    }
    _status(target, STAGES[0], details)
    print(json.dumps(details, indent=2))


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target(output)
    _require(target, STAGES[0])
    stored = load_json(target / "frozen_manifest.json")
    expected = _manifest(config, output, include_endpoints=False)
    for key, value in expected.items():
        if stored.get(key) != value:
            raise RuntimeError(f"Qwen2.5 Clean S5 manifest drift: {key}")
    rubric = _rubric(output)
    if StructuredRubric.load_json(
            target / "rubric_snapshot.json").rubric_sha256 != rubric.rubric_sha256:
        raise RuntimeError("Qwen2.5 Clean S5 rubric snapshot drift")
    return target, stored, rubric


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    live = qwen25.phase10._inspect_endpoints(_worker_config(config))
    if live != manifest.get("endpoint_identities"):
        raise RuntimeError("Qwen2.5 Clean S5 endpoint identity drift")


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    row = _internal_rows(config, "discovery100")[0]
    reports = [{
        "root_id": root_id,
        "criterion_name": rubric.get_node(root_id).criterion.name,
        "report": {"analysis_a": "a", "analysis_b": "b", "thought": "t", "answer": "None"},
    } for root_id in rubric.root_ids]
    subtree_prompt = internal.subtree_user_prompt(row, rubric, rubric.root_ids[0])
    arbiter_prompt = support.global_arbiter_user_prompt(row, reports)
    sources = _source_paths(output)
    s3_manifest = load_json(sources["qwen3_unified_subtree_manifest"])
    clean_manifest = load_json(sources["qwen3_clean_s5_manifest"])
    checks = {
        "worker_model_is_qwen25": manifest["worker_model"] == WORKER_MODEL,
        "phase17_e4_rubric": rubric.rubric_sha256 == SOURCE_RUBRIC_SHA256,
        "five_roots_27_nodes": len(rubric.root_ids) == 5 and len(rubric.nodes) == 27,
        "subtree_prompt_byte_identical": manifest["subtree_system_prompt_sha256"]
            == s3_manifest.get("system_prompt_sha256")
            == hashlib.sha256(internal.UNIFIED_SUBTREE_SYSTEM_PROMPT.encode()).hexdigest(),
        "arbiter_prompt_byte_identical": manifest["arbiter_system_prompt_sha256"]
            == clean_manifest.get("system_prompt_sha256")
            == hashlib.sha256(arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode()).hexdigest(),
        "subtree_parser_accepts_none": internal.parse_subtree_response(
            '{"answer":"None"}')["answer"] == "None",
        "arbiter_parser_accepts_none": arbiter.parse_global_arbiter_ab_only_response(
            '{"answer":"None"}')["answer"] == "None",
        "subtree_dynamic_pair_present": str(row["A"]) in subtree_prompt
            and str(row["B"]) in subtree_prompt,
        "arbiter_has_five_reports": sum(
            f"### {item['criterion_name']}" in arbiter_prompt for item in reports) == 5,
        "internal_k1_vlrb_k3": manifest["settings"]["internal_k"] == 1
            and manifest["settings"]["vl_rewardbench_k"] == 3,
        "model_only_variable_declared": manifest["single_changed_variable"] == "worker_model",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Qwen2.5 Clean S5 audit failed: {checks}")
    atomic_write_json(target / "offline_audit.json", {
        "schema_version": SCHEMA_VERSION, "offline_only": True, "checks": checks})
    _status(target, STAGES[1], checks)
    print(json.dumps(checks, indent=2))


def _call_subtree(
    config: Mapping[str, Any], target: Path, endpoint: BackendEndpointSpec,
    split: str, row: Mapping[str, Any], rubric: StructuredRubric,
    root_id: str, replicate: int, order: int, total_attempt_limit: int,
) -> dict[str, Any]:
    return support.call_one(
        config, endpoint, target / "cache" / split / "subtree",
        user_text=internal.subtree_user_prompt(row, rubric, root_id), row=row,
        request_key={
            "kind": "qwen25_clean_s5_unified_subtree",
            "split": split,
            "sample_id": str(row["sample_id"]),
            "root_id": root_id,
            "replicate": replicate,
            "order": order,
            "rubric_sha256": rubric.rubric_sha256,
        },
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=internal.SUBTREE_PROMPT_VERSION,
        system_prompt=internal.UNIFIED_SUBTREE_SYSTEM_PROMPT,
        response_parser=internal.parse_subtree_response,
        settings_loader=_settings,
    )


def _call_arbiter(
    config: Mapping[str, Any], target: Path, endpoint: BackendEndpointSpec,
    split: str, row: Mapping[str, Any], reports: Sequence[Mapping[str, Any]],
    replicate: int, order: int, total_attempt_limit: int,
) -> dict[str, Any]:
    digest = canonical_sha256(reports)
    return support.call_one(
        config, endpoint, target / "cache" / split / "arbiter",
        user_text=support.global_arbiter_user_prompt(row, reports), row=row,
        request_key={
            "kind": "qwen25_clean_s5_global_arbiter",
            "split": split,
            "sample_id": str(row["sample_id"]),
            "replicate": replicate,
            "order": order,
            "source_report_bundle_sha256": digest,
        },
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=internal.ARBITER_PROMPT_VERSION,
        system_prompt=arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        response_parser=arbiter.parse_global_arbiter_ab_only_response,
        settings_loader=_settings,
    )


def _blocked_call(endpoint: BackendEndpointSpec) -> dict[str, Any]:
    return {
        "parse_ok": False,
        "parsed": None,
        "model_generation_count": 0,
        "endpoint_id": endpoint.endpoint_id,
        "cache_hit": False,
        "metrics": {},
        "error": "blocked_by_unresolved_subtree",
    }


def _process_sample(
    config: Mapping[str, Any], target: Path, endpoint: BackendEndpointSpec,
    split: str, row: Mapping[str, Any], orders: Sequence[int],
    rubric: StructuredRubric, total_attempt_limit: int, label: str,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "endpoint_id": endpoint.endpoint_id,
        "orders": list(orders),
        "replicates": {},
    }
    displayed = {
        replicate: support.ordered_row(row, int(order), replicate)
        for replicate, order in enumerate(orders)
    }

    def invoke_subtree(replicate: int, root_id: str):
        order = int(orders[replicate])
        call = _call_subtree(
            config, target, endpoint, split, displayed[replicate], rubric,
            root_id, replicate, order, total_attempt_limit)
        return replicate, root_id, support.compact_call(call)

    subtree_calls: dict[int, dict[str, Any]] = {
        replicate: {} for replicate in range(len(orders))}
    max_workers = min(
        endpoint.max_concurrency, len(orders) * len(rubric.root_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(invoke_subtree, replicate, root_id)
            for replicate in range(len(orders))
            for root_id in rubric.root_ids
        ]
        for future in as_completed(futures):
            replicate, root_id, call = future.result()
            subtree_calls[replicate][root_id] = call

    def invoke_arbiter(replicate: int):
        reports = []
        for root_id in rubric.root_ids:
            call = subtree_calls[replicate][root_id]
            if call.get("parse_ok"):
                reports.append({
                    "root_id": root_id,
                    "criterion_name": rubric.get_node(root_id).criterion.name,
                    "report": call["parsed"],
                })
        if len(reports) != len(rubric.root_ids):
            return replicate, reports, _blocked_call(endpoint)
        call = _call_arbiter(
            config, target, endpoint, split, displayed[replicate], reports,
            replicate, int(orders[replicate]), total_attempt_limit)
        return replicate, reports, support.compact_call(call)

    arbiter_calls: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    with ThreadPoolExecutor(
            max_workers=min(endpoint.max_concurrency, len(orders))) as executor:
        futures = [executor.submit(invoke_arbiter, replicate)
                   for replicate in range(len(orders))]
        for future in as_completed(futures):
            replicate, reports, call = future.result()
            arbiter_calls[replicate] = (reports, call)

    for replicate in range(len(orders)):
        reports, final = arbiter_calls[replicate]
        result["replicates"][str(replicate)] = {
            "order": int(orders[replicate]),
            "subtrees": subtree_calls[replicate],
            "source_report_bundle_sha256": canonical_sha256(reports),
            "arbiter": final,
        }
    atomic_write_json(
        target / "bundles" / label / f"{canonical_sha256(sample_id)}.json",
        result)
    return result


def _run_rows(
    config: Mapping[str, Any], target: Path, split: str, label: str,
    rows: Sequence[Mapping[str, Any]], orders_by_id: Mapping[str, Sequence[int]],
    rubric: StructuredRubric, total_attempt_limit: int,
) -> dict[str, Any]:
    worker_config = _worker_config(config)
    endpoints = tuple(
        endpoint for endpoint in _pool(config).endpoints
        if endpoint.endpoint_id in ENDPOINT_IDS)
    if tuple(endpoint.endpoint_id for endpoint in endpoints) != ENDPOINT_IDS:
        raise RuntimeError("Qwen2.5 Clean S5 requires both frozen endpoints")
    assignment_path = target / "endpoint_assignment.json"
    assignments = load_json(assignment_path) if assignment_path.is_file() else {}
    pending: queue.Queue[Mapping[str, Any]] = queue.Queue()
    assigned = {endpoint.endpoint_id: queue.Queue() for endpoint in endpoints}
    for row in rows:
        key = f"{split}::{row['sample_id']}"
        endpoint_id = assignments.get(key)
        (assigned[endpoint_id] if endpoint_id in assigned else pending).put(row)
    lock = threading.Lock()
    completed = 0
    values: dict[str, Any] = {}
    started = time.perf_counter()
    prediction_path = target / "predictions" / f"{label}.json"
    previous_main_wall: float | None = None
    if prediction_path.is_file():
        previous = load_json(prediction_path)
        stored = previous.get("initial_main_run_wall_seconds")
        if isinstance(stored, (int, float)) and not isinstance(stored, bool):
            previous_main_wall = float(stored)
    progress_path = target / "progress" / f"{label}.json"
    atomic_write_json(progress_path, {
        "stage": label, "completed": 0, "total": len(rows),
        "percent": 0.0, "elapsed_seconds": 0.0, "eta_seconds": None,
        "endpoint_bundle_counts": {},
    })
    print(f"{label}: 0/{len(rows)} sample bundles started", flush=True)

    def worker(endpoint: BackendEndpointSpec) -> None:
        nonlocal completed
        while True:
            try:
                row = assigned[endpoint.endpoint_id].get_nowait()
            except queue.Empty:
                try:
                    row = pending.get_nowait()
                except queue.Empty:
                    return
                with lock:
                    assignments[f"{split}::{row['sample_id']}"] = endpoint.endpoint_id
                    atomic_write_json(assignment_path, assignments)
            sample_id = str(row["sample_id"])
            value = _process_sample(
                worker_config, target, endpoint, split, row,
                orders_by_id[sample_id], rubric, total_attempt_limit, label)
            with lock:
                values[sample_id] = value
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(rows) - completed)
                endpoint_counts = Counter(
                    str(item["endpoint_id"]) for item in values.values())
                atomic_write_json(progress_path, {
                    "stage": label, "completed": completed, "total": len(rows),
                    "percent": completed / len(rows) * 100 if rows else 100.0,
                    "current_sample_id": sample_id,
                    "elapsed_seconds": elapsed, "eta_seconds": eta,
                    "endpoint_bundle_counts": dict(endpoint_counts),
                })
                print(
                    f"{label}: {completed}/{len(rows)} sample={sample_id} "
                    f"endpoint={endpoint.endpoint_id} elapsed={elapsed/60:.1f}m "
                    f"ETA={eta/60:.1f}m", flush=True)

    max_orders = max(len(value) for value in orders_by_id.values())
    calls_per_subtree_wave = max_orders * len(rubric.root_ids)
    workers = [
        endpoint for endpoint in endpoints
        for _ in range(max(1, endpoint.max_concurrency // calls_per_subtree_wave))
    ]
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    pass_wall = time.perf_counter() - started
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "worker_model": WORKER_MODEL,
        "split": split,
        "label": label,
        "samples": [values[str(row["sample_id"])] for row in rows],
        "wall_seconds": pass_wall,
        "initial_main_run_wall_seconds": (
            previous_main_wall if previous_main_wall is not None else pass_wall),
    }
    atomic_write_json(prediction_path, result)
    return result


def _iter_calls(result: Mapping[str, Any]):
    for sample in result["samples"]:
        for replicate in sample["replicates"].values():
            yield from replicate["subtrees"].values()
            yield replicate["arbiter"]


def _telemetry(result: Mapping[str, Any]) -> dict[str, Any]:
    calls = list(_iter_calls(result))
    pass_wall = float(result.get("wall_seconds") or 0.0)
    main_wall = float(result.get("initial_main_run_wall_seconds") or pass_wall)
    model_generations = sum(int(call.get("model_generation_count", 0)) for call in calls)
    return {
        "logical_request_count": len(calls),
        "model_generation_count": model_generations,
        "parse_valid_count": sum(bool(call.get("parse_ok")) for call in calls),
        "parse_valid_rate": (sum(bool(call.get("parse_ok")) for call in calls) / len(calls)
                             if calls else 0.0),
        "semantic_none_count": sum(
            call.get("parse_ok") and (call.get("parsed") or {}).get("answer") == "None"
            for call in calls),
        "cache_hit_count": sum(bool(call.get("cache_hit")) for call in calls),
        "endpoint_call_counts": dict(Counter(
            str(call.get("endpoint_id")) for call in calls)),
        "api_attempts": sum(int(call.get("metrics", {}).get("api_attempts") or 0)
                            for call in calls),
        "errors": sum(int(call.get("metrics", {}).get("error_count") or 0)
                      for call in calls),
        "input_tokens": sum(int(call.get("metrics", {}).get("input_tokens") or 0)
                            for call in calls),
        "output_tokens": sum(int(call.get("metrics", {}).get("output_tokens") or 0)
                             for call in calls),
        "main_run_wall_seconds": main_wall,
        "latest_pass_wall_seconds": pass_wall,
        "logical_requests_per_minute": len(calls) / main_wall * 60 if main_wall else 0.0,
        "sample_bundles_per_minute": (
            len(result["samples"]) / main_wall * 60 if main_wall else 0.0),
    }


def _failures(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for sample in result["samples"]:
        for replicate_key, replicate in sample["replicates"].items():
            for root_id, call in replicate["subtrees"].items():
                if not call.get("parse_ok"):
                    failures.append({
                        "sample_id": sample["sample_id"],
                        "replicate": int(replicate_key),
                        "kind": "subtree", "root_id": root_id,
                        "error": call.get("error"),
                        "model_generation_count": call.get("model_generation_count", 0),
                    })
            call = replicate["arbiter"]
            if not call.get("parse_ok"):
                failures.append({
                    "sample_id": sample["sample_id"],
                    "replicate": int(replicate_key),
                    "kind": "arbiter", "root_id": None,
                    "error": call.get("error"),
                    "model_generation_count": call.get("model_generation_count", 0),
                })
    return failures


def _internal_orders(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int]]:
    return {str(row["sample_id"]): (0,) for row in rows}


def _vlrb_rows(records: Sequence[Mapping[str, Any]]):
    return support.vlrb_rows(records)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[1])
    _verify_live(config, manifest)
    settings = _settings(config)
    attempt_limit = 1 + settings["max_parse_retries"]
    details: dict[str, Any] = {}
    count = settings["smoke_internal_samples_per_split"]
    for split in INTERNAL_COUNTS:
        rows = _internal_rows(config, split)[:count]
        result = _run_rows(
            config, target, split, f"smoke_{split}", rows,
            _internal_orders(rows), rubric, attempt_limit)
        details[split] = {
            "sample_count": len(rows),
            "technical_failure_count": len(_failures(result)),
            "telemetry": _telemetry(result),
        }
    records = support.selected_records(
        _vlrb_records(output), settings["smoke_vlrb_sample_count"])
    rows = _vlrb_rows(records)
    full_schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in _vlrb_records(output)])
    result = _run_rows(
        config, target, "vl_rewardbench", "smoke_vlrb20", rows,
        {str(row["sample_id"]): full_schedule[str(row["sample_id"])] for row in rows},
        rubric, attempt_limit)
    details["vl_rewardbench"] = {
        "sample_count": len(rows),
        "technical_failure_count": len(_failures(result)),
        "telemetry": _telemetry(result),
    }
    if any(value["technical_failure_count"] for value in details.values()):
        raise RuntimeError(f"Qwen2.5 Clean S5 smoke failed: {details}")
    _status(target, STAGES[2], details)
    print(json.dumps(details, indent=2))


def internal_run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[2])
    _verify_live(config, manifest)
    details = {}
    failures = load_json(target / "parse_failures.json") \
        if (target / "parse_failures.json").is_file() else {}
    for split in INTERNAL_COUNTS:
        rows = _internal_rows(config, split)
        result = _run_rows(
            config, target, split, f"internal_{split}", rows,
            _internal_orders(rows), rubric, 1)
        current = _failures(result)
        failures[split] = current
        details[split] = {
            "sample_count": len(rows),
            "technical_failure_count": len(current),
            "telemetry": _telemetry(result),
        }
    atomic_write_json(target / "parse_failures.json", failures)
    _status(target, STAGES[3], details)
    print(json.dumps(details, indent=2))


def vlrb_run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[3])
    _verify_live(config, manifest)
    records = _vlrb_records(output)
    rows = _vlrb_rows(records)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in records])
    result = _run_rows(
        config, target, "vl_rewardbench", "vlrb_full", rows,
        schedule, rubric, 1)
    failures = load_json(target / "parse_failures.json")
    failures["vl_rewardbench"] = _failures(result)
    atomic_write_json(target / "parse_failures.json", failures)
    details = {
        "sample_count": len(records),
        "technical_failure_count": len(failures["vl_rewardbench"]),
        "telemetry": _telemetry(result),
    }
    _status(target, STAGES[4], details)
    print(json.dumps(details, indent=2))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[4])
    _verify_live(config, manifest)
    attempt_limit = 1 + _settings(config)["max_parse_retries"]
    details = {}
    failures = {}
    for split in INTERNAL_COUNTS:
        rows = _internal_rows(config, split)
        result = _run_rows(
            config, target, split, f"internal_{split}", rows,
            _internal_orders(rows), rubric, attempt_limit)
        failures[split] = _failures(result)
        details[split] = {
            "technical_failure_count": len(failures[split]),
            "telemetry": _telemetry(result),
        }
    records = _vlrb_records(output)
    rows = _vlrb_rows(records)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in records])
    result = _run_rows(
        config, target, "vl_rewardbench", "vlrb_full", rows,
        schedule, rubric, attempt_limit)
    failures["vl_rewardbench"] = _failures(result)
    details["vl_rewardbench"] = {
        "technical_failure_count": len(failures["vl_rewardbench"]),
        "telemetry": _telemetry(result),
    }
    atomic_write_json(target / "parse_failures.json", failures)
    atomic_write_json(target / "retry_summary.json", {
        "schema_version": SCHEMA_VERSION,
        "total_attempt_limit": attempt_limit,
        "splits": details,
    })
    _status(target, STAGES[5], details)
    print(json.dumps(details, indent=2))


def _answer(call: Mapping[str, Any]) -> str | None:
    if not call.get("parse_ok"):
        return None
    value = (call.get("parsed") or {}).get("answer")
    return value if value in {"A", "B"} else None


def _root_majority(replicate: Mapping[str, Any]) -> str | None:
    values = [_answer(call) for call in replicate["subtrees"].values()]
    a_count, b_count = values.count("A"), values.count("B")
    return "A" if a_count > b_count else "B" if b_count > a_count else None


def _internal_report(config: Mapping[str, Any], output: Path, target: Path) -> dict[str, Any]:
    qwen3_report = load_json(_source_paths(output)["qwen3_internal_clean_s5"])
    result = {}
    for split in INTERNAL_COUNTS:
        rows = _internal_rows(config, split)
        artifact = load_json(target / "predictions" / f"internal_{split}.json")
        treatment = [_answer(sample["replicates"]["0"]["arbiter"])
                     for sample in artifact["samples"]]
        technical = [not sample["replicates"]["0"]["arbiter"].get("parse_ok")
                     for sample in artifact["samples"]]
        qwen3 = qwen3_report.get("splits", {}).get(split, {}).get(
            "systems", {}).get("single_prompt_global_arbiter")
        if not isinstance(qwen3, dict):
            raise RuntimeError(f"Qwen3 internal Clean S5 control missing: {split}")
        result[split] = {
            "systems": {
                "qwen3_clean_s5_v2": qwen3,
                SYSTEM_NAME: internal._metrics(rows, treatment, technical),
            },
            "paired_qwen25_vs_qwen3": internal._paired(
                rows, qwen3["predictions"], treatment),
            "root_response_distribution": {
                root_id: dict(Counter(
                    (_answer(sample["replicates"]["0"]["subtrees"][root_id]) or "None")
                    for sample in artifact["samples"]))
                for root_id in _rubric(output).root_ids
            },
            "telemetry": _telemetry(artifact),
        }
    return result


def _vlrb_vote_matrix(result: Mapping[str, Any]) -> list[list[int | None]]:
    votes: list[list[int | None]] = [[] for _ in range(VLRB_K)]
    for sample in result["samples"]:
        if len(sample["orders"]) != VLRB_K or len(sample["replicates"]) != VLRB_K:
            raise RuntimeError("Qwen2.5 Clean S5 VL-RewardBench artifact is incomplete")
        for replicate in range(VLRB_K):
            value = sample["replicates"][str(replicate)]
            answer = _answer(value["arbiter"]) or "None"
            votes[replicate].append(support.display_to_original(
                answer, int(value["order"])))
    return votes


def _override_audit(
    records: Sequence[Mapping[str, Any]], result: Mapping[str, Any],
) -> dict[str, Any]:
    names = ("5-0-0", "4-1-0 / 4-0-1", "3-2-0 / 3-1-1 / 3-0-2",
             "tie / sparse / all-None")
    patterns = {name: {
        "replicate_count": 0, "arbiter_follows": 0, "arbiter_overrides": 0,
        "override_corrected": 0, "override_harmed": 0,
    } for name in names}
    for index, sample in enumerate(result["samples"]):
        gold = int(records[index]["preferred_original_index"])
        for replicate_index in range(VLRB_K):
            replicate = sample["replicates"][str(replicate_index)]
            values = [(_answer(call) or "None")
                      for call in replicate["subtrees"].values()]
            decisive = sorted((values.count("A"), values.count("B")), reverse=True)
            name = (names[0] if decisive == [5, 0] and values.count("None") == 0
                    else names[1] if decisive[0] == 4
                    else names[2] if decisive[0] == 3 else names[3])
            item = patterns[name]
            item["replicate_count"] += 1
            majority = _root_majority(replicate)
            decision = _answer(replicate["arbiter"])
            if decision == majority:
                item["arbiter_follows"] += 1
                continue
            item["arbiter_overrides"] += 1
            order = int(replicate["order"])
            majority_correct = support.display_to_original(
                majority or "None", order) == gold
            arbiter_correct = support.display_to_original(
                decision or "None", order) == gold
            item["override_corrected"] += int(arbiter_correct and not majority_correct)
            item["override_harmed"] += int(majority_correct and not arbiter_correct)
    totals = {key: sum(item[key] for item in patterns.values())
              for key in ("replicate_count", "arbiter_follows", "arbiter_overrides",
                          "override_corrected", "override_harmed")}
    totals["override_net_corrected"] = (
        totals["override_corrected"] - totals["override_harmed"])
    return {"schema_version": SCHEMA_VERSION, "patterns": patterns, "totals": totals}


def _vlrb_report(output: Path, target: Path) -> dict[str, Any]:
    records = _vlrb_records(output)
    result = load_json(target / "predictions" / "vlrb_full.json")
    treatment_votes = _vlrb_vote_matrix(result)
    treatment_metrics = vlrb_metrics._system_metrics(records, treatment_votes)

    qwen3_clean_path = _source_paths(output)["qwen3_clean_s5"]
    qwen3_clean_votes = support._validated_system(
        qwen3_clean_path, QWEN3_CLEAN_SYSTEM, records)
    qwen3_clean_metrics = vlrb_metrics._system_metrics(records, qwen3_clean_votes)
    qwen3_s0_path = output.parent / support.SOURCE_S3_EXPERIMENT / "final_report.json"
    qwen3_s0_votes = support._validated_system(
        qwen3_s0_path, "s0_explicit_recursive", records)
    qwen3_s0_metrics = vlrb_metrics._system_metrics(records, qwen3_s0_votes)

    qwen25_report = _qwen25_explicit_report(output)
    qwen25_explicit = dict(qwen25_report["metrics"][QWEN25_EXPLICIT_SYSTEM])
    qwen25_predictions = qwen25_explicit["original_index_predictions"]
    treatment_predictions = treatment_metrics["original_index_predictions"]
    qwen3_clean_predictions = qwen3_clean_metrics["original_index_predictions"]
    qwen3_s0_predictions = qwen3_s0_metrics["original_index_predictions"]

    systems = {
        "qwen3_explicit_recursive_e4": {"metrics": qwen3_s0_metrics},
        "qwen3_clean_s5_v2": {"metrics": qwen3_clean_metrics},
        "qwen25_explicit_recursive_e4": {"metrics": qwen25_explicit},
        SYSTEM_NAME: {"votes_by_replicate": treatment_votes,
                      "metrics": treatment_metrics},
    }
    paired = {
        "qwen25_clean_vs_qwen25_explicit": vlrb._paired(
            records, qwen25_predictions, treatment_predictions),
        "qwen25_clean_vs_qwen3_clean": vlrb._paired(
            records, qwen3_clean_predictions, treatment_predictions),
        "qwen3_clean_vs_qwen3_explicit": vlrb._paired(
            records, qwen3_s0_predictions, qwen3_clean_predictions),
    }
    aggregation_gain_qwen3 = (
        qwen3_clean_metrics["strict_accuracy"] - qwen3_s0_metrics["strict_accuracy"])
    aggregation_gain_qwen25 = (
        treatment_metrics["strict_accuracy"] - qwen25_explicit["strict_accuracy"])
    return {
        "systems": systems,
        "paired": paired,
        "model_by_aggregation": {
            "qwen3": {
                "explicit_recursive_strict_accuracy": qwen3_s0_metrics["strict_accuracy"],
                "clean_s5_strict_accuracy": qwen3_clean_metrics["strict_accuracy"],
                "aggregation_gain": aggregation_gain_qwen3,
            },
            "qwen25": {
                "explicit_recursive_strict_accuracy": qwen25_explicit["strict_accuracy"],
                "clean_s5_strict_accuracy": treatment_metrics["strict_accuracy"],
                "aggregation_gain": aggregation_gain_qwen25,
            },
            "difference_in_differences": aggregation_gain_qwen25 - aggregation_gain_qwen3,
        },
        "arbiter_override_analysis": _override_audit(records, result),
        "telemetry": _telemetry(result),
        "control_provenance": {
            "qwen3_clean_report": str(qwen3_clean_path.resolve()),
            "qwen3_clean_report_sha256": file_sha256(qwen3_clean_path),
            "qwen3_s0_report": str(qwen3_s0_path.resolve()),
            "qwen3_s0_report_sha256": file_sha256(qwen3_s0_path),
            "qwen25_explicit_report": str(
                _source_paths(output)["qwen25_explicit_e4"].resolve()),
            "qwen25_explicit_report_sha256": file_sha256(
                _source_paths(output)["qwen25_explicit_e4"]),
            "metrics_recomputed_where_vote_matrices_exist": True,
        },
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, _, rubric = _load_frozen(config, output)
    _require(target, STAGES[5])
    failures = load_json(target / "parse_failures.json")
    internal_report = _internal_report(config, output, target)
    vlrb_report = _vlrb_report(output, target)
    internal_wall = sum(
        float(item["telemetry"]["main_run_wall_seconds"])
        for item in internal_report.values())
    vlrb_wall = float(vlrb_report["telemetry"]["main_run_wall_seconds"])
    efficiency = {
        "logical_request_counts": {
            "internal_k1": sum(INTERNAL_COUNTS.values()) * 6,
            "vl_rewardbench_k3": VLRB_COUNT * VLRB_K * 6,
            "combined": sum(INTERNAL_COUNTS.values()) * 6 + VLRB_COUNT * VLRB_K * 6,
        },
        "main_run_wall_seconds": {
            "internal_total": internal_wall,
            "vl_rewardbench": vlrb_wall,
            "combined": internal_wall + vlrb_wall,
        },
        "internal_telemetry": {
            split: item["telemetry"] for split, item in internal_report.items()},
        "vlrb_telemetry": vlrb_report["telemetry"],
    }
    category_analysis = {
        name: {
            "groups": item["metrics"].get("groups", {}),
            "source_groups": item["metrics"].get("source_groups", {}),
        }
        for name, item in vlrb_report["systems"].items()
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "system": SYSTEM_NAME,
        "worker_model": WORKER_MODEL,
        "rubric_sha256": rubric.rubric_sha256,
        "internal": internal_report,
        "vl_rewardbench": vlrb_report,
        "category_analysis": category_analysis,
        "efficiency": efficiency,
        "unresolved_technical_failure_count": sum(
            len(items) for items in failures.values()),
        "selection_after_diagnostics_forbidden": True,
    }
    atomic_write_json(target / "reports" / "internal.json", internal_report)
    atomic_write_json(target / "reports" / "vl_rewardbench.json", vlrb_report)
    atomic_write_json(target / "paired_comparison.json", {
        "internal": {split: item["paired_qwen25_vs_qwen3"]
                     for split, item in internal_report.items()},
        "vl_rewardbench": vlrb_report["paired"],
    })
    atomic_write_json(
        target / "arbiter_override_analysis.json",
        vlrb_report["arbiter_override_analysis"])
    atomic_write_json(target / "category_analysis.json", category_analysis)
    atomic_write_json(target / "efficiency.json", efficiency)
    atomic_write_json(target / "final_report.json", value)

    lines = [
        "# Qwen2.5 Clean S5-v2 transfer", "",
        "The Phase17 epoch-4 rubric and Clean S5-v2 aggregation protocol are "
        "frozen. Only the worker model changes to Qwen2.5-VL-7B-Instruct.",
        "", "## Internal K=1", "",
        "| Split | Model / aggregation | Strict ACC | OverallAcc | MacroAcc | Coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, item in internal_report.items():
        for name, metrics in item["systems"].items():
            macro = "-" if metrics.get("macro_acc") is None else f"{metrics['macro_acc']:.2%}"
            lines.append(
                f"| {split} | {name} | {metrics['strict_accuracy']:.2%} | "
                f"{metrics['overall_acc']:.2%} | {macro} | {metrics['coverage']:.2%} |")
    lines.extend([
        "", "## VL-RewardBench K=3", "",
        "| Model / aggregation | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in vlrb_report["systems"].items():
        metrics = item["metrics"]
        lines.append(
            f"| {name} | {metrics['strict_accuracy']:.2%} | "
            f"{metrics['overall_acc']:.2%} | {metrics['macro_acc']:.2%} | "
            f"{metrics['coverage']:.2%} | {metrics['correct_count']} |")
    factorial = vlrb_report["model_by_aggregation"]
    lines.extend([
        "", "## Model-by-aggregation effect", "",
        "| Worker | Explicit Recursive | Clean S5-v2 | Aggregation gain |",
        "|---|---:|---:|---:|",
        f"| Qwen3-VL-8B | {factorial['qwen3']['explicit_recursive_strict_accuracy']:.2%} | "
        f"{factorial['qwen3']['clean_s5_strict_accuracy']:.2%} | "
        f"{factorial['qwen3']['aggregation_gain']:+.2%} |",
        f"| Qwen2.5-VL-7B | {factorial['qwen25']['explicit_recursive_strict_accuracy']:.2%} | "
        f"{factorial['qwen25']['clean_s5_strict_accuracy']:.2%} | "
        f"{factorial['qwen25']['aggregation_gain']:+.2%} |",
        "", f"Difference-in-differences: {factorial['difference_in_differences']:+.2%}.",
        "", "## Paired VL-RewardBench comparisons", "",
        "| Comparison | Corrected | Harmed | Net | Exact McNemar p |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in vlrb_report["paired"].items():
        lines.append(
            f"| {name} | {item['corrected_count']} | {item['harmed_count']} | "
            f"{item['net_corrected']} | {item['mcnemar_exact_two_sided_p']:.6g} |")
    lines.extend([
        "", "## Category Strict ACC", "",
        "| Model / aggregation | General | Hallucination | Reasoning |",
        "|---|---:|---:|---:|",
    ])
    for name, item in vlrb_report["systems"].items():
        groups = item["metrics"]["groups"]
        lines.append(
            f"| {name} | {groups['general']['strict_accuracy']:.2%} | "
            f"{groups['hallucination']['strict_accuracy']:.2%} | "
            f"{groups['reasoning']['strict_accuracy']:.2%} |")
    lines.extend([
        "", "## Efficiency", "",
        f"Internal logical requests: {efficiency['logical_request_counts']['internal_k1']:,}.",
        f"VL-RewardBench logical requests: {efficiency['logical_request_counts']['vl_rewardbench_k3']:,}.",
        f"Internal main-run wall time: {internal_wall:.1f}s; "
        f"VL-RewardBench main-run wall time: {vlrb_wall:.1f}s.",
        "", f"Unresolved technical failures: {value['unresolved_technical_failure_count']}.",
        "Semantic None is a valid decision; only parse/transport failures are retried.",
    ])
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "internal_strict_accuracy": {
            split: item["systems"][SYSTEM_NAME]["strict_accuracy"]
            for split, item in internal_report.items()},
        "vl_rewardbench": {
            key: vlrb_report["systems"][SYSTEM_NAME]["metrics"][key]
            for key in ("strict_accuracy", "overall_acc", "macro_acc", "coverage")},
        "qwen25_aggregation_gain": factorial["qwen25"]["aggregation_gain"],
        "difference_in_differences": factorial["difference_in_differences"],
        "unresolved_technical_failures": value["unresolved_technical_failure_count"],
    }
    _status(target, STAGES[6], summary)
    print(json.dumps(summary, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze,
        STAGES[1]: audit,
        STAGES[2]: smoke,
        STAGES[3]: internal_run,
        STAGES[4]: vlrb_run,
        STAGES[5]: retry,
        STAGES[6]: report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported Qwen2.5 Clean S5 stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

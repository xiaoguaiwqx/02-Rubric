"""Counterfactual Global-Arbiter audit for rejected Phase21 candidates.

The experiment never regenerates a Unified-Subtree report.  For every rejected
Phase21 candidate it replaces exactly one epoch-start root report, keeps the
other four epoch-start reports unchanged, and calls only the frozen Global
Arbiter.  The resulting system delta is compared with root-local selective
utility and the original strict acceptance statistic.
"""

from __future__ import annotations

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
import urllib.error

from critiq.structured.backend_pool import BackendEndpointSpec, BackendPoolSpec
from critiq.structured.cache import canonical_sha256
from critiq.structured.schema import StructuredRubric

from . import _global_arbiter_ab_only_support as support
from . import aligned_system_runtime as runtime
from . import global_arbiter_ab_only as arbiter
from . import unified_subtree_bundle_evolution as phase21
from .experiment_utils import atomic_write_json, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase21_rejected_candidate_counterfactual_arbiter_audit_v1"
SOURCE_EXPERIMENT_DIR = phase21.EXPERIMENT_DIR
PROTOCOL_VERSION = "phase21-rejected-candidate-counterfactual-arbiter-audit-v1"
CONFIG_KEY = "phase21_rejected_candidate_counterfactual_arbiter_audit_v1_experiment"
STAGES = (
    "counterfactual-arbiter-audit-freeze",
    "counterfactual-arbiter-audit-audit",
    "counterfactual-arbiter-audit-smoke",
    "counterfactual-arbiter-audit-stage1",
    "counterfactual-arbiter-audit-stage2",
    "counterfactual-arbiter-audit-retry",
    "counterfactual-arbiter-audit-report",
)

STAGE1_KEYS = (
    (1, "init_05_clarity_and_coherence"),
    (2, "init_02_visual_grounding_and_details"),
    (1, "init_03_factuality_no_hallucination"),
    (4, "init_01_completeness_and_coverage"),
    (5, "init_02_visual_grounding_and_details"),
    (5, "init_04_creativity_and_expressiveness"),
)
STAGE1_ROLES = (
    "positive", "positive", "borderline",
    "negative", "negative", "negative",
)

SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": SOURCE_EXPERIMENT_DIR,
    "source_candidate_count": 25,
    "source_candidate_status": "competition_rejected",
    "counterfactual_unit": "replace_exactly_one_root_report",
    "unchanged_root_source": "same_epoch_start_baseline",
    "subtree_regeneration_count": 0,
    "stage1_candidate_count": 6,
    "stage2_remaining_candidate_count": 19,
    "sample_count": 100,
    "k": 1,
    "ab_swap": False,
    "temperature": 0.5,
    "max_tokens": 2048,
    "max_parse_retries": 10,
    "generation_seed_policy": "unset",
    "primary_local_predictor": "frozen_scope_normalized_selective_utility_delta",
    "selective_utility_scores": {"correct": 1, "wrong": -1, "None": 0},
    "primary_system_metric": "strict_accuracy_delta_vs_common_epoch0_control",
    "ground_truth": "discovery100_human_reviewed_A_or_B",
    "bootstrap_seed": 42,
    "bootstrap_repetitions": 5000,
}


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _source(output: Path) -> Path:
    return output / SOURCE_EXPERIMENT_DIR


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


def _source_manifest(output: Path) -> dict[str, Any]:
    path = _source(output) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("Phase21 frozen manifest is missing")
    value = load_json(path)
    if value.get("protocol_version") != phase21.PROTOCOL_VERSION:
        raise RuntimeError("Phase21 source protocol drift")
    return value


def _source_history(output: Path) -> dict[str, Any]:
    path = _source(output) / "evolution_history.json"
    if not path.is_file():
        raise RuntimeError("Phase21 evolution history is missing")
    value = load_json(path)
    if not value.get("completed"):
        raise RuntimeError("Phase21 evolution must be complete")
    return value


def _candidate_id(epoch: int, root_index: int, attempt: int) -> str:
    return f"e{epoch:02d}_r{root_index:02d}_a{attempt:02d}_split"


def _relative(source: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(source.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Phase21 artifact is outside source experiment: {path}") from exc


def _candidate_entries(output: Path) -> list[dict[str, Any]]:
    source = _source(output)
    history = _source_history(output)
    epoch0_rubric = StructuredRubric.load_json(
        source / "epochs" / "epoch_00" / "rubric_committed.json")
    root_order = {root_id: index + 1
                  for index, root_id in enumerate(epoch0_rubric.root_ids)}
    entries = []
    for record in history.get("attempts", []):
        if (record.get("operator") != "split"
                or record.get("decision") != "competition_rejected"):
            raise RuntimeError(
                "counterfactual audit requires every Phase21 attempt to be a rejected Split")
        epoch = int(record["epoch"])
        attempt = int(record["attempt"])
        root_id = str(record["root_id"])
        if root_id not in root_order:
            raise RuntimeError(f"unknown Phase21 root: {root_id}")
        attempt_dir = Path(str(record["attempt_dir"]))
        if not attempt_dir.is_dir():
            matches = list((source / "epochs" / f"epoch_{epoch:02d}" / "roots").glob(
                f"*/attempt_{attempt:02d}"))
            attempt_dir = next((path for path in matches
                                if load_json(path / "root_unified_subtree.json").get(
                                    "root_id") == root_id), Path())
        candidate_path = attempt_dir / "root_unified_subtree.json"
        candidate_rubric_path = attempt_dir / "candidate_rubric.json"
        baseline_dir = source / "epochs" / f"epoch_{epoch:02d}" / "root_baselines_before"
        baseline_paths = {
            baseline_root_id: baseline_dir / f"{baseline_root_id}.json"
            for baseline_root_id in epoch0_rubric.root_ids
        }
        paths = [candidate_path, candidate_rubric_path, *baseline_paths.values()]
        if any(not path.is_file() for path in paths):
            missing = [str(path) for path in paths if not path.is_file()]
            raise RuntimeError(f"Phase21 candidate artifacts are missing: {missing}")
        key = (epoch, root_id)
        stage1_index = STAGE1_KEYS.index(key) if key in STAGE1_KEYS else None
        entries.append({
            "candidate_id": _candidate_id(epoch, root_order[root_id], attempt),
            "epoch": epoch,
            "attempt": attempt,
            "operator": "split",
            "root_id": root_id,
            "root_index": root_order[root_id],
            "selection_stage": "stage1" if stage1_index is not None else "stage2",
            "selection_role": (STAGE1_ROLES[stage1_index]
                               if stage1_index is not None else "full_audit_remainder"),
            "candidate_root_path": _relative(source, candidate_path),
            "candidate_rubric_path": _relative(source, candidate_rubric_path),
            "baseline_root_paths": {
                key: _relative(source, value) for key, value in baseline_paths.items()},
            "source_hashes": {
                "candidate_root": file_sha256(candidate_path),
                "candidate_rubric": file_sha256(candidate_rubric_path),
                "baseline_roots": {
                    key: file_sha256(value) for key, value in baseline_paths.items()},
            },
        })
    entries.sort(key=lambda item: (item["epoch"], item["root_index"], item["attempt"]))
    if len(entries) != SETTINGS["source_candidate_count"]:
        raise RuntimeError(f"expected 25 Phase21 candidates, found {len(entries)}")
    if sum(item["selection_stage"] == "stage1" for item in entries) != 6:
        raise RuntimeError("frozen Stage-1 candidate selection is incomplete")
    return entries


def _load_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _settings(config)
    path = _target(output) / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run counterfactual-arbiter-audit-freeze first")
    value = load_json(path)
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("counterfactual audit protocol drift")
    return value


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


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _settings(config)
    target, source = _target(output), _source(output)
    rows = _rows(config)
    source_manifest = _source_manifest(output)
    entries = _candidate_entries(output)
    control_path = source / "epochs" / "epoch_00" / "system" / "discovery.json"
    if not control_path.is_file():
        raise RuntimeError("Phase21 epoch-0 system control is missing")
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "settings": settings,
        "dataset": {
            "path": phase21.phase17.DISCOVERY_PATH,
            "count": len(rows),
            "sha256": file_sha256(phase21.base._path(phase21.phase17.DISCOVERY_PATH)),
            "sample_ids": [str(row["sample_id"]) for row in rows],
        },
        "source": {
            "experiment": SOURCE_EXPERIMENT_DIR,
            "manifest_sha256": file_sha256(source / "frozen_manifest.json"),
            "history_sha256": file_sha256(source / "evolution_history.json"),
            "control_path": _relative(source, control_path),
            "control_sha256": file_sha256(control_path),
            "unified_runtime_identity": source_manifest["unified_runtime_identity"],
            "worker_endpoint_identities": source_manifest["worker_endpoint_identities"],
        },
        "candidates": entries,
        "stage1_candidate_ids": [item["candidate_id"] for item in entries
                                 if item["selection_stage"] == "stage1"],
        "stage2_candidate_ids": [item["candidate_id"] for item in entries
                                 if item["selection_stage"] == "stage2"],
        "ground_truth_used": True,
        "subtree_regeneration_count": 0,
    }
    path = target / "frozen_manifest.json"
    if path.is_file():
        if load_json(path) != manifest:
            raise RuntimeError("counterfactual audit frozen manifest drift")
        print("counterfactual-arbiter-audit-freeze already completed")
        return
    _write(path, manifest)
    _status(target, STAGES[0], {
        "candidate_count": len(entries),
        "stage1_candidate_count": len(manifest["stage1_candidate_ids"]),
        "stage2_candidate_count": len(manifest["stage2_candidate_ids"]),
        "new_subtree_requests": 0,
    })
    print(json.dumps(_status_details(target, STAGES[0]), indent=2))


def _status_details(target: Path, name: str) -> dict[str, Any]:
    return load_json(target / "stage_status.json")["stages"][name]["details"]


def _artifact(source: Path, relative_path: str) -> dict[str, Any]:
    return load_json(source / Path(relative_path))


def _sample_index(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError("root artifact has no samples")
    result = {str(item["sample_id"]): item for item in samples}
    if len(result) != len(samples):
        raise RuntimeError("root artifact has duplicate sample IDs")
    return result


def _scientific_root_hash(value: Mapping[str, Any]) -> str:
    return phase21._root_calls_sha256(value)


def _audit_entry(source: Path, entry: Mapping[str, Any],
                 sample_ids: Sequence[str]) -> dict[str, Any]:
    candidate = _artifact(source, str(entry["candidate_root_path"]))
    candidate_rubric = StructuredRubric.load_json(
        source / str(entry["candidate_rubric_path"]))
    root_id = str(entry["root_id"])
    if candidate.get("root_id") != root_id or root_id not in candidate_rubric.root_ids:
        raise RuntimeError(f"candidate root identity mismatch: {entry['candidate_id']}")
    indexes = {root: _sample_index(_artifact(source, path))
               for root, path in entry["baseline_root_paths"].items()}
    indexes[root_id] = _sample_index(candidate)
    expected = list(sample_ids)
    if any(list(index) != expected for index in indexes.values()):
        raise RuntimeError(f"candidate sample order drift: {entry['candidate_id']}")
    calls = [sample["call"] for index in indexes.values() for sample in index.values()]
    if any(not call.get("parse_ok") for call in calls):
        raise RuntimeError(f"candidate contains unresolved root call: {entry['candidate_id']}")
    if any(int(sample.get("order", -1)) != 0
           for index in indexes.values() for sample in index.values()):
        raise RuntimeError(f"candidate root report is not K=1/order=0: {entry['candidate_id']}")
    return {
        "candidate_id": entry["candidate_id"],
        "root_count": len(indexes),
        "sample_count": len(expected),
        "candidate_root_scientific_sha256": _scientific_root_hash(candidate),
    }


def audit(config: Mapping[str, Any], output: Path) -> None:
    manifest = _load_manifest(config, output)
    target, source = _target(output), _source(output)
    rows = _rows(config)
    sample_ids = [str(row["sample_id"]) for row in rows]
    current_entries = _candidate_entries(output)
    checks = {
        "dataset_count_100": len(rows) == 100,
        "dataset_sample_order_match": sample_ids == manifest["dataset"]["sample_ids"],
        "dataset_hash_match": file_sha256(
            phase21.base._path(phase21.phase17.DISCOVERY_PATH))
            == manifest["dataset"]["sha256"],
        "candidate_manifest_match": current_entries == manifest["candidates"],
        "candidate_count_25": len(current_entries) == 25,
        "stage1_count_6": len(manifest["stage1_candidate_ids"]) == 6,
        "stage2_count_19": len(manifest["stage2_candidate_ids"]) == 19,
        "source_runtime_match": phase21._unified_runtime_identity(config)
            == manifest["source"]["unified_runtime_identity"],
        "subtree_regeneration_zero": manifest["subtree_regeneration_count"] == 0,
        "ground_truth_is_dataset": manifest["ground_truth_used"] is True,
    }
    audited = [_audit_entry(source, entry, sample_ids) for entry in current_entries]
    # Phase21 accepted no candidates.  Hence all epoch-start baseline root calls
    # must be scientifically identical to the common epoch-0 control reports.
    epoch0_roots = {
        root_id: _artifact(source, f"epochs/epoch_00/root_baselines_before/{root_id}.json")
        for root_id in current_entries[0]["baseline_root_paths"]
    }
    common_hashes = {root_id: _scientific_root_hash(value)
                     for root_id, value in epoch0_roots.items()}
    baseline_common = True
    for entry in current_entries:
        for root_id, path in entry["baseline_root_paths"].items():
            if _scientific_root_hash(_artifact(source, path)) != common_hashes[root_id]:
                baseline_common = False
    checks["all_epoch_baselines_match_common_epoch0_control"] = baseline_common
    if not all(checks.values()):
        raise RuntimeError(f"counterfactual Arbiter offline audit failed: {checks}")
    value = {
        "schema_version": "1.0.0", "offline_only": True,
        "checks": checks, "candidates": audited,
        "common_epoch0_root_call_hashes": common_hashes,
    }
    _write(target / "offline_audit.json", value)
    _status(target, STAGES[1], checks)
    print(json.dumps(checks, indent=2))


def _endpoint_identities_with_retry(
    config: Mapping[str, Any], *, attempts: int = 6,
) -> list[dict[str, Any]]:
    """Retry transient `/v1/models` timeouts before a live stage starts."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return phase21._endpoint_identities(config)
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(30, 2 ** attempt)
            print(
                f"counterfactual endpoint preflight attempt={attempt}/{attempts} "
                f"failed; retrying in {delay}s: {type(exc).__name__}: {exc}",
                flush=True)
            time.sleep(delay)
    raise RuntimeError(
        f"counterfactual endpoint preflight failed after {attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}") from last_error


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if _endpoint_identities_with_retry(config) != manifest["source"][
            "worker_endpoint_identities"]:
        raise RuntimeError("counterfactual Arbiter endpoint identity drift")
    if phase21._unified_runtime_identity(config) != manifest["source"][
            "unified_runtime_identity"]:
        raise RuntimeError("counterfactual Arbiter prompt/parser/model identity drift")


def _mapped_root_answer(sample: Mapping[str, Any]) -> str:
    return runtime._answer(sample["call"], int(sample["order"]))


def _utility(prediction: str, gold: str) -> int:
    if prediction == "None":
        return 0
    if prediction not in {"A", "B"}:
        raise RuntimeError("selective utility cannot score a technical failure")
    return 1 if prediction == gold else -1


def local_metrics(entry: Mapping[str, Any], source: Path,
                  rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    root_id = str(entry["root_id"])
    baseline = _artifact(source, entry["baseline_root_paths"][root_id])
    candidate = _artifact(source, str(entry["candidate_root_path"]))
    before = _sample_index(baseline)
    after = _sample_index(candidate)
    by_id = {str(row["sample_id"]): row for row in rows}
    scope = tuple(str(item) for item in baseline["metrics"]["root_scope_sample_ids"])

    def utility_for(index: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> int:
        return sum(_utility(_mapped_root_answer(index[sample_id]),
                            str(by_id[sample_id]["answer"]))
                   for sample_id in ids)

    before_scope = utility_for(before, scope)
    after_scope = utility_for(after, scope)
    all_ids = tuple(by_id)
    before_full = utility_for(before, all_ids)
    after_full = utility_for(after, all_ids)
    paired = runtime.paired_root(baseline, candidate, rows)
    stored = load_json(source / "epochs" / f"epoch_{entry['epoch']:02d}"
                       / "summary.json")
    stored_attempt = next(item for item in stored["attempts"]
                          if item["root_id"] == root_id)
    if stored_attempt["root_scope_evaluation"] != paired:
        raise RuntimeError(f"stored Phase21 paired metric drift: {entry['candidate_id']}")
    support_count = len(scope)
    return {
        "scope_support": support_count,
        "baseline_scope_selective_utility_count": before_scope,
        "candidate_scope_selective_utility_count": after_scope,
        "scope_selective_utility_delta_count": after_scope - before_scope,
        "scope_selective_utility_delta": (
            (after_scope - before_scope) / support_count if support_count else 0.0),
        "baseline_full_selective_utility_count": before_full,
        "candidate_full_selective_utility_count": after_full,
        "full_selective_utility_delta_count": after_full - before_full,
        "full_selective_utility_delta": (after_full - before_full) / len(rows),
        "formal_corrected": paired["root_scope_corrected"],
        "formal_harmed": paired["root_scope_harmed"],
        "formal_net_corrected": paired["root_scope_net_corrected"],
        "formal_net_corrected_normalized": (
            paired["root_scope_net_corrected"] / support_count if support_count else 0.0),
        "strict_accuracy_before": paired["root_scope_strict_accuracy_before"],
        "strict_accuracy_after": paired["root_scope_strict_accuracy_after"],
        "strict_accuracy_delta": paired["root_scope_strict_accuracy_delta"],
        "coverage_before": paired["root_scope_coverage_before"],
        "coverage_after": paired["root_scope_coverage_after"],
        "coverage_delta": (paired["root_scope_coverage_after"]
                           - paired["root_scope_coverage_before"]),
        "covered_accuracy_before": baseline["metrics"]["root_scope_covered_accuracy"],
        "covered_accuracy_after": candidate["metrics"]["root_scope_covered_accuracy"],
        "covered_accuracy_delta": (candidate["metrics"]["root_scope_covered_accuracy"]
                                   - baseline["metrics"]["root_scope_covered_accuracy"]),
    }


def _report_bundle_index(
    entry: Mapping[str, Any], source: Path, sample_ids: Sequence[str],
) -> dict[str, tuple[list[dict[str, Any]], str]]:
    rubric = StructuredRubric.load_json(source / str(entry["candidate_rubric_path"]))
    root_id = str(entry["root_id"])
    indexes = {}
    for current_root_id in rubric.root_ids:
        path = (str(entry["candidate_root_path"]) if current_root_id == root_id
                else entry["baseline_root_paths"][current_root_id])
        indexes[current_root_id] = _sample_index(_artifact(source, path))
    bundles = {}
    for sample_id in sample_ids:
        reports = []
        for current_root_id in rubric.root_ids:
            call = indexes[current_root_id][sample_id]["call"]
            if not call.get("parse_ok"):
                raise RuntimeError(f"unresolved cached subtree report: {sample_id}")
            reports.append({
                "root_id": current_root_id,
                "criterion_name": rubric.get_node(current_root_id).criterion.name,
                "report": call["parsed"],
            })
        bundles[sample_id] = (reports, canonical_sha256(reports))
    return bundles


def _arbiter_call(config: Mapping[str, Any], endpoint: BackendEndpointSpec,
                  target: Path, entry: Mapping[str, Any], row: Mapping[str, Any],
                  reports: Sequence[Mapping[str, Any]], digest: str,
                  total_attempt_limit: int, namespace: str) -> dict[str, Any]:
    settings = _runtime_settings(config)
    call = support.call_one(
        config, endpoint, target / "cache" / namespace,
        user_text=support.global_arbiter_user_prompt(row, reports), row=row,
        request_key={
            "kind": "phase21_rejected_candidate_counterfactual_arbiter",
            "candidate_id": str(entry["candidate_id"]),
            "sample_id": str(row["sample_id"]),
            "source_report_bundle_sha256": digest,
            "replicate": 0,
            "order": 0,
        },
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=runtime.unified.ARBITER_PROMPT_VERSION,
        system_prompt=arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        response_parser=arbiter.parse_global_arbiter_ab_only_response,
        settings_loader=settings.as_loader(),
    )
    return {
        "sample_id": str(row["sample_id"]),
        "order": 0,
        "source_report_bundle_sha256": digest,
        "arbiter": support.compact_call(call),
    }


def _prediction(call: Mapping[str, Any]) -> str:
    return runtime._answer(call, 0)


def _system_metrics(samples: Sequence[Mapping[str, Any]],
                    rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in rows}
    predictions = [_prediction(sample["arbiter"]) for sample in samples]
    gold = [str(by_id[str(sample["sample_id"])]["answer"]) for sample in samples]
    decisive = [item in {"A", "B"} for item in predictions]
    correct = [item == target for item, target in zip(predictions, gold)]
    covered = sum(decisive)
    return {
        "sample_count": len(samples),
        "strict_accuracy": sum(correct) / len(samples),
        "coverage": covered / len(samples),
        "covered_accuracy": (sum(ok and active for ok, active in zip(correct, decisive))
                             / covered if covered else 0.0),
        "none_rate": predictions.count("None") / len(samples),
        "technical_failure_count": predictions.count("technical_failure"),
        "prediction_distribution": {
            "A": predictions.count("A"), "B": predictions.count("B"),
            "None": predictions.count("None"),
            "technical_failure": predictions.count("technical_failure"),
        },
        "predictions": predictions,
    }


def _control_metrics(output: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = runtime.load(
        _source(output) / "epochs" / "epoch_00" / "system" / "discovery.json")
    metrics = dict(value["metrics"])
    if metrics.get("technical_failure_count"):
        raise RuntimeError("common Phase21 epoch-0 control has technical failures")
    if len(metrics.get("predictions", [])) != len(rows):
        raise RuntimeError("common Phase21 epoch-0 control prediction drift")
    return metrics


def _paired_system(control: Mapping[str, Any], candidate: Mapping[str, Any],
                   rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return runtime.paired(control, candidate, rows)


def _run_candidate(config: Mapping[str, Any], output: Path,
                   entry: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                   *, total_attempt_limit: int,
                   namespace: str = "full") -> dict[str, Any]:
    target = _target(output)
    path = target / "predictions" / f"{entry['candidate_id']}.json"
    if path.is_file():
        old = load_json(path)
        if old.get("metrics", {}).get("technical_failure_count") == 0:
            return old
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoints = tuple(item for item in pool.endpoints
                      if item.endpoint_id in runtime.ENDPOINT_IDS)
    if tuple(item.endpoint_id for item in endpoints) != runtime.ENDPOINT_IDS:
        raise RuntimeError("counterfactual audit requires vllm-8000 and vllm-8001")
    assignment_path = target / "endpoint_assignment.json"
    assignments = load_json(assignment_path) if assignment_path.is_file() else {}
    pending: queue.Queue[Mapping[str, Any]] = queue.Queue()
    assigned = {endpoint.endpoint_id: queue.Queue() for endpoint in endpoints}
    for row in rows:
        assignment_key = f"{entry['candidate_id']}::{row['sample_id']}"
        endpoint_id = assignments.get(assignment_key)
        (assigned[endpoint_id] if endpoint_id in assigned else pending).put(row)
    bundles = _report_bundle_index(
        entry, _source(output), [str(row["sample_id"]) for row in rows])
    values: dict[str, Any] = {}
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()
    label = str(entry["candidate_id"])
    print(f"counterfactual_{label}: 0/{len(rows)} Arbiter calls started", flush=True)

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
            assignment_key = f"{label}::{row['sample_id']}"
            with lock:
                if assignment_key not in assignments:
                    assignments[assignment_key] = endpoint.endpoint_id
                    _write(assignment_path, assignments)
            reports, digest = bundles[str(row["sample_id"])]
            value = _arbiter_call(
                config, endpoint, target, entry, row, reports, digest,
                total_attempt_limit, namespace)
            with lock:
                values[str(row["sample_id"])] = value
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(rows) - completed)
                print(
                    f"counterfactual_{label}: {completed}/{len(rows)} "
                    f"sample={row['sample_id']} endpoint={endpoint.endpoint_id} "
                    f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)

    workers = runtime._interleaved_workers(endpoints)
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    samples = [values[str(row["sample_id"])] for row in rows]
    metrics = _system_metrics(samples, rows)
    result = {
        "schema_version": "1.0.0",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": entry["candidate_id"],
        "epoch": entry["epoch"], "attempt": entry["attempt"],
        "root_id": entry["root_id"],
        "selection_stage": entry["selection_stage"],
        "selection_role": entry["selection_role"],
        "k": 1, "orders": [0],
        "new_subtree_request_count": 0,
        "new_arbiter_request_count": len(rows),
        "samples": samples,
        "metrics": metrics,
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
    path = target / "smoke" / "report.json"
    if path.is_file():
        print(json.dumps(load_json(path), indent=2))
        return
    entries = {item["candidate_id"]: item for item in manifest["candidates"]}
    selected = [entries[manifest["stage1_candidate_ids"][0]],
                entries[manifest["stage1_candidate_ids"][-1]]]
    rows = _smoke_rows(_rows(config))
    outputs = [_run_candidate(
        config, output, {**entry, "candidate_id": f"smoke_{entry['candidate_id']}"},
        rows, total_attempt_limit=1 + SETTINGS["max_parse_retries"],
        namespace="smoke") for entry in selected]
    technical = sum(item["metrics"]["technical_failure_count"] for item in outputs)
    value = {
        "schema_version": "1.0.0", "status": "passed" if technical == 0 else "failed",
        "candidate_count": len(outputs), "sample_count_per_candidate": len(rows),
        "new_subtree_request_count": 0,
        "new_arbiter_request_count": len(outputs) * len(rows),
        "technical_failure_count": technical,
        "parse_valid_rate": 1 - technical / (len(outputs) * len(rows)),
    }
    _write(path, value)
    if technical:
        raise RuntimeError(f"counterfactual smoke has {technical} technical failures")
    _status(target, STAGES[2], value)
    print(json.dumps(value, indent=2))


def _run_selection(config: Mapping[str, Any], output: Path, stage: str,
                   *, total_attempt_limit: int) -> dict[str, Any]:
    target = _target(output)
    manifest = _load_manifest(config, output)
    _verify_live(config, manifest)
    ids = (manifest["stage1_candidate_ids"] if stage == "stage1"
           else manifest["stage2_candidate_ids"])
    entries = {item["candidate_id"]: item for item in manifest["candidates"]}
    rows = _rows(config)
    values = [_run_candidate(
        config, output, entries[candidate_id], rows,
        total_attempt_limit=total_attempt_limit) for candidate_id in ids]
    technical = sum(item["metrics"]["technical_failure_count"] for item in values)
    details = {
        "candidate_count": len(values),
        "new_subtree_request_count": 0,
        "logical_arbiter_request_count": len(values) * len(rows),
        "technical_failure_count": technical,
    }
    if stage == "stage1":
        source = _source(output)
        control = _control_metrics(output, rows)
        candidates = []
        for entry, value in zip((entries[candidate_id] for candidate_id in ids), values):
            local = local_metrics(entry, source, rows)
            paired = _paired_system(control, value["metrics"], rows)
            candidates.append({
                "candidate_id": entry["candidate_id"],
                "root_id": entry["root_id"],
                "selection_role": entry["selection_role"],
                "scope_selective_utility_delta_count": local[
                    "scope_selective_utility_delta_count"],
                "scope_selective_utility_delta": local[
                    "scope_selective_utility_delta"],
                "formal_net_corrected": local["formal_net_corrected"],
                "system_strict_accuracy": value["metrics"]["strict_accuracy"],
                "system_strict_accuracy_delta": paired["strict_accuracy_delta"],
                "system_corrected": paired["corrected"],
                "system_harmed": paired["harmed"],
                "system_net_corrected": paired["net_corrected"],
            })
        role_summary = {}
        for role in sorted({item["selection_role"] for item in candidates}):
            group = [item for item in candidates if item["selection_role"] == role]
            role_summary[role] = {
                "candidate_count": len(group),
                "mean_selective_utility_delta": _mean([
                    item["scope_selective_utility_delta"] for item in group]),
                "mean_system_strict_accuracy_delta": _mean([
                    item["system_strict_accuracy_delta"] for item in group]),
                "system_net_corrected": sum(
                    item["system_net_corrected"] for item in group),
            }
        _write(target / "stage1_report.json", {
            "schema_version": "1.0.0",
            "common_control_strict_accuracy": control["strict_accuracy"],
            "pre_results_selection_frozen": True,
            "candidates": candidates,
            "role_summary": role_summary,
        })
    _status(target, STAGES[3] if stage == "stage1" else STAGES[4], details)
    print(json.dumps(details, indent=2))
    return details


def stage1(config: Mapping[str, Any], output: Path) -> None:
    _require_stage(_target(output), STAGES[2])
    _run_selection(config, output, "stage1",
                   total_attempt_limit=1 + SETTINGS["max_parse_retries"])


def stage2(config: Mapping[str, Any], output: Path) -> None:
    _require_stage(_target(output), STAGES[3])
    _run_selection(config, output, "stage2",
                   total_attempt_limit=1 + SETTINGS["max_parse_retries"])


def retry(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    manifest = _load_manifest(config, output)
    _verify_live(config, manifest)
    rows = _rows(config)
    unresolved = 0
    for entry in manifest["candidates"]:
        path = target / "predictions" / f"{entry['candidate_id']}.json"
        if not path.is_file():
            continue
        value = _run_candidate(
            config, output, entry, rows,
            total_attempt_limit=1 + 2 * SETTINGS["max_parse_retries"])
        unresolved += int(value["metrics"]["technical_failure_count"])
    details = {"unresolved_technical_failures": unresolved,
               "target_total_attempt_limit": 21}
    _status(target, STAGES[5], details)
    print(json.dumps(details, indent=2))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean, y_mean = _mean(xs), _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else None


def _ranks(values: Sequence[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            result[ordered[position]] = rank
        start = end
    return result


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def _kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    concordant = discordant = ties_x = ties_y = 0
    for left in range(len(xs)):
        for right in range(left + 1, len(xs)):
            dx, dy = xs[left] - xs[right], ys[left] - ys[right]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y))
    return ((concordant - discordant) / denominator if denominator else None)


def _center_by_root(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    groups: dict[str, list[float]] = {}
    for record in records:
        groups.setdefault(str(record["root_id"]), []).append(float(record[key]))
    means = {root_id: _mean(values) for root_id, values in groups.items()}
    return [float(record[key]) - means[str(record["root_id"])] for record in records]


def _fixed_effect(records: Sequence[Mapping[str, Any]], x_key: str,
                  y_key: str) -> dict[str, Any]:
    xs = _center_by_root(records, x_key)
    ys = _center_by_root(records, y_key)
    denominator = sum(x * x for x in xs)
    return {
        "pearson": _pearson(xs, ys),
        "slope": sum(x * y for x, y in zip(xs, ys)) / denominator
        if denominator else None,
    }


def _stratified_bootstrap_spearman(
    records: Sequence[Mapping[str, Any]], x_key: str, y_key: str,
    *, repetitions: int, seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["root_id"]), []).append(record)
    generator = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sampled = []
        for root_id in sorted(groups):
            group = groups[root_id]
            sampled.extend(generator.choice(group) for _ in range(len(group)))
        value = _spearman(
            [float(item[x_key]) for item in sampled],
            [float(item[y_key]) for item in sampled])
        if value is not None:
            values.append(value)
    values.sort()
    if not values:
        return {"repetitions": repetitions, "valid_repetitions": 0,
                "ci95": [None, None]}
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return {"repetitions": repetitions, "valid_repetitions": len(values),
            "ci95": [low, high]}


def _correlations(records: Sequence[Mapping[str, Any]], x_key: str,
                  y_key: str) -> dict[str, Any]:
    xs = [float(item[x_key]) for item in records]
    ys = [float(item[y_key]) for item in records]
    return {
        "x": x_key, "y": y_key, "candidate_count": len(records),
        "pearson": _pearson(xs, ys),
        "spearman": _spearman(xs, ys),
        "kendall_tau_b": _kendall_tau_b(xs, ys),
        "root_fixed_effect": _fixed_effect(records, x_key, y_key),
        "stratified_candidate_bootstrap_spearman": _stratified_bootstrap_spearman(
            records, x_key, y_key,
            repetitions=SETTINGS["bootstrap_repetitions"],
            seed=SETTINGS["bootstrap_seed"]),
    }


def _telemetry(value: Mapping[str, Any]) -> dict[str, Any]:
    calls = [item["arbiter"] for item in value["samples"]]
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


def report(config: Mapping[str, Any], output: Path) -> None:
    target, source = _target(output), _source(output)
    _require_stage(target, STAGES[4])
    manifest = _load_manifest(config, output)
    rows = _rows(config)
    control = _control_metrics(output, rows)
    records = []
    telemetry = []
    for entry in manifest["candidates"]:
        path = target / "predictions" / f"{entry['candidate_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"counterfactual candidate result is missing: {path}")
        value = load_json(path)
        if value["metrics"]["technical_failure_count"]:
            raise RuntimeError(
                f"counterfactual candidate has technical failures: {entry['candidate_id']}")
        local = local_metrics(entry, source, rows)
        paired = _paired_system(control, value["metrics"], rows)
        record = {
            "candidate_id": entry["candidate_id"],
            "epoch": entry["epoch"], "attempt": entry["attempt"],
            "root_id": entry["root_id"], "root_index": entry["root_index"],
            "selection_stage": entry["selection_stage"],
            "selection_role": entry["selection_role"],
            **local,
            "system_strict_accuracy": value["metrics"]["strict_accuracy"],
            "system_strict_accuracy_delta": paired["strict_accuracy_delta"],
            "system_corrected": paired["corrected"],
            "system_harmed": paired["harmed"],
            "system_net_corrected": paired["net_corrected"],
            "system_corrected_sample_ids": paired["corrected_sample_ids"],
            "system_harmed_sample_ids": paired["harmed_sample_ids"],
            "system_coverage": value["metrics"]["coverage"],
            "system_none_rate": value["metrics"]["none_rate"],
            "system_prediction_distribution": value["metrics"][
                "prediction_distribution"],
        }
        records.append(record)
        telemetry.append(_telemetry(value))
    records.sort(key=lambda item: (item["epoch"], item["root_index"]))
    primary_x = "scope_selective_utility_delta"
    outcome = "system_strict_accuracy_delta"
    correlations = {
        "primary_selective_utility": _correlations(records, primary_x, outcome),
        "formal_strict_net": _correlations(
            records, "formal_net_corrected_normalized", outcome),
        "covered_accuracy_delta": _correlations(
            records, "covered_accuracy_delta", outcome),
        "coverage_delta": _correlations(records, "coverage_delta", outcome),
        "full_data_selective_utility": _correlations(
            records, "full_selective_utility_delta", outcome),
    }
    stage1 = [item for item in records if item["selection_stage"] == "stage1"]
    role_summary = {}
    for role in STAGE1_ROLES:
        values = [item for item in stage1 if item["selection_role"] == role]
        if values and role not in role_summary:
            role_summary[role] = {
                "candidate_count": len(values),
                "mean_selective_utility_delta": _mean([
                    item[primary_x] for item in values]),
                "mean_system_strict_accuracy_delta": _mean([
                    item[outcome] for item in values]),
                "system_net_corrected": sum(item["system_net_corrected"]
                                            for item in values),
            }
    usage = {
        key: sum(float(item[key]) for item in telemetry)
        for key in telemetry[0]
    }
    for key in ("logical_arbiter_requests", "model_generations", "api_attempts",
                "input_tokens", "output_tokens", "errors"):
        usage[key] = int(usage[key])
    control_predictions = list(control["predictions"])
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "claim_scope": (
            "Discovery100 mechanism audit only: association between root-local "
            "selective utility and counterfactual Global-Arbiter system benefit."),
        "common_control": {
            "source": "Phase21 epoch-0 committed system",
            "strict_accuracy": control["strict_accuracy"],
            "coverage": control["coverage"],
            "none_rate": control_predictions.count("None") / len(control_predictions),
            "prediction_distribution": {
                "A": control_predictions.count("A"),
                "B": control_predictions.count("B"),
                "None": control_predictions.count("None"),
                "technical_failure": control_predictions.count("technical_failure"),
            },
        },
        "candidate_count": len(records),
        "candidate_results": records,
        "stage1_role_summary": role_summary,
        "correlations": correlations,
        "usage": usage,
        "new_subtree_request_count": 0,
        "new_arbiter_request_count": len(records) * len(rows),
        "selection_after_results_forbidden": True,
    }
    _write(target / "final_report.json", value)
    csv_path = target / "candidate_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat_keys = [key for key in records[0]
                 if not isinstance(records[0][key], (dict, list))]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in flat_keys} for row in records)
    primary = correlations["primary_selective_utility"]
    markdown = [
        "# Phase21 Rejected-Candidate Counterfactual Arbiter Audit", "",
        f"- Candidates: {len(records)}",
        f"- Common control Strict ACC: {control['strict_accuracy']:.2%}",
        f"- Selective-utility Pearson: {primary['pearson']}",
        f"- Selective-utility Spearman: {primary['spearman']}",
        f"- Selective-utility Kendall tau-b: {primary['kendall_tau_b']}",
        f"- New Unified-Subtree calls: 0",
        f"- New Global-Arbiter calls: {len(records) * len(rows)}", "",
        "This is an in-sample mechanism audit on Discovery100, not a heldout "
        "generalization result.",
    ]
    (target / "final_report.md").write_text("\n".join(markdown), encoding="utf-8")
    _status(target, STAGES[6], {
        "candidate_count": len(records),
        "primary_spearman": primary["spearman"],
        "primary_pearson": primary["pearson"],
        "new_subtree_request_count": 0,
        "new_arbiter_request_count": len(records) * len(rows),
    })
    print(json.dumps(_status_details(target, STAGES[6]), indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze,
        STAGES[1]: audit,
        STAGES[2]: smoke,
        STAGES[3]: stage1,
        STAGES[4]: stage2,
        STAGES[5]: retry,
        STAGES[6]: report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported counterfactual Arbiter stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

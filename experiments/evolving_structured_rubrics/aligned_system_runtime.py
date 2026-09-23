"""Shared Unified-Subtree -> Global-Arbiter runtime for aligned evolution.

The runtime deliberately knows nothing about Split or Refine.  It evaluates a
rubric as the deployed system does and supports the one-root incremental case:
unchanged root reports are copied from a frozen baseline artifact, the changed
root is regenerated, and the arbiter is always rerun on the resulting bundle.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Mapping, Sequence

from critiq.structured.backend_pool import BackendEndpointSpec, BackendPoolSpec
from critiq.structured.cache import canonical_sha256
from critiq.structured.schema import StructuredRubric

from . import _global_arbiter_ab_only_support as support
from . import global_arbiter_ab_only as arbiter
from . import internal_global_arbiter_k1 as unified
from .experiment_utils import atomic_write_json


SCHEMA_VERSION = "1.0.0"
PROTOCOL_VERSION = "unified-subtree-global-arbiter-aligned-runtime-v1"
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")


def _interleaved_workers(
    endpoints: Sequence[BackendEndpointSpec],
) -> list[BackendEndpointSpec]:
    """Start endpoint workers in an interleaved order.

    Submitting all workers for the first endpoint first lets a fast endpoint
    claim a small queue before workers for the second endpoint even start.
    Interleaving preserves each endpoint's configured concurrency while making
    the two-server pool usable for short smoke and pilot batches as well.
    """
    workers: list[BackendEndpointSpec] = []
    for slot in range(max((item.max_concurrency for item in endpoints), default=0)):
        workers.extend(
            item for item in endpoints if slot < item.max_concurrency)
    return workers


@dataclass(frozen=True)
class RuntimeSettings:
    temperature: float
    max_tokens: int
    max_parse_retries: int
    generation_seed_policy: str = "unset"
    retain_arbiter_reason: bool = False

    def as_loader(self):
        value = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_parse_retries": self.max_parse_retries,
            "generation_seed_policy": self.generation_seed_policy,
        }

        def load(_: Mapping[str, Any]) -> dict[str, Any]:
            return dict(value)

        return load


def _root_subtree_sha256(rubric: StructuredRubric, root_id: str) -> str:
    node_ids = tuple(
        node_id for node_id in rubric.preorder_node_ids()
        if rubric.root_id_for(node_id) == root_id)
    return canonical_sha256({
        "root_id": root_id,
        "nodes": [rubric.get_node(node_id).to_dict() for node_id in node_ids],
        "edges": [edge.to_dict() for edge in rubric.edges
                  if edge.parent_id in node_ids and edge.child_id in node_ids],
    })


def _call_subtree(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, cache_dir: Path,
    split_name: str, row: Mapping[str, Any], rubric: StructuredRubric,
    root_id: str, replicate: int, order: int, attempts: int,
    settings: RuntimeSettings,
) -> dict[str, Any]:
    return support.call_one(
        config, endpoint, cache_dir / "subtree",
        user_text=unified.subtree_user_prompt(row, rubric, root_id), row=row,
        request_key={
            "kind": "aligned_unified_subtree",
            "split": split_name,
            "sample_id": str(row["sample_id"]),
            "root_id": root_id,
            "root_subtree_sha256": _root_subtree_sha256(rubric, root_id),
            "replicate": replicate,
            "order": order,
        },
        total_attempt_limit=attempts,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=unified.SUBTREE_PROMPT_VERSION,
        system_prompt=unified.UNIFIED_SUBTREE_SYSTEM_PROMPT,
        response_parser=unified.parse_subtree_response,
        settings_loader=settings.as_loader(),
    )


def _call_arbiter(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, cache_dir: Path,
    split_name: str, row: Mapping[str, Any], reports: Sequence[Mapping[str, Any]],
    replicate: int, order: int, attempts: int, settings: RuntimeSettings,
) -> dict[str, Any]:
    bundle_sha256 = canonical_sha256(reports)
    return support.call_one(
        config, endpoint, cache_dir / "arbiter",
        user_text=support.global_arbiter_user_prompt(row, reports), row=row,
        request_key={
            "kind": "aligned_global_arbiter",
            "split": split_name,
            "sample_id": str(row["sample_id"]),
            "source_report_bundle_sha256": bundle_sha256,
            "replicate": replicate,
            "order": order,
        },
        total_attempt_limit=attempts,
        protocol_version=(PROTOCOL_VERSION + "-full-reason"
                          if settings.retain_arbiter_reason else PROTOCOL_VERSION),
        prompt_version=unified.ARBITER_PROMPT_VERSION,
        system_prompt=arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        response_parser=(arbiter.parse_global_arbiter_with_reason
                         if settings.retain_arbiter_reason
                         else arbiter.parse_global_arbiter_ab_only_response),
        settings_loader=settings.as_loader(),
    )


def _baseline_index(value: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if value is None:
        return {}
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise ValueError("baseline system artifact has no sample list")
    result = {str(item["sample_id"]): item for item in samples}
    if len(result) != len(samples):
        raise ValueError("baseline system artifact has duplicate sample IDs")
    return result


def _reports_from_calls(
    rubric: StructuredRubric, calls: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports = []
    for root_id in rubric.root_ids:
        call = calls[root_id]
        if not call.get("parse_ok"):
            continue
        reports.append({
            "root_id": root_id,
            "criterion_name": rubric.get_node(root_id).criterion.name,
            "report": call["parsed"],
        })
    return reports


def _blocked(endpoint: BackendEndpointSpec) -> dict[str, Any]:
    return {
        "parse_ok": False,
        "parsed": None,
        "model_generation_count": 0,
        "endpoint_id": endpoint.endpoint_id,
        "cache_hit": False,
        "error": "blocked_by_unresolved_subtree",
    }


def _one_sample(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, cache_dir: Path,
    split_name: str, row: Mapping[str, Any], orders: Sequence[int],
    rubric: StructuredRubric, baseline: Mapping[str, Any] | None,
    changed_root_ids: frozenset[str] | None, attempts: int,
    settings: RuntimeSettings,
) -> dict[str, Any]:
    sample = {
        "sample_id": str(row["sample_id"]),
        "endpoint_id": endpoint.endpoint_id,
        "orders": list(orders),
        "replicates": {},
    }
    baseline_replicates = {} if baseline is None else baseline.get("replicates", {})
    for replicate, order in enumerate(orders):
        displayed = support.ordered_row(row, int(order), replicate)
        old = baseline_replicates.get(str(replicate), {})
        old_calls = old.get("subtrees", {})
        calls: dict[str, Any] = {}
        regenerated = []
        for root_id in rubric.root_ids:
            reuse = (changed_root_ids is not None and root_id not in changed_root_ids
                     and root_id in old_calls)
            if reuse:
                calls[root_id] = dict(old_calls[root_id])
                calls[root_id]["incremental_reuse"] = True
            else:
                call = _call_subtree(
                    config, endpoint, cache_dir, split_name, displayed, rubric,
                    root_id, replicate, int(order), attempts, settings)
                calls[root_id] = support.compact_call(call)
                calls[root_id]["incremental_reuse"] = False
                regenerated.append(root_id)
        reports = _reports_from_calls(rubric, calls)
        final = (_blocked(endpoint) if len(reports) != len(rubric.root_ids)
                 else support.compact_call(_call_arbiter(
                     config, endpoint, cache_dir, split_name, displayed, reports,
                     replicate, int(order), attempts, settings)))
        sample["replicates"][str(replicate)] = {
            "order": int(order),
            "subtrees": calls,
            "regenerated_root_ids": regenerated,
            "report_bundle_sha256": canonical_sha256(reports),
            "arbiter": final,
        }
    return sample


def _answer(call: Mapping[str, Any], order: int) -> str:
    if not call.get("parse_ok"):
        return "technical_failure"
    value = (call.get("parsed") or {}).get("answer")
    if value == "None":
        return "None"
    if value not in {"A", "B"}:
        return "technical_failure"
    if order == 0:
        return str(value)
    return "B" if value == "A" else "A"


def metrics(value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in rows}
    per_replicate: list[list[str]] = []
    k = int(value["k"])
    for replicate in range(k):
        answers = []
        for sample in value["samples"]:
            item = sample["replicates"][str(replicate)]
            answers.append(_answer(item["arbiter"], int(item["order"])))
        per_replicate.append(answers)
    final = []
    for index in range(len(value["samples"])):
        votes = [items[index] for items in per_replicate]
        a_count, b_count = votes.count("A"), votes.count("B")
        if not a_count and not b_count and "technical_failure" in votes:
            final.append("technical_failure")
        else:
            final.append("A" if a_count > b_count else "B" if b_count > a_count else "None")
    gold = [str(by_id[str(item["sample_id"])]["answer"]) for item in value["samples"]]
    decisive = [item in {"A", "B"} for item in final]
    correct = [item == target for item, target in zip(final, gold)]
    covered = sum(decisive)
    technical_calls = 0
    total_calls = 0
    for sample in value["samples"]:
        for item in sample["replicates"].values():
            subtree_calls = tuple(item["subtrees"].values())
            technical_calls += sum(not call.get("parse_ok") for call in subtree_calls)
            technical_calls += not item["arbiter"].get("parse_ok")
            total_calls += len(subtree_calls) + 1
    return {
        "sample_count": len(final),
        "strict_accuracy": sum(correct) / len(final),
        "accuracy": sum(correct) / len(final),
        "coverage": covered / len(final),
        "covered_accuracy": (sum(ok and active for ok, active in zip(correct, decisive))
                             / covered if covered else 0.0),
        "none_rate": final.count("None") / len(final),
        "technical_failure_count": technical_calls,
        "technical_failure_rate": technical_calls / total_calls if total_calls else 0.0,
        "predictions": final,
        "predictions_by_replicate": per_replicate,
    }


def paired(before: Mapping[str, Any], after: Mapping[str, Any],
           rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old = before["predictions"]
    new = after["predictions"]
    if len(old) != len(rows) or len(new) != len(rows):
        raise ValueError("paired system metrics do not align with rows")
    corrected = [str(row["sample_id"]) for row, a, b in zip(rows, old, new)
                 if a != row["answer"] and b == row["answer"]]
    harmed = [str(row["sample_id"]) for row, a, b in zip(rows, old, new)
              if a == row["answer"] and b != row["answer"]]
    return {
        "corrected_sample_ids": corrected,
        "harmed_sample_ids": harmed,
        "corrected": len(corrected),
        "harmed": len(harmed),
        "net_corrected": len(corrected) - len(harmed),
        "strict_accuracy_delta": after["strict_accuracy"] - before["strict_accuracy"],
    }


def root_metrics(
    value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    scope_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Score one Unified-Subtree on a candidate-independent frozen scope."""
    by_id = {str(row["sample_id"]): row for row in rows}
    samples = {str(item["sample_id"]): item for item in value["samples"]}
    if set(samples) != set(by_id):
        raise ValueError("root artifact sample identity does not match rows")
    scope = tuple(str(item) for item in scope_sample_ids)
    if len(set(scope)) != len(scope) or not set(scope).issubset(by_id):
        raise ValueError("root scope contains duplicate or unknown sample IDs")
    predictions = {
        sample_id: _answer(samples[sample_id]["call"], int(samples[sample_id]["order"]))
        for sample_id in by_id
    }
    scope_predictions = [predictions[sample_id] for sample_id in scope]
    gold = [str(by_id[sample_id]["answer"]) for sample_id in scope]
    decisive = [item in {"A", "B"} for item in scope_predictions]
    correct = [item == target for item, target in zip(scope_predictions, gold)]
    covered = sum(decisive)
    all_predictions = [predictions[str(row["sample_id"])] for row in rows]
    return {
        "root_id": str(value["root_id"]),
        "root_scope_support": len(scope),
        "root_scope_sample_ids": list(scope),
        "root_scope_strict_accuracy": (
            sum(correct) / len(scope) if scope else 0.0),
        "root_scope_coverage": covered / len(scope) if scope else 0.0,
        "root_scope_covered_accuracy": (
            sum(ok and active for ok, active in zip(correct, decisive)) / covered
            if covered else 0.0),
        "root_scope_none_count": scope_predictions.count("None"),
        "technical_failure_count": all_predictions.count("technical_failure"),
        "predictions": all_predictions,
        "scope_predictions": scope_predictions,
        "outside_scope_coverage": (
            sum(predictions[sample_id] in {"A", "B"}
                for sample_id in by_id if sample_id not in set(scope))
            / max(1, len(by_id) - len(scope))),
    }


def paired_root(
    before: Mapping[str, Any], after: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Paired comparison for two root artifacts sharing one frozen scope."""
    old = before["metrics"]
    new = after["metrics"]
    scope = tuple(str(item) for item in old["root_scope_sample_ids"])
    if scope != tuple(str(item) for item in new["root_scope_sample_ids"]):
        raise ValueError("root candidate changed the frozen evaluation scope")
    by_id = {str(row["sample_id"]): row for row in rows}
    corrected = [
        sample_id for sample_id, a, b in zip(
            scope, old["scope_predictions"], new["scope_predictions"])
        if a != str(by_id[sample_id]["answer"])
        and b == str(by_id[sample_id]["answer"])
    ]
    harmed = [
        sample_id for sample_id, a, b in zip(
            scope, old["scope_predictions"], new["scope_predictions"])
        if a == str(by_id[sample_id]["answer"])
        and b != str(by_id[sample_id]["answer"])
    ]
    return {
        "root_id": old["root_id"],
        "root_scope_support": len(scope),
        "root_scope_strict_accuracy_before": old["root_scope_strict_accuracy"],
        "root_scope_strict_accuracy_after": new["root_scope_strict_accuracy"],
        "root_scope_coverage_before": old["root_scope_coverage"],
        "root_scope_coverage_after": new["root_scope_coverage"],
        "root_scope_corrected_sample_ids": corrected,
        "root_scope_harmed_sample_ids": harmed,
        "root_scope_corrected": len(corrected),
        "root_scope_harmed": len(harmed),
        "root_scope_net_corrected": len(corrected) - len(harmed),
        "root_scope_strict_accuracy_delta": (
            new["root_scope_strict_accuracy"]
            - old["root_scope_strict_accuracy"]),
        "technical_failure_count": new["technical_failure_count"],
    }


def paired_root_all_samples(
    before: Mapping[str, Any], after: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two root artifacts on every row using a +1/-1/0 ledger.

    ``None`` is a scientific wrong prediction. A parse or transport failure is
    represented as ``technical_failure`` and reported separately so callers can
    pause instead of turning infrastructure failure into scientific evidence.
    """
    if str(before.get("root_id")) != str(after.get("root_id")):
        raise ValueError("root candidate changed root identity")
    row_ids = [str(row["sample_id"]) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("rows contain duplicate sample IDs")
    by_id = {str(row["sample_id"]): row for row in rows}

    def predictions(value: Mapping[str, Any]) -> dict[str, str]:
        samples = {str(item["sample_id"]): item for item in value["samples"]}
        if (len(samples) != len(value["samples"])
                or set(samples) != set(by_id)):
            raise ValueError("root artifact sample identity does not match rows")
        return {
            sample_id: _answer(
                samples[sample_id]["call"], int(samples[sample_id]["order"]))
            for sample_id in row_ids
        }

    old_predictions = predictions(before)
    new_predictions = predictions(after)
    ledger = []
    corrected = []
    harmed = []
    unchanged = []
    none_to_correct = []
    correct_to_none = []
    technical = []
    for sample_id in row_ids:
        gold = str(by_id[sample_id]["answer"])
        old_prediction = old_predictions[sample_id]
        new_prediction = new_predictions[sample_id]
        old_correct = old_prediction == gold
        new_correct = new_prediction == gold
        gain = int(new_correct) - int(old_correct)
        if gain > 0:
            corrected.append(sample_id)
        elif gain < 0:
            harmed.append(sample_id)
        else:
            unchanged.append(sample_id)
        if old_prediction == "None" and new_correct:
            none_to_correct.append(sample_id)
        if old_correct and new_prediction == "None":
            correct_to_none.append(sample_id)
        if "technical_failure" in {old_prediction, new_prediction}:
            technical.append(sample_id)
        ledger.append({
            "sample_id": sample_id,
            "gold": gold,
            "before": old_prediction,
            "after": new_prediction,
            "before_correct": old_correct,
            "after_correct": new_correct,
            "gain": gain,
        })

    before_correct = sum(item["before_correct"] for item in ledger)
    after_correct = sum(item["after_correct"] for item in ledger)
    support = len(ledger)
    return {
        "root_id": str(before["root_id"]),
        "all_sample_support": support,
        "all_sample_ids": row_ids,
        "all_sample_corrected_sample_ids": corrected,
        "all_sample_harmed_sample_ids": harmed,
        "all_sample_unchanged_sample_ids": unchanged,
        "all_sample_corrected": len(corrected),
        "all_sample_harmed": len(harmed),
        "all_sample_unchanged": len(unchanged),
        "all_sample_net_gain": len(corrected) - len(harmed),
        "all_sample_strict_accuracy_before": (
            before_correct / support if support else 0.0),
        "all_sample_strict_accuracy_after": (
            after_correct / support if support else 0.0),
        "all_sample_strict_accuracy_delta": (
            (after_correct - before_correct) / support if support else 0.0),
        "none_to_correct_sample_ids": none_to_correct,
        "correct_to_none_sample_ids": correct_to_none,
        "technical_failure_sample_ids": technical,
        "technical_failure_count": len(technical),
        "technical_failure_count_before": sum(
            value == "technical_failure" for value in old_predictions.values()),
        "technical_failure_count_after": sum(
            value == "technical_failure" for value in new_predictions.values()),
        "gain_ledger": ledger,
        "gain_ledger_sha256": canonical_sha256(ledger),
    }


def root_from_system(
    value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], root_id: str,
    scope_sample_ids: Sequence[str], *, output_path: Path | None = None,
) -> dict[str, Any]:
    """Project a K=1 full-system artifact into a frozen root-only baseline."""
    if int(value.get("k", 0)) != 1:
        raise ValueError("root baseline projection requires a K=1 system artifact")
    samples = []
    for item in value["samples"]:
        replicate = item["replicates"]["0"]
        call = replicate["subtrees"].get(root_id)
        if not isinstance(call, Mapping):
            raise ValueError(f"system artifact is missing root report: {root_id}")
        samples.append({
            "sample_id": str(item["sample_id"]),
            "order": int(replicate["order"]),
            "call": dict(call),
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_kind": "root_unified_subtree",
        "split": str(value["split"]),
        "rubric_sha256": str(value["rubric_sha256"]),
        "root_id": root_id,
        "root_subtree_sha256": str(value["root_subtree_sha256"][root_id]),
        "scope_sample_ids": [str(item) for item in scope_sample_ids],
        "samples": samples,
        "wall_seconds": 0.0,
        "projected_from_full_system": True,
    }
    result["metrics"] = root_metrics(result, rows, scope_sample_ids)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path, result)
    return result


def evaluate_root(
    config: Mapping[str, Any], *, output_path: Path, cache_dir: Path,
    split_name: str, rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
    root_id: str, scope_sample_ids: Sequence[str], settings: RuntimeSettings,
    total_attempt_limit: int | None = None,
    endpoint_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly one Unified-Subtree; no Global-Arbiter call is made."""
    if root_id not in rubric.root_ids:
        raise ValueError("root-only evaluation target must be a rubric root")
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    selected_ids = ENDPOINT_IDS if endpoint_ids is None else tuple(endpoint_ids)
    endpoints = tuple(item for item in pool.endpoints
                      if item.endpoint_id in selected_ids)
    if not selected_ids or tuple(item.endpoint_id for item in endpoints) != selected_ids:
        raise RuntimeError(f"root runtime endpoint order must match {selected_ids}")
    if endpoint_ids is not None and sum(e.max_concurrency for e in endpoints) > pool.global_request_concurrency:
        raise ValueError("selected endpoint slots exceed global request concurrency")
    attempts = total_attempt_limit or 1 + settings.max_parse_retries
    pending: queue.Queue[Mapping[str, Any]] = queue.Queue()
    for row in rows:
        pending.put(row)
    values: dict[str, Any] = {}
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()
    print(f"{split_name}: 0/{len(rows)} root reports started", flush=True)

    def worker(endpoint: BackendEndpointSpec) -> None:
        nonlocal completed
        while True:
            try:
                row = pending.get_nowait()
            except queue.Empty:
                return
            sample_id = str(row["sample_id"])
            call = _call_subtree(
                config, endpoint, cache_dir, split_name, row, rubric, root_id,
                0, 0, attempts, settings)
            with lock:
                values[sample_id] = {
                    "sample_id": sample_id, "order": 0,
                    "call": support.compact_call(call),
                }
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(rows) - completed)
                print(
                    f"{split_name}: {completed}/{len(rows)} sample={sample_id} "
                    f"endpoint={endpoint.endpoint_id} elapsed={elapsed/60:.1f}m "
                    f"ETA={eta/60:.1f}m", flush=True)

    workers = _interleaved_workers(endpoints)
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_kind": "root_unified_subtree",
        "split": split_name,
        "rubric_sha256": rubric.rubric_sha256,
        "root_id": root_id,
        "root_subtree_sha256": _root_subtree_sha256(rubric, root_id),
        "scope_sample_ids": [str(item) for item in scope_sample_ids],
        "samples": [values[str(row["sample_id"])] for row in rows],
        "wall_seconds": time.perf_counter() - started,
        "projected_from_full_system": False,
    }
    result["metrics"] = root_metrics(result, rows, scope_sample_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    return result


def attributed_error_ids(
    value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    gold = {str(row["sample_id"]): str(row["answer"]) for row in rows}
    result: dict[str, list[str]] = {}
    for sample in value["samples"]:
        sample_id = str(sample["sample_id"])
        item = sample["replicates"]["0"]
        final = _answer(item["arbiter"], int(item["order"]))
        if final == gold[sample_id]:
            continue
        for root_id, call in item["subtrees"].items():
            answer = ((call.get("parsed") or {}).get("answer")
                      if call.get("parse_ok") else None)
            if answer != gold[sample_id]:
                result.setdefault(root_id, []).append(sample_id)
    return {root_id: tuple(values) for root_id, values in result.items()}


def evaluate(
    config: Mapping[str, Any], *, output_path: Path, cache_dir: Path,
    split_name: str, rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
    settings: RuntimeSettings, orders_by_id: Mapping[str, Sequence[int]] | None = None,
    baseline: Mapping[str, Any] | None = None,
    changed_root_ids: Sequence[str] | None = None,
    total_attempt_limit: int | None = None,
    endpoint_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run or resume a system evaluation and persist a replayable artifact."""
    changed = None if changed_root_ids is None else frozenset(changed_root_ids)
    if changed is not None and not changed.issubset(rubric.root_ids):
        raise ValueError("changed roots must belong to the candidate rubric")
    old = _baseline_index(baseline)
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    selected_ids = ENDPOINT_IDS if endpoint_ids is None else tuple(endpoint_ids)
    endpoints = tuple(item for item in pool.endpoints
                      if item.endpoint_id in selected_ids)
    if not selected_ids or tuple(item.endpoint_id for item in endpoints) != selected_ids:
        raise RuntimeError(f"aligned runtime endpoint order must match {selected_ids}")
    if endpoint_ids is not None and sum(e.max_concurrency for e in endpoints) > pool.global_request_concurrency:
        raise ValueError("selected endpoint slots exceed global request concurrency")
    attempts = total_attempt_limit or 1 + settings.max_parse_retries
    pending: queue.Queue[Mapping[str, Any]] = queue.Queue()
    for row in rows:
        pending.put(row)
    values: dict[str, Any] = {}
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()
    print(f"{split_name}: 0/{len(rows)} system bundles started", flush=True)

    def worker(endpoint: BackendEndpointSpec) -> None:
        nonlocal completed
        while True:
            try:
                row = pending.get_nowait()
            except queue.Empty:
                return
            sample_id = str(row["sample_id"])
            orders = ((0,) if orders_by_id is None
                      else tuple(int(item) for item in orders_by_id[sample_id]))
            value = _one_sample(
                config, endpoint, cache_dir, split_name, row, orders, rubric,
                old.get(sample_id), changed, attempts, settings)
            with lock:
                values[sample_id] = value
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(rows) - completed)
                print(
                    f"{split_name}: {completed}/{len(rows)} sample={sample_id} "
                    f"endpoint={endpoint.endpoint_id} elapsed={elapsed/60:.1f}m "
                    f"ETA={eta/60:.1f}m", flush=True)

    workers = _interleaved_workers(endpoints)
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    k_values = {len((0,) if orders_by_id is None else orders_by_id[str(row["sample_id"])])
                for row in rows}
    if len(k_values) != 1:
        raise ValueError("all samples must use the same replicate count")
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "split": split_name,
        "rubric_sha256": rubric.rubric_sha256,
        "root_subtree_sha256": {
            root_id: _root_subtree_sha256(rubric, root_id)
            for root_id in rubric.root_ids},
        "changed_root_ids": None if changed is None else sorted(changed),
        "k": next(iter(k_values)),
        "samples": [values[str(row["sample_id"])] for row in rows],
        "wall_seconds": time.perf_counter() - started,
    }
    result["metrics"] = metrics(result, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    return result


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"aligned system artifact identity drift: {path}")
    return value

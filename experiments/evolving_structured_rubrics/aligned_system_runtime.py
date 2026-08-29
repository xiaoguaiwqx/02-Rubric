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


@dataclass(frozen=True)
class RuntimeSettings:
    temperature: float
    max_tokens: int
    max_parse_retries: int
    generation_seed_policy: str = "unset"

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
        protocol_version=PROTOCOL_VERSION,
        prompt_version=unified.ARBITER_PROMPT_VERSION,
        system_prompt=arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        response_parser=arbiter.parse_global_arbiter_ab_only_response,
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
) -> dict[str, Any]:
    """Run or resume a system evaluation and persist a replayable artifact."""
    changed = None if changed_root_ids is None else frozenset(changed_root_ids)
    if changed is not None and not changed.issubset(rubric.root_ids):
        raise ValueError("changed roots must belong to the candidate rubric")
    old = _baseline_index(baseline)
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoints = tuple(item for item in pool.endpoints
                      if item.endpoint_id in ENDPOINT_IDS)
    if tuple(item.endpoint_id for item in endpoints) != ENDPOINT_IDS:
        raise RuntimeError("aligned runtime requires vllm-8000 and vllm-8001")
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

    workers = [endpoint for endpoint in endpoints
               for _ in range(endpoint.max_concurrency)]
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

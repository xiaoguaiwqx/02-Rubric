"""Cost-bounded Discovery100 demo built from Coverage150 and Hard90."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured.backend_pool import AvailableSlotBackendPool, BackendPoolSpec
from critiq.structured.telemetry import ModelCallMetrics, combine_model_call_metrics

from . import discovery_data_v2 as dv2
from . import discovery_data_v2_adjudication as adv2
from .experiment_utils import atomic_write_json, load_json, make_progress_callback
from .rubric_factory import file_sha256


PROTOCOL_VERSION = "discovery100-demo-single-order-balanced-v3"
SELECTION_VERSION = "discovery100-demo-coverage70-hard30-v1"
ORDER_SCHEDULE_VERSION = "single-order-balanced-seed42-v1"
DIRECTORY = "discovery100_demo_single_order_v3"
COVERAGE_QUOTAS = {
    "rlhf_v": 12,
    "mm_rlhf": 12,
    "vilreward_73k": 11,
    "mmpr_v1_2": 12,
    "vision_arena_battle": 11,
    "mm_ifdpo": 12,
}
HARD_PER_SOURCE = 5
EXPECTED_DOMAIN_COUNTS = {"visual": 34, "reasoning": 33, "general": 33}
DATA_EXPORT_DIRECTORY = "discovery_v2_demo_v3"
QUALITY_REPLACEMENTS = {
    "dv2-mm_rlhf-bad422af07f7d2dc": {
        "replacement_sample_id": "dv2-mm_rlhf-461af81926c8275c",
        "pool_role": "coverage",
        "reason": "mojibake_question_and_answers_with_narrow_text_strip",
    },
    "dv2-mm_rlhf-942b229986f17fb8": {
        "replacement_sample_id": "dv2-mm_rlhf-0663dc3c9a157222",
        "pool_role": "hard",
        "reason": "overly_narrow_low_diversity_line_diagram",
    },
}


def _paths(target: Path) -> tuple[Path, Path]:
    return (target / "selection_v2/coverage_candidates_150.jsonl",
            target / "selection_v2/hard_candidates_90.jsonl")


def _select_coverage(rows: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    selected = []
    for source, count in COVERAGE_QUOTAS.items():
        values = [row for row in rows if row.get("source") == source]
        if len(values) != 25:
            raise RuntimeError(f"Coverage source {source} must contain 25 candidates")
        ordered = dv2._metadata_selection_order(values, source=source, seed=seed)
        selected.extend({**row, "demo_pool_role": "coverage"} for row in ordered[:count])
    if len(selected) != 70:
        raise RuntimeError("Discovery100 demo Coverage selection must contain 70 rows")
    return selected


def _select_hard(rows: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    selected = []
    for source in COVERAGE_QUOTAS:
        values = [row for row in rows if row.get("source") == source]
        if len(values) != 15:
            raise RuntimeError(f"Hard source {source} must contain 15 candidates")
        for row in values:
            screen = row.get("generic_screen")
            if not isinstance(screen, Mapping) or screen.get("hardness") not in {0, 1, 2, 3}:
                raise RuntimeError(f"Hard candidate {row.get('sample_id')} lacks Generic hardness")
        values.sort(key=lambda row: (
            -int(row["generic_screen"]["hardness"]),
            dv2._sha([SELECTION_VERSION, seed, source, row["sample_id"]])))
        selected.extend({**row, "demo_pool_role": "hard"}
                        for row in values[:HARD_PER_SOURCE])
    if len(selected) != 30:
        raise RuntimeError("Discovery100 demo Hard selection must contain 30 rows")
    return selected


def select_demo_rows(coverage: Sequence[dict[str, Any]],
                     hard: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rows = _select_coverage(coverage, seed=seed) + _select_hard(hard, seed=seed)
    original_order_plan = _single_order_plan(rows, seed=seed)
    candidates = {row["sample_id"]: row for row in (*coverage, *hard)}
    replaced_orders: dict[str, str] = {}
    updated = []
    for row in rows:
        replacement = QUALITY_REPLACEMENTS.get(row["sample_id"])
        if replacement is None:
            updated.append(row)
            continue
        replacement_id = replacement["replacement_sample_id"]
        candidate = candidates.get(replacement_id)
        if candidate is None:
            raise RuntimeError(f"Discovery100 quality replacement is missing: {replacement_id}")
        if candidate.get("source") != row.get("source"):
            raise RuntimeError("Discovery100 quality replacement source drift")
        pool_role = replacement["pool_role"]
        if row["demo_pool_role"] != pool_role:
            raise RuntimeError("Discovery100 quality replacement pool-role drift")
        updated.append({**candidate, "demo_pool_role": pool_role,
            "quality_replacement": {
                "replaces_sample_id": row["sample_id"],
                "reason": replacement["reason"],
                "selection_uses_397b_result": False,
            }})
        replaced_orders[replacement_id] = original_order_plan[row["sample_id"]]
    rows = updated
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Discovery100 demo Coverage and Hard selections overlap")
    source_counts = Counter(row["source"] for row in rows)
    expected_sources = Counter({source: count + HARD_PER_SOURCE
                                for source, count in COVERAGE_QUOTAS.items()})
    if source_counts != expected_sources:
        raise RuntimeError(f"Discovery100 demo source quotas drift: {dict(source_counts)}")
    domain_counts = Counter(row["domain"] for row in rows)
    if domain_counts != Counter(EXPECTED_DOMAIN_COUNTS):
        raise RuntimeError(f"Discovery100 demo domain quotas drift: {dict(domain_counts)}")
    order_plan = _single_order_plan(rows, seed=seed)
    # Preserve the already frozen order of all retained v2 samples.  A quality
    # replacement inherits the removed slot's order, so the other 98 cached
    # model judgments remain byte-identical and the schedule stays 50/50.
    order_plan.update({row["sample_id"]: original_order_plan[row["sample_id"]]
                       for row in rows if row["sample_id"] in original_order_plan})
    order_plan.update(replaced_orders)
    if Counter(order_plan.values()) != {"original": 50, "swapped": 50}:
        raise RuntimeError("Discovery100 replacement order schedule is not balanced")
    return [{**row, "preadjudication_assigned_order": order_plan[row["sample_id"]]}
            for row in rows]


def _single_order_plan(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, str]:
    """Assign exactly half of the frozen samples to each presentation order."""

    ordered = sorted(rows, key=lambda row: dv2._sha([
        ORDER_SCHEDULE_VERSION, seed, row["sample_id"]]))
    swapped_count = len(ordered) // 2
    return {row["sample_id"]: ("swapped" if index < swapped_count else "original")
            for index, row in enumerate(ordered)}


def freeze(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = dv2._config(config), dv2._target(output)
    dv2._require(target, "discovery-v2-selection-freeze")
    dv2._require(target, "discovery-v2-generic-screen")
    coverage_path, hard_path = _paths(target)
    rows = select_demo_rows(dv2._read_jsonl(coverage_path), dv2._read_jsonl(hard_path),
                            seed=int(cfg["seed"]))
    directory = target / DIRECTORY
    rows_path = directory / "frozen_discovery100.jsonl"
    manifest_path = directory / "frozen_manifest.json"
    if rows_path.is_file() and dv2._read_jsonl(rows_path) != rows:
        raise RuntimeError("Discovery100 demo frozen rows drift")
    if not rows_path.is_file():
        dv2._write_jsonl(rows_path, rows)
    manifest = {
        "schema_version": dv2.SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(cfg["seed"]),
        "sample_count": 100, "logical_order_count": 100,
        "coverage_count": 70, "hard_count": 30,
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "order_schedule": {
            "version": ORDER_SCHEDULE_VERSION,
            "orders_per_sample": 1,
            "assignment_uses_source_gold": False,
            "counts": dict(sorted(Counter(
                row["preadjudication_assigned_order"] for row in rows).items())),
            "displayed_source_gold_counts": dict(sorted(Counter(
                (adv2._map_swapped_answer(row["answer"])
                 if row["preadjudication_assigned_order"] == "swapped"
                 else row["answer"]) for row in rows).items())),
        },
        "source_artifacts": {
            "coverage150": {"path": str(coverage_path.resolve()),
                "count": 150, "sha256": file_sha256(coverage_path)},
            "hard90": {"path": str(hard_path.resolve()),
                "count": 90, "sha256": file_sha256(hard_path)},
        },
        "frozen_rows": {"path": str(rows_path.resolve()), "count": 100,
                        "sha256": file_sha256(rows_path)},
        "selection": {
            "coverage": "metadata-only source-stratified deterministic selection",
            "hard": "top Generic hardness per source with seed-42 hash tie break",
            "397b_used_for_selection": False,
            "coverage_source_gold_used": False,
            "hard_uses_precomputed_generic_hardness": True,
            "generic_hardness_depends_on_source_gold": True,
            "quality_replacements": [
                {"removed_sample_id": removed, **replacement}
                for removed, replacement in QUALITY_REPLACEMENTS.items()
            ],
            "quality_replacement_uses_397b_result": False,
        },
        # This is the identity of one model call. The demo-level schedule above
        # decides that exactly one of the two possible orders is submitted.
        # Keeping the per-call identity unchanged permits safe reuse of an
        # already successful identical order from the full protocol.
        "per_order_model_request_spec": adv2._request_spec(config, cfg),
    }
    if manifest_path.is_file() and load_json(manifest_path) != manifest:
        raise RuntimeError("Discovery100 demo frozen manifest drift")
    if not manifest_path.is_file():
        atomic_write_json(manifest_path, manifest)
    dv2._set_status(target, "discovery-v2-demo-freeze", "passed", {
        "sample_count": 100, "coverage_count": 70, "hard_count": 30,
        "logical_order_count": 100})
    print(json.dumps({key: manifest[key] for key in (
        "sample_count", "coverage_count", "hard_count", "source_counts",
        "domain_counts", "order_schedule", "logical_order_count")}, indent=2))


def _load_frozen(config: Mapping[str, Any], output: Path) -> tuple[
        dict[str, Any], Path, dict[str, Any], list[dict[str, Any]]]:
    cfg, target = dv2._config(config), dv2._target(output)
    dv2._require(target, "discovery-v2-demo-freeze")
    manifest = load_json(target / DIRECTORY / "frozen_manifest.json")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Discovery100 demo protocol drift")
    if manifest.get("per_order_model_request_spec") != adv2._request_spec(config, cfg):
        raise RuntimeError("Discovery100 demo request identity drift")
    coverage_path, hard_path = _paths(target)
    for label, path in (("coverage150", coverage_path), ("hard90", hard_path)):
        if file_sha256(path) != manifest["source_artifacts"][label]["sha256"]:
            raise RuntimeError(f"Discovery100 demo source drift: {label}")
    rows_path = target / DIRECTORY / "frozen_discovery100.jsonl"
    if file_sha256(rows_path) != manifest["frozen_rows"]["sha256"]:
        raise RuntimeError("Discovery100 demo frozen row hash drift")
    rows = dv2._read_jsonl(rows_path)
    if len(rows) != 100:
        raise RuntimeError("Discovery100 demo frozen row count drift")
    if Counter(row.get("preadjudication_assigned_order") for row in rows) != {
            "original": 50, "swapped": 50}:
        raise RuntimeError("Discovery100 demo single-order schedule drift")
    return cfg, target, manifest, rows


def _smoke_rows(rows: Sequence[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for source_index, source in enumerate(COVERAGE_QUOTAS):
        for role_index, role in enumerate(("coverage", "hard")):
            values = [row for row in rows
                      if row["source"] == source and row["demo_pool_role"] == role]
            values.sort(key=lambda row: dv2._sha([
                PROTOCOL_VERSION, cfg["seed"], "smoke", source, role, row["sample_id"]]))
            if not values:
                raise RuntimeError(f"Discovery100 demo smoke lacks {source}/{role}")
            preferred = "original" if (source_index + role_index) % 2 == 0 else "swapped"
            matching = [row for row in values
                        if row["preadjudication_assigned_order"] == preferred]
            selected.append((matching or values)[0])
    if len(selected) != 12:
        raise RuntimeError("Discovery100 demo smoke must contain 12 rows")
    if set(row["preadjudication_assigned_order"] for row in selected) != {
            "original", "swapped"}:
        raise RuntimeError("Discovery100 demo smoke must exercise both orders")
    return selected


def _run_single_orders(config: Mapping[str, Any], cfg: Mapping[str, Any], target: Path,
                       rows: Sequence[Mapping[str, Any]], *, label: str) -> tuple[
                           list[dict[str, Any]], dict[str, Any]]:
    """Run exactly the one frozen presentation order assigned to each sample."""

    request_spec = adv2._request_spec(config, cfg)
    pool_spec = BackendPoolSpec.from_dict(cfg["adjudicator"]["backend_pool"])
    pool = AvailableSlotBackendPool(pool_spec)
    total = len(rows)
    work = target / DIRECTORY / label
    callback = make_progress_callback(work, label, total, pool)
    results: list[dict[str, Any] | None] = [None] * total
    metrics: list[ModelCallMetrics] = []
    cache_dir = target / "adjudication_v2/cache/orders"
    concurrency = min(pool_spec.global_request_concurrency,
                      adv2.EFFECTIVE_REQUEST_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for index, row in enumerate(rows):
            order = row["preadjudication_assigned_order"]
            if order not in {"original", "swapped"}:
                raise RuntimeError(f"invalid frozen order for {row['sample_id']}")
            future = executor.submit(
                adv2._order_cached, config, cfg, request_spec, row,
                swapped=order == "swapped", pool=pool, cache_dir=cache_dir)
            futures[future] = (index, order)
        for completed, future in enumerate(as_completed(futures), 1):
            index, order = futures[future]
            result, item_metrics = future.result()
            results[index] = {"sample_id": rows[index]["sample_id"],
                              "assigned_order": order, "result": result}
            metrics.append(item_metrics)
            callback(completed - 1, f"{rows[index]['sample_id']}::{order}", item_metrics)
    if any(item is None for item in results):
        raise RuntimeError("Discovery100 demo lost a scheduled result")
    completed_results = [item for item in results if item is not None]
    unresolved = [f"{item['sample_id']}::{item['assigned_order']}"
                  for item in completed_results if not item["result"]["parse_ok"]]
    combined = combine_model_call_metrics(metrics)
    summary = {
        "schema_version": dv2.SCHEMA_VERSION,
        "sample_count": total, "logical_order_count": total,
        "orders_per_sample": 1,
        "assigned_order_counts": dict(sorted(Counter(
            item["assigned_order"] for item in completed_results).items())),
        "parse_valid_count": total - len(unresolved),
        "parse_valid_rate": (total - len(unresolved)) / total if total else 1.0,
        "unresolved_sample_orders": unresolved,
        "current_run_metrics": combined.to_dict(),
        "endpoint_call_counts": dict(pool.records_by_endpoint()),
        "execution_concurrency": {
            "configured_capacity": pool_spec.global_request_concurrency,
            "effective_request_concurrency": concurrency,
            "reason": "provider_397b_busy_safety_cap",
        },
        "per_order_model_request_spec": request_spec,
    }
    dv2._write_jsonl(work / "judgments.jsonl", completed_results)
    atomic_write_json(work / "summary.json", summary)
    atomic_write_json(work / "provenance.json", pool.provenance_dict())
    return completed_results, summary


def _merge_single_order(row: Mapping[str, Any],
                        judgment: Mapping[str, Any]) -> dict[str, Any]:
    order = judgment["assigned_order"]
    parsed = judgment["result"]["parsed"]
    if parsed is None:
        raise ValueError("cannot merge unresolved single-order pre-adjudication")
    displayed_answer = parsed["answer"]
    mapped = (adv2._map_swapped_answer(displayed_answer)
              if order == "swapped" else displayed_answer)
    suggested = mapped if mapped in {"A", "B"} else None
    source_agreement = suggested == row["answer"] if suggested is not None else None
    requires_reconciliation = (
        suggested is None or source_agreement is False or parsed["confidence"] < 3)
    return {**row, "preadjudication": {
        "protocol_version": PROTOCOL_VERSION,
        "assigned_order": order,
        "parsed": parsed,
        "displayed_answer": displayed_answer,
        "suggested_answer": suggested,
        "confidence": parsed["confidence"],
        "source_answer_agreement": source_agreement,
        "requires_reconciliation": requires_reconciliation,
        "reconciliation_reasons": [reason for reason, active in (
            ("uncertain", suggested is None),
            ("source_disagreement", source_agreement is False),
            ("low_confidence", parsed["confidence"] < 3),
        ) if active],
    }}


def smoke(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, rows = _load_frozen(config, output)
    selected = _smoke_rows(rows, cfg)
    _, summary = _run_single_orders(config, cfg, target, selected, label="smoke")
    result = {**summary, "expected_sample_count": 12,
              "expected_logical_order_count": 12,
              "source_counts": dict(Counter(row["source"] for row in selected)),
              "pool_counts": dict(Counter(row["demo_pool_role"] for row in selected))}
    atomic_write_json(target / DIRECTORY / "smoke_report.json", result)
    if summary["unresolved_sample_orders"]:
        raise RuntimeError(
            f"Discovery100 demo smoke has {len(summary['unresolved_sample_orders'])} unresolved orders")
    dv2._set_status(target, "discovery-v2-demo-smoke", "passed", {
        "sample_count": 12, "logical_order_count": 12, "parse_valid_rate": 1.0})
    print(json.dumps({"sample_count": 12, "logical_order_count": 12,
                      "parse_valid_rate": 1.0,
                      "assigned_order_counts": summary["assigned_order_counts"],
                      "endpoint_call_counts": summary["endpoint_call_counts"]}, indent=2))


def adjudicate(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, rows = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-demo-smoke")
    judgments, summary = _run_single_orders(config, cfg, target, rows, label="full")
    atomic_write_json(target / DIRECTORY / "adjudication_run_report.json", {
        **summary, "smoke_cache_reuse_possible": 12,
        "maximum_new_requests_after_smoke": 88})
    if summary["unresolved_sample_orders"]:
        dv2._set_status(target, "discovery-v2-demo-adjudicate", "incomplete", {
            "unresolved_count": len(summary["unresolved_sample_orders"]),
            "parse_valid_rate": summary["parse_valid_rate"],
            "rerun_stage": "discovery-v2-demo-adjudicate"})
        print(json.dumps({"status": "incomplete",
            "unresolved_count": len(summary["unresolved_sample_orders"]),
            "parse_valid_rate": summary["parse_valid_rate"],
            "rerun_stage": "discovery-v2-demo-adjudicate"}, indent=2))
        return
    by_id = {item["sample_id"]: item for item in judgments}
    records = [_merge_single_order(row, by_id[row["sample_id"]]) for row in rows]
    records_path = target / DIRECTORY / "preadjudicated_discovery100.jsonl"
    dv2._write_jsonl(records_path, records)
    dv2._set_status(target, "discovery-v2-demo-adjudicate", "passed", {
        "sample_count": 100, "logical_order_count": 100,
        "parse_valid_rate": 1.0, "records_sha256": file_sha256(records_path)})
    print(json.dumps({"status": "passed", "sample_count": 100,
                      "logical_order_count": 100, "parse_valid_rate": 1.0}, indent=2))


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    suggested = [row for row in rows
                 if row["preadjudication"]["suggested_answer"] is not None]
    return {
        "count": len(rows),
        "suggested_answer_rate": len(suggested) / len(rows),
        "source_agreement_rate_on_suggested": (
            sum(row["preadjudication"]["source_answer_agreement"] for row in suggested)
            / len(suggested) if suggested else None),
        "reconciliation_rate": sum(
            row["preadjudication"]["requires_reconciliation"] for row in rows) / len(rows),
        "mean_confidence": sum(
            row["preadjudication"]["confidence"] for row in rows) / len(rows),
        "uncertain_rate": sum(
            row["preadjudication"]["suggested_answer"] is None for row in rows) / len(rows),
        "assigned_order_counts": dict(sorted(Counter(
            row["preadjudication"]["assigned_order"] for row in rows).items())),
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, _ = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-demo-adjudicate")
    path = target / DIRECTORY / "preadjudicated_discovery100.jsonl"
    rows = dv2._read_jsonl(path)
    if len(rows) != 100:
        raise RuntimeError("Discovery100 demo report requires 100 complete records")
    result = {
        "schema_version": dv2.SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "sample_count": 100, "logical_order_count": 100,
        "orders_per_sample": 1,
        "overall": _group(rows),
        "by_pool": {role: _group([row for row in rows
                                  if row["demo_pool_role"] == role])
                    for role in ("coverage", "hard")},
        "by_source": {source: _group([row for row in rows if row["source"] == source])
                      for source in COVERAGE_QUOTAS},
        "by_domain": {domain: _group([row for row in rows if row["domain"] == domain])
                      for domain in EXPECTED_DOMAIN_COUNTS},
        "human_review_required_for_final_gold": True,
        "records_sha256": file_sha256(path),
    }
    atomic_write_json(target / DIRECTORY / "report.json", result)
    dv2._set_status(target, "discovery-v2-demo-report", "passed", {
        "sample_count": 100,
        "source_agreement_rate_on_suggested": result["overall"][
            "source_agreement_rate_on_suggested"],
        "reconciliation_rate": result["overall"]["reconciliation_rate"]})
    print(json.dumps({"overall": result["overall"],
                      "by_pool": result["by_pool"]}, indent=2))


def _original_human_answer(item: Mapping[str, Any], swapped: bool) -> str | None:
    answer = item.get("human_review", {}).get("answer")
    if answer not in {"A", "B"}:
        return None
    return adv2._map_swapped_answer(answer) if swapped else answer


def review_export(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, _ = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-demo-report")
    rows = dv2._read_jsonl(target / DIRECTORY / "preadjudicated_discovery100.jsonl")
    review_dir = target / DIRECTORY / "review"
    queue_path = review_dir / "blind_review_queue.jsonl"
    hidden_path = review_dir / "hidden_reference.jsonl"
    existing = {row["review_id"]: row for row in dv2._read_jsonl(queue_path)} \
        if queue_path.is_file() else {}
    display = adv2._display_plan(rows, int(cfg["seed"]))
    ordered = sorted(rows, key=lambda row: dv2._sha([
        PROTOCOL_VERSION, cfg["seed"], "queue", row["sample_id"]]))
    queue, hidden = [], []
    for rank, row in enumerate(ordered, 1):
        review_id = f"discovery100-demo-{rank:03d}"
        swapped = display[row["sample_id"]]
        queue.append({
            "review_id": review_id, "queue_rank": rank,
            "image_path": row["image_path"], "question": row["question"],
            "A": row["B"] if swapped else row["A"],
            "B": row["A"] if swapped else row["B"],
            "human_review": existing.get(review_id, {}).get(
                "human_review", adv2._blank_human_review()),
        })
        hidden.append({
            "review_id": review_id, "sample_id": row["sample_id"],
            "display_swapped": swapped, "source": row["source"],
            "source_answer": row["answer"], "domain": row["domain"],
            "demo_pool_role": row["demo_pool_role"],
            "generic_screen": row.get("generic_screen"),
            "preadjudication": row["preadjudication"],
        })
    dv2._write_jsonl(queue_path, queue)
    dv2._write_jsonl(hidden_path, hidden)
    hidden_by_id = {row["review_id"]: row for row in hidden}
    reconciliation = []
    reviewed = 0
    for item in queue:
        if item["human_review"].get("reviewed") is not True:
            continue
        reviewed += 1
        reference = hidden_by_id[item["review_id"]]
        answer = _original_human_answer(item, reference["display_swapped"])
        pre = reference["preadjudication"]
        if (answer != reference["source_answer"] or answer != pre["suggested_answer"]
                or pre["requires_reconciliation"]):
            reconciliation.append({**item, "hidden_reference": reference,
                "human_answer_original_order": answer})
    reconciliation_path = review_dir / "reconciliation_queue.jsonl"
    dv2._write_jsonl(reconciliation_path, reconciliation)
    result = {
        "schema_version": dv2.SCHEMA_VERSION,
        "blind_queue_count": 100,
        "display_original_count": 50, "display_swapped_count": 50,
        "reviewed_count": reviewed,
        "reconciliation_queue_count": len(reconciliation),
        "first_pass_hidden_fields": [
            "source", "source_answer", "domain", "demo_pool_role",
            "generic_screen", "preadjudication"],
        "queue_sha256": file_sha256(queue_path),
        "hidden_reference_sha256": file_sha256(hidden_path),
    }
    atomic_write_json(review_dir / "export_report.json", result)
    dv2._set_status(target, "discovery-v2-demo-review-export", "passed", {
        "blind_queue_count": 100, "reviewed_count": reviewed,
        "reconciliation_queue_count": len(reconciliation)})
    print(json.dumps({key: result[key] for key in (
        "blind_queue_count", "display_original_count", "display_swapped_count",
        "reviewed_count", "reconciliation_queue_count")}, indent=2))


def _validate_human_review(item: Mapping[str, Any]) -> None:
    value = item.get("human_review")
    if not isinstance(value, Mapping) or value.get("reviewed") is not True:
        raise RuntimeError(f"review {item.get('review_id')} is incomplete")
    if value.get("decision") not in {"accept", "corrected", "reject"}:
        raise RuntimeError(f"review {item.get('review_id')} has invalid decision")
    if value["decision"] == "reject":
        return
    if value.get("answer") not in {"A", "B"}:
        raise RuntimeError(f"review {item.get('review_id')} lacks A/B answer")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or confidence < 3:
        raise RuntimeError(f"review {item.get('review_id')} confidence must be 3 or 4")
    if not isinstance(value.get("preference_rationale"), str) or not value["preference_rationale"].strip():
        raise RuntimeError(f"review {item.get('review_id')} lacks rationale")
    if value.get("task_type") not in dv2.PREADJUDICATION_TASK_TYPES:
        raise RuntimeError(f"review {item.get('review_id')} has invalid task_type")
    if value.get("evidence_type") not in dv2.PREADJUDICATION_EVIDENCE_TYPES:
        raise RuntimeError(f"review {item.get('review_id')} has invalid evidence_type")
    dimensions = value.get("preference_dimensions")
    if (not isinstance(dimensions, list) or not dimensions
            or not set(dimensions).issubset(dv2.PREADJUDICATION_DIMENSIONS)):
        raise RuntimeError(f"review {item.get('review_id')} has invalid preference_dimensions")
    ambiguity = value.get("ambiguity_flags")
    if (not isinstance(ambiguity, list)
            or not set(ambiguity).issubset(dv2.PREADJUDICATION_AMBIGUITY_FLAGS)):
        raise RuntimeError(f"review {item.get('review_id')} has invalid ambiguity_flags")


def _orient_final(row: Mapping[str, Any], answer: str, *, target_answer: str,
                  review: Mapping[str, Any], display_swapped: bool) -> dict[str, Any]:
    swap = answer != target_answer
    display_a_issues = list(review.get("candidate_a_issues", []))
    display_b_issues = list(review.get("candidate_b_issues", []))
    original_a_issues, original_b_issues = (
        (display_b_issues, display_a_issues) if display_swapped
        else (display_a_issues, display_b_issues))
    final_a_issues, final_b_issues = (
        (original_b_issues, original_a_issues) if swap
        else (original_a_issues, original_b_issues))
    return {
        "sample_id": row["sample_id"], "image_path": row["image_path"],
        "question": row["question"],
        "A": row["B"] if swap else row["A"],
        "B": row["A"] if swap else row["B"],
        "answer": target_answer,
        "split_role": "evolve", "domain": row["domain"],
        "subdomain": row["subdomain"], "source": row["source"],
        "source_family": row["source_family"],
        "source_sample_id": row["source_sample_id"],
        "demo_pool_role": row["demo_pool_role"],
        "label_origin": "human_reviewed",
        "preference_confidence": review["confidence"],
        "preference_rationale": review["preference_rationale"],
        "visual_evidence": review.get("visual_evidence", []),
        "task_type": review.get("task_type"),
        "preference_dimensions": review.get("preference_dimensions", []),
        "evidence_type": review.get("evidence_type"),
        "candidate_a_issues": final_a_issues,
        "candidate_b_issues": final_b_issues,
        "ambiguity_flags": review.get("ambiguity_flags", []),
        "image_sha256": row["image_sha256"],
        "question_sha256": row["question_sha256"],
        "unordered_pair_sha256": row["unordered_pair_sha256"],
        "human_review": {"decision": review["decision"], "reviewed": True,
                         "reconciled": bool(review.get("reconciled"))},
    }


def finalize(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, frozen = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-demo-review-export")
    queue_path = target / DIRECTORY / "review/blind_review_queue.jsonl"
    hidden_path = target / DIRECTORY / "review/hidden_reference.jsonl"
    queue = dv2._read_jsonl(queue_path)
    hidden = {item["review_id"]: item for item in dv2._read_jsonl(hidden_path)}
    if len(queue) != 100 or len(hidden) != 100:
        raise RuntimeError("Discovery100 demo finalize requires 100 review records")
    frozen_by_id = {row["sample_id"]: row for row in frozen}
    reviewed = []
    rejected = []
    for item in queue:
        _validate_human_review(item)
        reference = hidden[item["review_id"]]
        if item["human_review"]["decision"] == "reject":
            rejected.append(item["review_id"])
            continue
        answer = _original_human_answer(item, reference["display_swapped"])
        expected_decision = ("corrected" if answer != reference["source_answer"]
                             else "accept")
        if item["human_review"]["decision"] != expected_decision:
            raise RuntimeError(
                f"review {item['review_id']} decision must be {expected_decision}")
        pre = reference["preadjudication"]
        needs_reconciliation = (
            answer != reference["source_answer"]
            or answer != pre["suggested_answer"]
            or pre["requires_reconciliation"])
        if needs_reconciliation and item["human_review"].get("reconciled") is not True:
            raise RuntimeError(
                f"review {item['review_id']} requires completed reconciliation")
        reviewed.append((frozen_by_id[reference["sample_id"]], answer,
                         item["human_review"], bool(reference["display_swapped"])))
    if rejected:
        atomic_write_json(target / DIRECTORY / "finalize_blocked.json", {
            "schema_version": dv2.SCHEMA_VERSION,
            "reason": "rejected records require deterministic replacements",
            "rejected_count": len(rejected), "review_ids": rejected})
        raise RuntimeError(
            f"Discovery100 demo has {len(rejected)} rejected records; replacements are required")
    if len(reviewed) != 100:
        raise RuntimeError("Discovery100 demo requires 100 accepted/corrected reviews")
    ordered = sorted(reviewed, key=lambda item: dv2._sha([
        PROTOCOL_VERSION, cfg["seed"], "final-position", item[0]["sample_id"]]))
    final_rows = [_orient_final(row, answer, target_answer="A" if index < 50 else "B",
                                review=review, display_swapped=display_swapped)
                  for index, (row, answer, review, display_swapped) in enumerate(ordered)]
    final_dir = target / DIRECTORY / "final"
    final_path = final_dir / "discovery_100.jsonl"
    dv2._write_jsonl(final_path, final_rows)
    result = {
        "schema_version": dv2.SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "sample_count": 100,
        "coverage_count": sum(row["demo_pool_role"] == "coverage" for row in final_rows),
        "hard_count": sum(row["demo_pool_role"] == "hard" for row in final_rows),
        "gold_position_counts": dict(Counter(row["answer"] for row in final_rows)),
        "source_counts": dict(sorted(Counter(row["source"] for row in final_rows).items())),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in final_rows).items())),
        "human_corrected_count": sum(review["decision"] == "corrected"
                                     for _, _, review, _ in reviewed),
        "artifact": {"path": str(final_path.resolve()), "sha256": file_sha256(final_path)},
    }
    atomic_write_json(final_dir / "dataset_manifest.json", result)
    dv2._set_status(target, "discovery-v2-demo-finalize", "passed", {
        "sample_count": 100, "dataset_sha256": result["artifact"]["sha256"]})
    print(json.dumps(result, indent=2))


def _balanced_export_rows(rows: Sequence[Mapping[str, Any]], *, seed: int,
                          split_name: str,
                          image_paths: Mapping[str, str]) -> list[dict[str, Any]]:
    """Build loader-compatible rows with an exact deterministic A/B balance."""

    if len(rows) % 2:
        raise RuntimeError(f"{split_name} must contain an even number of rows")
    ordered = sorted(rows, key=lambda row: dv2._sha([
        PROTOCOL_VERSION, "source-label-export-v1", seed, split_name,
        row["sample_id"]]))
    target_b = {row["sample_id"] for row in ordered[:len(rows) // 2]}
    exported = []
    for row in rows:
        source_answer = row["answer"]
        if source_answer not in {"A", "B"}:
            raise RuntimeError(f"{split_name} source answer must be A/B")
        target_answer = "B" if row["sample_id"] in target_b else "A"
        swap = source_answer != target_answer
        exported.append({
            "sample_id": row["sample_id"],
            "image_path": image_paths[row["image_sha256"]],
            "question": row["question"],
            "A": row["B"] if swap else row["A"],
            "B": row["A"] if swap else row["B"],
            "answer": target_answer,
            "source": row["source"],
            "source_family": row["source_family"],
            "source_sample_id": row["source_sample_id"],
            "domain": row["domain"],
            "subdomain": row["subdomain"],
            "split_role": split_name,
            "image_sha256": row["image_sha256"],
            "question_sha256": row["question_sha256"],
            "unordered_pair_sha256": row["unordered_pair_sha256"],
            "label_origin": "source_label_exploratory",
            "orientation_swapped_from_source": swap,
        })
    expected = {"A": len(rows) // 2, "B": len(rows) // 2}
    if Counter(row["answer"] for row in exported) != expected:
        raise RuntimeError(f"{split_name} A/B export balance drift")
    return exported


def export_data(config: Mapping[str, Any], output: Path) -> None:
    """Export exploratory Discovery100/Dev150 and their images under data/."""

    cfg, target, manifest, discovery = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-demo-adjudicate")
    dev_path = target / "selection_v2/dev_150.jsonl"
    dev = dv2._read_jsonl(dev_path)
    if len(dev) != 150:
        raise RuntimeError("Discovery-v2 export requires frozen Dev150")

    repository = Path(__file__).resolve().parents[2]
    export_root = repository / "data" / DATA_EXPORT_DIRECTORY
    image_root = export_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, str] = {}
    for row in [*discovery, *dev]:
        source = Path(row["image_path"])
        if not source.is_file() or file_sha256(source) != row["image_sha256"]:
            raise RuntimeError(f"export image identity drift: {row['sample_id']}")
        destination = image_root / f"{row['image_sha256']}{source.suffix.lower() or '.img'}"
        if destination.is_file():
            if file_sha256(destination) != row["image_sha256"]:
                raise RuntimeError(f"export image collision: {destination.name}")
        else:
            shutil.copy2(source, destination)
        image_paths[row["image_sha256"]] = destination.relative_to(repository).as_posix()

    seed = int(cfg["seed"])
    discovery_export = _balanced_export_rows(
        discovery, seed=seed, split_name="discovery", image_paths=image_paths)
    dev_export = _balanced_export_rows(
        dev, seed=seed, split_name="dev", image_paths=image_paths)
    discovery_path = export_root / "discovery_100.jsonl"
    dev_export_path = export_root / "dev_150.jsonl"
    dv2._write_jsonl(discovery_path, discovery_export)
    dv2._write_jsonl(dev_export_path, dev_export)

    from .experiment_utils import load_jsonl_dataset

    load_jsonl_dataset(discovery_path, expected_count=100)
    load_jsonl_dataset(dev_export_path, expected_count=150)
    overlap = dv2._selection_overlap(discovery_export, dev_export)
    if any(overlap.values()):
        raise RuntimeError(f"Discovery100/Dev150 export overlap: {overlap}")
    image_files = sorted(image_root.iterdir())
    value = {
        "schema_version": dv2.SCHEMA_VERSION,
        "protocol_version": "discovery100-dev150-source-label-export-v1",
        "seed": seed,
        "label_status": "exploratory_source_labels_not_human_reviewed",
        "397b_used_to_change_gold": False,
        "manager_visible_397b_analysis": False,
        "splits": {
            "discovery": {"count": 100, "gold_positions": {"A": 50, "B": 50},
                "path": discovery_path.relative_to(repository).as_posix(),
                "sha256": file_sha256(discovery_path)},
            "dev": {"count": 150, "gold_positions": {"A": 75, "B": 75},
                "path": dev_export_path.relative_to(repository).as_posix(),
                "sha256": file_sha256(dev_export_path)},
        },
        "source_counts": {
            name: dict(sorted(Counter(row["source"] for row in values).items()))
            for name, values in (("discovery", discovery_export), ("dev", dev_export))
        },
        "domain_counts": {
            name: dict(sorted(Counter(row["domain"] for row in values).items()))
            for name, values in (("discovery", discovery_export), ("dev", dev_export))
        },
        "image_count": len(image_paths),
        "image_bytes": sum(path.stat().st_size for path in image_files),
        "cross_split_overlap": overlap,
        "source_artifacts": {
            "discovery100_v3_sha256": manifest["frozen_rows"]["sha256"],
            "dev150_sha256": file_sha256(dev_path),
        },
    }
    atomic_write_json(export_root / "dataset_manifest.json", value)
    readme = (
        "# Discovery-v2 Demo v3 (Exploratory)\n\n"
        "This local export contains a source-label Discovery100 and an independent "
        "Dev150 for exploratory Prompt-v2 Split+Refine evolution. Labels have not "
        "completed human review. Qwen3.5-397B analyses were not used to change gold.\n\n"
        "- `discovery_100.jsonl`: Manager-visible evolution/competition data.\n"
        "- `dev_150.jsonl`: diagnostic-only data; never expose it to Managers.\n"
        "- `images/`: content-addressed images used by both JSONL files.\n")
    (export_root / "README.md").write_text(readme, encoding="utf-8")
    dv2._set_status(target, "discovery-v2-demo-export", "passed", value)
    print(json.dumps(value, indent=2, ensure_ascii=False))

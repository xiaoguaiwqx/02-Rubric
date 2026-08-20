"""Criterion-agnostic 397B pre-adjudication for Discovery-v2.

This module is deliberately separate from the legacy root-aware adjudication
pipeline.  It freezes three already-selected pools, judges every pair in both
answer orders, and exports a blind human-review queue.  Source labels, Worker
results, pool roles, roots, and rubrics never enter model prompts.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured.backend_pool import AvailableSlotBackendPool, BackendPoolSpec
from critiq.structured.telemetry import ModelCallMetrics, combine_model_call_metrics

from . import discovery_data_v2 as dv2
from .experiment_utils import atomic_write_json, load_json, make_progress_callback
from .rubric_factory import file_sha256


ROLE_PATHS = {
    "coverage_candidate": "selection_v2/coverage_candidates_150.jsonl",
    "hard_candidate": "selection_v2/hard_candidates_90.jsonl",
    "dev": "selection_v2/dev_150.jsonl",
}
EXPECTED_SOURCE_COUNTS = {
    "coverage_candidate": 25,
    "hard_candidate": 15,
    "dev": 25,
}
MODEL_VISIBLE_FIELDS = ("image_path", "question", "A", "B")
# The provider advertises more capacity, but burst-submitting 30 concurrent
# 397B multimodal requests repeatedly returns provider error 50508.  Keep this
# execution-only cap outside the semantic request/cache identity so lowering
# scheduling pressure never invalidates successful model outputs.
EFFECTIVE_REQUEST_CONCURRENCY = 5
QWEN3_VL_MIN_IMAGE_SIDE = 28
QWEN3_VL_IMAGE_REPAIR_VERSION = "qwen3-vl-min-side-28-v1"


def _prompt_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_spec(config: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    profile = cfg["adjudicator"]
    pool = BackendPoolSpec.from_dict(profile["backend_pool"])
    return {
        "schema_version": dv2.SCHEMA_VERSION,
        "protocol_version": dv2.PREADJUDICATION_PROTOCOL_VERSION,
        "prompt_version": dv2.PREADJUDICATION_PROMPT_VERSION,
        "model": pool.common_checkpoint_id,
        "backend_id": pool.backend_id,
        "backend_pool": pool.to_dict(),
        "system_prompt_sha256": _prompt_hash(dv2.PREADJUDICATION_SYSTEM_PROMPT),
        "user_prompt_sha256": _prompt_hash(dv2.PREADJUDICATION_USER_PROMPT),
        "content_order": "image_then_pair_text",
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "hidden_fields": [
            "source", "source_family", "source_sample_id", "answer", "domain",
            "subdomain", "generic_screen", "_adjudication_role", "roots", "rubric",
        ],
        "decoding_config": dict(profile["request_kwargs"]),
        "api_retry_attempts": int(profile["api_retry_attempts"]),
        "structured_max_attempts": int(profile["structured_max_attempts"]),
        "swap_orders": 2,
        "allow_uncertain": bool(profile["allow_uncertain"]),
    }


def _source_paths(target: Path) -> dict[str, Path]:
    return {role: target / relative for role, relative in ROLE_PATHS.items()}


def _load_and_validate_inputs(cfg: Mapping[str, Any], target: Path) -> list[dict[str, Any]]:
    enabled_sources = [item["name"] for item in cfg["sources"] if item["enabled"]]
    all_rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for role, path in _source_paths(target).items():
        rows = dv2._read_jsonl(path)
        expected = dv2.PREADJUDICATION_POOL_COUNTS[role]
        if len(rows) != expected:
            raise RuntimeError(f"{role} count drift: expected={expected}, actual={len(rows)}")
        counts = Counter(str(row.get("source")) for row in rows)
        expected_per_source = EXPECTED_SOURCE_COUNTS[role]
        if counts != Counter({source: expected_per_source for source in enabled_sources}):
            raise RuntimeError(f"{role} source quota drift: {dict(counts)}")
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise RuntimeError(f"{role} contains an empty sample_id")
            if sample_id in seen:
                raise RuntimeError(
                    f"pre-adjudication pools overlap: {sample_id} in {seen[sample_id]} and {role}")
            for field in MODEL_VISIBLE_FIELDS:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise RuntimeError(f"{sample_id} has invalid {field}")
            if not Path(row["image_path"]).is_file():
                raise RuntimeError(f"{sample_id} image is missing")
            seen[sample_id] = role
            all_rows.append({**row, "_adjudication_role": role})
    if len(all_rows) != 390:
        raise RuntimeError("pre-adjudication must freeze exactly 390 unique samples")
    return all_rows


def freeze(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = dv2._config(config), dv2._target(output)
    dv2._require(target, "discovery-v2-selection-freeze")
    dv2._require(target, "discovery-v2-generic-screen")
    rows = _load_and_validate_inputs(cfg, target)
    directory = target / "adjudication_v2"
    inputs_path = directory / "frozen_inputs.jsonl"
    manifest_path = directory / "frozen_manifest.json"
    source_artifacts = {
        role: {"path": str(path.resolve()), "count": dv2.PREADJUDICATION_POOL_COUNTS[role],
               "sha256": file_sha256(path)}
        for role, path in _source_paths(target).items()
    }
    if inputs_path.is_file():
        existing = dv2._read_jsonl(inputs_path)
        if existing != rows:
            raise RuntimeError("pre-adjudication frozen inputs drift")
    else:
        dv2._write_jsonl(inputs_path, rows)
    manifest = {
        "schema_version": dv2.SCHEMA_VERSION,
        "protocol_version": dv2.PREADJUDICATION_PROTOCOL_VERSION,
        "prompt_version": dv2.PREADJUDICATION_PROMPT_VERSION,
        "seed": int(cfg["seed"]),
        "sample_count": len(rows),
        "logical_order_count": len(rows) * 2,
        "pool_counts": dict(Counter(row["_adjudication_role"] for row in rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "source_artifacts": source_artifacts,
        "frozen_inputs": {"path": str(inputs_path.resolve()), "count": len(rows),
                          "sha256": file_sha256(inputs_path)},
        "request_spec": _request_spec(config, cfg),
        "privacy": {
            "source_visible_to_model": False,
            "gold_visible_to_model": False,
            "generic_screen_visible_to_model": False,
            "pool_role_visible_to_model": False,
            "root_or_rubric_visible_to_model": False,
        },
    }
    if manifest_path.is_file() and load_json(manifest_path) != manifest:
        raise RuntimeError("pre-adjudication frozen manifest drift")
    if not manifest_path.is_file():
        atomic_write_json(manifest_path, manifest)
    dv2._set_status(target, "discovery-v2-adjudication-freeze", "passed", {
        "sample_count": len(rows), "logical_order_count": len(rows) * 2})
    print(json.dumps({"sample_count": len(rows), "logical_order_count": len(rows) * 2,
                      "pool_counts": manifest["pool_counts"]}, indent=2))


def _load_frozen(config: Mapping[str, Any], output: Path) -> tuple[
        dict[str, Any], Path, dict[str, Any], list[dict[str, Any]]]:
    cfg, target = dv2._config(config), dv2._target(output)
    dv2._require(target, "discovery-v2-adjudication-freeze")
    manifest_path = target / "adjudication_v2/frozen_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("request_spec") != _request_spec(config, cfg):
        raise RuntimeError("pre-adjudication request identity drift")
    for role, path in _source_paths(target).items():
        frozen = manifest["source_artifacts"][role]
        if not path.is_file() or file_sha256(path) != frozen["sha256"]:
            raise RuntimeError(f"pre-adjudication source artifact drift: {role}")
    inputs_path = target / "adjudication_v2/frozen_inputs.jsonl"
    if file_sha256(inputs_path) != manifest["frozen_inputs"]["sha256"]:
        raise RuntimeError("pre-adjudication frozen input hash drift")
    rows = dv2._read_jsonl(inputs_path)
    if len(rows) != manifest["sample_count"]:
        raise RuntimeError("pre-adjudication frozen input count drift")
    return cfg, target, manifest, rows


def validate_response(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "answer", "confidence", "preference_rationale", "visual_evidence",
        "task_type", "preference_dimensions", "evidence_type",
        "candidate_a_issues", "candidate_b_issues", "ambiguity_flags",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("pre-adjudication response fields are invalid")
    if value["answer"] not in {"A", "B", "uncertain"}:
        raise ValueError("pre-adjudication answer is invalid")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 4:
        raise ValueError("pre-adjudication confidence must be an integer from 1 to 4")
    if (not isinstance(value["preference_rationale"], str)
            or not value["preference_rationale"].strip()):
        raise ValueError("preference_rationale must be non-empty")
    if value["task_type"] not in dv2.PREADJUDICATION_TASK_TYPES:
        raise ValueError("pre-adjudication task_type is invalid")
    if value["evidence_type"] not in dv2.PREADJUDICATION_EVIDENCE_TYPES:
        raise ValueError("pre-adjudication evidence_type is invalid")
    for field in ("visual_evidence", "preference_dimensions", "candidate_a_issues",
                  "candidate_b_issues", "ambiguity_flags"):
        values = value[field]
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"pre-adjudication {field} must be a string list")
    if not value["preference_dimensions"]:
        raise ValueError("preference_dimensions must be non-empty")
    if not set(value["preference_dimensions"]).issubset(dv2.PREADJUDICATION_DIMENSIONS):
        raise ValueError("pre-adjudication preference_dimensions are invalid")
    if not set(value["ambiguity_flags"]).issubset(dv2.PREADJUDICATION_AMBIGUITY_FLAGS):
        raise ValueError("pre-adjudication ambiguity_flags are invalid")
    return {key: value[key] for key in fields}


def _qwen3_vl_image_repair(row: Mapping[str, Any]) -> str | None:
    """Return the deterministic repair identity required by tiny images."""

    from PIL import Image  # type: ignore

    with Image.open(row["image_path"]) as image:
        width, height = image.size
    return (QWEN3_VL_IMAGE_REPAIR_VERSION
            if min(width, height) < QWEN3_VL_MIN_IMAGE_SIDE else None)


def _qwen3_vl_image_data_url(row: Mapping[str, Any]) -> str:
    """Encode an image, minimally upscaling dimensions rejected by Qwen3-VL.

    The source artifact and its SHA remain untouched.  Only the request payload
    is repaired, and the matching repair version is included in the cache
    fingerprint below.
    """

    repair = _qwen3_vl_image_repair(row)
    if repair is None:
        return dv2._image_data_url(row["image_path"])

    from PIL import Image  # type: ignore

    with Image.open(row["image_path"]) as image:
        width, height = image.size
        scale = max(QWEN3_VL_MIN_IMAGE_SIDE / width,
                    QWEN3_VL_MIN_IMAGE_SIDE / height)
        size = (max(QWEN3_VL_MIN_IMAGE_SIDE, math.ceil(width * scale)),
                max(QWEN3_VL_MIN_IMAGE_SIDE, math.ceil(height * scale)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _user_content(row: Mapping[str, Any], *, swapped: bool, retry: bool) -> list[dict[str, Any]]:
    a, b = (row["B"], row["A"]) if swapped else (row["A"], row["B"])
    text = dv2.PREADJUDICATION_USER_PROMPT.format(question=row["question"], A=a, B=b)
    if retry:
        text += ("\n\nFormatting reminder: return exactly one JSON object using every required "
                 "key. answer must be A, B, or uncertain; confidence must be 1-4.")
    return [
        {"type": "image_url", "image_url": {"url": _qwen3_vl_image_data_url(row)}},
        {"type": "text", "text": text},
    ]


def _input_fingerprint(row: Mapping[str, Any], *, swapped: bool) -> str:
    payload = {"sample_id": row["sample_id"], "image_sha256": row["image_sha256"],
               "question": row["question"], "A": row["A"], "B": row["B"],
               "swapped": swapped}
    repair = _qwen3_vl_image_repair(row)
    if repair is not None:
        payload["image_request_repair"] = repair
    return dv2._sha(payload)


def _order_cached(config: Mapping[str, Any], cfg: Mapping[str, Any],
                  request_spec: Mapping[str, Any], row: Mapping[str, Any], *,
                  swapped: bool, pool: AvailableSlotBackendPool,
                  cache_dir: Path) -> tuple[dict[str, Any], ModelCallMetrics]:
    fingerprint = _input_fingerprint(row, swapped=swapped)
    key = dv2._sha({"request_spec": request_spec, "input_fingerprint": fingerprint})
    path = cache_dir / f"{key}.json"
    if path.is_file():
        cached = load_json(path)
        if (cached.get("request_spec") != request_spec
                or cached.get("input_fingerprint") != fingerprint):
            raise RuntimeError("pre-adjudication cache identity drift")
        if cached.get("result", {}).get("parse_ok"):
            result = dict(cached["result"])
            result["cache_hit"] = True
            return result, ModelCallMetrics.from_agent_calls((), cache_hit=True)

    profile = cfg["adjudicator"]
    key_env = profile["api_key_env"]
    api_keys = os.environ.get(key_env, "") if key_env else "EMPTY"
    if not api_keys:
        raise RuntimeError(f"environment variable {key_env!r} is required")
    agent_args = {
        "model": request_spec["model"], "api_keys": api_keys,
        "system": dv2.PREADJUDICATION_SYSTEM_PROMPT,
        "request_kwargs": dict(request_spec["decoding_config"]),
        "api_retry_attempts": int(request_spec["api_retry_attempts"]),
    }
    calls, raw_outputs = [], []
    parsed = None
    last_error = "pre-adjudicator did not return parseable JSON"
    order = "swapped" if swapped else "original"
    for attempt in range(1, int(request_spec["structured_max_attempts"]) + 1):
        try:
            raw, call_metrics = pool.call(
                _user_content(row, swapped=swapped, retry=attempt > 1),
                request_type="discovery_v2_preadjudication",
                request_key=f"{row['sample_id']}::{order}",
                structured_attempt=attempt, agent_args=agent_args)
            calls.append(call_metrics)
            raw_outputs.append(raw if isinstance(raw, str) else None)
            parsed = validate_response(dv2._extract_object(raw))
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        except Exception as exc:  # transport failures remain explicit and resumable
            last_error = f"{type(exc).__name__}: {exc}"
            raw_outputs.append(None)
    metrics = ModelCallMetrics.from_agent_calls(
        calls, logical_evaluations=1, parse_retries=max(0, len(raw_outputs) - 1))
    result = {
        "sample_id": row["sample_id"], "order": order,
        "parse_ok": parsed is not None, "parsed": parsed,
        "answer": parsed["answer"] if parsed is not None else None,
        "attempt_count": len(raw_outputs),
        "parse_error": None if parsed is not None else last_error,
        "raw_responses": raw_outputs, "cache_hit": False,
        "generation_metrics": metrics.to_dict(),
    }
    atomic_write_json(path, {"schema_version": dv2.SCHEMA_VERSION,
        "request_spec": dict(request_spec), "input_fingerprint": fingerprint,
        "result": result})
    return result, metrics


def _run_orders(config: Mapping[str, Any], cfg: Mapping[str, Any], target: Path,
                rows: Sequence[Mapping[str, Any]], *, label: str,
                artifact_directory: str = "adjudication_v2",
                cache_directory: str = "adjudication_v2/cache/orders") -> tuple[
                    list[dict[str, Any]], dict[str, Any]]:
    request_spec = _request_spec(config, cfg)
    pool_spec = BackendPoolSpec.from_dict(cfg["adjudicator"]["backend_pool"])
    pool = AvailableSlotBackendPool(pool_spec)
    total = len(rows) * 2
    work = target / artifact_directory / label
    callback = make_progress_callback(work, label, total, pool)
    results = [{"sample_id": row["sample_id"], "original": None, "swapped": None}
               for row in rows]
    metrics: list[ModelCallMetrics] = []
    cache_dir = target / cache_directory
    effective_concurrency = min(
        pool_spec.global_request_concurrency, EFFECTIVE_REQUEST_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        futures = {}
        for index, row in enumerate(rows):
            for swapped in (False, True):
                future = executor.submit(_order_cached, config, cfg, request_spec, row,
                    swapped=swapped, pool=pool, cache_dir=cache_dir)
                futures[future] = (index, swapped)
        for completed, future in enumerate(as_completed(futures), 1):
            index, swapped = futures[future]
            result, item_metrics = future.result()
            order = "swapped" if swapped else "original"
            results[index][order] = result
            metrics.append(item_metrics)
            callback(completed - 1, f"{rows[index]['sample_id']}::{order}", item_metrics)
    unresolved = [f"{item['sample_id']}::{order}" for item in results
                  for order in ("original", "swapped") if not item[order]["parse_ok"]]
    combined = combine_model_call_metrics(metrics)
    summary = {
        "schema_version": dv2.SCHEMA_VERSION,
        "sample_count": len(rows), "logical_order_count": total,
        "parse_valid_count": total - len(unresolved),
        "parse_valid_rate": (total - len(unresolved)) / total if total else 1.0,
        "unresolved_sample_orders": unresolved,
        "current_run_metrics": combined.to_dict(),
        "endpoint_call_counts": dict(pool.records_by_endpoint()),
        "request_spec": request_spec,
        "execution_concurrency": {
            "configured_capacity": pool_spec.global_request_concurrency,
            "effective_request_concurrency": effective_concurrency,
            "reason": "provider_397b_busy_safety_cap",
            "excluded_from_semantic_cache_identity": True,
        },
    }
    dv2._write_jsonl(work / "judgments.jsonl", results)
    atomic_write_json(work / "summary.json", summary)
    atomic_write_json(work / "provenance.json", pool.provenance_dict())
    return results, summary


def _smoke_rows(rows: Sequence[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = [item["name"] for item in cfg["sources"] if item["enabled"]]
    selected = []
    for role in ROLE_PATHS:
        for source in sources:
            values = [row for row in rows
                      if row["_adjudication_role"] == role and row["source"] == source]
            values.sort(key=lambda row: dv2._sha([
                dv2.PREADJUDICATION_PROTOCOL_VERSION, cfg["seed"], role,
                source, row["sample_id"]]))
            if not values:
                raise RuntimeError(f"smoke lacks {role}/{source}")
            selected.append(values[0])
    if len(selected) != 18:
        raise RuntimeError("pre-adjudication smoke must contain exactly 18 samples")
    return selected


def smoke(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, rows = _load_frozen(config, output)
    selected = _smoke_rows(rows, cfg)
    _, summary = _run_orders(config, cfg, target, selected, label="smoke")
    report_value = {**summary,
        "expected_sample_count": 18, "expected_logical_order_count": 36,
        "sources": dict(Counter(row["source"] for row in selected)),
        "roles": dict(Counter(row["_adjudication_role"] for row in selected)),
        "frozen_manifest_sha256": file_sha256(target / "adjudication_v2/frozen_manifest.json")}
    atomic_write_json(target / "adjudication_v2/smoke_report.json", report_value)
    if summary["unresolved_sample_orders"]:
        raise RuntimeError(
            f"pre-adjudication smoke has {len(summary['unresolved_sample_orders'])} unresolved orders")
    dv2._set_status(target, "discovery-v2-adjudication-smoke", "passed", {
        "sample_count": 18, "logical_order_count": 36, "parse_valid_rate": 1.0})
    print(json.dumps({"sample_count": 18, "logical_order_count": 36,
                      "parse_valid_rate": 1.0,
                      "endpoint_call_counts": summary["endpoint_call_counts"]}, indent=2))


def _map_swapped_answer(answer: str) -> str:
    return {"A": "B", "B": "A", "uncertain": "uncertain"}[answer]


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def merge_orders(row: Mapping[str, Any], judgment: Mapping[str, Any]) -> dict[str, Any]:
    original = judgment["original"]["parsed"]
    swapped = judgment["swapped"]["parsed"]
    if original is None or swapped is None:
        raise ValueError("cannot merge unresolved pre-adjudication orders")
    swapped_mapped = _map_swapped_answer(swapped["answer"])
    agreement = original["answer"] == swapped_mapped
    suggested = original["answer"] if agreement and original["answer"] in {"A", "B"} else None
    return {**row, "preadjudication": {
        "protocol_version": dv2.PREADJUDICATION_PROTOCOL_VERSION,
        "original": original, "swapped": swapped,
        "swapped_answer_mapped": swapped_mapped,
        "answer_agreement": agreement,
        "suggested_answer": suggested,
        "minimum_confidence": min(original["confidence"], swapped["confidence"]),
        "task_type_agreement": original["task_type"] == swapped["task_type"],
        "evidence_type_agreement": original["evidence_type"] == swapped["evidence_type"],
        "preference_dimension_jaccard": _jaccard(
            original["preference_dimensions"], swapped["preference_dimensions"]),
        "source_answer_agreement": suggested == row["answer"] if suggested else None,
        "requires_reconciliation": suggested is None or suggested != row["answer"],
    }}


def run(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, rows = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-adjudication-smoke")
    judgments, summary = _run_orders(config, cfg, target, rows, label="full")
    run_report = {**summary,
        "frozen_manifest_sha256": file_sha256(target / "adjudication_v2/frozen_manifest.json"),
        "successful_smoke_cache_reuse_possible": 36,
        "rerun_behavior": "successful orders are cached; only unresolved orders are retried"}
    atomic_write_json(target / "adjudication_v2/run_report.json", run_report)
    if summary["unresolved_sample_orders"]:
        dv2._set_status(target, "discovery-v2-adjudication-run", "incomplete", {
            "unresolved_count": len(summary["unresolved_sample_orders"]),
            "parse_valid_rate": summary["parse_valid_rate"],
            "rerun_stage": "discovery-v2-adjudication-run"})
        print(json.dumps({"status": "incomplete",
            "unresolved_count": len(summary["unresolved_sample_orders"]),
            "parse_valid_rate": summary["parse_valid_rate"],
            "rerun_stage": "discovery-v2-adjudication-run"}, indent=2))
        return
    by_id = {item["sample_id"]: item for item in judgments}
    records = [merge_orders(row, by_id[row["sample_id"]]) for row in rows]
    records_path = target / "adjudication_v2/records.jsonl"
    dv2._write_jsonl(records_path, records)
    dv2._set_status(target, "discovery-v2-adjudication-run", "passed", {
        "sample_count": len(records), "logical_order_count": len(records) * 2,
        "parse_valid_rate": 1.0, "records_sha256": file_sha256(records_path)})
    print(json.dumps({"status": "passed", "sample_count": len(records),
                      "logical_order_count": len(records) * 2,
                      "parse_valid_rate": 1.0}, indent=2))


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _group_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "answer_agreement_rate": _rate([
            bool(row["preadjudication"]["answer_agreement"]) for row in rows]),
        "suggested_answer_rate": _rate([
            row["preadjudication"]["suggested_answer"] is not None for row in rows]),
        "source_answer_agreement_rate_on_suggested": _rate([
            bool(row["preadjudication"]["source_answer_agreement"]) for row in rows
            if row["preadjudication"]["source_answer_agreement"] is not None]),
        "reconciliation_rate": _rate([
            bool(row["preadjudication"]["requires_reconciliation"]) for row in rows]),
        "mean_minimum_confidence": (
            sum(row["preadjudication"]["minimum_confidence"] for row in rows) / len(rows)
            if rows else None),
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, _ = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-adjudication-run")
    records_path = target / "adjudication_v2/records.jsonl"
    rows = dv2._read_jsonl(records_path)
    if len(rows) != 390:
        raise RuntimeError("pre-adjudication report requires 390 complete records")
    by_role = {role: _group_report([row for row in rows
                                   if row["_adjudication_role"] == role])
               for role in ROLE_PATHS}
    sources = sorted({row["source"] for row in rows})
    by_source = {source: _group_report([row for row in rows if row["source"] == source])
                 for source in sources}
    task_types = Counter()
    dimensions = Counter()
    ambiguity = Counter()
    for row in rows:
        for order in ("original", "swapped"):
            item = row["preadjudication"][order]
            task_types[item["task_type"]] += 1
            dimensions.update(item["preference_dimensions"])
            ambiguity.update(item["ambiguity_flags"])
    result = {
        "schema_version": dv2.SCHEMA_VERSION,
        "protocol_version": dv2.PREADJUDICATION_PROTOCOL_VERSION,
        "sample_count": len(rows), "logical_order_count": len(rows) * 2,
        "parse_valid_rate": 1.0,
        "overall": _group_report(rows), "by_pool_role": by_role,
        "by_source": by_source,
        "task_type_order_counts": dict(sorted(task_types.items())),
        "preference_dimension_order_counts": dict(sorted(dimensions.items())),
        "ambiguity_flag_order_counts": dict(sorted(ambiguity.items())),
        "records_sha256": file_sha256(records_path),
        "exploratory": True,
        "human_review_required_for_final_gold": True,
    }
    atomic_write_json(target / "adjudication_v2/report.json", result)
    dv2._set_status(target, "discovery-v2-adjudication-report", "passed", {
        "sample_count": len(rows),
        "answer_agreement_rate": result["overall"]["answer_agreement_rate"],
        "reconciliation_rate": result["overall"]["reconciliation_rate"]})
    print(json.dumps({"sample_count": len(rows), "parse_valid_rate": 1.0,
                      "overall": result["overall"], "by_pool_role": by_role}, indent=2))


def _display_plan(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, bool]:
    ordered = sorted(rows, key=lambda row: dv2._sha([
        dv2.PREADJUDICATION_PROTOCOL_VERSION, seed, "display", row["sample_id"]]))
    swap_count = len(ordered) // 2
    return {row["sample_id"]: index < swap_count for index, row in enumerate(ordered)}


def _blank_human_review() -> dict[str, Any]:
    return {"answer": None, "confidence": None, "preference_rationale": "",
            "visual_evidence": [], "task_type": None, "preference_dimensions": [],
            "evidence_type": None, "candidate_a_issues": [], "candidate_b_issues": [],
            "ambiguity_flags": [], "decision": None, "reviewed": False,
            "reconciled": False, "notes": ""}


def _original_human_answer(item: Mapping[str, Any], swapped: bool) -> str | None:
    answer = item.get("human_review", {}).get("answer")
    if answer not in {"A", "B"}:
        return None
    return _map_swapped_answer(answer) if swapped else answer


def review_export(config: Mapping[str, Any], output: Path) -> None:
    cfg, target, manifest, _ = _load_frozen(config, output)
    dv2._require(target, "discovery-v2-adjudication-report")
    rows = dv2._read_jsonl(target / "adjudication_v2/records.jsonl")
    review_dir = target / "adjudication_v2/review_v2"
    queue_path = review_dir / "blind_review_queue.jsonl"
    hidden_path = review_dir / "hidden_reference.jsonl"
    existing = {row["review_id"]: row for row in dv2._read_jsonl(queue_path)} \
        if queue_path.is_file() else {}
    display = _display_plan(rows, int(cfg["seed"]))
    ordered = sorted(rows, key=lambda row: dv2._sha([
        dv2.PREADJUDICATION_PROTOCOL_VERSION, cfg["seed"], "queue", row["sample_id"]]))
    queue, hidden = [], []
    for rank, row in enumerate(ordered, 1):
        review_id = f"dv2-review-{rank:04d}"
        swapped = display[row["sample_id"]]
        previous = existing.get(review_id, {})
        human = previous.get("human_review", _blank_human_review())
        queue.append({
            "review_id": review_id, "queue_rank": rank,
            "image_path": row["image_path"], "question": row["question"],
            "A": row["B"] if swapped else row["A"],
            "B": row["A"] if swapped else row["B"],
            "human_review": human,
        })
        hidden.append({
            "review_id": review_id, "sample_id": row["sample_id"],
            "display_swapped": swapped, "source": row["source"],
            "source_answer": row["answer"],
            "adjudication_role": row["_adjudication_role"],
            "generic_screen": row.get("generic_screen"),
            "preadjudication": row["preadjudication"],
        })
    dv2._write_jsonl(queue_path, queue)
    dv2._write_jsonl(hidden_path, hidden)

    reconciliation = []
    hidden_by_id = {item["review_id"]: item for item in hidden}
    reviewed_count = 0
    for item in queue:
        human = item["human_review"]
        if human.get("reviewed") is not True:
            continue
        reviewed_count += 1
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
        "schema_version": dv2.SCHEMA_VERSION, "blind_queue_count": len(queue),
        "display_swapped_count": sum(display.values()),
        "display_original_count": len(display) - sum(display.values()),
        "reviewed_count": reviewed_count,
        "reconciliation_queue_count": len(reconciliation),
        "blind_fields": ["review_id", "queue_rank", "image_path", "question", "A", "B",
                         "human_review"],
        "hidden_from_first_pass": ["source", "source_answer", "generic_screen",
                                    "adjudication_role", "preadjudication"],
        "queue_sha256": file_sha256(queue_path),
        "hidden_reference_sha256": file_sha256(hidden_path),
    }
    atomic_write_json(review_dir / "export_report.json", result)
    dv2._set_status(target, "discovery-v2-review-v2-export", "passed", {
        "blind_queue_count": len(queue), "reviewed_count": reviewed_count,
        "reconciliation_queue_count": len(reconciliation)})
    print(json.dumps({key: result[key] for key in (
        "blind_queue_count", "display_swapped_count", "display_original_count",
        "reviewed_count", "reconciliation_queue_count")}, indent=2))

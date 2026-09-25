"""Read-only request-equivalence audit for the seed11 fixed-root control.

The old S0/Final predictions may be cited by the generated-root experiment only
when their complete VLRB requests still match its frozen F5 protocol.  The
manifest points at the old artifacts; it does not copy or relabel predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured.cache import canonical_sha256
from critiq.structured.schema import StructuredRubric

from . import _global_arbiter_ab_only_support as support
from . import aligned_system_runtime as system
from . import global_arbiter_ab_only as arbiter
from . import internal_global_arbiter_k1 as unified
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_aligned_evolution as aligned
from . import vl_rewardbench_phase10 as official
from .experiment_utils import atomic_write_json, load_json
from .rubric_factory import build_multicrit_open_ended_init_rubric, file_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"F5 equivalence audit: {message}")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cache_request(
    old_target: Path, split: str, kind: str, call: Mapping[str, Any],
    digest: Any,
) -> dict[str, Any]:
    key = call.get("cache_key")
    _require(isinstance(key, str) and len(key) == 64,
             f"{split}/{kind}: missing cache key")
    path = old_target / "cache" / split / kind / key[:2] / f"{key}.json"
    _require(path.is_file(), f"missing request cache: {path}")
    raw = path.read_bytes()
    digest.update(path.relative_to(old_target).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(raw).digest())
    cached = json.loads(raw)
    _require(cached.get("cache_key") == key
             and canonical_sha256(cached.get("request")) == key,
             f"cache identity differs: {path}")
    for field in ("parse_ok", "parsed", "raw_response",
                  "model_generation_count", "endpoint_id"):
        _require(call.get(field) == cached.get(field),
                 f"{split}/{kind}: compact result differs from {path}: {field}")
    _require(call.get("parse_ok") is True and cached.get("parse_ok") is True,
             f"{split}/{kind}: unresolved technical failure: {path}")
    return cached["request"]


def _expected_request(
    *, config: Mapping[str, Any], endpoint: Mapping[str, Any],
    prompt_version: str, protocol_version: str, system_prompt: str,
    user_prompt: str, image_sha256: str, request_key: Mapping[str, Any],
) -> dict[str, Any]:
    worker = config["worker"]
    return {
        "schema_version": support.SCHEMA_VERSION,
        "protocol_version": protocol_version,
        "prompt_version": prompt_version,
        "model": worker["model"],
        "endpoint_checkpoint": endpoint["checkpoint_root"],
        "system_sha256": _sha_text(system_prompt),
        "user_sha256": _sha_text(user_prompt),
        "image_sha256": image_sha256,
        "request": {
            "temperature": worker["temperature"],
            "max_tokens": worker["max_tokens"],
            "generation_seed_policy": "unset",
        },
        "request_key": dict(request_key),
    }


def _audit_stage(
    old_target: Path, stage: str, config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Sequence[int]], image_hashes: Mapping[str, str],
    rubric: StructuredRubric, initial: StructuredRubric,
    endpoints: Mapping[str, Mapping[str, Any]], heldout_ids: set[str],
) -> dict[str, Any]:
    split_name = f"vlrb/{stage}"
    artifact = old_target / f"{split_name}.json"
    _require(artifact.is_file(), f"missing prediction artifact: {artifact}")
    value = load_json(artifact)
    record_ids = [str(row["sample_id"]) for row in records]
    _require(value.get("schema_version") == system.SCHEMA_VERSION
             and value.get("protocol_version") == system.PROTOCOL_VERSION
             and value.get("split") == split_name
             and value.get("rubric_sha256") == rubric.rubric_sha256
             and value.get("k") == 3,
             f"{split_name}: artifact identity differs")
    subtree_hashes = {
        root_id: system._root_subtree_sha256(rubric, root_id)
        for root_id in rubric.root_ids
    }
    _require(value.get("root_subtree_sha256") == subtree_hashes,
             f"{split_name}: subtree hashes differ")
    samples = value.get("samples")
    _require(isinstance(samples, list)
             and [str(item.get("sample_id")) for item in samples] == record_ids,
             f"{split_name}: VLRB sample order differs")
    row_by_id = {str(row["sample_id"]): row for row in rows}
    initial_hashes = {
        root_id: system._root_subtree_sha256(initial, root_id)
        for root_id in initial.root_ids
    }
    cache_digest = hashlib.sha256()
    request_count = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        expected_orders = tuple(int(order) for order in schedule[sample_id])
        _require(tuple(sample.get("orders", ())) == expected_orders
                 and set(sample.get("replicates", {})) == {"0", "1", "2"},
                 f"{split_name}/{sample_id}: A/B schedule differs")
        sample_endpoint = sample.get("endpoint_id")
        _require(sample_endpoint in endpoints,
                 f"{split_name}/{sample_id}: unknown endpoint")
        for replicate, order in enumerate(expected_orders):
            item = sample["replicates"][str(replicate)]
            _require(item.get("order") == order,
                     f"{split_name}/{sample_id}/r{replicate}: order differs")
            displayed = support.ordered_row(row_by_id[sample_id], order, replicate)
            calls = item.get("subtrees")
            _require(isinstance(calls, dict)
                     and tuple(calls) == rubric.root_ids,
                     f"{split_name}/{sample_id}/r{replicate}: root reports differ")
            for root_id in rubric.root_ids:
                call = calls[root_id]
                endpoint_id = call.get("endpoint_id")
                _require(endpoint_id == sample_endpoint,
                         f"{split_name}/{sample_id}/r{replicate}: subtree endpoint differs")
                reused = bool(call.get("incremental_reuse"))
                request_split = "vlrb/initial" if reused else split_name
                if reused:
                    _require(stage == "final"
                             and initial_hashes[root_id] == subtree_hashes[root_id],
                             f"{split_name}/{sample_id}/r{replicate}: invalid subtree reuse")
                actual = _cache_request(
                    old_target, request_split, "subtree", call, cache_digest)
                expected = _expected_request(
                    config=config, endpoint=endpoints[endpoint_id],
                    prompt_version=unified.SUBTREE_PROMPT_VERSION,
                    protocol_version=system.PROTOCOL_VERSION,
                    system_prompt=unified.UNIFIED_SUBTREE_SYSTEM_PROMPT,
                    user_prompt=unified.subtree_user_prompt(displayed, rubric, root_id),
                    image_sha256=image_hashes[sample_id],
                    request_key={
                        "kind": "aligned_unified_subtree",
                        "split": request_split,
                        "sample_id": str(displayed["sample_id"]),
                        "root_id": root_id,
                        "root_subtree_sha256": subtree_hashes[root_id],
                        "replicate": replicate,
                        "order": order,
                    },
                )
                _require(actual == expected,
                         f"{split_name}/{sample_id}/r{replicate}/{root_id}: request differs")
                request_count += 1
            reports = system._reports_from_calls(rubric, calls)
            _require(len(reports) == len(rubric.root_ids)
                     and item.get("report_bundle_sha256") == canonical_sha256(reports),
                     f"{split_name}/{sample_id}/r{replicate}: report bundle differs")
            final_call = item.get("arbiter", {})
            endpoint_id = final_call.get("endpoint_id")
            _require(endpoint_id == sample_endpoint,
                     f"{split_name}/{sample_id}/r{replicate}: arbiter endpoint differs")
            actual = _cache_request(
                old_target, split_name, "arbiter", final_call, cache_digest)
            expected = _expected_request(
                config=config, endpoint=endpoints[endpoint_id],
                prompt_version=unified.ARBITER_PROMPT_VERSION,
                protocol_version=system.PROTOCOL_VERSION + "-full-reason",
                system_prompt=arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
                user_prompt=support.global_arbiter_user_prompt(displayed, reports),
                image_sha256=image_hashes[sample_id],
                request_key={
                    "kind": "aligned_global_arbiter",
                    "split": split_name,
                    "sample_id": str(displayed["sample_id"]),
                    "source_report_bundle_sha256": canonical_sha256(reports),
                    "replicate": replicate,
                    "order": order,
                },
            )
            _require(actual == expected,
                     f"{split_name}/{sample_id}/r{replicate}: arbiter request differs")
            request_count += 1
    recomputed = system.metrics(value, rows)
    _require(value.get("metrics") == recomputed
             and recomputed["technical_failure_count"] == 0,
             f"{split_name}: stored metrics or technical status differ")
    official_result = official._system_metrics(records, aligned._votes(value))
    # Runtime metrics select the most frequent non-abstaining vote, while VLRB
    # requires two agreeing votes out of K=3. Validate each metric against its
    # own semantics instead of requiring their accuracies to be identical.
    predictions = official_result["original_index_predictions"]
    heldout_correct = sum(
        predictions[index] == int(record["preferred_original_index"])
        for index, record in enumerate(records)
        if str(record["sample_id"]) in heldout_ids
    )
    return {
        "artifact_sha256": file_sha256(artifact),
        "rubric_sha256": rubric.rubric_sha256,
        "sample_count": len(samples),
        "k": 3,
        "request_count": request_count,
        "request_cache_files_sha256": cache_digest.hexdigest(),
        "strict_correct": official_result["correct_count"],
        "heldout_hallucination_correct": heldout_correct,
        "heldout_hallucination_total": len(heldout_ids),
        "technical_failure_count": 0,
    }


def audit_fixed_seed11(
    old_target: Path, config: dict, split: dict,
    records: Sequence[Mapping[str, Any]], output_manifest: Path,
) -> dict[str, Any]:
    """Audit old F5 S0/Final predictions before reusing them as a control."""

    old_target = Path(old_target).resolve()
    output_manifest = Path(output_manifest).resolve()
    _require(old_target.is_dir(), f"missing old seed11 output: {old_target}")
    _require(old_target not in output_manifest.parents,
             "the reuse manifest must be outside the historical output")
    old_split = load_json(old_target / "split.json")
    old_config = load_json(old_target / "run_config.json")
    _require(split == old_split and split.get("seed") == 11,
             "frozen seed11 split differs")
    _require(config == old_config,
             "frozen F5 experiment configuration differs")
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    _require(parquet.is_file()
             and file_sha256(parquet) == split.get("parquet_sha256"),
             "VLRB parquet differs from the frozen split")
    record_ids = [str(record["sample_id"]) for record in records]
    heldout_ids = set(str(item) for item in split["heldout_ids"])
    train_ids = set(str(item) for item in split["train_ids"])
    _require(len(record_ids) == 1247 and len(set(record_ids)) == 1247
             and len(train_ids) == 100 and len(heldout_ids) == 648
             and not train_ids.intersection(heldout_ids)
             and train_ids.issubset(record_ids)
             and heldout_ids.issubset(record_ids),
             "VLRB records or train/heldout IDs differ")
    image_hashes = {}
    for record in records:
        sample_id = str(record["sample_id"])
        image_path = Path(str(record["image_path"]))
        _require(image_path.is_file(), f"missing image: {sample_id}")
        actual_hash = file_sha256(image_path)
        _require(actual_hash == record.get("image_sha256"),
                 f"image bytes differ: {sample_id}")
        image_hashes[sample_id] = actual_hash
    bare = StructuredRubric.load_json(old_target / "r0/rubric.json")
    initial = StructuredRubric.load_json(old_target / "init/rubric.json")
    final = StructuredRubric.load_json(old_target / "final.json")
    fixed = build_multicrit_open_ended_init_rubric()
    _require(bare.rubric_sha256 == fixed.rubric_sha256
             and initial.root_ids == fixed.root_ids
             and final.root_ids == fixed.root_ids,
             "old F5 roots differ from fixed MultiCrit roots")
    for rubric in (initial, final):
        _require(all(rubric.get_node(root_id).to_dict()
                     == fixed.get_node(root_id).to_dict()
                     for root_id in fixed.root_ids),
                 "a fixed root changed during Split or evolution")
    state = load_json(old_target / "state.json")
    _require(state.get("completed") is True
             and StructuredRubric.from_dict(state["rubric"]).rubric_sha256
             == final.rubric_sha256,
             "old Final is not the completed frozen rubric")
    endpoints = {
        endpoint["endpoint_id"]: endpoint
        for endpoint in config["worker"]["backend_pool"]["endpoints"]
    }
    _require(len(endpoints) == len(config["worker"]["backend_pool"]["endpoints"]),
             "duplicate Worker endpoint IDs")
    rows = support.vlrb_rows(records)
    schedule = vlrb._order_schedule(records)
    stages = {
        stage: _audit_stage(
            old_target, stage, config, records, rows, schedule, image_hashes,
            rubric, initial, endpoints, heldout_ids,
        )
        for stage, rubric in (("initial", initial), ("final", final))
    }
    _require(stages["initial"]["strict_correct"] == 906
             and stages["initial"]["heldout_hallucination_correct"] == 527
             and stages["final"]["strict_correct"] == 920
             and stages["final"]["heldout_hallucination_correct"] == 534,
             "old F5 scores differ from frozen seed11 results")
    files = (
        "run_config.json", "split.json", "r0/rubric.json", "init/rubric.json",
        "final.json", "state.json", "vlrb/initial.json", "vlrb/final.json",
    )
    manifest = {
        "schema_version": "1.0.0",
        "kind": "generated-roots-f5-old-seed11-equivalence-audit",
        "source_output": str(old_target),
        "source_artifact_sha256": {
            name: file_sha256(old_target / name) for name in files
        },
        "vlrb_parquet_sha256": split["parquet_sha256"],
        "fixed_r0_rubric_sha256": fixed.rubric_sha256,
        "arbiter_system_prompt_sha256": _sha_text(
            arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT),
        "stages": stages,
    }
    if output_manifest.is_file():
        _require(load_json(output_manifest) == manifest,
                 f"existing manifest differs: {output_manifest}")
    else:
        atomic_write_json(output_manifest, manifest)
    return manifest

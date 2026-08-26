"""Unified Full-Rubric Worker ablation on internal data and VL-RewardBench.

Every sample/replicate receives exactly one model call.  The complete frozen
Phase17 E4 rubric is placed in the static system prompt and treated as one
decision policy.  There are no node workers, subtree workers, gates, routers,
root votes, arbiters, or fallback model calls.
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
from critiq.structured.backend_pool import BackendEndpointSpec, BackendPoolSpec
from critiq.utils import parse_json

from . import _global_arbiter_ab_only_support as support
from . import internal_global_arbiter_k1 as internal
from . import run_rubric_evolution as base
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_phase10 as vlrb_metrics
from .experiment_utils import atomic_write_json, canonical_sha256, load_json
from .rubric_factory import file_sha256


SCHEMA_VERSION = "1.0.0"
PROTOCOL_VERSION = "unified-full-rubric-worker-v1"
PROMPT_VERSION = "pairwise-full-rubric-unified-v1"
EXPERIMENT_DIR = "full_rubric_unified_worker_v1"
CONFIG_KEY = "full_rubric_unified_worker_experiment"
SOURCE_EXPERIMENT = support.SOURCE_RUBRIC_EXPERIMENT
SOURCE_EPOCH = support.SOURCE_RUBRIC_EPOCH
SOURCE_RUBRIC_SHA256 = (
    "007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d")
CLEAN_S5_EXPERIMENT = "vl_rewardbench_global_arbiter_ab_preferred_none_v2"
CLEAN_S5_SYSTEM = "s5_v2_global_arbiter_ab_preferred_none_tolerant"
SYSTEM_NAME = "s6_unified_full_rubric"
ENDPOINT_IDS = support.ENDPOINT_IDS
INTERNAL_COUNTS = dict(internal.SPLIT_COUNTS)
VLRB_COUNT = support.VLRB_COUNT
VLRB_K = support.K

STAGES = (
    "full-rubric-worker-freeze",
    "full-rubric-worker-audit",
    "full-rubric-worker-smoke",
    "full-rubric-worker-internal-run",
    "full-rubric-worker-vlrb-run",
    "full-rubric-worker-retry",
    "full-rubric-worker-report",
)

SYSTEM_HEADER = """## Instruction

You are judging a multimodal image-text preference pair using one complete structured rubric. You are given the image, the source instruction or question, and two candidate responses.

Treat the full rubric hierarchy as one unified decision policy, not as independent votes. Use only the criteria that bear on the actual difference between the responses. Resolve conflicts using direct visual evidence, task requirements, factual validity, error severity, and likely human preference. Completeness, clarity, and creativity cannot compensate for a material factual or visual error.

Use the image whenever visual evidence is relevant. Return a relative preference for the pair. If both responses are imperfect or the evidence is limited, choose the response with stronger support and the less severe error.

"""

SYSTEM_FOOTER = """

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A using the relevant rubric evidence.",
    "analysis_b": "Analyze B using the relevant rubric evidence.",
    "thought": "Compare A and B and resolve any criterion conflicts.",
    "answer": "A / B"
}
```

Return exactly one JSON object and no additional prose. Do not output criterion-level scores, votes, routing decisions, or a list of applicable criteria.
"""


def _target(output: Path) -> Path:
    return output.parent / EXPERIMENT_DIR


def _source_target(output: Path) -> Path:
    return output / SOURCE_EXPERIMENT


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_epoch": SOURCE_EPOCH,
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
        "single_model_call_per_sample_replicate": True,
        "selection_after_diagnostics_forbidden": True,
    }
    value = config.get(CONFIG_KEY)
    if value != expected:
        raise RuntimeError(f"{CONFIG_KEY} does not match the frozen protocol")
    return dict(value)


def _rubric(output: Path) -> StructuredRubric:
    rubric = support.rubric(output)
    if rubric.rubric_sha256 != SOURCE_RUBRIC_SHA256:
        raise RuntimeError("Unified Full-Rubric source hash drift")
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 27:
        raise RuntimeError("Unified Full-Rubric requires Phase17 E4 5-root/27-node rubric")
    return rubric


def _serialize_node(rubric: StructuredRubric, node_id: str) -> dict[str, Any]:
    node = rubric.get_node(node_id)
    return {
        "node_id": node_id,
        "criterion_name": node.criterion.name,
        "description": node.criterion.description,
        "children": [
            {
                "edge_condition": edge.condition.value,
                "node": _serialize_node(rubric, edge.child_id),
            }
            for edge in rubric.child_edges(node_id)
        ],
    }


def full_rubric_serialization(rubric: StructuredRubric) -> dict[str, Any]:
    """Return the canonical prompt-facing hierarchy without training evidence."""

    return {
        "schema_version": SCHEMA_VERSION,
        "rubric_sha256": rubric.rubric_sha256,
        "root_ids": list(rubric.root_ids),
        "roots": [_serialize_node(rubric, root_id) for root_id in rubric.root_ids],
    }


def _render_node(node: Mapping[str, Any], depth: int, index: int) -> list[str]:
    heading = "#" * min(6, depth + 3)
    role = "Root" if depth == 0 else "Specialized criterion"
    lines = [
        f"{heading} {role} {index}: {node['criterion_name']}",
        str(node["description"]),
    ]
    for child_index, child in enumerate(node["children"], start=1):
        lines.extend(["", *_render_node(child["node"], depth + 1, child_index)])
    return lines


def render_full_rubric(serialization: Mapping[str, Any]) -> str:
    lines = ["## Complete Structured Rubric"]
    for index, root in enumerate(serialization["roots"], start=1):
        lines.extend(["", *_render_node(root, 0, index)])
    return "\n".join(lines)


def full_rubric_system_prompt(rubric: StructuredRubric) -> str:
    return SYSTEM_HEADER + render_full_rubric(
        full_rubric_serialization(rubric)) + SYSTEM_FOOTER


def full_rubric_user_prompt(row: Mapping[str, Any]) -> str:
    return "\n".join([
        "## Source Instruction or Question", str(row["question"]), "",
        "## Candidate A", str(row["A"]), "",
        "## Candidate B", str(row["B"]), "",
        "Which candidate better follows the complete structured rubric and is "
        "more likely to align with human preference?",
    ])


def parse_full_rubric_response(raw: object) -> dict[str, str]:
    if not isinstance(raw, str):
        raise ValueError("Full-Rubric response must be text")
    payload = parse_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("Full-Rubric response must be one JSON object")
    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise ValueError("Full-Rubric answer must be text")
    normalized = answer.strip()
    if normalized.upper() in {"A", "B"}:
        normalized = normalized.upper()
    elif normalized.lower() == "none":
        normalized = "None"
    else:
        raise ValueError("Full-Rubric answer must be A, B, or None")
    return {
        "analysis_a": payload.get("analysis_a")
        if isinstance(payload.get("analysis_a"), str) else "",
        "analysis_b": payload.get("analysis_b")
        if isinstance(payload.get("analysis_b"), str) else "",
        "thought": payload.get("thought")
        if isinstance(payload.get("thought"), str) else "",
        "answer": normalized,
    }


def _internal_rows(config: Mapping[str, Any], split: str) -> tuple[dict[str, Any], ...]:
    rows = internal._rows(config, split)
    if len(rows) != INTERNAL_COUNTS[split]:
        raise RuntimeError(f"Unified Full-Rubric {split} count drift")
    return rows


def _internal_dataset_path(config: Mapping[str, Any], split: str) -> Path:
    return internal._dataset_path(config, split)


def _vlrb_records(output: Path) -> tuple[dict[str, Any], ...]:
    return support.records(output)


def _vlrb_controls(
    output: Path, records: Sequence[Mapping[str, Any]], rubric: StructuredRubric,
) -> tuple[dict[str, list[list[int | None]]], dict[str, Any]]:
    s3_path = output.parent / support.SOURCE_S3_EXPERIMENT / "final_report.json"
    clean_path = output.parent / CLEAN_S5_EXPERIMENT / "final_report.json"
    s3_manifest = load_json(
        output.parent / support.SOURCE_S3_EXPERIMENT / "frozen_manifest.json")
    clean_manifest = load_json(
        output.parent / CLEAN_S5_EXPERIMENT / "frozen_manifest.json")
    if (s3_manifest.get("rubric_sha256") != rubric.rubric_sha256
            or clean_manifest.get("rubric_sha256") != rubric.rubric_sha256):
        raise RuntimeError("Unified Full-Rubric VL-RewardBench control rubric drift")
    controls = {
        "s0_explicit_recursive": support._validated_system(
            s3_path, "s0_explicit_recursive", records),
        "s3_unified_subtree": support._validated_system(
            s3_path, "s3_unified_subtree", records),
        "clean_s5_v2_global_arbiter": support._validated_system(
            clean_path, CLEAN_S5_SYSTEM, records),
    }
    return controls, {
        "s3_experiment": support.SOURCE_S3_EXPERIMENT,
        "clean_s5_experiment": CLEAN_S5_EXPERIMENT,
        "s3_report_sha256": file_sha256(s3_path),
        "clean_s5_report_sha256": file_sha256(clean_path),
        "metrics_recomputed_exact": True,
    }


def _source_artifacts(config: Mapping[str, Any], output: Path) -> dict[str, str]:
    source = _source_target(output)
    epoch = source / "epochs" / f"epoch_{SOURCE_EPOCH:02d}"
    return {
        "rubric": file_sha256(epoch / "rubric_committed.json"),
        "discovery_prediction": file_sha256(epoch / "discovery_pairwise.json"),
        "dev_prediction": file_sha256(epoch / "dev150" / "pairwise.json"),
        "heldout_prediction": file_sha256(source / "heldout500" / "combined_pairwise.json"),
        "vlrb_s3_report": file_sha256(
            output.parent / support.SOURCE_S3_EXPERIMENT / "final_report.json"),
        "vlrb_clean_s5_report": file_sha256(
            output.parent / CLEAN_S5_EXPERIMENT / "final_report.json"),
        "vlrb_schedule": file_sha256(
            output.parent / support.SOURCE_SCHEDULE_EXPERIMENT / "order_schedule.json"),
    }


def _manifest(
    config: Mapping[str, Any], output: Path, *, include_endpoints: bool,
) -> dict[str, Any]:
    settings = _settings(config)
    rubric = _rubric(output)
    serialization = full_rubric_serialization(rubric)
    system_prompt = full_rubric_system_prompt(rubric)
    records = _vlrb_records(output)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in records])
    controls, control_provenance = _vlrb_controls(output, records, rubric)
    datasets = {
        split: {
            "count": INTERNAL_COUNTS[split],
            "path": str(_internal_dataset_path(config, split).resolve()),
            "sha256": file_sha256(_internal_dataset_path(config, split)),
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
        "prompt_version": PROMPT_VERSION,
        "settings": settings,
        "settings_sha256": canonical_sha256(settings),
        "rubric_sha256": rubric.rubric_sha256,
        "root_ids": list(rubric.root_ids),
        "node_ids_in_preorder": list(rubric.preorder_node_ids()),
        "node_count": len(rubric.nodes),
        "edge_count": len(rubric.edges),
        "rubric_serialization_sha256": canonical_sha256(serialization),
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")).hexdigest(),
        "datasets": datasets,
        "vlrb_schedule_sha256": canonical_sha256(schedule),
        "vlrb_control_vote_matrices_sha256": canonical_sha256(controls),
        "control_provenance": control_provenance,
        "source_artifacts": _source_artifacts(config, output),
        "logical_request_budget": {
            "internal": sum(INTERNAL_COUNTS.values()),
            "vl_rewardbench": len(records) * VLRB_K,
            "total": sum(INTERNAL_COUNTS.values()) + len(records) * VLRB_K,
        },
        "execution": {
            "model_calls_per_sample_replicate": 1,
            "node_workers": 0,
            "subtree_workers": 0,
            "gate_workers": 0,
            "root_routers": 0,
            "global_arbiters": 0,
            "fallback_model_calls": 0,
        },
    }
    if include_endpoints:
        value["endpoint_identities"] = base._inspect_endpoints(
            config, BackendPoolSpec.from_dict(config["backend_pool"]))
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
        raise RuntimeError("Unified Full-Rubric frozen manifest drift")
    rubric = _rubric(output)
    serialization = full_rubric_serialization(rubric)
    system_prompt = full_rubric_system_prompt(rubric)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in _vlrb_records(output)])
    atomic_write_json(path, manifest)
    rubric.save_json(target / "rubric_snapshot.json")
    atomic_write_json(target / "rubric_serialization.json", serialization)
    atomic_write_json(target / "prompt_spec.json", {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system_prompt,
        "user_prompt_order": ["question", "A", "B"],
        "parser": "A_or_B_or_semantic_None_with_optional_reasoning_fields",
        "retry": "same_prompt_technical_failures_only",
    })
    atomic_write_json(target / "schedules" / "internal_k1.json", {
        split: {str(row["sample_id"]): [0] for row in _internal_rows(config, split)}
        for split in INTERNAL_COUNTS
    })
    atomic_write_json(target / "schedules" / "vlrb_k3.json", schedule)
    details = {
        "rubric_sha256": rubric.rubric_sha256,
        "node_count": len(rubric.nodes),
        "internal_request_count": sum(INTERNAL_COUNTS.values()),
        "vlrb_request_count": len(schedule) * VLRB_K,
        "total_logical_request_count": manifest["logical_request_budget"]["total"],
    }
    _status(target, STAGES[0], details)
    print(json.dumps(details, indent=2))


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target(output)
    _require(target, STAGES[0])
    manifest = load_json(target / "frozen_manifest.json")
    expected = _manifest(config, output, include_endpoints=False)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Unified Full-Rubric manifest drift: {key}")
    rubric = _rubric(output)
    if StructuredRubric.load_json(
            target / "rubric_snapshot.json").rubric_sha256 != rubric.rubric_sha256:
        raise RuntimeError("Unified Full-Rubric snapshot drift")
    if load_json(target / "rubric_serialization.json") != full_rubric_serialization(rubric):
        raise RuntimeError("Unified Full-Rubric serialization drift")
    return target, manifest, rubric


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    actual = base._inspect_endpoints(
        config, BackendPoolSpec.from_dict(config["backend_pool"]))
    if actual != manifest.get("endpoint_identities"):
        raise RuntimeError("Unified Full-Rubric endpoint identity drift")


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    serialization = full_rubric_serialization(rubric)
    serialized_ids: list[str] = []

    def visit(node: Mapping[str, Any]) -> None:
        serialized_ids.append(str(node["node_id"]))
        for child in node["children"]:
            visit(child["node"])

    for root in serialization["roots"]:
        visit(root)
    sentinel = dict(_internal_rows(config, "discovery100")[0])
    sentinel["answer"] = "__GOLD_SENTINEL__"
    user_prompt = full_rubric_user_prompt(sentinel)
    system_prompt = full_rubric_system_prompt(rubric)
    controls, _ = _vlrb_controls(output, _vlrb_records(output), rubric)
    checks = {
        "phase17_e4_rubric": rubric.rubric_sha256 == SOURCE_RUBRIC_SHA256,
        "five_roots_27_nodes": len(rubric.root_ids) == 5 and len(rubric.nodes) == 27,
        "all_nodes_once_in_serialization": serialized_ids
            == list(rubric.preorder_node_ids()) and len(set(serialized_ids)) == 27,
        "all_descriptions_in_system_prompt": all(
            node.criterion.description in system_prompt
            for node in rubric.nodes.values()),
        "rubric_only_in_static_prompt": "## Complete Structured Rubric" in system_prompt
            and "## Complete Structured Rubric" not in user_prompt,
        "dynamic_pair_in_user_prompt": str(sentinel["A"]) in user_prompt
            and str(sentinel["B"]) in user_prompt,
        "gold_not_in_prompt": "__GOLD_SENTINEL__" not in user_prompt,
        "four_field_schema": all(
            f'"{key}"' in system_prompt
            for key in ("analysis_a", "analysis_b", "thought", "answer")),
        "prompt_requests_ab": '"answer": "A / B"' in system_prompt,
        "parser_accepts_a": parse_full_rubric_response(
            '{"answer":"A"}')["answer"] == "A",
        "parser_accepts_b": parse_full_rubric_response(
            '{"answer":"B"}')["answer"] == "B",
        "parser_accepts_semantic_none": parse_full_rubric_response(
            '{"answer":"None"}')["answer"] == "None",
        "one_call_execution": manifest["execution"] == {
            "model_calls_per_sample_replicate": 1,
            "node_workers": 0,
            "subtree_workers": 0,
            "gate_workers": 0,
            "root_routers": 0,
            "global_arbiters": 0,
            "fallback_model_calls": 0,
        },
        "internal_budget_750": manifest["logical_request_budget"]["internal"] == 750,
        "vlrb_budget_3741": manifest["logical_request_budget"]["vl_rewardbench"] == 3741,
        "controls_exact": set(controls) == {
            "s0_explicit_recursive", "s3_unified_subtree",
            "clean_s5_v2_global_arbiter"},
        "vlrb_schedule_k3": all(
            tuple(value) in {(0, 1, 0), (1, 0, 1)}
            for value in load_json(target / "schedules" / "vlrb_k3.json").values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Unified Full-Rubric audit failed: {checks}")
    atomic_write_json(target / "offline_audit.json", {
        "schema_version": SCHEMA_VERSION, "offline_only": True, "checks": checks})
    _status(target, STAGES[1], checks)
    print(json.dumps(checks, indent=2))


def _call_one(
    config: Mapping[str, Any], target: Path, endpoint: BackendEndpointSpec,
    split: str, row: Mapping[str, Any], rubric: StructuredRubric,
    sample_id: str, replicate: int, order: int, total_attempt_limit: int,
) -> dict[str, Any]:
    serialization_sha = canonical_sha256(full_rubric_serialization(rubric))
    return support.call_one(
        config, endpoint, target / "cache" / split,
        user_text=full_rubric_user_prompt(row), row=row,
        request_key={
            "kind": "unified_full_rubric_worker",
            "split": split,
            "sample_id": sample_id,
            "replicate": replicate,
            "order": order,
            "rubric_serialization_sha256": serialization_sha,
        },
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=PROMPT_VERSION,
        system_prompt=full_rubric_system_prompt(rubric),
        response_parser=parse_full_rubric_response,
        settings_loader=_settings,
    )


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
        "calls": {},
    }

    def invoke(replicate: int) -> tuple[int, dict[str, Any]]:
        order = int(orders[replicate])
        displayed = support.ordered_row(row, order, replicate)
        call = _call_one(
            config, target, endpoint, split, displayed, rubric,
            sample_id, replicate, order, total_attempt_limit)
        return replicate, call

    with ThreadPoolExecutor(
            max_workers=min(endpoint.max_concurrency, len(orders))) as executor:
        futures = [executor.submit(invoke, replicate)
                   for replicate in range(len(orders))]
        for future in as_completed(futures):
            replicate, call = future.result()
            result["calls"][str(replicate)] = support.compact_call(call)
    atomic_write_json(
        target / "bundles" / label / f"{canonical_sha256(sample_id)}.json",
        result)
    return result


def _run_rows(
    config: Mapping[str, Any], target: Path, split: str, label: str,
    rows: Sequence[Mapping[str, Any]], orders_by_id: Mapping[str, Sequence[int]],
    rubric: StructuredRubric, total_attempt_limit: int,
) -> dict[str, Any]:
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoints = tuple(
        endpoint for endpoint in pool.endpoints if endpoint.endpoint_id in ENDPOINT_IDS)
    if tuple(endpoint.endpoint_id for endpoint in endpoints) != ENDPOINT_IDS:
        raise RuntimeError("Unified Full-Rubric requires both frozen endpoints")
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
        "endpoint_call_counts": {},
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
                config, target, endpoint, split, row,
                orders_by_id[sample_id], rubric, total_attempt_limit, label)
            with lock:
                values[sample_id] = value
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(rows) - completed)
                endpoint_counts = Counter(
                    str(item["endpoint_id"]) for item in values.values())
                atomic_write_json(progress_path, {
                    "stage": label,
                    "completed": completed,
                    "total": len(rows),
                    "percent": completed / len(rows) * 100 if rows else 100.0,
                    "current_sample_id": sample_id,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "endpoint_call_counts": dict(endpoint_counts),
                })
                print(
                    f"{label}: {completed}/{len(rows)} sample={sample_id} "
                    f"endpoint={endpoint.endpoint_id} elapsed={elapsed/60:.1f}m "
                    f"ETA={eta/60:.1f}m", flush=True)

    max_replicates = max(len(value) for value in orders_by_id.values())
    workers = [endpoint for endpoint in endpoints
               for _ in range(max(1, endpoint.max_concurrency // max_replicates))]
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    pass_wall = time.perf_counter() - started
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "split": split,
        "label": label,
        "samples": [values[str(row["sample_id"])] for row in rows],
        "wall_seconds": pass_wall,
        "initial_main_run_wall_seconds": (
            previous_main_wall if previous_main_wall is not None else pass_wall),
    }
    atomic_write_json(prediction_path, result)
    return result


def _calls(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [call for sample in result["samples"] for call in sample["calls"].values()]


def _telemetry(result: Mapping[str, Any]) -> dict[str, Any]:
    calls = _calls(result)
    pass_wall = float(result.get("wall_seconds") or 0.0)
    main_wall = float(
        result.get("initial_main_run_wall_seconds") or pass_wall)
    by_endpoint = Counter(str(call.get("endpoint_id")) for call in calls)
    latencies = sorted(
        float(call.get("metrics", {}).get("latency_seconds") or 0.0)
        for call in calls)

    def percentile(fraction: float) -> float:
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, max(
            0, int((len(latencies) - 1) * fraction + 0.5)))
        return latencies[index]

    return {
        "logical_request_count": len(calls),
        "model_generation_count": sum(
            int(call.get("model_generation_count", 0)) for call in calls),
        "parse_valid_count": sum(bool(call.get("parse_ok")) for call in calls),
        "parse_valid_rate": (sum(bool(call.get("parse_ok")) for call in calls)
                             / len(calls) if calls else 0.0),
        "semantic_none_count": sum(
            call.get("parse_ok")
            and (call.get("parsed") or {}).get("answer") == "None"
            for call in calls),
        "endpoint_call_counts": dict(by_endpoint),
        "cache_hit_count": sum(bool(call.get("cache_hit")) for call in calls),
        "api_attempts": sum(
            int(call.get("metrics", {}).get("api_attempts") or 0) for call in calls),
        "errors": sum(
            int(call.get("metrics", {}).get("error_count") or 0) for call in calls),
        "input_tokens": sum(
            int(call.get("metrics", {}).get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(
            int(call.get("metrics", {}).get("output_tokens") or 0) for call in calls),
        "wall_seconds": main_wall,
        "main_run_wall_seconds": main_wall,
        "latest_pass_wall_seconds": pass_wall,
        "logical_requests_per_minute": (
            len(calls) / main_wall * 60 if main_wall else 0.0),
        "sample_bundles_per_minute": (
            len(result["samples"]) / main_wall * 60 if main_wall else 0.0),
        "latency_seconds": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": latencies[-1] if latencies else 0.0,
        },
    }


def _failures(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{
        "sample_id": sample["sample_id"],
        "replicate": int(replicate),
        "error": call.get("error"),
        "model_generation_count": call.get("model_generation_count", 0),
        "cache_key": call.get("cache_key"),
    } for sample in result["samples"]
        for replicate, call in sample["calls"].items()
        if not call.get("parse_ok")]


def _internal_orders(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int]]:
    return {str(row["sample_id"]): (0,) for row in rows}


def _vlrb_rows(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return support.vlrb_rows(records)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[1])
    _verify_live(config, manifest)
    settings = _settings(config)
    limit = 1 + settings["max_parse_retries"]
    details: dict[str, Any] = {}
    count = settings["smoke_internal_samples_per_split"]
    for split in INTERNAL_COUNTS:
        rows = _internal_rows(config, split)[:count]
        result = _run_rows(
            config, target, split, f"smoke_{split}", rows,
            _internal_orders(rows), rubric, limit)
        details[split] = {
            "sample_count": len(rows),
            "technical_failure_count": len(_failures(result)),
            "telemetry": _telemetry(result),
        }
    records = support.selected_records(
        _vlrb_records(output), settings["smoke_vlrb_sample_count"])
    rows = _vlrb_rows(records)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in _vlrb_records(output)])
    result = _run_rows(
        config, target, "vl_rewardbench", "smoke_vlrb20", rows,
        {str(row["sample_id"]): schedule[str(row["sample_id"])] for row in rows},
        rubric, limit)
    details["vl_rewardbench"] = {
        "sample_count": len(rows),
        "technical_failure_count": len(_failures(result)),
        "telemetry": _telemetry(result),
    }
    if any(value["technical_failure_count"] for value in details.values()):
        raise RuntimeError(f"Unified Full-Rubric smoke failed: {details}")
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
        config, target, "vl_rewardbench", "vlrb_full", rows, schedule,
        rubric, 1)
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
    limit = 1 + _settings(config)["max_parse_retries"]
    details = {}
    failures = {}
    for split in INTERNAL_COUNTS:
        rows = _internal_rows(config, split)
        result = _run_rows(
            config, target, split, f"internal_{split}", rows,
            _internal_orders(rows), rubric, limit)
        current = _failures(result)
        failures[split] = current
        details[split] = {
            "technical_failure_count": len(current),
            "telemetry": _telemetry(result),
        }
    records = _vlrb_records(output)
    rows = _vlrb_rows(records)
    schedule = support.source_schedule(
        output, [str(record["sample_id"]) for record in records])
    result = _run_rows(
        config, target, "vl_rewardbench", "vlrb_full", rows,
        schedule, rubric, limit)
    current = _failures(result)
    failures["vl_rewardbench"] = current
    details["vl_rewardbench"] = {
        "technical_failure_count": len(current),
        "telemetry": _telemetry(result),
    }
    atomic_write_json(target / "parse_failures.json", failures)
    atomic_write_json(target / "retry_summary.json", {
        "schema_version": SCHEMA_VERSION,
        "total_attempt_limit": limit,
        "splits": details,
    })
    _status(target, STAGES[5], details)
    print(json.dumps(details, indent=2))


def _normalized_answer(call: Mapping[str, Any]) -> str | None:
    if not call.get("parse_ok"):
        return None
    answer = (call.get("parsed") or {}).get("answer")
    return answer if answer in {"A", "B"} else None


def _internal_report(
    config: Mapping[str, Any], output: Path, target: Path,
    rubric: StructuredRubric,
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for split in INTERNAL_COUNTS:
        rows = _internal_rows(config, split)
        result = load_json(target / "predictions" / f"internal_{split}.json")
        treatment = [_normalized_answer(sample["calls"]["0"])
                     for sample in result["samples"]]
        technical = [not sample["calls"]["0"].get("parse_ok")
                     for sample in result["samples"]]
        control, provenance = internal._control_predictions(
            output, split, rubric, rows)
        control_name = (
            "phase17_e4_explicit_recursive"
            if provenance["exact_same_rubric_as_treatment"]
            else "phase17_e5_explicit_recursive_reference")
        systems = {
            control_name: internal._metrics(rows, control),
            SYSTEM_NAME: internal._metrics(rows, treatment, technical),
        }
        value[split] = {
            "systems": systems,
            "paired": internal._paired(rows, control, treatment),
            "control_provenance": provenance,
            "telemetry": _telemetry(result),
        }
    return value


def _vlrb_vote_matrix(result: Mapping[str, Any]) -> list[list[int | None]]:
    votes: list[list[int | None]] = [[] for _ in range(VLRB_K)]
    for sample in result["samples"]:
        if len(sample["orders"]) != VLRB_K or len(sample["calls"]) != VLRB_K:
            raise RuntimeError("Unified Full-Rubric VL-RB K=3 artifact is incomplete")
        for replicate in range(VLRB_K):
            call = sample["calls"][str(replicate)]
            answer = ((call.get("parsed") or {}).get("answer")
                      if call.get("parse_ok") else "None")
            votes[replicate].append(support.display_to_original(
                answer, int(sample["orders"][replicate])))
    return votes


def _vlrb_report(
    output: Path, target: Path, rubric: StructuredRubric,
) -> dict[str, Any]:
    records = _vlrb_records(output)
    controls, provenance = _vlrb_controls(output, records, rubric)
    result = load_json(target / "predictions" / "vlrb_full.json")
    treatment_votes = _vlrb_vote_matrix(result)
    votes_by_system = {**controls, SYSTEM_NAME: treatment_votes}
    systems = {
        name: {
            "votes_by_replicate": votes,
            "metrics": vlrb_metrics._system_metrics(records, votes),
        }
        for name, votes in votes_by_system.items()
    }
    treatment = systems[SYSTEM_NAME]["metrics"]["original_index_predictions"]
    paired = {
        f"s6_vs_{name}": vlrb._paired(
            records, item["metrics"]["original_index_predictions"], treatment)
        for name, item in systems.items() if name != SYSTEM_NAME
    }
    return {
        "systems": systems,
        "paired": paired,
        "control_provenance": provenance,
        "telemetry": _telemetry(result),
        "response_distribution_by_replicate": [
            dict(Counter("None" if vote is None else str(vote) for vote in row))
            for row in treatment_votes
        ],
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, _, rubric = _load_frozen(config, output)
    _require(target, STAGES[5])
    failures = load_json(target / "parse_failures.json")
    internal_report = _internal_report(config, output, target, rubric)
    vlrb_report = _vlrb_report(output, target, rubric)
    internal_wall = sum(
        float(item["telemetry"]["wall_seconds"])
        for item in internal_report.values())
    vlrb_wall = float(vlrb_report["telemetry"]["wall_seconds"])
    efficiency = {
        "logical_request_counts": {
            "internal_k1": sum(INTERNAL_COUNTS.values()),
            "s0_explicit_recursive_vlrb_k3": VLRB_COUNT * VLRB_K * len(rubric.nodes),
            "s3_unified_subtree_vlrb_k3": VLRB_COUNT * VLRB_K * len(rubric.root_ids),
            "clean_s5_v2_vlrb_k3": VLRB_COUNT * VLRB_K * (len(rubric.root_ids) + 1),
            "s6_unified_full_rubric_vlrb_k3": VLRB_COUNT * VLRB_K,
        },
        "request_reduction": {
            "s6_vs_s0": 1.0 - 1.0 / len(rubric.nodes),
            "s6_vs_s3": 1.0 - 1.0 / len(rubric.root_ids),
            "s6_vs_clean_s5_v2": 1.0 - 1.0 / (len(rubric.root_ids) + 1),
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
            "groups": system["metrics"]["groups"],
            "source_groups": system["metrics"]["source_groups"],
        }
        for name, system in vlrb_report["systems"].items()
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system": SYSTEM_NAME,
        "rubric_sha256": rubric.rubric_sha256,
        "internal": internal_report,
        "vl_rewardbench": vlrb_report,
        "efficiency": efficiency,
        "category_analysis": category_analysis,
        "unresolved_technical_failure_count": sum(
            len(items) for items in failures.values()),
        "selection_after_diagnostics_forbidden": True,
    }
    atomic_write_json(target / "reports" / "internal.json", internal_report)
    atomic_write_json(target / "reports" / "vl_rewardbench.json", vlrb_report)
    atomic_write_json(target / "paired_comparison.json", {
        "internal": {split: item["paired"]
                     for split, item in internal_report.items()},
        "vl_rewardbench": vlrb_report["paired"],
    })
    atomic_write_json(target / "category_analysis.json", category_analysis)
    atomic_write_json(target / "efficiency.json", efficiency)
    atomic_write_json(target / "final_report.json", value)

    lines = [
        "# Unified Full-Rubric Worker v1", "",
        "One model call per sample/replicate. The complete Phase17 E4 rubric is "
        "used as one decision policy; no node/subtree votes, routing, Arbiter, or fallback.",
        "", "## Internal K=1", "",
        "| Split | System | Strict ACC | OverallAcc | MacroAcc | Coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, item in internal_report.items():
        for name, metrics in item["systems"].items():
            macro = "-" if metrics["macro_acc"] is None else f"{metrics['macro_acc']:.2%}"
            lines.append(
                f"| {split} | {name} | {metrics['strict_accuracy']:.2%} | "
                f"{metrics['overall_acc']:.2%} | {macro} | {metrics['coverage']:.2%} |")
    lines.extend([
        "", "## VL-RewardBench K=3", "",
        "| System | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in vlrb_report["systems"].items():
        metrics = item["metrics"]
        lines.append(
            f"| {name} | {metrics['strict_accuracy']:.2%} | "
            f"{metrics['overall_acc']:.2%} | {metrics['macro_acc']:.2%} | "
            f"{metrics['coverage']:.2%} | {metrics['correct_count']} |")
    lines.extend([
        "", "## Paired VL-RewardBench comparisons", "",
        "| Comparison | Corrected | Harmed | Net | Exact McNemar p |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in vlrb_report["paired"].items():
        lines.append(
            f"| {name} | {item['corrected_count']} | {item['harmed_count']} | "
            f"{item['net_corrected']} | {item['mcnemar_exact_two_sided_p']:.6g} |")
    lines.extend([
        "", "## VL-RewardBench category Strict ACC", "",
        "| System | General | Hallucination | Reasoning |",
        "|---|---:|---:|---:|",
    ])
    for name, item in vlrb_report["systems"].items():
        groups = item["metrics"]["groups"]
        lines.append(
            f"| {name} | {groups['general']['strict_accuracy']:.2%} | "
            f"{groups['hallucination']['strict_accuracy']:.2%} | "
            f"{groups['reasoning']['strict_accuracy']:.2%} |")
    counts = efficiency["logical_request_counts"]
    reductions = efficiency["request_reduction"]
    lines.extend([
        "", "## Efficiency", "",
        "| System | VL-RB logical requests | Reduction vs S0 |",
        "|---|---:|---:|",
        f"| S0 Explicit Recursive | {counts['s0_explicit_recursive_vlrb_k3']:,} | - |",
        f"| S3 Unified Subtree | {counts['s3_unified_subtree_vlrb_k3']:,} | "
        f"{1-counts['s3_unified_subtree_vlrb_k3']/counts['s0_explicit_recursive_vlrb_k3']:.2%} |",
        f"| Clean S5-v2 | {counts['clean_s5_v2_vlrb_k3']:,} | "
        f"{1-counts['clean_s5_v2_vlrb_k3']/counts['s0_explicit_recursive_vlrb_k3']:.2%} |",
        f"| S6 Unified Full-Rubric | {counts['s6_unified_full_rubric_vlrb_k3']:,} | "
        f"{reductions['s6_vs_s0']:.2%} |",
        "", f"VL-RewardBench main-run wall time: {vlrb_wall:.1f}s; "
        f"throughput: {vlrb_report['telemetry']['logical_requests_per_minute']:.2f} "
        "requests/min and "
        f"{vlrb_report['telemetry']['sample_bundles_per_minute']:.2f} samples/min.",
        "", f"Unresolved technical failures: {value['unresolved_technical_failure_count']}.",
        "Semantic None is valid and lowers Coverage/Strict ACC; only technical failures are retried.",
    ])
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "internal": {
            split: item["systems"][SYSTEM_NAME]["strict_accuracy"]
            for split, item in internal_report.items()},
        "vl_rewardbench": {
            key: vlrb_report["systems"][SYSTEM_NAME]["metrics"][key]
            for key in ("strict_accuracy", "overall_acc", "macro_acc", "coverage")},
        "unresolved_technical_failures": value[
            "unresolved_technical_failure_count"],
        "vlrb_logical_requests": counts["s6_unified_full_rubric_vlrb_k3"],
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
        raise ValueError(f"unsupported Unified Full-Rubric stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

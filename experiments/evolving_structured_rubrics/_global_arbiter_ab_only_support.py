"""Private runtime helpers for the final A/B-only Global Arbiter.

The earlier Implicit-All, Unified-Subtree, and abstaining Arbiter runners were
exploratory ablations.  This module keeps only the small read-only bridge that
the final experiment needs to consume their frozen S3/S4 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from critiq.agent import Agent, AgentCallMetrics
from critiq.evaluator import MultiModalPairEvaluator
from critiq.structured import StructuredRubric
from critiq.structured.aggregation import aggregate_selected_roots
from critiq.structured.backend_pool import BackendEndpointSpec, BackendPoolSpec
from critiq.structured.judgement import FinalPreference, Vote
from critiq.utils import parse_json

from . import vl_rewardbench as vlrb
from . import vl_rewardbench_phase10 as vlrb_metrics
from .experiment_utils import atomic_write_json, canonical_sha256, load_json
from .rubric_factory import file_sha256


SCHEMA_VERSION = "1.0.0"
K = 3
VLRB_COUNT = 1247
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")

SOURCE_RUBRIC_EXPERIMENT = "phase17_discovery_v2_prompt_v2_split_refine_v1"
SOURCE_RUBRIC_EPOCH = 4
SOURCE_SCHEDULE_EXPERIMENT = "vl_rewardbench_phase17_checkpoint_transfer_v1"
SOURCE_S3_EXPERIMENT = "vl_rewardbench_implicit_unified_subtree_prompt_v1"
SOURCE_S3_PROTOCOL = "implicit-unified-subtree-v1-vlrb-only"
SOURCE_S3_PROMPT = "implicit-unified-subtree-direct-judge-v1"
SOURCE_S4_EXPERIMENT = "vl_rewardbench_global_arbiter_v1"
SOURCE_S4_PROTOCOL = "global-arbiter-aggregation-v1-vlrb-only"
SOURCE_S4_PROMPT = "global-arbiter-evidence-synthesis-v1"


def rubric(output: Path) -> StructuredRubric:
    path = (output / SOURCE_RUBRIC_EXPERIMENT / "epochs"
            / f"epoch_{SOURCE_RUBRIC_EPOCH:02d}" / "rubric_committed.json")
    if not path.is_file():
        raise RuntimeError(f"Phase17 E4 rubric is missing: {path}")
    value = StructuredRubric.load_json(path)
    if len(value.root_ids) != 5 or len(value.nodes) != 27:
        raise RuntimeError("A/B-only Arbiter requires the frozen 5-root/27-node rubric")
    return value


def records(output: Path) -> tuple[dict[str, Any], ...]:
    value = vlrb._read_records(output.parent / SOURCE_S3_EXPERIMENT)
    if len(value) != VLRB_COUNT:
        raise RuntimeError("A/B-only Arbiter requires 1,247 VL-RewardBench rows")
    return value


def vlrb_rows(items: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple({
        "sample_id": str(item["sample_id"]),
        "image_path": str(item["image_path"]),
        "question": str(item["question"]),
        "A": str(item["responses"][0]),
        "B": str(item["responses"][1]),
        "answer": "A" if int(item["preferred_original_index"]) == 0 else "B",
    } for item in items)


def source_schedule(output: Path, sample_ids: Sequence[str]) -> dict[str, tuple[int, ...]]:
    path = output.parent / SOURCE_SCHEDULE_EXPERIMENT / "order_schedule.json"
    if not path.is_file():
        raise RuntimeError(f"Phase17 E4 A/B schedule is missing: {path}")
    raw = load_json(path)
    if not isinstance(raw, dict) or set(raw) != set(sample_ids):
        raise RuntimeError("Phase17 E4 A/B schedule identity drift")
    result = {sample_id: tuple(int(item) for item in raw[sample_id])
              for sample_id in sample_ids}
    if any(value not in {(0, 1, 0), (1, 0, 1)} for value in result.values()):
        raise RuntimeError("Phase17 E4 schedule is not frozen ABA/BAB K=3")
    return result


def ordered_row(row: Mapping[str, Any], order: int, replicate: int) -> dict[str, Any]:
    result = dict(row)
    result["sample_id"] = f"{row['sample_id']}::k{replicate + 1}"
    if order:
        result["A"], result["B"] = str(row["B"]), str(row["A"])
        result["answer"] = "B" if row["answer"] == "A" else "A"
    return result


def display_to_original(value: str, order: int) -> int | None:
    if value == "A":
        return order
    if value == "B":
        return 1 - order
    return None


def content(row: Mapping[str, Any], user_text: str) -> list[dict[str, Any]]:
    image_url = MultiModalPairEvaluator._image_path_to_data_url(str(row["image_path"]))
    return [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": user_text},
    ]


def metric_dict(metrics: AgentCallMetrics | None) -> dict[str, Any]:
    return dict((metrics or AgentCallMetrics()).__dict__)


def sum_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        key: sum(float(item.get(key) or 0) for item in items)
        for key in ("api_attempts", "error_count", "input_tokens", "output_tokens",
                    "latency_seconds")
    }
    for key in ("api_attempts", "error_count", "input_tokens", "output_tokens"):
        result[key] = int(result[key])
    return result


def compact_call(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "parse_ok", "parsed", "raw_response", "model_generation_count",
        "endpoint_id", "cache_hit", "metrics", "error", "cache_key",
    )
    return {key: value[key] for key in keys if key in value}


def global_arbiter_user_prompt(
    row: Mapping[str, Any], reports: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "## Source Instruction or Question", str(row["question"]), "",
        "## Candidate A", str(row["A"]), "",
        "## Candidate B", str(row["B"]), "",
        "## Subtree Assessments",
    ]
    for item in reports:
        lines.extend([
            "", f"### {item['criterion_name']}",
            json.dumps(item["report"], ensure_ascii=False, separators=(",", ":")),
        ])
    lines.extend([
        "", "Integrate the image evidence and all subtree assessments into one final preference.",
    ])
    return "\n".join(lines)


def _source_target(output: Path) -> Path:
    return output.parent / SOURCE_S3_EXPERIMENT


def source_prediction(output: Path) -> dict[str, Any]:
    path = _source_target(output) / "predictions" / "vlrb_full.json"
    if not path.is_file():
        raise RuntimeError("frozen S3 Unified-Subtree prediction is missing")
    value = load_json(path)
    if (value.get("protocol_version") != SOURCE_S3_PROTOCOL
            or value.get("prompt_version") != SOURCE_S3_PROMPT):
        raise RuntimeError("frozen S3 prediction identity drift")
    return value


def _source_cache_artifact(output: Path, cache_key: str) -> dict[str, Any]:
    path = (_source_target(output) / "cache" / "vlrb_full" / cache_key[:2]
            / f"{cache_key}.json")
    if not path.is_file():
        raise RuntimeError(f"S3 source cache artifact is missing: {cache_key}")
    value = load_json(path)
    if value.get("cache_key") != cache_key or not value.get("parse_ok"):
        raise RuntimeError(f"S3 source cache artifact is invalid: {cache_key}")
    return value


def _full_report(raw: object, expected_answer: str) -> dict[str, str]:
    if not isinstance(raw, str):
        raise RuntimeError("S3 source report raw response is missing")
    payload = parse_json(raw)
    if not isinstance(payload, dict) or payload.get("answer") != expected_answer:
        raise RuntimeError("S3 report does not match its compact prediction")
    return {
        "analysis_a": payload.get("analysis_a") if isinstance(payload.get("analysis_a"), str) else "",
        "analysis_b": payload.get("analysis_b") if isinstance(payload.get("analysis_b"), str) else "",
        "thought": payload.get("thought") if isinstance(payload.get("thought"), str) else "",
        "answer": expected_answer,
    }


def source_report_bundle(
    output: Path, sample: Mapping[str, Any], replicate: int,
    source_rubric: StructuredRubric,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    roots = sample.get("unified", {}).get(str(replicate), {})
    reports = []
    cache_keys = []
    for root_id in source_rubric.root_ids:
        compact = roots.get(root_id)
        if not isinstance(compact, dict) or not compact.get("parse_ok"):
            raise RuntimeError(f"S3 report missing: {sample.get('sample_id')} r{replicate}")
        answer = (compact.get("parsed") or {}).get("answer")
        cache_key = compact.get("cache_key")
        if answer not in {"A", "B", "None"} or not isinstance(cache_key, str):
            raise RuntimeError("S3 compact report is invalid")
        cached = _source_cache_artifact(output, cache_key)
        request = cached.get("request", {})
        if (request.get("protocol_version") != SOURCE_S3_PROTOCOL
                or request.get("prompt_version") != SOURCE_S3_PROMPT
                or request.get("request_key") != {
                    "kind": "unified_subtree",
                    "sample_id": str(sample["sample_id"]),
                    "replicate": replicate,
                    "root_id": root_id,
                }):
            raise RuntimeError("S3 source request identity drift")
        reports.append({
            "root_id": root_id,
            "criterion_name": source_rubric.get_node(root_id).criterion.name,
            "report": _full_report(cached.get("raw_response"), answer),
        })
        cache_keys.append(cache_key)
    digest = canonical_sha256({
        "sample_id": sample["sample_id"],
        "replicate": replicate,
        "order": int(sample["orders"][replicate]),
        "reports": reports,
    })
    return reports, digest, cache_keys


def source_index(
    output: Path, items: Sequence[Mapping[str, Any]], source_rubric: StructuredRubric,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction = source_prediction(output)
    samples = prediction.get("samples", [])
    ids = [str(item["sample_id"]) for item in items]
    if [str(sample.get("sample_id")) for sample in samples] != ids:
        raise RuntimeError("S3 source sample ordering drift")
    schedule = source_schedule(output, ids)
    entries = []
    by_id = {}
    full_fields = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if tuple(sample.get("orders", ())) != schedule[sample_id]:
            raise RuntimeError("S3 source A/B schedule drift")
        by_id[sample_id] = sample
        for replicate in range(K):
            reports, digest, keys = source_report_bundle(
                output, sample, replicate, source_rubric)
            full_fields += sum(all(report["report"][key].strip()
                                   for key in ("analysis_a", "analysis_b", "thought"))
                               for report in reports)
            entries.append({
                "sample_id": sample_id,
                "replicate": replicate,
                "order": int(sample["orders"][replicate]),
                "root_ids": list(source_rubric.root_ids),
                "source_cache_keys": keys,
                "report_bundle_sha256": digest,
            })
    return by_id, {
        "schema_version": SCHEMA_VERSION,
        "source_experiment": SOURCE_S3_EXPERIMENT,
        "sample_count": len(samples),
        "logical_bundle_count": len(entries),
        "source_root_report_count": len(entries) * len(source_rubric.root_ids),
        "full_field_report_count": full_fields,
        "entries": entries,
    }


def _validated_system(
    path: Path, name: str, items: Sequence[Mapping[str, Any]],
) -> list[list[int | None]]:
    report = load_json(path)
    system = report.get("systems", {}).get(name, {})
    votes = system.get("votes_by_replicate")
    if (not isinstance(votes, list) or len(votes) != K
            or any(len(row) != len(items) for row in votes)):
        raise RuntimeError(f"frozen {name} vote matrix is incomplete")
    recomputed = vlrb_metrics._system_metrics(items, votes)
    stored = system.get("metrics", {})
    for key in ("strict_accuracy", "overall_acc", "macro_acc", "coverage"):
        if stored.get(key) != recomputed.get(key):
            raise RuntimeError(f"frozen {name} metric drift: {key}")
    return votes


def source_controls(
    output: Path, items: Sequence[Mapping[str, Any]], source_rubric: StructuredRubric,
) -> tuple[dict[str, list[list[int | None]]], dict[str, Any]]:
    s3_path = _source_target(output) / "final_report.json"
    s4_path = output.parent / SOURCE_S4_EXPERIMENT / "final_report.json"
    if not s3_path.is_file() or not s4_path.is_file():
        raise RuntimeError("frozen S3/S4 control reports are missing")
    controls = {
        "s0_explicit_recursive": _validated_system(
            s3_path, "s0_explicit_recursive", items),
        "s3_unified_subtree": _validated_system(
            s3_path, "s3_unified_subtree", items),
        "s4_global_arbiter": _validated_system(
            s4_path, "s4_global_arbiter", items),
    }
    return controls, {
        "s3_experiment": SOURCE_S3_EXPERIMENT,
        "s4_experiment": SOURCE_S4_EXPERIMENT,
        "s3_report_sha256": file_sha256(s3_path),
        "s4_report_sha256": file_sha256(s4_path),
        "metrics_recomputed_exact": True,
    }


def status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {
        "schema_version": SCHEMA_VERSION, "stages": {}}
    value["stages"][stage] = {"status": "completed", "details": dict(details)}
    atomic_write_json(path, value)


def require(target: Path, stage: str) -> None:
    path = target / "stage_status.json"
    if (not path.is_file()
            or load_json(path).get("stages", {}).get(stage, {}).get("status") != "completed"):
        raise RuntimeError(f"run {stage} first")


def call_one(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, cache_dir: Path,
    *, user_text: str, row: Mapping[str, Any], request_key: Mapping[str, Any],
    total_attempt_limit: int, protocol_version: str, prompt_version: str,
    system_prompt: str, response_parser: Callable[[object], dict[str, str]],
    settings_loader: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    settings = settings_loader(config)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": protocol_version,
        "prompt_version": prompt_version,
        "model": config["model"],
        "endpoint_checkpoint": endpoint.checkpoint_root,
        "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
        "image_sha256": file_sha256(Path(str(row["image_path"]))),
        "request": {
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "generation_seed_policy": settings["generation_seed_policy"],
        },
        "request_key": dict(request_key),
    }
    key = canonical_sha256(payload)
    path = cache_dir / key[:2] / f"{key}.json"
    cached = None
    if path.is_file():
        candidate = load_json(path)
        if candidate.get("request") == payload:
            cached = candidate
            if candidate.get("parse_ok"):
                candidate["cache_hit"] = True
                return candidate
            try:
                parsed = response_parser(candidate.get("raw_response"))
            except Exception:
                pass
            else:
                candidate.update({
                    "parse_ok": True, "parsed": parsed, "cache_hit": True,
                    "parser_recovered": True,
                })
                candidate.pop("error", None)
                atomic_write_json(path, candidate)
                return candidate

    attempts = list((cached or {}).get("attempts", []))
    generations = int((cached or {}).get("model_generation_count", len(attempts)))
    metric_items = list((cached or {}).get("attempt_metrics", []))
    last_raw = (cached or {}).get("raw_response")
    last_error = (cached or {}).get("error")
    for _ in range(max(0, total_attempt_limit - generations)):
        attempt_number = generations + 1
        agent = Agent(
            system=system_prompt,
            model=config["model"],
            base_url=endpoint.base_url,
            api_keys="EMPTY",
            request_kwargs={
                "temperature": settings["temperature"],
                "max_tokens": settings["max_tokens"],
            },
            api_retry_attempts=config["api_retry_attempts"],
        )
        raw = None
        try:
            raw = agent(content(row, user_text), stream=False)
            parsed = response_parser(raw)
        except Exception as exc:
            last_raw = raw
            last_error = f"{type(exc).__name__}: {exc}"
            metric_items.append(metric_dict(agent.last_call_metrics))
            attempts.append({
                "attempt": attempt_number, "error": last_error,
                "raw_response": last_raw,
            })
            generations += 1
            continue
        metric_items.append(metric_dict(agent.last_call_metrics))
        result = {
            "request": payload, "parse_ok": True, "parsed": parsed,
            "raw_response": raw, "model_generation_count": attempt_number,
            "endpoint_id": endpoint.endpoint_id, "cache_hit": False,
            "metrics": sum_metrics(metric_items), "attempt_metrics": metric_items,
            "attempts": attempts, "cache_key": key,
        }
        atomic_write_json(path, result)
        return result
    result = {
        "request": payload, "parse_ok": False, "parsed": None,
        "raw_response": last_raw, "model_generation_count": generations,
        "endpoint_id": endpoint.endpoint_id, "cache_hit": bool(cached),
        "metrics": sum_metrics(metric_items), "attempt_metrics": metric_items,
        "attempts": attempts, "error": last_error or "parse failure",
        "cache_key": key,
    }
    atomic_write_json(path, result)
    return result


def _process_bundle(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, target: Path,
    source_output: Path, row: Mapping[str, Any], source_sample: Mapping[str, Any],
    source_rubric: StructuredRubric, *, label: str, total_attempt_limit: int,
    protocol_version: str, prompt_version: str, system_prompt: str,
    response_parser: Callable[[object], dict[str, str]],
    settings_loader: Callable[[Mapping[str, Any]], dict[str, Any]],
    request_kind: str,
    user_prompt_builder: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], str,
    ],
    cache_namespace: str,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    result = {
        "sample_id": sample_id, "endpoint_id": endpoint.endpoint_id,
        "orders": list(source_sample["orders"]), "arbiter": {},
        "source_report_bundle_sha256": {},
    }

    def invoke(replicate: int):
        order = int(source_sample["orders"][replicate])
        displayed = ordered_row(row, order, replicate)
        reports, digest, _ = source_report_bundle(
            source_output, source_sample, replicate, source_rubric)
        call = call_one(
            config, endpoint, target / "cache" / cache_namespace,
            user_text=user_prompt_builder(displayed, reports), row=displayed,
            request_key={
                "kind": request_kind, "sample_id": sample_id,
                "replicate": replicate, "order": order,
                "source_report_bundle_sha256": digest,
            },
            total_attempt_limit=total_attempt_limit,
            protocol_version=protocol_version, prompt_version=prompt_version,
            system_prompt=system_prompt, response_parser=response_parser,
            settings_loader=settings_loader,
        )
        return replicate, digest, call

    with ThreadPoolExecutor(max_workers=min(endpoint.max_concurrency, K)) as executor:
        futures = [executor.submit(invoke, replicate) for replicate in range(K)]
        for future in as_completed(futures):
            replicate, digest, call = future.result()
            result["arbiter"][str(replicate)] = compact_call(call)
            result["source_report_bundle_sha256"][str(replicate)] = digest
    atomic_write_json(
        target / "bundles" / label / f"{canonical_sha256(sample_id)}.json", result)
    return result


def run_bundles(
    config: Mapping[str, Any], output: Path, target: Path,
    items: Sequence[Mapping[str, Any]], source_by_id: Mapping[str, Any],
    source_rubric: StructuredRubric, *, label: str, total_attempt_limit: int,
    protocol_version: str, prompt_version: str, system_prompt: str,
    response_parser: Callable[[object], dict[str, str]],
    settings_loader: Callable[[Mapping[str, Any]], dict[str, Any]],
    request_kind: str,
    user_prompt_builder: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], str,
    ] = global_arbiter_user_prompt,
    cache_namespace: str = "arbiter",
) -> dict[str, Any]:
    spec = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoints = tuple(item for item in spec.endpoints if item.endpoint_id in ENDPOINT_IDS)
    if tuple(item.endpoint_id for item in endpoints) != ENDPOINT_IDS:
        raise RuntimeError("A/B-only Arbiter requires vllm-8000 and vllm-8001")
    assignment_path = target / "endpoint_assignment.json"
    assignments = load_json(assignment_path) if assignment_path.is_file() else {}
    unassigned: queue.Queue[Mapping[str, Any]] = queue.Queue()
    assigned = {endpoint.endpoint_id: queue.Queue() for endpoint in endpoints}
    for item in items:
        endpoint_id = assignments.get(str(item["sample_id"]))
        (assigned[endpoint_id] if endpoint_id in assigned else unassigned).put(item)

    lock = threading.Lock()
    completed = 0
    output_rows = {}
    started = time.perf_counter()
    print(f"{label}: 0/{len(items)} sample bundles started", flush=True)

    def worker(endpoint: BackendEndpointSpec) -> None:
        nonlocal completed
        while True:
            try:
                item = assigned[endpoint.endpoint_id].get_nowait()
            except queue.Empty:
                try:
                    item = unassigned.get_nowait()
                except queue.Empty:
                    return
                with lock:
                    assignments[str(item["sample_id"])] = endpoint.endpoint_id
                    atomic_write_json(assignment_path, assignments)
            sample_id = str(item["sample_id"])
            value = _process_bundle(
                config, endpoint, target, output, vlrb_rows((item,))[0],
                source_by_id[sample_id], source_rubric, label=label,
                total_attempt_limit=total_attempt_limit,
                protocol_version=protocol_version, prompt_version=prompt_version,
                system_prompt=system_prompt, response_parser=response_parser,
                settings_loader=settings_loader, request_kind=request_kind,
                user_prompt_builder=user_prompt_builder,
                cache_namespace=cache_namespace)
            with lock:
                output_rows[sample_id] = value
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(items) - completed)
                print(f"{label}: {completed}/{len(items)} sample={sample_id} "
                      f"endpoint={endpoint.endpoint_id} elapsed={elapsed/60:.1f}m "
                      f"ETA={eta/60:.1f}m", flush=True)

    workers = [endpoint for endpoint in endpoints
               for _ in range(max(1, endpoint.max_concurrency // K))]
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    result = {
        "schema_version": SCHEMA_VERSION, "protocol_version": protocol_version,
        "prompt_version": prompt_version, "label": label,
        "samples": [output_rows[str(item["sample_id"])] for item in items],
        "wall_seconds": time.perf_counter() - started,
        "endpoint_assignment": assignments,
    }
    atomic_write_json(target / "predictions" / f"{label}.json", result)
    return result


def telemetry(result: Mapping[str, Any]) -> dict[str, Any]:
    calls = [call for sample in result["samples"] for call in sample["arbiter"].values()]
    wall = float(result.get("wall_seconds") or 0.0)
    by_endpoint = {}
    for call in calls:
        endpoint = str(call.get("endpoint_id"))
        by_endpoint[endpoint] = by_endpoint.get(endpoint, 0) + int(
            call.get("model_generation_count", 0))
    return {
        "logical_request_count": len(calls),
        "model_generation_count": sum(int(call.get("model_generation_count", 0)) for call in calls),
        "parse_valid_count": sum(bool(call.get("parse_ok")) for call in calls),
        "parse_valid_rate": sum(bool(call.get("parse_ok")) for call in calls) / len(calls) if calls else 0.0,
        "endpoint_generation_counts": by_endpoint,
        "api_attempts": sum(int(call.get("metrics", {}).get("api_attempts") or 0) for call in calls),
        "errors": sum(int(call.get("metrics", {}).get("error_count") or 0) for call in calls),
        "input_tokens": sum(int(call.get("metrics", {}).get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(int(call.get("metrics", {}).get("output_tokens") or 0) for call in calls),
        "wall_seconds": wall,
        "logical_requests_per_minute": len(calls) / wall * 60 if wall else 0.0,
    }


def parse_failures(result: Mapping[str, Any]) -> dict[str, Any]:
    failures = [{
        "sample_id": sample["sample_id"], "replicate": int(replicate),
        "error": call.get("error"),
        "model_generation_count": call.get("model_generation_count", 0),
        "cache_key": call.get("cache_key"),
    } for sample in result["samples"]
        for replicate, call in sample["arbiter"].items()
        if not call.get("parse_ok")]
    return {
        "schema_version": SCHEMA_VERSION,
        "logical_request_count": len(result["samples"]) * K,
        "failure_count": len(failures), "failures": failures,
    }


def selected_records(
    items: Sequence[Mapping[str, Any]], count: int,
) -> tuple[Mapping[str, Any], ...]:
    groups = {}
    for item in items:
        groups.setdefault(str(item.get("group", "unknown")), []).append(item)
    for key, values in groups.items():
        values.sort(key=lambda item: hashlib.sha256(
            f"arbiter-smoke|42|{key}|{item['sample_id']}".encode()).hexdigest())
    selected = []
    keys = sorted(groups)
    offset = 0
    while len(selected) < count and any(groups.values()):
        key = keys[offset % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop(0))
        offset += 1
    if len(selected) != count:
        raise RuntimeError(f"cannot select {count} smoke rows")
    return tuple(selected)


def _five_root(values: Mapping[str, str], root_ids: Sequence[str]) -> str:
    votes = {root_id: Vote.A if values[root_id] == "A" else
             Vote.B if values[root_id] == "B" else Vote.ABSTAIN
             for root_id in root_ids}
    preference = aggregate_selected_roots(votes, root_ids)
    return preference.value if preference in {FinalPreference.A, FinalPreference.B} else "None"


def root_majority_override_audit(
    items: Sequence[Mapping[str, Any]], source_result: Mapping[str, Any],
    arbiter_result: Mapping[str, Any], source_rubric: StructuredRubric,
) -> dict[str, Any]:
    source_by_id = {str(item["sample_id"]): item for item in source_result["samples"]}
    names = ("5-0-0", "4-1-0 / 4-0-1", "3-2-0 / 3-1-1 / 3-0-2",
             "tie / sparse / all-None")
    categories = {name: {
        "replicate_count": 0, "arbiter_follows": 0, "arbiter_overrides": 0,
        "override_corrected": 0, "override_harmed": 0,
    } for name in names}
    for index, sample in enumerate(arbiter_result["samples"]):
        source = source_by_id[str(sample["sample_id"])]
        gold = int(items[index]["preferred_original_index"])
        for replicate in range(K):
            order = int(sample["orders"][replicate])
            mapping = {}
            for root_id in source_rubric.root_ids:
                call = source["unified"][str(replicate)][root_id]
                mapping[root_id] = ((call.get("parsed") or {}).get("answer", "None")
                                    if call.get("parse_ok") else "None")
            values = list(mapping.values())
            decisive = sorted((values.count("A"), values.count("B")), reverse=True)
            category_name = (names[0] if decisive == [5, 0] and values.count("None") == 0
                             else names[1] if decisive[0] == 4
                             else names[2] if decisive[0] == 3 else names[3])
            category = categories[category_name]
            category["replicate_count"] += 1
            majority = _five_root(mapping, source_rubric.root_ids)
            call = sample["arbiter"][str(replicate)]
            decision = ((call.get("parsed") or {}).get("answer", "None")
                        if call.get("parse_ok") else "None")
            if decision == majority:
                category["arbiter_follows"] += 1
                continue
            category["arbiter_overrides"] += 1
            majority_correct = display_to_original(majority, order) == gold
            arbiter_correct = display_to_original(decision, order) == gold
            category["override_corrected"] += int(arbiter_correct and not majority_correct)
            category["override_harmed"] += int(majority_correct and not arbiter_correct)
    totals = {key: sum(value[key] for value in categories.values())
              for key in ("replicate_count", "arbiter_follows", "arbiter_overrides",
                          "override_corrected", "override_harmed")}
    totals["override_net_corrected"] = (
        totals["override_corrected"] - totals["override_harmed"])
    return {"schema_version": SCHEMA_VERSION, "patterns": categories, "totals": totals}

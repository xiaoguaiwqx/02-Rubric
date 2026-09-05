"""Cache-oriented Pairwise Worker prompt and scheduling ablation.

The experiment keeps the frozen v1 worker untouched and compares it with a
versioned System/User prompt pilot.  Scheduler-only and prompt-v2 treatments
pin every criterion for one sample to one endpoint and warm that sample prefix
before fanning out the remaining criteria.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.dual_evaluator import (
    CacheOptimizedPairwiseVoteMultiModalEvaluator,
    PairwiseVoteMultiModalEvaluator,
)
from critiq.dual_worker_prompts import (
    PAIRWISE_MULTIMODAL_WORKER_PROMPT,
    PAIRWISE_MULTIMODAL_WORKER_SYSTEM_PROMPT_V2_CACHE,
    PAIRWISE_MULTIMODAL_WORKER_USER_PROMPT_V2_CACHE,
    PAIRWISE_WORKER_PROMPT_POSTFIX,
)
from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    CacheMode,
    JsonPredictionCache,
    ModelCallMetrics,
    OnlinePairwiseVoteBackend,
    PairwisePredictionOutput,
    StructuredRubric,
    aggregate_flat_votes,
)
from critiq.structured.cache import pairwise_cache_key_payload
from critiq.structured.evolution.specialize import execute_offline_m1
from critiq.structured.version import (
    PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
)
from critiq.structured.worker_output import StructuredCriterionSnapshot

from . import run_rubric_evolution as base
from . import vl_rewardbench_phase10 as phase10
from .experiment_utils import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    load_jsonl_dataset,
    make_progress_callback,
)
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "pairwise_worker_cache_prompt_ablation_v1"
PROTOCOL_VERSION = "pairwise-cache-prompt-ablation-v1"
SOURCE_EXPERIMENT = "phase10_five_root_locked_split_refine_v1"
SOURCE_RUBRIC_SHA256 = "17ad7a0b6350338b100384b701303008746463911662f4a628b90c4632cdf7ef"
SOURCE_NODE_COUNT = 22
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")
VARIANT_CONTROL = "s0_control"
VARIANT_SCHEDULER = "s1_scheduler_only"
VARIANT_PROMPT_V2 = "s2_prompt_v2"
VARIANT_PROMPT_V2_DYNAMIC = "s3_prompt_v2_dynamic"
DISCOVERY_VARIANTS = (VARIANT_CONTROL, VARIANT_SCHEDULER, VARIANT_PROMPT_V2)
HELDOUT_VARIANTS = (VARIANT_CONTROL, VARIANT_PROMPT_V2)
S3_EXTENSION_PROTOCOL_VERSION = "pairwise-cache-prompt-s3-extension-v1"
S3_HELDOUT_PROTOCOL_VERSION = "pairwise-cache-prompt-s3-heldout-v1"
S3_HELDOUT_VARIANTS = (VARIANT_CONTROL, VARIANT_PROMPT_V2_DYNAMIC)
PROMETHEUS_METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
)


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _source_rubric_path(output: Path) -> Path:
    return output / SOURCE_EXPERIMENT / "final/rubric.json"


def _status_path(target: Path) -> Path:
    return target / "stage_status.json"


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    value[stage] = {"status": "passed", "details": dict(details)}
    atomic_write_json(_status_path(target), value)


def _require(target: Path, stage: str) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_rubric_sha256": SOURCE_RUBRIC_SHA256,
        "source_node_count": SOURCE_NODE_COUNT,
        "endpoint_ids": list(ENDPOINT_IDS),
        "active_samples_per_endpoint": 2,
        "smoke_sample_count": 10,
        "cache_control_mode": "external_observation_no_reset",
        "cache_reset_paths": [],
        "discovery_variants": list(DISCOVERY_VARIANTS),
        "heldout_variants": list(HELDOUT_VARIANTS),
        "heldout_access": "final_stage_only",
    }
    value = config.get("pairwise_cache_prompt_ablation")
    if value != expected:
        raise RuntimeError(
            "pairwise_cache_prompt_ablation must match the frozen v1 protocol")
    request = config.get("worker_request_kwargs")
    if (not isinstance(request, dict)
            or request.get("temperature") != 0.5
            or request.get("max_tokens") != 2048):
        raise RuntimeError("cache ablation freezes temperature=0.5 and max_tokens=2048")
    return dict(value)


def _pool_spec(config: Mapping[str, Any]) -> BackendPoolSpec:
    spec = BackendPoolSpec.from_dict(config["backend_pool"])
    if spec.common_checkpoint_id != config["model"]:
        raise RuntimeError("cache ablation endpoint checkpoint identity mismatch")
    return spec


def _rubric(output: Path) -> StructuredRubric:
    path = _source_rubric_path(output)
    if not path.is_file():
        raise RuntimeError(f"Phase10 source rubric is missing: {path}")
    rubric = StructuredRubric.load_json(path)
    if rubric.rubric_sha256 != SOURCE_RUBRIC_SHA256 or len(rubric.nodes) != SOURCE_NODE_COUNT:
        raise RuntimeError("Phase10 source rubric hash or node count drift")
    return rubric


def _rows(config: Mapping[str, Any], split: str):
    if split == "discovery90":
        path = base._path(config["discovery_dataset"])
        expected_sha = config["discovery_dataset_sha256"]
        expected_count = 90
    elif split == "heldout500":
        path = base._path(config["heldout_dataset"])
        expected_sha = config["heldout_dataset_sha256"]
        expected_count = 500
    else:
        raise ValueError(f"unsupported cache-ablation split: {split}")
    if file_sha256(path).lower() != str(expected_sha).lower():
        raise RuntimeError(f"{split} dataset hash drift")
    return load_jsonl_dataset(path, expected_count=expected_count)


def _sample_endpoint(sample_id: str) -> str:
    digest = hashlib.sha256(
        f"{PROTOCOL_VERSION}|sample-affinity|{sample_id}".encode("utf-8")
    ).hexdigest()
    return ENDPOINT_IDS[int(digest, 16) % len(ENDPOINT_IDS)]


def _route_map(rows: Sequence[Mapping[str, Any]], rubric: StructuredRubric) -> dict[str, str]:
    result = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        endpoint_id = _sample_endpoint(sample_id)
        for node_id in rubric.preorder_node_ids():
            name = rubric.get_node(node_id).criterion.name
            result[f"{sample_id}::{name}"] = endpoint_id
    return result


class _AffinityRouter:
    """Thread-safe endpoint router with one frozen endpoint per sample."""

    def __init__(self, spec: BackendPoolSpec, routes: Mapping[str, str]) -> None:
        self.spec = spec
        self._routes = dict(routes)
        self._global = threading.BoundedSemaphore(spec.global_request_concurrency)
        self._pools = {}
        for endpoint in spec.endpoints:
            single = BackendPoolSpec(
                pool_id=f"{spec.pool_id}-cache-ablation-{endpoint.endpoint_id}",
                common_checkpoint_id=spec.common_checkpoint_id,
                global_request_concurrency=endpoint.max_concurrency,
                endpoints=(endpoint,),
            )
            self._pools[endpoint.endpoint_id] = AvailableSlotBackendPool(single)

    @property
    def backend_id(self) -> str:
        return self.spec.backend_id

    @property
    def records(self):
        return tuple(record for pool in self._pools.values() for record in pool.records)

    def records_by_endpoint(self):
        return {endpoint_id: len(pool.records)
                for endpoint_id, pool in self._pools.items()}

    def call(self, prompt: object, *, request_type: str, request_key: str,
             structured_attempt: int, agent_args: Mapping[str, Any]):
        try:
            endpoint_id = self._routes[request_key]
        except KeyError as exc:
            raise RuntimeError(f"request absent from sample-affinity map: {request_key}") from exc
        self._global.acquire()
        try:
            return self._pools[endpoint_id].call(
                prompt,
                request_type=request_type,
                request_key=request_key,
                structured_attempt=structured_attempt,
                agent_args=agent_args,
            )
        finally:
            self._global.release()

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "backend_pool": self.spec.to_dict(),
            "backend_id": self.backend_id,
            "routing_protocol": "sample-affinity-sha256-mod2-v1",
            "endpoint_call_counts": dict(self.records_by_endpoint()),
            "calls": [record.to_dict() for record in self.records],
        }


def _uses_v2_prompt(variant: str) -> bool:
    return variant in {VARIANT_PROMPT_V2, VARIANT_PROMPT_V2_DYNAMIC}


def _uses_affinity_scheduler(variant: str) -> bool:
    return variant in {VARIANT_SCHEDULER, VARIANT_PROMPT_V2}


def _make_evaluator(config: Mapping[str, Any], rows, backend, variant: str):
    request = dict(config["worker_request_kwargs"])
    request["temperature"] = 0.5
    request["max_tokens"] = 2048
    kwargs = {
        "worker_args": {
            "model": config["model"],
            "api_keys": "EMPTY",
            "request_kwargs": request,
            "api_retry_attempts": config["api_retry_attempts"],
        },
        "dataset": rows,
        "backend_id": backend.backend_id,
        "max_concurrent": _pool_spec(config).global_request_concurrency,
        "max_retries": config["structured_max_retries"],
        "max_data_chars": None,
        "encode_local_image": True,
        "call_backend": backend,
    }
    if _uses_v2_prompt(variant):
        return CacheOptimizedPairwiseVoteMultiModalEvaluator(**kwargs)
    if variant in {VARIANT_CONTROL, VARIANT_SCHEDULER}:
        return PairwiseVoteMultiModalEvaluator(**kwargs)
    raise ValueError(f"unknown cache-ablation variant: {variant}")


def _reset_caches(config: Mapping[str, Any], reset_paths: Sequence[str]) -> list[dict[str, Any]]:
    results = []
    for endpoint in sorted(_pool_spec(config).endpoints, key=lambda item: item.endpoint_id):
        base_url = endpoint.base_url.rstrip("/")
        root = base_url[:-3] if base_url.endswith("/v1") else base_url
        for reset_path in reset_paths:
            url = root + reset_path
            request = urllib.request.Request(
                url, data=b"", method="POST", headers={"Accept": "application/json"})
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    status = int(response.status)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(
                    f"{reset_path} is unavailable on {endpoint.endpoint_id}. "
                    "Restart the local vLLM server with VLLM_SERVER_DEV_MODE=1 "
                    "before running the cache ablation; no inference was started."
                ) from exc
            if status < 200 or status >= 300:
                raise RuntimeError(
                    f"cache reset failed for {endpoint.endpoint_id} {reset_path}: {status}")
            results.append({
                "endpoint_id": endpoint.endpoint_id,
                "path": reset_path,
                "url": url,
                "status": status,
                "elapsed_seconds": time.perf_counter() - started,
                "response": body[:500],
            })
    return results


def _reset_capabilities(config: Mapping[str, Any], reset_paths: Sequence[str]) -> dict[str, Any]:
    result = {}
    for endpoint in sorted(_pool_spec(config).endpoints, key=lambda item: item.endpoint_id):
        base_url = endpoint.base_url.rstrip("/")
        root = base_url[:-3] if base_url.endswith("/v1") else base_url
        try:
            with urllib.request.urlopen(root + "/openapi.json", timeout=10) as response:
                schema = json.loads(response.read().decode("utf-8"))
            paths = schema.get("paths", {}) if isinstance(schema, dict) else {}
            available = {path: path in paths for path in reset_paths}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            available = {path: False for path in reset_paths}
            result[endpoint.endpoint_id] = {
                "available": available, "error": str(exc)}
            continue
        result[endpoint.endpoint_id] = {"available": available, "error": None}
    return result


def _server_metrics_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    snapshots = {}
    for endpoint in sorted(_pool_spec(config).endpoints, key=lambda item: item.endpoint_id):
        base_url = endpoint.base_url.rstrip("/")
        root = base_url[:-3] if base_url.endswith("/v1") else base_url
        with urllib.request.urlopen(root + "/metrics", timeout=10) as response:
            text = response.read().decode("utf-8", errors="replace")
        values = {name: 0.0 for name in PROMETHEUS_METRICS}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
            if match and match.group(1) in values:
                values[match.group(1)] += float(match.group(2))
        snapshots[endpoint.endpoint_id] = values
    return snapshots


def _server_metrics_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    by_endpoint = {}
    totals = {name: 0.0 for name in PROMETHEUS_METRICS}
    for endpoint_id in ENDPOINT_IDS:
        values = {}
        for name in PROMETHEUS_METRICS:
            delta = float(after[endpoint_id][name]) - float(before[endpoint_id][name])
            if delta < -1e-9:
                raise RuntimeError(
                    f"vLLM metric counter reset during run: {endpoint_id} {name}")
            values[name] = max(0.0, delta)
            totals[name] += values[name]
        by_endpoint[endpoint_id] = values
    queries = totals["vllm:prefix_cache_queries_total"]
    hits = totals["vllm:prefix_cache_hits_total"]
    ttft_count = totals["vllm:time_to_first_token_seconds_count"]
    prefill_count = totals["vllm:request_prefill_time_seconds_count"]
    return {
        "by_endpoint": by_endpoint,
        "totals": totals,
        "prefix_cache_token_hit_rate": hits / queries if queries else None,
        "mean_ttft_seconds": (
            totals["vllm:time_to_first_token_seconds_sum"] / ttft_count
            if ttft_count else None),
        "mean_prefill_seconds": (
            totals["vllm:request_prefill_time_seconds_sum"] / prefill_count
            if prefill_count else None),
    }


def _cache_payload(evaluator, row, node):
    return pairwise_cache_key_payload(
        sample_fingerprint=evaluator.sample_fingerprint(row),
        criterion_name=node.criterion.name,
        criterion_description=node.criterion.description,
        request_spec=evaluator.request_spec(),
    )


def _affinity_prediction(evaluator, rows, rubric, cache, callback,
                         active_samples_per_endpoint: int,
                         prompt_version: str):
    backend = OnlinePairwiseVoteBackend(evaluator, cache)
    nodes = tuple(rubric.get_node(node_id) for node_id in rubric.preorder_node_ids())
    outputs: list[dict[str, Any]] = [dict() for _ in rows]
    sample_metrics: list[list[ModelCallMetrics]] = [[] for _ in rows]
    trace_lock = threading.Lock()
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()

    def record(sample_index: int, node, result) -> None:
        outputs[sample_index][node.criterion.name] = result.output
        sample_metrics[sample_index].append(result.metrics)
        if callback is not None:
            callback(
                sample_index,
                f"{rows[sample_index][evaluator.sample_id_field]}::{node.criterion.name}",
                result.metrics,
            )

    def pipeline(sample_index: int) -> None:
        row = rows[sample_index]
        sample_id = str(row[evaluator.sample_id_field])
        seed = nodes[0]
        entry = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "endpoint_id": _sample_endpoint(sample_id),
            "seed_criterion": seed.criterion.name,
            "pipeline_started_seconds": time.perf_counter() - started,
            "resume_warmup": False,
        }
        seed_cached = cache.get_pairwise(_cache_payload(evaluator, row, seed)) is not None
        missing_after_seed = any(
            cache.get_pairwise(_cache_payload(evaluator, row, node)) is None
            for node in nodes[1:]
        )
        if seed_cached and missing_after_seed:
            evaluator.infer_one(row, seed.criterion.to_criterion())
            entry["resume_warmup"] = True
        seed_result = backend.evaluate(row, seed)
        record(sample_index, seed, seed_result)
        entry["seed_completed_seconds"] = time.perf_counter() - started
        entry["fanout_submitted_seconds"] = time.perf_counter() - started
        with ThreadPoolExecutor(max_workers=len(nodes) - 1) as executor:
            futures = {executor.submit(backend.evaluate, row, node): node
                       for node in nodes[1:]}
            for future in as_completed(futures):
                node = futures[future]
                record(sample_index, node, future.result())
        entry["pipeline_completed_seconds"] = time.perf_counter() - started
        entry["seed_before_fanout"] = (
            entry["seed_completed_seconds"] <= entry["fanout_submitted_seconds"])
        with trace_lock:
            traces.append(entry)

    by_endpoint = {endpoint_id: [] for endpoint_id in ENDPOINT_IDS}
    for index, row in enumerate(rows):
        by_endpoint[_sample_endpoint(str(row[evaluator.sample_id_field]))].append(index)

    def endpoint_pipeline(indices: Sequence[int]) -> None:
        with ThreadPoolExecutor(max_workers=active_samples_per_endpoint) as executor:
            futures = [executor.submit(pipeline, index) for index in indices]
            for future in as_completed(futures):
                future.result()

    with ThreadPoolExecutor(max_workers=len(ENDPOINT_IDS)) as executor:
        futures = [executor.submit(endpoint_pipeline, by_endpoint[endpoint_id])
                   for endpoint_id in ENDPOINT_IDS]
        for future in as_completed(futures):
            future.result()

    answers = tuple(aggregate_flat_votes(item.vote for item in row.values())
                    for row in outputs)
    prediction = PairwisePredictionOutput(
        tuple(str(row[evaluator.sample_id_field]) for row in rows),
        tuple(evaluator.sample_fingerprint(row) for row in rows),
        tuple(StructuredCriterionSnapshot(node.criterion.name, node.criterion.description)
              for node in nodes),
        tuple(outputs),
        answers,
        evaluator.request_spec(),
        prompt_version=prompt_version,
    )
    return prediction, sorted(traces, key=lambda item: item["sample_index"])


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _telemetry(provenance: Mapping[str, Any], wall_seconds: float,
               logical_requests: int,
               server_metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    calls = list(provenance.get("calls", []))
    latencies = [float(item.get("latency_seconds") or 0.0) for item in calls]
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in calls)
    cached_known = [item.get("cached_input_tokens") for item in calls
                    if isinstance(item.get("cached_input_tokens"), int)]
    cached_tokens = sum(int(value) for value in cached_known)
    return {
        "logical_request_count": logical_requests,
        "recorded_model_call_count": len(calls),
        "api_attempts": sum(int(item.get("api_attempts") or 0) for item in calls),
        "errors": sum(int(item.get("error_count") or 0) for item in calls),
        "input_tokens": input_tokens,
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in calls),
        "cached_input_tokens": cached_tokens if cached_known else None,
        "cached_token_reporting_rate": len(cached_known) / len(calls) if calls else 0.0,
        "prefix_cache_token_hit_rate": (
            server_metrics["prefix_cache_token_hit_rate"] if server_metrics else None),
        "wall_seconds": wall_seconds,
        "requests_per_minute": logical_requests / wall_seconds * 60 if wall_seconds else 0.0,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p99": _percentile(latencies, 0.99),
        },
        "mean_ttft_seconds": server_metrics["mean_ttft_seconds"] if server_metrics else None,
        "mean_prefill_seconds": server_metrics["mean_prefill_seconds"] if server_metrics else None,
        "ttft_available": bool(server_metrics and server_metrics["mean_ttft_seconds"] is not None),
        "endpoint_call_counts": dict(provenance.get("endpoint_call_counts", {})),
        "vllm_server_metrics": dict(server_metrics) if server_metrics else None,
        "cache_measurement_source": (
            "runner_prometheus" if server_metrics else "external_backend_observation"),
    }


def _manifest(config: Mapping[str, Any], output: Path, *, include_endpoints: bool) -> dict[str, Any]:
    protocol = _protocol(config)
    rubric = _rubric(output)
    discovery = _rows(config, "discovery90")
    model_rows = base._model_rows(discovery[:1])
    pool = AvailableSlotBackendPool(_pool_spec(config))
    v1 = _make_evaluator(config, model_rows, pool, VARIANT_CONTROL)
    v2 = _make_evaluator(config, model_rows, pool, VARIANT_PROMPT_V2)
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol": protocol,
        "source": {
            "experiment": SOURCE_EXPERIMENT,
            "rubric_path": str(_source_rubric_path(output).resolve()),
            "rubric_file_sha256": file_sha256(_source_rubric_path(output)),
            "rubric_sha256": rubric.rubric_sha256,
            "node_count": len(rubric.nodes),
        },
        "datasets": {
            "discovery90": {
                "path": str(base._path(config["discovery_dataset"]).resolve()),
                "sha256": config["discovery_dataset_sha256"],
                "count": len(discovery),
                "sample_ids": [str(row["sample_id"]) for row in discovery],
            },
            "heldout500": {
                "path": str(base._path(config["heldout_dataset"]).resolve()),
                "sha256": config["heldout_dataset_sha256"],
                "count": 500,
                "accessed_during_freeze": False,
            },
        },
        "worker": {
            "model": config["model"],
            "decoding": dict(config["worker_request_kwargs"]),
            "structured_max_retries": config["structured_max_retries"],
            "api_retry_attempts": config["api_retry_attempts"],
            "v1_request_spec": v1.request_spec().to_dict(),
            "v2_request_spec": v2.request_spec().to_dict(),
            "v1_prompt_sha256": v1.request_spec().prompt_sha256,
            "v2_prompt_sha256": v2.request_spec().prompt_sha256,
            "v2_prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
            "v2_system_sha256": hashlib.sha256(
                PAIRWISE_MULTIMODAL_WORKER_SYSTEM_PROMPT_V2_CACHE.encode("utf-8")
            ).hexdigest(),
            "v2_user_sha256": hashlib.sha256(
                PAIRWISE_MULTIMODAL_WORKER_USER_PROMPT_V2_CACHE.encode("utf-8")
            ).hexdigest(),
            "v2_content_order": v2.content_order,
        },
        "backend_pool": _pool_spec(config).to_dict(),
        "variant_order": list(DISCOVERY_VARIANTS),
        "selection_after_heldout_forbidden": True,
    }
    if include_endpoints:
        value["endpoint_identities"] = phase10._inspect_endpoints(config)
    return value


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target(output)
    _require(target, "pairwise-cache-freeze")
    stored = load_json(target / "frozen_manifest.json")
    expected = _manifest(config, output, include_endpoints=False)
    static = {key: value for key, value in stored.items() if key != "endpoint_identities"}
    if static != expected:
        raise RuntimeError("pairwise cache ablation frozen manifest drift")
    return target, stored, _rubric(output)


def _verify_live_endpoints(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if manifest.get("endpoint_identities") != phase10._inspect_endpoints(config):
        raise RuntimeError("pairwise cache ablation live endpoint identity drift")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    manifest = _manifest(config, output, include_endpoints=True)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        downstream = (
            "pairwise-cache-smoke", "pairwise-cache-discovery-run",
            "pairwise-cache-heldout-run",
        )
        if any(status.get(stage, {}).get("status") == "passed" for stage in downstream):
            raise RuntimeError("pairwise cache ablation manifest drift after inference")
    atomic_write_json(path, manifest)
    details = {
        "rubric_sha256": manifest["source"]["rubric_sha256"],
        "node_count": manifest["source"]["node_count"],
        "discovery_count": manifest["datasets"]["discovery90"]["count"],
        "heldout_accessed": False,
        "variants": list(DISCOVERY_VARIANTS),
    }
    _status(target, "pairwise-cache-freeze", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    discovery = _rows(config, "discovery90")
    model_row = base._model_rows(discovery[:1])[0]
    pool = AvailableSlotBackendPool(_pool_spec(config))
    evaluator = _make_evaluator(config, (model_row,), pool, VARIANT_PROMPT_V2)
    criterion = rubric.get_node(rubric.preorder_node_ids()[0]).criterion.to_criterion()
    content = evaluator._make_user_content(model_row, criterion)
    if (not isinstance(content, list) or len(content) != 2
            or content[0].get("type") != "image_url"
            or content[1].get("type") != "text"):
        raise RuntimeError("Prompt v2 content is not image-first with one trailing text block")
    text = str(content[1]["text"])
    positions = [text.index(label) for label in (
        "## Source Instruction or Question", "## Candidate A",
        "## Candidate B", "## Criterion")]
    if positions != sorted(positions):
        raise RuntimeError("Prompt v2 sample/criterion order drift")
    system_lower = PAIRWISE_MULTIMODAL_WORKER_SYSTEM_PROMPT_V2_CACHE.lower()
    if any(name in system_lower for name in ("rlhf-v", "vl-rewardbench")):
        raise RuntimeError("Prompt v2 leaks a dataset name")
    route_map = _route_map(discovery, rubric)
    criteria = rubric.criteria_in_execution_order()
    per_sample_ok = all(len({route_map[f"{row['sample_id']}::{criterion.name}"]
                             for criterion in criteria}) == 1
                        for row in discovery)
    if not per_sample_ok:
        raise RuntimeError("sample-affinity route map splits a sample across endpoints")
    value = {
        "schema_version": "1.0.0",
        "status": "passed",
        "heldout_accessed": False,
        "v1_prompt_unchanged": (
            PAIRWISE_MULTIMODAL_WORKER_PROMPT.startswith("## Instruction")
            and bool(PAIRWISE_WORKER_PROMPT_POSTFIX)),
        "v1_v2_request_specs_distinct": (
            manifest["worker"]["v1_request_spec"]
            != manifest["worker"]["v2_request_spec"]),
        "v2_image_first": True,
        "v2_criterion_after_candidates": True,
        "v2_dataset_neutral": True,
        "sample_affinity_verified": per_sample_ok,
        "route_count": len(route_map),
        "endpoint_counts": {
            endpoint_id: sum(value == endpoint_id for value in route_map.values())
            for endpoint_id in ENDPOINT_IDS
        },
        "cache_control_mode": manifest["protocol"]["cache_control_mode"],
        "cache_reset_capabilities": {},
    }
    value["online_cache_reset_ready"] = None
    atomic_write_json(target / "offline_audit.json", value)
    _status(target, "pairwise-cache-audit", value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _prediction_valid_rate(prediction: PairwisePredictionOutput) -> float:
    values = [output for row in prediction.node_outputs for output in row.values()]
    return sum(item.parse_ok and item.answer_valid for item in values) / len(values)


def _validate_prediction(prediction: PairwisePredictionOutput, rows, rubric,
                         request_spec: Mapping[str, Any], prompt_version: str) -> None:
    if tuple(prediction.sample_ids) != tuple(str(row["sample_id"]) for row in rows):
        raise RuntimeError("cache ablation prediction sample order drift")
    names = tuple(rubric.get_node(node_id).criterion.name
                  for node_id in rubric.preorder_node_ids())
    if tuple(item.name for item in prediction.criteria) != names:
        raise RuntimeError("cache ablation prediction criterion order drift")
    if prediction.request_spec.to_dict() != request_spec:
        raise RuntimeError("cache ablation request spec drift")
    if prediction.prompt_version != prompt_version:
        raise RuntimeError("cache ablation prediction prompt version drift")


def _with_prompt_version(prediction: PairwisePredictionOutput,
                         prompt_version: str) -> PairwisePredictionOutput:
    """Correct scheduler-owned artifact metadata without changing predictions."""
    return PairwisePredictionOutput(
        prediction.sample_ids,
        prediction.sample_fingerprints,
        prediction.criteria,
        prediction.node_outputs,
        prediction.flat_answers,
        prediction.request_spec,
        semantics_version=prediction.semantics_version,
        schema_version=prediction.schema_version,
        prompt_version=prompt_version,
        parser_version=prediction.parser_version,
    )


def _run_variant(config: Mapping[str, Any], output: Path, target: Path,
                 manifest: Mapping[str, Any], rubric: StructuredRubric,
                 rows, split: str, variant: str) -> dict[str, Any]:
    work = target / split / variant
    artifact = work / "predictions.json"
    expected_spec = manifest["worker"][
        "v2_request_spec" if _uses_v2_prompt(variant) else "v1_request_spec"]
    prompt_version = (PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
                      if _uses_v2_prompt(variant) else "1.0.0")
    if artifact.exists():
        prediction = PairwisePredictionOutput.load_json(artifact)
        # The original dynamic scheduler creates an otherwise correct v2
        # prediction artifact with its legacy default prompt-version metadata.
        # Repair only that exact S3 condition so completed requests and timing
        # remain reusable; request-spec validation below still guards identity.
        metadata_repaired = (
            variant == VARIANT_PROMPT_V2_DYNAMIC
            and prediction.prompt_version == "1.0.0"
            and prediction.request_spec.to_dict() == expected_spec
        )
        if metadata_repaired:
            prediction = _with_prompt_version(prediction, prompt_version)
            atomic_write_json(artifact, prediction.to_dict())
            summary = load_json(work / "run_summary.json")
            summary["prediction_sha256"] = file_sha256(artifact)
            summary["prompt_version_metadata_repaired"] = True
            atomic_write_json(work / "run_summary.json", summary)
        _validate_prediction(prediction, rows, rubric, expected_spec, prompt_version)
        return load_json(work / "run_summary.json")

    measure_server_cache = manifest["protocol"]["cache_control_mode"] == "required_http_reset"
    reset = (_reset_caches(config, manifest["protocol"]["cache_reset_paths"])
             if measure_server_cache else [])
    atomic_write_json(work / "prefix_cache_reset.json", {
        "schema_version": "1.0.0",
        "mode": manifest["protocol"]["cache_control_mode"],
        "skipped": not measure_server_cache,
        "items": reset})
    metrics_before = _server_metrics_snapshot(config) if measure_server_cache else None
    if metrics_before is not None:
        atomic_write_json(work / "vllm_metrics_before.json", metrics_before)
    model_rows = base._model_rows(rows)
    route_map = _route_map(rows, rubric)
    if not _uses_affinity_scheduler(variant):
        backend = AvailableSlotBackendPool(_pool_spec(config))
    else:
        backend = _AffinityRouter(_pool_spec(config), route_map)
        atomic_write_json(work / "sample_affinity.json", {
            "schema_version": "1.0.0",
            "protocol": "sample-affinity-sha256-mod2-v1",
            "route_sha256": canonical_sha256(route_map),
            "endpoint_counts": {
                endpoint_id: sum(value == endpoint_id for value in route_map.values())
                for endpoint_id in ENDPOINT_IDS
            },
            "routes": route_map,
        })
    evaluator = _make_evaluator(config, model_rows, backend, variant)
    if evaluator.request_spec().to_dict() != expected_spec:
        raise RuntimeError(f"{variant} evaluator request identity drift")
    cache_root = work / "cache"
    resumed = cache_root.exists() and any(cache_root.rglob("*.json"))
    cache = JsonPredictionCache(cache_root, CacheMode.READ_WRITE)
    callback = make_progress_callback(
        work, f"{split}_{variant}", len(rows) * len(rubric.nodes), backend)
    started = time.perf_counter()
    if not _uses_affinity_scheduler(variant):
        prediction = base._pairwise_cached(
            evaluator, model_rows, rubric, cache, None,
            evaluation_callback=callback)
        if prediction.prompt_version != prompt_version:
            prediction = _with_prompt_version(prediction, prompt_version)
        traces = []
    else:
        prediction, traces = _affinity_prediction(
            evaluator, model_rows, rubric, cache, callback,
            manifest["protocol"]["active_samples_per_endpoint"],
            prompt_version)
        atomic_write_json(work / "seed_fanout_trace.json", {
            "schema_version": "1.0.0", "items": traces})
        if not all(item["seed_before_fanout"] for item in traces):
            raise RuntimeError("seed/fan-out ordering invariant failed")
    wall_seconds = time.perf_counter() - started
    metrics_after = _server_metrics_snapshot(config) if measure_server_cache else None
    metrics_delta = (_server_metrics_delta(metrics_before, metrics_after)
                     if metrics_before is not None and metrics_after is not None else None)
    if metrics_after is not None and metrics_delta is not None:
        atomic_write_json(work / "vllm_metrics_after.json", metrics_after)
        atomic_write_json(work / "vllm_metrics_delta.json", metrics_delta)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    prediction.save_json(artifact)
    provenance = backend.provenance_dict()
    atomic_write_json(work / "provenance.json", provenance)
    summary = {
        "schema_version": "1.0.0",
        "split": split,
        "variant": variant,
        "sample_count": len(rows),
        "node_count": len(rubric.nodes),
        "prompt_version": prompt_version,
        "request_spec": evaluator.request_spec().to_dict(),
        "prediction_sha256": file_sha256(artifact),
        "valid_rate": _prediction_valid_rate(prediction),
        "resumed_from_local_cache": bool(resumed),
        "fresh_timing_valid": not resumed,
        "cache_state_reset_before_variant": measure_server_cache,
        "timing_order_confounded_by_shared_cache": not measure_server_cache,
        "resume_warmup_count": sum(bool(item.get("resume_warmup")) for item in traces),
        "telemetry": _telemetry(
            provenance, wall_seconds, len(rows) * len(rubric.nodes), metrics_delta),
    }
    atomic_write_json(work / "run_summary.json", summary)
    _validate_prediction(prediction, rows, rubric, expected_spec, prompt_version)
    return summary


def _node_metrics(prediction: PairwisePredictionOutput, rows) -> dict[str, Any]:
    result = {}
    for criterion in prediction.criteria:
        outputs = [row[criterion.name] for row in prediction.node_outputs]
        decisive = [item.vote.value in {"A", "B"} for item in outputs]
        support = sum(decisive)
        correct = sum(active and item.vote.value == row["answer"]
                      for active, item, row in zip(decisive, outputs, rows))
        result[criterion.name] = {
            "support": support,
            "coverage": support / len(rows),
            "correct_count": correct,
            "covered_accuracy": correct / support if support else 0.0,
            "strict_accuracy": correct / len(rows),
            "valid_rate": sum(item.parse_ok and item.answer_valid for item in outputs)
                          / len(outputs),
        }
    return result


def _quality_metrics(prediction: PairwisePredictionOutput,
                     rubric: StructuredRubric, rows) -> dict[str, Any]:
    _, answers = execute_offline_m1(rubric, prediction, rows)
    m1 = base._metrics(answers, rows)
    return {
        "m1": m1,
        "valid_rate": _prediction_valid_rate(prediction),
        "node_metrics": _node_metrics(prediction, rows),
    }


def _paired(rows, before: Sequence[str], after: Sequence[str]) -> dict[str, Any]:
    corrected, harmed = [], []
    for row, old, new in zip(rows, before, after):
        gold = str(row["answer"])
        if new == gold and old != gold:
            corrected.append(str(row["sample_id"]))
        elif old == gold and new != gold:
            harmed.append(str(row["sample_id"]))
    discordant = len(corrected) + len(harmed)
    tail = (sum(math.comb(discordant, index)
                for index in range(min(len(corrected), len(harmed)) + 1))
            / (2 ** discordant) if discordant else 0.5)
    return {
        "corrected_count": len(corrected),
        "harmed_count": len(harmed),
        "net_corrected": len(corrected) - len(harmed),
        "mcnemar_exact_two_sided_p": min(1.0, 2.0 * tail),
        "corrected_sample_ids": corrected,
        "harmed_sample_ids": harmed,
    }


def _efficiency_delta(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    old = control["telemetry"]
    new = treatment["telemetry"]
    old_hit = old.get("prefix_cache_token_hit_rate")
    new_hit = new.get("prefix_cache_token_hit_rate")
    wall_reduction = ((old["wall_seconds"] - new["wall_seconds"]) / old["wall_seconds"]
                      if old["wall_seconds"] else 0.0)
    p50_reduction = ((old["latency_seconds"]["p50"] - new["latency_seconds"]["p50"])
                     / old["latency_seconds"]["p50"]
                     if old["latency_seconds"]["p50"] else 0.0)
    return {
        "wall_time_reduction_fraction": wall_reduction,
        "requests_per_minute_ratio": (
            new["requests_per_minute"] / old["requests_per_minute"]
            if old["requests_per_minute"] else 0.0),
        "p50_latency_reduction_fraction": p50_reduction,
        "prefix_cache_hit_rate_delta": (
            new_hit - old_hit if old_hit is not None and new_hit is not None else None),
        "control_fresh_timing_valid": control["fresh_timing_valid"],
        "treatment_fresh_timing_valid": treatment["fresh_timing_valid"],
    }


def _report_split(config: Mapping[str, Any], output: Path, split: str,
                  variants: Sequence[str]) -> dict[str, Any]:
    target, manifest, rubric = _load_frozen(config, output)
    rows = _rows(config, split)
    run_summaries, qualities = {}, {}
    for variant in variants:
        work = target / split / variant
        summary_path = work / "run_summary.json"
        prediction_path = work / "predictions.json"
        if not summary_path.exists() or not prediction_path.exists():
            raise RuntimeError(f"run {split} variant {variant} first")
        run_summaries[variant] = load_json(summary_path)
        prediction = PairwisePredictionOutput.load_json(prediction_path)
        expected_spec = manifest["worker"][
            "v2_request_spec" if variant == VARIANT_PROMPT_V2 else "v1_request_spec"]
        prompt_version = (PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
                          if variant == VARIANT_PROMPT_V2 else "1.0.0")
        _validate_prediction(prediction, rows, rubric, expected_spec, prompt_version)
        qualities[variant] = _quality_metrics(prediction, rubric, rows)
    control_predictions = qualities[VARIANT_CONTROL]["m1"]["predictions"]
    paired = {
        variant: _paired(rows, control_predictions,
                         qualities[variant]["m1"]["predictions"])
        for variant in variants if variant != VARIANT_CONTROL
    }
    efficiency = {
        variant: _efficiency_delta(run_summaries[VARIANT_CONTROL], run_summaries[variant])
        for variant in variants if variant != VARIANT_CONTROL
    }
    s2_quality = qualities[VARIANT_PROMPT_V2]
    control_quality = qualities[VARIANT_CONTROL]
    quality_non_degenerate = (
        s2_quality["m1"]["accuracy"] >= control_quality["m1"]["accuracy"] - 0.01
        and s2_quality["m1"]["coverage"] >= control_quality["m1"]["coverage"] - 0.01
        and s2_quality["valid_rate"] >= 0.99
    )
    s2_efficiency = efficiency[VARIANT_PROMPT_V2]
    prompt_efficiency_positive = (
        s2_efficiency["wall_time_reduction_fraction"] >= 0.20)
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "split": split,
        "sample_count": len(rows),
        "rubric_sha256": rubric.rubric_sha256,
        "run_summaries": run_summaries,
        "quality": qualities,
        "paired_vs_control": paired,
        "efficiency_vs_control": efficiency,
        "quality_non_degenerate": quality_non_degenerate,
        "prompt_efficiency_positive": prompt_efficiency_positive,
        "cache_control_mode": manifest["protocol"]["cache_control_mode"],
        "timing_interpretation": (
            "exploratory_order_confounded; cache utilization observed externally"),
        "go": quality_non_degenerate and prompt_efficiency_positive,
        "heldout_exploratory": split == "heldout500",
        "selection_after_heldout_forbidden": True,
    }
    atomic_write_json(target / split / "report.json", value)
    return value


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, "pairwise-cache-audit")
    _verify_live_endpoints(config, manifest)
    rows = _rows(config, "discovery90")[:manifest["protocol"]["smoke_sample_count"]]
    summaries = {
        variant: _run_variant(
            config, output, target, manifest, rubric, rows, "smoke", variant)
        for variant in DISCOVERY_VARIANTS
    }
    valid = {variant: summary["valid_rate"] for variant, summary in summaries.items()}
    if min(valid.values()) < 0.99:
        raise RuntimeError("pairwise cache smoke final-valid rate below 99%")
    details = {
        "sample_count": len(rows),
        "logical_requests_per_variant": len(rows) * len(rubric.nodes),
        "valid_rates": valid,
        "fresh_timing_valid": {
            variant: summary["fresh_timing_valid"] for variant, summary in summaries.items()},
    }
    _status(target, "pairwise-cache-smoke", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def discovery_run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, "pairwise-cache-smoke")
    _verify_live_endpoints(config, manifest)
    rows = _rows(config, "discovery90")
    summaries = {
        variant: _run_variant(
            config, output, target, manifest, rubric, rows, "discovery90", variant)
        for variant in DISCOVERY_VARIANTS
    }
    details = {
        "sample_count": len(rows),
        "logical_requests_per_variant": len(rows) * len(rubric.nodes),
        "variants": list(summaries),
    }
    _status(target, "pairwise-cache-discovery-run", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def discovery_report(config: Mapping[str, Any], output: Path) -> None:
    target, _, _ = _load_frozen(config, output)
    _require(target, "pairwise-cache-discovery-run")
    value = _report_split(config, output, "discovery90", DISCOVERY_VARIANTS)
    _status(target, "pairwise-cache-discovery-report", {
        "go": value["go"],
        "quality_non_degenerate": value["quality_non_degenerate"],
        "prompt_efficiency_positive": value["prompt_efficiency_positive"],
    })
    print(json.dumps({
        "go": value["go"],
        "m1_accuracy": {name: item["m1"]["accuracy"]
                        for name, item in value["quality"].items()},
        "wall_seconds": {name: item["telemetry"]["wall_seconds"]
                         for name, item in value["run_summaries"].items()},
    }, indent=2, ensure_ascii=False))


def heldout_run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, "pairwise-cache-discovery-report")
    discovery = load_json(target / "discovery90/report.json")
    if not discovery.get("go"):
        raise RuntimeError("discovery go gate failed; heldout access is forbidden")
    _verify_live_endpoints(config, manifest)
    rows = _rows(config, "heldout500")
    summaries = {
        variant: _run_variant(
            config, output, target, manifest, rubric, rows, "heldout500", variant)
        for variant in HELDOUT_VARIANTS
    }
    details = {
        "sample_count": len(rows),
        "logical_requests_per_variant": len(rows) * len(rubric.nodes),
        "variants": list(summaries),
        "exploratory": True,
    }
    _status(target, "pairwise-cache-heldout-run", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def final_report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, "pairwise-cache-heldout-run")
    heldout = _report_split(config, output, "heldout500", HELDOUT_VARIANTS)
    discovery = load_json(target / "discovery90/report.json")
    adopt = discovery["go"] and heldout["go"]
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "rubric_sha256": rubric.rubric_sha256,
        "manifest_sha256": file_sha256(target / "frozen_manifest.json"),
        "discovery": discovery,
        "heldout": heldout,
        "decision": "adopt_prompt_v2" if adopt else "do_not_adopt_prompt_v2",
        "old_prompt_remains_replayable": True,
        "selection_after_heldout_forbidden": True,
    }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# Pairwise Worker Cache Prompt Ablation", "",
        f"Decision: **{value['decision']}**", "",
        "| Split | Variant | M1 ACC | Coverage | Valid | Wall (s) | Req/min | Cache-hit tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, report in (("discovery90", discovery), ("heldout500", heldout)):
        for variant, quality in report["quality"].items():
            summary = report["run_summaries"][variant]
            telemetry = summary["telemetry"]
            cache_rate = telemetry["prefix_cache_token_hit_rate"]
            lines.append(
                f"| {split_name} | {variant} | {quality['m1']['accuracy']:.4f} | "
                f"{quality['m1']['coverage']:.4f} | {quality['valid_rate']:.4f} | "
                f"{telemetry['wall_seconds']:.1f} | {telemetry['requests_per_minute']:.2f} | "
                f"{'N/A' if cache_rate is None else f'{cache_rate:.4f}'} |")
    (target / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _status(target, "pairwise-cache-final-report", {
        "decision": value["decision"],
        "report": str(target / "final_report.json"),
    })
    print(json.dumps({
        "decision": value["decision"],
        "discovery_go": discovery["go"],
        "heldout_go": heldout["go"],
    }, indent=2, ensure_ascii=False))


def _s3_extension_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target, base_manifest, rubric = _load_frozen(config, output)
    _require(target, "pairwise-cache-discovery-run")
    sources = {}
    for variant in (VARIANT_CONTROL, VARIANT_PROMPT_V2):
        work = target / "discovery90" / variant
        prediction = work / "predictions.json"
        summary = work / "run_summary.json"
        if not prediction.exists() or not summary.exists():
            raise RuntimeError(f"S3 source artifact missing: {variant}")
        sources[variant] = {
            "prediction_sha256": file_sha256(prediction),
            "run_summary_sha256": file_sha256(summary),
        }
    return {
        "schema_version": "1.0.0",
        "protocol_version": S3_EXTENSION_PROTOCOL_VERSION,
        "base_protocol_version": PROTOCOL_VERSION,
        "base_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
        "rubric_sha256": rubric.rubric_sha256,
        "sample_count": 90,
        "node_count": len(rubric.nodes),
        "logical_request_count": 90 * len(rubric.nodes),
        "variant": VARIANT_PROMPT_V2_DYNAMIC,
        "prompt_request_spec": base_manifest["worker"]["v2_request_spec"],
        "scheduler": "available-slot-dynamic-v1",
        "cache_control_mode": base_manifest["protocol"]["cache_control_mode"],
        "source_artifacts": sources,
        "heldout_accessed": False,
        "selection_after_heldout_forbidden": True,
    }


def _load_s3_extension(config: Mapping[str, Any], output: Path):
    target, base_manifest, rubric = _load_frozen(config, output)
    _require(target, "pairwise-cache-s3-freeze")
    path = target / "s3_extension_manifest.json"
    if not path.exists() or load_json(path) != _s3_extension_manifest(config, output):
        raise RuntimeError("pairwise cache S3 extension manifest drift")
    return target, base_manifest, rubric


def s3_freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    value = _s3_extension_manifest(config, output)
    path = target / "s3_extension_manifest.json"
    if path.exists() and load_json(path) != value:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        if status.get("pairwise-cache-s3-run", {}).get("status") == "passed":
            raise RuntimeError("S3 extension manifest drift after inference")
    atomic_write_json(path, value)
    details = {
        "variant": value["variant"],
        "logical_request_count": value["logical_request_count"],
        "scheduler": value["scheduler"],
        "heldout_accessed": False,
    }
    _status(target, "pairwise-cache-s3-freeze", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def s3_run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_s3_extension(config, output)
    _verify_live_endpoints(config, manifest)
    rows = _rows(config, "discovery90")
    summary = _run_variant(
        config, output, target, manifest, rubric, rows, "discovery90",
        VARIANT_PROMPT_V2_DYNAMIC)
    details = {
        "variant": VARIANT_PROMPT_V2_DYNAMIC,
        "logical_request_count": len(rows) * len(rubric.nodes),
        "wall_seconds": summary["telemetry"]["wall_seconds"],
        "valid_rate": summary["valid_rate"],
        "heldout_accessed": False,
    }
    _status(target, "pairwise-cache-s3-run", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _effective_concurrency(summary: Mapping[str, Any]) -> float:
    telemetry = summary["telemetry"]
    wall = float(telemetry["wall_seconds"])
    if wall <= 0:
        return 0.0
    return (float(telemetry["recorded_model_call_count"])
            * float(telemetry["latency_seconds"]["mean"]) / wall)


def s3_report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_s3_extension(config, output)
    _require(target, "pairwise-cache-s3-run")
    rows = _rows(config, "discovery90")
    variants = (VARIANT_CONTROL, VARIANT_PROMPT_V2, VARIANT_PROMPT_V2_DYNAMIC)
    summaries, qualities = {}, {}
    for variant in variants:
        work = target / "discovery90" / variant
        summaries[variant] = load_json(work / "run_summary.json")
        prediction = PairwisePredictionOutput.load_json(work / "predictions.json")
        expected_spec = manifest["worker"][
            "v2_request_spec" if _uses_v2_prompt(variant) else "v1_request_spec"]
        prompt_version = (PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
                          if _uses_v2_prompt(variant) else "1.0.0")
        _validate_prediction(prediction, rows, rubric, expected_spec, prompt_version)
        qualities[variant] = _quality_metrics(prediction, rubric, rows)
    answers = {variant: qualities[variant]["m1"]["predictions"]
               for variant in variants}
    s3_quality = qualities[VARIANT_PROMPT_V2_DYNAMIC]
    control_quality = qualities[VARIANT_CONTROL]
    quality_non_degenerate = (
        s3_quality["m1"]["accuracy"] >= control_quality["m1"]["accuracy"] - 0.01
        and s3_quality["m1"]["coverage"] >= control_quality["m1"]["coverage"] - 0.01
        and s3_quality["valid_rate"] >= 0.99)
    efficiency_vs_control = _efficiency_delta(
        summaries[VARIANT_CONTROL], summaries[VARIANT_PROMPT_V2_DYNAMIC])
    efficiency_vs_s2 = _efficiency_delta(
        summaries[VARIANT_PROMPT_V2], summaries[VARIANT_PROMPT_V2_DYNAMIC])
    speed_target_met = efficiency_vs_control["wall_time_reduction_fraction"] >= 0.20
    value = {
        "schema_version": "1.0.0",
        "experiment": "pairwise-cache-prompt-s3-extension-v1",
        "split": "discovery90",
        "sample_count": len(rows),
        "variants": list(variants),
        "run_summaries": summaries,
        "quality": qualities,
        "paired": {
            "s3_vs_s0": _paired(
                rows, answers[VARIANT_CONTROL], answers[VARIANT_PROMPT_V2_DYNAMIC]),
            "s3_vs_s2": _paired(
                rows, answers[VARIANT_PROMPT_V2], answers[VARIANT_PROMPT_V2_DYNAMIC]),
        },
        "efficiency": {
            "s3_vs_s0": efficiency_vs_control,
            "s3_vs_s2": efficiency_vs_s2,
        },
        "effective_concurrency": {
            variant: _effective_concurrency(summaries[variant])
            for variant in variants},
        "quality_non_degenerate": quality_non_degenerate,
        "speed_target_met": speed_target_met,
        "go": quality_non_degenerate and speed_target_met,
        "cache_control_mode": manifest["protocol"]["cache_control_mode"],
        "timing_interpretation": (
            "exploratory_order_confounded; S3 executed after S0/S1/S2 without server cache reset"),
        "heldout_accessed": False,
        "selection_after_heldout_forbidden": True,
    }
    atomic_write_json(target / "discovery90" / "s3_report.json", value)
    details = {
        "go": value["go"],
        "speed_target_met": speed_target_met,
        "m1_accuracy": {
            variant: qualities[variant]["m1"]["accuracy"] for variant in variants},
        "wall_seconds": {
            variant: summaries[variant]["telemetry"]["wall_seconds"]
            for variant in variants},
        "effective_concurrency": value["effective_concurrency"],
        "heldout_accessed": False,
    }
    _status(target, "pairwise-cache-s3-report", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _s3_heldout_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target, base_manifest, rubric = _load_s3_extension(config, output)
    _require(target, "pairwise-cache-s3-report")
    discovery_report_path = target / "discovery90" / "s3_report.json"
    if not discovery_report_path.exists():
        raise RuntimeError("S3 discovery report is missing")
    heldout_path = base._path(config["heldout_dataset"])
    heldout_sha256 = file_sha256(heldout_path)
    if heldout_sha256.lower() != str(config["heldout_dataset_sha256"]).lower():
        raise RuntimeError("S3 heldout dataset hash drift")
    return {
        "schema_version": "1.0.0",
        "protocol_version": S3_HELDOUT_PROTOCOL_VERSION,
        "s3_extension_manifest_sha256": file_sha256(
            target / "s3_extension_manifest.json"),
        "discovery_s3_report_sha256": file_sha256(discovery_report_path),
        "rubric_sha256": rubric.rubric_sha256,
        "heldout_dataset_sha256": heldout_sha256,
        "sample_count": 500,
        "node_count": len(rubric.nodes),
        "variants": list(S3_HELDOUT_VARIANTS),
        "logical_requests_per_variant": 500 * len(rubric.nodes),
        "total_logical_requests": 500 * len(rubric.nodes) * 2,
        "request_specs": {
            VARIANT_CONTROL: base_manifest["worker"]["v1_request_spec"],
            VARIANT_PROMPT_V2_DYNAMIC: base_manifest["worker"]["v2_request_spec"],
        },
        "scheduler": "available-slot-dynamic-v1",
        "quality_non_inferiority_margin": 0.01,
        "exploratory_reused_heldout": True,
        "selection_after_heldout_forbidden": True,
    }


def _load_s3_heldout(config: Mapping[str, Any], output: Path):
    target, base_manifest, rubric = _load_s3_extension(config, output)
    _require(target, "pairwise-cache-s3-heldout-freeze")
    path = target / "s3_heldout_manifest.json"
    expected = _s3_heldout_manifest(config, output)
    if not path.exists() or load_json(path) != expected:
        raise RuntimeError("pairwise cache S3 heldout manifest drift")
    return target, base_manifest, rubric, expected


def s3_heldout_freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    value = _s3_heldout_manifest(config, output)
    path = target / "s3_heldout_manifest.json"
    if path.exists() and load_json(path) != value:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        if status.get("pairwise-cache-s3-heldout-run", {}).get("status") == "passed":
            raise RuntimeError("S3 heldout manifest drift after inference")
    atomic_write_json(path, value)
    details = {
        "variants": value["variants"],
        "sample_count": value["sample_count"],
        "total_logical_requests": value["total_logical_requests"],
        "exploratory_reused_heldout": True,
    }
    _status(target, "pairwise-cache-s3-heldout-freeze", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def s3_heldout_run(config: Mapping[str, Any], output: Path) -> None:
    target, base_manifest, rubric, frozen = _load_s3_heldout(config, output)
    _verify_live_endpoints(config, base_manifest)
    rows = _rows(config, "heldout500")
    summaries = {
        variant: _run_variant(
            config, output, target, base_manifest, rubric, rows, "heldout500", variant)
        for variant in S3_HELDOUT_VARIANTS
    }
    details = {
        "sample_count": len(rows),
        "logical_requests_per_variant": frozen["logical_requests_per_variant"],
        "variants": list(summaries),
        "wall_seconds": {
            variant: summary["telemetry"]["wall_seconds"]
            for variant, summary in summaries.items()},
        "exploratory_reused_heldout": True,
    }
    _status(target, "pairwise-cache-s3-heldout-run", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _node_delta_summary(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for name in control:
        old = control[name]
        new = treatment[name]
        rows.append({
            "criterion_name": name,
            "covered_accuracy_s0": old["covered_accuracy"],
            "covered_accuracy_s3": new["covered_accuracy"],
            "covered_accuracy_delta": new["covered_accuracy"] - old["covered_accuracy"],
            "coverage_s0": old["coverage"],
            "coverage_s3": new["coverage"],
            "coverage_delta": new["coverage"] - old["coverage"],
            "strict_accuracy_s0": old["strict_accuracy"],
            "strict_accuracy_s3": new["strict_accuracy"],
            "strict_accuracy_delta": new["strict_accuracy"] - old["strict_accuracy"],
        })
    return {
        "improved_strict_accuracy_count": sum(
            item["strict_accuracy_delta"] > 0 for item in rows),
        "unchanged_strict_accuracy_count": sum(
            item["strict_accuracy_delta"] == 0 for item in rows),
        "declined_strict_accuracy_count": sum(
            item["strict_accuracy_delta"] < 0 for item in rows),
        "criteria": rows,
    }


def s3_heldout_report(config: Mapping[str, Any], output: Path) -> None:
    target, base_manifest, rubric, frozen = _load_s3_heldout(config, output)
    _require(target, "pairwise-cache-s3-heldout-run")
    rows = _rows(config, "heldout500")
    summaries, qualities = {}, {}
    for variant in S3_HELDOUT_VARIANTS:
        work = target / "heldout500" / variant
        summaries[variant] = load_json(work / "run_summary.json")
        prediction = PairwisePredictionOutput.load_json(work / "predictions.json")
        expected_spec = base_manifest["worker"][
            "v2_request_spec" if _uses_v2_prompt(variant) else "v1_request_spec"]
        prompt_version = (PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
                          if _uses_v2_prompt(variant) else "1.0.0")
        _validate_prediction(prediction, rows, rubric, expected_spec, prompt_version)
        qualities[variant] = _quality_metrics(prediction, rubric, rows)
    control = qualities[VARIANT_CONTROL]
    treatment = qualities[VARIANT_PROMPT_V2_DYNAMIC]
    margin = float(frozen["quality_non_inferiority_margin"])
    accuracy_delta = treatment["m1"]["accuracy"] - control["m1"]["accuracy"]
    coverage_delta = treatment["m1"]["coverage"] - control["m1"]["coverage"]
    semantic_non_degenerate = (
        accuracy_delta >= -margin
        and coverage_delta >= -margin
        and treatment["valid_rate"] >= 0.99)
    efficiency = _efficiency_delta(
        summaries[VARIANT_CONTROL], summaries[VARIANT_PROMPT_V2_DYNAMIC])
    discovery = load_json(target / "discovery90" / "s3_report.json")
    value = {
        "schema_version": "1.0.0",
        "experiment": S3_HELDOUT_PROTOCOL_VERSION,
        "split": "heldout500",
        "sample_count": len(rows),
        "rubric_sha256": rubric.rubric_sha256,
        "variants": list(S3_HELDOUT_VARIANTS),
        "run_summaries": summaries,
        "quality": qualities,
        "paired_s3_vs_s0": _paired(
            rows, control["m1"]["predictions"], treatment["m1"]["predictions"]),
        "efficiency_s3_vs_s0": efficiency,
        "node_deltas_s3_vs_s0": _node_delta_summary(
            control["node_metrics"], treatment["node_metrics"]),
        "semantic_drift_diagnostic": {
            "non_inferiority_margin": margin,
            "m1_accuracy_delta": accuracy_delta,
            "coverage_delta": coverage_delta,
            "valid_rate_s0": control["valid_rate"],
            "valid_rate_s3": treatment["valid_rate"],
            "semantic_non_degenerate": semantic_non_degenerate,
            "discovery_accuracy_delta": (
                discovery["quality"][VARIANT_PROMPT_V2_DYNAMIC]["m1"]["accuracy"]
                - discovery["quality"][VARIANT_CONTROL]["m1"]["accuracy"]),
        },
        "speed_target_met": efficiency["wall_time_reduction_fraction"] >= 0.20,
        "go": semantic_non_degenerate,
        "timing_interpretation": (
            "exploratory_order_confounded; S0 then S3 without server cache reset"),
        "exploratory_reused_heldout": True,
        "selection_after_heldout_forbidden": True,
    }
    path = target / "heldout500" / "s3_report.json"
    atomic_write_json(path, value)
    details = {
        "go": value["go"],
        "semantic_non_degenerate": semantic_non_degenerate,
        "speed_target_met": value["speed_target_met"],
        "m1_accuracy": {
            variant: qualities[variant]["m1"]["accuracy"]
            for variant in S3_HELDOUT_VARIANTS},
        "coverage": {
            variant: qualities[variant]["m1"]["coverage"]
            for variant in S3_HELDOUT_VARIANTS},
        "wall_seconds": {
            variant: summaries[variant]["telemetry"]["wall_seconds"]
            for variant in S3_HELDOUT_VARIANTS},
        "report": str(path),
    }
    _status(target, "pairwise-cache-s3-heldout-report", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {
        "pairwise-cache-freeze": freeze,
        "pairwise-cache-audit": audit,
        "pairwise-cache-smoke": smoke,
        "pairwise-cache-discovery-run": discovery_run,
        "pairwise-cache-discovery-report": discovery_report,
        "pairwise-cache-heldout-run": heldout_run,
        "pairwise-cache-final-report": final_report,
        "pairwise-cache-s3-freeze": s3_freeze,
        "pairwise-cache-s3-run": s3_run,
        "pairwise-cache-s3-report": s3_report,
        "pairwise-cache-s3-heldout-freeze": s3_heldout_freeze,
        "pairwise-cache-s3-heldout-run": s3_heldout_run,
        "pairwise-cache-s3-heldout-report": s3_heldout_report,
    }
    try:
        actions[stage](config, output)
    except KeyError as exc:
        raise ValueError(f"unsupported pairwise cache ablation stage: {stage}") from exc

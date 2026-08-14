"""Phase10 dual-endpoint external transfer evaluation on VL-RewardBench.

This protocol freezes the final 22-node Split+Refine rubric, evaluates every
criterion once per sample/order, and reuses those predictions for equal-root,
weighted-root, initial-root, and root-local offline aggregations.  Physical
requests are deterministically pinned to vLLM ports 8000 and 8001.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    CacheMode,
    FinalPreference,
    JsonPredictionCache,
    ModelCallMetrics,
    PairwisePredictionOutput,
    StructuredRubric,
    aggregate_flat_votes,
    aggregate_weighted_root_votes,
)

from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from .experiment_utils import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    make_progress_callback,
)
from .rubric_factory import build_multicrit_open_ended_init_rubric, file_sha256


EXPERIMENT_DIR = "vl_rewardbench_phase10_transfer_v1"
SOURCE_EXPERIMENT = "phase10_five_root_locked_split_refine_v1"
SOURCE_RUBRIC_SHA256 = "17ad7a0b6350338b100384b701303008746463911662f4a628b90c4632cdf7ef"
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")
FINAL_SYSTEM = "phase10_final"
INITIAL_SYSTEM = "initial_five_root_equal"
EQUAL_SYSTEM = "phase10_final_equal"
WEIGHTED_SYSTEM = "phase10_final_weighted"
NATIVE_SYSTEM = "native_vlrb_prompt"
VISUAL_SYSTEM = "phase10_visual_only"
VISUAL_ROOT_ID = "init_02_visual_grounding_and_details"
ROOT_WEIGHTS = {
    "init_01_completeness_and_coverage": 0.15,
    VISUAL_ROOT_ID: 0.40,
    "init_03_factuality_no_hallucination": 0.15,
    "init_04_creativity_and_expressiveness": 0.15,
    "init_05_clarity_and_coherence": 0.15,
}
ROUTE_PROTOCOL = "vlrb-phase10-sha256-mod2-v1"
PARITY_COUNT = 20


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _source_rubric_path(output: Path) -> Path:
    return (output / SOURCE_EXPERIMENT / "final/rubric.json").resolve()


def _status_path(target: Path) -> Path:
    return target / "stage_status.json"


def _status(target: Path, stage: str, details: Mapping[str, Any] | None = None) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    value[stage] = {"status": "passed", "details": dict(details or {})}
    atomic_write_json(_status_path(target), value)


def _require(target: Path, stage: str) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _pool_spec(config: Mapping[str, Any]) -> BackendPoolSpec:
    spec = BackendPoolSpec.from_dict(config["backend_pool"])
    if tuple(sorted(item.endpoint_id for item in spec.endpoints)) != ENDPOINT_IDS:
        raise RuntimeError("Phase10 VL-RewardBench requires exactly vllm-8000 and vllm-8001")
    if spec.common_checkpoint_id != config["model"]:
        raise RuntimeError("dual endpoints do not declare the configured common checkpoint")
    return spec


def _inspect_endpoints(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    identities = []
    for endpoint in sorted(_pool_spec(config).endpoints, key=lambda item: item.endpoint_id):
        base_url = endpoint.base_url.rstrip("/")
        version = base.request_json(base_url[:-3] + "/version")
        models = base.request_json(base_url + "/models")
        matches = [item for item in models.get("data", []) if item.get("id") == config["model"]]
        if len(matches) != 1:
            raise RuntimeError(f"{endpoint.endpoint_id} does not expose the configured model exactly once")
        identity = {
            "endpoint_id": endpoint.endpoint_id,
            "base_url": endpoint.base_url,
            "vllm_version": version.get("version"),
            "model": matches[0].get("id"),
            "checkpoint_root": matches[0].get("root"),
            "max_model_len": matches[0].get("max_model_len"),
            "max_concurrency": endpoint.max_concurrency,
        }
        if (identity["vllm_version"] != config["vllm_version"]
                or identity["model"] != config["model"]):
            raise RuntimeError(f"Phase10 endpoint identity mismatch: {identity}")
        if not isinstance(identity["max_model_len"], int) or identity["max_model_len"] < 4096:
            raise RuntimeError(f"Phase10 endpoint context window is too small: {identity}")
        identities.append(identity)
    return identities


def _criterion_hash(name: str, description: str) -> str:
    return canonical_sha256({"name": name, "description": description})


def _route_key(system: str, replicate: int, sample_id: str,
               criterion_hash: str = "native") -> str:
    return f"{system}|r{replicate + 1:02d}|{sample_id}|{criterion_hash}"


def _route_endpoint(route_key: str) -> str:
    index = int(hashlib.sha256(route_key.encode("utf-8")).hexdigest(), 16) % len(ENDPOINT_IDS)
    return ENDPOINT_IDS[index]


def _endpoint_schedule(records: Sequence[Mapping[str, Any]],
                       rubric: StructuredRubric) -> dict[str, Any]:
    routes: dict[str, str] = {}
    criteria = [rubric.get_node(node_id).criterion
                for node_id in rubric.preorder_node_ids()]
    for replicate in range(legacy.K):
        for record in records:
            sample_id = str(record["sample_id"])
            native_key = _route_key(NATIVE_SYSTEM, replicate, sample_id)
            routes[native_key] = _route_endpoint(native_key)
            for criterion in criteria:
                key = _route_key(
                    FINAL_SYSTEM, replicate, sample_id,
                    _criterion_hash(criterion.name, criterion.description),
                )
                routes[key] = _route_endpoint(key)
    counts = {endpoint_id: sum(value == endpoint_id for value in routes.values())
              for endpoint_id in ENDPOINT_IDS}
    return {
        "schema_version": "1.0.0",
        "protocol": ROUTE_PROTOCOL,
        "endpoint_ids": list(ENDPOINT_IDS),
        "route_count": len(routes),
        "endpoint_counts": counts,
        "routes": routes,
    }


def _validate_endpoint_schedule(value: Mapping[str, Any],
                                records: Sequence[Mapping[str, Any]],
                                rubric: StructuredRubric) -> None:
    expected_count = len(records) * legacy.K * (1 + len(rubric.nodes))
    routes = value.get("routes")
    if (value.get("schema_version") != "1.0.0"
            or value.get("protocol") != ROUTE_PROTOCOL
            or value.get("endpoint_ids") != list(ENDPOINT_IDS)
            or value.get("route_count") != expected_count
            or not isinstance(routes, dict)
            or len(routes) != expected_count
            or set(routes.values()) != set(ENDPOINT_IDS)):
        raise RuntimeError("Phase10 endpoint schedule schema or size drift")
    counts = {endpoint_id: sum(item == endpoint_id for item in routes.values())
              for endpoint_id in ENDPOINT_IDS}
    if value.get("endpoint_counts") != counts:
        raise RuntimeError("Phase10 endpoint schedule count drift")
    if (expected_count >= 1000
            and abs(counts[ENDPOINT_IDS[0]] - counts[ENDPOINT_IDS[1]]) / expected_count > 0.02):
        raise RuntimeError("Phase10 endpoint schedule is not approximately balanced")
    for key, endpoint_id in routes.items():
        if _route_endpoint(str(key)) != endpoint_id:
            raise RuntimeError("Phase10 endpoint schedule hash assignment drift")


def _root_reuse_contract(initial: StructuredRubric,
                         final: StructuredRubric) -> dict[str, str]:
    if tuple(initial.root_ids) != tuple(final.root_ids):
        raise RuntimeError("Phase10 final root IDs differ from the initial rubric")
    mapping = {}
    for root_id in initial.root_ids:
        old = initial.get_node(root_id).criterion
        new = final.get_node(root_id).criterion
        if (old.name, old.description, old.score) != (new.name, new.description, new.score):
            raise RuntimeError(f"Phase10 root cannot be reused exactly: {root_id}")
        mapping[old.name] = root_id
    return mapping


def _manifest(config: Mapping[str, Any], output: Path,
              records: Sequence[Mapping[str, Any]],
              order_schedule: Mapping[str, Sequence[int]],
              endpoint_schedule: Mapping[str, Any], *,
              include_endpoints: bool) -> dict[str, Any]:
    source_path = _source_rubric_path(output)
    if not source_path.is_file():
        raise RuntimeError(f"Phase10 final rubric is missing: {source_path}")
    final = StructuredRubric.load_json(source_path)
    initial = build_multicrit_open_ended_init_rubric()
    if final.rubric_sha256 != SOURCE_RUBRIC_SHA256 or len(final.nodes) != 22:
        raise RuntimeError("Phase10 final rubric hash or node count drift")
    reuse = _root_reuse_contract(initial, final)
    if set(ROOT_WEIGHTS) != set(final.root_ids) or abs(sum(ROOT_WEIGHTS.values()) - 1.0) > 1e-12:
        raise RuntimeError("Phase10 root weights are invalid")
    rows = legacy._ordered_rows(records, order_schedule, 0)
    structured_spec = base._expected_pairwise_request_spec(config, rows).to_dict()
    spec = _pool_spec(config)
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "overlap_audit": "skipped_by_protocol",
        "dataset": {
            "path": str(legacy._parquet_path()),
            "sha256": file_sha256(legacy._parquet_path()),
            "count": len(records),
            "record_sha256": canonical_sha256(list(records)),
        },
        "order_schedule_sha256": canonical_sha256(order_schedule),
        "endpoint_schedule_sha256": canonical_sha256(endpoint_schedule),
        "counterbalance": {"k": legacy.K, "seed": legacy.SEED,
                           "protocol": "balanced_b_1minusb_b"},
        "native": {
            "model": config["model"],
            "backend_id": spec.backend_id,
            "prompt_sha256": file_sha256(legacy._prompt_path()),
            "decoding": legacy.NATIVE_DECODING,
            "parser": "vl_rewardbench_overall_judgment_regex_v1",
            "k": legacy.K,
            "seed": legacy.SEED,
        },
        "structured_worker_request_spec": structured_spec,
        "systems": {
            NATIVE_SYSTEM: {"prompt_path": str(legacy._prompt_path())},
            INITIAL_SYSTEM: {
                "rubric_path": "deterministic:build_multicrit_open_ended_init_rubric",
                "rubric_sha256": initial.rubric_sha256,
                "node_count": len(initial.nodes),
                "root_weights": {root_id: 0.2 for root_id in initial.root_ids},
            },
            EQUAL_SYSTEM: {
                "rubric_path": str(source_path),
                "file_sha256": file_sha256(source_path),
                "rubric_sha256": final.rubric_sha256,
                "node_count": len(final.nodes),
                "root_weights": {root_id: 0.2 for root_id in final.root_ids},
            },
            WEIGHTED_SYSTEM: {
                "prediction_source": EQUAL_SYSTEM,
                "rubric_sha256": final.rubric_sha256,
                "root_weights": dict(ROOT_WEIGHTS),
            },
        },
        "root_prediction_reuse": reuse,
        "endpoint_pool": spec.to_dict(),
        "selection_after_benchmark_forbidden": True,
    }
    if include_endpoints:
        value["endpoint_identities"] = _inspect_endpoints(config)
    return value


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target()
    _require(target, "vlrb-phase10-freeze")
    records = legacy._read_records(target)
    order_schedule = legacy._order_schedule(records)
    endpoint_schedule = load_json(target / "endpoint_schedule.json")
    final = StructuredRubric.load_json(_source_rubric_path(output))
    _validate_endpoint_schedule(endpoint_schedule, records, final)
    expected = _manifest(
        config, output, records, order_schedule, endpoint_schedule,
        include_endpoints=False,
    )
    stored = load_json(target / "frozen_manifest.json")
    stored_static = {key: value for key, value in stored.items()
                     if key != "endpoint_identities"}
    if expected != stored_static:
        raise RuntimeError("Phase10 VL-RewardBench frozen manifest drift")
    return target, stored, records, order_schedule, endpoint_schedule, final


def _verify_live_endpoints(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if manifest.get("endpoint_identities") != _inspect_endpoints(config):
        raise RuntimeError("Phase10 VL-RewardBench live endpoint identity drift")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target()
    records = legacy._read_records(target)
    order_schedule = legacy._order_schedule(records)
    final = StructuredRubric.load_json(_source_rubric_path(output))
    endpoint_schedule = _endpoint_schedule(records, final)
    _validate_endpoint_schedule(endpoint_schedule, records, final)
    manifest = _manifest(
        config, output, records, order_schedule, endpoint_schedule,
        include_endpoints=True,
    )
    manifest_path = target / "frozen_manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        downstream = ("vlrb-phase10-smoke", "vlrb-phase10-run", "vlrb-phase10-report")
        if any(status.get(stage, {}).get("status") == "passed" for stage in downstream):
            raise RuntimeError("Phase10 VL-RewardBench manifest drift after inference")
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(target / "dataset_manifest.json", manifest["dataset"])
    atomic_write_json(target / "order_schedule.json", order_schedule)
    atomic_write_json(target / "endpoint_schedule.json", endpoint_schedule)
    _status(target, "vlrb-phase10-freeze", {
        "dataset_count": len(records),
        "rubric_sha256": final.rubric_sha256,
        "node_count": len(final.nodes),
        "endpoint_counts": endpoint_schedule["endpoint_counts"],
    })
    print(json.dumps({
        "dataset_count": len(records),
        "rubric_sha256": final.rubric_sha256,
        "node_count": len(final.nodes),
        "endpoint_counts": endpoint_schedule["endpoint_counts"],
    }, indent=2, ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, order_schedule, endpoint_schedule, final = _load_frozen(
        config, output)
    del order_schedule
    initial = build_multicrit_open_ended_init_rubric()
    reuse = _root_reuse_contract(initial, final)
    expected_routes = len(records) * legacy.K * (1 + len(final.nodes))
    value = {
        "schema_version": "1.0.0",
        "status": "passed",
        "heldout_accessed": False,
        "dataset_count": len(records),
        "source_rubric_sha256": final.rubric_sha256,
        "source_node_count": len(final.nodes),
        "initial_root_reuse_count": len(reuse),
        "root_weights": dict(ROOT_WEIGHTS),
        "root_weight_sum": sum(ROOT_WEIGHTS.values()),
        "endpoint_route_count": endpoint_schedule["route_count"],
        "expected_route_count": expected_routes,
        "endpoint_counts": endpoint_schedule["endpoint_counts"],
        "endpoint_identities_frozen": len(manifest["endpoint_identities"]),
        "benchmark_labels_in_model_prompt": False,
        "selection_after_benchmark_forbidden": True,
    }
    atomic_write_json(target / "offline_audit.json", value)
    _status(target, "vlrb-phase10-audit", value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


class _FrozenEndpointRouter:
    """Route each logical request to its pre-frozen physical endpoint."""

    def __init__(self, spec: BackendPoolSpec, routes: Mapping[str, str],
                 schedule_sha256: str) -> None:
        self.spec = spec
        self._routes = dict(routes)
        self._schedule_sha256 = schedule_sha256
        self._global = threading.BoundedSemaphore(spec.global_request_concurrency)
        self._pools = {}
        for endpoint in spec.endpoints:
            single = BackendPoolSpec(
                pool_id=f"{spec.pool_id}-frozen-{endpoint.endpoint_id}",
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

    def records_by_endpoint(self) -> Mapping[str, int]:
        return {endpoint_id: len(pool.records)
                for endpoint_id, pool in self._pools.items()}

    def call(self, prompt: object, *, request_type: str, request_key: str,
             structured_attempt: int, agent_args: Mapping[str, Any]):
        try:
            endpoint_id = self._routes[request_key]
        except KeyError as exc:
            raise RuntimeError(f"request is absent from frozen endpoint schedule: {request_key}") from exc
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
            "routing_protocol": ROUTE_PROTOCOL,
            "endpoint_schedule_sha256": self._schedule_sha256,
            "endpoint_call_counts": dict(self.records_by_endpoint()),
            "calls": [record.to_dict() for record in self.records],
        }


def _structured_route_map(records: Sequence[Mapping[str, Any]],
                          rubric: StructuredRubric, replicate: int,
                          endpoint_schedule: Mapping[str, Any]) -> dict[str, str]:
    routes = endpoint_schedule["routes"]
    result = {}
    for record in records:
        sample_id = str(record["sample_id"])
        for node_id in rubric.preorder_node_ids():
            criterion = rubric.get_node(node_id).criterion
            evaluator_key = f"{sample_id}::{criterion.name}"
            frozen_key = _route_key(
                FINAL_SYSTEM, replicate, sample_id,
                _criterion_hash(criterion.name, criterion.description),
            )
            result[evaluator_key] = routes[frozen_key]
    return result


def _native_route_map(records: Sequence[Mapping[str, Any]], replicate: int,
                      endpoint_schedule: Mapping[str, Any]) -> dict[str, str]:
    routes = endpoint_schedule["routes"]
    return {
        f"{record['sample_id']}::replicate_{replicate + 1:02d}": routes[
            _route_key(NATIVE_SYSTEM, replicate, str(record["sample_id"]))
        ]
        for record in records
    }


def _validate_prediction(prediction: PairwisePredictionOutput,
                         rubric: StructuredRubric,
                         rows: Sequence[Mapping[str, Any]],
                         manifest: Mapping[str, Any]) -> None:
    if tuple(prediction.sample_ids) != tuple(str(row["sample_id"]) for row in rows):
        raise RuntimeError("Phase10 structured sample order drift")
    expected = tuple(
        rubric.get_node(node_id).criterion.name
        for node_id in rubric.preorder_node_ids()
    )
    if tuple(item.name for item in prediction.criteria) != expected:
        raise RuntimeError("Phase10 structured criterion order drift")
    if prediction.request_spec.to_dict() != manifest["structured_worker_request_spec"]:
        raise RuntimeError("Phase10 structured request identity drift")


def _structured_prediction(config: Mapping[str, Any], work: Path,
                           manifest: Mapping[str, Any], rubric: StructuredRubric,
                           rows: Sequence[Mapping[str, Any]], replicate: int,
                           endpoint_schedule: Mapping[str, Any]) -> PairwisePredictionOutput:
    run_dir = work / "structured" / FINAL_SYSTEM / f"replicate_{replicate + 1:02d}"
    artifact = run_dir / "predictions" / f"{FINAL_SYSTEM}.json"
    route_map = _structured_route_map(rows, rubric, replicate, endpoint_schedule)
    assignments = {
        "schema_version": "1.0.0",
        "replicate": replicate + 1,
        "endpoint_schedule_sha256": manifest["endpoint_schedule_sha256"],
        "assignment_count": len(route_map),
        "endpoint_counts": {
            endpoint_id: sum(value == endpoint_id for value in route_map.values())
            for endpoint_id in ENDPOINT_IDS
        },
        "assignments": route_map,
    }
    assignment_path = run_dir / "endpoint_assignments.json"
    if assignment_path.exists() and load_json(assignment_path) != assignments:
        raise RuntimeError("Phase10 structured endpoint assignment drift")
    atomic_write_json(assignment_path, assignments)
    if artifact.exists():
        prediction = PairwisePredictionOutput.load_json(artifact)
        _validate_prediction(prediction, rubric, rows, manifest)
        return prediction
    router = _FrozenEndpointRouter(
        _pool_spec(config), route_map, manifest["endpoint_schedule_sha256"])
    model_rows = base._model_rows(rows)
    evaluator = base._pairwise_evaluator(
        config, model_rows, router,
        request_backend_id=manifest["structured_worker_request_spec"]["backend_id"],
    )
    cache = JsonPredictionCache(run_dir / "cache/phase10_pairwise", CacheMode.READ_WRITE)
    callback = make_progress_callback(
        run_dir, f"phase10_replicate_{replicate + 1:02d}",
        len(rows) * len(rubric.nodes), router)
    prediction = base._pairwise_cached(
        evaluator, model_rows, rubric, cache, None,
        evaluation_callback=callback,
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    prediction.save_json(artifact)
    valid = sum(
        output.parse_ok and output.answer_valid
        for row in prediction.node_outputs for output in row.values()
    )
    total = len(rows) * len(rubric.nodes)
    provenance = router.provenance_dict()
    provenance["request_backend_id"] = evaluator.request_spec().backend_id
    atomic_write_json(run_dir / "provenance.json", provenance)
    if valid / total < 0.95:
        raise RuntimeError("Phase10 structured final-valid rate below 95%")
    _validate_prediction(prediction, rubric, rows, manifest)
    return prediction


def _native_cache_path(work: Path, sample_id: str, replicate: int) -> Path:
    digest = hashlib.sha256(
        f"phase10-native-v1|{sample_id}|{replicate}".encode("utf-8")
    ).hexdigest()
    return work / "native/cache" / f"{digest}.json"


def _native_result(config: Mapping[str, Any], work: Path,
                   manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                   order_schedule: Mapping[str, Sequence[int]], replicate: int,
                   endpoint_schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    output_path = work / "native" / f"replicate_{replicate + 1:02d}.json"
    request_identity = canonical_sha256(manifest["native"])
    route_map = _native_route_map(records, replicate, endpoint_schedule)
    expected_ids = [str(record["sample_id"]) for record in records]
    if output_path.exists():
        value = load_json(output_path)
        if (isinstance(value, list) and len(value) == len(records)
                and [item.get("sample_id") for item in value] == expected_ids
                and all(item.get("request_identity") == request_identity for item in value)
                and all(item.get("endpoint_id") == route_map[
                    f"{item['sample_id']}::replicate_{replicate + 1:02d}"] for item in value)):
            return value
        raise RuntimeError("Phase10 native output identity drift")
    router = _FrozenEndpointRouter(
        _pool_spec(config), route_map, manifest["endpoint_schedule_sha256"])
    callback = make_progress_callback(
        work / "native", f"native_replicate_{replicate + 1:02d}",
        len(records), router)

    def one(index: int, record: Mapping[str, Any]):
        sample_id = str(record["sample_id"])
        order = int(order_schedule[sample_id][replicate])
        request_key = f"{sample_id}::replicate_{replicate + 1:02d}"
        endpoint_id = route_map[request_key]
        cache_path = _native_cache_path(work, sample_id, replicate)
        if cache_path.exists():
            cached = load_json(cache_path)
            if (cached.get("sample_id") == sample_id
                    and cached.get("order") == order
                    and cached.get("request_identity") == request_identity
                    and cached.get("endpoint_id") == endpoint_id):
                return index, cached, ModelCallMetrics.from_agent_calls((), cache_hit=True)
            raise RuntimeError("Phase10 native cache identity drift")
        raw, metrics = router.call(
            legacy._native_content(record, order),
            request_type="vlrb_phase10_native",
            request_key=request_key,
            structured_attempt=1,
            agent_args={
                "model": config["model"], "api_keys": "EMPTY",
                "request_kwargs": legacy.NATIVE_DECODING,
                "api_retry_attempts": config["api_retry_attempts"],
            },
        )
        display_choice = legacy._parse_native(raw)
        normalized = ModelCallMetrics.from_agent_calls((metrics,))
        result = {
            "sample_id": sample_id,
            "order": order,
            "endpoint_id": endpoint_id,
            "request_identity": request_identity,
            "display_choice": display_choice,
            "parse_ok": display_choice is not None,
            "raw_response": raw,
            "metrics": normalized.to_dict(),
        }
        atomic_write_json(cache_path, result)
        return index, result, metrics

    results: list[dict[str, Any] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=router.spec.global_request_concurrency) as executor:
        futures = {executor.submit(one, index, record): (index, record)
                   for index, record in enumerate(records)}
        for future in as_completed(futures):
            index, record = futures[future]
            result_index, result, metrics = future.result()
            results[result_index] = result
            callback(index, str(record["sample_id"]), metrics)
    completed = [item for item in results if item is not None]
    atomic_write_json(output_path, completed)
    atomic_write_json(
        work / "native" / f"provenance_replicate_{replicate + 1:02d}.json",
        router.provenance_dict(),
    )
    return completed


def _subset_prediction(prediction: PairwisePredictionOutput,
                       rubric: StructuredRubric) -> PairwisePredictionOutput:
    by_name = {item.name: item for item in prediction.criteria}
    criteria = tuple(
        by_name[rubric.get_node(node_id).criterion.name]
        for node_id in rubric.preorder_node_ids()
    )
    for node_id, snapshot in zip(rubric.preorder_node_ids(), criteria):
        criterion = rubric.get_node(node_id).criterion
        if (snapshot.name, snapshot.description) != (criterion.name, criterion.description):
            raise RuntimeError("Initial prediction reuse description drift")
    names = tuple(item.name for item in criteria)
    outputs = tuple({name: row[name] for name in names}
                    for row in prediction.node_outputs)
    answers = tuple(aggregate_flat_votes(item.vote for item in row.values())
                    for row in outputs)
    return PairwisePredictionOutput(
        prediction.sample_ids,
        prediction.sample_fingerprints,
        criteria,
        outputs,
        answers,
        prediction.request_spec,
    )


def _weighted_answers(execution, weights: Mapping[str, float]):
    answers = []
    for trace in execution.traces:
        selected = tuple(root.root_id for root in trace.roots if root.selected)
        votes = {root.root_id: root.subtree_vote for root in trace.roots if root.selected}
        if any(value is None for value in votes.values()):
            raise RuntimeError("Phase10 selected root trace lacks a subtree vote")
        answers.append(aggregate_weighted_root_votes(
            votes, selected, {root_id: weights[root_id] for root_id in selected}))
    return tuple(answers)


def _logical_votes(config: Mapping[str, Any], output: Path, work: Path,
                   manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                   order_schedule: Mapping[str, Sequence[int]], final: StructuredRubric,
                   endpoint_schedule: Mapping[str, Any]):
    initial = build_multicrit_open_ended_init_rubric()
    native_by_replicate = []
    final_predictions = []
    votes = {name: [] for name in (
        NATIVE_SYSTEM, INITIAL_SYSTEM, EQUAL_SYSTEM, WEIGHTED_SYSTEM, VISUAL_SYSTEM)}
    root_votes = {root_id: [] for root_id in final.root_ids}
    for replicate in range(legacy.K):
        native = _native_result(
            config, work, manifest, records, order_schedule, replicate,
            endpoint_schedule)
        native_by_replicate.append(native)
        rows = legacy._ordered_rows(records, order_schedule, replicate)
        prediction = _structured_prediction(
            config, work, manifest, final, rows, replicate, endpoint_schedule)
        final_predictions.append(prediction)
        execution, equal_answers = base.execute_offline_m1(final, prediction, rows)
        initial_prediction = _subset_prediction(prediction, initial)
        _, initial_answers = base.execute_offline_m1(initial, initial_prediction, rows)
        weighted_answers = _weighted_answers(execution, ROOT_WEIGHTS)
        orders = [int(order_schedule[str(row["sample_id"])][replicate]) for row in rows]
        votes[INITIAL_SYSTEM].append([
            legacy._original_index(answer.value, order)
            for answer, order in zip(initial_answers, orders)])
        votes[EQUAL_SYSTEM].append([
            legacy._original_index(answer.value, order)
            for answer, order in zip(equal_answers, orders)])
        votes[WEIGHTED_SYSTEM].append([
            legacy._original_index(answer.value, order)
            for answer, order in zip(weighted_answers, orders)])
        for root_id in final.root_ids:
            display_votes = []
            for trace in execution.traces:
                matches = [root for root in trace.roots if root.root_id == root_id]
                if len(matches) != 1 or matches[0].subtree_vote is None:
                    raise RuntimeError(f"Phase10 root trace is incomplete: {root_id}")
                display_votes.append(matches[0].subtree_vote.value)
            normalized = [legacy._original_index(value, order)
                          for value, order in zip(display_votes, orders)]
            root_votes[root_id].append(normalized)
        votes[VISUAL_SYSTEM].append(root_votes[VISUAL_ROOT_ID][-1])
    votes[NATIVE_SYSTEM] = [
        [legacy._original_index(item["display_choice"], int(item["order"]))
         for item in replicate]
        for replicate in native_by_replicate
    ]
    value = {
        "schema_version": "1.0.0",
        "sample_ids": [str(row["sample_id"]) for row in records],
        "k": legacy.K,
        "systems": {name: {"votes_by_replicate": item} for name, item in votes.items()},
        "root_votes_by_replicate": root_votes,
        "prediction_reuse": {
            "phase10_equal_and_weighted_shared": True,
            "initial_roots_projected_from_phase10": True,
        },
    }
    atomic_write_json(work / "combined/logical_votes.json", value)
    return value, final_predictions


def _parity_diagnostic(config: Mapping[str, Any], work: Path,
                       manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                       order_schedule: Mapping[str, Sequence[int]], final: StructuredRubric,
                       endpoint_schedule: Mapping[str, Any],
                       prediction: PairwisePredictionOutput) -> dict[str, Any]:
    replicate = 0
    rows = legacy._ordered_rows(records, order_schedule, replicate)
    model_rows = base._model_rows(rows)
    spec = _pool_spec(config)
    pools = {}
    evaluators = {}
    for endpoint_id in ENDPOINT_IDS:
        pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(
            base._single_endpoint_execution_pool(config, endpoint_id)))
        pools[endpoint_id] = pool
        evaluators[endpoint_id] = base._pairwise_evaluator(
            config, model_rows, pool,
            request_backend_id=manifest["structured_worker_request_spec"]["backend_id"],
        )
    tasks = []
    for sample_index, row in enumerate(rows):
        for node_id in final.preorder_node_ids():
            criterion = final.get_node(node_id).criterion
            frozen_key = _route_key(
                FINAL_SYSTEM, replicate, str(row["sample_id"]),
                _criterion_hash(criterion.name, criterion.description),
            )
            scheduled = endpoint_schedule["routes"][frozen_key]
            opposite = ENDPOINT_IDS[1] if scheduled == ENDPOINT_IDS[0] else ENDPOINT_IDS[0]
            tasks.append((sample_index, node_id, scheduled, opposite))
            if len(tasks) == PARITY_COUNT:
                break
        if len(tasks) == PARITY_COUNT:
            break
    callback = make_progress_callback(work / "parity", "endpoint_parity", len(tasks))

    def one(index: int, task):
        sample_index, node_id, scheduled, opposite = task
        criterion = final.get_node(node_id).criterion
        cache_key = canonical_sha256({
            "sample_id": rows[sample_index]["sample_id"],
            "criterion_hash": _criterion_hash(criterion.name, criterion.description),
            "opposite_endpoint": opposite,
            "request_spec": manifest["structured_worker_request_spec"],
        })
        cache_path = work / "parity/cache" / f"{cache_key}.json"
        if cache_path.exists():
            cached = load_json(cache_path)
            return index, cached, ModelCallMetrics.from_agent_calls((), cache_hit=True)
        output, metrics = evaluators[opposite].infer_one(
            model_rows[sample_index], criterion.to_criterion())
        source = prediction.node_outputs[sample_index][criterion.name]
        value = {
            "sample_id": rows[sample_index]["sample_id"],
            "criterion_name": criterion.name,
            "scheduled_endpoint": scheduled,
            "parity_endpoint": opposite,
            "scheduled_vote": source.vote.value,
            "scheduled_valid": source.parse_ok and source.answer_valid,
            "parity_vote": output.vote.value,
            "parity_valid": output.parse_ok and output.answer_valid,
            "vote_agreement": source.vote == output.vote,
            "parity_output": output.to_dict(),
            "metrics": metrics.to_dict(),
        }
        atomic_write_json(cache_path, value)
        return index, value, metrics

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=min(20, spec.global_request_concurrency)) as executor:
        futures = {executor.submit(one, index, task): index
                   for index, task in enumerate(tasks)}
        for future in as_completed(futures):
            index, value, metrics = future.result()
            results[index] = value
            callback(index, str(value["sample_id"]), metrics)
    completed = [item for item in results if item is not None]
    scheduled_valid = sum(bool(item["scheduled_valid"]) for item in completed) / len(completed)
    parity_valid = sum(bool(item["parity_valid"]) for item in completed) / len(completed)
    report = {
        "schema_version": "1.0.0",
        "request_count": len(completed),
        "scheduled_valid_rate": scheduled_valid,
        "parity_valid_rate": parity_valid,
        "vote_agreement_rate": sum(bool(item["vote_agreement"]) for item in completed) / len(completed),
        "by_parity_endpoint": {
            endpoint_id: sum(item["parity_endpoint"] == endpoint_id for item in completed)
            for endpoint_id in ENDPOINT_IDS
        },
        "items": completed,
    }
    if scheduled_valid < 0.95 or parity_valid < 0.95:
        raise RuntimeError("Phase10 endpoint parity valid rate below 95%")
    atomic_write_json(work / "parity/report.json", report)
    atomic_write_json(work / "parity/provenance.json", {
        endpoint_id: pool.provenance_dict() for endpoint_id, pool in pools.items()})
    return report


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, order_schedule, endpoint_schedule, final = _load_frozen(
        config, output)
    _require(target, "vlrb-phase10-audit")
    _verify_live_endpoints(config, manifest)
    selected = tuple(records[:20])
    _, predictions = _logical_votes(
        config, output, target / "smoke", manifest, selected,
        order_schedule, final, endpoint_schedule)
    parity = _parity_diagnostic(
        config, target / "smoke", manifest, selected, order_schedule,
        final, endpoint_schedule, predictions[0])
    details = {
        "sample_count": len(selected),
        "main_request_count": len(selected) * legacy.K * (1 + len(final.nodes)),
        "parity_request_count": parity["request_count"],
        "parity_valid_rate": parity["parity_valid_rate"],
    }
    _status(target, "vlrb-phase10-smoke", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, order_schedule, endpoint_schedule, final = _load_frozen(
        config, output)
    _require(target, "vlrb-phase10-smoke")
    _verify_live_endpoints(config, manifest)
    _logical_votes(
        config, output, target / "run", manifest, records,
        order_schedule, final, endpoint_schedule)
    details = {
        "sample_count": len(records),
        "unique_request_count": len(records) * legacy.K * (1 + len(final.nodes)),
        "node_count": len(final.nodes),
        "endpoint_ids": list(ENDPOINT_IDS),
    }
    _status(target, "vlrb-phase10-run", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _system_metrics(records: Sequence[Mapping[str, Any]], votes_by_replicate):
    value = legacy._system_metrics(records, votes_by_replicate)
    # Match VL-RewardBench ``cal.py::distribution_dataset``: unmatched/other
    # decisions are excluded from OverallAcc, and MacroAcc is the unweighted
    # mean of General, Hallucination, and Reasoning covered accuracies.
    value["overall_acc"] = value["covered_accuracy"]
    value["macro_acc"] = sum(
        item["covered_accuracy"] for item in value["groups"].values()) / 3
    value["official_metric_semantics"] = (
        "VL-RewardBench agree/reject accuracy; tie/abstain excluded")
    decisions = value["original_index_predictions"]
    source_groups = {}
    for source in ("povid", "reasoning_tasks", "rlaif-v", "rlhf-v",
                   "vlfeedback", "wildvision-battle"):
        indices = [index for index, record in enumerate(records)
                   if legacy._official_dataset(str(record["benchmark_id"])) == source]
        correct = sum(decisions[index] == int(records[index]["preferred_original_index"])
                      for index in indices)
        covered = sum(decisions[index] is not None for index in indices)
        source_groups[source] = {
            "sample_count": len(indices),
            "correct_count": correct,
            "strict_accuracy": correct / len(indices) if indices else 0.0,
            "coverage": covered / len(indices) if indices else 0.0,
            "covered_accuracy": correct / covered if covered else 0.0,
        }
    nonempty = [item for item in source_groups.values() if item["sample_count"]]
    value["source_groups"] = source_groups
    value["source_macro_strict_accuracy"] = (
        sum(item["strict_accuracy"] for item in nonempty) / len(nonempty))
    return value


def _provenance_summary(target: Path, endpoint_schedule: Mapping[str, Any]) -> dict[str, Any]:
    paths = list((target / "run/native").glob("provenance_replicate_*.json"))
    paths.extend((target / "run/structured" / FINAL_SYSTEM).glob("replicate_*/provenance.json"))
    calls = []
    for path in sorted(paths):
        value = load_json(path)
        calls.extend(value.get("calls", []))
    by_endpoint = {
        endpoint_id: sum(call.get("endpoint_id") == endpoint_id for call in calls)
        for endpoint_id in ENDPOINT_IDS
    }
    return {
        "scheduled_unique_requests": endpoint_schedule["route_count"],
        "scheduled_endpoint_counts": endpoint_schedule["endpoint_counts"],
        "recorded_model_calls": len(calls),
        "recorded_calls_by_endpoint": by_endpoint,
        "api_attempts": sum(int(call.get("api_attempts", 0)) for call in calls),
        "errors": sum(int(call.get("error_count", 0)) for call in calls),
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
        "latency_seconds_sum": sum(float(call.get("latency_seconds") or 0.0) for call in calls),
        "note": "recorded_model_calls excludes cache hits from interrupted/resumed processes",
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, order_schedule, endpoint_schedule, final = _load_frozen(
        config, output)
    del order_schedule, final
    _require(target, "vlrb-phase10-run")
    combined = load_json(target / "run/combined/logical_votes.json")
    if combined.get("sample_ids") != [str(record["sample_id"]) for record in records]:
        raise RuntimeError("Phase10 report sample order drift")
    metrics = {
        name: _system_metrics(records, item["votes_by_replicate"])
        for name, item in combined["systems"].items()
    }
    roots = {
        root_id: _system_metrics(records, votes)
        for root_id, votes in combined["root_votes_by_replicate"].items()
    }
    predictions = {name: value["original_index_predictions"]
                   for name, value in metrics.items()}
    paired = {
        "initial_to_phase10_equal": legacy._paired(
            records, predictions[INITIAL_SYSTEM], predictions[EQUAL_SYSTEM]),
        "phase10_equal_to_weighted": legacy._paired(
            records, predictions[EQUAL_SYSTEM], predictions[WEIGHTED_SYSTEM]),
        "native_to_phase10_weighted": legacy._paired(
            records, predictions[NATIVE_SYSTEM], predictions[WEIGHTED_SYSTEM]),
    }
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "overlap_audit": "skipped_by_protocol",
        "frozen_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
        "selection_after_benchmark_forbidden": True,
        "phase10_rubric_sha256": SOURCE_RUBRIC_SHA256,
        "root_weights": dict(ROOT_WEIGHTS),
        "metrics": metrics,
        "root_metrics": roots,
        "paired": paired,
        "prediction_reuse": combined["prediction_reuse"],
        "execution_provenance": _provenance_summary(target, endpoint_schedule),
        "primary_system": WEIGHTED_SYSTEM,
    }
    atomic_write_json(target / "report.json", value)
    lines = [
        "# VL-RewardBench Phase10 External Transfer Report", "",
        "Exploratory K=3 counterbalanced evaluation; overlap audit skipped by protocol.", "",
        "| System | Strict ACC | Source Macro ACC | Coverage |", "|---|---:|---:|---:|",
    ]
    for name in (NATIVE_SYSTEM, INITIAL_SYSTEM, EQUAL_SYSTEM, WEIGHTED_SYSTEM, VISUAL_SYSTEM):
        item = metrics[name]
        lines.append(
            f"| {name} | {item['strict_accuracy']:.4f} | "
            f"{item['source_macro_strict_accuracy']:.4f} | {item['coverage']:.4f} |")
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _status(target, "vlrb-phase10-report", {"report": str(target / "report.json")})
    print(json.dumps({name: metrics[name]["strict_accuracy"] for name in (
        NATIVE_SYSTEM, INITIAL_SYSTEM, EQUAL_SYSTEM, WEIGHTED_SYSTEM, VISUAL_SYSTEM)},
        indent=2, ensure_ascii=False))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {
        "vlrb-phase10-freeze": freeze,
        "vlrb-phase10-audit": audit,
        "vlrb-phase10-smoke": smoke,
        "vlrb-phase10-run": run,
        "vlrb-phase10-report": report,
    }
    try:
        actions[stage](config, output)
    except KeyError as exc:
        raise ValueError(f"unsupported Phase10 VL-RewardBench stage: {stage}") from exc

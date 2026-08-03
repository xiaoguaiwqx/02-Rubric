"""双服务器 Shared-Output 离线实验入口。

运行顺序（PowerShell）：

    $module = "experiments.evolving_structured_rubrics.run_shared_output_pool"
    # Copy shared_output_pool.example.json to configs/local and fill local paths first.
    $config = "experiments/evolving_structured_rubrics/configs/local/shared_output_pool.json"
    $output = "output/evolving_structured_rubrics/shared_output_pool_v1"

    python -m $module --config $config --output-dir $output freeze
    python -m $module --config $config --output-dir $output endpoint-check
    python -m $module --config $config --output-dir $output pairwise-p05
    python -m $module --config $config --output-dir $output pairwise-p00
    python -m $module --config $config --output-dir $output gate
    python -m $module --config $config --output-dir $output router
    python -m $module --config $config --output-dir $output reserve-p05
    python -m $module --config $config --output-dir $output offline
    python -m $module --config $config --output-dir $output report

模型生成阶段只保存输出完整性和调用 provenance。只有 ``offline`` 会读取 gold
answer 并计算 B1/H1/G1/M1/M2 accuracy。
"""

# 单节点修复示例（仅刷新一个 P00 sample×criterion，其余节点读取缓存）：
# python -m experiments.evolving_structured_rubrics.run_shared_output_pool \
#   --config experiments/evolving_structured_rubrics/configs/local/shared_output_pool.json \
#   --output-dir output/evolving_structured_rubrics/shared_output_pool_v1 \
#   --pairwise-tag p00 --sample-id rlhfv-003841 \
#   --criterion temporal_coherence retry-pairwise-node

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.dual_evaluator import GateStateMultiModalEvaluator, PairwiseVoteMultiModalEvaluator
from critiq.dual_worker_prompts import GATE_STATE_WORKER_PROMPT_V2
from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    CacheMode,
    DualCascadeExecutor,
    ExecutionConfig,
    GatePredictionOutput,
    JsonPredictionCache,
    OfflineGateStateBackend,
    OfflinePairwiseVoteBackend,
    OnlineGateStateBackend,
    OnlinePairwiseVoteBackend,
    PairwisePredictionOutput,
    RootRoutingPredictionOutput,
    StructuredRootRouter,
    StructuredSystemVariant,
)
from critiq.structured.aggregation import aggregate_flat_votes
from critiq.structured.backend import OfflineRouterBackend, OnlineRouterBackend
from critiq.structured.cache import pairwise_cache_key_payload
from critiq.structured.root_router import root_routing_consistency_errors, root_snapshots
from critiq.structured.semantics import EdgeCondition, resolve_root_routing
from critiq.structured.telemetry import ModelCallMetrics, combine_model_call_metrics
from critiq.structured.worker_output import StructuredCriterionSnapshot

from .rubric_factory import build_static_rubrics, file_sha256
from .experiment_utils import (
    atomic_write_json as _write,
    canonical_sha256 as _canonical_sha256,
    load_json as _load,
    load_jsonl_dataset as _load_jsonl_dataset,
    make_progress_callback as _progress,
    request_json as _request_json,
    validate_manifest_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/evolving_structured_rubrics/configs/local/shared_output_pool.json"
DEFAULT_OUTPUT = ROOT / "output/evolving_structured_rubrics/shared_output_pool_v1"
VARIANTS = tuple(StructuredSystemVariant)
REPRESENTATIVE_CRITERIA = ("visual_grounding", "logical_consistency", "multimodal_alignment")


def _config(path: Path) -> dict[str, Any]:
    value = _load(path)
    required = {"experiment_id", "model", "vllm_version", "max_model_len",
                "criteria_source", "criteria_source_sha256", "heldout_dataset",
                "heldout_dataset_sha256", "discovery_dataset", "reserve_pool",
                "reserve_pool_sha256", "image_root", "backend_pool",
                "worker_request_kwargs", "router_request_kwargs",
                "structured_max_retries", "api_retry_attempts", "reserve_seed"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"shared-output config fields mismatch: {sorted(set(value) ^ required)}")
    BackendPoolSpec.from_dict(value["backend_pool"])
    if value["worker_request_kwargs"].get("temperature") is not None:
        raise ValueError("worker_request_kwargs must not freeze temperature")
    if value["router_request_kwargs"].get("temperature") != 0:
        raise ValueError("Root Router must use temperature=0")
    return value


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _inspect_endpoint(config: Mapping[str, Any], endpoint: Mapping[str, Any]) -> dict[str, Any]:
    base = str(endpoint["base_url"]).rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("endpoint base_url must end with /v1")
    version = _request_json(base[:-3] + "/version")
    models = _request_json(base + "/models")
    matches = [item for item in models.get("data", [])
               if isinstance(item, dict) and item.get("id") == config["model"]]
    if len(matches) != 1:
        raise ValueError(f"{endpoint['endpoint_id']} does not expose the configured model exactly once")
    model = matches[0]
    actual = {"endpoint_id": endpoint["endpoint_id"], "base_url": base,
              "vllm_version": version.get("version"), "model": model.get("id"),
              "checkpoint_root": model.get("root"),
              "max_model_len": model.get("max_model_len")}
    expected = {"endpoint_id": endpoint["endpoint_id"], "base_url": base,
                "vllm_version": config["vllm_version"], "model": config["model"],
                "checkpoint_root": endpoint["checkpoint_root"],
                "max_model_len": config["max_model_len"]}
    if actual != expected:
        raise ValueError(f"endpoint identity mismatch: expected={expected}, actual={actual}")
    return actual


def _pool(config: Mapping[str, Any]) -> AvailableSlotBackendPool:
    return AvailableSlotBackendPool(BackendPoolSpec.from_dict(config["backend_pool"]))


def _agent_args(config: Mapping[str, Any], temperature: float, base_url: str | None = None,
                *, router: bool = False) -> dict[str, Any]:
    request = dict(config["router_request_kwargs"] if router else config["worker_request_kwargs"])
    request["temperature"] = temperature
    value = {"model": config["model"], "api_keys": "EMPTY", "request_kwargs": request,
             "api_retry_attempts": config["api_retry_attempts"]}
    if base_url is not None:
        value["base_url"] = base_url
    return value


def _pair_eval(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
               backend_id: str, temperature: float, pool: AvailableSlotBackendPool | None = None):
    return PairwiseVoteMultiModalEvaluator(
        worker_args=_agent_args(config, temperature), dataset=rows, backend_id=backend_id,
        max_concurrent=(pool.spec.global_request_concurrency if pool else 1),
        max_retries=config["structured_max_retries"], max_data_chars=None,
        encode_local_image=True, call_backend=pool)


def _gate_eval(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
               backend_id: str, pool: AvailableSlotBackendPool | None = None,
               base_url: str | None = None):
    return GateStateMultiModalEvaluator(
        worker_args=_agent_args(config, 0.0, base_url), dataset=rows, backend_id=backend_id,
        worker_prompt=GATE_STATE_WORKER_PROMPT_V2,
        max_concurrent=(pool.spec.global_request_concurrency if pool else 1),
        max_retries=config["structured_max_retries"], max_data_chars=None,
        encode_local_image=True, call_backend=pool)


def _router(config: Mapping[str, Any], backend_id: str,
            pool: AvailableSlotBackendPool | None = None,
            base_url: str | None = None) -> StructuredRootRouter:
    return StructuredRootRouter(
        router_args=_agent_args(config, 0.0, base_url, router=True),
        router_backend_id=backend_id, max_retries=config["structured_max_retries"],
        max_data_chars=None, encode_local_image=True, call_backend=pool)


def _reserve50(config: Mapping[str, Any], heldout: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    source = _path(config["reserve_pool"])
    if file_sha256(source).lower() != config["reserve_pool_sha256"].lower():
        raise ValueError("reserve pool SHA-256 mismatch")
    image_root = _path(config["image_root"])
    heldout_ids = {row["sample_id"] for row in heldout}
    heldout_images = {str(Path(row["image_path"]).resolve()).lower() for row in heldout}
    groups: dict[str, list[dict[str, Any]]] = {"question_answering": [], "detailed_description": []}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = (row.get("origin_split") or {}).get("type")
        if task not in groups or row.get("sample_id") in heldout_ids:
            continue
        image_path = (image_root / row["image_path"]).resolve()
        if str(image_path).lower() in heldout_images or not image_path.is_file():
            continue
        groups[task].append({**row, "image_path": str(image_path).replace("\\", "/")})
    rng = random.Random(config["reserve_seed"])
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for task in ("question_answering", "detailed_description"):
        candidates = list(groups[task]); rng.shuffle(candidates)
        for row in candidates:
            image = row["image_path"].lower()
            if image in used_images:
                continue
            selected.append(row); used_images.add(image)
            if sum((item.get("origin_split") or {}).get("type") == task for item in selected) == 25:
                break
    if len(selected) != 50:
        raise ValueError("reserve pool cannot supply the frozen balanced unique-image reserve-50")
    rng.shuffle(selected)
    converted = []
    for index, row in enumerate(selected):
        chosen_first = index < 25
        converted.append({"sample_id": row["sample_id"], "image_path": row["image_path"],
                          "question": row["question"],
                          "A": row["chosen"] if chosen_first else row["rejected"],
                          "B": row["rejected"] if chosen_first else row["chosen"],
                          "answer": "A" if chosen_first else "B"})
    return tuple(converted)


def _inputs(config: Mapping[str, Any]):
    heldout = _load_jsonl_dataset(_path(config["heldout_dataset"]), expected_count=500)
    reserve = _reserve50(config, heldout)
    return heldout, reserve, tuple(heldout) + reserve


def _model_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Remove gold labels before any online model request is constructed."""
    return tuple({key: value for key, value in row.items() if key != "answer"} for row in rows)


def _status_path(output: Path) -> Path:
    return output / "stage_status.json"


def _set_status(output: Path, stage: str, status: str, details: object = None) -> None:
    value = _load(_status_path(output)) if _status_path(output).exists() else {}
    value[stage] = {"status": status, "details": details}
    _write(_status_path(output), value)


def _require(output: Path, stage: str, *, allow_telemetry_gap: bool = False) -> None:
    value = _load(_status_path(output)) if _status_path(output).exists() else {}
    allowed = {"passed"}
    if allow_telemetry_gap:
        allowed.add("passed_with_telemetry_gap")
    if value.get(stage, {}).get("status") not in allowed:
        raise RuntimeError(f"stage {stage!r} must pass first")


def _generation_stage_status(final_valid_rate: float, telemetry_complete: bool) -> str:
    """Separate artifact usability from generation-cost completeness."""
    if final_valid_rate < .95:
        return "failed"
    return "passed" if telemetry_complete else "passed_with_telemetry_gap"


def _generation_summary(pool: AvailableSlotBackendPool, *, valid: int, total: int) -> dict[str, Any]:
    records = pool.records
    usage = all(item.usage_complete for item in records)
    return {"logical_outputs": total, "final_valid": valid,
            "final_valid_rate": valid / total if total else 0.0,
            "pool_backend_id": pool.backend_id,
            "endpoint_call_counts": dict(pool.records_by_endpoint()),
            "api_attempts": sum(item.api_attempts for item in records),
            "input_tokens": sum(item.input_tokens or 0 for item in records) if usage else None,
            "output_tokens": sum(item.output_tokens or 0 for item in records) if usage else None,
            "total_tokens": sum(item.total_tokens or 0 for item in records) if usage else None,
            "usage_complete": usage, "error_count": sum(item.error_count for item in records)}


def freeze(config: Mapping[str, Any], output: Path) -> None:
    if file_sha256(_path(config["criteria_source"])).lower() != config["criteria_source_sha256"].lower():
        raise ValueError("criteria source SHA-256 mismatch")
    if file_sha256(_path(config["heldout_dataset"])).lower() != config["heldout_dataset_sha256"].lower():
        raise ValueError("heldout source SHA-256 mismatch")
    pool_spec = BackendPoolSpec.from_dict(config["backend_pool"])
    identities = [_inspect_endpoint(config, item.to_dict()) for item in pool_spec.endpoints]
    heldout, reserve, eval550 = _inputs(config)
    strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    fingerprint_eval = _pair_eval(config, _model_rows(eval550), pool_spec.backend_id, .5)
    p05_spec = _pair_eval(config, _model_rows(eval550[:1]), pool_spec.backend_id, .5).request_spec().to_dict()
    p00_spec = _pair_eval(config, _model_rows(eval550[:1]), pool_spec.backend_id, 0).request_spec().to_dict()
    p05_without_temperature = json.loads(json.dumps(p05_spec))
    p00_without_temperature = json.loads(json.dumps(p00_spec))
    p05_without_temperature["decoding_config"].pop("temperature")
    p00_without_temperature["decoding_config"].pop("temperature")
    if p05_without_temperature != p00_without_temperature:
        raise RuntimeError("P05/P00 request specs differ by more than temperature")
    manifest = {"manifest_version": "shared-output-pool-v1",
                "config_sha256": _canonical_sha256(config),
                "pool_spec": pool_spec.to_dict(), "pool_backend_id": pool_spec.backend_id,
                "endpoint_identities": identities, "rubric_sha256": strict.rubric_sha256,
                "heldout_sample_ids": [row["sample_id"] for row in heldout],
                "reserve50_rows": list(reserve),
                "reserve50_fingerprints": [fingerprint_eval.sample_fingerprint(row) for row in reserve],
                "eval550_fingerprints": [fingerprint_eval.sample_fingerprint(row) for row in eval550],
                "request_specs": {
                    "pairwise_p05": p05_spec,
                    "pairwise_p00": p00_spec,
                    "gate_v2_t0": _gate_eval(config, _model_rows(eval550[:1]), pool_spec.backend_id).request_spec().to_dict(),
                    "router_t0": _router(config, pool_spec.backend_id).request_spec(strict).to_dict()}}
    path = output / "frozen_manifest.json"
    if path.exists() and _load(path) != manifest:
        raise RuntimeError("existing manifest differs from current configuration/runtime")
    _write(path, manifest); _set_status(output, "freeze", "passed")
    print(f"Frozen manifest: {path}; heldout=500 reserve=50 eval=550")


def _validate_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _require(output, "freeze")
    manifest = _load(output / "frozen_manifest.json")
    validate_manifest_config(manifest, config)
    for item in BackendPoolSpec.from_dict(config["backend_pool"]).endpoints:
        _inspect_endpoint(config, item.to_dict())
    return manifest


def endpoint_check(config: Mapping[str, Any], output: Path) -> None:
    manifest = _validate_manifest(config, output)
    discovery = _load_jsonl_dataset(_path(config["discovery_dataset"]), expected_count=90)
    rows = _model_rows(discovery[:10])
    strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    by_name = {item.name: item for item in strict.criteria_in_execution_order()}
    criteria = tuple(by_name[name] for name in REPRESENTATIVE_CRITERIA)
    endpoint_results = []
    progress = _progress(output, "endpoint-check", 100)
    progress_index = 0
    for endpoint in BackendPoolSpec.from_dict(config["backend_pool"]).endpoints:
        pair_eval = PairwiseVoteMultiModalEvaluator(
            worker_args=_agent_args(config, 0, endpoint.base_url), dataset=rows,
            backend_id=endpoint.endpoint_id, max_concurrent=endpoint.max_concurrency,
            max_retries=config["structured_max_retries"])
        gate_eval = _gate_eval(config, rows, endpoint.endpoint_id,
                               base_url=endpoint.base_url)
        router_eval = _router(config, endpoint.endpoint_id, base_url=endpoint.base_url)
        pair_outputs: list[Any] = [None] * 30; pair_metrics: list[Any] = [None] * 30
        gate_outputs: list[Any] = [None] * 10; gate_metrics: list[Any] = [None] * 10
        router_outputs: list[Any] = [None] * 10; router_metrics: list[Any] = [None] * 10
        with ThreadPoolExecutor(max_workers=endpoint.max_concurrency) as executor:
            futures = {}
            for row_index, row in enumerate(rows):
                for criterion_index, criterion in enumerate(criteria):
                    future = executor.submit(pair_eval.infer_one, row, criterion)
                    futures[future] = ("pair", row_index * len(criteria) + criterion_index)
                futures[executor.submit(gate_eval.infer_one, row, by_name["multimodal_alignment"])] = ("gate", row_index)
                futures[executor.submit(router_eval.route_one_uncached, row, strict)] = ("router", row_index)
            for future in as_completed(futures):
                kind, index = futures[future]; model_output, metrics = future.result()
                if kind == "pair": pair_outputs[index], pair_metrics[index] = model_output, metrics
                elif kind == "gate": gate_outputs[index], gate_metrics[index] = model_output, metrics
                else: router_outputs[index], router_metrics[index] = model_output, metrics
                progress(progress_index, f"{endpoint.endpoint_id}::{kind}::{index}", metrics)
                progress_index += 1
        endpoint_results.append({"endpoint": endpoint, "pair_outputs": pair_outputs,
            "pair_metrics": pair_metrics, "gate_outputs": gate_outputs,
            "gate_metrics": gate_metrics, "router_outputs": router_outputs,
            "router_metrics": router_metrics,
            "pair_spec": pair_eval.request_spec(), "gate_spec": gate_eval.request_spec(),
            "router_spec": router_eval.request_spec(strict)})
    left, right = endpoint_results
    pair_checks = [a.vote is b.vote for a, b in zip(left["pair_outputs"], right["pair_outputs"])]
    gate_checks = []
    for a, b in zip(left["gate_outputs"], right["gate_outputs"]):
        activates = lambda item: (item.valid and item.judgement.applicable.value == "yes"
                                  and item.judgement.status_a.value == "pass"
                                  and item.judgement.status_b.value == "pass")
        gate_checks.append(activates(a) == activates(b))
    router_checks = []
    for a, b in zip(left["router_outputs"], right["router_outputs"]):
        ar = resolve_root_routing(a.decision, strict.root_ids, enabled=True)
        br = resolve_root_routing(b.decision, strict.root_ids, enabled=True)
        router_checks.append(ar.selected_root_ids == br.selected_root_ids)
    def rates(item):
        pair_valid = sum(v.parse_ok and v.answer_valid for v in item["pair_outputs"]) / 30
        gate_valid = sum(v.valid for v in item["gate_outputs"]) / 10
        router_valid = sum(not root_routing_consistency_errors(v.decision, strict.root_ids)
                           and v.parse_error is None for v in item["router_outputs"]) / 10
        all_metrics = item["pair_metrics"] + item["gate_metrics"] + item["router_metrics"]
        return {"endpoint_id": item["endpoint"].endpoint_id, "pairwise_final_valid_rate": pair_valid,
                "gate_final_valid_rate": gate_valid, "router_final_valid_rate": router_valid,
                "api_success_rate": sum(v.api_attempts >= 1 for v in all_metrics) / len(all_metrics),
                "usage_complete": all(v.usage_complete for v in all_metrics),
                "error_count": sum(v.error_count for v in all_metrics)}
    endpoint_rates = [rates(item) for item in endpoint_results]
    report = {"sample_count": 10, "gold_answers_accessed": False,
              "pairwise_semantic_agreement": sum(pair_checks) / len(pair_checks),
              "gate_edge_activation_agreement": sum(gate_checks) / len(gate_checks),
              "router_selected_root_agreement": sum(router_checks) / len(router_checks),
              "endpoint_rates": endpoint_rates,
              "identical_request_specs": {
                  "pairwise": ((left["pair_spec"].to_dict() | {"backend_id": manifest["pool_backend_id"]})
                               == (right["pair_spec"].to_dict() | {"backend_id": manifest["pool_backend_id"]})),
                  "gate": ((left["gate_spec"].to_dict() | {"backend_id": manifest["pool_backend_id"]})
                           == (right["gate_spec"].to_dict() | {"backend_id": manifest["pool_backend_id"]})),
                  "router": ((left["router_spec"].to_dict() | {"router_backend_id": manifest["pool_backend_id"]})
                             == (right["router_spec"].to_dict() | {"router_backend_id": manifest["pool_backend_id"]}))}}
    passed = (all(item["pairwise_final_valid_rate"] >= .95
                  and item["gate_final_valid_rate"] >= .95
                  and item["router_final_valid_rate"] >= .95
                  and item["api_success_rate"] == 1.0
                  and item["usage_complete"] for item in endpoint_rates)
              and report["pairwise_semantic_agreement"] >= .90
              and report["gate_edge_activation_agreement"] >= .80
              and report["router_selected_root_agreement"] >= .80
              and all(report["identical_request_specs"].values()))
    report["passed"] = passed
    _write(output / "reports/endpoint_equivalence.json", report)
    _set_status(output, "endpoint_check", "passed" if passed else "failed", report)
    print(json.dumps(report, indent=2));
    if not passed: raise RuntimeError("endpoint equivalence check failed")


def _save_provenance(output: Path, stage: str, pool: AvailableSlotBackendPool,
                     summary: Mapping[str, Any]) -> None:
    _write(output / f"provenance/{stage}.json", pool.provenance_dict())
    _write(output / f"reports/{stage}_generation.json", dict(summary))


def _cache(output: Path, stage: str) -> JsonPredictionCache:
    """Return the isolated, resumable cache namespace for one stage."""
    return JsonPredictionCache(output / f"cache/{stage}", CacheMode.READ_WRITE)


def _pairwise_pred_cached(evaluator, rows, rubric, cache, on_sample_complete):
    backend = OnlinePairwiseVoteBackend(evaluator, cache)
    nodes = tuple(rubric.get_node(node_id) for node_id in rubric.preorder_node_ids())
    outputs: list[dict[str, Any]] = [dict() for _ in rows]
    metrics_by_sample: list[list[ModelCallMetrics]] = [[] for _ in rows]
    remaining = [len(nodes)] * len(rows); cache_hits = 0
    with ThreadPoolExecutor(max_workers=evaluator.max_concurrent) as executor:
        futures = {executor.submit(backend.evaluate, row, node): (sample_index, node)
                   for sample_index, row in enumerate(rows) for node in nodes}
        for future in as_completed(futures):
            sample_index, node = futures[future]; result = future.result()
            outputs[sample_index][node.criterion.name] = result.output
            metrics_by_sample[sample_index].append(result.metrics)
            cache_hits += result.metrics.cache_hits
            remaining[sample_index] -= 1
            if remaining[sample_index] == 0:
                on_sample_complete(sample_index, str(rows[sample_index][evaluator.sample_id_field]),
                    combine_model_call_metrics(metrics_by_sample[sample_index]))
    answers = tuple(aggregate_flat_votes(item.vote for item in row.values()) for row in outputs)
    return PairwisePredictionOutput(
        tuple(str(row[evaluator.sample_id_field]) for row in rows),
        tuple(evaluator.sample_fingerprint(row) for row in rows),
        tuple(StructuredCriterionSnapshot(node.criterion.name, node.criterion.description)
              for node in nodes), tuple(outputs), answers, evaluator.request_spec()), cache_hits


def _gate_pred_cached(evaluator, rows, nodes, cache, on_sample_complete):
    backend = OnlineGateStateBackend(evaluator, cache)
    outputs: list[dict[str, Any]] = [dict() for _ in rows]
    metrics_by_sample: list[list[ModelCallMetrics]] = [[] for _ in rows]
    remaining = [len(nodes)] * len(rows); cache_hits = 0
    with ThreadPoolExecutor(max_workers=evaluator.max_concurrent) as executor:
        futures = {executor.submit(backend.evaluate, row, node): (sample_index, node)
                   for sample_index, row in enumerate(rows) for node in nodes}
        for future in as_completed(futures):
            sample_index, node = futures[future]; result = future.result()
            outputs[sample_index][node.criterion.name] = result.output
            metrics_by_sample[sample_index].append(result.metrics)
            cache_hits += result.metrics.cache_hits
            remaining[sample_index] -= 1
            if remaining[sample_index] == 0:
                on_sample_complete(sample_index, str(rows[sample_index][evaluator.sample_id_field]),
                    combine_model_call_metrics(metrics_by_sample[sample_index]))
    return GatePredictionOutput(
        tuple(str(row[evaluator.sample_id_field]) for row in rows),
        tuple(evaluator.sample_fingerprint(row) for row in rows),
        tuple(StructuredCriterionSnapshot(node.criterion.name, node.criterion.description)
              for node in nodes), tuple(outputs), evaluator.request_spec(),
        prompt_version=evaluator.prompt_version), cache_hits


def pairwise_generate(config: Mapping[str, Any], output: Path, temperature: float) -> None:
    _require(output, "endpoint_check"); _validate_manifest(config, output)
    heldout, _, _ = _inputs(config); strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    pool = _pool(config); tag = "p05" if temperature == .5 else "p00"
    rows = _model_rows(heldout)
    evaluator = _pair_eval(config, rows, pool.backend_id, temperature, pool)
    cache = _cache(output, f"pairwise_{tag}")
    prediction, cache_hits = _pairwise_pred_cached(
        evaluator, rows, strict, cache,
        _progress(output, f"pairwise-{tag}", len(heldout), pool))
    artifact = output / f"predictions/pairwise_{tag}_heldout500.json"
    artifact.parent.mkdir(parents=True, exist_ok=True); prediction.save_json(artifact)
    total = len(heldout) * len(strict.nodes)
    valid = sum(item.parse_ok and item.answer_valid for row in prediction.node_outputs for item in row.values())
    cumulative = cache.cumulative_generation_metrics(("pairwise",))
    summary = _generation_summary(pool, valid=valid, total=total) | {
        "artifact": str(artifact), "temperature": temperature, "accuracy_computed": False,
        "cache_hits_current_run": cache_hits,
        "cached_generation_entries": len(cache.generation_provenance("pairwise")),
        "cumulative_api_attempts": cumulative.api_attempts,
        "cumulative_usage_complete": cumulative.usage_complete,
        "artifact_ready_for_offline": valid / total >= .95,
        "telemetry_status": ("complete" if cumulative.usage_complete else "incomplete")}
    _save_provenance(output, f"pairwise_{tag}", pool, summary)
    stage = f"pairwise_{tag}"
    status = _generation_stage_status(
        summary["final_valid_rate"], summary["cumulative_usage_complete"])
    _set_status(output, stage, status, summary)
    print(f"{stage}: {valid}/{total} valid; endpoint calls={summary['endpoint_call_counts']}")
    if status == "passed_with_telemetry_gap":
        print(f"WARNING: {stage} artifact is usable, but generation telemetry is incomplete")
    if status == "failed": raise RuntimeError(f"{stage} generation gate failed")


def _replace_pairwise_node_output(
    prediction: PairwisePredictionOutput,
    sample_id: str,
    criterion_name: str,
    replacement: Any,
) -> PairwisePredictionOutput:
    try:
        sample_index = prediction.sample_ids.index(sample_id)
    except ValueError as exc:
        raise KeyError(f"sample {sample_id!r} is not present in Pairwise artifact") from exc
    if criterion_name not in prediction.node_outputs[sample_index]:
        raise KeyError(f"criterion {criterion_name!r} is not present in Pairwise artifact")
    rows = [dict(row) for row in prediction.node_outputs]
    rows[sample_index][criterion_name] = replacement
    answers = tuple(
        aggregate_flat_votes(item.vote for item in row.values()) for row in rows
    )
    return PairwisePredictionOutput(
        prediction.sample_ids,
        prediction.sample_fingerprints,
        prediction.criteria,
        tuple(rows),
        answers,
        prediction.request_spec,
        prediction.semantics_version,
        prediction.schema_version,
        prediction.prompt_version,
        prediction.parser_version,
    )


def retry_pairwise_node(
    config: Mapping[str, Any],
    output: Path,
    *,
    pairwise_tag: str,
    sample_id: str,
    criterion_name: str,
) -> None:
    """Regenerate one cached Pairwise node and patch its frozen artifact."""
    _require(output, "endpoint_check")
    _validate_manifest(config, output)
    if pairwise_tag not in {"p05", "p00"}:
        raise ValueError("pairwise_tag must be p05 or p00")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be non-empty")
    if not isinstance(criterion_name, str) or not criterion_name.strip():
        raise ValueError("criterion_name must be non-empty")

    artifact = output / f"predictions/pairwise_{pairwise_tag}_heldout500.json"
    prediction = PairwisePredictionOutput.load_json(artifact)
    heldout, _, _ = _inputs(config)
    rows = _model_rows(heldout)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    if sample_id not in row_by_id:
        raise KeyError(f"sample {sample_id!r} is not present in heldout-500")
    strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    nodes_by_name = {node.criterion.name: node for node in strict.nodes.values()}
    if criterion_name not in nodes_by_name:
        raise KeyError(f"criterion {criterion_name!r} is not present in rubric")
    node = nodes_by_name[criterion_name]
    sample_index = prediction.sample_ids.index(sample_id)
    current = prediction.node_outputs[sample_index][criterion_name]
    if current.parse_ok and current.answer_valid:
        raise RuntimeError("refusing to replace an already-valid Pairwise node")

    pool = _pool(config)
    temperature = .5 if pairwise_tag == "p05" else 0.0
    evaluator = _pair_eval(config, (row_by_id[sample_id],), pool.backend_id, temperature, pool)
    if evaluator.request_spec() != prediction.request_spec:
        raise RuntimeError("repair request spec does not match the frozen Pairwise artifact")
    fingerprint = evaluator.sample_fingerprint(row_by_id[sample_id])
    if fingerprint != prediction.sample_fingerprints[sample_index]:
        raise RuntimeError("repair sample fingerprint does not match the frozen Pairwise artifact")

    cache = _cache(output, f"pairwise_{pairwise_tag}")
    payload = pairwise_cache_key_payload(
        sample_fingerprint=fingerprint,
        criterion_name=node.criterion.name,
        criterion_description=node.criterion.description,
        request_spec=evaluator.request_spec(),
    )
    cached = cache.get_pairwise(payload)
    quarantine_path: Path | None = None
    if cached is not None and cached.output == current:
        quarantine_path = cache.quarantine("pairwise", payload)

    result = OnlinePairwiseVoteBackend(evaluator, cache).evaluate(
        row_by_id[sample_id], node
    )
    repair_id = f"pairwise_{pairwise_tag}_{sample_id}_{criterion_name}_{time.time_ns()}"
    repair_summary = {
        "pairwise_tag": pairwise_tag,
        "sample_id": sample_id,
        "criterion_name": criterion_name,
        "previous_output": current.to_dict(),
        "replacement_output": result.output.to_dict(),
        "backend_source": result.source.value,
        "quarantined_cache": str(quarantine_path) if quarantine_path else None,
        "generation_metrics": result.generation_metrics.to_dict(),
    }
    _write(output / f"provenance/repairs/{repair_id}.json", {
        "repair": repair_summary,
        "backend_pool": pool.provenance_dict(),
    })
    if not (result.output.parse_ok and result.output.answer_valid):
        _write(output / f"reports/repairs/{repair_id}.json", repair_summary)
        raise RuntimeError("replacement Pairwise node is still invalid; artifact was not modified")

    repaired = _replace_pairwise_node_output(
        prediction, sample_id, criterion_name, result.output
    )
    _write(artifact, repaired.to_dict())
    total = len(repaired.sample_ids) * len(repaired.criteria)
    valid = sum(
        item.parse_ok and item.answer_valid
        for row in repaired.node_outputs for item in row.values()
    )
    cumulative = cache.cumulative_generation_metrics(("pairwise",))
    status = _generation_stage_status(valid / total, cumulative.usage_complete)
    repair_summary.update({
        "artifact": str(artifact),
        "final_valid": valid,
        "logical_outputs": total,
        "final_valid_rate": valid / total,
        "artifact_ready_for_offline": valid / total >= .95,
        "cumulative_api_attempts": cumulative.api_attempts,
        "cumulative_usage_complete": cumulative.usage_complete,
        "telemetry_status": ("complete" if cumulative.usage_complete else "incomplete"),
    })
    _write(output / f"reports/repairs/{repair_id}.json", repair_summary)
    _set_status(output, f"pairwise_{pairwise_tag}", status, repair_summary)
    print(
        f"repaired {sample_id}::{criterion_name}; valid={valid}/{total}; "
        f"api_attempts={result.metrics.api_attempts}; status={status}"
    )
    if status == "passed_with_telemetry_gap":
        print("WARNING: artifact is usable for offline replay, but historical telemetry is incomplete")


def gate_generate(config: Mapping[str, Any], output: Path) -> None:
    _require(output, "endpoint_check"); _validate_manifest(config, output)
    _, _, eval550 = _inputs(config); strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    parent_names = {strict.get_node(edge.parent_id).criterion.name for edge in strict.edges
                    if edge.condition in {EdgeCondition.PARENT_BOTH_PASS, EdgeCondition.PARENT_BOTH_FAIL}}
    nodes_by_name = {node.criterion.name: node for node in strict.nodes.values()}
    nodes = tuple(nodes_by_name[item.name] for item in strict.criteria_in_execution_order()
                  if item.name in parent_names)
    rows = _model_rows(eval550); pool = _pool(config)
    evaluator = _gate_eval(config, rows, pool.backend_id, pool)
    cache = _cache(output, "gate")
    prediction, cache_hits = _gate_pred_cached(
        evaluator, rows, nodes, cache,
        _progress(output, "gate-v2-t0", len(eval550), pool))
    artifact = output / "predictions/gate_v2_t0_eval550.json"
    artifact.parent.mkdir(parents=True, exist_ok=True); prediction.save_json(artifact)
    total = len(eval550) * len(nodes); valid = sum(item.valid for row in prediction.node_outputs for item in row.values())
    cumulative = cache.cumulative_generation_metrics(("gate",))
    summary = _generation_summary(pool, valid=valid, total=total) | {
        "artifact": str(artifact), "temperature": 0, "accuracy_computed": False,
        "cache_hits_current_run": cache_hits,
        "cached_generation_entries": len(cache.generation_provenance("gate")),
        "cumulative_api_attempts": cumulative.api_attempts,
        "cumulative_usage_complete": cumulative.usage_complete,
        "artifact_ready_for_offline": valid / total >= .95,
        "telemetry_status": ("complete" if cumulative.usage_complete else "incomplete")}
    _save_provenance(output, "gate", pool, summary)
    status = _generation_stage_status(
        summary["final_valid_rate"], summary["cumulative_usage_complete"])
    _set_status(output, "gate", status, summary)
    print(f"gate: {valid}/{total} valid; endpoint calls={summary['endpoint_call_counts']}")
    if status == "passed_with_telemetry_gap":
        print("WARNING: Gate artifact is usable, but generation telemetry is incomplete")
    if status == "failed": raise RuntimeError("Gate generation gate failed")


def _router_pred_parallel(router: StructuredRootRouter, rows: Sequence[Mapping[str, Any]], rubric,
                          workers: int, on_complete=None,
                          cache: JsonPredictionCache | None = None):
    outputs: list[Any] = [None] * len(rows); metrics: list[Any] = [None] * len(rows)
    backend = OnlineRouterBackend(router, cache) if cache is not None else None
    def run(index, row):
        if backend is None:
            output, metric = router.route_one_uncached(row, rubric)
            return index, output, metric, metric
        result = backend.route(row, rubric)
        return index, result.output, result.metrics, result.generation_metrics
    current_metrics: list[Any] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, i, row) for i, row in enumerate(rows)]
        for future in as_completed(futures):
            index, outputs[index], current_metrics[index], metrics[index] = future.result()
            if on_complete is not None:
                on_complete(index, str(rows[index][router.sample_id_field]), current_metrics[index])
    prediction = RootRoutingPredictionOutput(
        tuple(row[router.sample_id_field] for row in rows),
        tuple(router.sample_fingerprint(row) for row in rows), root_snapshots(rubric),
        tuple(outputs), tuple(metrics), router.request_spec(rubric), rubric.rubric_sha256)
    return prediction, sum(item.cache_hits for item in current_metrics)


def router_generate(config: Mapping[str, Any], output: Path) -> None:
    _require(output, "endpoint_check"); _validate_manifest(config, output)
    _, _, eval550 = _inputs(config); strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    pool = _pool(config); router = _router(config, pool.backend_id, pool)
    cache = _cache(output, "router")
    prediction, cache_hits = _router_pred_parallel(router, _model_rows(eval550), strict,
        pool.spec.global_request_concurrency,
        on_complete=_progress(output, "router-t0", len(eval550), pool), cache=cache)
    artifact = output / "predictions/root_router_t0_eval550.json"
    artifact.parent.mkdir(parents=True, exist_ok=True); prediction.save_json(artifact)
    valid = sum(item.parse_error is None and not root_routing_consistency_errors(item.decision, strict.root_ids)
                for item in prediction.outputs)
    cumulative = cache.cumulative_generation_metrics(("router",))
    summary = _generation_summary(pool, valid=valid, total=len(eval550)) | {
        "artifact": str(artifact), "temperature": 0, "accuracy_computed": False,
        "cache_hits_current_run": cache_hits,
        "cached_generation_entries": len(cache.generation_provenance("router")),
        "cumulative_api_attempts": cumulative.api_attempts,
        "cumulative_usage_complete": cumulative.usage_complete,
        "artifact_ready_for_offline": valid / len(eval550) >= .95,
        "telemetry_status": ("complete" if cumulative.usage_complete else "incomplete")}
    _save_provenance(output, "router", pool, summary)
    status = _generation_stage_status(
        summary["final_valid_rate"], summary["cumulative_usage_complete"])
    _set_status(output, "router", status, summary)
    print(f"router: {valid}/{len(eval550)} valid; endpoint calls={summary['endpoint_call_counts']}")
    if status == "passed_with_telemetry_gap":
        print("WARNING: Router artifact is usable, but generation telemetry is incomplete")
    if status == "failed": raise RuntimeError("Router generation gate failed")


def reserve_generate(config: Mapping[str, Any], output: Path) -> None:
    _require(output, "endpoint_check"); _validate_manifest(config, output)
    _, reserve, _ = _inputs(config); strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    reports = []
    for replicate in range(3):
        pool = _pool(config)
        prediction = _pair_eval(config, _model_rows(reserve), pool.backend_id, .5, pool).pred(
            strict.criteria_in_execution_order(),
            on_sample_complete=_progress(output, f"reserve-p05-r{replicate}", len(reserve), pool))
        artifact = output / f"predictions/pairwise_p05_reserve50_r{replicate}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True); prediction.save_json(artifact)
        total = len(reserve) * len(strict.nodes)
        valid = sum(item.parse_ok and item.answer_valid for row in prediction.node_outputs for item in row.values())
        summary = _generation_summary(pool, valid=valid, total=total) | {
            "artifact": str(artifact), "temperature": .5, "replicate": replicate,
            "cache_reused": False, "accuracy_computed": False}
        _save_provenance(output, f"reserve_p05_r{replicate}", pool, summary); reports.append(summary)
    artifacts_ready = all(item["final_valid_rate"] >= .95 for item in reports)
    telemetry_complete = all(item["usage_complete"] for item in reports)
    status = _generation_stage_status(
        min(item["final_valid_rate"] for item in reports), telemetry_complete)
    _set_status(output, "reserve_p05", status, reports)
    print(f"reserve-p05: generated three independent artifacts; status={status}")
    if status == "passed_with_telemetry_gap":
        print("WARNING: reserve artifacts are usable, but generation telemetry is incomplete")
    if not artifacts_ready: raise RuntimeError("reserve P05 generation gate failed")


def _system_metrics(answers: Sequence[Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisive = [answer.value in {"A", "B"} for answer in answers]
    correct = [answer.value == row["answer"] for answer, row in zip(answers, rows)]
    covered = sum(decisive)
    return {"accuracy": sum(correct) / len(rows), "coverage": covered / len(rows),
            "covered_accuracy": (sum(ok and dec for ok, dec in zip(correct, decisive)) / covered
                                  if covered else 0.0),
            "tie_rate": sum(not value for value in decisive) / len(rows),
            "predictions": [answer.value for answer in answers]}


def _offline_one(pairwise: PairwisePredictionOutput, gate: GatePredictionOutput,
                 router: RootRoutingPredictionOutput, rows: Sequence[Mapping[str, Any]], rubric,
                 *, label: str = "offline", on_variant=None):
    pair_backend = OfflinePairwiseVoteBackend(pairwise, rubric)
    gate_backend = OfflineGateStateBackend(gate, rubric)
    router_backend = OfflineRouterBackend(router, rubric)
    result = {}
    for variant in VARIANTS:
        executor = DualCascadeExecutor(
            rubric, pair_backend,
            gate_backend if variant in {StructuredSystemVariant.G1_GATING_ONLY,
                                        StructuredSystemVariant.M1_ALL_ROOTS_CASCADE,
                                        StructuredSystemVariant.M2_ROUTED_ROOTS_CASCADE} else None,
            router_backend if variant is StructuredSystemVariant.M2_ROUTED_ROOTS_CASCADE else None)
        execution = executor.execute_batch(rows, ExecutionConfig(variant))
        answers = tuple(trace.final_preference for trace in execution.traces)
        result[variant.value] = _system_metrics(answers, rows)
        if on_variant is not None:
            on_variant(label, variant, result[variant.value])
    if tuple(result[StructuredSystemVariant.B1_FLAT.value]["predictions"]) != tuple(
            answer.value for answer in pairwise.flat_answers):
        raise RuntimeError("offline B1 changed Pairwise artifact flat answers")
    return result


def _agreement(values: Sequence[Sequence[str]]) -> float:
    checks = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            checks.extend(a == b for a, b in zip(values[left], values[right]))
    return sum(checks) / len(checks) if checks else 1.0


def offline(config: Mapping[str, Any], output: Path) -> None:
    for stage in ("pairwise_p05", "pairwise_p00", "gate", "router", "reserve_p05"):
        _require(output, stage, allow_telemetry_gap=True)
    _validate_manifest(config, output)
    heldout, reserve, _ = _inputs(config); strict, _ = build_static_rubrics(_path(config["criteria_source"]))
    gate = GatePredictionOutput.load_json(output / "predictions/gate_v2_t0_eval550.json")
    router = RootRoutingPredictionOutput.load_json(output / "predictions/root_router_t0_eval550.json")
    heldout_results = {}
    log_path = output / "logs/offline.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
    offline_completed = 0
    offline_total = 25
    def variant_progress(label, variant, metrics):
        nonlocal offline_completed
        offline_completed += 1
        line = (f"offline: {offline_completed}/{offline_total} run={label} "
                f"variant={variant.value} accuracy={metrics['accuracy']:.3f} "
                f"coverage={metrics['coverage']:.3f}")
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        try:
            _write(output / "progress.json", {
                "stage": "offline", "completed": offline_completed,
                "total": offline_total, "percent": offline_completed / offline_total * 100,
                "current_run": label, "current_variant": variant.value,
                "accuracy": metrics["accuracy"], "coverage": metrics["coverage"]})
        except OSError as exc:
            warning = f"WARNING: progress.json update skipped: {exc}"
            print(warning, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(warning + "\n")
    for tag in ("p05", "p00"):
        pairwise = PairwisePredictionOutput.load_json(output / f"predictions/pairwise_{tag}_heldout500.json")
        heldout_results[tag] = _offline_one(pairwise, gate, router, heldout, strict,
                                             label=f"heldout-{tag}",
                                             on_variant=variant_progress)
        baseline = heldout_results[tag][StructuredSystemVariant.B1_FLAT.value]["accuracy"]
        for name, result in heldout_results[tag].items():
            result["paired_accuracy_difference_from_b1"] = result["accuracy"] - baseline
    reserve_runs = []
    for replicate in range(3):
        pairwise = PairwisePredictionOutput.load_json(
            output / f"predictions/pairwise_p05_reserve50_r{replicate}.json")
        reserve_runs.append(_offline_one(pairwise, gate, router, reserve, strict,
                                         label=f"reserve-r{replicate}",
                                         on_variant=variant_progress))
    variance = {}
    for variant in VARIANTS:
        name = variant.value; accuracies = [run[name]["accuracy"] for run in reserve_runs]
        variance[name] = {"accuracies": accuracies, "mean": statistics.mean(accuracies),
                          "sample_std": statistics.stdev(accuracies),
                          "min": min(accuracies), "max": max(accuracies),
                          "final_prediction_agreement": _agreement(
                              [run[name]["predictions"] for run in reserve_runs])}
    node_runs = [PairwisePredictionOutput.load_json(
        output / f"predictions/pairwise_p05_reserve50_r{i}.json") for i in range(3)]
    node_sequences = [[item.vote.value for row in run.node_outputs for item in row.values()]
                      for run in node_runs]
    stage_status = _load(_status_path(output))
    report = {"accuracy_source": "offline_shared_output_only",
              "input_stage_status": {
                  stage: stage_status[stage]["status"]
                  for stage in ("pairwise_p05", "pairwise_p00", "gate", "router", "reserve_p05")
              },
              "telemetry_warning": any(
                  stage_status[stage]["status"] == "passed_with_telemetry_gap"
                  for stage in ("pairwise_p05", "pairwise_p00", "gate", "router", "reserve_p05")
              ),
              "heldout500": heldout_results,
              "p05_p00_final_prediction_agreement": {
                  variant.value: _agreement([
                      heldout_results["p05"][variant.value]["predictions"],
                      heldout_results["p00"][variant.value]["predictions"]])
                  for variant in VARIANTS},
              "reserve50_p05_replicates": reserve_runs,
              "reserve50_variance": variance,
              "reserve50_node_vote_agreement": _agreement(node_sequences)}
    _write(output / "reports/offline_shared_output_report.json", report)
    _set_status(output, "offline", "passed", {"report": "reports/offline_shared_output_report.json"})
    print("Offline shared-output report generated; all accuracy values originate here.")


def report(output: Path) -> None:
    _require(output, "offline")
    value = _load(output / "reports/offline_shared_output_report.json")
    for temperature in ("p05", "p00"):
        print(temperature.upper())
        for variant, metrics in value["heldout500"][temperature].items():
            print(f"  {variant}: acc={metrics['accuracy']:.3f} coverage={metrics['coverage']:.3f}")
    print("Reserve P05 random variance")
    for variant, metrics in value["reserve50_variance"].items():
        print(f"  {variant}: mean={metrics['mean']:.3f} std={metrics['sample_std']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairwise-tag", choices=("p05", "p00"))
    parser.add_argument("--sample-id")
    parser.add_argument("--criterion")
    parser.add_argument("stage", choices=("freeze", "endpoint-check", "pairwise-p05",
        "pairwise-p00", "gate", "router", "reserve-p05", "retry-pairwise-node",
        "offline", "report"))
    return parser


def main() -> int:
    args = build_parser().parse_args(); config = _config(args.config.resolve())
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.stage == "retry-pairwise-node":
        missing = [name for name, value in (
            ("--pairwise-tag", args.pairwise_tag),
            ("--sample-id", args.sample_id),
            ("--criterion", args.criterion),
        ) if value is None]
        if missing:
            raise ValueError(
                "retry-pairwise-node requires " + ", ".join(missing)
            )
        retry_pairwise_node(
            config,
            output,
            pairwise_tag=args.pairwise_tag,
            sample_id=args.sample_id,
            criterion_name=args.criterion,
        )
        return 0
    actions = {"freeze": lambda: freeze(config, output),
               "endpoint-check": lambda: endpoint_check(config, output),
               "pairwise-p05": lambda: pairwise_generate(config, output, .5),
               "pairwise-p00": lambda: pairwise_generate(config, output, 0.0),
               "gate": lambda: gate_generate(config, output),
               "router": lambda: router_generate(config, output),
               "reserve-p05": lambda: reserve_generate(config, output),
               "offline": lambda: offline(config, output),
               "report": lambda: report(output)}
    actions[args.stage](); return 0


if __name__ == "__main__":
    raise SystemExit(main())

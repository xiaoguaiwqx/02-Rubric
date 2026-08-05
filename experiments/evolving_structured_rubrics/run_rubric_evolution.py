"""Evolving Structured Rubrics 的 Phase 5 与 Phase 6B 实验入口。

PowerShell 公共变量：

    $module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
    $config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5.json"
    $output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

    $module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
    $config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
    $output = "output/evolving_structured_rubrics/rubric_evolution_phase5_prompt_8001"

Phase 5：构建 Multi-Crit Init Rubric、生成 baseline、提取反馈并冻结阈值：

    python -m $module --config $config --output-dir $output freeze
    python -m $module --config $config --output-dir $output init-baseline
    python -m $module --config $config --output-dir $output feedback
    python -m $module --config $config --output-dir $output phase5-report
    python -m $module --config $config --output-dir $output finalize-phase5

Phase 6B Specialize：冻结目标 parent 并为其 decisive-wrong 样本生成错误签名：

    python -m $module --config $config --output-dir $output specialize-freeze `
      --parent-node-id init_02_visual_grounding_and_details
    python -m $module --config $config --output-dir $output specialize-signatures

Phase 6B Specialize：将错误签名聚类为 2–5 个语义失败子域：

    python -m $module --config $config --output-dir $output specialize-cluster

聚类完成后，先人工检查：

    output/evolving_structured_rubrics/rubric_evolution_phase5/
      phase6_specialize/cluster_proposal.json

确认 cluster 的样本划分与语义合理后，再生成 children、运行新增 child 的
Pairwise 推理，并通过 discovery-90 完整 M1 before/after 决定是否接受：

    python -m $module --config $config --output-dir $output specialize-propose
    python -m $module --config $config --output-dir $output specialize-evaluate
    python -m $module --config $config --output-dir $output specialize-report

``init-baseline`` 在线生成 discovery-90 与 heldout-500 的五节点 Pairwise
artifacts，但 gold answer 不进入模型请求；accuracy 只由随后同一阶段的 offline
B1/M1 replay 计算。Specialize 中间阶段只读取 discovery-90，禁止访问 heldout-500。
三个 Manager 阶段的模型与 backend pool 都由 ``specialize_managers`` 独立配置。
当前 ErrorSignature 使用本地 Qwen3-VL-8B；Semantic Clustering 与 Child
Generation 使用硅基流动的 Qwen3.5-397B-A17B 并开启思考模式。后两个阶段
使用文本输入，Child Generation 接收代表样本的 question/A/B/gold 而不发送图片。
Clustering 使用 prompt v1.0 的一次完整 partition 输出，temperature=0.2。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.dual_evaluator import PairwiseVoteMultiModalEvaluator
from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    CacheMode,
    CandidateAcceptancePolicy,
    DualCascadeExecutor,
    DualWorkerRequestSpec,
    ExecutionConfig,
    FinalPreference,
    JsonPredictionCache,
    ModelCallMetrics,
    OfflinePairwiseVoteBackend,
    OnlinePairwiseVoteBackend,
    PairwisePredictionOutput,
    StructuredCriterionSnapshot,
    StructuredRubric,
    StructuredSystemVariant,
    combine_model_call_metrics,
    ChildCriterionProposal,
    ClusterProposal,
    ErrorSignatureOutput,
    EvolutionContext,
    RubricFeedback,
    SpecializeCandidate,
    SpecializeEvaluation,
    SpecializeManager,
    SpecializeManagerFailure,
    SpecializeTriggerDecision,
    apply_rubric_patch,
    assemble_specialized_pairwise_prediction,
    build_specialize_candidate,
    detect_specialize_trigger,
    evaluate_specialize_candidate,
    extract_rubric_feedback,
    parse_child_proposal_response,
    parse_cluster_proposal_response,
    plan_artifact_refresh,
)
from critiq.structured.aggregation import aggregate_flat_votes

from .experiment_utils import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    load_jsonl_dataset,
    load_local_env,
    make_progress_callback,
    request_json,
)
from .rubric_factory import (
    HELDOUT_DATASET_SHA256,
    build_multicrit_open_ended_init_rubric,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5.json"
DEFAULT_OUTPUT = ROOT / "output/evolving_structured_rubrics/rubric_evolution_phase5"
DISCOVERY_SHA256 = "7c3c5b399ac9826980c8b2d1f451f3b326ae0c575572c390864155d7f3b50ca7"
PHASE5_EVOLUTION_POLICY_V1 = {
    "policy_version": "phase5-v1",
    "trigger_thresholds": {
        "tau_acc": 0.55,
        "tau_split": 0.70,
        "tau_cov_high": 0.80,
        "tau_refine": 0.80,
        "N_min_support": 15,
        "N_min_wrong": 15,
        "N_min_cluster": 5,
        "max_children": 5,
    },
    "candidate_acceptance": {
        "min_accuracy_delta": 2 / 90,
        "min_corrected": 2,
        "max_coverage_drop": 1 / 90,
        "min_valid_rate": 0.95,
        "require_corrected_gt_harmed": True,
        "require_valid_rate_not_decrease": True,
    },
    "p05_confirmation": {
        "total_replicates": 3,
        "additional_replicates": 2,
        "require_primary_pass": True,
        "min_passing_replicates": 2,
        "min_mean_accuracy_delta": 0.0,
        "independent_cache_per_replicate": True,
    },
}
PHASE5_EVOLUTION_POLICY_V2 = {
    "policy_version": "phase5-v2",
    "trigger_thresholds": dict(PHASE5_EVOLUTION_POLICY_V1["trigger_thresholds"]),
    "candidate_acceptance": {
        "min_accuracy_delta": 2 / 90,
        "min_valid_rate": 0.95,
    },
    "p05_execution": {
        "replicates": 1,
    },
}
SPECIALIZE_MANAGER_POLICY = {
    "analysis_request_kwargs": {"temperature": 0.2, "seed": 42},
    "clustering_request_kwargs": {"temperature": 0.2, "seed": 42},
    "generation_request_kwargs": {"temperature": 0.7, "seed": 42},
    "structured_max_retries": 1,
    "api_retry_attempts": 10,
}


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    required_fields = {
        "experiment_id", "model", "vllm_version", "max_model_len",
        "discovery_dataset", "discovery_dataset_sha256", "heldout_dataset",
        "heldout_dataset_sha256", "backend_pool", "worker_request_kwargs",
        "structured_max_retries", "api_retry_attempts", "evolution_policy",
    }
    optional_fields = {"specialize_managers"}
    if (not isinstance(value, dict)
            or not required_fields.issubset(value)
            or set(value) - required_fields - optional_fields):
        raise ValueError(
            "Phase 5 config fields mismatch: "
            f"missing={sorted(required_fields - set(value))}, "
            f"unknown={sorted(set(value) - required_fields - optional_fields)}")
    BackendPoolSpec.from_dict(value["backend_pool"])
    request = value["worker_request_kwargs"]
    if not isinstance(request, dict) or request.get("temperature") != 0.5:
        raise ValueError("Pairwise P05 requires an explicit temperature=0.5")
    if value["evolution_policy"] != PHASE5_EVOLUTION_POLICY_V2:
        raise ValueError("Phase 5 evolution policy does not match frozen phase5-v2")
    CandidateAcceptancePolicy(**value["evolution_policy"]["candidate_acceptance"])
    _specialize_profiles(value)
    return value


def _phase5_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude Phase 6-only Manager identities from the frozen Phase 5 identity."""

    value = dict(config)
    value.pop("specialize_managers", None)
    return value


def _specialize_profiles(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return validated, stage-specific Manager profiles.

    Older Phase 5 configs remain readable by mapping all three stages to the
    original Pairwise backend. New experiments should set ``specialize_managers``
    explicitly so each stage has an auditable model and backend identity.
    """

    configured = config.get("specialize_managers")
    if configured is None:
        request_by_stage = {
            "error_signature": SPECIALIZE_MANAGER_POLICY["analysis_request_kwargs"],
            "semantic_cluster": SPECIALIZE_MANAGER_POLICY["clustering_request_kwargs"],
            "child_generation": SPECIALIZE_MANAGER_POLICY["generation_request_kwargs"],
        }
        configured = {
            stage: {
                "model": config["model"],
                "backend_pool": config["backend_pool"],
                "api_key_env": None,
                "input_mode": "multimodal" if stage != "semantic_cluster" else "text",
                "request_kwargs": request_by_stage[stage],
                "vllm_identity": {
                    "version": config["vllm_version"],
                    "max_model_len": config["max_model_len"],
                },
            }
            for stage in ("error_signature", "semantic_cluster", "child_generation")
        }
    expected_stages = {"error_signature", "semantic_cluster", "child_generation"}
    if not isinstance(configured, dict) or set(configured) != expected_stages:
        raise ValueError("specialize_managers must define exactly the three Manager stages")
    result: dict[str, dict[str, Any]] = {}
    for stage in sorted(expected_stages):
        profile = configured[stage]
        fields = {"model", "backend_pool", "api_key_env", "input_mode",
                  "request_kwargs", "vllm_identity"}
        if not isinstance(profile, dict) or set(profile) != fields:
            raise ValueError(f"invalid {stage} Manager profile fields")
        if not isinstance(profile["model"], str) or not profile["model"].strip():
            raise ValueError(f"{stage} Manager model must be non-empty")
        BackendPoolSpec.from_dict(profile["backend_pool"])
        if profile["input_mode"] not in {"text", "multimodal"}:
            raise ValueError(f"{stage} input_mode must be text or multimodal")
        if not isinstance(profile["request_kwargs"], dict):
            raise ValueError(f"{stage} request_kwargs must be an object")
        if profile["api_key_env"] is not None and (
                not isinstance(profile["api_key_env"], str)
                or not profile["api_key_env"].strip()):
            raise ValueError(f"{stage} api_key_env must be null or a non-empty string")
        identity = profile["vllm_identity"]
        if identity is not None and (
                not isinstance(identity, dict)
                or set(identity) != {"version", "max_model_len"}
                or not isinstance(identity["version"], str)
                or isinstance(identity["max_model_len"], bool)
                or not isinstance(identity["max_model_len"], int)):
            raise ValueError(f"{stage} vllm_identity is invalid")
        result[stage] = dict(profile)
    return result


def _model_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple({key: value for key, value in row.items() if key != "answer"} for row in rows)


def _status_path(output: Path) -> Path:
    return output / "stage_status.json"


def _set_status(output: Path, stage: str, status: str, details: object = None) -> None:
    value = load_json(_status_path(output)) if _status_path(output).exists() else {}
    value[stage] = {"status": status, "details": details}
    atomic_write_json(_status_path(output), value)


def _require(output: Path, stage: str) -> None:
    value = load_json(_status_path(output)) if _status_path(output).exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"stage {stage!r} must pass first")


def _inspect_vllm_endpoints(*, model_name: str, version_name: str,
                            max_model_len: int,
                            spec: BackendPoolSpec) -> list[dict[str, Any]]:
    identities = []
    for endpoint in spec.endpoints:
        base = endpoint.base_url.rstrip("/")
        if not base.endswith("/v1"):
            raise ValueError("endpoint base_url must end with /v1")
        version = request_json(base[:-3] + "/version")
        models = request_json(base + "/models")
        matches = [item for item in models.get("data", []) if item.get("id") == model_name]
        if len(matches) != 1:
            raise ValueError(f"{endpoint.endpoint_id} does not expose the configured model exactly once")
        model = matches[0]
        actual = {
            "endpoint_id": endpoint.endpoint_id,
            "vllm_version": version.get("version"),
            "model": model.get("id"),
            "checkpoint_root": model.get("root"),
            "max_model_len": model.get("max_model_len"),
        }
        expected = {
            "endpoint_id": endpoint.endpoint_id,
            "vllm_version": version_name,
            "model": model_name,
            "checkpoint_root": endpoint.checkpoint_root,
            "max_model_len": max_model_len,
        }
        if actual != expected:
            raise ValueError(f"endpoint identity mismatch: expected={expected}, actual={actual}")
        identities.append(actual)
    return identities


def _inspect_endpoints(config: Mapping[str, Any], spec: BackendPoolSpec) -> list[dict[str, Any]]:
    return _inspect_vllm_endpoints(
        model_name=config["model"], version_name=config["vllm_version"],
        max_model_len=config["max_model_len"], spec=spec)


def _manager_runtime(config: Mapping[str, Any], stage: str):
    profile = _specialize_profiles(config)[stage]
    spec = BackendPoolSpec.from_dict(profile["backend_pool"])
    identity = profile["vllm_identity"]
    endpoint_identities = (
        _inspect_vllm_endpoints(
            model_name=profile["model"], version_name=identity["version"],
            max_model_len=identity["max_model_len"], spec=spec)
        if identity is not None else []
    )
    api_key_env = profile["api_key_env"]
    api_keys = "EMPTY"
    if api_key_env is not None:
        api_keys = os.environ.get(api_key_env, "")
        if not api_keys:
            raise RuntimeError(
                f"environment variable {api_key_env!r} required by {stage} is missing")
    pool = AvailableSlotBackendPool(spec)
    request = profile["request_kwargs"]
    manager = SpecializeManager(
        model=profile["model"], backend_pool=pool, api_keys=api_keys,
        api_retry_attempts=SPECIALIZE_MANAGER_POLICY["api_retry_attempts"],
        structured_max_retries=SPECIALIZE_MANAGER_POLICY["structured_max_retries"],
        analysis_request_kwargs=(request if stage == "error_signature"
                                 else SPECIALIZE_MANAGER_POLICY["analysis_request_kwargs"]),
        clustering_request_kwargs=(request if stage == "semantic_cluster"
                                   else SPECIALIZE_MANAGER_POLICY["clustering_request_kwargs"]),
        generation_request_kwargs=(request if stage == "child_generation"
                                   else SPECIALIZE_MANAGER_POLICY["generation_request_kwargs"]),
        child_input_mode=profile["input_mode"],
    )
    public_profile = {
        "model": profile["model"], "backend_pool": spec.to_dict(),
        "api_key_env": api_key_env, "input_mode": profile["input_mode"],
        "request_kwargs": request,
        "vllm_identity": identity,
    }
    return manager, pool, public_profile, endpoint_identities


def _pairwise_evaluator(config: Mapping[str, Any], rows, pool: AvailableSlotBackendPool):
    request = dict(config["worker_request_kwargs"])
    request["temperature"] = 0.5
    return PairwiseVoteMultiModalEvaluator(
        worker_args={
            "model": config["model"],
            "api_keys": "EMPTY",
            "request_kwargs": request,
            "api_retry_attempts": config["api_retry_attempts"],
        },
        dataset=rows,
        backend_id=pool.backend_id,
        max_concurrent=pool.spec.global_request_concurrency,
        max_retries=config["structured_max_retries"],
        max_data_chars=None,
        encode_local_image=True,
        call_backend=pool,
    )


def _expected_pairwise_request_spec(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> DualWorkerRequestSpec:
    if not rows:
        raise ValueError("at least one row is required to construct request identity")
    pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(config["backend_pool"]))
    return _pairwise_evaluator(config, _model_rows(rows[:1]), pool).request_spec()


def _pairwise_cached(evaluator, rows, rubric, cache, callback):
    backend = OnlinePairwiseVoteBackend(evaluator, cache)
    nodes = tuple(rubric.get_node(node_id) for node_id in rubric.preorder_node_ids())
    outputs: list[dict[str, Any]] = [dict() for _ in rows]
    metrics: list[list[ModelCallMetrics]] = [[] for _ in rows]
    remaining = [len(nodes)] * len(rows)
    with ThreadPoolExecutor(max_workers=evaluator.max_concurrent) as executor:
        futures = {
            executor.submit(backend.evaluate, row, node): (sample_index, node)
            for sample_index, row in enumerate(rows)
            for node in nodes
        }
        for future in as_completed(futures):
            sample_index, node = futures[future]
            result = future.result()
            outputs[sample_index][node.criterion.name] = result.output
            metrics[sample_index].append(result.metrics)
            remaining[sample_index] -= 1
            if remaining[sample_index] == 0:
                callback(
                    sample_index,
                    str(rows[sample_index][evaluator.sample_id_field]),
                    combine_model_call_metrics(metrics[sample_index]),
                )
    answers = tuple(aggregate_flat_votes(item.vote for item in row.values()) for row in outputs)
    return PairwisePredictionOutput(
        tuple(str(row[evaluator.sample_id_field]) for row in rows),
        tuple(evaluator.sample_fingerprint(row) for row in rows),
        tuple(StructuredCriterionSnapshot(node.criterion.name, node.criterion.description) for node in nodes),
        tuple(outputs),
        answers,
        evaluator.request_spec(),
    )


def _metrics(answers: Sequence[FinalPreference], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisive = [answer in {FinalPreference.A, FinalPreference.B} for answer in answers]
    correct = [answer.value == row["answer"] for answer, row in zip(answers, rows)]
    covered = sum(decisive)
    return {
        "sample_count": len(rows),
        "accuracy": sum(correct) / len(rows),
        "coverage": covered / len(rows),
        "covered_accuracy": sum(ok and dec for ok, dec in zip(correct, decisive)) / covered if covered else 0.0,
        "tie_rate": 1.0 - covered / len(rows),
        "predictions": [answer.value for answer in answers],
    }


def freeze(config: Mapping[str, Any], output: Path) -> None:
    discovery_path, heldout_path = _path(config["discovery_dataset"]), _path(config["heldout_dataset"])
    discovery_sha = file_sha256(discovery_path).lower()
    if discovery_sha != config["discovery_dataset_sha256"].lower() or discovery_sha != DISCOVERY_SHA256:
        raise ValueError("discovery-90 SHA-256 mismatch")
    heldout_sha = file_sha256(heldout_path).lower()
    if heldout_sha != config["heldout_dataset_sha256"].lower() or heldout_sha != HELDOUT_DATASET_SHA256:
        raise ValueError("heldout-500 SHA-256 mismatch")
    discovery = load_jsonl_dataset(discovery_path, expected_count=90)
    heldout = load_jsonl_dataset(heldout_path, expected_count=500)
    rubric = build_multicrit_open_ended_init_rubric()
    spec = BackendPoolSpec.from_dict(config["backend_pool"])
    identities = _inspect_endpoints(config, spec)
    rubric_path = output / "rubrics/init_multicrit_open_ended_v1.json"
    rubric_path.parent.mkdir(parents=True, exist_ok=True)
    rubric.save_json(rubric_path)
    manifest = {
        "experiment_id": config["experiment_id"],
        "config_sha256": canonical_sha256(_phase5_config_view(config)),
        "discovery_dataset": str(discovery_path),
        "discovery_dataset_sha256": file_sha256(discovery_path),
        "discovery_count": len(discovery),
        "heldout_dataset": str(heldout_path),
        "heldout_dataset_sha256": file_sha256(heldout_path),
        "heldout_count": len(heldout),
        "rubric_sha256": rubric.rubric_sha256,
        "rubric_path": str(rubric_path),
        "backend_pool_id": spec.pool_id,
        "endpoint_identities": identities,
        "pairwise_temperature": 0.5,
        "pairwise_request_spec": _expected_pairwise_request_spec(config, discovery).to_dict(),
        "heldout_access_policy": ["init-baseline", "final-evaluation"],
        "evolution_policy": config["evolution_policy"],
        "thresholds_status": "frozen_phase5_v2",
        "phase6_allowed": False,
    }
    atomic_write_json(output / "frozen_manifest.json", manifest)
    _set_status(output, "freeze", "passed", manifest)
    print(f"Frozen Phase 5 manifest; rubric={rubric.rubric_sha256}")


def _validate_manifest(config: Mapping[str, Any], output: Path):
    _require(output, "freeze")
    manifest = load_json(output / "frozen_manifest.json")
    if manifest["config_sha256"] != canonical_sha256(_phase5_config_view(config)):
        raise RuntimeError("config changed after freeze")
    for key in ("discovery", "heldout"):
        path = _path(config[f"{key}_dataset"])
        if file_sha256(path).lower() != manifest[f"{key}_dataset_sha256"].lower():
            raise RuntimeError(f"{key} dataset changed after freeze")
    rubric = build_multicrit_open_ended_init_rubric()
    if rubric.rubric_sha256 != manifest["rubric_sha256"]:
        raise RuntimeError("Init Rubric changed after freeze")
    stored_rubric = StructuredRubric.load_json(manifest["rubric_path"])
    if stored_rubric.rubric_sha256 != rubric.rubric_sha256:
        raise RuntimeError("stored Init Rubric changed after freeze")
    if manifest.get("evolution_policy") != config["evolution_policy"]:
        raise RuntimeError("evolution policy changed after freeze")
    discovery = load_jsonl_dataset(_path(config["discovery_dataset"]), expected_count=90)
    expected_spec = _expected_pairwise_request_spec(config, discovery)
    if DualWorkerRequestSpec.from_dict(manifest["pairwise_request_spec"]) != expected_spec:
        raise RuntimeError("Pairwise request identity changed after freeze")
    return manifest, rubric


def _generate_pairwise(config, output, rubric, rows, label, *, min_valid_rate: float | None = .95):
    pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(config["backend_pool"]))
    model_rows = _model_rows(rows)
    evaluator = _pairwise_evaluator(config, model_rows, pool)
    cache = JsonPredictionCache(output / f"cache/{label}", CacheMode.READ_WRITE)
    prediction = _pairwise_cached(
        evaluator, model_rows, rubric, cache,
        make_progress_callback(output, label, len(rows), pool),
    )
    artifact = output / f"predictions/{label}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    prediction.save_json(artifact)
    valid = sum(
        item.parse_ok and item.answer_valid
        for row in prediction.node_outputs for item in row.values()
    )
    total = len(rows) * len(rubric.nodes)
    atomic_write_json(output / f"provenance/{label}.json", pool.provenance_dict())
    if min_valid_rate is not None and valid / total < min_valid_rate:
        raise RuntimeError(f"{label} final-valid rate below 95%")
    return prediction, artifact, valid / total


def _offline_variants(rubric, prediction, rows):
    backend = OfflinePairwiseVoteBackend(prediction, rubric)
    executor = DualCascadeExecutor(rubric, backend)
    results = {}
    answers_by_variant = {}
    for variant in (
        StructuredSystemVariant.B1_FLAT,
        StructuredSystemVariant.M1_ALL_ROOTS_CASCADE,
    ):
        execution = executor.execute_batch(rows, ExecutionConfig(variant))
        answers = tuple(trace.final_preference for trace in execution.traces)
        answers_by_variant[variant.value] = answers
        results[variant.value] = _metrics(answers, rows)
    if answers_by_variant[StructuredSystemVariant.B1_FLAT.value] != answers_by_variant[StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]:
        raise RuntimeError("Init Rubric B1 and M1 predictions differ")
    return results, answers_by_variant[StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]


def init_baseline(config: Mapping[str, Any], output: Path) -> None:
    status = load_json(_status_path(output)) if _status_path(output).exists() else {}
    if status.get("init_baseline", {}).get("status") == "passed":
        print("init-baseline already passed; heldout-500 will not be evaluated again")
        return
    manifest, rubric = _validate_manifest(config, output)
    discovery = load_jsonl_dataset(_path(config["discovery_dataset"]), expected_count=90)
    heldout = load_jsonl_dataset(_path(config["heldout_dataset"]), expected_count=500)
    discovery_prediction, discovery_artifact, discovery_valid = _generate_pairwise(
        config, output, rubric, discovery, "init_pairwise_p05_discovery90"
    )
    heldout_prediction, heldout_artifact, heldout_valid = _generate_pairwise(
        config, output, rubric, heldout, "init_pairwise_p05_heldout500"
    )
    discovery_metrics, _ = _offline_variants(rubric, discovery_prediction, discovery)
    heldout_metrics, _ = _offline_variants(rubric, heldout_prediction, heldout)
    report = {
        "rubric_sha256": rubric.rubric_sha256,
        "discovery_artifact": str(discovery_artifact),
        "heldout_artifact": str(heldout_artifact),
        "discovery_final_valid_rate": discovery_valid,
        "heldout_final_valid_rate": heldout_valid,
        "discovery90": discovery_metrics,
        "heldout500": heldout_metrics,
        "b1_m1_identical": True,
        "heldout_evaluation_number": 1,
    }
    atomic_write_json(output / "reports/init_baseline.json", report)
    _set_status(output, "init_baseline", "passed", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def feedback(config: Mapping[str, Any], output: Path) -> None:
    _require(output, "init_baseline")
    manifest, rubric = _validate_manifest(config, output)
    discovery = load_jsonl_dataset(_path(config["discovery_dataset"]), expected_count=90)
    prediction = PairwisePredictionOutput.load_json(
        output / "predictions/init_pairwise_p05_discovery90.json"
    )
    _, m1_answers = _offline_variants(rubric, prediction, discovery)
    value = extract_rubric_feedback(
        rubric,
        prediction,
        discovery,
        m1_answers,
        expected_request_spec=DualWorkerRequestSpec.from_dict(manifest["pairwise_request_spec"]),
    )
    report = value.to_dict()
    atomic_write_json(output / "reports/init_feedback.json", report)
    _set_status(output, "feedback", "passed", {
        "rubric_gap_count": len(value.rubric_gap_sample_ids),
        "cascade_failure_count": len(value.cascade_failure_sample_ids),
        "aggregation_conflict_count": len(value.aggregation_conflict_sample_ids),
    })
    print(f"feedback: gaps={len(value.rubric_gap_sample_ids)} failures={len(value.cascade_failure_sample_ids)} conflicts={len(value.aggregation_conflict_sample_ids)}")


def _cluster_capacity(
    nodes: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for node_id, node in nodes.items():
        wrong = node["wrong"]
        value = {"decisive_wrong": wrong}
        for count in range(2, thresholds["max_children"] + 1):
            value[f"max_equal_size_for_{count}_clusters"] = wrong // count
        value["max_supported_clusters"] = min(
            thresholds["max_children"], wrong // thresholds["N_min_cluster"]
        )
        result[node_id] = value
    return result


def phase5_report(config: Mapping[str, Any], output: Path) -> None:
    _require(output, "feedback")
    _validate_manifest(config, output)
    baseline = load_json(output / "reports/init_baseline.json")
    feedback_value = load_json(output / "reports/init_feedback.json")
    policy = config["evolution_policy"]
    threshold = policy["trigger_thresholds"]
    cluster_capacity = _cluster_capacity(feedback_value["nodes"], threshold)
    report = {
        "rubric_sha256": baseline["rubric_sha256"],
        "discovery90": baseline["discovery90"],
        "heldout500_init_only": baseline["heldout500"],
        "nodes": feedback_value["nodes"],
        "agreement_by_pair": feedback_value["agreement_by_pair"],
        "rubric_gap_count": len(feedback_value["rubric_gap_sample_ids"]),
        "cascade_failure_count": len(feedback_value["cascade_failure_sample_ids"]),
        "aggregation_conflict_count": len(feedback_value["aggregation_conflict_sample_ids"]),
        "cluster_capacity": cluster_capacity,
        "evolution_policy": policy,
        "thresholds_status": "FROZEN_PHASE5_V2",
        "phase6_allowed": True,
    }
    atomic_write_json(output / "reports/phase5_report.json", report)
    manifest = load_json(output / "frozen_manifest.json")
    manifest["phase6_allowed"] = True
    atomic_write_json(output / "frozen_manifest.json", manifest)
    _set_status(output, "phase5_report", "passed", {
        "thresholds_status": "FROZEN_PHASE5_V2", "phase6_allowed": True,
    })
    print(json.dumps(report, indent=2, ensure_ascii=False))


def finalize_phase5(config: Mapping[str, Any], output: Path) -> None:
    """Migrate a completed Phase 5 run into the current frozen policy contract."""

    _require(output, "init_baseline")
    _require(output, "feedback")
    _require(output, "phase5_report")
    manifest = load_json(output / "frozen_manifest.json")
    phase5_config = _phase5_config_view(config)
    current_hash = canonical_sha256(phase5_config)
    legacy_config = dict(phase5_config)
    legacy_config.pop("evolution_policy")
    v1_config = dict(phase5_config)
    v1_config["evolution_policy"] = PHASE5_EVOLUTION_POLICY_V1
    migratable_hashes = {
        current_hash,
        canonical_sha256(legacy_config),
        canonical_sha256(v1_config),
    }
    if manifest.get("config_sha256") not in migratable_hashes:
        raise RuntimeError("existing Phase 5 manifest does not match current or migratable config")

    discovery = load_jsonl_dataset(_path(config["discovery_dataset"]), expected_count=90)
    heldout = load_jsonl_dataset(_path(config["heldout_dataset"]), expected_count=500)
    if file_sha256(_path(config["discovery_dataset"])).lower() != DISCOVERY_SHA256:
        raise RuntimeError("discovery dataset changed before Phase 5 finalization")
    if file_sha256(_path(config["heldout_dataset"])).lower() != HELDOUT_DATASET_SHA256:
        raise RuntimeError("heldout dataset changed before Phase 5 finalization")
    rubric = build_multicrit_open_ended_init_rubric()
    if rubric.rubric_sha256 != manifest.get("rubric_sha256"):
        raise RuntimeError("Init Rubric changed before Phase 5 finalization")

    expected_spec = _expected_pairwise_request_spec(config, discovery)
    for label, rows in (
        ("init_pairwise_p05_discovery90", discovery),
        ("init_pairwise_p05_heldout500", heldout),
    ):
        prediction = PairwisePredictionOutput.load_json(output / f"predictions/{label}.json")
        if prediction.request_spec != expected_spec:
            raise RuntimeError(f"{label} request identity does not match frozen P05 spec")
        backend = OfflinePairwiseVoteBackend(prediction, rubric)
        for row in rows:
            backend.sample_fingerprint(row)

    manifest.update({
        "config_sha256": current_hash,
        "pairwise_request_spec": expected_spec.to_dict(),
        "evolution_policy": config["evolution_policy"],
        "thresholds_status": "frozen_phase5_v2",
        "phase6_allowed": True,
    })
    atomic_write_json(output / "frozen_manifest.json", manifest)
    phase5_report(config, output)
    _set_status(output, "finalize_phase5", "passed", {
        "policy_version": config["evolution_policy"]["policy_version"],
        "phase6_allowed": True,
    })
    print("Phase 5 finalized; frozen policy=phase5-v2; phase6_allowed=true")


def _specialize_output(output: Path) -> Path:
    return output / "phase6_specialize"


def _specialize_status_path(output: Path) -> Path:
    return _specialize_output(output) / "stage_status.json"


def _set_specialize_status(output: Path, stage: str, status: str, details: object = None) -> None:
    path = _specialize_status_path(output)
    value = load_json(path) if path.exists() else {}
    value[stage] = {"status": status, "details": details}
    atomic_write_json(path, value)


def _require_specialize(output: Path, stage: str) -> None:
    path = _specialize_status_path(output)
    value = load_json(path) if path.exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"Specialize stage {stage!r} must pass first")


def _specialize_already_passed(output: Path, stage: str) -> bool:
    path = _specialize_status_path(output)
    value = load_json(path) if path.exists() else {}
    return value.get(stage, {}).get("status") == "passed"


def _specialize_log(output: Path, message: str) -> None:
    path = _specialize_output(output) / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _safe_artifact_name(value: str) -> str:
    return canonical_sha256(value)[:24] + ".json"


def _phase6_inputs(config: Mapping[str, Any], output: Path):
    """Validate discovery-only Phase 6 inputs without reading heldout-500."""

    _require(output, "finalize_phase5")
    manifest = load_json(output / "frozen_manifest.json")
    if not manifest.get("phase6_allowed"):
        raise RuntimeError("Phase 5 manifest has not enabled Phase 6")
    if manifest.get("config_sha256") != canonical_sha256(_phase5_config_view(config)):
        raise RuntimeError("Phase 5 config changed before Specialize")
    discovery_path = _path(config["discovery_dataset"])
    if file_sha256(discovery_path).lower() != manifest["discovery_dataset_sha256"].lower():
        raise RuntimeError("discovery-90 changed before Specialize")
    rows = load_jsonl_dataset(discovery_path, expected_count=90)
    rubric = StructuredRubric.load_json(manifest["rubric_path"])
    if rubric.rubric_sha256 != manifest["rubric_sha256"]:
        raise RuntimeError("Phase 5 Init Rubric changed before Specialize")
    prediction = PairwisePredictionOutput.load_json(
        output / "predictions/init_pairwise_p05_discovery90.json")
    expected_spec = DualWorkerRequestSpec.from_dict(manifest["pairwise_request_spec"])
    if prediction.request_spec != expected_spec:
        raise RuntimeError("Phase 5 discovery Pairwise identity changed")
    offline = OfflinePairwiseVoteBackend(prediction, rubric)
    for row in rows:
        offline.sample_fingerprint(row)
    feedback_value = RubricFeedback.from_dict(load_json(output / "reports/init_feedback.json"))
    return manifest, rubric, rows, prediction, feedback_value


def specialize_freeze(config: Mapping[str, Any], output: Path, parent_node_id: str | None) -> None:
    if not parent_node_id:
        raise ValueError("specialize-freeze requires --parent-node-id")
    if _specialize_already_passed(output, "freeze"):
        existing = load_json(_specialize_output(output) / "frozen_manifest.json")
        if existing["parent_node_id"] != parent_node_id:
            raise RuntimeError("Specialize is already frozen for a different parent")
        print("specialize-freeze already passed; reusing frozen manifest")
        return
    phase5, rubric, rows, prediction, feedback_value = _phase6_inputs(config, output)
    manager_profiles = {}
    manager_request_specs = {}
    endpoint_identities = {}
    for stage in ("error_signature", "semantic_cluster", "child_generation"):
        manager, _, profile, identities = _manager_runtime(config, stage)
        manager_profiles[stage] = profile
        manager_request_specs[stage] = manager.request_specs()[stage].to_dict()
        endpoint_identities[stage] = identities
    context = EvolutionContext(rubric, feedback_value)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    trigger = detect_specialize_trigger(context, parent_node_id, thresholds)
    target = _specialize_output(output)
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "trigger.json", trigger.to_dict())
    rubric.save_json(target / "rubric_before.json")
    manifest = {
        "stage": "phase6b_specialize_v1", "phase5_manifest_sha256": canonical_sha256(phase5),
        "discovery_dataset_sha256": phase5["discovery_dataset_sha256"],
        "rubric_before_sha256": rubric.rubric_sha256,
        "base_pairwise_artifact_sha256": canonical_sha256(prediction.to_dict()),
        "pairwise_request_spec": prediction.request_spec.to_dict(),
        "parent_node_id": parent_node_id, "trigger": trigger.to_dict(),
        "thresholds": thresholds,
        "candidate_acceptance": config["evolution_policy"]["candidate_acceptance"],
        "manager_policy": SPECIALIZE_MANAGER_POLICY,
        "manager_request_specs": manager_request_specs,
        "manager_profiles": manager_profiles,
        "endpoint_identities": endpoint_identities,
        "heldout_access": "forbidden",
    }
    atomic_write_json(target / "frozen_manifest.json", manifest)
    if not trigger.triggered:
        _set_specialize_status(output, "freeze", "not_triggered", trigger.to_dict())
        print(json.dumps(trigger.to_dict(), indent=2, ensure_ascii=False))
        return
    _set_specialize_status(output, "freeze", "passed", {
        "parent_node_id": parent_node_id,
        "wrong_samples": len(trigger.decisive_wrong_sample_ids),
        "remaining_capacity": trigger.remaining_capacity})
    _specialize_log(output, f"freeze passed parent={parent_node_id}")
    print(f"Specialize frozen: parent={parent_node_id} wrong={len(trigger.decisive_wrong_sample_ids)}")


def _load_specialize_context(config: Mapping[str, Any], output: Path):
    _require_specialize(output, "freeze")
    phase5, rubric, rows, prediction, feedback_value = _phase6_inputs(config, output)
    target = _specialize_output(output)
    manifest = load_json(target / "frozen_manifest.json")
    if manifest["phase5_manifest_sha256"] != canonical_sha256(phase5):
        raise RuntimeError("Phase 5 manifest changed after Specialize freeze")
    if manifest["rubric_before_sha256"] != rubric.rubric_sha256:
        raise RuntimeError("Specialize base Rubric changed")
    if manifest["base_pairwise_artifact_sha256"] != canonical_sha256(prediction.to_dict()):
        raise RuntimeError("Specialize base Pairwise artifact changed")
    return manifest, EvolutionContext(rubric, feedback_value), rows, prediction


def specialize_signatures(config: Mapping[str, Any], output: Path) -> None:
    if _specialize_already_passed(output, "signatures"):
        print("specialize-signatures already passed; reusing signatures")
        return
    manifest, context, rows, _ = _load_specialize_context(config, output)
    target = _specialize_output(output)
    trigger = SpecializeTriggerDecision.from_dict(manifest["trigger"])
    parent = context.rubric.get_node(trigger.parent_node_id)
    errors = {item.sample_id: item for item in context.feedback.nodes[parent.node_id].errors
              if item.outcome == "wrong"}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    manager, pool, profile, identities = _manager_runtime(config, "error_signature")
    current_spec = manager.request_specs()["error_signature"].to_dict()
    if manifest["manager_request_specs"]["error_signature"] != current_spec:
        raise RuntimeError("ErrorSignature Manager identity changed after freeze")
    outputs: dict[str, ErrorSignatureOutput] = {}
    shard_dir = target / "signatures"
    expected_spec = manager.request_specs()["error_signature"]
    for sample_id in errors:
        shard = shard_dir / _safe_artifact_name(sample_id)
        if shard.exists():
            saved = ErrorSignatureOutput.from_dict(load_json(shard))
            if saved.signature is not None and saved.signature.sample_id == sample_id and saved.request_spec == expected_spec:
                outputs[sample_id] = saved
    pending = {sample_id: error for sample_id, error in errors.items() if sample_id not in outputs}
    callback = make_progress_callback(target, "specialize-signatures", len(pending), pool) if pending else None
    with ThreadPoolExecutor(max_workers=min(max(1, len(pending)), pool.spec.global_request_concurrency)) as executor:
        futures = {executor.submit(manager.infer_signature, row_by_id[sample_id], parent, error): sample_id
                   for sample_id, error in pending.items()}
        for index, future in enumerate(as_completed(futures), 1):
            sample_id = futures[future]
            result = future.result(); outputs[sample_id] = result
            atomic_write_json(shard_dir / _safe_artifact_name(sample_id), result.to_dict())
            if callback is not None:
                callback(index - 1, sample_id, result.metrics)
    artifact = {"schema_version": "1.0.0", "parent_node_id": parent.node_id,
                "outputs": [outputs[sample_id].to_dict()
                            for sample_id in trigger.decisive_wrong_sample_ids]}
    atomic_write_json(target / "error_signatures.json", artifact)
    atomic_write_json(target / "provenance/signatures.json", pool.provenance_dict())
    invalid = [sample_id for sample_id, result in outputs.items() if result.signature is None]
    if invalid:
        _set_specialize_status(output, "signatures", "invalid", {"sample_ids": invalid})
        raise RuntimeError(f"Specialize ErrorSignature failed for {len(invalid)} samples")
    _set_specialize_status(output, "signatures", "passed", {"count": len(outputs)})
    _specialize_log(output, f"signatures passed count={len(outputs)} reused={len(outputs)-len(pending)}")
    print(f"Specialize signatures complete: {len(outputs)}/{len(outputs)} valid")


def _load_signatures(output: Path) -> tuple[ErrorSignatureOutput, ...]:
    value = load_json(_specialize_output(output) / "error_signatures.json")
    return tuple(ErrorSignatureOutput.from_dict(item) for item in value["outputs"])


def specialize_cluster(config: Mapping[str, Any], output: Path) -> None:
    if _specialize_already_passed(output, "cluster"):
        print("specialize-cluster already passed; reusing cluster proposal")
        return
    _require_specialize(output, "signatures")
    manifest, context, _, _ = _load_specialize_context(config, output)
    outputs = _load_signatures(output)
    trigger = SpecializeTriggerDecision.from_dict(manifest["trigger"])
    expected_signature_spec = manifest["manager_request_specs"]["error_signature"]
    if tuple(item.signature.sample_id for item in outputs if item.signature) != trigger.decisive_wrong_sample_ids:
        raise RuntimeError("ErrorSignature artifact does not cover the frozen wrong set in order")
    if any(item.request_spec.to_dict() != expected_signature_spec for item in outputs):
        raise RuntimeError("ErrorSignature request identity changed")
    signatures = tuple(item.signature for item in outputs if item.signature is not None)
    parent = context.rubric.get_node(trigger.parent_node_id)
    manager, pool, profile, identities = _manager_runtime(config, "semantic_cluster")
    current_cluster_spec = manager.request_specs()["semantic_cluster"].to_dict()
    if manifest["manager_request_specs"]["semantic_cluster"] != current_cluster_spec:
        status = load_json(_specialize_output(output) / "stage_status.json")
        if status.get("cluster", {}).get("status") not in {None, "invalid", "not_triggered"}:
            raise RuntimeError("cannot change Semantic Clustering protocol after a valid result")
        manifest["manager_request_specs"]["semantic_cluster"] = current_cluster_spec
        manifest.setdefault("manager_profiles", {})["semantic_cluster"] = profile
        manifest.setdefault("endpoint_identities", {})["semantic_cluster"] = identities
        manifest["manager_policy"] = SPECIALIZE_MANAGER_POLICY
        atomic_write_json(_specialize_output(output) / "frozen_manifest.json", manifest)
        _specialize_log(output, "semantic clustering Manager identity updated before a valid result")
    target = _specialize_output(output)
    proposal = None
    proposal_source = "model_call"
    failure_path = target / "cluster_failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        try:
            proposal = parse_cluster_proposal_response(
                failure["raw_response"],
                expected_sample_ids=trigger.decisive_wrong_sample_ids,
                min_cluster_size=manifest["thresholds"]["N_min_cluster"],
                max_clusters=trigger.remaining_capacity,
                attempt_count=failure["attempt_count"],
                metrics=ModelCallMetrics.from_dict(failure["metrics"]),
                request_spec=manager.request_specs()["semantic_cluster"],
            )
            proposal_source = "recovered_existing_raw_response"
        except (KeyError, TypeError, ValueError):
            proposal = None
    if proposal is None:
        try:
            proposal = manager.cluster(
                signatures, criterion_name=parent.criterion.name,
                min_cluster_size=manifest["thresholds"]["N_min_cluster"],
                max_clusters=trigger.remaining_capacity)
        except SpecializeManagerFailure as exc:
            atomic_write_json(target / "cluster_failure.json", exc.to_dict())
            _set_specialize_status(output, "cluster", "invalid", exc.to_dict())
            raise
    atomic_write_json(target / "cluster_proposal.json", proposal.to_dict())
    atomic_write_json(target / "provenance/cluster.json", {
        "proposal_source": proposal_source,
        "current_run": pool.provenance_dict(),
    })
    if len(proposal.clusters) < 2:
        details = {"status": "NOT_TRIGGERED_AFTER_CLUSTERING",
                   "valid_clusters": len(proposal.clusters)}
        _set_specialize_status(output, "cluster", "not_triggered", details)
        print(json.dumps(details, indent=2))
        return
    _set_specialize_status(output, "cluster", "passed", {
        "clusters": len(proposal.clusters),
        "unclustered": len(proposal.unclustered_sample_ids),
        "proposal_source": proposal_source})
    _specialize_log(output, f"cluster passed clusters={len(proposal.clusters)} "
                            f"source={proposal_source}")
    print(f"Specialize clustering complete: {len(proposal.clusters)} clusters; "
          f"source={proposal_source}")


def specialize_propose(config: Mapping[str, Any], output: Path) -> None:
    if _specialize_already_passed(output, "propose"):
        print("specialize-propose already passed; reusing child proposals")
        return
    _require_specialize(output, "cluster")
    manifest, context, rows, _ = _load_specialize_context(config, output)
    target = _specialize_output(output)
    proposal = ClusterProposal.from_dict(load_json(target / "cluster_proposal.json"))
    trigger = SpecializeTriggerDecision.from_dict(manifest["trigger"])
    assigned = [sample_id for cluster in proposal.clusters for sample_id in cluster.sample_ids]
    if (len(set(assigned)) != len(assigned)
            or set(assigned) & set(proposal.unclustered_sample_ids)
            or set(assigned) | set(proposal.unclustered_sample_ids) != set(trigger.decisive_wrong_sample_ids)
            or any(len(cluster.sample_ids) < manifest["thresholds"]["N_min_cluster"]
                   for cluster in proposal.clusters)
            or not 2 <= len(proposal.clusters) <= trigger.remaining_capacity):
        raise RuntimeError("saved cluster proposal violates the frozen partition contract")
    if proposal.request_spec.to_dict() != manifest["manager_request_specs"]["semantic_cluster"]:
        raise RuntimeError("cluster request identity changed")
    outputs = _load_signatures(output)
    signatures = {item.signature.sample_id: item.signature for item in outputs if item.signature}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    parent = context.rubric.get_node(manifest["parent_node_id"])
    manager, pool, profile, identities = _manager_runtime(config, "child_generation")
    current_child_spec = manager.request_specs()["child_generation"].to_dict()
    if manifest["manager_request_specs"]["child_generation"] != current_child_spec:
        status = load_json(_specialize_output(output) / "stage_status.json")
        if status.get("propose", {}).get("status") not in {None, "invalid"}:
            raise RuntimeError("cannot change Child Generation Manager after a valid proposal")
        manifest["manager_request_specs"]["child_generation"] = current_child_spec
        manifest.setdefault("manager_profiles", {})["child_generation"] = profile
        manifest.setdefault("endpoint_identities", {})["child_generation"] = identities
        atomic_write_json(target / "frozen_manifest.json", manifest)
        _specialize_log(output, "child generation Manager identity updated before a valid proposal")
    children = []
    failure_path = target / "child_generation_failure.json"
    saved_failure = load_json(failure_path) if failure_path.exists() else None
    recovered_failure = False
    try:
        for cluster in proposal.clusters:
            representative_ids = cluster.sample_ids[:3]
            input_identity = canonical_sha256({
                "parent": parent.to_dict(), "cluster": cluster.to_dict(),
                "signatures": [signatures[sample_id].to_dict() for sample_id in cluster.sample_ids],
                "representative_ids": representative_ids,
                "siblings": [{"name": item.criterion_name, "description": item.description}
                             for item in children],
                "request_spec": manager.request_specs()["child_generation"].to_dict(),
            })
            shard = target / "children" / _safe_artifact_name(cluster.cluster_id)
            child = None
            if shard.exists():
                saved = load_json(shard)
                if saved.get("input_sha256") == input_identity:
                    child = ChildCriterionProposal.from_dict(saved["proposal"])
            if child is None and saved_failure is not None and not recovered_failure:
                try:
                    child = parse_child_proposal_response(
                        saved_failure["raw_response"],
                        cluster=cluster,
                        attempt_count=saved_failure["attempt_count"],
                        metrics=ModelCallMetrics.from_dict(saved_failure["metrics"]),
                        request_spec=manager.request_specs()["child_generation"],
                    )
                except (KeyError, TypeError, ValueError):
                    child = None
                else:
                    recovered_failure = True
                    atomic_write_json(shard, {
                        "input_sha256": input_identity,
                        "proposal": child.to_dict(),
                        "source": "recovered_failure_response",
                    })
            if child is None:
                child = manager.generate_child(
                    parent=parent, cluster=cluster,
                    signatures=tuple(signatures[sample_id] for sample_id in cluster.sample_ids),
                    representative_rows=tuple(row_by_id[sample_id] for sample_id in representative_ids),
                    siblings=tuple(children))
                atomic_write_json(shard, {"input_sha256": input_identity,
                                          "proposal": child.to_dict()})
            children.append(child)
    except SpecializeManagerFailure as exc:
        atomic_write_json(target / "child_generation_failure.json", exc.to_dict())
        _set_specialize_status(output, "propose", "invalid", exc.to_dict())
        raise
    try:
        candidate = build_specialize_candidate(context, parent.node_id, proposal, children, rows, signatures)
        rubric_candidate = apply_rubric_patch(context.rubric, candidate.edit_candidate.patch)
        refresh = plan_artifact_refresh(context.rubric, rubric_candidate)
    except Exception as exc:
        failure = {"stage": "candidate_validation", "error": str(exc)}
        atomic_write_json(target / "candidate_validation_failure.json", failure)
        _set_specialize_status(output, "propose", "invalid", failure)
        raise
    atomic_write_json(target / "child_proposals.json", {
        "schema_version": "1.0.0", "children": [item.to_dict() for item in children]})
    atomic_write_json(target / "candidate.json", candidate.to_dict())
    rubric_candidate.save_json(target / "rubric_candidate.json")
    atomic_write_json(target / "artifact_refresh_plan.json", refresh.to_dict())
    atomic_write_json(target / "provenance/children.json", {
        "recovered_existing_raw_response": recovered_failure,
        "current_run": pool.provenance_dict(),
    })
    _set_specialize_status(output, "propose", "passed", {
        "candidate_id": candidate.edit_candidate.candidate_id,
        "children": len(children), "rubric_sha256": rubric_candidate.rubric_sha256})
    _specialize_log(output, f"propose passed children={len(children)}")
    print(f"Specialize candidate generated: {len(children)} children")


def specialize_evaluate(config: Mapping[str, Any], output: Path) -> None:
    if _specialize_already_passed(output, "evaluate"):
        print("specialize-evaluate already passed; reusing evaluation")
        return
    _require_specialize(output, "propose")
    started = time.perf_counter()
    target = _specialize_output(output)

    def announce(message: str) -> None:
        line = f"[specialize-evaluate +{time.perf_counter() - started:.1f}s] {message}"
        print(line, flush=True)
        _specialize_log(output, line)

    announce("validating candidate and frozen discovery inputs")
    manifest, context, rows, base_prediction = _load_specialize_context(config, output)
    candidate = SpecializeCandidate.from_dict(load_json(target / "candidate.json"))
    rubric_candidate = StructuredRubric.load_json(target / "rubric_candidate.json")
    reconstructed = apply_rubric_patch(context.rubric, candidate.edit_candidate.patch)
    if reconstructed.rubric_sha256 != rubric_candidate.rubric_sha256:
        raise RuntimeError("saved candidate Rubric does not match its patch")
    if plan_artifact_refresh(context.rubric, rubric_candidate).to_dict() != load_json(
            target / "artifact_refresh_plan.json"):
        raise RuntimeError("saved artifact refresh plan was modified")
    child_ids = tuple(candidate.node_id_by_cluster[item.cluster_id] for item in candidate.children)
    child_nodes = {node_id: rubric_candidate.get_node(node_id) for node_id in child_ids}
    child_rubric = StructuredRubric(nodes=child_nodes, edges=(), root_ids=child_ids)
    try:
        announce(
            f"generating/loading Pairwise outputs for {len(child_ids)} children × {len(rows)} samples")
        child_prediction, artifact, _ = _generate_pairwise(
            config, target, child_rubric, rows, "specialize_children_p05_discovery90",
            min_valid_rate=None)
        announce(f"Pairwise artifact ready: {artifact}")
        announce("assembling base and child shared-output artifacts")
        combined = assemble_specialized_pairwise_prediction(
            base_prediction, child_prediction, rubric_candidate)
        combined_path = target / "predictions/combined_pairwise_p05_discovery90.json"
        combined.save_json(combined_path)
        parent_fitness = context.feedback.nodes[candidate.parent_node_id].fitness
        policy = CandidateAcceptancePolicy(**manifest["candidate_acceptance"])
        announce("replaying M1 before/after and computing local diagnostics")
        evaluation, before_execution, after_execution = evaluate_specialize_candidate(
            before_rubric=context.rubric, after_rubric=rubric_candidate,
            combined_prediction=combined, child_prediction=child_prediction,
            dataset=rows, parent_node_id=candidate.parent_node_id,
            cluster_proposal=candidate.cluster_proposal, candidate=candidate,
            policy=policy, parent_fitness=parent_fitness)
        baseline = load_json(output / "reports/init_baseline.json")
        expected = baseline["discovery90"][StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]["predictions"]
        actual = [trace.final_preference.value for trace in before_execution.traces]
        if actual != expected:
            raise RuntimeError("Before M1 does not reproduce the Phase 5 baseline")
        before_execution.save_json(target / "m1_before.json")
        after_execution.save_json(target / "m1_after.json")
        atomic_write_json(target / "evaluation.json", evaluation.to_dict())
        atomic_write_json(target / "local_diagnostics.json", {
            "child_diagnostics": [item.to_dict() for item in evaluation.child_diagnostics],
            "subtree_diagnostic": evaluation.subtree_diagnostic.to_dict()})
        decision = {"decision": evaluation.candidate_evaluation.decision.value,
                    "reasons": list(evaluation.candidate_evaluation.reasons),
                    "mechanism_risks": list(evaluation.subtree_diagnostic.mechanism_risks),
                    "candidate_id": candidate.edit_candidate.candidate_id,
                    "accepted_rubric": (str(target / "rubric_candidate.json")
                                        if evaluation.candidate_evaluation.decision.value == "accept" else None)}
        atomic_write_json(target / "decision.json", decision)
    except Exception as exc:
        decision = {"decision": "invalid", "reasons": [str(exc)],
                    "mechanism_risks": [], "candidate_id": candidate.edit_candidate.candidate_id,
                    "accepted_rubric": None}
        atomic_write_json(target / "decision.json", decision)
        _set_specialize_status(output, "evaluate", "invalid", decision)
        announce(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    _set_specialize_status(output, "evaluate", "passed", decision)
    announce(f"complete decision={decision['decision']}")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def specialize_report(config: Mapping[str, Any], output: Path) -> None:
    if _specialize_already_passed(output, "report"):
        print("specialize-report already passed; reusing report")
        return
    _require_specialize(output, "evaluate")
    _load_specialize_context(config, output)
    target = _specialize_output(output)
    report = {
        "trigger": load_json(target / "trigger.json"),
        "clusters": load_json(target / "cluster_proposal.json"),
        "candidate": load_json(target / "candidate.json"),
        "refresh_plan": load_json(target / "artifact_refresh_plan.json"),
        "evaluation": load_json(target / "evaluation.json"),
        "decision": load_json(target / "decision.json"),
        "heldout_accessed": False,
    }
    atomic_write_json(target / "specialize_report.json", report)
    _set_specialize_status(output, "report", "passed", report["decision"])
    _specialize_log(output, f"report passed decision={report['decision']['decision']}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


def main() -> int:
    load_local_env(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("stage", choices=(
        "freeze", "init-baseline", "feedback", "phase5-report", "finalize-phase5",
        "specialize-freeze", "specialize-signatures", "specialize-cluster",
        "specialize-propose", "specialize-evaluate", "specialize-report"))
    parser.add_argument("--parent-node-id")
    args = parser.parse_args()
    config = _config(args.config.resolve())
    output = args.output_dir.resolve()
    actions = {
        "freeze": lambda: freeze(config, output),
        "init-baseline": lambda: init_baseline(config, output),
        "feedback": lambda: feedback(config, output),
        "phase5-report": lambda: phase5_report(config, output),
        "finalize-phase5": lambda: finalize_phase5(config, output),
        "specialize-freeze": lambda: specialize_freeze(config, output, args.parent_node_id),
        "specialize-signatures": lambda: specialize_signatures(config, output),
        "specialize-cluster": lambda: specialize_cluster(config, output),
        "specialize-propose": lambda: specialize_propose(config, output),
        "specialize-evaluate": lambda: specialize_evaluate(config, output),
        "specialize-report": lambda: specialize_report(config, output),
    }
    actions[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

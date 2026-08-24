r"""
Five-root Split-only evolution (397B Manager, Pairwise only on 8001):

    $ErrorActionPreference = "Stop"
    $python = "C:\Users\wenqx\miniconda3\envs\critiq\python.exe"
    $module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
    $config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
    $output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

    function Invoke-EvolutionStage([string]$stage) {
        & $python -m $module --config $config --output-dir $output $stage
        if ($LASTEXITCODE -ne 0) { throw "$stage failed: $LASTEXITCODE" }
    }

    Invoke-EvolutionStage "split-evolution-repair-audit"
    Invoke-EvolutionStage "split-evolution-freeze"
    Invoke-EvolutionStage "split-evolution-smoke"
    Invoke-RestMethod "http://localhost:8001/v1/models" | Out-Null
    Invoke-EvolutionStage "split-evolution-run"
    Invoke-EvolutionStage "split-evolution-report"
    Invoke-EvolutionStage "split-evolution-heldout"
    Invoke-EvolutionStage "split-evolution-final-report"

Artifacts are isolated under ``$output/phase6_split_only_evolution_v2``; v1 is read-only.

Manager Global-Rubric Memory ablation (reuses completed v2 signatures read-only):

    Invoke-EvolutionStage "split-memory-freeze"
    Invoke-EvolutionStage "split-memory-smoke"
    Invoke-EvolutionStage "split-memory-run"
    Invoke-EvolutionStage "split-memory-report"
    Invoke-EvolutionStage "split-memory-heldout"
    Invoke-EvolutionStage "split-memory-final-report"

Treatment artifacts are isolated under
``$output/phase6_split_only_evolution_global_memory_v1``. The heldout stage is
exploratory and must be run only after the discovery report freezes the final Rubric.
Evolving Structured Rubrics 的 Phase 5 与 Phase 6B 实验入口。

当前 Split v2 完整运行脚本（397B Manager；Pairwise Worker 仅使用本地 8001）：

    $ErrorActionPreference = "Stop"
    $python = "C:\Users\wenqx\miniconda3\envs\critiq\python.exe"
    $module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
    $config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
    # 使用已完成 Phase 5、但尚未运行旧 Split 的新实验目录。
    $output = "output/evolving_structured_rubrics/rubric_evolution_phase5"
    $parent = "init_02_visual_grounding_and_details"

    function Invoke-SplitStage {
        param([string]$Stage, [string[]]$ExtraArgs = @())
        & $python -m $module --config $config --output-dir $output $Stage @ExtraArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Split stage $Stage failed with exit code $LASTEXITCODE"
        }
    }

    Invoke-RestMethod "http://localhost:8001/v1/models" | Out-Null
    Invoke-SplitStage "split-freeze" @("--parent-node-id", $parent)
    Invoke-SplitStage "split-signatures"
    Invoke-SplitStage "split-cluster"
    Invoke-SplitStage "split-propose"
    Invoke-SplitStage "split-evaluate"
    Invoke-SplitStage "split-report"

当前 ErrorSignature 8B vs Qwen3.5-397B 严格对照实验：

    # 前置条件：原 split-signatures 已完成；不会覆盖 phase6_split。
    Invoke-SplitStage "split-signature-qwen35-generate"
    Invoke-SplitStage "split-signature-qwen35-report"

    # B1 质量门控通过后运行完整 R2 传播实验：
    Invoke-SplitStage "split-signature-qwen35-cluster"
    Invoke-SplitStage "split-signature-qwen35-propose"

    # Pairwise evaluate 仅路由到 8001；不要求 8000 在线。
    Invoke-RestMethod "http://localhost:8001/v1/models" | Out-Null
    Invoke-SplitStage "split-signature-qwen35-evaluate"
    Invoke-SplitStage "split-signature-qwen35-compare"

    # 一次性 heldout-500 Visual subtree 泛化测试；四个 children 仅通过 8001 推理。
    Invoke-RestMethod "http://localhost:8001/v1/models" | Out-Null
    Invoke-SplitStage "split-signature-qwen35-heldout-visual"

结果写入 ``$output/phase6_split_signature_qwen35_397b``。前两阶段只比较 paired
ErrorSignature，并通过 ``paired_signature_report.json`` 与
``manual_audit_template.json`` 完成质量门控；后四阶段以397B signatures 重新聚类和生成
children，同时固定下游 Manager、Pairwise Worker 语义与 Fitness；Pairwise 实际请求仅走 8001，最终报告为
``end_to_end_comparison_report.json``。一次性 heldout 结果写入
``$output/phase6_split_signature_qwen35_397b/heldout_visual_only/report.json``；生成报告后不得再依据 heldout 选择或改写 children。
当前严格 text-only vs multimodal child-generation 对照实验：

    # 前置条件：上面的 text-only Split 已完成；以下阶段不会覆盖 phase6_split。
    Invoke-SplitStage "split-multimodal-propose"
    Invoke-SplitStage "split-multimodal-evaluate"
    Invoke-SplitStage "split-multimodal-report"

multimodal treatment 结果写入 ``$output/phase6_split_multimodal_child``；最终对照
报告为 ``$output/phase6_split_multimodal_child/comparison_report.json``。实验固定
parent、ErrorSignatures、cluster、representative IDs、sibling context、模型、提示词、
解码参数、Pairwise Worker 与 Fitness；唯一处理差异是 child generation 是否发送图片。
原 text-only 结果 ``$output/phase6_split`` 保持不变。

结果位于 ``$output/phase6_split``，跨次 Split 历史位于
``$output/split_history.json``。运行 ``split-cluster`` 和 ``split-propose`` 前，
需要在项目 ``.env`` 中配置 ``GUIJI_API_KEY``。各阶段支持通过 stage status 续跑。

PowerShell 公共变量：

    $module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
    $config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
    # 使用已完成 Phase 5、但尚未运行旧 Split 的新实验目录。
    $output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

    $module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
    $config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
    $output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

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
ErrorSignature、Semantic Clustering 与 Child Generation 统一使用
Qwen/Qwen3.5-397B-A17B。ErrorSignature 为多模态输入；后两个阶段使用文本输入，
Child Generation 接收代表样本的 question/A/B/gold 而不发送图片。
Clustering 使用 prompt v1.0 的一次完整 partition 输出，temperature=0.2。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.dual_evaluator import (
    CacheOptimizedPairwiseVoteMultiModalEvaluator,
    PairwiseVoteMultiModalEvaluator,
)
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
    execute_offline_m1,
    extract_rubric_feedback,
    parse_child_proposal_response,
    parse_cluster_proposal_response,
    plan_artifact_refresh,
    project_pairwise_prediction,
)
from critiq.structured.aggregation import aggregate_flat_votes
from critiq.structured.version import (
    PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    PAIRWISE_WORKER_PROMPT_VERSION,
)

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
    optional_fields = {"specialize_managers", "split_evolution",
                       "split_manager_memory_ablation", "refine_manager",
                       "refine_operator", "split_retry_experiment",
                       "split_retry_v2_experiment", "refine_role_experiment",
                       "visual_split_refine_experiment",
                       "five_root_locked_split_refine_experiment",
                       "prompt_v2_aligned_evolution",
                       "discovery_v2_prompt_v2_evolution",
                       "root_boundary_pre_refine_experiment",
                       "pairwise_cache_prompt_ablation",
                       "vlrb_prompt_v2_transfer",
                       "vlrb_prompt_v2_evolved",
                       "vlrb_discovery_v2",
                       "vlrb_qwen25_transfer",
                       "qwen25_full_evolution",
                       "vlrb_qwen25_evolved",
                       "vlrb_phase16_checkpoint_transfer",
                       "vlrb_phase17_checkpoint_transfer",
                       "visual_gate_experiment",
                       "full_child_gate_experiment",
                       "vlrb_full_child_gate_experiment",
                       "vlrb_phase16_root_child_router",
                       "discovery_data_v2"}
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
    value.pop("split_evolution", None)
    value.pop("split_manager_memory_ablation", None)
    value.pop("refine_manager", None)
    value.pop("refine_operator", None)
    value.pop("split_retry_experiment", None)
    value.pop("split_retry_v2_experiment", None)
    value.pop("refine_role_experiment", None)
    value.pop("visual_split_refine_experiment", None)
    value.pop("five_root_locked_split_refine_experiment", None)
    value.pop("prompt_v2_aligned_evolution", None)
    value.pop("root_boundary_pre_refine_experiment", None)
    value.pop("vlrb_prompt_v2_transfer", None)
    value.pop("vlrb_prompt_v2_evolved", None)
    value.pop("vlrb_qwen25_transfer", None)
    value.pop("qwen25_full_evolution", None)
    value.pop("vlrb_qwen25_evolved", None)
    value.pop("visual_gate_experiment", None)
    value.pop("full_child_gate_experiment", None)
    value.pop("vlrb_full_child_gate_experiment", None)
    value.pop("discovery_data_v2", None)
    return value


def _validate_phase6_config_against_manifest(
        config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Validate frozen scientific identity while allowing Phase-6 execution routing.

    Phase 5 was originally frozen with a two-endpoint execution pool.  Later
    Phase-6 experiments may execute the same frozen Pairwise request identity on
    only vllm-8001, so the mutable runtime ``backend_pool`` must not be compared
    through the old whole-config hash.  Dataset hashes, model/decoding identity,
    evolution policy, and stored prediction identity remain strictly checked.
    """

    expected_spec = DualWorkerRequestSpec.from_dict(manifest["pairwise_request_spec"])
    if config["experiment_id"] != manifest["experiment_id"]:
        raise RuntimeError("Phase 5 experiment identity changed before Specialize")
    if config["model"] != expected_spec.model:
        raise RuntimeError("Phase 5 Pairwise model changed before Specialize")
    if config["worker_request_kwargs"] != expected_spec.decoding_config:
        raise RuntimeError("Phase 5 Pairwise decoding changed before Specialize")
    if config["evolution_policy"] != manifest["evolution_policy"]:
        raise RuntimeError("Phase 5 evolution policy changed before Specialize")


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


def _manager_runtime(config: Mapping[str, Any], stage: str, *,
                     rubric_memory_mode: str = "none"):
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
        rubric_memory_mode=rubric_memory_mode,
    )
    public_profile = {
        "model": profile["model"], "backend_pool": spec.to_dict(),
        "api_key_env": api_key_env, "input_mode": profile["input_mode"],
        "request_kwargs": request,
        "vllm_identity": identity,
    }
    if rubric_memory_mode != "none":
        public_profile["rubric_memory_mode"] = rubric_memory_mode
    return manager, pool, public_profile, endpoint_identities


def _pairwise_evaluator(
    config: Mapping[str, Any], rows, pool: AvailableSlotBackendPool,
    *, request_backend_id: str | None = None,
):
    request = dict(config["worker_request_kwargs"])
    request["temperature"] = 0.5
    prompt_mode = config.get("_pairwise_prompt_mode", "v1")
    evaluator_type = {
        "v1": PairwiseVoteMultiModalEvaluator,
        "v2_cache": CacheOptimizedPairwiseVoteMultiModalEvaluator,
    }.get(prompt_mode)
    if evaluator_type is None:
        raise ValueError(f"unsupported Pairwise prompt mode: {prompt_mode!r}")
    evaluator = evaluator_type(
        worker_args={
            "model": config["model"],
            "api_keys": "EMPTY",
            "request_kwargs": request,
            "api_retry_attempts": config["api_retry_attempts"],
        },
        dataset=rows,
        backend_id=request_backend_id or pool.backend_id,
        max_concurrent=pool.spec.global_request_concurrency,
        max_retries=config["structured_max_retries"],
        max_data_chars=None,
        encode_local_image=True,
        call_backend=pool,
    )
    evaluator.prompt_version = (
        PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
        if prompt_mode == "v2_cache" else PAIRWISE_WORKER_PROMPT_VERSION)
    return evaluator


def _expected_pairwise_request_spec(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> DualWorkerRequestSpec:
    if not rows:
        raise ValueError("at least one row is required to construct request identity")
    pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(config["backend_pool"]))
    return _pairwise_evaluator(config, _model_rows(rows[:1]), pool).request_spec()


def _pairwise_cached(
    evaluator, rows, rubric, cache, callback, evaluation_callback=None,
):
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
            if evaluation_callback is not None:
                evaluation_callback(
                    sample_index,
                    f"{rows[sample_index][evaluator.sample_id_field]}::{node.criterion.name}",
                    result.metrics,
                )
            outputs[sample_index][node.criterion.name] = result.output
            metrics[sample_index].append(result.metrics)
            remaining[sample_index] -= 1
            if remaining[sample_index] == 0 and callback is not None:
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
        prompt_version=getattr(
            evaluator, "prompt_version", PAIRWISE_WORKER_PROMPT_VERSION),
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


def _generate_pairwise(
    config, output, rubric, rows, label, *, min_valid_rate: float | None = .95,
    execution_backend_pool: Mapping[str, Any] | None = None,
    request_backend_id: str | None = None,
    request_level_progress: bool = False,
):
    execution_spec = BackendPoolSpec.from_dict(
        execution_backend_pool or config["backend_pool"])
    pool = AvailableSlotBackendPool(execution_spec)
    model_rows = _model_rows(rows)
    evaluator = _pairwise_evaluator(
        config, model_rows, pool, request_backend_id=request_backend_id)
    cache = JsonPredictionCache(output / f"cache/{label}", CacheMode.READ_WRITE)
    sample_callback = None if request_level_progress else make_progress_callback(
        output, label, len(rows), pool)
    evaluation_callback = (
        make_progress_callback(output, label, len(rows) * len(rubric.nodes), pool)
        if request_level_progress else None)
    prediction = _pairwise_cached(
        evaluator, model_rows, rubric, cache, sample_callback,
        evaluation_callback=evaluation_callback)
    artifact = output / f"predictions/{label}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    prediction.save_json(artifact)
    valid = sum(
        item.parse_ok and item.answer_valid
        for row in prediction.node_outputs for item in row.values()
    )
    total = len(rows) * len(rubric.nodes)
    provenance = pool.provenance_dict()
    provenance["request_backend_id"] = evaluator.request_spec().backend_id
    provenance["execution_backend_id"] = pool.backend_id
    atomic_write_json(output / f"provenance/{label}.json", provenance)
    if min_valid_rate is not None and valid / total < min_valid_rate:
        raise RuntimeError(f"{label} final-valid rate below 95%")
    return prediction, artifact, valid / total


def _single_endpoint_execution_pool(
    config: Mapping[str, Any], endpoint_id: str,
) -> dict[str, Any]:
    """Build an execution-only route without changing frozen request identity."""

    logical = BackendPoolSpec.from_dict(config["backend_pool"])
    matches = [item for item in logical.endpoints if item.endpoint_id == endpoint_id]
    if len(matches) != 1:
        raise ValueError(f"Pairwise endpoint {endpoint_id!r} is not uniquely configured")
    endpoint = matches[0]
    execution = BackendPoolSpec(
        pool_id=f"{logical.pool_id}-route-{endpoint.endpoint_id}",
        common_checkpoint_id=logical.common_checkpoint_id,
        global_request_concurrency=min(
            logical.global_request_concurrency, endpoint.max_concurrency),
        endpoints=(endpoint,),
    )
    return execution.to_dict()


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
    return output / "phase6_split"


def _split_multimodal_output(output: Path) -> Path:
    return output / "phase6_split_multimodal_child"


def _multimodal_child_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Clone a text-child config while changing only child input_mode."""

    value = json.loads(json.dumps(config))
    managers = value.get("specialize_managers")
    if not isinstance(managers, dict) or "child_generation" not in managers:
        raise ValueError("multimodal ablation requires explicit child_generation profile")
    control = managers["child_generation"]
    if control.get("input_mode") != "text":
        raise ValueError("control child_generation profile must be text-only")
    treatment = dict(control)
    treatment["input_mode"] = "multimodal"
    managers["child_generation"] = treatment
    if {key: item for key, item in control.items() if key != "input_mode"} != {
            key: item for key, item in treatment.items() if key != "input_mode"}:
        raise RuntimeError("multimodal treatment changed more than input_mode")
    return value


def _signature_qwen35_output(output: Path) -> Path:
    return output / "phase6_split_signature_qwen35_397b"


def _qwen35_signature_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Clone config while changing only the ErrorSignature model/backend identity."""

    value = json.loads(json.dumps(config))
    managers = value.get("specialize_managers")
    if not isinstance(managers, dict):
        raise ValueError("Qwen3.5 signature ablation requires explicit Manager profiles")
    control = managers.get("error_signature")
    donor = managers.get("child_generation")
    if not isinstance(control, dict) or not isinstance(donor, dict):
        raise ValueError("signature and child-generation profiles are required")
    if control.get("input_mode") != "multimodal":
        raise ValueError("control ErrorSignature Manager must already be multimodal")
    if donor.get("model") != "Qwen/Qwen3.5-397B-A17B":
        raise ValueError("child-generation profile must provide Qwen/Qwen3.5-397B-A17B")
    treatment = dict(control)
    for field in ("model", "backend_pool", "api_key_env", "vllm_identity"):
        treatment[field] = json.loads(json.dumps(donor[field]))
    treatment["input_mode"] = control["input_mode"]
    treatment["request_kwargs"] = json.loads(json.dumps(control["request_kwargs"]))
    managers["error_signature"] = treatment
    return value


def _set_signature_qwen35_status(
    output: Path, stage: str, status: str, details: object = None,
) -> None:
    path = _signature_qwen35_output(output) / "stage_status.json"
    value = load_json(path) if path.exists() else {}
    value[stage] = {"status": status, "details": details}
    atomic_write_json(path, value)


def _require_signature_qwen35(output: Path, stage: str) -> None:
    path = _signature_qwen35_output(output) / "stage_status.json"
    value = load_json(path) if path.exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"Qwen3.5 ErrorSignature stage {stage!r} must pass first")

def _set_split_multimodal_status(
    output: Path, stage: str, status: str, details: object = None,
) -> None:
    path = _split_multimodal_output(output) / "stage_status.json"
    value = load_json(path) if path.exists() else {}
    value[stage] = {"status": status, "details": details}
    atomic_write_json(path, value)


def _require_split_multimodal(output: Path, stage: str) -> None:
    path = _split_multimodal_output(output) / "stage_status.json"
    value = load_json(path) if path.exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"multimodal Split stage {stage!r} must pass first")

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


def _split_parent_scope_predictions(
    *,
    context: EvolutionContext,
    candidate: SpecializeCandidate,
    combined_prediction: PairwisePredictionOutput,
    after_execution: Any,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze paired parent/subtree predictions on the parent's applicability domain."""

    parent_name = context.rubric.get_node(candidate.parent_node_id).criterion.name
    trace_by_id = {trace.sample_id: trace for trace in after_execution.traces}
    records = []
    for row, outputs in zip(rows, combined_prediction.node_outputs):
        parent_output = outputs[parent_name]
        if (not parent_output.parse_ok or not parent_output.answer_valid
                or parent_output.vote.value not in {"A", "B"}):
            continue
        sample_id = str(row["sample_id"])
        trace = trace_by_id[sample_id]
        node_trace = next(
            node for node in trace.nodes if node.node_id == candidate.parent_node_id)
        parent_vote = parent_output.vote.value
        specialized_vote = node_trace.subtree_vote.value
        gold = str(row["answer"])
        parent_correct = parent_vote == gold
        specialized_correct = specialized_vote == gold
        outcome = (
            "corrected" if not parent_correct and specialized_correct
            else "harmed" if parent_correct and not specialized_correct
            else "unchanged_correct" if parent_correct and specialized_correct
            else "unchanged_wrong")
        records.append({
            "sample_id": sample_id,
            "gold": gold,
            "parent_vote": parent_vote,
            "specialized_vote": specialized_vote,
            "parent_correct": parent_correct,
            "specialized_correct": specialized_correct,
            "outcome": outcome,
        })
    return records


def _generate_split_failure_attribution(
    config: Mapping[str, Any],
    output: Path,
    context: EvolutionContext,
    candidate: SpecializeCandidate,
    evaluation: SpecializeEvaluation,
    prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Ask the 397B Manager to diagnose a rejected complete child set."""

    manager, pool, profile, identities = _manager_runtime(config, "semantic_cluster")
    if profile["model"] != "Qwen/Qwen3.5-397B-A17B":
        raise RuntimeError("Split failure attribution must use Qwen/Qwen3.5-397B-A17B")
    signatures = tuple(
        output_value.signature for output_value in _load_signatures(output)
        if output_value.signature is not None)
    changed = [record for record in prediction_records
               if record["outcome"] in {"corrected", "harmed"}]
    local_metrics = {
        "parent_scope_size": len(prediction_records),
        "parent_accuracy": evaluation.parent_accuracy,
        "specialized_accuracy": evaluation.specialized_accuracy,
        "accuracy_delta": evaluation.accuracy_delta,
        "corrected_count": len(evaluation.subtree_diagnostic.corrected_sample_ids),
        "harmed_count": len(evaluation.subtree_diagnostic.harmed_sample_ids),
        "sibling_conflict_count": evaluation.subtree_diagnostic.sibling_conflict_count,
        "children": [
            {"criterion_name": item.criterion_name,
             "support": item.support,
             "accuracy": item.accuracy,
             "coverage": item.coverage}
            for item in evaluation.child_diagnostics],
    }
    artifact = manager.attribute_split_failure(
        parent=context.rubric.get_node(candidate.parent_node_id),
        signatures=signatures,
        cluster_proposal=candidate.cluster_proposal.to_dict(),
        children=candidate.children,
        local_metrics=local_metrics,
        changed_predictions=changed)
    target = _specialize_output(output)
    atomic_write_json(target / "failure_attribution.json", artifact)
    atomic_write_json(target / "provenance/failure_attribution.json", {
        "manager_profile": profile,
        "endpoint_identities": identities,
        "backend_pool": pool.provenance_dict(),
    })
    return artifact

def _record_split_history(
    output: Path,
    context: EvolutionContext,
    candidate: SpecializeCandidate,
    decision: Mapping[str, Any],
    evaluation: Any | None,
    *,
    prediction_records: Sequence[Mapping[str, Any]] = (),
    failure_attribution: Mapping[str, Any] | None = None,
) -> None:
    path = output / "split_history.json"
    history = load_json(path) if path.exists() else {"schema_version": "2.0.0", "entries": []}
    if (history.get("schema_version") not in {"1.0.0", "2.0.0"}
            or not isinstance(history.get("entries"), list)):
        raise RuntimeError("split_history.json has an unsupported schema")
    history["schema_version"] = "2.0.0"
    parent = context.rubric.get_node(candidate.parent_node_id)
    signature_outputs = _load_signatures(output)
    signatures = [item.signature.to_dict() for item in signature_outputs
                  if item.signature is not None]
    entry = {
        "attempt_index": len(history["entries"]) + 1,
        "parent_node_id": candidate.parent_node_id,
        "parent_rubric_sha256": context.rubric.rubric_sha256,
        "parent_criterion": {
            "name": parent.criterion.name,
            "description": parent.criterion.description,
        },
        "candidate_id": candidate.edit_candidate.candidate_id,
        "trigger": load_json(_specialize_output(output) / "trigger.json"),
        "error_signatures": signatures,
        "clusters": candidate.cluster_proposal.to_dict(),
        "children": [item.to_dict() for item in candidate.children],
        "child_diagnostics": ([] if evaluation is None else
                              [item.to_dict() for item in evaluation.child_diagnostics]),
        "parent_accuracy": None if evaluation is None else evaluation.parent_accuracy,
        "specialized_accuracy": None if evaluation is None else evaluation.specialized_accuracy,
        "accuracy_delta": None if evaluation is None else evaluation.accuracy_delta,
        "parent_scope_predictions": [dict(item) for item in prediction_records],
        "corrected_sample_ids": ([] if evaluation is None else
                                 list(evaluation.subtree_diagnostic.corrected_sample_ids)),
        "harmed_sample_ids": ([] if evaluation is None else
                              list(evaluation.subtree_diagnostic.harmed_sample_ids)),
        "structured_failure": {
            "decision": decision["decision"],
            "reasons": list(decision.get("reasons", [])),
            "mechanism_risks": list(decision.get("mechanism_risks", [])),
        },
        "failure_attribution": None if failure_attribution is None else dict(failure_attribution),
        "decision": decision["decision"],
        "reasons": list(decision.get("reasons", [])),
    }
    history["entries"].append(entry)
    atomic_write_json(path, history)


def _prior_split_failures(output: Path, parent_node_id: str) -> list[dict[str, Any]]:
    path = output / "split_history.json"
    if not path.exists():
        return []
    history = load_json(path)
    if (history.get("schema_version") not in {"1.0.0", "2.0.0"}
            or not isinstance(history.get("entries"), list)):
        raise RuntimeError("split_history.json has an unsupported schema")
    summaries = []
    for entry in history["entries"]:
        if (entry.get("parent_node_id") != parent_node_id
                or entry.get("decision") == "accept"):
            continue
        clusters = entry.get("clusters", {}).get("clusters", [])
        attribution_artifact = entry.get("failure_attribution")
        attribution = (
            attribution_artifact.get("attribution")
            if isinstance(attribution_artifact, Mapping) else None)
        summaries.append({
            "attempt_index": entry.get("attempt_index"),
            "structured_failure": entry.get("structured_failure", {
                "decision": entry.get("decision"),
                "reasons": entry.get("reasons", []),
                "mechanism_risks": [],
            }),
            "failure_attribution": attribution,
            "clusters": [{key: cluster.get(key) for key in
                          ("cluster_id", "label", "shared_failure", "distinction")}
                         for cluster in clusters],
            "children": [{key: child.get(key) for key in
                          ("cluster_id", "criterion_name", "description", "rationale")}
                         for child in entry.get("children", [])],
            "child_diagnostics": [{key: diagnostic.get(key) for key in
                                   ("criterion_name", "support", "accuracy", "coverage")}
                                  for diagnostic in entry.get("child_diagnostics", [])],
            "parent_accuracy": entry.get("parent_accuracy", entry.get("parent_fitness")),
            "specialized_accuracy": entry.get(
                "specialized_accuracy", entry.get("collective_fitness")),
            "accuracy_delta": entry.get("accuracy_delta", entry.get("fitness_delta")),
            "corrected_sample_ids": entry.get("corrected_sample_ids", []),
            "harmed_sample_ids": entry.get("harmed_sample_ids", []),
        })
    return summaries

def _safe_artifact_name(value: str) -> str:
    return canonical_sha256(value)[:24] + ".json"


def _phase6_inputs(config: Mapping[str, Any], output: Path):
    """Validate discovery-only Phase 6 inputs without reading heldout-500."""

    _require(output, "finalize_phase5")
    manifest = load_json(output / "frozen_manifest.json")
    if not manifest.get("phase6_allowed"):
        raise RuntimeError("Phase 5 manifest has not enabled Phase 6")
    _validate_phase6_config_against_manifest(config, manifest)
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
        if existing.get("stage") != "phase6_split_v2_local_specialized_accuracy":
            raise RuntimeError(
                "Existing phase6_split artifacts use legacy Split semantics; "
                "use a fresh Phase-5 output copy for Split v2")
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
    manager_models = {profile["model"] for profile in manager_profiles.values()}
    if manager_models != {"Qwen/Qwen3.5-397B-A17B"}:
        raise RuntimeError(
            "Split requires Qwen/Qwen3.5-397B-A17B for ErrorSignature, "
            "semantic clustering, and child generation")
    context = EvolutionContext(rubric, feedback_value)
    thresholds = config["evolution_policy"]["trigger_thresholds"]
    trigger = detect_specialize_trigger(context, parent_node_id, thresholds)
    target = _specialize_output(output)
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "trigger.json", trigger.to_dict())
    rubric.save_json(target / "rubric_before.json")
    manifest = {
        "stage": "phase6_split_v2_local_specialized_accuracy", "phase5_manifest_sha256": canonical_sha256(phase5),
        "discovery_dataset_sha256": phase5["discovery_dataset_sha256"],
        "rubric_before_sha256": rubric.rubric_sha256,
        "base_pairwise_artifact_sha256": canonical_sha256(prediction.to_dict()),
        "pairwise_request_spec": prediction.request_spec.to_dict(),
        "parent_node_id": parent_node_id, "trigger": trigger.to_dict(),
        "prior_split_failures": _prior_split_failures(output, parent_node_id),
        "thresholds": thresholds,
        "candidate_acceptance": config["evolution_policy"]["candidate_acceptance"],
        "split_acceptance": {
            "metric": "local_specialized_accuracy",
            "scope": "parent_valid_decisive_ab",
            "comparator": ">=",
            "fallback": "children_tie_or_all_none_uses_parent",
            "children_unit": "complete_set",
        },
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
    if manifest.get("stage") != "phase6_split_v2_local_specialized_accuracy":
        raise RuntimeError("Specialize manifest does not use Split v2 semantics")
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
                max_clusters=trigger.remaining_capacity,
                prior_failures=manifest["prior_split_failures"])
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
                "prior_split_failures": manifest["prior_split_failures"],
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
                    siblings=tuple(children),
                    prior_failures=manifest["prior_split_failures"])
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
            config, target, child_rubric, rows, "split_children_p05_discovery90",
            min_valid_rate=None)
        announce(f"Pairwise artifact ready: {artifact}")
        announce("assembling base and child shared-output artifacts")
        combined = assemble_specialized_pairwise_prediction(
            base_prediction, child_prediction, rubric_candidate)
        combined_path = target / "predictions/combined_pairwise_p05_discovery90.json"
        combined.save_json(combined_path)
        policy = CandidateAcceptancePolicy(**manifest["candidate_acceptance"])
        announce("replaying M1 before/after and computing local diagnostics")
        evaluation, before_execution, after_execution = evaluate_specialize_candidate(
            before_rubric=context.rubric, after_rubric=rubric_candidate,
            combined_prediction=combined, child_prediction=child_prediction,
            dataset=rows, parent_node_id=candidate.parent_node_id,
            cluster_proposal=candidate.cluster_proposal, candidate=candidate,
            policy=policy)
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
        prediction_records = _split_parent_scope_predictions(
            context=context, candidate=candidate, combined_prediction=combined,
            after_execution=after_execution, rows=rows)
        decision = {
            "decision": evaluation.candidate_evaluation.decision.value,
            "reasons": list(evaluation.candidate_evaluation.reasons),
            "parent_scope_size": len(prediction_records),
            "parent_accuracy": evaluation.parent_accuracy,
            "specialized_accuracy": evaluation.specialized_accuracy,
            "accuracy_delta": evaluation.accuracy_delta,
            "local_corrected_count": len(
                evaluation.subtree_diagnostic.corrected_sample_ids),
            "local_harmed_count": len(
                evaluation.subtree_diagnostic.harmed_sample_ids),
            "global_m1_before_accuracy": evaluation.candidate_evaluation.before_accuracy,
            "global_m1_after_accuracy": evaluation.candidate_evaluation.after_accuracy,
            "global_m1_accuracy_delta": evaluation.candidate_evaluation.accuracy_delta,
            "global_m1_corrected_count": (
                evaluation.candidate_evaluation.corrected_count),
            "global_m1_harmed_count": (
                evaluation.candidate_evaluation.harmed_count),
            "parent_global_coverage": evaluation.subtree_diagnostic.parent_coverage,
            "specialized_global_coverage": evaluation.subtree_diagnostic.specialized_coverage,
            "mechanism_risks": list(evaluation.subtree_diagnostic.mechanism_risks),
            "candidate_id": candidate.edit_candidate.candidate_id,
            "accepted_rubric": (str(target / "rubric_candidate.json")
                                if evaluation.candidate_evaluation.decision.value == "accept"
                                else None),
        }
        failure_attribution = None
        if decision["decision"] == "reject":
            announce("requesting 397B failure attribution for rejected Split")
            try:
                failure_attribution = _generate_split_failure_attribution(
                    config, output, context, candidate, evaluation, prediction_records)
            except SpecializeManagerFailure as attribution_error:
                failure_attribution = {
                    "schema_version": "1.0.0",
                    "attribution": None,
                    "generation_failure": attribution_error.to_dict(),
                }
                atomic_write_json(target / "failure_attribution.json", failure_attribution)
        atomic_write_json(target / "decision.json", decision)
        _record_split_history(
            output, context, candidate, decision, evaluation,
            prediction_records=prediction_records,
            failure_attribution=failure_attribution)
    except Exception as exc:
        decision = {"decision": "invalid", "reasons": [str(exc)],
                    "mechanism_risks": [], "candidate_id": candidate.edit_candidate.candidate_id,
                    "accepted_rubric": None}
        atomic_write_json(target / "decision.json", decision)
        _record_split_history(output, context, candidate, decision, None)
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
        "failure_attribution": (load_json(target / "failure_attribution.json")
                                if (target / "failure_attribution.json").exists() else None),
        "heldout_accessed": False,
    }
    atomic_write_json(target / "split_report.json", report)
    _set_specialize_status(output, "report", "passed", report["decision"])
    _specialize_log(output, f"report passed decision={report['decision']['decision']}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


def split_signature_qwen35_generate(config: Mapping[str, Any], output: Path) -> None:
    """Generate paired Qwen3.5-397B ErrorSignatures without changing downstream stages."""

    source = _specialize_output(output)
    source_status = load_json(source / "stage_status.json")
    if source_status.get("signatures", {}).get("status") != "passed":
        raise RuntimeError("completed Qwen3-VL-8B control signatures are required")
    target = _signature_qwen35_output(output)
    status_path = target / "stage_status.json"
    if status_path.exists() and load_json(status_path).get("generate", {}).get("status") == "passed":
        print("split-signature-qwen35-generate already passed; reusing signatures")
        return

    source_manifest, context, rows, _ = _load_specialize_context(config, output)
    trigger = SpecializeTriggerDecision.from_dict(source_manifest["trigger"])
    parent = context.rubric.get_node(trigger.parent_node_id)
    control_artifact = load_json(source / "error_signatures.json")
    control_outputs = tuple(
        ErrorSignatureOutput.from_dict(item) for item in control_artifact["outputs"])
    if tuple(item.signature.sample_id for item in control_outputs if item.signature) != (
            trigger.decisive_wrong_sample_ids):
        raise RuntimeError("control ErrorSignature sample order differs from frozen trigger")

    treatment_config = _qwen35_signature_config(config)
    manager, pool, treatment_profile, endpoint_identities = _manager_runtime(
        treatment_config, "error_signature")
    control_profile = source_manifest["manager_profiles"]["error_signature"]
    if (control_profile["input_mode"] != "multimodal"
            or treatment_profile["input_mode"] != "multimodal"):
        raise RuntimeError("ErrorSignature ablation must keep multimodal input")
    if control_profile["request_kwargs"] != treatment_profile["request_kwargs"]:
        raise RuntimeError("ErrorSignature ablation changed decoding parameters")
    allowed_profile_changes = {"model", "backend_pool", "api_key_env", "vllm_identity"}
    for key in set(control_profile) - allowed_profile_changes:
        if control_profile[key] != treatment_profile[key]:
            raise RuntimeError(f"ErrorSignature ablation unexpectedly changed {key}")

    control_spec = source_manifest["manager_request_specs"]["error_signature"]
    treatment_spec = manager.request_specs()["error_signature"]
    treatment_spec_value = treatment_spec.to_dict()
    for key in ("prompt_sha256", "decoding_config", "prompt_version", "parser_version"):
        if control_spec[key] != treatment_spec_value[key]:
            raise RuntimeError(f"ErrorSignature request changed controlled field {key}")
    if control_spec["model"] == treatment_spec_value["model"]:
        raise RuntimeError("ErrorSignature treatment did not change the model")

    target.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema_version": "1.0.0",
        "experiment": "qwen3vl8b_vs_qwen35_397b_error_signature",
        "source_split_dir": str(source),
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "source_error_signatures_sha256": canonical_sha256(control_artifact),
        "fixed_parent_node_id": trigger.parent_node_id,
        "fixed_sample_ids": list(trigger.decisive_wrong_sample_ids),
        "control_profile": control_profile,
        "treatment_profile": treatment_profile,
        "control_request_spec": control_spec,
        "treatment_request_spec": treatment_spec_value,
        "endpoint_identities": endpoint_identities,
        "controlled_fields": [
            "samples", "images", "question", "A", "B", "gold", "parent_vote",
            "parent_thought", "prompt", "decoding_config", "input_mode"],
        "treatment_fields": ["model", "backend_pool"],
        "heldout_access": "forbidden",
    }
    atomic_write_json(target / "frozen_ablation_manifest.json", frozen)

    errors = {item.sample_id: item for item in context.feedback.nodes[parent.node_id].errors
              if item.outcome == "wrong"}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    expected_spec = manager.request_specs()["error_signature"]
    outputs: dict[str, ErrorSignatureOutput] = {}
    shard_dir = target / "signatures"
    for sample_id in trigger.decisive_wrong_sample_ids:
        shard = shard_dir / _safe_artifact_name(sample_id)
        if shard.exists():
            saved = ErrorSignatureOutput.from_dict(load_json(shard))
            if (saved.signature is not None and saved.signature.sample_id == sample_id
                    and saved.request_spec == expected_spec):
                outputs[sample_id] = saved
    pending = {sample_id: errors[sample_id] for sample_id in trigger.decisive_wrong_sample_ids
               if sample_id not in outputs}
    callback = make_progress_callback(
        target, "split-signature-qwen35-generate", len(pending), pool) if pending else None
    with ThreadPoolExecutor(
            max_workers=min(max(1, len(pending)), pool.spec.global_request_concurrency)) as executor:
        futures = {
            executor.submit(manager.infer_signature, row_by_id[sample_id], parent, error): sample_id
            for sample_id, error in pending.items()}
        for index, future in enumerate(as_completed(futures), 1):
            sample_id = futures[future]
            result = future.result()
            outputs[sample_id] = result
            atomic_write_json(shard_dir / _safe_artifact_name(sample_id), result.to_dict())
            if callback is not None:
                callback(index - 1, sample_id, result.metrics)

    artifact = {
        "schema_version": "1.0.0", "parent_node_id": parent.node_id,
        "outputs": [outputs[sample_id].to_dict()
                    for sample_id in trigger.decisive_wrong_sample_ids],
    }
    atomic_write_json(target / "error_signatures.json", artifact)
    atomic_write_json(target / "provenance/signatures.json", pool.provenance_dict())
    invalid = [sample_id for sample_id, result in outputs.items() if result.signature is None]
    if invalid:
        _set_signature_qwen35_status(output, "generate", "invalid", {"sample_ids": invalid})
        raise RuntimeError(f"Qwen3.5 ErrorSignature failed for {len(invalid)} samples")
    details = {"count": len(outputs), "reused": len(outputs) - len(pending),
               "model": treatment_profile["model"]}
    _set_signature_qwen35_status(output, "generate", "passed", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def split_signature_qwen35_report(config: Mapping[str, Any], output: Path) -> None:
    """Create a paired, human-auditable signature comparison without judging semantics."""

    status_path = _signature_qwen35_output(output) / "stage_status.json"
    if status_path.exists() and load_json(status_path).get("report", {}).get("status") == "passed":
        print("split-signature-qwen35-report already passed; preserving manual audit")
        return
    _require_signature_qwen35(output, "generate")
    source = _specialize_output(output)
    target = _signature_qwen35_output(output)
    frozen = load_json(target / "frozen_ablation_manifest.json")
    control_artifact = load_json(source / "error_signatures.json")
    if frozen["source_error_signatures_sha256"] != canonical_sha256(control_artifact):
        raise RuntimeError("frozen Qwen3-VL-8B control signatures changed")
    treatment_artifact = load_json(target / "error_signatures.json")
    control_outputs = {
        item.signature.sample_id: item for item in
        (ErrorSignatureOutput.from_dict(value) for value in control_artifact["outputs"])
        if item.signature is not None}
    treatment_outputs = {
        item.signature.sample_id: item for item in
        (ErrorSignatureOutput.from_dict(value) for value in treatment_artifact["outputs"])
        if item.signature is not None}
    fixed_ids = tuple(frozen["fixed_sample_ids"])
    if set(control_outputs) != set(fixed_ids) or set(treatment_outputs) != set(fixed_ids):
        raise RuntimeError("paired ErrorSignature sample sets differ")

    source_manifest, context, rows, _ = _load_specialize_context(config, output)
    parent_id = frozen["fixed_parent_node_id"]
    row_by_id = {str(row["sample_id"]): row for row in rows}
    error_by_id = {item.sample_id: item for item in context.feedback.nodes[parent_id].errors
                   if item.outcome == "wrong"}
    signature_fields = (
        "task_pattern", "visual_focus", "candidate_difference",
        "parent_failure", "suggested_subdomain")
    changed_counts = {field: 0 for field in signature_fields}
    pairs = []
    audit_items = []
    for sample_id in fixed_ids:
        left = control_outputs[sample_id].signature
        right = treatment_outputs[sample_id].signature
        assert left is not None and right is not None
        left_value = left.to_dict()
        right_value = right.to_dict()
        for field in signature_fields:
            changed_counts[field] += int(left_value[field] != right_value[field])
        row = row_by_id[sample_id]
        error = error_by_id[sample_id]
        pairs.append({
            "sample_id": sample_id, "question": row["question"],
            "A": row["A"], "B": row["B"], "gold": row["answer"],
            "parent_vote": error.vote.value, "parent_thought": error.thought,
            "control_qwen3vl8b": left_value,
            "treatment_qwen35_397b": right_value,
        })
        audit_items.append({
            "sample_id": sample_id,
            "direction_consistent": None,
            "visual_facts_consistent": None,
            "internally_consistent": None,
            "actionable_for_child_generation": None,
            "preferred_signature": None,
            "notes": "",
        })
    report = {
        "schema_version": "1.0.0",
        "experiment": "qwen3vl8b_vs_qwen35_397b_error_signature",
        "sample_count": len(fixed_ids),
        "changed_field_counts": changed_counts,
        "control_request_spec": frozen["control_request_spec"],
        "treatment_request_spec": frozen["treatment_request_spec"],
        "pairs": pairs,
        "heldout_accessed": False,
    }
    atomic_write_json(target / "paired_signature_report.json", report)
    atomic_write_json(target / "manual_audit_template.json", {
        "schema_version": "1.0.0",
        "instructions": {
            "direction_consistent": "Signature supports gold over parent_vote without reversing A/B.",
            "visual_facts_consistent": "Visual claims agree with the supplied image.",
            "internally_consistent": "Fields do not contradict one another.",
            "actionable_for_child_generation": "Failure is expressible as a non-circular pairwise rule.",
            "preferred_signature": "control, treatment, tie, or neither.",
        },
        "items": audit_items,
    })
    details = {"sample_count": len(fixed_ids),
               "paired_report": str(target / "paired_signature_report.json"),
               "manual_audit": str(target / "manual_audit_template.json")}
    _set_signature_qwen35_status(output, "report", "passed", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))

def _load_qwen35_signatures(output: Path) -> tuple[ErrorSignatureOutput, ...]:
    value = load_json(_signature_qwen35_output(output) / "error_signatures.json")
    return tuple(ErrorSignatureOutput.from_dict(item) for item in value["outputs"])


def split_signature_qwen35_cluster(config: Mapping[str, Any], output: Path) -> None:
    """Re-cluster the frozen wrong set using only the 397B ErrorSignatures."""

    _require_signature_qwen35(output, "report")
    target = _signature_qwen35_output(output)
    status = load_json(target / "stage_status.json")
    if status.get("cluster", {}).get("status") == "passed":
        print("split-signature-qwen35-cluster already passed; reusing cluster proposal")
        return
    source = _specialize_output(output)
    source_manifest, context, _, base_prediction = _load_specialize_context(config, output)
    frozen = load_json(target / "frozen_ablation_manifest.json")
    treatment_artifact = load_json(target / "error_signatures.json")
    if frozen["source_error_signatures_sha256"] != canonical_sha256(
            load_json(source / "error_signatures.json")):
        raise RuntimeError("8B control ErrorSignatures changed before R2")
    outputs = _load_qwen35_signatures(output)
    trigger = SpecializeTriggerDecision.from_dict(source_manifest["trigger"])
    if tuple(item.signature.sample_id for item in outputs if item.signature) != (
            trigger.decisive_wrong_sample_ids):
        raise RuntimeError("397B ErrorSignatures do not cover the frozen wrong set in order")
    expected_signature_spec = frozen["treatment_request_spec"]
    if any(item.request_spec.to_dict() != expected_signature_spec for item in outputs):
        raise RuntimeError("397B ErrorSignature request identity changed")
    signatures = tuple(item.signature for item in outputs if item.signature is not None)
    parent = context.rubric.get_node(trigger.parent_node_id)

    manager, pool, cluster_profile, cluster_identities = _manager_runtime(
        config, "semantic_cluster")
    cluster_spec = manager.request_specs()["semantic_cluster"].to_dict()
    if (cluster_profile != source_manifest["manager_profiles"]["semantic_cluster"]
            or cluster_spec != source_manifest["manager_request_specs"]["semantic_cluster"]):
        raise RuntimeError("R2 changed Semantic Clustering configuration")
    child_manager, _, child_profile, child_identities = _manager_runtime(
        config, "child_generation")
    child_spec = child_manager.request_specs()["child_generation"].to_dict()
    if (child_profile != source_manifest["manager_profiles"]["child_generation"]
            or child_spec != source_manifest["manager_request_specs"]["child_generation"]):
        raise RuntimeError("R2 changed Child Generation configuration")

    end_to_end_manifest = {
        "schema_version": "1.0.0",
        "experiment": "8b_vs_397b_error_signature_end_to_end_split",
        "controlled_difference": "ErrorSignature model and resulting signature artifact",
        "source_split_dir": str(source),
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "source_cluster_sha256": canonical_sha256(load_json(source / "cluster_proposal.json")),
        "source_children_sha256": canonical_sha256(load_json(source / "child_proposals.json")),
        "source_evaluation_sha256": canonical_sha256(load_json(source / "evaluation.json")),
        "source_decision_sha256": canonical_sha256(load_json(source / "decision.json")),
        "control_signatures_sha256": canonical_sha256(load_json(source / "error_signatures.json")),
        "treatment_signatures_sha256": canonical_sha256(treatment_artifact),
        "fixed_parent_node_id": trigger.parent_node_id,
        "fixed_sample_ids": list(trigger.decisive_wrong_sample_ids),
        "semantic_cluster_profile": cluster_profile,
        "semantic_cluster_request_spec": cluster_spec,
        "child_generation_profile": child_profile,
        "child_generation_request_spec": child_spec,
        "cluster_endpoint_identities": cluster_identities,
        "child_endpoint_identities": child_identities,
        "pairwise_request_spec": source_manifest["pairwise_request_spec"],
        "candidate_acceptance": source_manifest["candidate_acceptance"],
        "thresholds": source_manifest["thresholds"],
        "prior_split_failures": source_manifest["prior_split_failures"],
        "heldout_access": "forbidden",
    }
    atomic_write_json(target / "end_to_end_manifest.json", end_to_end_manifest)
    try:
        proposal = manager.cluster(
            signatures, criterion_name=parent.criterion.name,
            min_cluster_size=source_manifest["thresholds"]["N_min_cluster"],
            max_clusters=trigger.remaining_capacity,
            prior_failures=source_manifest["prior_split_failures"])
    except SpecializeManagerFailure as exc:
        atomic_write_json(target / "cluster_failure.json", exc.to_dict())
        _set_signature_qwen35_status(output, "cluster", "invalid", exc.to_dict())
        raise
    atomic_write_json(target / "cluster_proposal.json", proposal.to_dict())
    atomic_write_json(target / "provenance/cluster.json", pool.provenance_dict())
    if len(proposal.clusters) < 2:
        details = {"status": "NOT_TRIGGERED_AFTER_CLUSTERING",
                   "valid_clusters": len(proposal.clusters)}
        _set_signature_qwen35_status(output, "cluster", "not_triggered", details)
        print(json.dumps(details, indent=2, ensure_ascii=False))
        return
    details = {"clusters": len(proposal.clusters),
               "unclustered": len(proposal.unclustered_sample_ids)}
    _set_signature_qwen35_status(output, "cluster", "passed", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def split_signature_qwen35_propose(config: Mapping[str, Any], output: Path) -> None:
    """Generate text-only children from the new clusters and 397B signatures."""

    _require_signature_qwen35(output, "cluster")
    target = _signature_qwen35_output(output)
    status = load_json(target / "stage_status.json")
    if status.get("propose", {}).get("status") == "passed":
        print("split-signature-qwen35-propose already passed; reusing children")
        return
    source_manifest, context, rows, _ = _load_specialize_context(config, output)
    manifest = load_json(target / "end_to_end_manifest.json")
    treatment_artifact = load_json(target / "error_signatures.json")
    if manifest["treatment_signatures_sha256"] != canonical_sha256(treatment_artifact):
        raise RuntimeError("397B ErrorSignatures changed before child generation")
    trigger = SpecializeTriggerDecision.from_dict(source_manifest["trigger"])
    proposal = ClusterProposal.from_dict(load_json(target / "cluster_proposal.json"))
    assigned = [sample_id for cluster in proposal.clusters for sample_id in cluster.sample_ids]
    if (len(set(assigned)) != len(assigned)
            or set(assigned) & set(proposal.unclustered_sample_ids)
            or set(assigned) | set(proposal.unclustered_sample_ids) != set(
                trigger.decisive_wrong_sample_ids)
            or any(len(cluster.sample_ids) < manifest["thresholds"]["N_min_cluster"]
                   for cluster in proposal.clusters)
            or not 2 <= len(proposal.clusters) <= trigger.remaining_capacity):
        raise RuntimeError("397B-signature cluster proposal violates the partition contract")
    if proposal.request_spec.to_dict() != manifest["semantic_cluster_request_spec"]:
        raise RuntimeError("397B-signature cluster request identity changed")
    outputs = _load_qwen35_signatures(output)
    signatures = {item.signature.sample_id: item.signature for item in outputs if item.signature}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    parent = context.rubric.get_node(trigger.parent_node_id)
    manager, pool, profile, identities = _manager_runtime(config, "child_generation")
    if (profile != manifest["child_generation_profile"]
            or manager.request_specs()["child_generation"].to_dict()
            != manifest["child_generation_request_spec"]
            or identities != manifest["child_endpoint_identities"]):
        raise RuntimeError("R2 Child Generation identity changed")

    children = []
    try:
        for cluster in proposal.clusters:
            representative_ids = cluster.sample_ids[:3]
            input_identity = canonical_sha256({
                "parent": parent.to_dict(), "cluster": cluster.to_dict(),
                "signatures": [signatures[sample_id].to_dict()
                               for sample_id in cluster.sample_ids],
                "representative_ids": list(representative_ids),
                "siblings": [{"name": item.criterion_name, "description": item.description}
                             for item in children],
                "prior_split_failures": manifest["prior_split_failures"],
                "request_spec": manifest["child_generation_request_spec"],
            })
            shard = target / "children" / _safe_artifact_name(cluster.cluster_id)
            child = None
            if shard.exists():
                saved = load_json(shard)
                if saved.get("input_sha256") == input_identity:
                    child = ChildCriterionProposal.from_dict(saved["proposal"])
            if child is None:
                child = manager.generate_child(
                    parent=parent, cluster=cluster,
                    signatures=tuple(signatures[sample_id] for sample_id in cluster.sample_ids),
                    representative_rows=tuple(row_by_id[sample_id]
                                              for sample_id in representative_ids),
                    siblings=tuple(children),
                    prior_failures=manifest["prior_split_failures"])
                atomic_write_json(shard, {
                    "input_sha256": input_identity, "proposal": child.to_dict()})
            children.append(child)
    except SpecializeManagerFailure as exc:
        atomic_write_json(target / "child_generation_failure.json", exc.to_dict())
        _set_signature_qwen35_status(output, "propose", "invalid", exc.to_dict())
        raise

    candidate = build_specialize_candidate(
        context, parent.node_id, proposal, children, rows, signatures)
    rubric_candidate = apply_rubric_patch(context.rubric, candidate.edit_candidate.patch)
    refresh = plan_artifact_refresh(context.rubric, rubric_candidate)
    atomic_write_json(target / "child_proposals.json", {
        "schema_version": "1.0.0", "children": [item.to_dict() for item in children]})
    atomic_write_json(target / "candidate.json", candidate.to_dict())
    rubric_candidate.save_json(target / "rubric_candidate.json")
    atomic_write_json(target / "artifact_refresh_plan.json", refresh.to_dict())
    atomic_write_json(target / "provenance/children.json", pool.provenance_dict())
    details = {"candidate_id": candidate.edit_candidate.candidate_id,
               "children": len(children), "rubric_sha256": rubric_candidate.rubric_sha256}
    _set_signature_qwen35_status(output, "propose", "passed", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def split_signature_qwen35_evaluate(config: Mapping[str, Any], output: Path) -> None:
    """Evaluate the 397B-signature Split through only the local 8001 endpoint."""

    _require_signature_qwen35(output, "propose")
    target = _signature_qwen35_output(output)
    status = load_json(target / "stage_status.json")
    if status.get("evaluate", {}).get("status") == "passed":
        print("split-signature-qwen35-evaluate already passed; reusing evaluation")
        return
    source = _specialize_output(output)
    source_manifest, context, rows, base_prediction = _load_specialize_context(config, output)
    manifest = load_json(target / "end_to_end_manifest.json")
    frozen_checks = {
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "source_cluster_sha256": canonical_sha256(load_json(source / "cluster_proposal.json")),
        "source_children_sha256": canonical_sha256(load_json(source / "child_proposals.json")),
        "source_evaluation_sha256": canonical_sha256(load_json(source / "evaluation.json")),
        "source_decision_sha256": canonical_sha256(load_json(source / "decision.json")),
        "control_signatures_sha256": canonical_sha256(load_json(source / "error_signatures.json")),
        "treatment_signatures_sha256": canonical_sha256(
            load_json(target / "error_signatures.json")),
    }
    for key, current in frozen_checks.items():
        if manifest[key] != current:
            raise RuntimeError(f"R2 frozen artifact changed: {key}")
    candidate = SpecializeCandidate.from_dict(load_json(target / "candidate.json"))
    rubric_candidate = StructuredRubric.load_json(target / "rubric_candidate.json")
    reconstructed = apply_rubric_patch(context.rubric, candidate.edit_candidate.patch)
    if reconstructed.rubric_sha256 != rubric_candidate.rubric_sha256:
        raise RuntimeError("397B-signature candidate Rubric does not match its patch")
    if plan_artifact_refresh(context.rubric, rubric_candidate).to_dict() != load_json(
            target / "artifact_refresh_plan.json"):
        raise RuntimeError("397B-signature refresh plan changed")
    child_ids = tuple(candidate.node_id_by_cluster[item.cluster_id] for item in candidate.children)
    child_nodes = {node_id: rubric_candidate.get_node(node_id) for node_id in child_ids}
    child_rubric = StructuredRubric(nodes=child_nodes, edges=(), root_ids=child_ids)
    logical_spec = BackendPoolSpec.from_dict(config["backend_pool"])
    execution_pool = _single_endpoint_execution_pool(config, "vllm-8001")
    atomic_write_json(target / "pairwise_execution_policy.json", {
        "stage": "split-signature-qwen35-evaluate",
        "routing": "single_endpoint",
        "endpoint_id": "vllm-8001",
        "execution_backend_pool": execution_pool,
        "logical_request_backend_id": logical_spec.backend_id,
        "identity_policy": (
            "Preserve the frozen Pairwise request identity because vllm-8001 is a "
            "member of the original equivalent-checkpoint pool; record the actual "
            "single-endpoint route separately in provenance."),
    })
    child_prediction, artifact, _ = _generate_pairwise(
        config, target, child_rubric, rows,
        "signature_qwen35_children_p05_discovery90_8001", min_valid_rate=None,
        execution_backend_pool=execution_pool,
        request_backend_id=logical_spec.backend_id)
    combined = assemble_specialized_pairwise_prediction(
        base_prediction, child_prediction, rubric_candidate)
    combined.save_json(target / "predictions/combined_pairwise_p05_discovery90.json")
    evaluation, before_execution, after_execution = evaluate_specialize_candidate(
        before_rubric=context.rubric, after_rubric=rubric_candidate,
        combined_prediction=combined, child_prediction=child_prediction,
        dataset=rows, parent_node_id=candidate.parent_node_id,
        cluster_proposal=candidate.cluster_proposal, candidate=candidate,
        policy=CandidateAcceptancePolicy(**manifest["candidate_acceptance"]))
    baseline = load_json(output / "reports/init_baseline.json")
    expected = baseline["discovery90"][
        StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]["predictions"]
    if [trace.final_preference.value for trace in before_execution.traces] != expected:
        raise RuntimeError("397B-signature R2 does not reproduce Phase 5 M1 baseline")
    before_execution.save_json(target / "m1_before.json")
    after_execution.save_json(target / "m1_after.json")
    atomic_write_json(target / "evaluation.json", evaluation.to_dict())
    atomic_write_json(target / "local_diagnostics.json", {
        "child_diagnostics": [item.to_dict() for item in evaluation.child_diagnostics],
        "subtree_diagnostic": evaluation.subtree_diagnostic.to_dict()})
    decision = {
        "decision": evaluation.candidate_evaluation.decision.value,
        "reasons": list(evaluation.candidate_evaluation.reasons),
        "parent_fitness": evaluation.parent_accuracy,
        "collective_fitness": evaluation.specialized_accuracy,
        "fitness_delta": evaluation.accuracy_delta,
        "mechanism_risks": list(evaluation.subtree_diagnostic.mechanism_risks),
        "candidate_id": candidate.edit_candidate.candidate_id,
        "pairwise_artifact": str(artifact),
        "pairwise_execution_endpoint": "vllm-8001",
        "accepted_rubric": (str(target / "rubric_candidate.json")
                            if evaluation.candidate_evaluation.decision.value == "accept" else None),
    }
    atomic_write_json(target / "decision.json", decision)
    _set_signature_qwen35_status(output, "evaluate", "passed", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def _split_set_summary(evaluation: SpecializeEvaluation) -> dict[str, Any]:
    diagnostics = evaluation.child_diagnostics
    parent_accuracy = (evaluation.parent_accuracy if hasattr(evaluation, "parent_accuracy")
                       else evaluation.parent_fitness)
    specialized_accuracy = (
        evaluation.specialized_accuracy if hasattr(evaluation, "specialized_accuracy")
        else evaluation.collective_fitness)
    accuracy_delta = (evaluation.accuracy_delta if hasattr(evaluation, "accuracy_delta")
                      else evaluation.fitness_delta)
    target_support = sum(item.cluster_support for item in diagnostics)
    target_correct = sum(round(item.cluster_support * item.cluster_accuracy)
                         for item in diagnostics)
    return {
        "children": len(diagnostics),
        "mean_accuracy": sum(item.accuracy for item in diagnostics) / len(diagnostics),
        "mean_coverage": sum(item.coverage for item in diagnostics) / len(diagnostics),
        "mean_fitness": sum(item.fitness for item in diagnostics) / len(diagnostics),
        "target_cluster_support": target_support,
        "target_cluster_correct": target_correct,
        "target_cluster_accuracy": target_correct / target_support if target_support else 0.0,
        "non_target_decisive": sum(item.non_target_decisive for item in diagnostics),
        "non_target_wrong": sum(item.non_target_wrong for item in diagnostics),
        "collective_fitness": specialized_accuracy,
        "parent_fitness": parent_accuracy,
        "fitness_delta": accuracy_delta,
        "m1_before_accuracy": evaluation.candidate_evaluation.before_accuracy,
        "m1_after_accuracy": evaluation.candidate_evaluation.after_accuracy,
        "m1_accuracy_delta": evaluation.candidate_evaluation.accuracy_delta,
        "corrected_count": evaluation.candidate_evaluation.corrected_count,
        "harmed_count": evaluation.candidate_evaluation.harmed_count,
        "subtree_accuracy": evaluation.subtree_diagnostic.specialized_accuracy,
        "subtree_coverage": evaluation.subtree_diagnostic.specialized_coverage,
    }


def _activation_summary(base_path: Path, child_path: Path, parent_name: str) -> dict[str, Any]:
    base = load_json(base_path)
    child = load_json(child_path)
    base_by_id = {item["sample_id"]: item for item in base["samples"]}
    distribution: dict[int, int] = {}
    conflicts = 0
    parent_domain = 0
    for sample in child["samples"]:
        parent_vote = base_by_id[sample["sample_id"]]["node_outputs"][parent_name]["vote"]
        if parent_vote not in {"A", "B"}:
            continue
        parent_domain += 1
        votes = [value["vote"] for value in sample["node_outputs"].values()]
        decisive = [vote for vote in votes if vote in {"A", "B"}]
        distribution[len(decisive)] = distribution.get(len(decisive), 0) + 1
        conflicts += int("A" in decisive and "B" in decisive)
    child_count = len(child["criteria"])
    return {
        "parent_domain": parent_domain,
        "activation_distribution": {str(key): distribution[key] for key in sorted(distribution)},
        "mean_active_children": ((sum(key * value for key, value in distribution.items())
                                  / parent_domain) if parent_domain else 0.0),
        "samples_with_at_least_two_active": sum(
            value for key, value in distribution.items() if key >= 2),
        "samples_with_all_children_active": distribution.get(child_count, 0),
        "sibling_conflict_samples": conflicts,
    }


def split_signature_qwen35_compare(config: Mapping[str, Any], output: Path) -> None:
    """Compare original 8B-signature Split against the full 397B-signature Split."""

    del config
    _require_signature_qwen35(output, "evaluate")
    target = _signature_qwen35_output(output)
    status = load_json(target / "stage_status.json")
    if status.get("compare", {}).get("status") == "passed":
        print("split-signature-qwen35-compare already passed; reusing report")
        return
    source = _specialize_output(output)
    manifest = load_json(target / "end_to_end_manifest.json")
    if (manifest["source_evaluation_sha256"] != canonical_sha256(
            load_json(source / "evaluation.json"))
            or manifest["source_decision_sha256"] != canonical_sha256(
                load_json(source / "decision.json"))):
        raise RuntimeError("8B-signature control result changed before comparison")
    control_evaluation = SpecializeEvaluation.from_dict(load_json(source / "evaluation.json"))
    treatment_evaluation = SpecializeEvaluation.from_dict(load_json(target / "evaluation.json"))
    control_summary = _split_set_summary(control_evaluation)
    treatment_summary = _split_set_summary(treatment_evaluation)
    base_path = output / "predictions/init_pairwise_p05_discovery90.json"
    parent_name = StructuredRubric.load_json(source / "rubric_before.json").get_node(
        manifest["fixed_parent_node_id"]).criterion.name
    control_activation = _activation_summary(
        base_path, source / "predictions/split_children_p05_discovery90.json", parent_name)
    treatment_activation = _activation_summary(
        base_path, target / "predictions/signature_qwen35_children_p05_discovery90_8001.json",
        parent_name)
    report = {
        "schema_version": "1.0.0",
        "experiment": "8b_vs_397b_error_signature_end_to_end_split",
        "controlled_difference": "ErrorSignature model and propagated signature artifact",
        "execution_note": (
            "Treatment Pairwise requests were routed only through vllm-8001, an "
            "equivalent-checkpoint member of the frozen logical backend pool."),
        "pairwise_execution_policy": load_json(
            target / "pairwise_execution_policy.json"),
        "control": {
            "signature_model": "Qwen/Qwen3-VL-8B-Instruct",
            "decision": load_json(source / "decision.json"),
            "cluster_proposal": load_json(source / "cluster_proposal.json"),
            "children": [item.to_dict() for item in control_evaluation.child_diagnostics],
            "set_summary": control_summary,
            "activation_summary": control_activation,
        },
        "treatment": {
            "signature_model": "Qwen/Qwen3.5-397B-A17B",
            "decision": load_json(target / "decision.json"),
            "cluster_proposal": load_json(target / "cluster_proposal.json"),
            "children": [item.to_dict() for item in treatment_evaluation.child_diagnostics],
            "set_summary": treatment_summary,
            "activation_summary": treatment_activation,
        },
        "delta": {
            key: treatment_summary[key] - control_summary[key]
            for key in (
                "mean_accuracy", "mean_coverage", "mean_fitness",
                "target_cluster_accuracy", "non_target_wrong",
                "collective_fitness", "fitness_delta", "m1_after_accuracy",
                "m1_accuracy_delta", "subtree_accuracy", "subtree_coverage")
        },
        "heldout_accessed": False,
    }
    atomic_write_json(target / "end_to_end_comparison_report.json", report)
    details = {
        "control_collective_fitness": control_summary["collective_fitness"],
        "treatment_collective_fitness": treatment_summary["collective_fitness"],
        "delta_collective_fitness": report["delta"]["collective_fitness"],
        "delta_m1_after_accuracy": report["delta"]["m1_after_accuracy"],
        "report": str(target / "end_to_end_comparison_report.json"),
    }
    _set_signature_qwen35_status(output, "compare", "passed", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))

def split_multimodal_propose(config: Mapping[str, Any], output: Path) -> None:
    """Generate multimodal children from the exact frozen text-control inputs."""

    source = _specialize_output(output)
    source_status = load_json(source / "stage_status.json")
    if source_status.get("report", {}).get("status") != "passed":
        raise RuntimeError("completed text-only Split report is required as the control")
    target = _split_multimodal_output(output)
    status_path = target / "stage_status.json"
    if status_path.exists() and load_json(status_path).get("propose", {}).get("status") == "passed":
        print("split-multimodal-propose already passed; reusing candidate")
        return

    source_manifest, context, rows, _ = _load_specialize_context(config, output)
    cluster_value = load_json(source / "cluster_proposal.json")
    proposal = ClusterProposal.from_dict(cluster_value)
    signature_outputs = _load_signatures(output)
    signatures = {item.signature.sample_id: item.signature for item in signature_outputs
                  if item.signature is not None}
    source_child_value = load_json(source / "child_proposals.json")
    source_children = tuple(ChildCriterionProposal.from_dict(item)
                            for item in source_child_value["children"])
    if tuple(item.cluster_id for item in source_children) != tuple(
            item.cluster_id for item in proposal.clusters):
        raise RuntimeError("text-control children do not align with frozen clusters")

    treatment_config = _multimodal_child_config(config)
    manager, pool, treatment_profile, endpoint_identities = _manager_runtime(
        treatment_config, "child_generation")
    control_profile = source_manifest["manager_profiles"]["child_generation"]
    control_without_mode = {key: value for key, value in control_profile.items()
                            if key != "input_mode"}
    treatment_without_mode = {key: value for key, value in treatment_profile.items()
                              if key != "input_mode"}
    if control_without_mode != treatment_without_mode:
        raise RuntimeError("text and multimodal child profiles differ beyond input_mode")
    if control_profile["input_mode"] != "text" or treatment_profile["input_mode"] != "multimodal":
        raise RuntimeError("ablation input modes are not text vs multimodal")

    current_spec = manager.request_specs()["child_generation"]
    source_specs = {item.request_spec.prompt_version for item in source_children}
    if source_specs != {current_spec.prompt_version}:
        raise RuntimeError("text and multimodal prompt versions differ")
    row_by_id = {str(row["sample_id"]): row for row in rows}
    parent = context.rubric.get_node(source_manifest["parent_node_id"])
    target.mkdir(parents=True, exist_ok=True)
    ablation_manifest = {
        "schema_version": "1.0.0",
        "treatment": "representative_images_added_to_child_generation",
        "only_changed_field": "specialize_managers.child_generation.input_mode",
        "source_split_dir": str(source),
        "source_rubric_sha256": context.rubric.rubric_sha256,
        "source_cluster_sha256": canonical_sha256(cluster_value),
        "source_signatures_sha256": canonical_sha256(load_json(source / "error_signatures.json")),
        "source_children_sha256": canonical_sha256(source_child_value),
        "control_profile": control_profile,
        "treatment_profile": treatment_profile,
        "control_request_spec": source_children[0].request_spec.to_dict(),
        "treatment_request_spec": current_spec.to_dict(),
        "endpoint_identities": endpoint_identities,
        "fixed_representative_ids": {
            item.cluster_id: list(item.representative_sample_ids) for item in source_children},
        "fixed_sibling_context": "text-control preceding siblings",
        "cluster_and_fitness_unchanged": True,
        "heldout_access": "forbidden",
    }
    atomic_write_json(target / "frozen_ablation_manifest.json", ablation_manifest)

    children = []
    for index, (cluster, control_child) in enumerate(zip(proposal.clusters, source_children)):
        representative_ids = control_child.representative_sample_ids
        fixed_siblings = source_children[:index]
        input_identity = canonical_sha256({
            "cluster": cluster.to_dict(),
            "signatures": [signatures[sample_id].to_dict() for sample_id in cluster.sample_ids],
            "representative_ids": list(representative_ids),
            "fixed_siblings": [{"criterion_name": item.criterion_name,
                                "description": item.description}
                               for item in fixed_siblings],
            "request_spec": current_spec.to_dict(),
            "input_mode": "multimodal",
        })
        shard = target / "children" / _safe_artifact_name(cluster.cluster_id)
        child = None
        if shard.exists():
            saved = load_json(shard)
            if saved.get("input_sha256") == input_identity:
                child = ChildCriterionProposal.from_dict(saved["proposal"])
        if child is None:
            child = manager.generate_child(
                parent=parent, cluster=cluster,
                signatures=tuple(signatures[sample_id] for sample_id in cluster.sample_ids),
                representative_rows=tuple(row_by_id[sample_id] for sample_id in representative_ids),
                siblings=fixed_siblings,
                prior_failures=source_manifest["prior_split_failures"])
            # Representative examples are an ablation-controlled input, not a
            # model-selected output. The prompt permits the model to cite a
            # subset, so pin persisted metadata to the control IDs while
            # retaining raw_response for an exact audit of model output.
            if child.representative_sample_ids != representative_ids:
                child = dataclasses.replace(
                    child, representative_sample_ids=representative_ids)
            atomic_write_json(shard, {"input_sha256": input_identity,
                                      "proposal": child.to_dict()})
        children.append(child)

    candidate = build_specialize_candidate(
        context, parent.node_id, proposal, children, rows, signatures)
    rubric_candidate = apply_rubric_patch(context.rubric, candidate.edit_candidate.patch)
    atomic_write_json(target / "child_proposals.json", {
        "schema_version": "1.0.0", "children": [item.to_dict() for item in children]})
    atomic_write_json(target / "candidate.json", candidate.to_dict())
    rubric_candidate.save_json(target / "rubric_candidate.json")
    atomic_write_json(target / "artifact_refresh_plan.json",
                      plan_artifact_refresh(context.rubric, rubric_candidate).to_dict())
    atomic_write_json(target / "provenance/children.json", pool.provenance_dict())
    details = {"candidate_id": candidate.edit_candidate.candidate_id,
               "children": len(children), "rubric_sha256": rubric_candidate.rubric_sha256}
    _set_split_multimodal_status(output, "propose", "passed", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def split_multimodal_evaluate(config: Mapping[str, Any], output: Path) -> None:
    _require_split_multimodal(output, "propose")
    target = _split_multimodal_output(output)
    status = load_json(target / "stage_status.json")
    if status.get("evaluate", {}).get("status") == "passed":
        print("split-multimodal-evaluate already passed; reusing evaluation")
        return
    source = _specialize_output(output)
    source_manifest, context, rows, base_prediction = _load_specialize_context(config, output)
    frozen = load_json(target / "frozen_ablation_manifest.json")
    if (frozen["source_cluster_sha256"] != canonical_sha256(load_json(source / "cluster_proposal.json"))
            or frozen["source_signatures_sha256"] != canonical_sha256(
                load_json(source / "error_signatures.json"))
            or frozen["source_children_sha256"] != canonical_sha256(
                load_json(source / "child_proposals.json"))):
        raise RuntimeError("frozen text-control inputs changed before multimodal evaluation")
    candidate = SpecializeCandidate.from_dict(load_json(target / "candidate.json"))
    rubric_candidate = StructuredRubric.load_json(target / "rubric_candidate.json")
    reconstructed = apply_rubric_patch(context.rubric, candidate.edit_candidate.patch)
    if reconstructed.rubric_sha256 != rubric_candidate.rubric_sha256:
        raise RuntimeError("multimodal candidate Rubric does not match its patch")
    child_ids = tuple(candidate.node_id_by_cluster[item.cluster_id] for item in candidate.children)
    child_nodes = {node_id: rubric_candidate.get_node(node_id) for node_id in child_ids}
    child_rubric = StructuredRubric(nodes=child_nodes, edges=(), root_ids=child_ids)
    child_prediction, artifact, _ = _generate_pairwise(
        config, target, child_rubric, rows,
        "multimodal_children_p05_discovery90", min_valid_rate=None)
    combined = assemble_specialized_pairwise_prediction(
        base_prediction, child_prediction, rubric_candidate)
    combined.save_json(target / "predictions/combined_pairwise_p05_discovery90.json")
    evaluation, before_execution, after_execution = evaluate_specialize_candidate(
        before_rubric=context.rubric, after_rubric=rubric_candidate,
        combined_prediction=combined, child_prediction=child_prediction,
        dataset=rows, parent_node_id=candidate.parent_node_id,
        cluster_proposal=candidate.cluster_proposal, candidate=candidate,
        policy=CandidateAcceptancePolicy(**source_manifest["candidate_acceptance"]))
    baseline = load_json(output / "reports/init_baseline.json")
    expected = baseline["discovery90"][StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]["predictions"]
    if [trace.final_preference.value for trace in before_execution.traces] != expected:
        raise RuntimeError("multimodal ablation does not reproduce the Phase 5 M1 baseline")
    before_execution.save_json(target / "m1_before.json")
    after_execution.save_json(target / "m1_after.json")
    atomic_write_json(target / "evaluation.json", evaluation.to_dict())
    decision = {
        "decision": evaluation.candidate_evaluation.decision.value,
        "reasons": list(evaluation.candidate_evaluation.reasons),
        "parent_fitness": evaluation.parent_accuracy,
        "collective_fitness": evaluation.specialized_accuracy,
        "fitness_delta": evaluation.accuracy_delta,
        "mechanism_risks": list(evaluation.subtree_diagnostic.mechanism_risks),
        "candidate_id": candidate.edit_candidate.candidate_id,
        "pairwise_artifact": str(artifact),
        "accepted_rubric": (str(target / "rubric_candidate.json")
                            if evaluation.candidate_evaluation.decision.value == "accept" else None),
    }
    atomic_write_json(target / "decision.json", decision)
    _set_split_multimodal_status(output, "evaluate", "passed", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def split_multimodal_report(config: Mapping[str, Any], output: Path) -> None:
    del config
    _require_split_multimodal(output, "evaluate")
    source = _specialize_output(output)
    target = _split_multimodal_output(output)
    control_evaluation = SpecializeEvaluation.from_dict(load_json(source / "evaluation.json"))
    treatment_evaluation = SpecializeEvaluation.from_dict(load_json(target / "evaluation.json"))
    control_candidate = SpecializeCandidate.from_dict(load_json(source / "candidate.json"))
    treatment_candidate = SpecializeCandidate.from_dict(load_json(target / "candidate.json"))
    control_by_cluster = {item.cluster_id: item for item in control_candidate.children}
    treatment_by_cluster = {item.cluster_id: item for item in treatment_candidate.children}
    control_diag = {item.criterion_name: item for item in control_evaluation.child_diagnostics}
    treatment_diag = {item.criterion_name: item for item in treatment_evaluation.child_diagnostics}
    children = []
    for cluster in control_candidate.cluster_proposal.clusters:
        left = control_by_cluster[cluster.cluster_id]
        right = treatment_by_cluster[cluster.cluster_id]
        left_diag = control_diag[left.criterion_name]
        right_diag = treatment_diag[right.criterion_name]
        children.append({
            "cluster_id": cluster.cluster_id,
            "control_criterion_name": left.criterion_name,
            "treatment_criterion_name": right.criterion_name,
            "control": left_diag.to_dict(),
            "treatment": right_diag.to_dict(),
            "delta_accuracy": right_diag.accuracy - left_diag.accuracy,
            "delta_coverage": right_diag.coverage - left_diag.coverage,
            "delta_fitness": right_diag.fitness - left_diag.fitness,
            "delta_cluster_accuracy": right_diag.cluster_accuracy - left_diag.cluster_accuracy,
        })
    report = {
        "schema_version": "1.0.0",
        "experiment": "text_only_vs_multimodal_child_generation",
        "controlled_difference": "representative images supplied to Qwen/Qwen3.5-397B-A17B",
        "control": load_json(source / "decision.json"),
        "treatment": load_json(target / "decision.json"),
        "delta_collective_fitness": (
            treatment_evaluation.specialized_accuracy - control_evaluation.specialized_accuracy),
        "delta_m1_accuracy": (
            treatment_evaluation.candidate_evaluation.after_accuracy
            - control_evaluation.candidate_evaluation.after_accuracy),
        "children": children,
        "heldout_accessed": False,
    }
    atomic_write_json(target / "comparison_report.json", report)
    _set_split_multimodal_status(output, "report", "passed", {
        "delta_collective_fitness": report["delta_collective_fitness"],
        "delta_m1_accuracy": report["delta_m1_accuracy"],
    })
    print(json.dumps(report, indent=2, ensure_ascii=False))

def _heldout_visual_output(output: Path) -> Path:
    return _signature_qwen35_output(output) / "heldout_visual_only"


def _visual_variant_rubric(
    candidate_rubric: StructuredRubric,
    parent_node_id: str,
    child_node_ids: Sequence[str],
) -> StructuredRubric:
    selected = {parent_node_id, *child_node_ids}
    nodes = {node_id: candidate_rubric.get_node(node_id) for node_id in selected}
    edges = tuple(
        edge for edge in candidate_rubric.edges
        if edge.parent_id in selected and edge.child_id in selected
    )
    return StructuredRubric(nodes=nodes, edges=edges, root_ids=(parent_node_id,))


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid Wilson interval counts")
    z = 1.959963984540054
    p = successes / total
    scale = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / scale
    margin = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / scale
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _heldout_vote_metrics(votes, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(votes) != len(rows):
        raise ValueError("heldout votes and rows differ in length")
    predictions = [vote.value for vote in votes]
    correct = [prediction == row["answer"] for prediction, row in zip(predictions, rows)]
    decisive = [prediction in {"A", "B"} for prediction in predictions]
    correct_count = sum(correct)
    coverage_count = sum(decisive)
    return {
        "sample_count": len(rows),
        "correct_count": correct_count,
        "accuracy": correct_count / len(rows),
        "accuracy_ci95_wilson": _wilson_interval(correct_count, len(rows)),
        "coverage_count": coverage_count,
        "coverage": coverage_count / len(rows),
        "coverage_ci95_wilson": _wilson_interval(coverage_count, len(rows)),
        "covered_accuracy": (
            sum(ok and covered for ok, covered in zip(correct, decisive)) / coverage_count
            if coverage_count else 0.0),
        "predictions": predictions,
    }


def _paired_heldout_comparison(
    baseline_votes, treatment_votes, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(baseline_votes) != len(treatment_votes) or len(baseline_votes) != len(rows):
        raise ValueError("paired heldout inputs differ in length")
    corrected = []
    harmed = []
    for baseline, treatment, row in zip(baseline_votes, treatment_votes, rows):
        baseline_correct = baseline.value == row["answer"]
        treatment_correct = treatment.value == row["answer"]
        if treatment_correct and not baseline_correct:
            corrected.append(str(row["sample_id"]))
        elif baseline_correct and not treatment_correct:
            harmed.append(str(row["sample_id"]))
    discordant = len(corrected) + len(harmed)
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(len(corrected), len(harmed)) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "corrected_count": len(corrected),
        "harmed_count": len(harmed),
        "net_corrected": len(corrected) - len(harmed),
        "corrected_sample_ids": corrected,
        "harmed_sample_ids": harmed,
        "mcnemar_exact_two_sided_p": p_value,
    }


def split_signature_qwen35_heldout_visual(
    config: Mapping[str, Any], output: Path,
) -> None:
    """One-shot heldout-500 evaluation of the frozen Visual subtree via 8001."""

    _require_signature_qwen35(output, "evaluate")
    source = _specialize_output(output)
    treatment = _signature_qwen35_output(output)
    target = _heldout_visual_output(output)
    status_path = target / "stage_status.json"
    status = load_json(status_path) if status_path.exists() else {}
    if status.get("heldout_visual", {}).get("status") == "passed":
        print("split-signature-qwen35-heldout-visual already passed; reusing one-shot report")
        return

    heldout_path = _path(config["heldout_dataset"])
    heldout_sha256 = file_sha256(heldout_path)
    if (heldout_sha256.lower() != HELDOUT_DATASET_SHA256.lower()
            or heldout_sha256.lower()
            != str(config["heldout_dataset_sha256"]).lower()):
        raise RuntimeError("heldout-500 identity changed")
    base_path = output / "predictions/init_pairwise_p05_heldout500.json"
    base_prediction = PairwisePredictionOutput.load_json(base_path)
    candidate_value = load_json(treatment / "candidate.json")
    candidate = SpecializeCandidate.from_dict(candidate_value)
    candidate_rubric = StructuredRubric.load_json(treatment / "rubric_candidate.json")
    before_rubric = StructuredRubric.load_json(source / "rubric_before.json")
    if apply_rubric_patch(
            before_rubric,
            candidate.edit_candidate.patch).rubric_sha256 != candidate_rubric.rubric_sha256:
        raise RuntimeError("heldout candidate Rubric does not match its frozen patch")

    child_by_name = {
        child.criterion_name: candidate.node_id_by_cluster[child.cluster_id]
        for child in candidate.children
    }
    selected_names = (
        "spatial_geometric_grounding_accuracy",
        "direct_answer_visual_accuracy",
    )
    if any(name not in child_by_name for name in selected_names):
        raise RuntimeError("pre-registered discovery-selected child subset is unavailable")
    all_child_ids = tuple(child_by_name[child.criterion_name] for child in candidate.children)
    logical_spec = BackendPoolSpec.from_dict(config["backend_pool"])
    if base_prediction.request_spec.backend_id != logical_spec.backend_id:
        raise RuntimeError("heldout base artifact and frozen logical Pairwise identity differ")
    execution_pool = _single_endpoint_execution_pool(config, "vllm-8001")
    frozen = {
        "schema_version": "1.0.0",
        "experiment": "visual_subtree_heldout500_one_shot",
        "heldout_dataset_sha256": heldout_sha256,
        "base_pairwise_sha256": file_sha256(base_path),
        "candidate_sha256": canonical_sha256(candidate_value),
        "candidate_rubric_sha256": candidate_rubric.rubric_sha256,
        "child_proposals_sha256": canonical_sha256(
            load_json(treatment / "child_proposals.json")),
        "parent_node_id": candidate.parent_node_id,
        "variants": {
            "parent_only": [],
            "all_children": [child.criterion_name for child in candidate.children],
            "discovery_selected_spatial_direct": list(selected_names),
        },
        "pairwise_request_spec": base_prediction.request_spec.to_dict(),
        "execution_backend_pool": execution_pool,
        "execution_endpoint": "vllm-8001",
        "heldout_policy": (
            "One-shot final diagnostic. Do not select, rewrite, or tune children after "
            "viewing this report; new development requires a new validation/test split."),
    }
    frozen_path = target / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != frozen:
        raise RuntimeError("heldout visual experiment was already frozen with different inputs")
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(frozen_path, frozen)

    rows = load_jsonl_dataset(heldout_path, expected_count=500)
    child_nodes = {
        node_id: candidate_rubric.get_node(node_id) for node_id in all_child_ids}
    child_rubric = StructuredRubric(
        nodes=child_nodes, edges=(), root_ids=all_child_ids)
    child_prediction, child_artifact, child_valid_rate = _generate_pairwise(
        config, target, child_rubric, rows,
        "visual_children_h500_8001",
        execution_backend_pool=execution_pool,
        request_backend_id=logical_spec.backend_id,
        request_level_progress=True)
    combined = assemble_specialized_pairwise_prediction(
        base_prediction, child_prediction, candidate_rubric)
    combined_path = target / "predictions/combined_pairwise_p05_heldout500.json"
    combined.save_json(combined_path)

    variant_specs = {
        "parent_only": (),
        "all_children": all_child_ids,
        "discovery_selected_spatial_direct": tuple(
            child_by_name[name] for name in selected_names),
    }
    variant_votes = {}
    variant_metrics = {}
    for name, child_ids in variant_specs.items():
        rubric = _visual_variant_rubric(
            candidate_rubric, candidate.parent_node_id, child_ids)
        prediction = project_pairwise_prediction(combined, rubric)
        execution, votes = execute_offline_m1(rubric, prediction, rows)
        traces_path = target / f"traces/{name}.json"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        execution.save_json(traces_path)
        variant_votes[name] = votes
        variant_metrics[name] = _heldout_vote_metrics(votes, rows)

    full_m1_execution, full_m1_votes = execute_offline_m1(
        before_rubric, base_prediction, rows)
    full_m1_execution.save_json(target / "traces/original_full_m1.json")
    full_m1_metrics = _heldout_vote_metrics(full_m1_votes, rows)
    baseline = load_json(output / "reports/init_baseline.json")["heldout500"][
        StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]
    if (full_m1_metrics["accuracy"] != baseline["accuracy"]
            or full_m1_metrics["coverage"] != baseline["coverage"]):
        raise RuntimeError("heldout full-M1 replay does not reproduce the frozen baseline")

    gold = [str(row["answer"]) for row in rows]
    parent_name = candidate_rubric.get_node(candidate.parent_node_id).criterion.name
    parent_outputs = tuple(row[parent_name] for row in base_prediction.node_outputs)
    parent_scope = tuple(
        item.parse_ok and item.answer_valid and item.vote.value in {"A", "B"}
        for item in parent_outputs)
    parent_support = sum(parent_scope)
    child_diagnostics = []
    active_distribution = {count: 0 for count in range(len(candidate.children) + 1)}
    sibling_conflicts = 0
    for index, row in enumerate(child_prediction.node_outputs):
        if not parent_scope[index]:
            continue
        active = [item.vote.value for item in row.values()
                  if item.parse_ok and item.answer_valid
                  and item.vote.value in {"A", "B"}]
        active_distribution[len(active)] += 1
        if "A" in active and "B" in active:
            sibling_conflicts += 1
    for child in candidate.children:
        outputs = tuple(row[child.criterion_name] for row in child_prediction.node_outputs)
        decisive = [
            parent_scope[index] and item.parse_ok and item.answer_valid
            and item.vote.value in {"A", "B"}
            for index, item in enumerate(outputs)]
        correct = [
            decisive[index] and item.vote.value == gold[index]
            for index, item in enumerate(outputs)]
        support = sum(decisive)
        all_decisive = [
            item.parse_ok and item.answer_valid and item.vote.value in {"A", "B"}
            for item in outputs]
        all_correct = [
            all_decisive[index] and item.vote.value == gold[index]
            for index, item in enumerate(outputs)]
        all_support = sum(all_decisive)
        child_diagnostics.append({
            "criterion_name": child.criterion_name,
            "parent_domain_support": support,
            "parent_domain_accuracy": sum(correct) / support if support else 0.0,
            "parent_domain_coverage": support / parent_support,
            "all_sample_support": all_support,
            "all_sample_accuracy": sum(all_correct) / all_support if all_support else 0.0,
            "all_sample_coverage": all_support / len(rows),
        })

    report = {
        "schema_version": "1.0.0",
        "experiment": "visual_subtree_heldout500_one_shot",
        "heldout_accessed": True,
        "selection_or_tuning_after_report_forbidden": True,
        "child_pairwise_artifact": str(child_artifact),
        "combined_pairwise_artifact": str(combined_path),
        "child_final_valid_rate": child_valid_rate,
        "variants": variant_metrics,
        "original_full_m1": full_m1_metrics,
        "paired_comparisons": {
            "parent_to_all_children": _paired_heldout_comparison(
                variant_votes["parent_only"], variant_votes["all_children"], rows),
            "parent_to_discovery_selected": _paired_heldout_comparison(
                variant_votes["parent_only"],
                variant_votes["discovery_selected_spatial_direct"], rows),
            "full_m1_to_all_children": _paired_heldout_comparison(
                full_m1_votes, variant_votes["all_children"], rows),
            "full_m1_to_discovery_selected": _paired_heldout_comparison(
                full_m1_votes,
                variant_votes["discovery_selected_spatial_direct"], rows),
        },
        "child_diagnostics": child_diagnostics,
        "activation_summary_in_parent_domain": {
            "parent_support": parent_support,
            "active_children_distribution": {
                str(key): value for key, value in active_distribution.items()},
            "samples_with_at_least_two_active": sum(
                value for key, value in active_distribution.items() if key >= 2),
            "samples_with_all_children_active": active_distribution[len(candidate.children)],
            "sibling_conflict_samples": sibling_conflicts,
        },
    }
    atomic_write_json(target / "report.json", report)
    details = {
        "report": str(target / "report.json"),
        "all_children_accuracy": variant_metrics["all_children"]["accuracy"],
        "selected_subset_accuracy": variant_metrics[
            "discovery_selected_spatial_direct"]["accuracy"],
        "parent_only_accuracy": variant_metrics["parent_only"]["accuracy"],
        "original_full_m1_accuracy": full_m1_metrics["accuracy"],
    }
    _set_status(target, "heldout_visual", "passed", details)
    print(json.dumps(details, indent=2, ensure_ascii=False))
def main() -> int:
    load_local_env(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("stage", choices=(
        "freeze", "init-baseline", "feedback", "phase5-report", "finalize-phase5",
        "split-freeze", "split-signatures", "split-cluster",
        "split-propose", "split-evaluate", "split-report",
        "split-multimodal-propose", "split-multimodal-evaluate",
        "split-multimodal-report",
        "split-signature-qwen35-generate", "split-signature-qwen35-report",
        "split-signature-qwen35-cluster", "split-signature-qwen35-propose",
        "split-signature-qwen35-evaluate", "split-signature-qwen35-compare",
        "split-signature-qwen35-heldout-visual",
        "specialize-freeze", "specialize-signatures", "specialize-cluster",
        "specialize-propose", "specialize-evaluate", "specialize-report",
        "split-evolution-repair-audit", "split-evolution-freeze", "split-evolution-smoke", "split-evolution-run", "split-evolution-report",
        "split-evolution-heldout", "split-evolution-final-report",
        "split-memory-freeze", "split-memory-smoke", "split-memory-run",
        "split-memory-report", "split-memory-heldout", "split-memory-final-report",
        "split-retry-visual-freeze", "split-retry-visual-audit",
        "split-retry-visual-run", "split-retry-visual-report",
        "split-retry-v2-freeze", "split-retry-v2-audit",
        "split-retry-v2-run", "split-retry-v2-report",
        "split-retry-v2-heldout-freeze", "split-retry-v2-heldout-run",
        "split-retry-v2-heldout-report",
        "refine-freeze", "refine-smoke", "refine-smoke-heldout",
        "refine-smoke-report", "split-refine-freeze", "split-refine-run",
        "split-refine-report", "split-refine-heldout",
        "split-refine-final-report", "refine-role-freeze",
        "refine-role-audit", "refine-role-run", "refine-role-report",
        "refine-role-heldout", "refine-role-final-report",
        "refine-role-checkpoints-freeze", "refine-role-checkpoints-run",
        "refine-role-checkpoints-report",
        "refine-role-checkpoints-v2-freeze", "refine-role-checkpoints-v2-run",
        "refine-role-checkpoints-v2-report",
        "visual-split-refine-freeze", "visual-split-refine-audit",
        "visual-split-refine-run", "visual-split-refine-report",
        "visual-split-refine-heldout", "visual-split-refine-final-report",
        "five-root-locked-split-refine-freeze",
        "five-root-locked-split-refine-audit",
        "five-root-locked-split-refine-run",
        "five-root-locked-split-refine-report",
        "five-root-locked-split-refine-heldout",
        "five-root-locked-split-refine-final-report",
        "prompt-v2-evolution-freeze", "prompt-v2-evolution-audit",
        "prompt-v2-evolution-smoke", "prompt-v2-evolution-run",
        "prompt-v2-evolution-report", "prompt-v2-evolution-heldout",
        "prompt-v2-evolution-final-report",
        "prompt-v2-evolution-checkpoint-heldout-freeze",
        "prompt-v2-evolution-checkpoint-heldout-audit",
        "prompt-v2-evolution-checkpoint-heldout-smoke",
        "prompt-v2-evolution-checkpoint-heldout-run",
        "prompt-v2-evolution-checkpoint-heldout-report",
        "discovery-v2-evolution-freeze", "discovery-v2-evolution-audit",
        "discovery-v2-evolution-smoke", "discovery-v2-evolution-run",
        "discovery-v2-evolution-report", "discovery-v2-evolution-heldout",
        "discovery-v2-evolution-final-report",
        "root-pre-refine-freeze", "root-pre-refine-baseline",
        "root-pre-refine-audit",
        "root-pre-refine-smoke",
        "root-pre-refine-run", "root-pre-refine-report",
        "root-pre-refine-split-refine-run",
        "root-pre-refine-split-refine-report",
        "root-pre-refine-heldout", "root-pre-refine-final-report",
        "vlrb-freeze", "vlrb-smoke", "vlrb-run", "vlrb-report",
        "vlrb-phase10-freeze", "vlrb-phase10-audit", "vlrb-phase10-smoke",
        "vlrb-phase10-run", "vlrb-phase10-report",
        "vlrb-phase10-capped-freeze", "vlrb-phase10-capped-audit",
        "vlrb-phase10-capped-promote", "vlrb-phase10-capped-smoke",
        "vlrb-phase10-capped-run", "vlrb-phase10-capped-report",
        "vlrb-phase10-capped-format-repair-freeze",
        "vlrb-phase10-capped-format-repair-run",
        "vlrb-phase10-capped-format-repair-report",
        "vlrb-phase10-capped-failure-retry-freeze",
        "vlrb-phase10-capped-failure-retry-run",
        "vlrb-phase10-capped-failure-retry-report",
        "vlrb-phase10-capped-failure-rescue-run",
        "vlrb-phase10-capped-failure-rescue-report",
        "vlrb-phase10-capped-native-fallback-freeze",
        "vlrb-phase10-capped-native-fallback-run",
        "vlrb-phase10-capped-native-fallback-report",
        "vlrb-phase10-capped-native-retry-freeze",
        "vlrb-phase10-capped-native-retry-run",
        "vlrb-phase10-capped-native-retry-report",
        "pairwise-cache-freeze", "pairwise-cache-audit",
        "pairwise-cache-smoke", "pairwise-cache-discovery-run",
        "pairwise-cache-discovery-report", "pairwise-cache-heldout-run",
        "pairwise-cache-final-report", "pairwise-cache-s3-freeze",
        "pairwise-cache-s3-run", "pairwise-cache-s3-report",
        "pairwise-cache-s3-heldout-freeze",
        "pairwise-cache-s3-heldout-run",
        "pairwise-cache-s3-heldout-report",
        "visual-gate-freeze", "visual-gate-smoke",
        "visual-gate-discovery", "visual-gate-report",
        "visual-gate-heldout", "visual-gate-heldout-exploratory",
        "visual-gate-final-report",
        "full-child-gate-freeze", "full-child-gate-smoke",
        "full-child-gate-discovery", "full-child-gate-report",
        "full-child-gate-heldout-exploratory",
        "full-child-gate-final-report",
        "vlrb-prompt-v2-freeze", "vlrb-prompt-v2-audit",
        "vlrb-prompt-v2-smoke", "vlrb-prompt-v2-run",
        "vlrb-prompt-v2-retry", "vlrb-prompt-v2-report",
        "vlrb-prompt-v2-evolved-freeze", "vlrb-prompt-v2-evolved-audit",
        "vlrb-prompt-v2-evolved-smoke", "vlrb-prompt-v2-evolved-run",
        "vlrb-prompt-v2-evolved-retry", "vlrb-prompt-v2-evolved-report",
        "vlrb-discovery-v2-freeze", "vlrb-discovery-v2-audit",
        "vlrb-discovery-v2-smoke", "vlrb-discovery-v2-run",
        "vlrb-discovery-v2-retry", "vlrb-discovery-v2-report",
        "vlrb-qwen25-transfer-freeze", "vlrb-qwen25-transfer-audit",
        "vlrb-qwen25-transfer-smoke", "vlrb-qwen25-transfer-run",
        "vlrb-qwen25-transfer-retry", "vlrb-qwen25-transfer-report",
        "qwen25-evolution-freeze", "qwen25-evolution-audit",
        "qwen25-evolution-smoke", "qwen25-evolution-run",
        "qwen25-evolution-report", "qwen25-evolution-heldout",
        "qwen25-evolution-final-report",
        "vlrb-qwen25-evolved-freeze", "vlrb-qwen25-evolved-audit",
        "vlrb-qwen25-evolved-smoke", "vlrb-qwen25-evolved-run",
        "vlrb-qwen25-evolved-retry", "vlrb-qwen25-evolved-report",
        "vlrb-phase16-checkpoint-freeze", "vlrb-phase16-checkpoint-audit",
        "vlrb-phase16-checkpoint-smoke", "vlrb-phase16-checkpoint-run",
        "vlrb-phase16-checkpoint-retry", "vlrb-phase16-checkpoint-report",
        "vlrb-phase17-checkpoint-freeze", "vlrb-phase17-checkpoint-audit",
        "vlrb-phase17-checkpoint-smoke", "vlrb-phase17-checkpoint-run",
        "vlrb-phase17-checkpoint-retry", "vlrb-phase17-checkpoint-report",
        "vlrb-child-gate-freeze", "vlrb-child-gate-audit",
        "vlrb-child-gate-smoke", "vlrb-child-gate-run",
        "vlrb-child-gate-retry", "vlrb-child-gate-report",
        "vlrb-phase16-router-freeze", "vlrb-phase16-router-audit",
        "vlrb-phase16-router-smoke", "vlrb-phase16-router-run",
        "vlrb-phase16-router-retry", "vlrb-phase16-router-report",
        "discovery-v2-source-lock", "discovery-v2-ingest",
        "discovery-v2-selection-freeze",
        "discovery-v2-generic-screen-smoke",
        "discovery-v2-generic-screen",
        "discovery-v2-adjudication-freeze",
        "discovery-v2-adjudication-smoke",
        "discovery-v2-adjudication-run",
        "discovery-v2-adjudication-report",
        "discovery-v2-review-v2-export",
        "discovery-v2-demo-freeze",
        "discovery-v2-demo-smoke",
        "discovery-v2-demo-adjudicate",
        "discovery-v2-demo-report",
        "discovery-v2-demo-review-export",
        "discovery-v2-demo-finalize",
        "discovery-v2-demo-export",
        "discovery-v2-screen-smoke", "discovery-v2-screen",
        "discovery-v2-adjudicate", "discovery-v2-review-export",
        "discovery-v2-finalize", "discovery-v2-report"))
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
        "split-freeze": lambda: specialize_freeze(config, output, args.parent_node_id),
        "split-signatures": lambda: specialize_signatures(config, output),
        "split-cluster": lambda: specialize_cluster(config, output),
        "split-propose": lambda: specialize_propose(config, output),
        "split-evaluate": lambda: specialize_evaluate(config, output),
        "split-report": lambda: specialize_report(config, output),
        "split-multimodal-propose": lambda: split_multimodal_propose(config, output),
        "split-multimodal-evaluate": lambda: split_multimodal_evaluate(config, output),
        "split-multimodal-report": lambda: split_multimodal_report(config, output),
        "split-signature-qwen35-generate": lambda: split_signature_qwen35_generate(config, output),
        "split-signature-qwen35-report": lambda: split_signature_qwen35_report(config, output),
        "split-signature-qwen35-cluster": lambda: split_signature_qwen35_cluster(config, output),
        "split-signature-qwen35-propose": lambda: split_signature_qwen35_propose(config, output),
        "split-signature-qwen35-evaluate": lambda: split_signature_qwen35_evaluate(config, output),
        "split-signature-qwen35-compare": lambda: split_signature_qwen35_compare(config, output),
        "split-signature-qwen35-heldout-visual": lambda: (
            split_signature_qwen35_heldout_visual(config, output)),
        "specialize-freeze": lambda: specialize_freeze(config, output, args.parent_node_id),
        "specialize-signatures": lambda: specialize_signatures(config, output),
        "specialize-cluster": lambda: specialize_cluster(config, output),
        "specialize-propose": lambda: specialize_propose(config, output),
        "specialize-evaluate": lambda: specialize_evaluate(config, output),
        "specialize-report": lambda: specialize_report(config, output),
    }
    if args.stage.startswith("discovery-v2-evolution-"):
        from .discovery_v2_prompt_v2_evolution import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("discovery-v2-"):
        from .discovery_data_v2 import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith(("split-evolution-", "split-memory-",
                              "split-retry-visual-", "split-retry-v2-")):
        from .split_evolution import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith(("refine-", "split-refine-",
                                "visual-split-refine-",
                                "five-root-locked-split-refine-")):
        from .refine_evolution import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-phase10-capped-"):
        from .vl_rewardbench_phase10_capped import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-phase10-"):
        from .vl_rewardbench_phase10 import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("root-pre-refine-"):
        from .root_boundary_evolution import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("prompt-v2-evolution-"):
        from .prompt_v2_aligned_evolution import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-phase16-router-"):
        from .vl_rewardbench_phase16_root_child_router import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-child-gate-"):
        from .vl_rewardbench_child_gate import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-prompt-v2-evolved-"):
        from .vl_rewardbench_prompt_v2_evolved import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-discovery-v2-"):
        from .vl_rewardbench_discovery_v2 import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("qwen25-evolution-"):
        from .qwen25_full_evolution import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-qwen25-evolved-"):
        from .vl_rewardbench_qwen25_evolved import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-qwen25-transfer-"):
        from .vl_rewardbench_qwen25_transfer import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-phase16-checkpoint-"):
        from .vl_rewardbench_phase16_checkpoints import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-phase17-checkpoint-"):
        from .vl_rewardbench_phase17_checkpoints import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-prompt-v2-"):
        from .vl_rewardbench_prompt_v2 import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("vlrb-"):
        from .vl_rewardbench import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("pairwise-cache-"):
        from .pairwise_cache_ablation import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("visual-gate-"):
        from .visual_gate import run_stage
        run_stage(config, output, args.stage)
    elif args.stage.startswith("full-child-gate-"):
        from .full_child_gate import run_stage
        run_stage(config, output, args.stage)
    else:
        actions[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

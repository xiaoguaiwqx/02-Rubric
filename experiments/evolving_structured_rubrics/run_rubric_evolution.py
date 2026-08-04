"""Phase 5：Multi-Crit Init Rubric、baseline 与反馈校准入口。

PowerShell 运行顺序：

    $module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
    $config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5.json"
    $output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

    python -m $module --config $config --output-dir $output freeze
    python -m $module --config $config --output-dir $output init-baseline
    python -m $module --config $config --output-dir $output feedback
    python -m $module --config $config --output-dir $output phase5-report
    python -m $module --config $config --output-dir $output finalize-phase5

``init-baseline`` 在线生成 discovery-90 与 heldout-500 的五节点 Pairwise
artifacts，但 gold answer 不进入模型请求；accuracy 只由随后同一阶段的 offline
B1/M1 replay 计算。中间 evolution stages 尚未实现，也不能读取 heldout-500。
"""

from __future__ import annotations

import argparse
import json
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
    extract_rubric_feedback,
)
from critiq.structured.aggregation import aggregate_flat_votes

from .experiment_utils import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    load_jsonl_dataset,
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


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    fields = {
        "experiment_id", "model", "vllm_version", "max_model_len",
        "discovery_dataset", "discovery_dataset_sha256", "heldout_dataset",
        "heldout_dataset_sha256", "backend_pool", "worker_request_kwargs",
        "structured_max_retries", "api_retry_attempts", "evolution_policy",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Phase 5 config fields mismatch: {sorted(set(value) ^ fields)}")
    BackendPoolSpec.from_dict(value["backend_pool"])
    request = value["worker_request_kwargs"]
    if not isinstance(request, dict) or request.get("temperature") != 0.5:
        raise ValueError("Pairwise P05 requires an explicit temperature=0.5")
    if value["evolution_policy"] != PHASE5_EVOLUTION_POLICY_V2:
        raise ValueError("Phase 5 evolution policy does not match frozen phase5-v2")
    CandidateAcceptancePolicy(**value["evolution_policy"]["candidate_acceptance"])
    return value


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


def _inspect_endpoints(config: Mapping[str, Any], spec: BackendPoolSpec) -> list[dict[str, Any]]:
    identities = []
    for endpoint in spec.endpoints:
        base = endpoint.base_url.rstrip("/")
        if not base.endswith("/v1"):
            raise ValueError("endpoint base_url must end with /v1")
        version = request_json(base[:-3] + "/version")
        models = request_json(base + "/models")
        matches = [item for item in models.get("data", []) if item.get("id") == config["model"]]
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
            "vllm_version": config["vllm_version"],
            "model": config["model"],
            "checkpoint_root": endpoint.checkpoint_root,
            "max_model_len": config["max_model_len"],
        }
        if actual != expected:
            raise ValueError(f"endpoint identity mismatch: expected={expected}, actual={actual}")
        identities.append(actual)
    return identities


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
        "config_sha256": canonical_sha256(config),
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
    if manifest["config_sha256"] != canonical_sha256(config):
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


def _generate_pairwise(config, output, rubric, rows, label):
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
    if valid / total < 0.95:
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
    current_hash = canonical_sha256(config)
    legacy_config = dict(config)
    legacy_config.pop("evolution_policy")
    v1_config = dict(config)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("stage", choices=("freeze", "init-baseline", "feedback", "phase5-report", "finalize-phase5"))
    args = parser.parse_args()
    config = _config(args.config.resolve())
    output = args.output_dir.resolve()
    actions = {
        "freeze": lambda: freeze(config, output),
        "init-baseline": lambda: init_baseline(config, output),
        "feedback": lambda: feedback(config, output),
        "phase5-report": lambda: phase5_report(config, output),
        "finalize-phase5": lambda: finalize_phase5(config, output),
    }
    actions[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

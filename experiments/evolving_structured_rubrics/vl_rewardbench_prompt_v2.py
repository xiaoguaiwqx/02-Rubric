"""Prompt-v2 external transfer evaluation on VL-RewardBench.

The experiment reuses the frozen Phase10 rubric, VL-RewardBench records, and
K=3 order schedule, but regenerates all 22 criterion predictions with the
cache-oriented Pairwise Worker prompt.  Initial-root, visual-only, weighted,
and equal-root systems are derived offline from the same predictions.  The
native judge and Prompt-v1 predictions are read-only historical references.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    CacheMode,
    FinalPreference,
    JsonPredictionCache,
    ModelCallMetrics,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    StructuredRubric,
    aggregate_flat_votes,
    combine_model_call_metrics,
)
from critiq.structured.version import PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION

from . import pairwise_cache_ablation as cache_exp
from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from . import vl_rewardbench_phase10 as phase10
from .experiment_utils import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    make_progress_callback,
)
from .rubric_factory import build_multicrit_open_ended_init_rubric, file_sha256


EXPERIMENT_DIR = "vl_rewardbench_phase10_prompt_v2_transfer_v1"
PROTOCOL_VERSION = "vlrb-phase10-prompt-v2-transfer-v1"
SOURCE_EXPERIMENT = "phase10_five_root_locked_split_refine_v1"
SOURCE_RUBRIC_SHA256 = (
    "17ad7a0b6350338b100384b701303008746463911662f4a628b90c4632cdf7ef"
)
SOURCE_NODE_COUNT = 22
SOURCE_V1_EXPERIMENT = "vl_rewardbench_phase10_transfer_v2_max2048"
SOURCE_V1_LOGICAL = "native_retry_max10/combined/logical_votes.json"
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")
SMOKE_COUNT = 20
MAX_RETRY_ATTEMPTS = 10

INITIAL_SYSTEM = phase10.INITIAL_SYSTEM
EQUAL_SYSTEM = phase10.EQUAL_SYSTEM
WEIGHTED_SYSTEM = phase10.WEIGHTED_SYSTEM
VISUAL_SYSTEM = phase10.VISUAL_SYSTEM
NATIVE_SYSTEM = phase10.NATIVE_SYSTEM

STAGE_FREEZE = "vlrb-prompt-v2-freeze"
STAGE_AUDIT = "vlrb-prompt-v2-audit"
STAGE_SMOKE = "vlrb-prompt-v2-smoke"
STAGE_RUN = "vlrb-prompt-v2-run"
STAGE_RETRY = "vlrb-prompt-v2-retry"
STAGE_REPORT = "vlrb-prompt-v2-report"


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _source_target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / SOURCE_V1_EXPERIMENT


def _source_rubric_path(output: Path) -> Path:
    return (output / SOURCE_EXPERIMENT / "final/rubric.json").resolve()


def _source_logical_path() -> Path:
    return _source_target() / SOURCE_V1_LOGICAL


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
        "source_v1_experiment": SOURCE_V1_EXPERIMENT,
        "source_v1_logical": SOURCE_V1_LOGICAL,
        "endpoint_ids": list(ENDPOINT_IDS),
        "scheduler": "available_slot_dynamic",
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "smoke_sample_count": SMOKE_COUNT,
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "k": legacy.K,
        "seed": legacy.SEED,
        "selection_after_benchmark_forbidden": True,
    }
    value = config.get("vlrb_prompt_v2_transfer")
    if value != expected:
        raise RuntimeError("vlrb_prompt_v2_transfer must match the frozen v1 protocol")
    request = config.get("worker_request_kwargs")
    if (not isinstance(request, dict)
            or request.get("temperature") != 0.5
            or request.get("max_tokens") != 2048):
        raise RuntimeError("VL-RewardBench Prompt v2 freezes temperature=0.5/max_tokens=2048")
    return dict(value)


def _pool_spec(config: Mapping[str, Any]):
    spec = phase10._pool_spec(config)
    if spec.global_request_concurrency != sum(
            endpoint.max_concurrency for endpoint in spec.endpoints):
        raise RuntimeError("Prompt v2 requires a fully available-slot dynamic pool")
    return spec


def _rubric(output: Path) -> StructuredRubric:
    path = _source_rubric_path(output)
    if not path.is_file():
        raise RuntimeError(f"Phase10 source rubric is missing: {path}")
    rubric = StructuredRubric.load_json(path)
    if (rubric.rubric_sha256 != SOURCE_RUBRIC_SHA256
            or len(rubric.nodes) != SOURCE_NODE_COUNT):
        raise RuntimeError("Phase10 source rubric hash or node count drift")
    return rubric


def _records() -> tuple[dict[str, Any], ...]:
    """Reuse the previously materialized images and exact record identity."""

    if not _source_target().is_dir():
        raise RuntimeError(f"Prompt-v1 source experiment is missing: {_source_target()}")
    return legacy._read_records(_source_target())


def _v2_evaluator(config: Mapping[str, Any], rows, pool):
    return cache_exp._make_evaluator(
        config, rows, pool, cache_exp.VARIANT_PROMPT_V2_DYNAMIC)


def _v2_request_spec(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                     schedule: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    rows = legacy._ordered_rows(records[:1], schedule, 0)
    model_rows = base._model_rows(rows)
    pool = AvailableSlotBackendPool(_pool_spec(config))
    evaluator = _v2_evaluator(config, model_rows, pool)
    return evaluator.request_spec().to_dict()


def _validate_source_logical(records: Sequence[Mapping[str, Any]],
                             logical: Mapping[str, Any]) -> None:
    expected_ids = [str(record["sample_id"]) for record in records]
    if logical.get("sample_ids") != expected_ids or logical.get("k") != legacy.K:
        raise RuntimeError("Prompt-v1 logical vote source identity drift")
    required = {NATIVE_SYSTEM, INITIAL_SYSTEM, EQUAL_SYSTEM, WEIGHTED_SYSTEM, VISUAL_SYSTEM}
    if not required.issubset(logical.get("systems", {})):
        raise RuntimeError("Prompt-v1 logical vote source is incomplete")


def _manifest(config: Mapping[str, Any], output: Path,
              records: Sequence[Mapping[str, Any]],
              schedule: Mapping[str, Sequence[int]], *,
              include_endpoints: bool) -> dict[str, Any]:
    protocol = _protocol(config)
    final = _rubric(output)
    initial = build_multicrit_open_ended_init_rubric()
    reuse = phase10._root_reuse_contract(initial, final)
    source_manifest_path = _source_target() / "frozen_manifest.json"
    source_logical_path = _source_logical_path()
    if not source_manifest_path.is_file() or not source_logical_path.is_file():
        raise RuntimeError("completed Prompt-v1 VL-RewardBench artifacts are required")
    source_manifest = load_json(source_manifest_path)
    source_logical = load_json(source_logical_path)
    _validate_source_logical(records, source_logical)
    dataset = source_manifest.get("dataset")
    if (not isinstance(dataset, dict)
            or dataset.get("count") != legacy.EXPECTED_COUNT
            or dataset.get("record_sha256") != canonical_sha256(list(records))):
        raise RuntimeError("Prompt-v1 frozen dataset identity drift")
    request_spec = _v2_request_spec(config, records, schedule)
    heldout_manifest_path = output / cache_exp.EXPERIMENT_DIR / "s3_heldout_manifest.json"
    if not heldout_manifest_path.is_file():
        raise RuntimeError("completed heldout Prompt-v2 ablation manifest is required")
    heldout_manifest = load_json(heldout_manifest_path)
    heldout_spec = heldout_manifest["request_specs"][
        cache_exp.VARIANT_PROMPT_V2_DYNAMIC]
    if request_spec != heldout_spec:
        raise RuntimeError("Prompt v2 request identity differs from heldout ablation")
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol": protocol,
        "exploratory": True,
        "dataset": dict(dataset),
        "order_schedule_sha256": canonical_sha256(schedule),
        "counterbalance": {
            "k": legacy.K,
            "seed": legacy.SEED,
            "protocol": "balanced_b_1minusb_b",
        },
        "structured_worker_request_spec": request_spec,
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "source": {
            "phase10_rubric_path": str(_source_rubric_path(output)),
            "phase10_rubric_file_sha256": file_sha256(_source_rubric_path(output)),
            "phase10_rubric_sha256": final.rubric_sha256,
            "prompt_v1_manifest_path": str(source_manifest_path),
            "prompt_v1_manifest_sha256": file_sha256(source_manifest_path),
            "prompt_v1_logical_path": str(source_logical_path),
            "prompt_v1_logical_sha256": file_sha256(source_logical_path),
            "heldout_prompt_v2_manifest_path": str(heldout_manifest_path),
            "heldout_prompt_v2_manifest_sha256": file_sha256(heldout_manifest_path),
        },
        "systems": {
            INITIAL_SYSTEM: {
                "node_count": len(initial.nodes),
                "root_weights": {root_id: 0.2 for root_id in initial.root_ids},
            },
            EQUAL_SYSTEM: {
                "node_count": len(final.nodes),
                "root_weights": {root_id: 0.2 for root_id in final.root_ids},
            },
            WEIGHTED_SYSTEM: {
                "prediction_source": EQUAL_SYSTEM,
                "root_weights": dict(phase10.ROOT_WEIGHTS),
            },
            VISUAL_SYSTEM: {
                "prediction_source": EQUAL_SYSTEM,
                "root_id": phase10.VISUAL_ROOT_ID,
            },
        },
        "root_prediction_reuse": reuse,
        "endpoint_pool": _pool_spec(config).to_dict(),
        "selection_after_benchmark_forbidden": True,
    }
    if include_endpoints:
        value["endpoint_identities"] = phase10._inspect_endpoints(config)
    return value


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target()
    _require(target, STAGE_FREEZE)
    records = _records()
    schedule = legacy._order_schedule(records)
    expected = _manifest(config, output, records, schedule, include_endpoints=False)
    stored = load_json(target / "frozen_manifest.json")
    stored_static = {key: value for key, value in stored.items()
                     if key != "endpoint_identities"}
    if expected != stored_static:
        raise RuntimeError("VL-RewardBench Prompt v2 frozen manifest drift")
    return target, stored, records, schedule, _rubric(output)


def _verify_live_endpoints(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if manifest.get("endpoint_identities") != phase10._inspect_endpoints(config):
        raise RuntimeError("VL-RewardBench Prompt v2 live endpoint identity drift")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target()
    records = _records()
    schedule = legacy._order_schedule(records)
    manifest = _manifest(config, output, records, schedule, include_endpoints=True)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        downstream = (STAGE_SMOKE, STAGE_RUN, STAGE_RETRY, STAGE_REPORT)
        if any(status.get(stage, {}).get("status") == "passed" for stage in downstream):
            raise RuntimeError("Prompt v2 manifest drift after inference")
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    details = {
        "dataset_count": len(records),
        "rubric_sha256": manifest["source"]["phase10_rubric_sha256"],
        "node_count": SOURCE_NODE_COUNT,
        "prompt_version": manifest["prompt_version"],
        "logical_request_count": len(records) * legacy.K * SOURCE_NODE_COUNT,
    }
    _status(target, STAGE_FREEZE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, final = _load_frozen(config, output)
    initial = build_multicrit_open_ended_init_rubric()
    reuse = phase10._root_reuse_contract(initial, final)
    source_logical = load_json(_source_logical_path())
    _validate_source_logical(records, source_logical)
    schedule_values = list(schedule.values())
    value = {
        "schema_version": "1.0.0",
        "status": "passed",
        "dataset_count": len(records),
        "k": legacy.K,
        "source_node_count": len(final.nodes),
        "initial_root_reuse_count": len(reuse),
        "new_logical_request_count": len(records) * legacy.K * len(final.nodes),
        "native_new_request_count": 0,
        "offline_system_count": 4,
        "schedule_aba_count": sum(tuple(item) == (0, 1, 0) for item in schedule_values),
        "schedule_bab_count": sum(tuple(item) == (1, 0, 1) for item in schedule_values),
        "prompt_v1_cache_reuse_allowed": False,
        "benchmark_labels_in_model_prompt": False,
        "selection_after_benchmark_forbidden": True,
        "endpoint_identity_count": len(manifest["endpoint_identities"]),
    }
    if abs(value["schedule_aba_count"] - value["schedule_bab_count"]) > 1:
        raise RuntimeError("K=3 order schedule is not globally balanced")
    atomic_write_json(target / "offline_audit.json", value)
    _status(target, STAGE_AUDIT, value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _validate_prediction(prediction: PairwisePredictionOutput,
                         rubric: StructuredRubric,
                         rows: Sequence[Mapping[str, Any]],
                         manifest: Mapping[str, Any]) -> None:
    cache_exp._validate_prediction(
        prediction, rows, rubric,
        manifest["structured_worker_request_spec"],
        PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    )


def _prediction_path(work: Path, replicate: int) -> Path:
    return (work / "structured/prompt_v2"
            / f"replicate_{replicate + 1:02d}/predictions.json")


def _structured_prediction(config: Mapping[str, Any], work: Path,
                           manifest: Mapping[str, Any], rubric: StructuredRubric,
                           rows: Sequence[Mapping[str, Any]], replicate: int):
    run_dir = _prediction_path(work, replicate).parent
    artifact = _prediction_path(work, replicate)
    if artifact.exists():
        prediction = PairwisePredictionOutput.load_json(artifact)
        _validate_prediction(prediction, rubric, rows, manifest)
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Prompt v2 prediction exists without run summary")
        return prediction, load_json(summary_path)

    model_rows = base._model_rows(rows)
    pool = AvailableSlotBackendPool(_pool_spec(config))
    evaluator = _v2_evaluator(config, model_rows, pool)
    if evaluator.request_spec().to_dict() != manifest["structured_worker_request_spec"]:
        raise RuntimeError("Prompt v2 evaluator request identity drift")
    cache = JsonPredictionCache(run_dir / "cache", CacheMode.READ_WRITE)
    callback = make_progress_callback(
        run_dir, f"prompt_v2_replicate_{replicate + 1:02d}",
        len(rows) * len(rubric.nodes), pool)
    resumed = (run_dir / "cache").exists() and any((run_dir / "cache").rglob("*.json"))
    started = time.perf_counter()
    prediction = base._pairwise_cached(
        evaluator, model_rows, rubric, cache, None,
        evaluation_callback=callback)
    if prediction.prompt_version != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION:
        prediction = cache_exp._with_prompt_version(
            prediction, PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION)
    wall_seconds = time.perf_counter() - started
    artifact.parent.mkdir(parents=True, exist_ok=True)
    prediction.save_json(artifact)
    provenance = pool.provenance_dict()
    atomic_write_json(run_dir / "provenance.json", provenance)
    summary = {
        "schema_version": "1.0.0",
        "replicate": replicate + 1,
        "sample_count": len(rows),
        "node_count": len(rubric.nodes),
        "prompt_version": prediction.prompt_version,
        "request_spec": evaluator.request_spec().to_dict(),
        "prediction_sha256": file_sha256(artifact),
        "valid_rate": cache_exp._prediction_valid_rate(prediction),
        "resumed_from_local_cache": resumed,
        "telemetry": cache_exp._telemetry(
            provenance, wall_seconds, len(rows) * len(rubric.nodes), None),
    }
    atomic_write_json(run_dir / "run_summary.json", summary)
    if summary["valid_rate"] < 0.95:
        raise RuntimeError("Prompt v2 structured valid rate below 95%")
    _validate_prediction(prediction, rubric, rows, manifest)
    return prediction, summary


def _logical_from_predictions(records: Sequence[Mapping[str, Any]],
                              schedule: Mapping[str, Sequence[int]],
                              final: StructuredRubric,
                              predictions: Sequence[PairwisePredictionOutput]):
    if len(predictions) != legacy.K:
        raise RuntimeError("Prompt v2 requires exactly K=3 prediction artifacts")
    initial = build_multicrit_open_ended_init_rubric()
    votes = {name: [] for name in (
        INITIAL_SYSTEM, EQUAL_SYSTEM, WEIGHTED_SYSTEM, VISUAL_SYSTEM)}
    root_votes = {root_id: [] for root_id in final.root_ids}
    for replicate, prediction in enumerate(predictions):
        rows = legacy._ordered_rows(records, schedule, replicate)
        execution, equal_answers = base.execute_offline_m1(final, prediction, rows)
        initial_prediction = phase10._subset_prediction(prediction, initial)
        _, initial_answers = base.execute_offline_m1(initial, initial_prediction, rows)
        weighted_answers = phase10._weighted_answers(execution, phase10.ROOT_WEIGHTS)
        orders = [int(schedule[str(row["sample_id"])][replicate]) for row in rows]
        for name, answers in (
                (INITIAL_SYSTEM, initial_answers),
                (EQUAL_SYSTEM, equal_answers),
                (WEIGHTED_SYSTEM, weighted_answers)):
            votes[name].append([
                legacy._original_index(answer.value, order)
                for answer, order in zip(answers, orders)
            ])
        for root_id in final.root_ids:
            display_votes = []
            for trace in execution.traces:
                matches = [root for root in trace.roots if root.root_id == root_id]
                if len(matches) != 1 or matches[0].subtree_vote is None:
                    raise RuntimeError(f"Prompt v2 root trace is incomplete: {root_id}")
                display_votes.append(matches[0].subtree_vote.value)
            root_votes[root_id].append([
                legacy._original_index(value, order)
                for value, order in zip(display_votes, orders)
            ])
        votes[VISUAL_SYSTEM].append(root_votes[phase10.VISUAL_ROOT_ID][-1])
    return {
        "schema_version": "1.0.0",
        "sample_ids": [str(record["sample_id"]) for record in records],
        "k": legacy.K,
        "systems": {name: {"votes_by_replicate": item}
                    for name, item in votes.items()},
        "root_votes_by_replicate": root_votes,
        "prediction_reuse": {
            "all_four_systems_share_prompt_v2_predictions": True,
            "initial_roots_projected_from_phase10": True,
            "new_model_calls_for_offline_variants": 0,
        },
    }


def _run_predictions(config: Mapping[str, Any], work: Path,
                     manifest: Mapping[str, Any], records,
                     schedule, final):
    predictions = []
    summaries = []
    for replicate in range(legacy.K):
        rows = legacy._ordered_rows(records, schedule, replicate)
        prediction, summary = _structured_prediction(
            config, work, manifest, final, rows, replicate)
        predictions.append(prediction)
        summaries.append(summary)
    logical = _logical_from_predictions(records, schedule, final, predictions)
    atomic_write_json(work / "combined/logical_votes.json", logical)
    atomic_write_json(work / "run_summaries.json", {
        "schema_version": "1.0.0", "items": summaries})
    return logical, tuple(predictions), tuple(summaries)


def _smoke_system_metrics(records: Sequence[Mapping[str, Any]],
                          votes_by_replicate: Sequence[Sequence[int | None]]):
    """Validate a small shard without assuming all official groups occur."""

    if (len(votes_by_replicate) != legacy.K
            or any(len(votes) != len(records) for votes in votes_by_replicate)):
        raise RuntimeError("Prompt v2 smoke K=3 vote matrix shape drift")
    decisions = [legacy._majority([
        votes[index] for votes in votes_by_replicate
    ]) for index in range(len(records))]
    covered = sum(vote is not None for vote in decisions)
    correct = sum(
        vote == int(record["preferred_original_index"])
        for vote, record in zip(decisions, records)
    )
    return {
        "sample_count": len(records),
        "correct_count": correct,
        "strict_accuracy": correct / len(records) if records else 0.0,
        "coverage": covered / len(records) if records else 0.0,
    }


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, final = _load_frozen(config, output)
    _require(target, STAGE_AUDIT)
    _verify_live_endpoints(config, manifest)
    selected = tuple(records[:SMOKE_COUNT])
    logical, _, summaries = _run_predictions(
        config, target / "smoke", manifest, selected, schedule, final)
    calls = {
        endpoint_id: sum(
            int(summary["telemetry"]["endpoint_call_counts"].get(endpoint_id, 0))
            for summary in summaries)
        for endpoint_id in ENDPOINT_IDS
    }
    valid_rate = sum(summary["valid_rate"] for summary in summaries) / len(summaries)
    if valid_rate < 0.99 or any(count == 0 for count in calls.values()):
        raise RuntimeError("Prompt v2 smoke validity or dual-endpoint use failed")
    smoke_metrics = {
        name: _smoke_system_metrics(selected, item["votes_by_replicate"])
        for name, item in logical["systems"].items()
    }
    details = {
        "sample_count": len(selected),
        "logical_request_count": len(selected) * legacy.K * len(final.nodes),
        "valid_rate": valid_rate,
        "endpoint_call_counts": calls,
        "offline_system_count": len(logical["systems"]),
        "offline_system_metrics": smoke_metrics,
    }
    _status(target, STAGE_SMOKE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, final = _load_frozen(config, output)
    _require(target, STAGE_SMOKE)
    _verify_live_endpoints(config, manifest)
    _, _, summaries = _run_predictions(
        config, target / "run", manifest, records, schedule, final)
    details = {
        "sample_count": len(records),
        "node_count": len(final.nodes),
        "logical_request_count": len(records) * legacy.K * len(final.nodes),
        "valid_rates": [item["valid_rate"] for item in summaries],
        "native_new_request_count": 0,
    }
    _status(target, STAGE_RUN, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _retry_key(replicate: int, sample_index: int, criterion_name: str) -> str:
    return f"replicate_{replicate + 1:02d}::{sample_index}::{criterion_name}"


def _failure_items(predictions: Sequence[PairwisePredictionOutput]):
    items = []
    for replicate, prediction in enumerate(predictions):
        for sample_index, (sample_id, row) in enumerate(zip(
                prediction.sample_ids, prediction.node_outputs)):
            for criterion_name, output in row.items():
                if output.parse_ok and output.answer_valid:
                    continue
                items.append({
                    "key": _retry_key(replicate, sample_index, criterion_name),
                    "replicate": replicate,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "criterion_name": criterion_name,
                    "source_output_sha256": canonical_sha256(output.to_dict()),
                    "source_parse_error": output.parse_error,
                })
    return items


def _replace_prediction_outputs(
        prediction: PairwisePredictionOutput,
        replacements: Mapping[tuple[int, str], Any]) -> PairwisePredictionOutput:
    rows = []
    for sample_index, row in enumerate(prediction.node_outputs):
        updated = dict(row)
        for criterion_name in tuple(updated):
            replacement = replacements.get((sample_index, criterion_name))
            if replacement is not None:
                updated[criterion_name] = replacement
        rows.append(updated)
    answers = tuple(aggregate_flat_votes(output.vote for output in row.values())
                    for row in rows)
    return PairwisePredictionOutput(
        prediction.sample_ids,
        prediction.sample_fingerprints,
        prediction.criteria,
        tuple(rows),
        answers,
        prediction.request_spec,
        semantics_version=prediction.semantics_version,
        schema_version=prediction.schema_version,
        prompt_version=prediction.prompt_version,
        parser_version=prediction.parser_version,
    )


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, final = _load_frozen(config, output)
    _require(target, STAGE_RUN)
    _verify_live_endpoints(config, manifest)
    source_predictions = tuple(
        PairwisePredictionOutput.load_json(_prediction_path(target / "run", replicate))
        for replicate in range(legacy.K)
    )
    for replicate, prediction in enumerate(source_predictions):
        _validate_prediction(
            prediction, final,
            legacy._ordered_rows(records, schedule, replicate), manifest)
    items = _failure_items(source_predictions)
    work = target / "retry"
    atomic_write_json(work / "failure_manifest.json", {
        "schema_version": "1.0.0",
        "source_prediction_sha256": [
            file_sha256(_prediction_path(target / "run", replicate))
            for replicate in range(legacy.K)
        ],
        "target_count": len(items),
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "selection_uses_gold": False,
        "items": items,
    })
    pool = AvailableSlotBackendPool(_pool_spec(config))
    ordered_rows = {
        replicate: legacy._ordered_rows(records, schedule, replicate)
        for replicate in range(legacy.K)
    }
    model_rows = {replicate: base._model_rows(rows)
                  for replicate, rows in ordered_rows.items()}
    retry_config = dict(config)
    retry_config["structured_max_retries"] = 0
    evaluators = {replicate: _v2_evaluator(
        retry_config, model_rows[replicate], pool)
                  for replicate in range(legacy.K)}
    if any(evaluator.request_spec().to_dict()
           != manifest["structured_worker_request_spec"]
           for evaluator in evaluators.values()):
        raise RuntimeError("Prompt v2 retry request identity drift")
    criteria = {
        final.get_node(node_id).criterion.name: final.get_node(node_id).criterion
        for node_id in final.preorder_node_ids()
    }
    callback = make_progress_callback(work, "prompt_v2_technical_retry", len(items), pool)

    def one(index: int, item: Mapping[str, Any]):
        cache_path = work / "attempts" / f"{canonical_sha256(item['key'])}.json"
        state = load_json(cache_path) if cache_path.exists() else {
            "schema_version": "1.0.0",
            "key": item["key"],
            "source_output_sha256": item["source_output_sha256"],
            "attempts": [],
            "complete": False,
            "success": False,
            "final_output": None,
        }
        if (state.get("key") != item["key"]
                or state.get("source_output_sha256") != item["source_output_sha256"]):
            raise RuntimeError(f"Prompt v2 retry cache identity drift: {item['key']}")
        replicate = int(item["replicate"])
        sample_index = int(item["sample_index"])
        criterion_name = str(item["criterion_name"])
        if not state["complete"]:
            for attempt_number in range(
                    len(state["attempts"]) + 1, MAX_RETRY_ATTEMPTS + 1):
                candidate, metrics = evaluators[replicate].infer_one(
                    model_rows[replicate][sample_index],
                    criteria[criterion_name].to_criterion())
                state["attempts"].append({
                    "attempt_number": attempt_number,
                    "output": candidate.to_dict(),
                    "metrics": metrics.to_dict(),
                })
                if candidate.parse_ok and candidate.answer_valid:
                    state["success"] = True
                    state["complete"] = True
                    state["final_output"] = candidate.to_dict()
                elif attempt_number == MAX_RETRY_ATTEMPTS:
                    state["complete"] = True
                atomic_write_json(cache_path, state)
                if state["success"]:
                    break
        metrics = combine_model_call_metrics(
            ModelCallMetrics.from_dict(item["metrics"])
            for item in state["attempts"])
        return index, dict(item), state, metrics

    results: list[Any | None] = [None] * len(items)
    if items:
        with ThreadPoolExecutor(max_workers=_pool_spec(config).global_request_concurrency) as executor:
            futures = {executor.submit(one, index, item): index
                       for index, item in enumerate(items)}
            for future in as_completed(futures):
                index, item, state, metrics = future.result()
                results[index] = (item, state)
                callback(index, str(item["sample_id"]), metrics)
    completed = [item for item in results if item is not None]
    if len(completed) != len(items):
        raise RuntimeError("Prompt v2 retry result matrix is incomplete")
    by_replicate: dict[int, dict[tuple[int, str], Any]] = {
        replicate: {} for replicate in range(legacy.K)}
    report_items = []
    for item, state in completed:
        output_value = (PairwiseVoteOutput.from_dict(state["final_output"])
                        if state["success"] else None)
        if output_value is not None:
            by_replicate[int(item["replicate"])][(
                int(item["sample_index"]), str(item["criterion_name"]))] = output_value
        report_items.append({
            **item,
            "success": bool(state["success"]),
            "attempt_count": len(state["attempts"]),
            "final_output": state["final_output"],
        })
    repaired_predictions = []
    for replicate, source in enumerate(source_predictions):
        repaired = _replace_prediction_outputs(source, by_replicate[replicate])
        path = _prediction_path(work, replicate)
        path.parent.mkdir(parents=True, exist_ok=True)
        repaired.save_json(path)
        _validate_prediction(repaired, final, ordered_rows[replicate], manifest)
        repaired_predictions.append(repaired)
    logical = _logical_from_predictions(records, schedule, final, repaired_predictions)
    atomic_write_json(work / "combined/logical_votes.json", logical)
    atomic_write_json(work / "retry_items.json", {
        "schema_version": "1.0.0", "items": report_items})
    atomic_write_json(work / "provenance.json", pool.provenance_dict())
    recovered = sum(item["success"] for item in report_items)
    details = {
        "target_count": len(items),
        "recovered_count": recovered,
        "still_failed_count": len(items) - recovered,
        "new_model_requests": sum(item["attempt_count"] for item in report_items),
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
    }
    atomic_write_json(work / "report.json", details)
    _status(target, STAGE_RETRY, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _node_metrics(records, schedule, final, predictions):
    result = {}
    for node_id in final.preorder_node_ids():
        name = final.get_node(node_id).criterion.name
        votes = []
        for replicate, prediction in enumerate(predictions):
            orders = [int(schedule[str(record["sample_id"])][replicate])
                      for record in records]
            votes.append([
                legacy._original_index(row[name].vote.value, order)
                for row, order in zip(prediction.node_outputs, orders)
            ])
        result[name] = phase10._system_metrics(records, votes)
    return result


def _position_metrics(records, schedule, votes_by_replicate):
    counts = {"A": 0, "B": 0, "None": 0}
    by_gold = {"A": {"correct": 0, "total": 0},
               "B": {"correct": 0, "total": 0}}
    all_valid = 0
    all_agree = 0
    for index, record in enumerate(records):
        normalized = [votes[index] for votes in votes_by_replicate]
        valid = [vote for vote in normalized if vote is not None]
        if len(valid) == legacy.K:
            all_valid += 1
            all_agree += len(set(valid)) == 1
        for replicate, vote in enumerate(normalized):
            order = int(schedule[str(record["sample_id"])][replicate])
            gold_display = ("A" if int(record["preferred_original_index"]) == order
                            else "B")
            by_gold[gold_display]["total"] += 1
            if vote is None:
                counts["None"] += 1
                continue
            display = "A" if vote == order else "B"
            counts[display] += 1
            by_gold[gold_display]["correct"] += display == gold_display
    for value in by_gold.values():
        value["accuracy"] = value["correct"] / value["total"]
    return {
        "display_prediction_counts": counts,
        "by_gold_display": by_gold,
        "all_three_valid_count": all_valid,
        "all_three_original_vote_agreement_count": all_agree,
        "all_three_original_vote_agreement_rate": (
            all_agree / all_valid if all_valid else 0.0),
    }


def _execution_summary(target: Path) -> dict[str, Any]:
    summaries = load_json(target / "run/run_summaries.json")["items"]
    total_calls = sum(item["telemetry"]["recorded_model_call_count"]
                      for item in summaries)
    total_wall = sum(item["telemetry"]["wall_seconds"] for item in summaries)
    return {
        "logical_request_count": legacy.EXPECTED_COUNT * legacy.K * SOURCE_NODE_COUNT,
        "recorded_model_call_count": total_calls,
        "wall_seconds_sum_across_replicates": total_wall,
        "inferences_per_minute": (
            legacy.EXPECTED_COUNT * legacy.K * SOURCE_NODE_COUNT / total_wall * 60
            if total_wall else 0.0),
        "fresh_timing_valid": not any(
            item.get("resumed_from_local_cache") for item in summaries),
        "endpoint_call_counts": {
            endpoint_id: sum(item["telemetry"]["endpoint_call_counts"].get(
                endpoint_id, 0) for item in summaries)
            for endpoint_id in ENDPOINT_IDS
        },
        "api_attempts": sum(item["telemetry"]["api_attempts"] for item in summaries),
        "errors": sum(item["telemetry"]["errors"] for item in summaries),
        "input_tokens": sum(item["telemetry"]["input_tokens"] for item in summaries),
        "output_tokens": sum(item["telemetry"]["output_tokens"] for item in summaries),
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, final = _load_frozen(config, output)
    _require(target, STAGE_RETRY)
    logical = load_json(target / "retry/combined/logical_votes.json")
    if logical.get("sample_ids") != [str(record["sample_id"]) for record in records]:
        raise RuntimeError("Prompt v2 report sample order drift")
    old_logical = load_json(_source_logical_path())
    _validate_source_logical(records, old_logical)
    new_metrics = {
        name: phase10._system_metrics(records, item["votes_by_replicate"])
        for name, item in logical["systems"].items()
    }
    old_metrics = {
        name: phase10._system_metrics(records, item["votes_by_replicate"])
        for name, item in old_logical["systems"].items()
    }
    new_predictions = {
        name: value["original_index_predictions"]
        for name, value in new_metrics.items()
    }
    old_predictions = {
        name: value["original_index_predictions"]
        for name, value in old_metrics.items()
    }
    paired = {
        "initial_v2_to_final_equal_v2": legacy._paired(
            records, new_predictions[INITIAL_SYSTEM], new_predictions[EQUAL_SYSTEM]),
        "final_equal_v1_to_final_equal_v2": legacy._paired(
            records, old_predictions[EQUAL_SYSTEM], new_predictions[EQUAL_SYSTEM]),
        "initial_v1_to_initial_v2": legacy._paired(
            records, old_predictions[INITIAL_SYSTEM], new_predictions[INITIAL_SYSTEM]),
        "final_equal_v2_to_weighted_v2": legacy._paired(
            records, new_predictions[EQUAL_SYSTEM], new_predictions[WEIGHTED_SYSTEM]),
    }
    root_metrics = {
        root_id: phase10._system_metrics(records, votes)
        for root_id, votes in logical["root_votes_by_replicate"].items()
    }
    predictions = tuple(
        PairwisePredictionOutput.load_json(_prediction_path(target / "retry", replicate))
        for replicate in range(legacy.K)
    )
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "frozen_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
        "selection_after_benchmark_forbidden": True,
        "phase10_rubric_sha256": SOURCE_RUBRIC_SHA256,
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "new_metrics": new_metrics,
        "historical_prompt_v1_metrics": old_metrics,
        "historical_native_metrics": old_metrics[NATIVE_SYSTEM],
        "root_metrics": root_metrics,
        "node_metrics": _node_metrics(records, schedule, final, predictions),
        "paired": paired,
        "position_metrics": {
            name: _position_metrics(records, schedule, item["votes_by_replicate"])
            for name, item in logical["systems"].items()
        },
        "prediction_reuse": logical["prediction_reuse"],
        "retry": load_json(target / "retry/report.json"),
        "execution": _execution_summary(target),
        "primary_system": EQUAL_SYSTEM,
        "primary_comparison": "initial_v2_to_final_equal_v2",
        "supporting_comparison": "final_equal_v1_to_final_equal_v2",
    }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# VL-RewardBench Prompt v2 Transfer Report", "",
        "Exploratory K=3 evaluation using the frozen Phase10 rubric and order schedule.", "",
        "| System | OverallAcc | MacroAcc | Coverage | Strict ACC |", "|---|---:|---:|---:|---:|",
    ]
    for name in (INITIAL_SYSTEM, VISUAL_SYSTEM, WEIGHTED_SYSTEM, EQUAL_SYSTEM):
        item = new_metrics[name]
        lines.append(
            f"| {name} | {item['overall_acc']:.4f} | {item['macro_acc']:.4f} | "
            f"{item['coverage']:.4f} | {item['strict_accuracy']:.4f} |")
    lines.extend(["", "## Historical references", "",
                  "| System | OverallAcc | MacroAcc | Coverage | Strict ACC |",
                  "|---|---:|---:|---:|---:|"])
    for name in (NATIVE_SYSTEM, INITIAL_SYSTEM, EQUAL_SYSTEM):
        item = old_metrics[name]
        lines.append(
            f"| Prompt v1 {name} | {item['overall_acc']:.4f} | "
            f"{item['macro_acc']:.4f} | {item['coverage']:.4f} | "
            f"{item['strict_accuracy']:.4f} |")
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    details = {
        "initial_v2_overall_acc": new_metrics[INITIAL_SYSTEM]["overall_acc"],
        "final_equal_v2_overall_acc": new_metrics[EQUAL_SYSTEM]["overall_acc"],
        "final_equal_v1_overall_acc": old_metrics[EQUAL_SYSTEM]["overall_acc"],
        "evolution_net_corrected": paired[
            "initial_v2_to_final_equal_v2"]["net_corrected"],
        "prompt_net_corrected": paired[
            "final_equal_v1_to_final_equal_v2"]["net_corrected"],
        "unresolved_technical_failures": value["retry"]["still_failed_count"],
    }
    _status(target, STAGE_REPORT, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {
        STAGE_FREEZE: freeze,
        STAGE_AUDIT: audit,
        STAGE_SMOKE: smoke,
        STAGE_RUN: run,
        STAGE_RETRY: retry,
        STAGE_REPORT: report,
    }
    try:
        actions[stage](config, output)
    except KeyError as exc:
        raise ValueError(f"unsupported Prompt v2 VL-RewardBench stage: {stage}") from exc

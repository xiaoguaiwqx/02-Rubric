"""Qwen2.5-VL worker transfer on the frozen Phase17 Epoch-4 rubric.

The structured treatment and Initial five-root baseline share one Prompt-v2
prediction set.  The Native baseline uses the benchmark prompt and reproduces
the historical regex -> Qwen3.5-397B text parser -> same-prompt max-10 retry
recovery protocol.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    ModelCallMetrics,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    StructuredRubric,
    combine_model_call_metrics,
)
from critiq.structured.version import PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION

from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from . import vl_rewardbench_phase10 as phase10
from . import vl_rewardbench_phase10_capped as capped
from . import vl_rewardbench_prompt_v2 as prompt_v2
from .experiment_utils import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    make_progress_callback,
)
from .rubric_factory import build_multicrit_open_ended_init_rubric, file_sha256


EXPERIMENT_DIR = "vl_rewardbench_qwen25_phase17_e4_transfer_v1"
PROTOCOL_VERSION = "vlrb-qwen25-phase17-e4-transfer-v1"
CONFIG_KEY = "vlrb_qwen25_transfer"
WORKER_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SOURCE_EVOLUTION_EXPERIMENT = "phase17_discovery_v2_prompt_v2_split_refine_v1"
SOURCE_EPOCH = 4
SOURCE_RUBRIC_SHA256 = (
    "007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d"
)
SOURCE_NODE_COUNT = 27
HISTORICAL_PROMPT_V2_EXPERIMENT = "vl_rewardbench_phase10_prompt_v2_transfer_v1"
HISTORICAL_E4_EXPERIMENT = "vl_rewardbench_phase17_checkpoint_transfer_v1"
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")
SMOKE_COUNT = 20
MAX_RETRY_ATTEMPTS = 10

INITIAL_SYSTEM = "qwen25_initial_five_root_prompt_v2"
E4_SYSTEM = "qwen25_phase17_epoch_04_equal"
NATIVE_SYSTEM = "qwen25_native_vlrb_prompt"

STAGE_FREEZE = "vlrb-qwen25-transfer-freeze"
STAGE_AUDIT = "vlrb-qwen25-transfer-audit"
STAGE_SMOKE = "vlrb-qwen25-transfer-smoke"
STAGE_RUN = "vlrb-qwen25-transfer-run"
STAGE_RETRY = "vlrb-qwen25-transfer-retry"
STAGE_REPORT = "vlrb-qwen25-transfer-report"

SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "worker_model": WORKER_MODEL,
    "source_experiment": SOURCE_EVOLUTION_EXPERIMENT,
    "source_epoch": SOURCE_EPOCH,
    "source_rubric_sha256": SOURCE_RUBRIC_SHA256,
    "source_node_count": SOURCE_NODE_COUNT,
    "endpoint_ids": list(ENDPOINT_IDS),
    "scheduler": "sample_major_available_slot_dynamic",
    "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    "structured_temperature": 0.5,
    "structured_max_tokens": 2048,
    "native_decoding": dict(legacy.NATIVE_DECODING),
    "native_regex_parser": "vl_rewardbench_overall_judgment_regex_v1",
    "native_fallback_model": capped.NATIVE_FALLBACK_MODEL,
    "native_fallback_prompt_version": capped.NATIVE_FALLBACK_PROMPT_VERSION,
    "native_retry_max_attempts": MAX_RETRY_ATTEMPTS,
    "smoke_sample_count": SMOKE_COUNT,
    "k": legacy.K,
    "seed": legacy.SEED,
    "selection_after_benchmark_forbidden": True,
}


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _rubric_path(output: Path) -> Path:
    return (
        output / SOURCE_EVOLUTION_EXPERIMENT
        / f"epochs/epoch_{SOURCE_EPOCH:02d}/rubric_committed.json"
    ).resolve()


def _historical_prompt_v2_report() -> Path:
    return (
        base.ROOT / "output/evolving_structured_rubrics"
        / HISTORICAL_PROMPT_V2_EXPERIMENT / "final_report.json"
    )


def _historical_e4_report() -> Path:
    return (
        base.ROOT / "output/evolving_structured_rubrics"
        / HISTORICAL_E4_EXPERIMENT / "final_report.json"
    )


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
    value = config.get(CONFIG_KEY)
    if value != SETTINGS:
        raise RuntimeError(f"{CONFIG_KEY} must match the frozen v1 protocol")
    return dict(value)


def _worker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an experiment-local Qwen2.5 view without mutating old protocols."""

    value = deepcopy(dict(config))
    value["model"] = WORKER_MODEL
    value["worker_request_kwargs"] = {
        "temperature": SETTINGS["structured_temperature"],
        "max_tokens": SETTINGS["structured_max_tokens"],
    }
    pool = deepcopy(dict(value["backend_pool"]))
    pool["pool_id"] = "qwen25vl7b-vlrb-transfer-pool-v1"
    pool["common_checkpoint_id"] = WORKER_MODEL
    value["backend_pool"] = pool
    return value


def _pool_spec(config: Mapping[str, Any]) -> BackendPoolSpec:
    spec = phase10._pool_spec(_worker_config(config))
    if spec.global_request_concurrency != sum(
            endpoint.max_concurrency for endpoint in spec.endpoints):
        raise RuntimeError("Qwen2.5 transfer requires a full available-slot pool")
    return spec


def _rubric(output: Path) -> StructuredRubric:
    path = _rubric_path(output)
    if not path.is_file():
        raise RuntimeError(f"Phase17 Epoch-4 rubric is missing: {path}")
    rubric = StructuredRubric.load_json(path)
    if (rubric.rubric_sha256 != SOURCE_RUBRIC_SHA256
            or len(rubric.nodes) != SOURCE_NODE_COUNT):
        raise RuntimeError("Phase17 Epoch-4 rubric identity drift")
    initial = build_multicrit_open_ended_init_rubric()
    phase10._root_reuse_contract(initial, rubric)
    return rubric


def _records() -> tuple[dict[str, Any], ...]:
    return prompt_v2._records()


def _native_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": WORKER_MODEL,
        "prompt_sha256": file_sha256(legacy._prompt_path()),
        "decoding": dict(legacy.NATIVE_DECODING),
        "parser": SETTINGS["native_regex_parser"],
        "fallback_parser": capped._native_fallback_profile(config),
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "k": legacy.K,
        "seed": legacy.SEED,
    }


def _manifest(config: Mapping[str, Any], output: Path, records, schedule,
              *, include_endpoints: bool) -> dict[str, Any]:
    protocol = _protocol(config)
    worker = _worker_config(config)
    rubric = _rubric(output)
    for path in (_historical_prompt_v2_report(), _historical_e4_report()):
        if not path.is_file():
            raise RuntimeError(f"historical Qwen3 report is missing: {path}")
    rows = legacy._ordered_rows(records[:1], schedule, 0)
    request_spec = prompt_v2._v2_request_spec(worker, records, schedule)
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol": protocol,
        "exploratory": True,
        "dataset": {
            "count": len(records),
            "record_sha256": canonical_sha256(list(records)),
            "sample_ids_sha256": canonical_sha256(
                [str(record["sample_id"]) for record in records]),
        },
        "order_schedule_sha256": canonical_sha256(schedule),
        "counterbalance": {
            "k": legacy.K, "seed": legacy.SEED,
            "protocol": "balanced_b_1minusb_b",
        },
        "structured_worker_request_spec": request_spec,
        "structured_request_probe_sha256": canonical_sha256(rows),
        "native_request_spec": _native_spec(config),
        "source": {
            "rubric_path": str(_rubric_path(output)),
            "rubric_file_sha256": file_sha256(_rubric_path(output)),
            "rubric_sha256": rubric.rubric_sha256,
            "node_count": len(rubric.nodes),
            "historical_prompt_v2_report": str(
                _historical_prompt_v2_report().resolve()),
            "historical_prompt_v2_report_sha256": file_sha256(
                _historical_prompt_v2_report()),
            "historical_e4_report": str(_historical_e4_report().resolve()),
            "historical_e4_report_sha256": file_sha256(
                _historical_e4_report()),
        },
        "endpoint_pool": _pool_spec(config).to_dict(),
        "systems": {
            INITIAL_SYSTEM: "offline_projection_from_qwen25_e4_predictions",
            E4_SYSTEM: "qwen25_structured_treatment",
            NATIVE_SYSTEM: "qwen25_native_auxiliary_baseline",
            "qwen3_corresponding_systems": "read_only_historical_controls",
        },
        "selection_after_benchmark_forbidden": True,
    }
    if include_endpoints:
        value["endpoint_identities"] = phase10._inspect_endpoints(worker)
    return value


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target()
    _require(target, STAGE_FREEZE)
    records = _records()
    schedule = legacy._order_schedule(records)
    expected = _manifest(config, output, records, schedule, include_endpoints=False)
    stored = load_json(target / "frozen_manifest.json")
    without_live = {key: item for key, item in stored.items()
                    if key != "endpoint_identities"}
    if without_live != expected:
        raise RuntimeError("Qwen2.5 transfer frozen manifest drift")
    return target, stored, records, schedule, _rubric(output)


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    frozen = {item["endpoint_id"]: item
              for item in manifest["endpoint_identities"]}
    live = {item["endpoint_id"]: item
            for item in phase10._inspect_endpoints(_worker_config(config))}
    for endpoint_id in ENDPOINT_IDS:
        if live[endpoint_id] != frozen[endpoint_id]:
            raise RuntimeError(
                f"Qwen2.5 transfer endpoint identity drift: {endpoint_id}")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target()
    records = _records()
    schedule = legacy._order_schedule(records)
    manifest = _manifest(config, output, records, schedule, include_endpoints=True)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        if any(status.get(stage, {}).get("status") == "passed"
               for stage in (STAGE_SMOKE, STAGE_RUN, STAGE_RETRY, STAGE_REPORT)):
            raise RuntimeError("Qwen2.5 transfer manifest drift after inference")
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    details = {
        "dataset_count": len(records),
        "worker_model": WORKER_MODEL,
        "rubric_sha256": manifest["source"]["rubric_sha256"],
        "node_count": len(_rubric(output).nodes),
        "structured_logical_requests": len(records) * legacy.K * SOURCE_NODE_COUNT,
        "native_logical_requests": len(records) * legacy.K,
    }
    _status(target, STAGE_FREEZE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    initial = build_multicrit_open_ended_init_rubric()
    phase10._root_reuse_contract(initial, rubric)
    schedules = list(schedule.values())
    value = {
        "schema_version": "1.0.0",
        "status": "passed",
        "offline_only": True,
        "dataset_count": len(records),
        "k": legacy.K,
        "worker_model": WORKER_MODEL,
        "node_count": len(rubric.nodes),
        "initial_root_projection_valid": True,
        "schedule_aba_count": sum(tuple(item) == (0, 1, 0) for item in schedules),
        "schedule_bab_count": sum(tuple(item) == (1, 0, 1) for item in schedules),
        "native_parser_chain": [
            SETTINGS["native_regex_parser"],
            SETTINGS["native_fallback_prompt_version"],
            "same_native_prompt_max10",
        ],
        "benchmark_labels_in_prompt": False,
        "selection_after_benchmark_forbidden": True,
        "endpoint_identity_count": len(manifest["endpoint_identities"]),
    }
    atomic_write_json(target / "offline_audit.json", value)
    _status(target, STAGE_AUDIT, value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _native_cache_path(work: Path, sample_id: str, replicate: int) -> Path:
    return work / "native/cache" / (
        canonical_sha256({
            "protocol": PROTOCOL_VERSION,
            "sample_id": sample_id,
            "replicate": replicate + 1,
        }) + ".json"
    )


def _native_run(config: Mapping[str, Any], work: Path,
                manifest: Mapping[str, Any], records, schedule) -> tuple[list, ...]:
    request_identity = canonical_sha256(manifest["native_request_spec"])
    pool = AvailableSlotBackendPool(_pool_spec(config))
    total = len(records) * legacy.K
    callback = make_progress_callback(work / "native", "qwen25_native", total, pool)
    results: list[list[dict[str, Any] | None]] = [
        [None] * len(records) for _ in range(legacy.K)]

    def one(sample_index: int, replicate: int, record: Mapping[str, Any]):
        sample_id = str(record["sample_id"])
        order = int(schedule[sample_id][replicate])
        path = _native_cache_path(work, sample_id, replicate)
        if path.exists():
            cached = load_json(path)
            if (cached.get("sample_id") != sample_id
                    or cached.get("order") != order
                    or cached.get("request_identity") != request_identity):
                raise RuntimeError("Qwen2.5 Native cache identity drift")
            metrics = ModelCallMetrics.from_agent_calls((), cache_hit=True)
            return sample_index, replicate, cached, metrics
        raw, call_metrics = pool.call(
            legacy._native_content(record, order),
            request_type="vlrb_qwen25_native",
            request_key=f"{sample_id}::replicate_{replicate + 1:02d}",
            structured_attempt=1,
            agent_args={
                "model": WORKER_MODEL,
                "api_keys": "EMPTY",
                "request_kwargs": legacy.NATIVE_DECODING,
                "api_retry_attempts": config["api_retry_attempts"],
            },
        )
        choice = legacy._parse_native(raw)
        normalized = ModelCallMetrics.from_agent_calls((call_metrics,))
        value = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "replicate": replicate + 1,
            "order": order,
            "request_identity": request_identity,
            "display_choice": choice,
            "parse_ok": choice is not None,
            "raw_response": raw,
            "metrics": normalized.to_dict(),
        }
        atomic_write_json(path, value)
        return sample_index, replicate, value, normalized

    with ThreadPoolExecutor(max_workers=pool.spec.global_request_concurrency) as executor:
        futures = []
        for sample_index, record in enumerate(records):
            for replicate in range(legacy.K):
                futures.append(executor.submit(one, sample_index, replicate, record))
        for future in as_completed(futures):
            sample_index, replicate, value, metrics = future.result()
            results[replicate][sample_index] = value
            callback(sample_index, str(value["sample_id"]), metrics)
    completed = []
    for replicate, values in enumerate(results):
        if any(item is None for item in values):
            raise RuntimeError("Qwen2.5 Native result matrix is incomplete")
        typed = [dict(item) for item in values if item is not None]
        atomic_write_json(
            work / "native" / f"replicate_{replicate + 1:02d}.json", typed)
        completed.append(typed)
    atomic_write_json(work / "native/provenance.json", pool.provenance_dict())
    return tuple(completed)


def _run_both(config: Mapping[str, Any], work: Path,
              manifest: Mapping[str, Any], records, schedule, rubric):
    worker = _worker_config(config)
    logical, predictions, summaries = prompt_v2._run_predictions(
        worker, work / "structured", manifest, records, schedule, rubric)
    native = _native_run(config, work, manifest, records, schedule)
    return logical, predictions, summaries, native


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_AUDIT)
    _verify_live(config, manifest)
    selected = tuple(records[:SMOKE_COUNT])
    logical, _, summaries, native = _run_both(
        config, target / "smoke", manifest, selected, schedule, rubric)
    calls = {endpoint_id: sum(
        int(item["telemetry"]["endpoint_call_counts"].get(endpoint_id, 0))
        for item in summaries) for endpoint_id in ENDPOINT_IDS}
    valid_rate = sum(item["valid_rate"] for item in summaries) / len(summaries)
    native_parse_rate = sum(
        item["parse_ok"] for values in native for item in values) / (
            len(selected) * legacy.K)
    parser_pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(
        manifest["native_request_spec"]["fallback_parser"]["backend_pool"]))
    fallback_choice, _, _ = _parser_choice(
        config, parser_pool,
        "Overall Judgment: Answer 1 is the better response.",
        "qwen25-native-fallback-smoke-probe", 1)
    if fallback_choice != 1:
        raise RuntimeError("Qwen2.5 Native 397B fallback parser smoke failed")
    atomic_write_json(
        target / "smoke/native_fallback_parser_provenance.json",
        parser_pool.provenance_dict())
    metrics = {
        INITIAL_SYSTEM: prompt_v2._smoke_system_metrics(
            selected, logical["systems"][prompt_v2.INITIAL_SYSTEM][
                "votes_by_replicate"]),
        E4_SYSTEM: prompt_v2._smoke_system_metrics(
            selected, logical["systems"][prompt_v2.EQUAL_SYSTEM][
                "votes_by_replicate"]),
    }
    details = {
        "sample_count": len(selected),
        "structured_valid_rate": valid_rate,
        "native_regex_parse_rate": native_parse_rate,
        "native_fallback_parser_probe": "passed",
        "endpoint_call_counts": calls,
        "metrics": metrics,
    }
    _status(target, STAGE_SMOKE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_SMOKE)
    _verify_live(config, manifest)
    _, _, summaries, native = _run_both(
        config, target / "run", manifest, records, schedule, rubric)
    details = {
        "sample_count": len(records),
        "node_count": len(rubric.nodes),
        "structured_logical_request_count": len(records) * legacy.K * len(rubric.nodes),
        "native_logical_request_count": len(records) * legacy.K,
        "structured_valid_rates": [item["valid_rate"] for item in summaries],
        "native_regex_failure_count": sum(
            not item["parse_ok"] for values in native for item in values),
    }
    _status(target, STAGE_RUN, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _retry_structured(config: Mapping[str, Any], target: Path,
                      manifest: Mapping[str, Any], records, schedule,
                      rubric: StructuredRubric) -> tuple[PairwisePredictionOutput, ...]:
    source = tuple(PairwisePredictionOutput.load_json(
        prompt_v2._prediction_path(target / "run/structured", replicate))
        for replicate in range(legacy.K))
    items = prompt_v2._failure_items(source)
    work = target / "retry/structured"
    pool = AvailableSlotBackendPool(_pool_spec(config))
    rows = {replicate: legacy._ordered_rows(records, schedule, replicate)
            for replicate in range(legacy.K)}
    model_rows = {key: base._model_rows(value) for key, value in rows.items()}
    worker = _worker_config(config)
    worker["structured_max_retries"] = 0
    evaluators = {replicate: prompt_v2._v2_evaluator(
        worker, model_rows[replicate], pool) for replicate in range(legacy.K)}
    criteria = {rubric.get_node(node_id).criterion.name:
                rubric.get_node(node_id).criterion
                for node_id in rubric.preorder_node_ids()}
    callback = make_progress_callback(
        work, "qwen25_structured_retry", len(items), pool)

    def one(index: int, item: Mapping[str, Any]):
        path = work / "attempts" / f"{canonical_sha256(item['key'])}.json"
        state = load_json(path) if path.exists() else {
            "key": item["key"], "attempts": [], "success": False,
            "complete": False, "final_output": None,
        }
        replicate = int(item["replicate"])
        sample_index = int(item["sample_index"])
        name = str(item["criterion_name"])
        for number in range(len(state["attempts"]) + 1, MAX_RETRY_ATTEMPTS + 1):
            if state["complete"]:
                break
            candidate, metrics = evaluators[replicate].infer_one(
                model_rows[replicate][sample_index], criteria[name].to_criterion())
            state["attempts"].append({
                "attempt_number": number,
                "output": candidate.to_dict(),
                "metrics": metrics.to_dict(),
            })
            if candidate.parse_ok and candidate.answer_valid:
                state.update({"success": True, "complete": True,
                              "final_output": candidate.to_dict()})
            elif number == MAX_RETRY_ATTEMPTS:
                state["complete"] = True
            atomic_write_json(path, state)
        metrics = combine_model_call_metrics(
            ModelCallMetrics.from_dict(value["metrics"])
            for value in state["attempts"])
        return index, dict(item), state, metrics

    results = [None] * len(items)
    if items:
        with ThreadPoolExecutor(max_workers=pool.spec.global_request_concurrency) as executor:
            futures = [executor.submit(one, index, item)
                       for index, item in enumerate(items)]
            for future in as_completed(futures):
                index, item, state, metrics = future.result()
                results[index] = (item, state)
                callback(index, str(item["sample_id"]), metrics)
    replacements = {replicate: {} for replicate in range(legacy.K)}
    report_items = []
    for item, state in (value for value in results if value is not None):
        if state["success"]:
            replacements[int(item["replicate"])][(
                int(item["sample_index"]), str(item["criterion_name"]))] = (
                    PairwiseVoteOutput.from_dict(state["final_output"]))
        report_items.append({**item, "success": bool(state["success"]),
                             "attempt_count": len(state["attempts"])})
    repaired = []
    for replicate, prediction in enumerate(source):
        updated = prompt_v2._replace_prediction_outputs(
            prediction, replacements[replicate])
        path = prompt_v2._prediction_path(work, replicate)
        path.parent.mkdir(parents=True, exist_ok=True)
        updated.save_json(path)
        repaired.append(updated)
    logical = prompt_v2._logical_from_predictions(
        records, schedule, rubric, repaired)
    atomic_write_json(work / "combined/logical_votes.json", logical)
    details = {
        "target_count": len(items),
        "recovered_count": sum(item["success"] for item in report_items),
        "still_failed_count": sum(not item["success"] for item in report_items),
        "new_model_requests": sum(item["attempt_count"] for item in report_items),
        "items": report_items,
    }
    atomic_write_json(work / "report.json", details)
    return tuple(repaired)


def _parser_choice(config: Mapping[str, Any], parser_pool,
                   raw_response: str, request_key: str, attempt: int):
    profile = capped._native_fallback_profile(config)
    api_key = os.environ.get(profile["api_key_env"], "")
    if not api_key:
        raise RuntimeError(
            f"environment variable {profile['api_key_env']!r} is required "
            "for Native fallback parsing")
    raw, call_metrics = parser_pool.call(
        capped._native_fallback_prompt(raw_response),
        request_type="vlrb_qwen25_native_text_parser",
        request_key=request_key,
        structured_attempt=attempt,
        agent_args={
            "model": profile["model"],
            "api_keys": api_key,
            "request_kwargs": profile["request_kwargs"],
            "api_retry_attempts": config["api_retry_attempts"],
        },
    )
    return capped._parse_native_fallback(raw), raw, ModelCallMetrics.from_agent_calls(
        (call_metrics,))


def _recover_native(config: Mapping[str, Any], target: Path,
                    manifest: Mapping[str, Any], records, schedule):
    source = tuple(load_json(
        target / "run/native" / f"replicate_{replicate + 1:02d}.json")
        for replicate in range(legacy.K))
    failures = [(replicate, index, item)
                for replicate, values in enumerate(source)
                for index, item in enumerate(values) if not item["parse_ok"]]
    work = target / "retry/native"
    worker_pool = AvailableSlotBackendPool(_pool_spec(config))
    profile = manifest["native_request_spec"]["fallback_parser"]
    parser_pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(
        profile["backend_pool"]))
    callback = make_progress_callback(
        work, "qwen25_native_recovery", len(failures), worker_pool)

    def one(position: int, replicate: int, sample_index: int,
            source_item: Mapping[str, Any]):
        key = f"r{replicate + 1:02d}:{sample_index}"
        path = work / "attempts" / f"{canonical_sha256(key)}.json"
        if path.exists():
            state = load_json(path)
            metrics = combine_model_call_metrics(
                ModelCallMetrics.from_dict(item["metrics"])
                for item in state["calls"])
            return position, replicate, sample_index, state, metrics
        calls = []
        choice = None
        for parser_attempt in range(1, capped.NATIVE_FALLBACK_MAX_ATTEMPTS + 1):
            parser_choice, parser_raw, parser_metrics = _parser_choice(
                config, parser_pool, str(source_item["raw_response"]),
                f"{key}:initial_{parser_attempt:02d}", parser_attempt)
            calls.append({
                "kind": "initial_397b_parser", "attempt": parser_attempt,
                "choice": parser_choice, "raw_response": parser_raw,
                "metrics": parser_metrics.to_dict(),
            })
            if parser_choice is not None:
                choice = parser_choice
                break
        choice_source = "initial_397b_parser" if choice is not None else None
        record = records[sample_index]
        order = int(schedule[str(record["sample_id"])][replicate])
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            if choice is not None:
                break
            native_raw, native_call = worker_pool.call(
                legacy._native_content(record, order),
                request_type="vlrb_qwen25_native_retry",
                request_key=f"{key}:native_retry_{attempt:02d}",
                structured_attempt=attempt,
                agent_args={
                    "model": WORKER_MODEL, "api_keys": "EMPTY",
                    "request_kwargs": legacy.NATIVE_DECODING,
                    "api_retry_attempts": config["api_retry_attempts"],
                },
            )
            native_metrics = ModelCallMetrics.from_agent_calls((native_call,))
            regex_choice = legacy._parse_native(native_raw)
            calls.append({
                "kind": "native_retry", "attempt": attempt,
                "choice": regex_choice, "raw_response": native_raw,
                "metrics": native_metrics.to_dict(),
            })
            if regex_choice is not None:
                choice = regex_choice
                choice_source = "native_retry_regex"
                break
            parser_choice, parser_raw, parser_metrics = _parser_choice(
                config, parser_pool, str(native_raw),
                f"{key}:retry_parser_{attempt:02d}", attempt)
            calls.append({
                "kind": "retry_397b_parser", "attempt": attempt,
                "choice": parser_choice, "raw_response": parser_raw,
                "metrics": parser_metrics.to_dict(),
            })
            if parser_choice is not None:
                choice = parser_choice
                choice_source = "native_retry_397b_parser"
                break
        state = {
            "key": key, "replicate": replicate + 1,
            "sample_index": sample_index,
            "sample_id": source_item["sample_id"],
            "order": source_item["order"],
            "choice": choice, "choice_source": choice_source,
            "parse_ok": choice is not None, "calls": calls,
        }
        atomic_write_json(path, state)
        metrics = combine_model_call_metrics(
            ModelCallMetrics.from_dict(item["metrics"]) for item in calls)
        return position, replicate, sample_index, state, metrics

    results = [None] * len(failures)
    if failures:
        with ThreadPoolExecutor(max_workers=worker_pool.spec.global_request_concurrency) as executor:
            futures = [executor.submit(one, position, replicate, index, item)
                       for position, (replicate, index, item) in enumerate(failures)]
            for future in as_completed(futures):
                position, replicate, index, state, metrics = future.result()
                results[position] = state
                callback(index, str(state["sample_id"]), metrics)
    votes = []
    overlays = {(int(item["replicate"]) - 1, int(item["sample_index"])): item
                for item in results if item is not None}
    for replicate, values in enumerate(source):
        replicate_votes = []
        for index, item in enumerate(values):
            choice = item["display_choice"] if item["parse_ok"] else None
            overlay = overlays.get((replicate, index))
            if overlay is not None and overlay["parse_ok"]:
                choice = overlay["choice"]
            replicate_votes.append(legacy._original_index(choice, int(item["order"])))
        votes.append(replicate_votes)
    logical = {
        "schema_version": "1.0.0",
        "sample_ids": [str(record["sample_id"]) for record in records],
        "k": legacy.K,
        "systems": {NATIVE_SYSTEM: {"votes_by_replicate": votes}},
    }
    atomic_write_json(work / "combined/logical_votes.json", logical)
    recovered = [item for item in results if item is not None and item["parse_ok"]]
    report = {
        "regex_failure_count": len(failures),
        "initial_397b_recovered_count": sum(
            item["choice_source"] == "initial_397b_parser" for item in recovered),
        "native_retry_regex_recovered_count": sum(
            item["choice_source"] == "native_retry_regex" for item in recovered),
        "native_retry_397b_recovered_count": sum(
            item["choice_source"] == "native_retry_397b_parser" for item in recovered),
        "unresolved_count": len(failures) - len(recovered),
        "new_native_requests": sum(
            call["kind"] == "native_retry" for item in results if item
            for call in item["calls"]),
        "new_parser_requests": sum(
            "397b_parser" in call["kind"] for item in results if item
            for call in item["calls"]),
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
    }
    atomic_write_json(work / "report.json", report)
    return logical, report


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_RUN)
    _verify_live(config, manifest)
    _retry_structured(config, target, manifest, records, schedule, rubric)
    _, native_report = _recover_native(
        config, target, manifest, records, schedule)
    structured_report = load_json(target / "retry/structured/report.json")
    details = {
        "structured": {key: structured_report[key] for key in (
            "target_count", "recovered_count", "still_failed_count",
            "new_model_requests")},
        "native": native_report,
    }
    atomic_write_json(target / "retry/report.json", details)
    _status(target, STAGE_RETRY, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _historical_metrics() -> dict[str, Any]:
    prompt_report = load_json(_historical_prompt_v2_report())
    e4_report = load_json(_historical_e4_report())
    return {
        "qwen3_native": prompt_report["historical_native_metrics"],
        "qwen3_initial_five_root_prompt_v2": (
            prompt_report["new_metrics"]["initial_five_root_equal"]),
        "qwen3_phase17_epoch_04_equal": (
            e4_report["metrics"]["phase17_epoch_04_equal"]),
    }


def _metric_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item[key] for key in (
        "overall_acc", "macro_acc", "coverage", "strict_accuracy")}


def report(config: Mapping[str, Any], output: Path) -> None:
    target, _, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_RETRY)
    structured = load_json(target / "retry/structured/combined/logical_votes.json")
    native = load_json(target / "retry/native/combined/logical_votes.json")
    initial_metrics = phase10._system_metrics(
        records, structured["systems"][prompt_v2.INITIAL_SYSTEM]["votes_by_replicate"])
    e4_metrics = phase10._system_metrics(
        records, structured["systems"][prompt_v2.EQUAL_SYSTEM]["votes_by_replicate"])
    native_metrics = phase10._system_metrics(
        records, native["systems"][NATIVE_SYSTEM]["votes_by_replicate"])
    historical = _historical_metrics()
    metrics = {
        NATIVE_SYSTEM: native_metrics,
        INITIAL_SYSTEM: initial_metrics,
        E4_SYSTEM: e4_metrics,
        **historical,
    }
    paired = {
        "qwen25_initial_to_e4": legacy._paired(
            records, initial_metrics["original_index_predictions"],
            e4_metrics["original_index_predictions"]),
        "qwen3_to_qwen25_native": legacy._paired(
            records, historical["qwen3_native"]["original_index_predictions"],
            native_metrics["original_index_predictions"]),
        "qwen3_to_qwen25_initial": legacy._paired(
            records,
            historical["qwen3_initial_five_root_prompt_v2"][
                "original_index_predictions"],
            initial_metrics["original_index_predictions"]),
        "qwen3_to_qwen25_e4": legacy._paired(
            records,
            historical["qwen3_phase17_epoch_04_equal"][
                "original_index_predictions"],
            e4_metrics["original_index_predictions"]),
    }
    predictions = tuple(PairwisePredictionOutput.load_json(
        prompt_v2._prediction_path(target / "retry/structured", replicate))
        for replicate in range(legacy.K))
    root_metrics = {
        root_id: phase10._system_metrics(records, votes)
        for root_id, votes in structured["root_votes_by_replicate"].items()
    }
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "worker_model": WORKER_MODEL,
        "rubric_sha256": rubric.rubric_sha256,
        "node_count": len(rubric.nodes),
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "metrics": metrics,
        "metric_summary": {name: _metric_row(item)
                           for name, item in metrics.items()},
        "paired": paired,
        "root_metrics": root_metrics,
        "node_metrics": prompt_v2._node_metrics(
            records, schedule, rubric, predictions),
        "position_metrics": prompt_v2._position_metrics(
            records, schedule,
            structured["systems"][prompt_v2.EQUAL_SYSTEM]["votes_by_replicate"]),
        "retry": load_json(target / "retry/report.json"),
        "selection_after_benchmark_forbidden": True,
        "primary_comparison": "qwen25_initial_to_e4",
        "native_is_auxiliary": True,
    }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# Qwen2.5-VL Phase17 E4 Worker Transfer", "",
        "Exploratory K=3 model-swap evaluation on VL-RewardBench.", "",
        "| System | OverallAcc | MacroAcc | Coverage | Strict ACC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in value["metric_summary"].items():
        lines.append(
            f"| {name} | {item['overall_acc']:.4f} | {item['macro_acc']:.4f} | "
            f"{item['coverage']:.4f} | {item['strict_accuracy']:.4f} |")
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    details = {
        "qwen25_native_overall_acc": native_metrics["overall_acc"],
        "qwen25_initial_overall_acc": initial_metrics["overall_acc"],
        "qwen25_e4_overall_acc": e4_metrics["overall_acc"],
        "rubric_net_corrected": paired["qwen25_initial_to_e4"]["net_corrected"],
        "structured_unresolved": value["retry"]["structured"]["still_failed_count"],
        "native_unresolved": value["retry"]["native"]["unresolved_count"],
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
    if stage not in actions:
        raise ValueError(f"unsupported Qwen2.5 transfer stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

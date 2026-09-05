"""VL-RewardBench transfer for the Prompt-v2-evolved Phase-16 rubric."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    ModelCallMetrics,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    StructuredRubric,
    combine_model_call_metrics,
)
from critiq.structured.version import PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION

from . import pairwise_cache_ablation as cache_exp
from . import prompt_v2_aligned_evolution as evolution
from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from . import vl_rewardbench_phase10 as phase10
from . import vl_rewardbench_prompt_v2 as control
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, make_progress_callback
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_phase16_prompt_v2_evolved_v2"
PROTOCOL_VERSION = "vlrb-prompt-v2-evolved-v2"
CONTROL_EXPERIMENT = control.EXPERIMENT_DIR
ENDPOINT_IDS = control.ENDPOINT_IDS
SMOKE_COUNT = 20
MAX_RETRY_ATTEMPTS = 10
TREATMENT_SYSTEM = "prompt_v2_evolved_final_equal"
SCHEDULER = "available_slot_dynamic"
EXTRA_BASELINE_REPORTS: dict[str, tuple[Path, str]] = {}

STAGE_FREEZE = "vlrb-prompt-v2-evolved-freeze"
STAGE_AUDIT = "vlrb-prompt-v2-evolved-audit"
STAGE_SMOKE = "vlrb-prompt-v2-evolved-smoke"
STAGE_RUN = "vlrb-prompt-v2-evolved-run"
STAGE_RETRY = "vlrb-prompt-v2-evolved-retry"
STAGE_REPORT = "vlrb-prompt-v2-evolved-report"


def _target(config: Mapping[str, Any] | None = None) -> Path:
    if config is not None and config.get("_vlrb_output_dir"):
        return base._path(config["_vlrb_output_dir"])
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _control_target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / CONTROL_EXPERIMENT


def _rubric_path(output: Path, config: Mapping[str, Any] | None = None) -> Path:
    if config is not None and config.get("_vlrb_rubric_path"):
        return base._path(config["_vlrb_rubric_path"])
    return output / evolution.EXPERIMENT_DIR / "final" / "rubric.json"


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
        "source_experiment": evolution.EXPERIMENT_DIR,
        "control_experiment": CONTROL_EXPERIMENT,
        "endpoint_ids": list(ENDPOINT_IDS),
        "scheduler": SCHEDULER,
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "smoke_sample_count": SMOKE_COUNT,
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "k": legacy.K,
        "seed": legacy.SEED,
        "run_regardless_of_heldout_result": True,
        "selection_after_benchmark_forbidden": True,
    }
    value = config.get("vlrb_prompt_v2_evolved")
    if value != expected:
        raise RuntimeError("vlrb_prompt_v2_evolved must match the frozen v1 protocol")
    request = config.get("worker_request_kwargs")
    if (not isinstance(request, Mapping)
            or request.get("temperature") != 0.5
            or request.get("max_tokens") != 2048):
        raise RuntimeError("VL-RB evolved treatment requires temperature=.5/max_tokens=2048")
    control._pool_spec(config)
    return dict(value)


def _rubric(output: Path, config: Mapping[str, Any] | None = None) -> StructuredRubric:
    path = _rubric_path(output, config)
    if config is not None and config.get("_vlrb_rubric_path"):
        return StructuredRubric.load_json(path)
    if not path.is_file():
        raise RuntimeError("run prompt-v2-evolution-report before VL-RewardBench")
    final_report_path = output / evolution.EXPERIMENT_DIR / "final_report.json"
    if not final_report_path.is_file():
        raise RuntimeError(
            "run prompt-v2-evolution-heldout/final-report before VL-RewardBench")
    rubric = StructuredRubric.load_json(path)
    discovery = load_json(
        output / evolution.EXPERIMENT_DIR / "final" / "discovery_report.json")
    if rubric.rubric_sha256 != discovery["final_rubric_sha256"]:
        raise RuntimeError("Prompt-v2-evolved final rubric hash drift")
    final_report = load_json(final_report_path)
    if (not final_report.get("vl_rewardbench_required")
            or final_report.get("discovery", {}).get("final_rubric_sha256")
            != rubric.rubric_sha256):
        raise RuntimeError("Prompt-v2 evolution final report contract drift")
    return rubric


def _records():
    return control._records()


def _control_logical_path() -> Path:
    path = _control_target() / "retry" / "combined" / "logical_votes.json"
    if not path.is_file():
        raise RuntimeError(f"completed Prompt v2 Control votes are required: {path}")
    return path


def _validate_prompt_v2_control_logical(
    records: Sequence[Mapping[str, Any]], logical: Mapping[str, Any],
) -> None:
    """Validate the Phase10 Prompt-v2 control, not the older Prompt-v1 source.

    The Prompt-v2 transfer artifact intentionally contains only structured
    systems derived from the Prompt-v2 worker.  In particular, it has no
    native-judge system, so ``control._validate_source_logical`` is not the
    appropriate validator here.
    """
    expected_ids = [str(record["sample_id"]) for record in records]
    if logical.get("sample_ids") != expected_ids or logical.get("k") != legacy.K:
        raise RuntimeError("Prompt-v2 Control logical vote identity drift")
    required = {control.INITIAL_SYSTEM, control.EQUAL_SYSTEM}
    if not required.issubset(logical.get("systems", {})):
        raise RuntimeError("Prompt-v2 Control logical votes are incomplete")


def _manifest(config: Mapping[str, Any], output: Path, records, schedule,
              *, include_endpoints: bool) -> dict[str, Any]:
    protocol = _protocol(config)
    rubric = _rubric(output, config)
    control_manifest_path = _control_target() / "frozen_manifest.json"
    control_report_path = _control_target() / "final_report.json"
    control_logical_path = _control_logical_path()
    for path in (control_manifest_path, control_report_path, control_logical_path):
        if not path.is_file():
            raise RuntimeError(f"completed Prompt v2 Control artifact missing: {path}")
    control_manifest = load_json(control_manifest_path)
    _validate_prompt_v2_control_logical(
        records, load_json(control_logical_path))
    dataset = control_manifest["dataset"]
    if (dataset.get("count") != legacy.EXPECTED_COUNT
            or dataset.get("record_sha256") != canonical_sha256(list(records))):
        raise RuntimeError("VL-RewardBench record identity drift")
    request_spec = control._v2_request_spec(config, records, schedule)
    request_scientific = dict(request_spec); request_scientific.pop("backend_id")
    control_scientific = dict(control_manifest["structured_worker_request_spec"])
    control_scientific.pop("backend_id")
    if request_scientific != control_scientific:
        raise RuntimeError("treatment and Control Prompt v2 request identity differ")
    value = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol": protocol,
        "exploratory": True,
        "dataset": dataset,
        "order_schedule_sha256": canonical_sha256(schedule),
        "counterbalance": {"k": legacy.K, "seed": legacy.SEED,
                           "protocol": "balanced_b_1minusb_b"},
        "structured_worker_request_spec": request_spec,
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "source": {
            "rubric_path": str(_rubric_path(output, config).resolve()),
            "rubric_file_sha256": file_sha256(_rubric_path(output, config)),
            "rubric_sha256": rubric.rubric_sha256,
            "node_count": len(rubric.nodes),
            "control_manifest_path": str(control_manifest_path.resolve()),
            "control_manifest_sha256": file_sha256(control_manifest_path),
            "control_report_path": str(control_report_path.resolve()),
            "control_report_sha256": file_sha256(control_report_path),
            "control_logical_path": str(control_logical_path.resolve()),
            "control_logical_sha256": file_sha256(control_logical_path),
        },
        "endpoint_pool": control._pool_spec(config).to_dict(),
        "systems": {
            "initial_five_root_prompt_v2": "read_only_control",
            "phase10_final_equal_prompt_v2": "read_only_control",
            **{name: "read_only_external_baseline"
               for name in EXTRA_BASELINE_REPORTS},
            TREATMENT_SYSTEM: "treatment",
        },
        "selection_after_benchmark_forbidden": True,
        "run_regardless_of_heldout_result": True,
    }
    if include_endpoints:
        value["endpoint_identities"] = phase10._inspect_endpoints(config)
    if EXTRA_BASELINE_REPORTS:
        frozen_baselines = {}
        for name, (report_path, metric_name) in EXTRA_BASELINE_REPORTS.items():
            if not report_path.is_file():
                raise RuntimeError(f"completed external baseline missing: {report_path}")
            report = load_json(report_path)
            if metric_name not in report.get("metrics", {}):
                raise RuntimeError(
                    f"external baseline metric missing: {name}/{metric_name}")
            frozen_baselines[name] = {
                "report_path": str(report_path.resolve()),
                "report_sha256": file_sha256(report_path),
                "metric_name": metric_name,
            }
        value["external_baselines"] = frozen_baselines
    return value


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target(config)
    _require(target, STAGE_FREEZE)
    records = _records()
    schedule = legacy._order_schedule(records)
    expected = _manifest(config, output, records, schedule, include_endpoints=False)
    stored = load_json(target / "frozen_manifest.json")
    if {key: value for key, value in stored.items()
            if key != "endpoint_identities"} != expected:
        raise RuntimeError("VL-RB evolved frozen manifest drift")
    return target, stored, records, schedule, _rubric(output, config)


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    frozen = {item["endpoint_id"]: item for item in manifest["endpoint_identities"]}
    live = {item["endpoint_id"]: item for item in phase10._inspect_endpoints(config)}
    for endpoint_id in ENDPOINT_IDS:
        for key in ("endpoint_id", "base_url", "vllm_version", "model", "max_concurrency"):
            if frozen[endpoint_id].get(key) != live[endpoint_id].get(key):
                raise RuntimeError(
                    f"VL-RB evolved endpoint scientific identity drift: "
                    f"{endpoint_id}/{key}")
        if int(live[endpoint_id].get("max_model_len") or 0) < 4096:
            raise RuntimeError(f"VL-RB endpoint context too short: {endpoint_id}")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(config)
    records = _records()
    schedule = legacy._order_schedule(records)
    manifest = _manifest(config, output, records, schedule, include_endpoints=True)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        if any(status.get(stage, {}).get("status") == "passed"
               for stage in (STAGE_SMOKE, STAGE_RUN, STAGE_RETRY, STAGE_REPORT)):
            raise RuntimeError("VL-RB evolved manifest drift after inference")
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    details = {
        "dataset_count": len(records),
        "rubric_sha256": manifest["source"]["rubric_sha256"],
        "node_count": manifest["source"]["node_count"],
        "logical_request_count": (
            len(records) * legacy.K * manifest["source"]["node_count"]),
    }
    _status(target, STAGE_FREEZE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    logical = load_json(_control_logical_path())
    _validate_prompt_v2_control_logical(records, logical)
    schedules = list(schedule.values())
    value = {
        "schema_version": "1.0.0",
        "status": "passed",
        "dataset_count": len(records),
        "k": legacy.K,
        "node_count": len(rubric.nodes),
        "new_logical_request_count": len(records) * legacy.K * len(rubric.nodes),
        "control_new_request_count": 0,
        "schedule_aba_count": sum(tuple(item) == (0, 1, 0) for item in schedules),
        "schedule_bab_count": sum(tuple(item) == (1, 0, 1) for item in schedules),
        "prompt_v1_cache_reuse_allowed": False,
        "benchmark_labels_in_prompt": False,
        "selection_after_benchmark_forbidden": True,
        "endpoint_identity_count": len(manifest["endpoint_identities"]),
    }
    if abs(value["schedule_aba_count"] - value["schedule_bab_count"]) > 1:
        raise RuntimeError("VL-RB K=3 schedule is not balanced")
    atomic_write_json(target / "offline_audit.json", value)
    _status(target, STAGE_AUDIT, value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_AUDIT)
    _verify_live(config, manifest)
    selected = tuple(records[:SMOKE_COUNT])
    logical, _, summaries = control._run_predictions(
        config, target / "smoke", manifest, selected, schedule, rubric)
    calls = {endpoint_id: sum(
        int(item["telemetry"]["endpoint_call_counts"].get(endpoint_id, 0))
        for item in summaries) for endpoint_id in ENDPOINT_IDS}
    valid_rate = sum(item["valid_rate"] for item in summaries) / len(summaries)
    if valid_rate < .99 or any(count == 0 for count in calls.values()):
        raise RuntimeError("VL-RB evolved smoke validity or dual-endpoint use failed")
    metrics = control._smoke_system_metrics(
        selected, logical["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"])
    details = {"sample_count": len(selected), "valid_rate": valid_rate,
               "endpoint_call_counts": calls, "treatment_metrics": metrics}
    _status(target, STAGE_SMOKE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_SMOKE)
    _verify_live(config, manifest)
    _, _, summaries = control._run_predictions(
        config, target / "run", manifest, records, schedule, rubric)
    details = {
        "sample_count": len(records), "node_count": len(rubric.nodes),
        "logical_request_count": len(records) * legacy.K * len(rubric.nodes),
        "valid_rates": [item["valid_rate"] for item in summaries],
    }
    _status(target, STAGE_RUN, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_RUN)
    _verify_live(config, manifest)
    source_predictions = tuple(PairwisePredictionOutput.load_json(
        control._prediction_path(target / "run", replicate))
        for replicate in range(legacy.K))
    items = control._failure_items(source_predictions)
    work = target / "retry"
    atomic_write_json(work / "failure_manifest.json", {
        "schema_version": "1.0.0", "target_count": len(items),
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "selection_uses_gold": False, "items": items})
    pool = AvailableSlotBackendPool(control._pool_spec(config))
    ordered_rows = {replicate: legacy._ordered_rows(records, schedule, replicate)
                    for replicate in range(legacy.K)}
    model_rows = {key: base._model_rows(value)
                  for key, value in ordered_rows.items()}
    retry_config = dict(config); retry_config["structured_max_retries"] = 0
    evaluators = {replicate: control._v2_evaluator(
        retry_config, model_rows[replicate], pool) for replicate in range(legacy.K)}
    criteria = {rubric.get_node(node_id).criterion.name:
                rubric.get_node(node_id).criterion
                for node_id in rubric.preorder_node_ids()}
    callback = make_progress_callback(
        work, "prompt_v2_evolved_technical_retry", len(items), pool)

    def one(index: int, item: Mapping[str, Any]):
        path = work / "attempts" / f"{canonical_sha256(item['key'])}.json"
        state = load_json(path) if path.exists() else {
            "schema_version": "1.0.0", "key": item["key"],
            "source_output_sha256": item["source_output_sha256"],
            "attempts": [], "complete": False, "success": False,
            "final_output": None}
        replicate = int(item["replicate"]); sample_index = int(item["sample_index"])
        name = str(item["criterion_name"])
        if not state["complete"]:
            for number in range(len(state["attempts"]) + 1, MAX_RETRY_ATTEMPTS + 1):
                candidate, metrics = evaluators[replicate].infer_one(
                    model_rows[replicate][sample_index], criteria[name].to_criterion())
                state["attempts"].append({"attempt_number": number,
                                          "output": candidate.to_dict(),
                                          "metrics": metrics.to_dict()})
                if candidate.parse_ok and candidate.answer_valid:
                    state.update({"success": True, "complete": True,
                                  "final_output": candidate.to_dict()})
                elif number == MAX_RETRY_ATTEMPTS:
                    state["complete"] = True
                atomic_write_json(path, state)
                if state["success"]: break
        metrics = combine_model_call_metrics(
            ModelCallMetrics.from_dict(value["metrics"])
            for value in state["attempts"])
        return index, dict(item), state, metrics

    results = [None] * len(items)
    if items:
        with ThreadPoolExecutor(
                max_workers=control._pool_spec(config).global_request_concurrency) as executor:
            futures = {executor.submit(one, index, item): index
                       for index, item in enumerate(items)}
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
                             "attempt_count": len(state["attempts"]),
                             "final_output": state["final_output"]})
    repaired = []
    for replicate, source in enumerate(source_predictions):
        prediction = control._replace_prediction_outputs(
            source, replacements[replicate])
        path = control._prediction_path(work, replicate)
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.save_json(path); repaired.append(prediction)
    logical = control._logical_from_predictions(records, schedule, rubric, repaired)
    atomic_write_json(work / "combined/logical_votes.json", logical)
    atomic_write_json(work / "retry_items.json", {
        "schema_version": "1.0.0", "items": report_items})
    recovered = sum(item["success"] for item in report_items)
    details = {"target_count": len(items), "recovered_count": recovered,
               "still_failed_count": len(items) - recovered,
               "new_model_requests": sum(item["attempt_count"] for item in report_items),
               "max_retry_attempts": MAX_RETRY_ATTEMPTS}
    atomic_write_json(work / "report.json", details)
    _status(target, STAGE_RETRY, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _execution_summary(target: Path, node_count: int) -> dict[str, Any]:
    summaries = load_json(target / "run" / "run_summaries.json")["items"]
    wall_seconds = sum(float(item["telemetry"]["wall_seconds"])
                       for item in summaries)
    logical_requests = legacy.EXPECTED_COUNT * legacy.K * node_count
    return {
        "logical_request_count": logical_requests,
        "wall_seconds_sum_across_replicates": wall_seconds,
        "inferences_per_minute": (
            logical_requests / wall_seconds * 60 if wall_seconds else 0.0),
        "recorded_model_call_count": sum(
            int(item["telemetry"]["recorded_model_call_count"])
            for item in summaries),
        "endpoint_call_counts": {
            endpoint_id: sum(int(item["telemetry"]["endpoint_call_counts"].get(
                endpoint_id, 0)) for item in summaries)
            for endpoint_id in ENDPOINT_IDS},
        "api_attempts": sum(int(item["telemetry"]["api_attempts"])
                            for item in summaries),
        "errors": sum(int(item["telemetry"]["errors"])
                      for item in summaries),
        "input_tokens": sum(int(item["telemetry"]["input_tokens"])
                            for item in summaries),
        "output_tokens": sum(int(item["telemetry"]["output_tokens"])
                             for item in summaries),
        "fresh_timing_valid": not any(
            item.get("resumed_from_local_cache") for item in summaries),
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule, rubric = _load_frozen(config, output)
    _require(target, STAGE_RETRY)
    logical = load_json(target / "retry" / "combined" / "logical_votes.json")
    control_logical = load_json(_control_logical_path())
    treatment_metrics = phase10._system_metrics(
        records, logical["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"])
    initial_metrics = phase10._system_metrics(
        records, control_logical["systems"][control.INITIAL_SYSTEM]["votes_by_replicate"])
    phase10_metrics = phase10._system_metrics(
        records, control_logical["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"])
    paired = {
        "phase10_to_prompt_v2_evolved": legacy._paired(
            records, phase10_metrics["original_index_predictions"],
            treatment_metrics["original_index_predictions"]),
        "initial_to_prompt_v2_evolved": legacy._paired(
            records, initial_metrics["original_index_predictions"],
            treatment_metrics["original_index_predictions"]),
    }
    external_metrics = {}
    for name, (report_path, metric_name) in EXTRA_BASELINE_REPORTS.items():
        item = load_json(report_path)["metrics"][metric_name]
        external_metrics[name] = item
        paired[f"{name}_to_prompt_v2_evolved"] = legacy._paired(
            records, item["original_index_predictions"],
            treatment_metrics["original_index_predictions"])
    predictions = tuple(PairwisePredictionOutput.load_json(
        control._prediction_path(target / "retry", replicate))
        for replicate in range(legacy.K))
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "rubric_sha256": rubric.rubric_sha256,
        "node_count": len(rubric.nodes),
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "metrics": {
            "initial_five_root_prompt_v2": initial_metrics,
            "phase10_final_equal_prompt_v2": phase10_metrics,
            **external_metrics,
            TREATMENT_SYSTEM: treatment_metrics,
        },
        "paired": paired,
        "node_metrics": control._node_metrics(
            records, schedule, rubric, predictions),
        "position_metrics": control._position_metrics(
            records, schedule,
            logical["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"]),
        "retry": load_json(target / "retry" / "report.json"),
        "execution": _execution_summary(target, len(rubric.nodes)),
        "selection_after_benchmark_forbidden": True,
    }
    atomic_write_json(target / "final_report.json", value)
    lines = ["# VL-RewardBench Prompt-v2-evolved Rubric", "",
             "Exploratory K=3 paired evaluation.", "",
             "| System | OverallAcc | MacroAcc | Coverage | Strict ACC |",
             "|---|---:|---:|---:|---:|"]
    for name, item in value["metrics"].items():
        lines.append(f"| {name} | {item['overall_acc']:.4f} | "
                     f"{item['macro_acc']:.4f} | {item['coverage']:.4f} | "
                     f"{item['strict_accuracy']:.4f} |")
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    details = {
        "control_overall_acc": phase10_metrics["overall_acc"],
        "treatment_overall_acc": treatment_metrics["overall_acc"],
        "control_macro_acc": phase10_metrics["macro_acc"],
        "treatment_macro_acc": treatment_metrics["macro_acc"],
        "net_corrected": paired["phase10_to_prompt_v2_evolved"]["net_corrected"],
        "unresolved_technical_failures": value["retry"]["still_failed_count"],
    }
    _status(target, STAGE_REPORT, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {STAGE_FREEZE: freeze, STAGE_AUDIT: audit, STAGE_SMOKE: smoke,
               STAGE_RUN: run, STAGE_RETRY: retry, STAGE_REPORT: report}
    if stage not in actions:
        raise ValueError(f"unsupported VL-RB Prompt-v2-evolved stage: {stage}")
    started = time.monotonic(); actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

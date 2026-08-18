"""K=3 VL-RewardBench transfer evaluation for the five-root Child-Gate.

The Phase10 Prompt-v2 Pairwise predictions are immutable inputs.  Each of the
three counterbalanced replicates receives an independent Gate execution with a
replicate-specific seed and cache namespace.  All five roots remain active;
only direct-child activation changes.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import PairwisePredictionOutput, StructuredRubric

from . import full_child_gate as gate
from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from . import vl_rewardbench_phase10 as phase10
from . import vl_rewardbench_prompt_v2 as prompt_v2
from .experiment_utils import atomic_write_json, canonical_sha256, load_json
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_phase14_full_child_gate_v1"
PROTOCOL_VERSION = "vlrb-full-child-gate-v1"
SCHEMA_VERSION = "1.0.0"
SOURCE_EXPERIMENT = prompt_v2.EXPERIMENT_DIR
SOURCE_STAGE = "retry"
SOURCE_SYSTEM = prompt_v2.EQUAL_SYSTEM
REPLICATE_SEEDS = (42, 43, 44)

STAGE_FREEZE = "vlrb-child-gate-freeze"
STAGE_AUDIT = "vlrb-child-gate-audit"
STAGE_SMOKE = "vlrb-child-gate-smoke"
STAGE_RUN = "vlrb-child-gate-run"
STAGE_RETRY = "vlrb-child-gate-retry"
STAGE_REPORT = "vlrb-child-gate-report"


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _status_path(target: Path) -> Path:
    return target / "stage_status.json"


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    path = _status_path(target)
    value = load_json(path) if path.is_file() else {}
    value[stage] = {"status": "passed", **dict(details)}
    atomic_write_json(path, value)


def _require(target: Path, stage: str) -> None:
    value = load_json(_status_path(target)) if _status_path(target).is_file() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_stage": SOURCE_STAGE,
        "source_system": SOURCE_SYSTEM,
        "source_rubric_sha256": gate.vg.SOURCE_RUBRIC_SHA256,
        "root_policy": "all_roots_always_active",
        "routing_scope": "per_root_direct_children_only",
        "gate_endpoints": ["vllm-8000", "vllm-8001"],
        "gate_temperature": 0.2,
        "gate_max_tokens": 2048,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "gate_max_retries": 5,
        "always_format_reminder": True,
        "uncertain_policy": "activate",
        "invalid_policy": "all_children_for_failed_root",
        "smoke_sample_count": 20,
        "k": legacy.K,
        "counterbalance_seed": legacy.SEED,
        "cross_replicate_cache_reuse": False,
        "selection_after_benchmark_forbidden": True,
    }
    value = config.get("vlrb_full_child_gate_experiment")
    if value != expected:
        raise RuntimeError(
            "vlrb_full_child_gate_experiment does not match the frozen v1 protocol")
    return dict(value)


def _gate_protocol(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Adapt the frozen Gate protocol without sharing a seed across replicates."""

    value = gate._protocol(config)
    value["gate_seed"] = seed
    value["prompt_version"] = gate.PROMPT_VERSION
    return value


def _rubric(output: Path) -> StructuredRubric:
    return gate._rubric(output)


def _records():
    return prompt_v2._records()


def _schedule(records):
    return legacy._order_schedule(records)


def _source_target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / SOURCE_EXPERIMENT


def _source_prediction_path(replicate: int) -> Path:
    return prompt_v2._prediction_path(_source_target() / SOURCE_STAGE, replicate)


def _source_prediction(output: Path, records, schedule, rubric, replicate: int):
    path = _source_prediction_path(replicate)
    if not path.is_file():
        raise RuntimeError(f"Prompt-v2 source prediction is missing: {path}")
    prediction = PairwisePredictionOutput.load_json(path)
    rows = legacy._ordered_rows(records, schedule, replicate)
    source_manifest = load_json(_source_target() / "frozen_manifest.json")
    prompt_v2._validate_prediction(prediction, rubric, rows, source_manifest)
    return prediction


def _subset_prediction(prediction: PairwisePredictionOutput, count: int):
    return replace(
        prediction,
        sample_ids=prediction.sample_ids[:count],
        sample_fingerprints=prediction.sample_fingerprints[:count],
        node_outputs=prediction.node_outputs[:count],
        flat_answers=prediction.flat_answers[:count],
    )


def _replicate_request_specs(config, protocol, contracts, spec):
    return {
        f"replicate_{index + 1:02d}": gate._request_specs(
            config, _gate_protocol(config, seed), contracts, spec)
        for index, seed in enumerate(protocol["replicate_seeds"])
    }


def _manifest(config: Mapping[str, Any], output: Path, *, inspect_live: bool):
    protocol = _protocol(config)
    rubric = _rubric(output)
    records = _records()
    schedule = _schedule(records)
    contracts = gate._contracts(rubric)
    spec = gate.vg._gate_pool_spec(config, protocol["gate_endpoints"])
    target = _target()
    prior = target / "frozen_manifest.json"
    identities = gate.vg._inspect_endpoints(config, spec) if inspect_live else None
    if identities is None and prior.is_file():
        identities = load_json(prior)["gate_endpoint_identities"]
    source_manifest = _source_target() / "frozen_manifest.json"
    source_logical = _source_target() / SOURCE_STAGE / "combined/logical_votes.json"
    source_report = _source_target() / "final_report.json"
    paths = [_source_prediction_path(index) for index in range(legacy.K)]
    for path in (source_manifest, source_logical, source_report, *paths):
        if not path.is_file():
            raise RuntimeError(f"required Prompt-v2 source artifact is missing: {path}")
    logical = load_json(source_logical)
    if (logical.get("sample_ids") != [str(item["sample_id"]) for item in records]
            or logical.get("k") != legacy.K
            or SOURCE_SYSTEM not in logical.get("systems", {})):
        raise RuntimeError("Prompt-v2 source logical artifact identity drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_DIR,
        "protocol": protocol,
        "dataset": {
            "sample_count": len(records),
            "record_sha256": canonical_sha256(list(records)),
            "sample_ids": [str(item["sample_id"]) for item in records],
            "order_schedule_sha256": canonical_sha256(schedule),
        },
        "source": {
            "rubric_path": str(gate.vg._rubric_path(output).resolve()),
            "rubric_file_sha256": file_sha256(gate.vg._rubric_path(output)),
            "rubric_sha256": rubric.rubric_sha256,
            "node_count": len(rubric.nodes),
            "root_count": len(rubric.root_ids),
            "root_ids": list(rubric.root_ids),
            "child_count": len(rubric.edges),
            "prompt_v2_manifest_path": str(source_manifest.resolve()),
            "prompt_v2_manifest_sha256": file_sha256(source_manifest),
            "prompt_v2_logical_path": str(source_logical.resolve()),
            "prompt_v2_logical_sha256": file_sha256(source_logical),
            "prompt_v2_report_path": str(source_report.resolve()),
            "prompt_v2_report_sha256": file_sha256(source_report),
            "prediction_paths": [str(path.resolve()) for path in paths],
            "prediction_sha256": [file_sha256(path) for path in paths],
        },
        "routing_contracts": {
            root_id: {
                "contract_sha256": item["contract_sha256"],
                "child_ids": [child["node_id"] for child in item["children"]],
            }
            for root_id, item in contracts.items()
        },
        "replicate_gate_request_specs": _replicate_request_specs(
            config, protocol, contracts, spec),
        "gate_endpoint_identities": identities,
        "cross_replicate_cache_reuse": False,
    }


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target()
    _require(target, STAGE_FREEZE)
    stored = load_json(target / "frozen_manifest.json")
    expected = _manifest(config, output, inspect_live=False)
    if stored != expected:
        raise RuntimeError("VL-RewardBench Child-Gate frozen manifest drift")
    rubric = _rubric(output)
    records = _records()
    schedule = _schedule(records)
    contracts = gate._contracts(rubric)
    spec = gate.vg._gate_pool_spec(config, stored["protocol"]["gate_endpoints"])
    return target, stored, rubric, records, schedule, contracts, spec


def _verify_live(config, manifest, spec):
    if gate.vg._inspect_endpoints(config, spec) != manifest["gate_endpoint_identities"]:
        raise RuntimeError("VL-RewardBench Child-Gate endpoint identity drift")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target()
    value = _manifest(config, output, inspect_live=True)
    path = target / "frozen_manifest.json"
    if path.is_file() and load_json(path) != value:
        status = load_json(_status_path(target)) if _status_path(target).is_file() else {}
        if any(status.get(stage, {}).get("status") == "passed" for stage in (
                STAGE_AUDIT, STAGE_SMOKE, STAGE_RUN, STAGE_RETRY, STAGE_REPORT)):
            raise RuntimeError("Child-Gate manifest drift after downstream execution")
    atomic_write_json(path, value)
    atomic_write_json(target / "order_schedule.json", _schedule(_records()))
    rubric = _rubric(output)
    for root_id, contract in gate._contracts(rubric).items():
        atomic_write_json(target / "routing_contracts" / f"{root_id}.json", contract)
    details = {
        "sample_count": value["dataset"]["sample_count"],
        "k": legacy.K,
        "logical_gate_requests": value["dataset"]["sample_count"] * legacy.K * 5,
        "replicate_seeds": value["protocol"]["replicate_seeds"],
    }
    _status(target, STAGE_FREEZE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _display_to_original(labels, records, schedule, replicate):
    return [
        legacy._original_index(label, int(schedule[str(record["sample_id"])][replicate]))
        for label, record in zip(labels, records)
    ]


def _source_equal_votes():
    logical = load_json(
        _source_target() / SOURCE_STAGE / "combined/logical_votes.json")
    return logical["systems"][SOURCE_SYSTEM]["votes_by_replicate"]


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(
        config, output)
    del contracts, spec
    replay = []
    for replicate in range(legacy.K):
        prediction = _source_prediction(output, records, schedule, rubric, replicate)
        selections = gate._constant_selection(rubric, len(records), all_children=True)
        _roots, labels = gate._selection_labels(rubric, prediction, selections)
        replay.append(_display_to_original(labels, records, schedule, replicate))
    source = _source_equal_votes()
    if replay != source:
        raise RuntimeError("all-children replay differs from Prompt-v2 source votes")
    report = load_json(_source_target() / "final_report.json")
    replay_metrics = phase10._system_metrics(records, replay)
    source_metrics = report["new_metrics"][SOURCE_SYSTEM]
    for key in ("overall_acc", "macro_acc", "coverage", "strict_accuracy"):
        if replay_metrics[key] != source_metrics[key]:
            raise RuntimeError(f"Prompt-v2 source metric replay drift: {key}")
    details = {
        "sample_count": len(records),
        "all_children_vote_exact_match": True,
        "overall_acc": replay_metrics["overall_acc"],
        "macro_acc": replay_metrics["macro_acc"],
        "cross_replicate_cache_reuse": manifest["cross_replicate_cache_reuse"],
    }
    atomic_write_json(target / "offline_audit.json", details)
    _status(target, STAGE_AUDIT, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def _run_replicates(config, output, target, manifest, rubric, records, schedule,
                    contracts, spec, *, work_name: str, sample_count: int | None):
    selected = tuple(records[:sample_count] if sample_count is not None else records)
    artifacts = []
    started = time.perf_counter()
    for replicate, seed in enumerate(manifest["protocol"]["replicate_seeds"]):
        rows = legacy._ordered_rows(selected, schedule, replicate)
        protocol = _gate_protocol(config, seed)
        expected_specs = manifest["replicate_gate_request_specs"][
            f"replicate_{replicate + 1:02d}"]
        if gate._request_specs(config, protocol, contracts, spec) != expected_specs:
            raise RuntimeError(f"Gate request identity drift for replicate {replicate + 1}")
        work = target / work_name / "gate" / f"replicate_{replicate + 1:02d}"
        # The cache directory is nested under the replicate work directory;
        # no path is shared across independent K=3 samples.
        artifact = gate._run_all_roots(
            config, work, {
                **manifest,
                "gate_request_specs": expected_specs,
            }, protocol, contracts, spec, rows, work,
            f"vlrb_child_gate_replicate_{replicate + 1:02d}",
            cache_root=work / "cache/gate",
            experiment_name=EXPERIMENT_DIR,
        )
        artifacts.append(artifact)
    return tuple(artifacts), time.perf_counter() - started


def _artifact_failures(artifacts):
    return [
        {"replicate": replicate + 1, "root_id": root_id,
         "sample_id": item["sample_id"], "parse_error": item["parse_error"]}
        for replicate, artifact in enumerate(artifacts)
        for root_id, root in artifact["roots"].items()
        for item in root["samples"] if not item["parse_ok"]
    ]


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(
        config, output)
    _require(target, STAGE_AUDIT)
    _verify_live(config, manifest, spec)
    count = manifest["protocol"]["smoke_sample_count"]
    artifacts, wall = _run_replicates(
        config, output, target, manifest, rubric, records, schedule,
        contracts, spec, work_name="smoke", sample_count=count)
    calls = {
        endpoint_id: sum(
            int(item["summary"]["endpoint_call_counts"].get(endpoint_id, 0))
            for item in artifacts)
        for endpoint_id in manifest["protocol"]["gate_endpoints"]
    }
    failures = _artifact_failures(artifacts)
    if failures:
        raise RuntimeError("Child-Gate smoke parse check failed")
    fresh_calls = sum(calls.values())
    if (fresh_calls == count * legacy.K * len(rubric.root_ids)
            and any(value == 0 for value in calls.values())):
        raise RuntimeError("fresh Child-Gate smoke did not use both endpoints")
    cache_hits = [item["summary"]["cache_hit_count"] for item in artifacts]
    details = {
        "sample_count": count,
        "logical_gate_requests": count * legacy.K * len(rubric.root_ids),
        "parse_valid_rate": 1.0,
        "endpoint_call_counts": calls,
        "cache_hits_by_replicate": cache_hits,
        "replicate_cache_roots": [
            str((target / "smoke/gate" / f"replicate_{index + 1:02d}"
                 / "cache/gate").resolve())
            for index in range(legacy.K)
        ],
        "wall_seconds": wall,
    }
    atomic_write_json(target / "smoke/summary.json", details)
    _status(target, STAGE_SMOKE, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(
        config, output)
    _require(target, STAGE_SMOKE)
    _verify_live(config, manifest, spec)
    artifacts, wall = _run_replicates(
        config, output, target, manifest, rubric, records, schedule,
        contracts, spec, work_name="run", sample_count=None)
    failures = _artifact_failures(artifacts)
    details = {
        "sample_count": len(records),
        "logical_gate_requests": len(records) * legacy.K * len(rubric.root_ids),
        "parse_failure_count": len(failures),
        "wall_seconds": wall,
    }
    atomic_write_json(target / "run/failure_manifest.json", failures)
    _status(target, STAGE_RUN, details)
    print(json.dumps(details, indent=2, ensure_ascii=False))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(
        config, output)
    _require(target, STAGE_RUN)
    _verify_live(config, manifest, spec)
    # Preserve the original run telemetry.  Retry artifacts are written to a
    # separate tree, while each replicate first copies its private run cache.
    # Failed cache entries are then retried by ``_route_one`` and overwritten.
    for replicate in range(legacy.K):
        source = (target / "run/gate" / f"replicate_{replicate + 1:02d}"
                  / "cache/gate")
        destination = (target / "retry/gate" / f"replicate_{replicate + 1:02d}"
                       / "cache/gate")
        if not destination.exists():
            shutil.copytree(source, destination)
    artifacts, wall = _run_replicates(
        config, output, target, manifest, rubric, records, schedule,
        contracts, spec, work_name="retry", sample_count=None)
    failures = _artifact_failures(artifacts)
    details = {
        "sample_count": len(records),
        "still_failed_count": len(failures),
        "parse_valid_rate": 1.0 - len(failures) / (len(records) * legacy.K * 5),
        "retry_wall_seconds": wall,
        "failed_items": failures,
    }
    atomic_write_json(target / "retry/report.json", details)
    _status(target, STAGE_RETRY, details)
    print(json.dumps({key: details[key] for key in (
        "still_failed_count", "parse_valid_rate", "retry_wall_seconds")},
        indent=2, ensure_ascii=False))


def _load_artifacts(target, manifest, records, schedule, *, work_name="retry"):
    artifacts = []
    for replicate in range(legacy.K):
        rows = legacy._ordered_rows(records, schedule, replicate)
        path = (target / work_name / "gate" / f"replicate_{replicate + 1:02d}"
                / "gate_predictions.json")
        artifacts.append(gate._load_gate(path, {
            **manifest,
            "gate_request_specs": manifest["replicate_gate_request_specs"][
                f"replicate_{replicate + 1:02d}"],
        }, rows))
    return tuple(artifacts)


def _system_votes(output, rubric, records, schedule, predictions, artifacts):
    names = ["parent_only", "all_children", "full_child_gate",
             "oracle_child_routing_upper_bound"]
    names.extend(f"single_root_gate::{root_id}" for root_id in rubric.root_ids)
    votes = {name: [] for name in names}
    diagnostics = []
    for replicate, (prediction, artifact) in enumerate(zip(predictions, artifacts)):
        rows = legacy._ordered_rows(records, schedule, replicate)
        parent = gate._constant_selection(rubric, len(rows), all_children=False)
        all_children = gate._constant_selection(rubric, len(rows), all_children=True)
        gated = gate._selection_from_gate(artifact, rubric.root_ids)
        oracle = gate._oracle_selection(rubric, prediction, rows)
        selections = {
            "parent_only": parent,
            "all_children": all_children,
            "full_child_gate": gated,
            "oracle_child_routing_upper_bound": oracle,
        }
        for root_id in rubric.root_ids:
            single = [dict(item) for item in all_children]
            for index in range(len(single)):
                single[index][root_id] = gated[index][root_id]
            selections[f"single_root_gate::{root_id}"] = single
        for name, selected in selections.items():
            _root_labels, labels = gate._selection_labels(rubric, prediction, selected)
            votes[name].append(_display_to_original(labels, records, schedule, replicate))
        diagnostics.append(gate._routing_diagnostics(
            rubric, prediction, rows, artifact, gated))
    return votes, diagnostics


def _routing_summary(diagnostics, artifacts, sample_count):
    logical = sample_count * legacy.K * 5
    failures = _artifact_failures(artifacts)
    total_active = sum(
        item["mean_active_children_total"] * sample_count for item in diagnostics)
    all_conflicts = sum(
        root["all_children_sibling_conflict_count"]
        for item in diagnostics for root in item["roots"].values())
    gated_conflicts = sum(
        root["gated_sibling_conflict_count"]
        for item in diagnostics for root in item["roots"].values())
    roots = {}
    for root_id in diagnostics[0]["roots"]:
        per_child = {}
        child_ids = diagnostics[0]["roots"][root_id]["per_child"]
        for child_id in child_ids:
            active = sum(item["roots"][root_id]["per_child"][child_id]["active"]
                         for item in diagnostics)
            decisive = sum(item["roots"][root_id]["per_child"][child_id]["decisive"]
                           for item in diagnostics)
            correct = sum(item["roots"][root_id]["per_child"][child_id]["correct"]
                          for item in diagnostics)
            per_child[child_id] = {
                "active": active,
                "decisive": decisive,
                "correct": correct,
                "accuracy_on_active_decisive": (
                    correct / decisive if decisive else 0.0),
            }
        roots[root_id] = {
            "mean_active_children": sum(
                item["roots"][root_id]["mean_active_children"]
                for item in diagnostics) / legacy.K,
            "empty_route_count": sum(
                item["roots"][root_id]["empty_route_count"]
                for item in diagnostics),
            "all_children_sibling_conflict_count": sum(
                item["roots"][root_id]["all_children_sibling_conflict_count"]
                for item in diagnostics),
            "gated_sibling_conflict_count": sum(
                item["roots"][root_id]["gated_sibling_conflict_count"]
                for item in diagnostics),
            "per_child": per_child,
        }
    return {
        "logical_route_count": logical,
        "parse_valid_count": logical - len(failures),
        "parse_valid_rate": 1.0 - len(failures) / logical,
        "parse_failure_count": len(failures),
        "mean_active_children_total": total_active / (sample_count * legacy.K),
        "activation_reduction_rate": 1.0 - total_active / (
            sample_count * legacy.K * 17),
        "all_children_sibling_conflict_count": all_conflicts,
        "gated_sibling_conflict_count": gated_conflicts,
        "sibling_conflict_reduction": all_conflicts - gated_conflicts,
        "roots": roots,
        "replicates": diagnostics,
    }


def _gate_stability(artifacts, rubric):
    """Compare independent routing samples after aligning by original pair ID."""

    roots = {}
    for root_id in rubric.root_ids:
        child_ids = tuple(
            child.node_id for child in rubric.children(root_id))
        samples = [artifact["roots"][root_id]["samples"] for artifact in artifacts]

        def status(item, child_id):
            if not item["parse_ok"]:
                return "invalid"
            return item["decision"]["decisions"][child_id]["status"]

        def pair(left, right):
            exact = 0
            jaccard = 0.0
            per_child = {child_id: 0 for child_id in child_ids}
            for a, b in zip(samples[left], samples[right]):
                vector_a = tuple(status(a, child_id) for child_id in child_ids)
                vector_b = tuple(status(b, child_id) for child_id in child_ids)
                exact += vector_a == vector_b
                active_a, active_b = set(a["active_child_ids"]), set(b["active_child_ids"])
                union = active_a | active_b
                jaccard += len(active_a & active_b) / len(union) if union else 1.0
                for child_id in child_ids:
                    per_child[child_id] += status(a, child_id) == status(b, child_id)
            count = len(samples[left])
            return {
                "sample_count": count,
                "exact_status_vector_match_rate": exact / count,
                "mean_active_set_jaccard": jaccard / count,
                "per_child_status_agreement": {
                    child_id: value / count for child_id, value in per_child.items()
                },
            }

        roots[root_id] = {
            "replicate_01_vs_03_same_order_independent": pair(0, 2),
            "replicate_01_vs_02_swapped_order": pair(0, 1),
        }
    return {"roots": roots}


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric, records, schedule, contracts, spec = _load_frozen(
        config, output)
    del contracts, spec
    _require(target, STAGE_RETRY)
    predictions = tuple(
        _source_prediction(output, records, schedule, rubric, replicate)
        for replicate in range(legacy.K))
    artifacts = _load_artifacts(target, manifest, records, schedule)
    votes, diagnostics = _system_votes(
        output, rubric, records, schedule, predictions, artifacts)
    if votes["all_children"] != _source_equal_votes():
        raise RuntimeError("report all-children votes differ from Prompt-v2 source")
    metrics = {
        name: phase10._system_metrics(records, value)
        for name, value in votes.items()
    }
    baseline = metrics["all_children"]["original_index_predictions"]
    treatment = metrics["full_child_gate"]["original_index_predictions"]
    parent = metrics["parent_only"]["original_index_predictions"]
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "selection_after_benchmark_forbidden": True,
        "frozen_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
        "metrics": metrics,
        "paired": {
            "all_children_to_full_child_gate": legacy._paired(
                records, baseline, treatment),
            "parent_only_to_full_child_gate": legacy._paired(
                records, parent, treatment),
        },
        "routing": _routing_summary(diagnostics, artifacts, len(records)),
        "gate_stability": _gate_stability(artifacts, rubric),
        "system_position_metrics": {
            name: prompt_v2._position_metrics(records, schedule, system_votes)
            for name, system_votes in votes.items()
        },
        "source_pairwise_model_calls": 0,
        "oracle_uses_gold_and_is_diagnostic_only": True,
    }
    atomic_write_json(target / "final_report.json", value)
    lines = [
        "# VL-RewardBench Full Child-Gate v1", "",
        "Exploratory K=3 transfer evaluation. Pairwise predictions are frozen; "
        "each Gate replicate is independently sampled.", "",
        "| System | OverallAcc | MacroAcc | Coverage | Strict ACC |",
        "|---|---:|---:|---:|---:|",
    ]
    order = ["parent_only", "all_children", "full_child_gate",
             "oracle_child_routing_upper_bound"]
    order.extend(f"single_root_gate::{root_id}" for root_id in rubric.root_ids)
    for name in order:
        item = metrics[name]
        lines.append(
            f"| {name} | {item['overall_acc']:.4f} | {item['macro_acc']:.4f} | "
            f"{item['coverage']:.4f} | {item['strict_accuracy']:.4f} |")
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    details = {
        "all_children_overall_acc": metrics["all_children"]["overall_acc"],
        "full_child_gate_overall_acc": metrics["full_child_gate"]["overall_acc"],
        "full_child_gate_macro_acc": metrics["full_child_gate"]["macro_acc"],
        "net_corrected": value["paired"][
            "all_children_to_full_child_gate"]["net_corrected"],
        "parse_valid_rate": value["routing"]["parse_valid_rate"],
        "activation_reduction_rate": value["routing"]["activation_reduction_rate"],
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
        raise ValueError(f"unsupported VL-RewardBench Child-Gate stage: {stage}") from exc

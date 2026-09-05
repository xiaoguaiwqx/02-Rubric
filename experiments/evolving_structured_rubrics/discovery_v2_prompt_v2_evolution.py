"""Discovery-v2 Prompt-v2 Locked-Split + Role-aware Refine experiment.

Discovery100 is the only optimization split.  Dev150 is evaluated from frozen
epoch snapshots and is deliberately absent from every Manager/trigger/fitness
input.  RLHF-V heldout-500 remains an exploratory regression check because the
current demo export contains known source overlap; VL-RewardBench is the main
external evaluation.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    EvolutionContext,
    PairwisePredictionOutput,
    RubricFeedback,
    StructuredRubric,
    detect_specialize_trigger,
    execute_offline_m1,
    project_pairwise_prediction,
)
from critiq.structured.version import (
    PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
)

from . import discovery_data_v2 as discovery_data
from . import prompt_v2_aligned_evolution as phase16
from . import refine_evolution as refine
from . import run_rubric_evolution as base
from . import split_evolution as split
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, load_jsonl_dataset
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "phase17_discovery_v2_prompt_v2_split_refine_v1"
PROTOCOL_VERSION = "discovery-v2-prompt-v2-locked-split-refine-v1"
PAIRWISE_PROMPT_MODE = "v2_cache"
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")
DISCOVERY_COUNT = 100
DEV_COUNT = 150
HELDOUT_COUNT = 500
DISCOVERY_PATH = "data/discovery_v2_demo_v3/discovery_100.jsonl"
DEV_PATH = "data/discovery_v2_demo_v3/dev_150.jsonl"

SETTINGS = {
    **refine.FIVE_ROOT_LOCKED_SPLIT_REFINE_V1,
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": "independent_initial_five_roots",
    "discovery_dataset": DISCOVERY_PATH,
    "dev_dataset": DEV_PATH,
    "discovery_count": DISCOVERY_COUNT,
    "dev_count": DEV_COUNT,
    "dev_policy": "diagnostic_only_no_selection",
    "worker_endpoint": "configured_available_slot_pool",
    "worker_endpoints": list(ENDPOINT_IDS),
    "pairwise_prompt_mode": PAIRWISE_PROMPT_MODE,
    "pairwise_prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    "split_trigger": {"tau_split": 0.75, "tau_cov_high": 0.80},
    "error_signature_policy": "fresh_discovery100_only",
    "heldout_policy": "exploratory_overlap_recorded",
    "vl_rewardbench_required": True,
}

PROTOCOL = split.EvolutionProtocol(
    EXPERIMENT_DIR,
    "discovery-v2-evolution",
    rubric_memory_mode="global_rubric_v1",
    control_experiment_dir=None,
    read_only_control_signatures=False,
    pairwise_endpoint="vllm-8000",
    allow_configured_endpoint_pool=True,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _heldout_reference_output(config: Mapping[str, Any], output: Path) -> Path:
    value = config.get("discovery_v2_heldout_reference_output")
    if value is None:
        return output
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "discovery_v2_heldout_reference_output must be a non-empty path")
    return base._path(value)


def _protocol(config: Mapping[str, Any]) -> split.EvolutionProtocol:
    manager_models = {
        str(item["model"]) for item in config["specialize_managers"].values()}
    manager_models.add(str(config["refine_manager"]["model"]))
    if len(manager_models) != 1:
        raise ValueError("Phase17 requires one shared model for every Manager role")
    return replace(
        PROTOCOL,
        manager_model=manager_models.pop(),
        pairwise_endpoint=config["backend_pool"]["endpoints"][0]["endpoint_id"],
    )


def _same_pairwise_scientific_request_identity(
    source: PairwisePredictionOutput,
    generated: PairwisePredictionOutput,
) -> bool:
    """Compare Pairwise identities while treating the execution pool as provenance.

    A combined offline artifact may reuse old node outputs and add newly generated
    node outputs. The two batches remain scientifically compatible when their
    model, prompt, parser, semantics, input contract, and decoding configuration
    match. The backend pool ID identifies where a batch ran and is preserved in
    ``prediction_merge_provenance.json`` instead of being treated as prompt
    semantics.
    """

    if (
        source.semantics_version != generated.semantics_version
        or source.schema_version != generated.schema_version
        or source.prompt_version != generated.prompt_version
        or source.parser_version != generated.parser_version
    ):
        return False
    source_spec = source.request_spec.to_dict()
    generated_spec = generated.request_spec.to_dict()
    source_spec.pop("backend_id")
    generated_spec.pop("backend_id")
    return source_spec == generated_spec


def _normalize_generated_request_for_merge(
    source: PairwisePredictionOutput,
    generated: PairwisePredictionOutput,
) -> PairwisePredictionOutput:
    """Normalize a compatible generated batch to the composite source spec."""

    if generated.request_spec == source.request_spec:
        return generated
    if not _same_pairwise_scientific_request_identity(source, generated):
        raise ValueError(
            "heldout generated/source scientific request identity mismatch"
        )
    return PairwisePredictionOutput(
        generated.sample_ids,
        generated.sample_fingerprints,
        generated.criteria,
        generated.node_outputs,
        generated.flat_answers,
        source.request_spec,
        semantics_version=generated.semantics_version,
        schema_version=generated.schema_version,
        prompt_version=generated.prompt_version,
        parser_version=generated.parser_version,
    )


def _dataset_paths() -> dict[str, str]:
    return {"discovery": DISCOVERY_PATH, "dev": DEV_PATH}


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["_pairwise_prompt_mode"] = PAIRWISE_PROMPT_MODE
    value["_experiment_dataset_paths"] = _dataset_paths()
    value["_experiment_dataset_counts"] = {
        "discovery": DISCOVERY_COUNT, "dev": DEV_COUNT,
        "heldout": HELDOUT_COUNT,
    }
    value["evolution_policy"]["trigger_thresholds"]["tau_split"] = 0.75
    experiment = config.get("discovery_v2_prompt_v2_evolution", {})
    value["_manager_compact_sample_ids"] = bool(
        experiment.get("compact_manager_sample_ids", False))
    value["evolution_policy"]["trigger_thresholds"]["N_min_cluster"] = int(
        experiment.get(
            "split_min_cluster_size",
            value["evolution_policy"]["trigger_thresholds"]["N_min_cluster"],
        )
    )
    return value


def _config(config: Mapping[str, Any]) -> dict[str, Any]:
    refine._config(config)
    value = config.get("discovery_v2_prompt_v2_evolution")
    if not isinstance(value, Mapping):
        raise ValueError(
            "discovery_v2_prompt_v2_evolution must equal the frozen v1 protocol")
    expected = dict(SETTINGS)
    configured_settings = dict(value)
    expected.pop("worker_endpoints")
    configured_endpoints = configured_settings.pop("worker_endpoints", None)
    split_min_cluster_size = configured_settings.pop(
        "split_min_cluster_size", 5)
    compact_manager_sample_ids = configured_settings.pop(
        "compact_manager_sample_ids", False)
    if (isinstance(split_min_cluster_size, bool)
            or not isinstance(split_min_cluster_size, int)
            or split_min_cluster_size < 1):
        raise ValueError("split_min_cluster_size must be a positive integer")
    if not isinstance(compact_manager_sample_ids, bool):
        raise ValueError("compact_manager_sample_ids must be bool")
    if configured_settings != expected:
        raise ValueError(
            "discovery_v2_prompt_v2_evolution must equal the frozen v1 protocol")
    request = config.get("worker_request_kwargs")
    if (not isinstance(request, Mapping)
            or request.get("temperature") != 0.5
            or request.get("max_tokens") != 2048):
        raise ValueError("Phase17 requires temperature=.5/max_tokens=2048")
    pool = base.BackendPoolSpec.from_dict(config["backend_pool"])
    pool_endpoint_ids = [item.endpoint_id for item in pool.endpoints]
    if configured_endpoints != pool_endpoint_ids:
        raise ValueError(
            "Phase17 worker_endpoints must match backend_pool endpoint order")
    if not configured_endpoints:
        raise ValueError("Phase17 requires at least one Worker endpoint")
    if pool.global_request_concurrency != sum(
            item.max_concurrency for item in pool.endpoints):
        raise ValueError("Phase17 requires the full available-slot pool capacity")
    # The repository-wide Split-v1 policy remains frozen at .70; Phase16/17
    # broaden scheduling to .75 only in their private runtime config.
    split.validate_policy(config, PROTOCOL)
    return dict(value)


def _rows(config: Mapping[str, Any], split_name: str):
    return refine._rows(_runtime_config(config), split_name)


def _manager_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime_config(config)
    protocol = _protocol(runtime)
    managers, profiles, specs, identities = split._managers(runtime, protocol)
    refine_manager, refine_profile = refine._manager(
        runtime, expected_model=protocol.manager_model)
    _, retry_specs = refine._integrated_retry_specs(runtime, protocol)
    return {
        "profiles": profiles,
        "specs": specs,
        "identities": identities,
        "refine_profile": refine_profile,
        "refine_specs": {
            key: spec.to_dict()
            for key, spec in refine_manager.request_specs().items()},
        "retry_specs": retry_specs,
    }


def _file_sha(path: str | Path) -> str:
    return file_sha256(base._path(path))


def _load_frozen(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target = _target(output)
    path = target / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("run discovery-v2-evolution-freeze first")
    manifest = load_json(path)
    current = {
        "discovery": _file_sha(DISCOVERY_PATH),
        "dev": _file_sha(DEV_PATH),
        "heldout": _file_sha(config["heldout_dataset"]),
    }
    drift = {name: (manifest["datasets"][name]["sha256"], digest)
             for name, digest in current.items()
             if manifest["datasets"][name]["sha256"] != digest}
    if drift:
        raise RuntimeError(f"Phase17 frozen dataset drift: {drift}")
    return manifest


def _heldout_overlap(rows: Sequence[Mapping[str, Any]],
                     heldout: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    heldout_images: dict[str, Mapping[str, Any]] = {}
    for row in heldout:
        path = base._path(str(row["image_path"]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        heldout_images[digest] = row
    matches = []
    for row in rows:
        other = heldout_images.get(str(row["image_sha256"]))
        if other is not None:
            matches.append({
                "sample_id": str(row["sample_id"]),
                "heldout_sample_id": str(other["sample_id"]),
                "same_question": str(row["question"]).strip()
                == str(other["question"]).strip(),
            })
    return {"count": len(matches), "matches": matches}


def _vlrb_overlap(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    cfg = config["discovery_data_v2"]
    exclusion = discovery_data._benchmark_exclusion(
        base._path(cfg["vl_rewardbench_path"]))
    values = {
        "sample_id": sum(str(row["sample_id"]) in exclusion["sample_ids"]
                         for row in rows),
        "image_sha256": sum(str(row["image_sha256"]) in exclusion["image_sha256"]
                            for row in rows),
        "question_sha256": sum(str(row["question_sha256"])
                               in exclusion["question_sha256"] for row in rows),
        "unordered_pair_sha256": sum(str(row["unordered_pair_sha256"])
                                     in exclusion["unordered_pair_sha256"]
                                     for row in rows),
    }
    return values


def _cross_overlap(left: Sequence[Mapping[str, Any]],
                   right: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fields = ("sample_id", "image_sha256", "question_sha256",
              "unordered_pair_sha256")
    result = {}
    for field in fields:
        result[field] = len({str(row[field]) for row in left}
                            & {str(row[field]) for row in right})
    left_source = {(str(row["source"]), str(row["source_sample_id"]))
                   for row in left}
    right_source = {(str(row["source"]), str(row["source_sample_id"]))
                    for row in right}
    result["source_sample_id"] = len(left_source & right_source)
    return result


def freeze(config: Mapping[str, Any], output: Path) -> None:
    settings = _config(config)
    refine._validate_output(output)
    target = _target(output)
    manifest_path = target / "frozen_manifest.json"
    discovery_rows = _rows(config, "discovery")
    dev_rows = _rows(config, "dev")
    heldout_rows = _rows(config, "heldout")
    rubric = base.build_multicrit_open_ended_init_rubric()
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 5:
        raise RuntimeError("Phase17 requires exactly five initial roots")
    contract = _manager_contract(config)
    runtime = _runtime_config(config)
    expected_spec = base._expected_pairwise_request_spec(runtime, discovery_rows)
    memory, memory_hash = split.freeze_rubric_memory(split._epoch(target, 0), rubric)
    rubric.save_json(split._epoch(target, 0) / "rubric_initial.json")
    overlaps = {
        "discovery_vs_dev": _cross_overlap(discovery_rows, dev_rows),
        "discovery_vs_heldout": _heldout_overlap(discovery_rows, heldout_rows),
        "dev_vs_heldout": _heldout_overlap(dev_rows, heldout_rows),
    }
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "protocol_version": PROTOCOL_VERSION,
        "exploratory": True,
        "integration_settings": settings,
        "policy": split.POLICY_V1,
        "datasets": {
            "discovery": {"path": str(base._path(DISCOVERY_PATH).resolve()),
                          "count": len(discovery_rows),
                          "sha256": _file_sha(DISCOVERY_PATH)},
            "dev": {"path": str(base._path(DEV_PATH).resolve()),
                    "count": len(dev_rows), "sha256": _file_sha(DEV_PATH)},
            "heldout": {"path": str(base._path(config["heldout_dataset"]).resolve()),
                        "count": len(heldout_rows),
                        "sha256": _file_sha(config["heldout_dataset"])},
        },
        "known_overlap": overlaps,
        "heldout_interpretation": "exploratory_regression_check_not_unbiased",
        "initial_rubric_sha256": rubric.rubric_sha256,
        "initial_root_ids": list(rubric.root_ids),
        "pairwise_prompt_mode": PAIRWISE_PROMPT_MODE,
        "pairwise_prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "pairwise_request_spec": expected_spec.to_dict(),
        "pairwise_execution_pool": base.BackendPoolSpec.from_dict(
            config["backend_pool"]).to_dict(),
        "manager_profiles": contract["profiles"],
        "manager_request_specs": contract["specs"],
        "manager_endpoint_identities": contract["identities"],
        "refine_protocol": refine.REFINE_V1,
        "refine_manager_profile": contract["refine_profile"],
        "refine_manager_request_specs": contract["refine_specs"],
        "locked_retry_manager_request_specs": contract["retry_specs"],
        "epoch_00_rubric_memory_sha256": memory_hash,
        "epoch_00_rubric_memory": memory,
        "error_signature_source": "fresh_discovery100_only",
        "dev_visible_to_manager": False,
        "dev_affects_acceptance": False,
        "dev_affects_early_stop": False,
        "selection_after_dev_forbidden": True,
        "heldout_access": "final_only",
        "vl_rewardbench_required": True,
    }
    if manifest_path.exists():
        stored = load_json(manifest_path)
        if any(stored.get(key) != item for key, item in manifest.items()):
            raise RuntimeError("Phase17 frozen manifest drift")
        print("discovery-v2-evolution-freeze already completed")
        return
    _write(manifest_path, manifest)
    _write(target / "stage_status.json", {"freeze": {
        "status": "passed", "details": {
            "discovery_count": len(discovery_rows), "dev_count": len(dev_rows),
            "known_heldout_overlap": (
                overlaps["discovery_vs_heldout"]["count"]
                + overlaps["dev_vs_heldout"]["count"]),
        }}})
    print(json.dumps({
        "target": str(target), "discovery_count": len(discovery_rows),
        "dev_count": len(dev_rows), "known_overlap": overlaps,
    }, indent=2, ensure_ascii=False))


def audit(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    manifest = _load_frozen(config, output)
    discovery_rows = _rows(config, "discovery")
    dev_rows = _rows(config, "dev")
    rubric = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    contract = _manager_contract(config)
    vlrb = _vlrb_overlap(config, tuple(discovery_rows) + tuple(dev_rows))
    discovery_gold = Counter(str(row["answer"]) for row in discovery_rows)
    dev_gold = Counter(str(row["answer"]) for row in dev_rows)
    discovery_sources = Counter(str(row["source"]) for row in discovery_rows)
    dev_sources = Counter(str(row["source"]) for row in dev_rows)
    checks = {
        "discovery_count_100": len(discovery_rows) == DISCOVERY_COUNT,
        "dev_count_150": len(dev_rows) == DEV_COUNT,
        "discovery_dev_isolated": not any(
            manifest["known_overlap"]["discovery_vs_dev"].values()),
        "all_images_readable": all(
            base._path(str(row["image_path"])).is_file()
            for row in tuple(discovery_rows) + tuple(dev_rows)),
        "gold_is_ab": all(row["answer"] in {"A", "B"}
                          for row in tuple(discovery_rows) + tuple(dev_rows)),
        "gold_positions_balanced": (
            discovery_gold == Counter({"A": 50, "B": 50})
            and dev_gold == Counter({"A": 75, "B": 75})),
        "source_distribution_frozen": (
            sorted(discovery_sources.values()) == [16, 16, 17, 17, 17, 17]
            and set(dev_sources.values()) == {25}
            and len(discovery_sources) == len(dev_sources) == 6),
        "initial_five_roots": len(rubric.root_ids) == 5 and len(rubric.nodes) == 5,
        "manager_specs_match": contract["specs"] == manifest["manager_request_specs"],
        "refine_specs_match": (
            contract["refine_specs"] == manifest["refine_manager_request_specs"]),
        "retry_specs_match": (
            contract["retry_specs"] == manifest["locked_retry_manager_request_specs"]),
        "vl_rewardbench_strong_overlap_zero": (
            vlrb["sample_id"] == 0 and vlrb["image_sha256"] == 0
            and vlrb["unordered_pair_sha256"] == 0),
        "dev_manager_visibility_false": manifest["dev_visible_to_manager"] is False,
        "heldout_accessed": False,
    }
    value = {"schema_version": "1.0.0", "offline_only": True,
             "checks": checks, "vl_rewardbench_overlap": vlrb,
             "heldout_overlap_is_nonblocking_and_reported": True}
    _write(target / "offline_audit.json", value)
    status = load_json(target / "stage_status.json")
    status["audit"] = {"status": "passed" if all(
        item for key, item in checks.items() if key != "heldout_accessed") else "failed",
        "details": value}
    _write(target / "stage_status.json", status)
    if status["audit"]["status"] != "passed":
        raise RuntimeError(f"Phase17 audit failed: {checks}")
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _worker_prediction(config: Mapping[str, Any], work: Path,
                       rubric: StructuredRubric,
                       rows: Sequence[Mapping[str, Any]], label: str):
    return base._generate_pairwise(
        _runtime_config(config), work, rubric, rows, label,
        execution_backend_pool=config["backend_pool"],
        request_level_progress=True,
    )[0]


def smoke(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    _load_frozen(config, output)
    status = load_json(target / "stage_status.json")
    if status.get("audit", {}).get("status") != "passed":
        raise RuntimeError("run discovery-v2-evolution-audit first")
    report_path = target / "smoke" / "report.json"
    if report_path.exists():
        print(json.dumps(load_json(report_path), indent=2, ensure_ascii=False))
        return
    rubric = StructuredRubric.load_json(split._epoch(target, 0) / "rubric_initial.json")
    discovery_rows = _rows(config, "discovery")[:20]
    dev_rows = _rows(config, "dev")[:20]
    started = time.monotonic()
    discovery_prediction = _worker_prediction(
        config, target / "smoke" / "discovery", rubric, discovery_rows,
        "initial_five_roots")
    dev_prediction = _worker_prediction(
        config, target / "smoke" / "dev", rubric, dev_rows,
        "initial_five_roots")
    _, discovery_answers = execute_offline_m1(
        rubric, discovery_prediction, discovery_rows)
    _, dev_answers = execute_offline_m1(rubric, dev_prediction, dev_rows)
    value = {
        "schema_version": "1.0.0", "status": "passed",
        "discovery_sample_count": 20, "dev_sample_count": 20,
        "discovery_m1": base._metrics(discovery_answers, discovery_rows),
        "dev_m1": base._metrics(dev_answers, dev_rows),
        "prompt_version": discovery_prediction.prompt_version,
        "dev_used_for_selection": False,
        "heldout_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    if (discovery_prediction.prompt_version
            != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
            or dev_prediction.prompt_version
            != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION):
        raise RuntimeError("Phase17 smoke did not use Pairwise Worker Prompt v2")
    _write(report_path, value)
    status["smoke"] = {"status": "passed", "details": value}
    _write(target / "stage_status.json", status)
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _initialize_baseline(config: Mapping[str, Any], target: Path,
                         manifest: dict[str, Any]) -> None:
    epoch0 = split._epoch(target, 0)
    required = (
        epoch0 / "discovery_pairwise.json", epoch0 / "rubric_committed.json",
        epoch0 / "feedback.json", epoch0 / "summary.json",
        target / "evolution_history.json",
    )
    if all(path.exists() for path in required):
        return
    rows = _rows(config, "discovery")
    rubric = StructuredRubric.load_json(epoch0 / "rubric_initial.json")
    prediction_path = target / "baseline" / "predictions" / "discovery_initial.json"
    prediction = (PairwisePredictionOutput.load_json(prediction_path)
                  if prediction_path.exists() else _worker_prediction(
                      config, target / "baseline", rubric, rows,
                      "discovery_initial"))
    votes, feedback = split._snapshot(epoch0, rubric, prediction, rows, manifest)
    thresholds = _runtime_config(config)["evolution_policy"]["trigger_thresholds"]
    context = EvolutionContext(rubric, feedback)
    triggers = {root_id: detect_specialize_trigger(
        context, root_id, thresholds).to_dict() for root_id in rubric.root_ids}
    eligible = [root_id for root_id in rubric.root_ids
                if triggers[root_id]["triggered"]]
    manifest.update({
        "initial_triggers": triggers,
        "initial_eligible_root_ids": eligible,
        "initial_eligible_root_count": len(eligible),
        "initial_prediction_sha256": canonical_sha256(prediction.to_dict()),
    })
    _write(target / "frozen_manifest.json", manifest)
    _write(epoch0 / "summary.json", {
        "epoch": 0, "m1": base._metrics(votes, rows), "triggers": triggers,
        "accepted_roots": [], "pairwise_prompt_version": prediction.prompt_version})
    _write(target / "evolution_history.json", {
        "schema_version": "1.0.0", "current_epoch": 0, "completed": False,
        "stop_reason": None,
        "root_states": {root_id: {
            "status": "eligible" if triggers[root_id]["triggered"] else "not_eligible",
            "attempt_count": 0, "accepted_epoch": None, "children": [],
            "locked_retry": None,
        } for root_id in rubric.root_ids},
        "attempts": [], "refine_states": {}, "refine_attempts": [],
    })


def _group_metrics(answers, rows, field: str) -> dict[str, Any]:
    values = {}
    groups = sorted({str(row.get(field, "unknown")) for row in rows})
    for group in groups:
        indices = [index for index, row in enumerate(rows)
                   if str(row.get(field, "unknown")) == group]
        values[group] = base._metrics(
            [answers[index] for index in indices], [rows[index] for index in indices])
    return values


def _root_metrics(execution, rubric: StructuredRubric, rows) -> dict[str, Any]:
    values = {}
    for root_id in rubric.root_ids:
        votes = []
        for trace in execution.traces:
            match = [item for item in trace.roots if item.root_id == root_id]
            if len(match) != 1 or match[0].subtree_vote is None:
                raise RuntimeError(f"incomplete Dev root trace: {root_id}")
            votes.append(match[0].subtree_vote)
        values[root_id] = base._heldout_vote_metrics(votes, rows)
    return values


def _node_metrics(prediction: PairwisePredictionOutput, rows) -> dict[str, Any]:
    values = {}
    for criterion in prediction.criteria:
        name = criterion.name
        raw = [row[name].vote.value for row in prediction.node_outputs]
        support = sum(value in {"A", "B"} for value in raw)
        correct = sum(value == row["answer"] for value, row in zip(raw, rows)
                      if value in {"A", "B"})
        values[name] = {
            "support": support, "wrong": support - correct,
            "coverage": support / len(rows),
            "accuracy": correct / support if support else 0.0,
        }
    return values


def _evaluate_dev_epoch(config: Mapping[str, Any], target: Path,
                        epoch_no: int) -> dict[str, Any]:
    epoch_dir = split._epoch(target, epoch_no)
    report_path = epoch_dir / "dev150" / "report.json"
    rubric = StructuredRubric.load_json(epoch_dir / "rubric_committed.json")
    if report_path.exists():
        value = load_json(report_path)
        if value.get("rubric_sha256") != rubric.rubric_sha256:
            raise RuntimeError(f"epoch {epoch_no} Dev report rubric drift")
        return value
    rows = _rows(config, "dev")
    # One shared cache namespace is intentional: exact criterion descriptions
    # survive across epochs, while new/refined descriptions produce new keys.
    prediction = _worker_prediction(
        config, target / "dev150_shared", rubric, rows, "node_versions")
    execution, answers = execute_offline_m1(rubric, prediction, rows)
    dev_dir = epoch_dir / "dev150"
    dev_dir.mkdir(parents=True, exist_ok=True)
    prediction.save_json(dev_dir / "pairwise.json")
    execution.save_json(dev_dir / "m1_execution.json")
    overall = base._metrics(answers, rows)
    value = {
        "schema_version": "1.0.0", "epoch": epoch_no,
        "rubric_sha256": rubric.rubric_sha256, "node_count": len(rubric.nodes),
        "m1": overall,
        "domains": _group_metrics(answers, rows, "domain"),
        "sources": _group_metrics(answers, rows, "source"),
        "root_subtrees": _root_metrics(execution, rubric, rows),
        "nodes": _node_metrics(prediction, rows),
        "diagnostic_only": True, "selection_forbidden": True,
        "manager_visible": False,
    }
    _write(report_path, value)
    return value


def _ensure_dev_trajectory(config: Mapping[str, Any], target: Path) -> None:
    history = load_json(target / "evolution_history.json")
    reports = []
    for epoch_no in range(int(history["current_epoch"]) + 1):
        reports.append(_evaluate_dev_epoch(config, target, epoch_no))
    initial = reports[0]["m1"]["predictions"]
    summary = []
    for item in reports:
        predictions = item["m1"]["predictions"]
        rows = _rows(config, "dev")
        corrected = sum(a != row["answer"] and b == row["answer"]
                        for a, b, row in zip(initial, predictions, rows))
        harmed = sum(a == row["answer"] and b != row["answer"]
                     for a, b, row in zip(initial, predictions, rows))
        summary.append({
            "epoch": item["epoch"], "rubric_sha256": item["rubric_sha256"],
            "node_count": item["node_count"], "m1_accuracy": item["m1"]["accuracy"],
            "m1_coverage": item["m1"]["coverage"],
            "corrected_vs_epoch0": corrected, "harmed_vs_epoch0": harmed,
            "net_corrected_vs_epoch0": corrected - harmed,
        })
    _write(target / "dev150_trajectory.json", {
        "schema_version": "1.0.0", "diagnostic_only": True,
        "selection_forbidden": True, "epochs": summary})


def run(config: Mapping[str, Any], output: Path) -> None:
    settings = _config(config)
    target = _target(output)
    status = load_json(target / "stage_status.json")
    if status.get("smoke", {}).get("status") != "passed":
        raise RuntimeError("run discovery-v2-evolution-smoke first")
    manifest = _load_frozen(config, output)
    _initialize_baseline(config, target, manifest)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        refine._run_five_root_locked_split_refine_impl(
            _runtime_config(config), target=target, manifest=manifest,
            history=history, settings=settings, protocol=_protocol(config),
            log_prefix="Discovery-v2 Prompt-v2 evolution",
            execution_backend_pool=config["backend_pool"])
    _ensure_dev_trajectory(config, target)


def report(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    _load_frozen(config, output)
    history = load_json(target / "evolution_history.json")
    if not history.get("completed"):
        raise RuntimeError("run discovery-v2-evolution-run to completion first")
    _ensure_dev_trajectory(config, target)
    epoch_dir = split._epoch(target, int(history["current_epoch"]))
    rubric = StructuredRubric.load_json(epoch_dir / "rubric_committed.json")
    prediction = PairwisePredictionOutput.load_json(epoch_dir / "discovery_pairwise.json")
    rows = _rows(config, "discovery")
    execution, answers = execute_offline_m1(rubric, prediction, rows)
    final_dir = target / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    rubric.save_json(final_dir / "rubric.json")
    prediction.save_json(final_dir / "discovery_pairwise.json")
    execution.save_json(final_dir / "m1_execution.json")
    initial = load_json(split._epoch(target, 0) / "summary.json")["m1"]
    metrics = base._metrics(answers, rows)
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "exploratory": True, "final_epoch": history["current_epoch"],
        "initial_m1": initial, "final_m1": metrics,
        "m1_accuracy_delta": metrics["accuracy"] - initial["accuracy"],
        "final_rubric_sha256": rubric.rubric_sha256,
        "node_count_initial": 5, "node_count_final": len(rubric.nodes),
        "split_attempts": history["attempts"],
        "refine_attempts": history["refine_attempts"],
        "accepted_split_count": sum(item["decision"] == split.ACCEPTED
                                    for item in history["attempts"]),
        "accepted_refine_count": sum(item["decision"] == "accepted"
                                     for item in history["refine_attempts"]),
        "dev_trajectory": load_json(target / "dev150_trajectory.json"),
        "dev_used_for_selection": False,
        "selection_after_external_evaluation_forbidden": True,
        "vl_rewardbench_required": True,
    }
    _write(final_dir / "discovery_report.json", value)
    status = load_json(target / "stage_status.json")
    status["report"] = {"status": "passed", "details": {
        "final_epoch": history["current_epoch"],
        "final_m1_accuracy": metrics["accuracy"],
        "final_rubric_sha256": rubric.rubric_sha256}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["report"]["details"], indent=2))


def heldout(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    reference_output = _heldout_reference_output(config, output)
    _load_frozen(config, output)
    discovery_report_path = target / "final" / "discovery_report.json"
    if not discovery_report_path.exists():
        raise RuntimeError("run discovery-v2-evolution-report first")
    final_rubric = StructuredRubric.load_json(target / "final" / "rubric.json")
    initial_rubric = base.build_multicrit_open_ended_init_rubric()
    phase10_rubric = StructuredRubric.load_json(
        reference_output / phase16.SOURCE_PHASE10 / "final" / "rubric.json")
    phase16_rubric = StructuredRubric.load_json(
        reference_output / phase16.EXPERIMENT_DIR / "final" / "rubric.json")
    phase16_prediction_path = (reference_output / phase16.EXPERIMENT_DIR
                               / "heldout500" / "combined_pairwise.json")
    if not phase16_prediction_path.is_file():
        raise RuntimeError("completed Phase16 Prompt-v2 heldout artifact is required")
    phase16_prediction = PairwisePredictionOutput.load_json(phase16_prediction_path)
    rows = _rows(config, "heldout")
    phase10_prediction, phase10_source = phase16._load_v2_source(
        config, reference_output, phase10_rubric, rows, "heldout")
    initial_prediction = project_pairwise_prediction(
        phase10_prediction, initial_rubric)
    source_descriptions = {item.name: item.description
                           for item in phase16_prediction.criteria}
    changed_ids = tuple(
        node_id for node_id in final_rubric.preorder_node_ids()
        if source_descriptions.get(final_rubric.get_node(node_id).criterion.name)
        != final_rubric.get_node(node_id).criterion.description)
    heldout_dir = target / "heldout500"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_frozen(config, output)
    frozen = {
        "schema_version": "1.0.0", "exploratory": True,
        "interpretation": "regression_check_not_unbiased",
        "known_overlap": manifest["known_overlap"],
        "final_rubric_sha256": final_rubric.rubric_sha256,
        "final_discovery_report_sha256": file_sha256(discovery_report_path),
        "heldout_dataset_sha256": _file_sha(config["heldout_dataset"]),
        "phase16_prediction_sha256": file_sha256(phase16_prediction_path),
        "phase10_prediction_sha256": file_sha256(phase10_source),
        "pairwise_prompt_version": phase16_prediction.prompt_version,
        "selection_after_heldout_forbidden": True,
    }
    frozen_path = heldout_dir / "frozen_manifest.json"
    if frozen_path.exists() and load_json(frozen_path) != frozen:
        raise RuntimeError("Phase17 heldout manifest drift")
    _write(frozen_path, frozen)
    generated = None
    generated_path = heldout_dir / "changed_node_predictions.json"
    if changed_ids:
        if generated_path.exists():
            generated = PairwisePredictionOutput.load_json(generated_path)
        else:
            nodes = {node_id: final_rubric.get_node(node_id) for node_id in changed_ids}
            variant = StructuredRubric(nodes, (), changed_ids)
            generated = _worker_prediction(
                config, heldout_dir / "worker", variant, rows, "changed_nodes")
            generated.save_json(generated_path)
    generated_for_merge = generated
    if generated is not None:
        scientific_identity_match = _same_pairwise_scientific_request_identity(
            phase16_prediction, generated)
        _write(heldout_dir / "prediction_merge_provenance.json", {
            "schema_version": "1.0.0",
            "source_artifact": str(phase16_prediction_path.resolve()),
            "generated_artifact": str(generated_path.resolve()),
            "source_backend_id": phase16_prediction.request_spec.backend_id,
            "generated_backend_id": generated.request_spec.backend_id,
            "exact_request_spec_match": (
                phase16_prediction.request_spec == generated.request_spec),
            "scientific_request_identity_match": scientific_identity_match,
            "normalization": (
                "backend_id_only" if scientific_identity_match
                and phase16_prediction.request_spec != generated.request_spec
                else "none"),
            "source_prediction_sha256": file_sha256(phase16_prediction_path),
            "generated_prediction_sha256": file_sha256(generated_path),
        })
        generated_for_merge = _normalize_generated_request_for_merge(
            phase16_prediction, generated)
    combined = refine._combine_heldout_predictions(
        phase16_prediction, generated_for_merge, final_rubric)
    combined.save_json(heldout_dir / "combined_pairwise.json")

    def evaluate(rubric, prediction):
        execution, answers = execute_offline_m1(rubric, prediction, rows)
        return execution, answers, base._heldout_vote_metrics(answers, rows)

    _, initial_answers, initial_metrics = evaluate(initial_rubric, initial_prediction)
    _, phase10_answers, phase10_metrics = evaluate(phase10_rubric, phase10_prediction)
    _, phase16_answers, phase16_metrics = evaluate(phase16_rubric, phase16_prediction)
    execution, final_answers, final_metrics = evaluate(final_rubric, combined)
    execution.save_json(heldout_dir / "m1_execution.json")
    value = {
        "schema_version": "1.0.0", "exploratory": True,
        "interpretation": "known_overlap_regression_check_only",
        "initial_five_root_prompt_v2": initial_metrics,
        "phase10_prompt_v2": phase10_metrics,
        "phase16_prompt_v2": phase16_metrics,
        "phase17_discovery_v2_prompt_v2": final_metrics,
        "phase17_vs_initial": base._paired_heldout_comparison(
            initial_answers, final_answers, rows),
        "phase17_vs_phase10": base._paired_heldout_comparison(
            phase10_answers, final_answers, rows),
        "phase17_vs_phase16": base._paired_heldout_comparison(
            phase16_answers, final_answers, rows),
        "reused_exact_criteria": len(final_rubric.nodes) - len(changed_ids),
        "generated_changed_criteria": len(changed_ids),
        "selection_after_heldout_forbidden": True,
    }
    _write(heldout_dir / "report.json", value)
    status = load_json(target / "stage_status.json")
    status["heldout"] = {"status": "passed", "details": {
        "phase16_accuracy": phase16_metrics["accuracy"],
        "phase17_accuracy": final_metrics["accuracy"],
        "known_overlap_count": (
            manifest["known_overlap"]["discovery_vs_heldout"]["count"]
            + manifest["known_overlap"]["dev_vs_heldout"]["count"])}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["heldout"]["details"], indent=2))


def final_report(config: Mapping[str, Any], output: Path) -> None:
    _config(config)
    target = _target(output)
    _load_frozen(config, output)
    discovery = load_json(target / "final" / "discovery_report.json")
    heldout_value = load_json(target / "heldout500" / "report.json")
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "exploratory": True, "discovery": discovery,
        "heldout": heldout_value,
        "heldout_is_primary": False,
        "next_required_stage": "vlrb-discovery-v2-freeze",
        "vl_rewardbench_required": True,
    }
    _write(target / "final_report.json", value)
    lines = [
        "# Discovery-v2 Prompt-v2 Split+Refine", "",
        "Discovery100 is the only optimization split; Dev150 is diagnostic only.",
        "Heldout-500 contains known source overlap and is only a regression check.",
        "VL-RewardBench is the primary external evaluation.", "",
        "| System | heldout-500 strict ACC | Coverage |", "|---|---:|---:|",
    ]
    for name in ("initial_five_root_prompt_v2", "phase10_prompt_v2",
                 "phase16_prompt_v2", "phase17_discovery_v2_prompt_v2"):
        item = heldout_value[name]
        lines.append(f"| {name} | {item['accuracy']:.4f} | {item['coverage']:.4f} |")
    (target / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    status = load_json(target / "stage_status.json")
    status["final_report"] = {"status": "passed", "details": {
        "vl_rewardbench_required": True}}
    _write(target / "stage_status.json", status)
    print(json.dumps(status["final_report"]["details"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {
        "discovery-v2-evolution-freeze": freeze,
        "discovery-v2-evolution-audit": audit,
        "discovery-v2-evolution-smoke": smoke,
        "discovery-v2-evolution-run": run,
        "discovery-v2-evolution-report": report,
        "discovery-v2-evolution-heldout": heldout,
        "discovery-v2-evolution-final-report": final_report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported Discovery-v2 evolution stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

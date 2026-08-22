"""External K=3 transfer diagnostic for frozen Phase16 epoch checkpoints."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    CacheMode,
    JsonPredictionCache,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    RubricNode,
    StructuredCriterionSnapshot,
    StructuredRubric,
)
from critiq.structured.dual_backend import OnlinePairwiseVoteBackend
from critiq.structured.version import PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
from critiq.utils import Criterion

from . import prompt_v2_aligned_evolution as evolution
from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from . import vl_rewardbench_phase10 as phase10
from . import vl_rewardbench_prompt_v2 as control
from . import vl_rewardbench_prompt_v2_evolved as evolved
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, make_progress_callback
from .rubric_factory import file_sha256


EXPERIMENT_DIR = "vl_rewardbench_phase16_checkpoint_transfer_v3_sample_major_dynamic"
PROTOCOL_VERSION = "vlrb-phase16-checkpoint-transfer-v3-sample-major-dynamic"
EPOCHS = (2, 3)
SMOKE_COUNT = 20
MAX_RETRY_ATTEMPTS = 10
CONFIG_KEY = "vlrb_phase16_checkpoint_transfer"
SOURCE_EVOLUTION_EXPERIMENT_DIR = evolution.EXPERIMENT_DIR
SOURCE_VLRB_EXPERIMENT_DIR = evolved.EXPERIMENT_DIR
SYSTEM_PREFIX = "phase16"
REPORT_TITLE = "Phase16 checkpoint VL-RewardBench transfer diagnostic"
EXPECTED_VARIANT_DESCRIPTION_COUNT: int | None = None
CHECKPOINT_ROLES: Mapping[int, str] = {
    2: "phase16_intermediate_epoch_2",
    3: "phase16_intermediate_epoch_3",
}
STAGE_PREFIX = "vlrb-phase16-checkpoint"
STAGES = tuple(f"{STAGE_PREFIX}-{name}" for name in (
    "freeze", "audit", "smoke", "run", "retry", "report"))


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _evolved_target() -> Path:
    return (base.ROOT / "output/evolving_structured_rubrics"
            / SOURCE_VLRB_EXPERIMENT_DIR)


def _control_target() -> Path:
    return (base.ROOT / "output/evolving_structured_rubrics"
            / control.EXPERIMENT_DIR)


def _status_path(target: Path) -> Path:
    return target / "stage_status.json"


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    value[stage] = {"status": "passed", "details": dict(details)}
    atomic_write_json(_status_path(target), value)


def _require(target: Path, stage: str) -> None:
    status = load_json(_status_path(target)) if _status_path(target).exists() else {}
    if status.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment": SOURCE_EVOLUTION_EXPERIMENT_DIR,
        "source_vlrb_experiment": SOURCE_VLRB_EXPERIMENT_DIR,
        "epochs": list(EPOCHS),
        "endpoint_ids": list(control.ENDPOINT_IDS),
        "scheduler": "sample_major_available_slot_dynamic",
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "smoke_sample_count": SMOKE_COUNT,
        "max_retry_attempts": MAX_RETRY_ATTEMPTS,
        "k": legacy.K,
        "seed": legacy.SEED,
        "selection_after_benchmark_forbidden": True,
    }
    if EXPECTED_VARIANT_DESCRIPTION_COUNT is not None:
        expected["expected_unique_variant_description_count"] = (
            EXPECTED_VARIANT_DESCRIPTION_COUNT)
    if config.get(CONFIG_KEY) != expected:
        raise RuntimeError(f"{CONFIG_KEY} must match frozen protocol")
    request = config.get("worker_request_kwargs")
    if not isinstance(request, Mapping) or request.get("temperature") != .5 or request.get("max_tokens") != 2048:
        raise RuntimeError("Phase16 checkpoint transfer requires temperature=.5/max_tokens=2048")
    return expected


def _checkpoint_rubric(output: Path, epoch: int) -> StructuredRubric:
    return StructuredRubric.load_json(
        output / SOURCE_EVOLUTION_EXPERIMENT_DIR / "epochs"
        / f"epoch_{epoch:02d}" / "rubric_committed.json")


def _final_rubric(output: Path) -> StructuredRubric:
    return StructuredRubric.load_json(
        output / SOURCE_EVOLUTION_EXPERIMENT_DIR / "final" / "rubric.json")


def _records():
    return control._records()


def _source_predictions() -> tuple[PairwisePredictionOutput, ...]:
    source = _evolved_target() / "retry"
    paths = tuple(control._prediction_path(source, replicate) for replicate in range(legacy.K))
    if not all(path.is_file() for path in paths):
        raise RuntimeError("completed Phase16 epoch-5 VL-RewardBench predictions are required")
    return tuple(PairwisePredictionOutput.load_json(path) for path in paths)


def _description_hash(value: str) -> str:
    return canonical_sha256({"description": value})


def _variant_key(node_id: str, description: str) -> tuple[str, str]:
    return node_id, _description_hash(description)


def _variant_path(target: Path, node_id: str, description: str, replicate: int) -> Path:
    key = canonical_sha256({"node_id": node_id, "description": description})[:16]
    # The first smoke implementation accidentally wrote Prompt-v1 artifacts
    # under ``predictions.json``.  A versioned filename prevents those invalid
    # local shards from ever being mistaken for Prompt-v2 outputs on resume.
    return target / "variants" / key / f"r{replicate + 1}" / "predictions_v2.json"


def _single_node_rubric(node: RubricNode) -> StructuredRubric:
    return StructuredRubric({node.node_id: node}, (), (node.node_id,))


def _changed_nodes(checkpoint: StructuredRubric, final: StructuredRubric) -> tuple[RubricNode, ...]:
    changed = []
    for node_id in checkpoint.preorder_node_ids():
        node = checkpoint.get_node(node_id)
        final_node = final.get_node(node_id)
        if node.criterion.description != final_node.criterion.description:
            changed.append(node)
    return tuple(changed)


def _manifest(config: Mapping[str, Any], output: Path, records, schedule, *, live: bool) -> dict[str, Any]:
    protocol = _protocol(config)
    final = _final_rubric(output)
    checkpoints = {epoch: _checkpoint_rubric(output, epoch) for epoch in EPOCHS}
    source_predictions = _source_predictions()
    source_logical_path = (
        _evolved_target() / "retry" / "combined" / "logical_votes.json")
    source_report_path = _evolved_target() / "final_report.json"
    control_logical_path = (
        _control_target() / "retry" / "combined" / "logical_votes.json")
    control_report_path = _control_target() / "final_report.json"
    for required in (source_logical_path, source_report_path,
                     control_logical_path, control_report_path):
        if not required.is_file():
            raise RuntimeError(
                f"completed checkpoint control artifact is required: {required}")
    request_spec = control._v2_request_spec(config, records, schedule)
    source_request = source_predictions[0].request_spec.to_dict()
    if request_spec != source_request or any(
            prediction.request_spec.to_dict() != source_request
            for prediction in source_predictions):
        raise RuntimeError("checkpoint worker request identity differs from epoch-5 source")
    if any(prediction.prompt_version != PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
           for prediction in source_predictions):
        raise RuntimeError("epoch-5 source prediction is not Prompt v2")
    variants: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoint_nodes = {}
    for epoch, rubric in checkpoints.items():
        checkpoint_nodes[str(epoch)] = len(rubric.nodes)
        for node in _changed_nodes(rubric, final):
            key = _variant_key(node.node_id, node.criterion.description)
            variants[key] = {"node_id": node.node_id, "criterion_name": node.criterion.name,
                             "description": node.criterion.description,
                             "description_sha256": key[1]}
    value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR,
        "protocol": protocol, "exploratory": True,
        "dataset": {"count": len(records), "record_sha256": canonical_sha256(list(records))},
        "order_schedule_sha256": canonical_sha256(schedule),
        "counterbalance": {"k": legacy.K, "seed": legacy.SEED, "protocol": "balanced_b_1minusb_b"},
        "structured_worker_request_spec": request_spec,
        "prompt_version": PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        "checkpoint_rubrics": {str(epoch): {"rubric_sha256": rubric.rubric_sha256,
                                                "node_count": len(rubric.nodes),
                                                "selection_role": CHECKPOINT_ROLES[epoch]}
                               for epoch, rubric in checkpoints.items()},
        "final_epoch_5": {
            "rubric_sha256": final.rubric_sha256,
            "node_count": len(final.nodes),
            "prediction_sha256": [file_sha256(control._prediction_path(
                _evolved_target() / "retry", replicate))
                for replicate in range(legacy.K)],
            "logical_votes_sha256": file_sha256(source_logical_path),
            "final_report_sha256": file_sha256(source_report_path),
        },
        "phase10_control": {
            "experiment": control.EXPERIMENT_DIR,
            "logical_votes_sha256": file_sha256(control_logical_path),
            "final_report_sha256": file_sha256(control_report_path),
            "generated": 0,
        },
        "variant_descriptions": [variants[key] for key in sorted(variants)],
        "reuse": {"unit": "sample_replicate_node_description_request_identity",
                  "source": SOURCE_VLRB_EXPERIMENT_DIR,
                  "fallback_reuse_by_name_forbidden": True},
        "selection_after_benchmark_forbidden": True,
    }
    if (EXPECTED_VARIANT_DESCRIPTION_COUNT is not None
            and len(value["variant_descriptions"])
            != EXPECTED_VARIANT_DESCRIPTION_COUNT):
        raise RuntimeError(
            "checkpoint unique historical description count drift: "
            f"expected={EXPECTED_VARIANT_DESCRIPTION_COUNT}, "
            f"actual={len(value['variant_descriptions'])}")
    if live:
        value["endpoint_identities"] = phase10._inspect_endpoints(config)
    return value


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target(); _require(target, STAGES[0])
    records = _records(); schedule = legacy._order_schedule(records)
    expected = _manifest(config, output, records, schedule, live=False)
    manifest = load_json(target / "frozen_manifest.json")
    if {key: value for key, value in manifest.items() if key != "endpoint_identities"} != expected:
        raise RuntimeError("Phase16 checkpoint VL-RewardBench manifest drift")
    return target, manifest, records, schedule


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if manifest.get("endpoint_identities") != phase10._inspect_endpoints(config):
        raise RuntimeError("Phase16 checkpoint VL-RewardBench endpoint identity drift")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(); records = _records(); schedule = legacy._order_schedule(records)
    manifest = _manifest(config, output, records, schedule, live=True)
    path = target / "frozen_manifest.json"
    if path.exists() and load_json(path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        if any(status.get(stage, {}).get("status") == "passed"
               for stage in STAGES[2:]):
            raise RuntimeError("checkpoint manifest drift after inference")
        atomic_write_json(_status_path(target), {})
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    for epoch in EPOCHS:
        rubric_path = target / "checkpoint_rubrics" / f"epoch_{epoch:02d}.json"
        rubric_path.parent.mkdir(parents=True, exist_ok=True)
        _checkpoint_rubric(output, epoch).save_json(rubric_path)
    atomic_write_json(target / "variant_manifest.json", {
        "schema_version": "1.0.0",
        "items": manifest["variant_descriptions"],
    })
    atomic_write_json(target / "reuse_manifest.json", manifest["reuse"])
    details = {"dataset_count": len(records), "epochs": list(EPOCHS),
               "unique_variant_description_count": len(manifest["variant_descriptions"]),
               "worst_case_new_logical_requests": len(manifest["variant_descriptions"]) * len(records) * legacy.K}
    _status(target, STAGES[0], details); print(json.dumps(details, indent=2))


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output)
    values = list(schedule.values())
    source_predictions = _source_predictions()
    reconstructed = control._logical_from_predictions(
        records, schedule, _final_rubric(output), source_predictions)
    source_logical = load_json(
        _evolved_target() / "retry" / "combined" / "logical_votes.json")
    source_reconstruction_exact = (
        reconstructed["sample_ids"] == source_logical["sample_ids"]
        and reconstructed["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"]
        == source_logical["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"])
    result = {"schema_version": "1.0.0", "status": "passed", "offline_only": True,
              "dataset_count": len(records), "k": legacy.K,
              "schedule_aba_count": sum(tuple(x) == (0, 1, 0) for x in values),
              "schedule_bab_count": sum(tuple(x) == (1, 0, 1) for x in values),
              "variant_descriptions": len(manifest["variant_descriptions"]),
              "description_keyed_reuse": True,
              "source_epoch_5_reconstruction_exact": source_reconstruction_exact,
              "benchmark_labels_in_prompt": False}
    if abs(result["schedule_aba_count"] - result["schedule_bab_count"]) > 1:
        raise RuntimeError("K=3 schedule imbalance")
    if not source_reconstruction_exact:
        raise RuntimeError("source epoch-5 logical reconstruction drift")
    atomic_write_json(target / "offline_audit.json", result); _status(target, STAGES[1], result)
    print(json.dumps(result, indent=2))


def _load_variant_prediction(work: Path, manifest: Mapping[str, Any], node: RubricNode,
                             rows, replicate: int) -> PairwisePredictionOutput | None:
    """Load a complete one-description artifact, if a prior run produced one."""

    path = _variant_path(work, node.node_id, node.criterion.description, replicate)
    if not path.is_file():
        return None
    prediction = PairwisePredictionOutput.load_json(path)
    control._validate_prediction(prediction, _single_node_rubric(node), rows, manifest)
    return prediction


def _sample_major_variant_predictions(
    config: Mapping[str, Any], work: Path, manifest: Mapping[str, Any],
    nodes: Mapping[tuple[str, str], RubricNode], rows, replicate: int,
) -> tuple[dict[tuple[str, str], PairwisePredictionOutput], int, dict[str, Any]]:
    """Evaluate missing descriptions in sample-major waves.

    Tasks are inserted into one shared executor queue in `(sample, description)`
    order.  The available-slot pool then gives each newly free endpoint the
    next queued request immediately.  This is exactly the original dynamic
    scheduling mechanism, except that we now submit all changed descriptions
    together rather than calling the one-description runner repeatedly.
    """

    predictions: dict[tuple[str, str], PairwisePredictionOutput] = {}
    pending: dict[tuple[str, str], RubricNode] = {}
    generated_shards = 0
    for key, node in nodes.items():
        prediction = _load_variant_prediction(work, manifest, node, rows, replicate)
        if prediction is None:
            pending[key] = node
            generated_shards += 1
        else:
            predictions[key] = prediction
    if not pending:
        return predictions, generated_shards, {"scheduled_requests": 0, "endpoint_call_counts": {}}

    model_rows = base._model_rows(rows)
    pool = AvailableSlotBackendPool(control._pool_spec(config))
    evaluator = control._v2_evaluator(config, model_rows, pool)
    if evaluator.request_spec().to_dict() != manifest["structured_worker_request_spec"]:
        raise RuntimeError("checkpoint variant Prompt-v2 request identity drift")
    backends = {
        key: OnlinePairwiseVoteBackend(
            evaluator,
            JsonPredictionCache(
                _variant_path(work, node.node_id, node.criterion.description, replicate).parent / "cache",
                CacheMode.READ_WRITE,
            ),
        )
        for key, node in pending.items()
    }
    outputs: dict[tuple[str, str], list[PairwiseVoteOutput | None]] = {
        key: [None] * len(model_rows) for key in pending
    }
    total = len(model_rows) * len(pending)
    callback = make_progress_callback(
        work / "sample_major_progress" / f"replicate_{replicate + 1:02d}",
        f"checkpoint_sample_major_r{replicate + 1}", total, pool,
    )

    def one(index: int, key: tuple[str, str]):
        result = backends[key].evaluate(model_rows[index], pending[key])
        return index, key, result

    with ThreadPoolExecutor(max_workers=pool.spec.global_request_concurrency) as executor:
        futures = {
            executor.submit(one, index, key): (index, key)
            for index in range(len(model_rows))
            for key in pending
        }
        for future in as_completed(futures):
            index, key, result = future.result()
            outputs[key][index] = result.output
            callback(index, f"{model_rows[index][evaluator.sample_id_field]}::{pending[key].criterion.name}", result.metrics)

    for key, node in pending.items():
        values = outputs[key]
        if any(value is None for value in values):
            raise RuntimeError("sample-major checkpoint prediction matrix is incomplete")
        criterion = StructuredCriterionSnapshot(node.criterion.name, node.criterion.description)
        node_rows = tuple({node.criterion.name: value} for value in values)
        answers = tuple(
            base.aggregate_flat_votes((row[node.criterion.name].vote,))
            for row in node_rows
        )
        prediction = PairwisePredictionOutput(
            tuple(str(row[evaluator.sample_id_field]) for row in model_rows),
            tuple(evaluator.sample_fingerprint(row) for row in model_rows),
            (criterion,), node_rows, answers, evaluator.request_spec(),
            prompt_version=PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
        )
        path = _variant_path(work, node.node_id, node.criterion.description, replicate)
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.save_json(path)
        control._validate_prediction(prediction, _single_node_rubric(node), rows, manifest)
        predictions[key] = prediction
    return predictions, generated_shards, {
        "scheduled_requests": total,
        "endpoint_call_counts": dict(pool.records_by_endpoint()),
        "submission_order": "sample_then_description",
    }


def _assemble(checkpoint: StructuredRubric, source: PairwisePredictionOutput,
              variants: Mapping[tuple[str, str], PairwisePredictionOutput]) -> PairwisePredictionOutput:
    criteria = []; outputs = []
    for node_id in checkpoint.preorder_node_ids():
        node = checkpoint.get_node(node_id)
        criteria.append(StructuredCriterionSnapshot(node.criterion.name, node.criterion.description))
    for index in range(len(source.sample_ids)):
        row = {}
        for node_id in checkpoint.preorder_node_ids():
            node = checkpoint.get_node(node_id); key = _variant_key(node_id, node.criterion.description)
            prediction = variants.get(key, source)
            row[node.criterion.name] = prediction.node_outputs[index][node.criterion.name]
        outputs.append(row)
    answers = tuple(base.aggregate_flat_votes(output.vote for output in row.values()) for row in outputs)
    return PairwisePredictionOutput(source.sample_ids, source.sample_fingerprints, tuple(criteria), tuple(outputs), answers,
                                    source.request_spec, source.semantics_version, source.schema_version,
                                    source.prompt_version, source.parser_version)


def _prefix_prediction(prediction: PairwisePredictionOutput, expected_ids: Sequence[str]) -> PairwisePredictionOutput:
    """Project a full epoch-5 artifact to the frozen smoke prefix."""
    count = len(expected_ids)
    if prediction.sample_ids[:count] != tuple(expected_ids):
        raise RuntimeError("checkpoint smoke source prediction sample order drift")
    rows = prediction.node_outputs[:count]
    answers = tuple(base.aggregate_flat_votes(output.vote for output in row.values()) for row in rows)
    return PairwisePredictionOutput(prediction.sample_ids[:count], prediction.sample_fingerprints[:count],
                                    prediction.criteria, rows, answers, prediction.request_spec,
                                    prediction.semantics_version, prediction.schema_version,
                                    prediction.prompt_version, prediction.parser_version)


def _write_composites(config: Mapping[str, Any], output: Path, target: Path, manifest, records, schedule, work: Path) -> tuple[dict[int, tuple[PairwisePredictionOutput, ...]], int, list[dict[str, Any]]]:
    final = _final_rubric(output)
    sources = _source_predictions(); generated = 0; provenance = []; per_epoch = {epoch: [] for epoch in EPOCHS}
    for replicate in range(legacy.K):
        rows = legacy._ordered_rows(records, schedule, replicate)
        source = _prefix_prediction(sources[replicate], [str(row["sample_id"]) for row in rows])
        nodes = {}
        for epoch in EPOCHS:
            for node in _changed_nodes(_checkpoint_rubric(output, epoch), final):
                nodes[_variant_key(node.node_id, node.criterion.description)] = node
        variants, count, details = _sample_major_variant_predictions(
            config, work, manifest, nodes, rows, replicate)
        generated += count
        provenance.append({"replicate": replicate + 1, **details})
        for epoch in EPOCHS:
            checkpoint = _checkpoint_rubric(output, epoch)
            composite = _assemble(checkpoint, source, variants)
            path = work / f"epoch_{epoch:02d}" / f"replicate_{replicate + 1:02d}.json"; path.parent.mkdir(parents=True, exist_ok=True)
            composite.save_json(path); per_epoch[epoch].append(composite)
    return {epoch: tuple(items) for epoch, items in per_epoch.items()}, generated, provenance


def _logical(output: Path, records, schedule, predictions: Mapping[int, Sequence[PairwisePredictionOutput]]) -> dict[str, Any]:
    systems = {}; root_votes = {}
    for epoch, items in predictions.items():
        rubric = _checkpoint_rubric(output, epoch)
        votes = []; roots = {root_id: [] for root_id in rubric.root_ids}
        for replicate, prediction in enumerate(items):
            rows = legacy._ordered_rows(records, schedule, replicate)
            execution, answers = base.execute_offline_m1(rubric, prediction, rows)
            orders = [int(schedule[str(row["sample_id"])][replicate]) for row in rows]
            votes.append([legacy._original_index(answer.value, order) for answer, order in zip(answers, orders)])
            for root_id in rubric.root_ids:
                root_answers = [next(root.subtree_vote for root in trace.roots if root.root_id == root_id) for trace in execution.traces]
                roots[root_id].append([legacy._original_index(answer.value, order) for answer, order in zip(root_answers, orders)])
        systems[f"{SYSTEM_PREFIX}_epoch_{epoch:02d}_equal"] = {
            "votes_by_replicate": votes}
        root_votes[str(epoch)] = roots
    return {"schema_version": "1.0.0", "sample_ids": [str(x["sample_id"]) for x in records], "k": legacy.K,
            "systems": systems, "root_votes_by_epoch": root_votes}


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output); _require(target, STAGES[1]); _verify_live(config, manifest)
    selected = tuple(records[:SMOKE_COUNT]); sub_schedule = {str(row["sample_id"]): schedule[str(row["sample_id"])] for row in selected}
    predictions, generated, provenance = _write_composites(
        config, output, target, manifest, selected, sub_schedule, target / "smoke")
    logical = _logical(output, selected, sub_schedule, predictions)
    metrics = {name: control._smoke_system_metrics(selected, item["votes_by_replicate"])
               for name, item in logical["systems"].items()}
    details = {"sample_count": len(selected), "generated_variant_shards": generated,
               "sample_major_provenance": provenance, "metrics": metrics}
    if any(metric["coverage"] < .95 for metric in metrics.values()): raise RuntimeError("checkpoint smoke coverage below 95%")
    _status(target, STAGES[2], details); print(json.dumps(details, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output); _require(target, STAGES[2]); _verify_live(config, manifest)
    predictions, generated, provenance = _write_composites(
        config, output, target, manifest, records, schedule, target / "run")
    logical = _logical(output, records, schedule, predictions); atomic_write_json(target / "run" / "logical_votes.json", logical)
    details = {"sample_count": len(records), "generated_variant_shards": generated,
               "logical_request_count": generated * len(records), "epochs": list(EPOCHS),
               "sample_major_provenance": provenance}
    _status(target, STAGES[3], details); print(json.dumps(details, indent=2))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output); _require(target, STAGES[3]); _verify_live(config, manifest)
    source = {epoch: [PairwisePredictionOutput.load_json(
        target / "run" / f"epoch_{epoch:02d}" / f"replicate_{replicate + 1:02d}.json")
        for replicate in range(legacy.K)] for epoch in EPOCHS}
    failure_index: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for epoch in EPOCHS:
        checkpoint = _checkpoint_rubric(output, epoch)
        descriptions = {
            node.criterion.name: node.criterion.description
            for node in checkpoint.nodes.values()
        }
        for replicate in range(legacy.K):
            prediction = source[epoch][replicate]
            for index, row in enumerate(prediction.node_outputs):
                for name, vote in row.items():
                    if not (vote.parse_ok and vote.answer_valid):
                        description = descriptions[name]
                        key = (replicate, index, name,
                               _description_hash(description))
                        source_hash = canonical_sha256(vote.to_dict())
                        item = failure_index.setdefault(key, {
                            "epochs": [], "replicate": replicate,
                            "sample_index": index,
                            "sample_id": prediction.sample_ids[index],
                            "criterion_name": name,
                            "description": description,
                            "description_sha256": key[3],
                            "source_output_sha256": source_hash,
                            "parse_error": vote.parse_error,
                        })
                        if item["source_output_sha256"] != source_hash:
                            raise RuntimeError(
                                "identical checkpoint request has divergent "
                                "source outputs")
                        item["epochs"].append(epoch)
    failures = list(failure_index.values())
    pool = AvailableSlotBackendPool(control._pool_spec(config))
    ordered = {replicate: legacy._ordered_rows(records, schedule, replicate) for replicate in range(legacy.K)}
    model_rows = {replicate: base._model_rows(rows) for replicate, rows in ordered.items()}
    retry_config = dict(config); retry_config["structured_max_retries"] = 0
    evaluators = {replicate: control._v2_evaluator(retry_config, model_rows[replicate], pool) for replicate in range(legacy.K)}
    states = []
    def one(item):
        path = target / "retry" / "attempts" / f"{canonical_sha256(item)}.json"
        state = load_json(path) if path.is_file() else {"item": item, "attempts": [], "complete": False, "success": False, "final_output": None}
        if not state["complete"]:
            for number in range(len(state["attempts"]) + 1, MAX_RETRY_ATTEMPTS + 1):
                criterion = Criterion(
                    item["criterion_name"], item["description"], 1.0)
                candidate, metrics = evaluators[item["replicate"]].infer_one(
                    model_rows[item["replicate"]][item["sample_index"]],
                    criterion)
                state["attempts"].append({"attempt": number, "output": candidate.to_dict(), "metrics": dict(metrics.__dict__)})
                if candidate.parse_ok and candidate.answer_valid:
                    state.update({"complete": True, "success": True, "final_output": candidate.to_dict()})
                elif number == MAX_RETRY_ATTEMPTS: state["complete"] = True
                atomic_write_json(path, state)
                if state["success"]: break
        return item, state
    if failures:
        with ThreadPoolExecutor(max_workers=control._pool_spec(config).global_request_concurrency) as executor:
            states = list(executor.map(one, failures))
    replacements = {epoch: {replicate: {} for replicate in range(legacy.K)} for epoch in EPOCHS}
    for item, state in states:
        if state["success"]:
            output_value = PairwiseVoteOutput.from_dict(state["final_output"])
            for epoch in item["epochs"]:
                replacements[epoch][item["replicate"]][(
                    item["sample_index"], item["criterion_name"])] = output_value
    repaired = {epoch: [] for epoch in EPOCHS}
    for epoch in EPOCHS:
        for replicate, prediction in enumerate(source[epoch]):
            value = control._replace_prediction_outputs(prediction, replacements[epoch][replicate])
            path = target / "retry" / f"epoch_{epoch:02d}" / f"replicate_{replicate + 1:02d}.json"; path.parent.mkdir(parents=True, exist_ok=True); value.save_json(path)
            repaired[epoch].append(value)
    logical = _logical(output, records, schedule, repaired); atomic_write_json(target / "retry" / "logical_votes.json", logical)
    recovered = sum(state["success"] for _, state in states)
    details = {"target_count": len(failures), "recovered_count": recovered, "still_failed_count": len(failures) - recovered,
               "new_model_requests": sum(len(state["attempts"]) for _, state in states), "max_retry_attempts": MAX_RETRY_ATTEMPTS}
    atomic_write_json(target / "retry" / "failure_manifest.json", {"items": failures, "states": [state for _, state in states], **details})
    _status(target, STAGES[4], details); print(json.dumps(details, indent=2))


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output); _require(target, STAGES[4])
    logical = load_json(target / "retry" / "logical_votes.json")
    source_logical = load_json(_evolved_target() / "retry" / "combined" / "logical_votes.json")
    control_logical = load_json(base.ROOT / "output/evolving_structured_rubrics" / control.EXPERIMENT_DIR / "retry" / "combined" / "logical_votes.json")
    metrics = {name: phase10._system_metrics(records, item["votes_by_replicate"]) for name, item in logical["systems"].items()}
    epoch5 = phase10._system_metrics(records, source_logical["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"])
    initial = phase10._system_metrics(records, control_logical["systems"][control.INITIAL_SYSTEM]["votes_by_replicate"])
    phase10_value = phase10._system_metrics(records, control_logical["systems"][control.EQUAL_SYSTEM]["votes_by_replicate"])
    paired = {name: legacy._paired(records, epoch5["original_index_predictions"], value["original_index_predictions"])
              for name, value in metrics.items()}
    paired_phase10 = {
        name: legacy._paired(
            records, phase10_value["original_index_predictions"],
            item["original_index_predictions"])
        for name, item in metrics.items()
    }
    root_metrics = {
        str(epoch): {
            root_id: phase10._system_metrics(records, votes)
            for root_id, votes in roots.items()
        }
        for epoch, roots in logical["root_votes_by_epoch"].items()
    }
    value = {"schema_version": "1.0.0", "experiment": EXPERIMENT_DIR, "exploratory": True,
             "metrics": {"initial_five_root_prompt_v2": initial, "phase10_final_equal_prompt_v2": phase10_value,
                         f"{SYSTEM_PREFIX}_epoch_05_equal": epoch5, **metrics},
             "paired_vs_epoch_05": paired,
             "paired_vs_phase10": paired_phase10,
             "root_metrics_by_epoch": root_metrics,
             "retry": load_json(target / "retry" / "failure_manifest.json"),
             "selection_after_benchmark_forbidden": True}
    atomic_write_json(target / "final_report.json", value)
    lines = [f"# {REPORT_TITLE}", "",
             "Exploratory K=3 diagnostic; no post-hoc checkpoint selection.",
             "", "| System | OverallAcc | MacroAcc | Coverage | Strict ACC |",
             "|---|---:|---:|---:|---:|"]
    for name, item in value["metrics"].items():
        lines.append(
            f"| {name} | {item['overall_acc']:.4f} | "
            f"{item['macro_acc']:.4f} | {item['coverage']:.4f} | "
            f"{item['strict_accuracy']:.4f} |")
    (target / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    details = {name: {"overall_acc": item["overall_acc"], "macro_acc": item["macro_acc"]} for name, item in metrics.items()}
    _status(target, STAGES[5], details); print(json.dumps(details, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = dict(zip(STAGES, (freeze, audit, smoke, run, retry, report)))
    if stage not in actions: raise ValueError(f"unsupported checkpoint VLRB stage: {stage}")
    started = time.monotonic(); actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)

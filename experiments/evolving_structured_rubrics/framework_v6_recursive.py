"""Frozen framework-v6 Rubrics evaluated with the historical recursive Worker.

Reuse Prompt-v2 node inference, caches, recursive aggregation and official VLRB
metrics. No Manager calls or rubric evolution occur in this control experiment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from critiq.structured import AvailableSlotBackendPool, PairwiseVoteOutput, StructuredRubric

from . import run_rubric_evolution as base
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_phase10 as official
from . import vl_rewardbench_prompt_v2 as recursive
from .experiment_utils import atomic_write_json, canonical_sha256, load_json


def worker_config(config):
    worker = deepcopy(config["worker"])
    if (worker["temperature"], worker["max_tokens"]) != (0.5, 2048):
        raise ValueError("Historical Prompt-v2 requires temperature=0.5, max_tokens=2048")
    # Historical pool validation compares these two model identity strings.
    worker["backend_pool"]["common_checkpoint_id"] = worker["model"]
    worker["worker_request_kwargs"] = {"temperature": 0.5, "max_tokens": 2048}
    worker["structured_max_retries"] = 0
    return worker


def repair(config, work, records, schedule, rubric, predictions, attempts):
    """Retry only technical failures; persist each attempt independently."""
    items = recursive._failure_items(predictions)
    if not items:
        return predictions
    pool = AvailableSlotBackendPool(recursive._pool_spec(config))
    rows = {i: base._model_rows(vlrb._ordered_rows(records, schedule, i))
            for i in range(vlrb.K)}
    evaluators = {i: recursive._v2_evaluator(config, rows[i], pool) for i in rows}
    criteria = {node.criterion.name: node.criterion for node in rubric.nodes.values()}

    def one(item):
        path = work / "retry" / f"{canonical_sha256(item['key'])}.json"
        state = load_json(path) if path.exists() else {"key": item["key"], "attempts": []}
        while not state.get("final_output") and len(state["attempts"]) < attempts - 1:
            output, metrics = evaluators[item["replicate"]].infer_one(
                rows[item["replicate"]][item["sample_index"]],
                criteria[item["criterion_name"]].to_criterion())
            state["attempts"].append({"output": output.to_dict(), "metrics": metrics.to_dict()})
            if output.parse_ok and output.answer_valid:
                state["final_output"] = output.to_dict()
            atomic_write_json(path, state)
        print(f"retry {item['key']}: success={bool(state.get('final_output'))}", flush=True)
        return item, state.get("final_output")

    replacements = {i: {} for i in range(vlrb.K)}
    with ThreadPoolExecutor(max_workers=recursive._pool_spec(config).global_request_concurrency) as pool:
        for item, output in pool.map(one, items):
            if output:
                replacements[item["replicate"]][item["sample_index"], item["criterion_name"]] = (
                    PairwiseVoteOutput.from_dict(output))
    repaired = tuple(recursive._replace_prediction_outputs(p, replacements[i])
                     for i, p in enumerate(predictions))
    unresolved = recursive._failure_items(repaired)
    atomic_write_json(work / "retry_report.json", {
        "original_failures": len(items), "unresolved": unresolved})
    if unresolved:
        raise RuntimeError(f"{work}: {len(unresolved)} failures; resume with larger --attempt-limit")
    return repaired


def run(config_path, source, target, attempts, check_only=False):
    config = load_json(config_path)
    worker = worker_config(config)
    state = load_json(source / "state.json")
    if not state["completed"]:
        raise ValueError("Freeze the completed evolution before this comparison")
    records = vlrb._read_records(
        source / "vlrb", parquet_path=Path(config["data_root"]) / config["datasets"]["vlrb"])
    schedule = vlrb._order_schedule(records)
    rubrics = {"initial": StructuredRubric.load_json(source / "init/rubric.json"),
               "final": StructuredRubric.from_dict(state["rubric"])}
    for name, rubric in rubrics.items():
        control = load_json(source / f"vlrb/{name}.json")
        if control["rubric_sha256"] != rubric.rubric_sha256 or control["k"] != vlrb.K:
            raise ValueError(f"{name}: control rubric or K mismatch")
        if [s["sample_id"] for s in control["samples"]] != [r["sample_id"] for r in records]:
            raise ValueError("Control sample order mismatch")
        if any(tuple(s["orders"]) != tuple(schedule[s["sample_id"]]) for s in control["samples"]):
            raise ValueError("Control A/B presentation order mismatch")
        print(f"{name}: nodes={len(rubric.nodes)}, logical_calls={len(records)*vlrb.K*len(rubric.nodes)}", flush=True)
    manifest = {"structured_worker_request_spec": recursive._v2_request_spec(worker, records, schedule),
                "rubrics": {n: r.rubric_sha256 for n, r in rubrics.items()},
                "sample_count": len(records), "k": vlrb.K,
                "worker": worker, "source": str(source),
                "comparison": "node-wise Prompt-v2 recursive vs unified-subtree plus Arbiter"}
    path = target / "manifest.json"
    if path.exists() and load_json(path) != manifest:
        raise ValueError("Recursive experiment configuration changed")
    if check_only:
        print("Offline check passed; no API calls", flush=True)
        return
    atomic_write_json(path, manifest)
    atomic_write_json(target / "order_schedule.json", schedule)
    controls = load_json(source / "vlrb/report.json")["official"]
    results = {}
    for name, rubric in rubrics.items():
        work = target / name
        work.mkdir(parents=True, exist_ok=True)
        print(f"=== recursive {name} ===", flush=True)
        rubric.save_json(work / "rubric.json")
        _, predictions, _ = recursive._run_predictions(worker, work, manifest, records, schedule, rubric)
        predictions = repair(worker, work, records, schedule, rubric, predictions, attempts)
        logical = recursive._logical_from_predictions(records, schedule, rubric, predictions)
        atomic_write_json(work / "combined/logical_votes.json", logical)
        metrics = official._system_metrics(records, logical["systems"][recursive.EQUAL_SYSTEM]["votes_by_replicate"])
        results[name] = metrics
        atomic_write_json(work / "report.json", {
            "rubric_sha256": rubric.rubric_sha256, "recursive": metrics,
            "arbiter": controls[name], "paired_arbiter_to_recursive": vlrb._paired(
                records, controls[name]["original_index_predictions"], metrics["original_index_predictions"])})
        print(f"{name}: recursive Strict ACC={metrics['strict_accuracy']:.2%}", flush=True)
    atomic_write_json(target / "report.json", {
        "recursive": results, "arbiter": {n: controls[n] for n in results},
        "paired_recursive_initial_to_final": vlrb._paired(
            records, results["initial"]["original_index_predictions"], results["final"]["original_index_predictions"])})
    print("Recursive comparison complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attempt-limit", type=int, default=8)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.attempt_limit < 1:
        parser.error("--attempt-limit must be positive")
    source = args.source_dir.resolve()
    run(args.config, source, (args.output_dir or source / "recursive").resolve(),
        args.attempt_limit, args.check)


if __name__ == "__main__":
    main()

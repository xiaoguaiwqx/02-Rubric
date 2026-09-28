"""Hallucination100: five-shot generated roots with unchanged local evolution.

F5 uses the previous fixed-root result only after a request-equivalence audit.
G5 and GN share one five-example Manager warm-up, then branch at root generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from critiq.structured.schema import StructuredRubric
from . import aligned_system_runtime as system
from . import generated_root_initialization as generated
from . import subtree_local_reflection as local
from . import vlrb_hallucination_fresh_init as fresh
from . import vlrb_hallucination_transfer as transfer
from . import vlrb_generated_roots_f5_audit as fixed_audit
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_aligned_evolution as aligned
from . import vl_rewardbench_phase10 as official
from .experiment_utils import atomic_write_json as write, load_json
from .rubric_factory import file_sha256


DEFAULT_OUTPUT = local.ROOT / "output/vlrb_hallucination100_generated_roots"
VARIANTS = ("f5", "g5", "gn")


def _seed_paths(seed: int, source_root: Path, fixed_root: Path,
                output_root: Path) -> tuple[Path, Path, Path]:
    if seed not in transfer.SEEDS:
        raise ValueError(f"seed must be one of {transfer.SEEDS}")
    return (source_root / f"seed{seed}").resolve(), (
        fixed_root / f"seed{seed}").resolve(), (
        output_root / f"seed{seed}").resolve()


def _frozen(target: Path) -> tuple[dict, dict, list[dict]]:
    config = load_json(target / "g5/config.json")
    split = load_json(target / "g5/split.json")
    rows = local.base.load_rows(config, "discovery")
    if ([row["sample_id"] for row in rows] != split["train_ids"]
            or len(rows) != 100):
        raise RuntimeError("frozen Hallucination100 training rows changed")
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    if file_sha256(parquet) != split["parquet_sha256"]:
        raise RuntimeError("frozen VLRB parquet differs from the split")
    records = _records(config, target)
    by_id = {record["sample_id"]: record for record in records}
    schedule = vlrb._order_schedule(records)
    for row in rows:
        sample_id = row["sample_id"]
        record = by_id[sample_id]
        order = schedule[sample_id][0]
        expected = {
            "question": record["question"],
            "A": record["responses"][order],
            "B": record["responses"][1 - order],
            "answer": ("A" if record["preferred_original_index"] == order
                       else "B"),
            "image_sha256": record["image_sha256"],
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise RuntimeError(f"{sample_id}: frozen training pair differs from VLRB")
        if file_sha256(row["image_path"]) != record["image_sha256"]:
            raise RuntimeError(f"{sample_id}: frozen training image bytes changed")
    for variant in VARIANTS:
        if load_json(target / variant / "config.json") != config:
            raise RuntimeError(f"{variant}: frozen config differs")
        if load_json(target / variant / "split.json") != split:
            raise RuntimeError(f"{variant}: frozen split differs")
    return config, split, rows


def prepare(source: Path, target: Path) -> None:
    if not (source / "run_config.json").is_file():
        raise FileNotFoundError(f"frozen transfer seed missing: {source}")
    for variant in VARIANTS:
        fresh.prepare(source, target / variant)
    _, split, _ = _frozen(target)
    print(f"prepared seed={split['seed']} train=100 "
          f"heldout={len(split['heldout_ids'])}", flush=True)


def generate(source: Path, target: Path, manager_attempts: int) -> None:
    del source  # Preparation already freezes the source data under each variant.
    config, split, rows = _frozen(target)
    load_dotenv(config.get("env_file"), override=False)
    generated.generate_r0_pair(
        rows, seed=split["seed"],
        manager_config=dict(config["manager"],
                            env_file=config.get("env_file", ".env")),
        output_dir=target, attempt_limit=manager_attempts,
        count_instructions=generated.LEGACY_COUNT_INSTRUCTIONS)


def _r0(target: Path, variant: str) -> StructuredRubric:
    if variant == "f5":
        return local.base.build_multicrit_open_ended_init_rubric()
    rubric = StructuredRubric.load_json(target / variant / "r0/rubric.json")
    count = len(rubric.root_ids)
    if ((variant == "g5" and count != 5)
            or (variant == "gn" and not 2 <= count <= 7)
            or rubric.edges):
        raise RuntimeError(f"{variant}: generated R0 has invalid root structure")
    return rubric


def run(source: Path, fixed: Path, target: Path, *, attempts: int,
        manager_attempts: int, variants: Sequence[str]) -> None:
    del source
    config, split, rows = _frozen(target)
    load_dotenv(config.get("env_file"), override=False)
    infeasible = []
    for variant in variants:
        if variant == "f5" and (fixed / "vlrb/final.json").is_file():
            try:
                fixed_audit.audit_fixed_seed11(
                    fixed, config, split, _records(config, target),
                    target / "f5/reuse_manifest.json")
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"f5: historical result is not equivalent ({exc}); "
                      "running the fixed control", flush=True)
            else:
                print("f5: verified historical S0/Final predictions reused",
                      flush=True)
                continue
        output = target / variant
        rubric = _r0(target, variant)
        existing = output / "r0/rubric.json"
        if existing.is_file() and StructuredRubric.load_json(existing).rubric_sha256 != rubric.rubric_sha256:
            raise RuntimeError(f"{variant}: R0 changed in an existing run")
        manager = local.make_manager(
            dict(config["manager"], env_file=config.get("env_file", ".env")),
            manager_attempts, n_roots=len(rubric.root_ids))
        try:
            local.run(config, output, attempts=attempts, manager=manager,
                      rows=[dict(row) for row in rows], r0=rubric)
        except RuntimeError as exc:
            if "insufficient supported patterns for two initial clusters; S0 not created" not in str(exc):
                raise
            write(output / "init_feasibility.json", {
                "status": "infeasible", "reason": str(exc),
                "r0_rubric_sha256": rubric.rubric_sha256,
            })
            infeasible.append(f"{variant}: {exc}")
            print(f"{variant}: initial Split infeasible ({exc})", flush=True)
            continue
        state = load_json(output / "state.json")
        if not state["completed"]:
            raise RuntimeError(f"{variant}: evolution did not complete")
        write(output / "init_feasibility.json", {
            "status": "completed", "r0_rubric_sha256": rubric.rubric_sha256,
        })
        print(f"{variant}: roots={len(rubric.root_ids)} "
              f"epochs={state['epoch']} completed", flush=True)
    if infeasible:
        raise RuntimeError("; ".join(infeasible))


def _records(config: dict, target: Path) -> tuple[dict, ...]:
    records = transfer._records(config, target)
    if len(records) != 1247:
        raise RuntimeError("frozen VLRB does not contain 1247 rows")
    return records


def _audit_fixed(fixed: Path, target: Path, config: dict, split: dict,
                 records: Sequence[Mapping[str, Any]]) -> Path:
    local_state = target / "f5/state.json"
    if (fixed / "vlrb/final.json").is_file():
        try:
            fixed_audit.audit_fixed_seed11(
                fixed, config, split, records,
                target / "f5/reuse_manifest.json")
        except (FileNotFoundError, ValueError, RuntimeError):
            if not (local_state.is_file() and load_json(local_state)["completed"]):
                raise
        else:
            return fixed
    if local_state.is_file() and load_json(local_state)["completed"]:
        return target / "f5"
    raise RuntimeError("F5 has neither an equivalent historical nor a completed local run")


def _r0_heldout(config: dict, split: dict, records: Sequence[Mapping[str, Any]],
                target: Path, variant: str, attempts: int) -> None:
    output = target / variant
    rubric = _r0(target, variant)
    heldout = set(split["heldout_ids"])
    rows = tuple(row for row in system.support.vlrb_rows(records)
                 if row["sample_id"] in heldout)
    if len(rows) != 648:
        raise RuntimeError("frozen hallucination heldout is not 648 rows")
    # The schedule depends on the full VLRB order; never recalculate it on 648.
    schedule = vlrb._order_schedule(records)
    system_orders = {row["sample_id"]: schedule[row["sample_id"]] for row in rows}
    local.base.evaluate(config, output, "vlrb/r0_heldout", rows, rubric,
                        orders=system_orders, attempts=attempts)


def external(fixed: Path, target: Path, *, attempts: int,
             variants: Sequence[str]) -> None:
    config, split, _ = _frozen(target)
    records = _records(config, target)
    # Complete both generated evolutions before reading any new VLRB accuracy.
    for variant in ("g5", "gn"):
        state = load_json(target / variant / "state.json")
        if not state["completed"]:
            raise RuntimeError(f"{variant}: freeze Final before VLRB evaluation")
    fixed_source = _audit_fixed(fixed, target, config, split, records)
    for variant in variants:
        if variant != "f5" or fixed_source != fixed:
            local.base.external(config, target / variant, "vlrb", attempts)
        _r0_heldout(config, split, records, target, variant, attempts)


def _subsets(records: Sequence[Mapping[str, Any]], split: dict) -> dict[str, set[str]]:
    all_ids = {row["sample_id"] for row in records}
    train = set(split["train_ids"])
    heldout = set(split["heldout_ids"])
    groups = {name: {row["sample_id"] for row in records if row["group"] == name}
              for name in ("general", "hallucination", "reasoning")}
    clean = heldout | groups["general"] | groups["reasoning"]
    if (len(all_ids), len(train), len(heldout), len(clean)) != (1247, 100, 648, 1146):
        raise RuntimeError("VLRB subset sizes changed")
    if train & clean or clean - all_ids:
        raise RuntimeError("clean evaluation set overlaps training or is absent")
    return dict(full=all_ids, train=train, nontrain_1147=all_ids-train,
                clean_1146=clean, heldout_hallucination=heldout, **groups)


def _load_predictions(path: Path, rubric: StructuredRubric,
                      records: Sequence[Mapping[str, Any]],
                      orders: Mapping[str, Sequence[int]]) -> tuple[dict, list[int | None]]:
    value = system.load(path)
    ids = [row["sample_id"] for row in records]
    if value["rubric_sha256"] != rubric.rubric_sha256 or value["k"] != 3:
        raise RuntimeError(f"{path}: Rubric or K differs")
    if [row["sample_id"] for row in value["samples"]] != ids:
        raise RuntimeError(f"{path}: VLRB sample order differs")
    if value["metrics"]["technical_failure_count"]:
        raise RuntimeError(f"{path}: unresolved technical failure")
    for sample in value["samples"]:
        sid = sample["sample_id"]
        if tuple(sample["orders"]) != tuple(orders[sid]):
            raise RuntimeError(f"{path}: A/B schedule differs for {sid}")
    if value["metrics"] != system.metrics(value, system.support.vlrb_rows(records)):
        raise RuntimeError(f"{path}: saved metrics differ from sample-level predictions")
    votes = aligned._votes(value)
    predictions = [vlrb._majority([replicate[index] for replicate in votes])
                   for index in range(len(records))]
    if len(records) == 1247:
        official_predictions = official._system_metrics(
            records, votes)["original_index_predictions"]
        if predictions != official_predictions:
            raise RuntimeError(f"{path}: official vote aggregation differs")
    return value, predictions


def _root_predictions(value: Mapping[str, Any], root_id: str) -> list[int | None]:
    predictions = []
    for sample in value["samples"]:
        votes = []
        for replicate in range(value["k"]):
            item = sample["replicates"][str(replicate)]
            answer = system._answer(item["subtrees"][root_id], int(item["order"]))
            votes.append(0 if answer == "A" else 1 if answer == "B" else None)
        predictions.append(vlrb._majority(votes) if len(votes) == 3 else votes[0])
    return predictions


def _root_metrics(value: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                  subsets: Mapping[str, set[str]], rubric: StructuredRubric) -> dict:
    return {root: {name: transfer._subset(records, predictions, ids)
                   for name, ids in subsets.items()}
            for root in rubric.root_ids
            for predictions in [_root_predictions(value, root)]}


def _training_root_metrics(value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                           rubric: StructuredRubric) -> dict:
    result = {}
    for root in rubric.root_ids:
        predictions = []
        for sample in value["samples"]:
            replicate = sample["replicates"]["0"]
            answer = system._answer(replicate["subtrees"][root], int(replicate["order"]))
            if answer not in {"A", "B", "None"}:
                raise RuntimeError(f"training root {root}: unresolved technical failure")
            predictions.append(answer)
        total = len(rows)
        covered = sum(answer in {"A", "B"} for answer in predictions)
        correct = sum(answer == row["answer"] for answer, row in zip(predictions, rows))
        result[root] = {
            "correct": correct,
            "covered": covered,
            "total": total,
            "covered_acc": correct / covered if covered else None,
            "coverage": covered / total,
            "strict_acc": correct / total,
        }
    return result


def _paired_ci(records: Sequence[Mapping[str, Any]], before: Sequence[int | None],
               after: Sequence[int | None], ids: set[str]) -> dict:
    selected = [(row, x, y) for row, x, y in zip(records, before, after)
                if row["sample_id"] in ids]
    return aligned._paired_bootstrap_delta_ci(
        [item[0] for item in selected], [item[1] for item in selected],
        [item[2] for item in selected], iterations=10_000, seed=20260925)


def _worker_cost(directory: Path) -> dict:
    """Count training and VLRB requests, excluding unrelated Dev/smoke runs."""
    totals = {kind: dict(logical_calls=0, model_generations=0, api_attempts=0,
                         input_tokens=0, output_tokens=0, latency_seconds=0.0,
                         missing_usage_calls=0, missing_usage_attempts=0)
              for kind in ("subtree", "arbiter")}
    if not directory.is_dir():
        return totals
    for path in directory.rglob("*.json"):
        section = path.relative_to(directory).parts[0]
        if section not in {"r0", "init", "vlrb"} and not (
                len(section) == 3 and section.startswith("e")
                and section[1:].isdigit()):
            continue
        item = load_json(path)
        if "cache_key" not in item:
            continue
        request_key = item.get("request", {}).get("request_key", {})
        kind = ("subtree" if request_key.get("kind") == "aligned_unified_subtree"
                else "arbiter" if request_key.get("kind") == "aligned_global_arbiter"
                else None)
        if kind is None:
            raise RuntimeError(f"unrecognized Worker cache request: {path}")
        metrics = item.get("metrics", {})
        cost = totals[kind]
        cost["logical_calls"] += 1
        cost["model_generations"] += item.get("model_generation_count", 0)
        cost["api_attempts"] += metrics.get("api_attempts", 0) or 0
        for field in ("input_tokens", "output_tokens", "latency_seconds"):
            cost[field] += metrics.get(field, 0) or 0
        missing = sum(
            attempt.get("usage_complete") is False
            or attempt.get("input_tokens") is None
            or attempt.get("output_tokens") is None
            for attempt in item.get("attempt_metrics", ()))
        cost["missing_usage_attempts"] += missing
        if missing:
            cost["missing_usage_calls"] += 1
    return totals


def _generation_cost(target: Path, variant: str) -> dict | None:
    if variant == "f5":
        return None
    paths = (target / "warmup/transcript.json",
             target / variant / "r0/generation.json")
    result = {}
    for label, path in zip(("shared_warmup", "root_generation"), paths):
        value = load_json(path)
        entries = value["turns"] if label == "shared_warmup" else [value]
        attempts = [attempt for entry in entries for attempt in entry["attempts"]]
        result[label] = {
            "successful_turns": (sum("response" in entry for entry in entries)
                                  if label == "shared_warmup" else int("parsed" in value)),
            "attempts": len(attempts),
            "api_attempts": sum(item.get("metrics", {}).get("api_attempts", 0)
                                for item in attempts),
            "input_tokens": sum(item.get("metrics", {}).get("input_tokens") or 0
                                for item in attempts),
            "output_tokens": sum(item.get("metrics", {}).get("output_tokens") or 0
                                 for item in attempts),
            "latency_seconds": sum(item.get("metrics", {}).get("latency_seconds", 0)
                                   for item in attempts),
            "missing_usage_attempts": sum(
                item.get("metrics", {}).get("input_tokens") is None
                or item.get("metrics", {}).get("output_tokens") is None
                for item in attempts),
        }
    return result


def _costs(source: Path, target: Path, variant: str,
           rubrics: Mapping[str, StructuredRubric]) -> dict:
    manager_dirs = [source / "init", *sorted(source.glob("e[0-9][0-9]"))]
    epochs = [load_json(path) for path in sorted(source.glob("e[0-9][0-9]/summary.json"))]
    result = {
        "nodes": {stage: len(rubric.nodes) for stage, rubric in rubrics.items()},
        "epochs_completed": load_json(source / "state.json")["epoch"],
        "evolution_wall_seconds": sum(item.get("wall_seconds", 0) for item in epochs),
        "manager": local.base.manager_cost(manager_dirs),
        "worker": _worker_cost(source / "cache"),
        "root_generation": _generation_cost(target, variant),
        "wall_seconds_by_artifact": {},
        "source_output": str(source),
    }
    if variant == "f5" and source != target / "f5":
        result["r0_heldout_supplement_worker"] = _worker_cost(
            target / "f5/cache")
        result["historical_cost_reused"] = True
    else:
        result["historical_cost_reused"] = False
    return result


def report(fixed: Path, target: Path) -> None:
    config, split, rows = _frozen(target)
    records = _records(config, target)
    fixed_source = _audit_fixed(fixed, target, config, split, records)
    orders = vlrb._order_schedule(records)
    subsets = _subsets(records, split)
    warmup = load_json(target / "warmup/transcript.json")
    warmup_samples = warmup["request"]["samples"]
    result: dict[str, Any] = dict(
        seed=split["seed"], fixed_source=str(fixed_source),
        warmup={"sample_ids": [sample["sample_id"] for sample in warmup_samples],
                "source_counts": dict(Counter(sample["source"] for sample in warmup_samples)),
                "flipped_sample_ids": [sample["sample_id"] for sample in warmup_samples
                                       if sample["flipped"]]},
        root_counts={}, root_catalog={}, root_generation={},
        systems={}, roots={}, paired={}, costs={})
    predictions = {}
    for variant in VARIANTS:
        source = fixed_source if variant == "f5" else target / variant
        rubrics = {
            "r0": _r0(target, variant),
            "s0": StructuredRubric.load_json(source / "init/rubric.json"),
            "final": StructuredRubric.load_json(source / "final.json"),
        }
        result["root_counts"][variant] = len(rubrics["r0"].root_ids)
        result["root_catalog"][variant] = [
            {"root_id": root, "name": rubrics["r0"].get_node(root).criterion.name,
             "description": rubrics["r0"].get_node(root).criterion.description}
            for root in rubrics["r0"].root_ids]
        if variant != "f5":
            generation = load_json(target / variant / "r0/generation.json")
            result["root_generation"][variant] = {
                "count_reason": generation["parsed"]["count_reason"],
                "warmup_history_sha256": generation["request"]["warmup_history_sha256"],
                "attempt_count": len(generation["attempts"]),
            }
        result["costs"][variant] = _costs(source, target, variant, rubrics)
        result["systems"][variant] = {}
        result["roots"][variant] = {}
        for stage in ("s0", "final"):
            path = source / "vlrb" / ("initial.json" if stage == "s0" else "final.json")
            value, votes = _load_predictions(path, rubrics[stage], records, orders)
            result["costs"][variant]["wall_seconds_by_artifact"][stage + "_vlrb"] = (
                value.get("wall_seconds"))
            predictions[variant, stage] = votes
            result["systems"][variant][stage] = {
                name: transfer._subset(records, votes, ids)
                for name, ids in subsets.items()}
            result["roots"][variant][stage] = _root_metrics(
                value, records, subsets, rubrics[stage])
        heldout = subsets["heldout_hallucination"]
        heldout_records = tuple(row for row in records if row["sample_id"] in heldout)
        heldout_orders = {row["sample_id"]: orders[row["sample_id"]]
                          for row in heldout_records}
        r0_value, r0_votes = _load_predictions(
            target / variant / "vlrb/r0_heldout.json", rubrics["r0"],
            heldout_records, heldout_orders)
        result["costs"][variant]["wall_seconds_by_artifact"]["r0_heldout"] = (
            r0_value.get("wall_seconds"))
        result["systems"][variant]["r0_heldout"] = transfer._subset(
            heldout_records, r0_votes, heldout)
        result["roots"][variant]["r0_heldout"] = _root_metrics(
            r0_value, heldout_records, {"heldout_hallucination": heldout},
            rubrics["r0"])
        for stage, relative in (("r0", "r0/system.json"),
                                ("s0", "init/system.json"),
                                ("final", load_json(source / "state.json")["baseline"] + ".json")):
            train_value = system.load(source / relative)
            if ([sample["sample_id"] for sample in train_value["samples"]]
                    != [row["sample_id"] for row in rows]
                    or train_value["k"] != 1
                    or train_value["rubric_sha256"] != rubrics[stage].rubric_sha256):
                raise RuntimeError(f"{source / relative}: training artifact differs")
            if train_value["metrics"] != system.metrics(train_value, rows):
                raise RuntimeError(f"{source / relative}: saved training metrics differ")
            result["costs"][variant]["wall_seconds_by_artifact"][stage + "_train"] = (
                train_value.get("wall_seconds"))
            result["systems"][variant][stage + "_train"] = train_value["metrics"]
            result["roots"][variant][stage + "_train"] = _training_root_metrics(
                train_value, rows, rubrics[stage])
    comparisons = (
        ("gn_final_vs_f5_final", ("f5", "final"), ("gn", "final")),
        ("g5_final_vs_f5_final", ("f5", "final"), ("g5", "final")),
        ("gn_final_vs_g5_final", ("g5", "final"), ("gn", "final")),
        *((f"{variant}_final_vs_s0", (variant, "s0"), (variant, "final"))
          for variant in VARIANTS),
    )
    for label, before_key, after_key in comparisons:
        before, after = predictions[before_key], predictions[after_key]
        result["paired"][label] = {}
        for name, ids in subsets.items():
            item = transfer._paired(records, before, after, ids)
            if name == "heldout_hallucination":
                item["bootstrap"] = _paired_ci(records, before, after, ids)
            result["paired"][label][name] = item
    write(target / "report.json", result)
    for variant in VARIANTS:
        item = result["systems"][variant]["final"]["heldout_hallucination"]
        print(f"{variant} Final heldout: {item['correct']}/{item['total']} "
              f"({item['strict_acc']:.2%})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "run", "vlrb",
                                          "report", "all"))
    parser.add_argument("--seed", type=int, choices=transfer.SEEDS, default=11)
    parser.add_argument("--source-root", type=Path, default=transfer.DEFAULT_OUTPUT)
    parser.add_argument("--fixed-root", type=Path, default=fresh.DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=("f5", "g5", "gn", "both"),
                        default="both")
    parser.add_argument("--attempt-limit", type=int, default=10)
    parser.add_argument("--manager-attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if min(args.attempt_limit, args.manager_attempt_limit) < 1:
        parser.error("attempt limits must be positive")
    if args.stage in {"all", "report"} and args.variant != "both":
        parser.error("all and report require --variant both")
    source, fixed, target = _seed_paths(
        args.seed, args.source_root, args.fixed_root, args.output_root)
    stages = ("prepare", "generate", "run", "vlrb", "report") if args.stage == "all" else (args.stage,)
    selected = VARIANTS if args.variant == "both" else (args.variant,)
    for stage in stages:
        if stage == "prepare":
            prepare(source, target)
        elif stage == "generate":
            generate(source, target, args.manager_attempt_limit)
        elif stage == "run":
            run(source, fixed, target, attempts=args.attempt_limit,
                manager_attempts=args.manager_attempt_limit, variants=selected)
        elif stage == "vlrb":
            external(fixed, target, attempts=args.attempt_limit, variants=selected)
        else:
            report(fixed, target)


if __name__ == "__main__":
    main()

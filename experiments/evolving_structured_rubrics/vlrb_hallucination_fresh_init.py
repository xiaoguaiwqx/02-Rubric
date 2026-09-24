"""Hallucination100 Split initialization followed by the existing local evolution.

Reuse a prepared transfer seed's exact data and settings. Evaluate both the new
Init and Final, and compare saved predictions on the original frozen held-out IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from . import subtree_local_reflection as local
from . import vlrb_hallucination_transfer as transfer
from .experiment_utils import atomic_write_json as write, load_json


DEFAULT_OUTPUT = local.ROOT / "output/vlrb_hallucination100_fresh_init"


def prepare(source: Path, target: Path) -> dict:
    config = load_json(source / "run_config.json")
    split = load_json(source / "split.json")
    rows = local.base.load_rows(config, "discovery")
    if [row["sample_id"] for row in rows] != split["train_ids"]:
        raise ValueError("source data differs from frozen training IDs")
    for name, value in (("config.json", config), ("split.json", split)):
        path = target / name
        if path.exists() and load_json(path) != value:
            raise ValueError(f"existing experiment differs: {path}")
        write(path, value)
    local.check(config, target)
    return config


def report(source: Path, target: Path, config: dict) -> None:
    split = load_json(target / "split.json")
    previous = load_json(source / "transfer_report.json")
    history = Path(previous["historical_output"])
    records = transfer._records(config, source.parent)
    ids = [row["sample_id"] for row in records]
    subsets = {
        "full": set(ids), "train": set(split["train_ids"]),
        "heldout_hallucination": set(split["heldout_ids"]),
        **{group: {r["sample_id"] for r in records if r["group"] == group}
           for group in ("general", "hallucination", "reasoning")},
    }
    artifacts = {
        "historical_init": history / "vlrb/initial.json",
        "historical_init_hallucination_final": source / "vlrb/final.json",
        "hallucination_init": target / "vlrb/initial.json",
        "hallucination_final": target / "vlrb/final.json",
    }
    predictions, metrics = {}, {}
    for name, path in artifacts.items():
        value = load_json(path)
        if [s["sample_id"] for s in value["samples"]] != ids or value["k"] != 3:
            raise ValueError(f"VLRB sample order or K differs: {path}")
        if value["metrics"]["technical_failure_count"]:
            raise ValueError(f"unresolved technical failures: {path}")
        official = transfer.official._system_metrics(records, transfer.aligned._votes(value))
        predictions[name] = official["original_index_predictions"]
        metrics[name] = {label: transfer._subset(records, predictions[name], subset)
                         for label, subset in subsets.items()}
    pairs = (("historical_init", "hallucination_init"),
             ("hallucination_init", "hallucination_final"),
             ("historical_init_hallucination_final", "hallucination_final"))
    paired = {f"{after}_vs_{before}": {
        label: transfer._paired(records, predictions[before], predictions[after], subset)
        for label, subset in subsets.items()} for before, after in pairs}
    write(target / "transfer_report.json", dict(
        source_output=str(source), metrics=metrics, paired=paired,
        artifacts={name: str(path) for name, path in artifacts.items()}))
    for name, values in metrics.items():
        print(name, {k: f"{v['correct']}/{v['total']}" for k, v in values.items()}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "vlrb", "report", "all"))
    parser.add_argument("--seed", type=int, choices=transfer.SEEDS, default=11)
    parser.add_argument("--source-root", type=Path, default=transfer.DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt-limit", type=int, default=10)
    parser.add_argument("--manager-attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if min(args.attempt_limit, args.manager_attempt_limit) < 1:
        parser.error("attempt limits must be positive")
    source = (args.source_root / f"seed{args.seed}").resolve()
    target = (args.output_root / f"seed{args.seed}").resolve()
    if target == source:
        parser.error("fresh initialization requires a separate output directory")
    config = prepare(source, target)
    load_dotenv(config.get("env_file"), override=False)
    stages = ("run", "vlrb", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "run":
            manager = local.make_manager(
                dict(config["manager"], env_file=config.get("env_file", ".env")),
                args.manager_attempt_limit)
            local.run(config, target, attempts=args.attempt_limit, manager=manager)
        elif stage == "vlrb":
            local.base.external(config, target, "vlrb", args.attempt_limit)
        elif stage == "report":
            report(source, target, config)


if __name__ == "__main__":
    main()

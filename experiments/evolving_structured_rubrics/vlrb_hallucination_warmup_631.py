"""Hallucination150: generate roots after a 6/3/1 ten-example warmup."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from critiq.structured.schema import StructuredRubric

from . import aligned_system_runtime as system
from . import framework_v6 as base
from . import generated_root_initialization as generated
from . import subtree_local_reflection as local
from . import vl_rewardbench as vlrb
from . import vlrb_category_specialized_rubrics as category
from . import vlrb_hallucination_generated_roots as previous
from . import vlrb_hallucination_transfer as transfer
from .experiment_utils import load_json
from .rubric_factory import file_sha256


PROTOCOL = "vlrb-hallucination-warmup-631-10-v1"
SOURCE_ROOT = base.ROOT / "output/vlrb_category_specialized_rubrics"
HISTORY_ROOT = base.ROOT / "output/vlrb_hallucination100_generated_roots"
DEFAULT_OUTPUT = base.ROOT / "output/vlrb_hallucination_warmup631_10"
QUOTAS = {"povid": 6, "rlaif-v": 3, "rlhf-v": 1}
STAGE_CONCURRENCY = 10


def _select(rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict]:
    selected = []
    for source, quota in QUOTAS.items():
        candidates = sorted((row for row in rows if row["source"] == source),
                            key=lambda row: row["sample_id"])
        if len(candidates) < quota:
            raise ValueError(f"{source}: need {quota} warmup examples, found {len(candidates)}")
        selected.extend(random.Random(
            f"{seed}:hallucination:{source}:warmup10").sample(candidates, quota))
    random.Random(f"{seed}:hallucination:warmup10-order").shuffle(selected)
    return [dict(row) for row in selected]


def prepare(source: Path, target: Path, seed: int) -> tuple[dict, dict, tuple[dict, ...], list[dict]]:
    config = deepcopy(category._config(source, "hallucination"))
    config["manager"]["stage_concurrency"].update(
        signature=STAGE_CONCURRENCY, case_reflection=STAGE_CONCURRENCY)
    split = load_json(source / "split.json")
    if split["seed"] != seed:
        raise ValueError(f"frozen source split uses seed {split['seed']}, not {seed}")
    rows = base.load_rows(config, "discovery")
    if [row["sample_id"] for row in rows] != split["train_ids"]["hallucination"]:
        raise ValueError("Hallucination150 discovery rows differ from the frozen split")
    records = category._records(config, source)
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    if file_sha256(parquet) != split["parquet_sha256"]:
        raise ValueError("VLRB parquet differs from the frozen source split")
    examples = _select(rows, seed)
    category._freeze_json(target / "selection.json", dict(
        protocol=PROTOCOL, seed=seed, quotas=QUOTAS,
        source_split_sha256=file_sha256(source / "split.json"),
        source_discovery_sha256=file_sha256(
            Path(config["datasets"]["discovery"])),
        sample_ids=[row["sample_id"] for row in examples],
        sources=[row["source"] for row in examples],
    ))
    category._freeze_json(target / "gn/config.json", config)
    local.check(config, target / "gn")
    print("prepared Hallucination150 warmup: povid=6 rlaif-v=3 rlhf-v=1",
          flush=True)
    return config, split, records, examples


def generate(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
             examples: Sequence[Mapping[str, Any]], target: Path, seed: int,
             manager_attempts: int) -> None:
    load_dotenv(config.get("env_file"), override=False)
    rubrics = generated.generate_r0_pair(
        rows, seed=seed,
        manager_config=dict(config["manager"],
                            env_file=config.get("env_file", ".env")),
        output_dir=target, attempt_limit=manager_attempts,
        expected_count=150, warmup_examples=examples,
        variants=("gn",), protocol=PROTOCOL,
    )
    print(f"generated {len(rubrics['gn'].root_ids)} Hallucination roots", flush=True)


def evolve(config: Mapping[str, Any], rows: list[dict], target: Path,
           attempts: int, manager_attempts: int) -> None:
    branch = target / "gn"
    r0 = StructuredRubric.load_json(branch / "r0/rubric.json")
    manager = local.make_manager(
        dict(config["manager"], env_file=config.get("env_file", ".env")),
        manager_attempts, n_roots=len(r0.root_ids))
    load_dotenv(config.get("env_file"), override=False)
    local.run(config, branch, attempts=attempts, manager=manager,
              rows=rows, r0=r0)
    print(f"Hallucination evolution complete: roots={len(r0.root_ids)}",
          flush=True)


def external(config: Mapping[str, Any], records: tuple[dict, ...],
             target: Path, attempts: int) -> None:
    branch = target / "gn"
    if not load_json(branch / "state.json")["completed"]:
        raise RuntimeError("freeze Final before external evaluation")
    rubric = StructuredRubric.load_json(branch / "final.json")
    base.evaluate(config, branch, "vlrb/final", system.support.vlrb_rows(records),
                  rubric, orders=vlrb._order_schedule(records), attempts=attempts)


def report(config: Mapping[str, Any], split: Mapping[str, Any],
           records: tuple[dict, ...], source: Path, history: Path,
           target: Path) -> None:
    branch = target / "gn"
    schedule = vlrb._order_schedule(records)
    scopes = category._scopes(records, split)
    sources = {
        "warmup_631_10": branch,
        "category_random_5": source / "hallucination/gn",
        "historical_generated_5": history / "g5",
    }
    if load_json(sources["historical_generated_5"] / "split.json")["parquet_sha256"] != split["parquet_sha256"]:
        raise ValueError("historical generated-five VLRB parquet differs")
    if config["worker"] != load_json(sources["historical_generated_5"] / "config.json")["worker"]:
        raise ValueError("historical generated-five Worker differs")
    predictions = {}
    metrics = {}
    for name, directory in sources.items():
        rubric = StructuredRubric.load_json(directory / "final.json")
        _, votes = previous._load_predictions(directory / "vlrb/final.json",
                                             rubric, records, schedule)
        predictions[name] = votes
        metrics[name] = {
            "root_count": len(rubric.root_ids),
            "full": transfer._subset(records, votes, scopes["full"]),
            "heldout_998": transfer._subset(records, votes, scopes["heldout"]),
            "hallucination_749": transfer._subset(
                records, votes, scopes["full_group"]["hallucination"]),
            "hallucination_599": transfer._subset(
                records, votes, scopes["heldout_group"]["hallucination"]),
        }
    paired = {name: category._paired_summary(
        records, predictions[name], predictions["warmup_631_10"],
        scopes["heldout_group"]["hallucination"])
        for name in ("category_random_5", "historical_generated_5")}
    state = load_json(branch / "state.json")
    training = {}
    for stage, path in (("r0", "r0/system.json"),
                        ("s0", "init/system.json"),
                        ("final", f"{state['baseline']}.json")):
        values = system.load(branch / path)["metrics"]
        training[stage] = {key: values[key] for key in (
            "sample_count", "strict_accuracy", "coverage",
            "covered_accuracy", "technical_failure_count")}
    r0 = StructuredRubric.load_json(branch / "r0/rubric.json")
    category._freeze_json(target / "report.json", dict(
        protocol=PROTOCOL, seed=split["seed"],
        selection=str(target / "selection.json"),
        roots=[dict(name=r0.get_node(root).criterion.name,
                    description=r0.get_node(root).criterion.description)
               for root in r0.root_ids],
        training=training, metrics=metrics, paired_hallucination_599=paired,
    ))
    result = metrics["warmup_631_10"]["hallucination_599"]
    print(f"Hallucination heldout: {result['correct']}/{result['total']} "
          f"({result['strict_acc']:.2%})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "evolve",
                                          "vlrb", "report", "all"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt-limit", type=int, default=10)
    parser.add_argument("--manager-attempt-limit", type=int, default=10)
    args = parser.parse_args()
    source = (args.source_root / f"seed{args.seed}").resolve()
    history = (args.history_root / f"seed{args.seed}").resolve()
    target = (args.output_root / f"seed{args.seed}").resolve()
    config, split, records, examples = prepare(source, target, args.seed)
    rows = base.load_rows(config, "discovery")
    stages = (("generate", "evolve", "vlrb", "report")
              if args.stage == "all" else (args.stage,))
    for stage in stages:
        if stage == "generate":
            generate(config, rows, examples, target, args.seed,
                     args.manager_attempt_limit)
        elif stage == "evolve":
            evolve(config, rows, target, args.attempt_limit,
                   args.manager_attempt_limit)
        elif stage == "vlrb":
            external(config, records, target, args.attempt_limit)
        elif stage == "report":
            report(config, split, records, source, history, target)


if __name__ == "__main__":
    main()

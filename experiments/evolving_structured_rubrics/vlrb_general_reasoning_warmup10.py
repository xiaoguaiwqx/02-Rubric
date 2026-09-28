"""Ten-example source-stratified warmup for VLRB General and Reasoning."""

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
from .experiment_utils import atomic_write_json as write, load_json
from .rubric_factory import file_sha256


PROTOCOL = "vlrb-general-reasoning-warmup10-v1"
GROUPS = ("general", "reasoning")
QUOTAS = {
    "general": {"wildvision-battle": 9, "vlfeedback": 1},
    "reasoning": {"mathverse": 6, "mmmu": 4},
}
TRAIN_COUNTS = {"general": 36, "reasoning": 63}
SOURCE_COUNTS = {
    "general": {"wildvision-battle": 34, "vlfeedback": 2},
    "reasoning": {"mathverse": 40, "mmmu": 23},
}
SOURCE_ROOT = base.ROOT / "output/vlrb_category_specialized_rubrics"
HALLUCINATION_ROOT = base.ROOT / "output/vlrb_hallucination_warmup631_10"
DEFAULT_OUTPUT = base.ROOT / "output/vlrb_general_reasoning_warmup10"
STAGE_CONCURRENCY = 15


def _origin(row: Mapping[str, Any], group: str) -> str:
    if group == "general":
        return str(row["source"])
    sample_id = str(row["sample_id"])
    for source in QUOTAS["reasoning"]:
        if sample_id.startswith(f"{source}_"):
            return source
    raise ValueError(f"Reasoning source cannot be inferred from {sample_id}")


def _select(rows: Sequence[Mapping[str, Any]], seed: int,
            group: str) -> list[dict]:
    selected = []
    for source, quota in QUOTAS[group].items():
        candidates = sorted((row for row in rows if _origin(row, group) == source),
                            key=lambda row: row["sample_id"])
        if len(candidates) < quota:
            raise ValueError(f"{group}/{source}: need {quota} warmup examples, "
                             f"found {len(candidates)}")
        selected.extend(random.Random(
            f"{seed}:{group}:{source}:warmup10").sample(candidates, quota))
    random.Random(f"{seed}:{group}:warmup10-order").shuffle(selected)
    return [dict(row) for row in selected]


def _branch(target: Path, group: str) -> Path:
    return target / group / "gn"


def _save_branch_config(path: Path, config: dict) -> None:
    if path.is_file():
        previous = load_json(path)
        previous["manager"]["stage_concurrency"] = config["manager"]["stage_concurrency"]
        if previous != config:
            raise ValueError(f"frozen branch configuration changed beyond Manager stage concurrency: {path}")
    write(path, config)


def prepare(source: Path, hallucination: Path, target: Path,
            seed: int) -> tuple[dict, tuple[dict, ...]]:
    split = load_json(source / "split.json")
    if split["seed"] != seed:
        raise ValueError(f"frozen source split uses seed {split['seed']}, not {seed}")
    hall_selection = load_json(hallucination / "selection.json")
    if (hall_selection["source_split_sha256"] != file_sha256(source / "split.json")
            or hall_selection["seed"] != seed):
        raise ValueError("reused Hallucination branch has a different frozen split")
    configs = {group: deepcopy(category._config(source, group)) for group in GROUPS}
    records = category._records(configs["general"], source)
    parquet = Path(configs["general"]["data_root"]) / configs["general"]["datasets"]["vlrb"]
    if file_sha256(parquet) != split["parquet_sha256"]:
        raise ValueError("VLRB parquet differs from the frozen source split")
    hall_config = load_json(hallucination / "gn/config.json")
    if any(config["worker"] != hall_config["worker"] for config in configs.values()):
        raise ValueError("reused Hallucination Worker differs from the new branches")
    category._scopes(records, split)
    for group, config in configs.items():
        config["manager"]["stage_concurrency"].update(
            signature=STAGE_CONCURRENCY, case_reflection=STAGE_CONCURRENCY)
        rows = base.load_rows(config, "discovery")
        if (len(rows) != TRAIN_COUNTS[group]
                or [row["sample_id"] for row in rows] != split["train_ids"][group]):
            raise ValueError(f"{group} discovery rows differ from the frozen split")
        counts = {origin: sum(_origin(row, group) == origin for row in rows)
                  for origin in SOURCE_COUNTS[group]}
        if counts != SOURCE_COUNTS[group]:
            raise ValueError(f"{group} source counts differ from the frozen protocol: {counts}")
        examples = _select(rows, seed, group)
        group_target = target / group
        category._freeze_json(group_target / "selection.json", dict(
            protocol=PROTOCOL, seed=seed, group=group, quotas=QUOTAS[group],
            source_split_sha256=file_sha256(source / "split.json"),
            source_discovery_sha256=file_sha256(
                Path(config["datasets"]["discovery"])),
            sample_ids=[row["sample_id"] for row in examples],
            sources=[_origin(row, group) for row in examples],
        ))
        _save_branch_config(_branch(target, group) / "config.json", config)
        local.check(config, _branch(target, group))
        print(f"prepared {group}: {len(rows)} train, "
              f"warmup={QUOTAS[group]}", flush=True)
    return split, records


def generate(target: Path, split: Mapping[str, Any],
             manager_attempts: int) -> None:
    for group in GROUPS:
        config = load_json(_branch(target, group) / "config.json")
        rows = base.load_rows(config, "discovery")
        selected = load_json(target / group / "selection.json")["sample_ids"]
        by_id = {row["sample_id"]: row for row in rows}
        examples = [by_id[sample_id] for sample_id in selected]
        load_dotenv(config.get("env_file"), override=False)
        rubrics = generated.generate_r0_pair(
            rows, seed=int(split["seed"]),
            manager_config=dict(config["manager"],
                                env_file=config.get("env_file", ".env")),
            output_dir=target / group, attempt_limit=manager_attempts,
            expected_count=TRAIN_COUNTS[group], warmup_examples=examples,
            variants=("gn",), protocol=PROTOCOL,
        )
        print(f"{group}: generated {len(rubrics['gn'].root_ids)} roots", flush=True)


def evolve(target: Path, *, attempts: int, manager_attempts: int) -> None:
    infeasible = []
    for group in GROUPS:
        branch = _branch(target, group)
        config = load_json(branch / "config.json")
        rows = base.load_rows(config, "discovery")
        r0 = StructuredRubric.load_json(branch / "r0/rubric.json")
        manager = local.make_manager(
            dict(config["manager"], env_file=config.get("env_file", ".env")),
            manager_attempts, n_roots=len(r0.root_ids))
        load_dotenv(config.get("env_file"), override=False)
        try:
            local.run(config, branch, attempts=attempts, manager=manager,
                      rows=rows, r0=r0)
        except RuntimeError as exc:
            if "insufficient supported patterns for two initial clusters; S0 not created" not in str(exc):
                raise
            category._freeze_json(branch / "init_feasibility.json", dict(
                status="infeasible", reason=str(exc),
                r0_rubric_sha256=r0.rubric_sha256))
            infeasible.append(f"{group}: {exc}")
            print(f"{group}: initial Split infeasible ({exc})", flush=True)
            continue
        state = load_json(branch / "state.json")
        if not state["completed"]:
            raise RuntimeError(f"{group}: evolution did not complete")
        category._freeze_json(branch / "init_feasibility.json", dict(
            status="completed", r0_rubric_sha256=r0.rubric_sha256))
        print(f"{group}: roots={len(r0.root_ids)} "
              f"epochs={state['epoch']} completed", flush=True)
    if infeasible:
        raise RuntimeError("; ".join(infeasible))


def external(target: Path, records: tuple[dict, ...], split: Mapping[str, Any],
             attempts: int) -> None:
    for group in GROUPS:
        if not load_json(_branch(target, group) / "state.json")["completed"]:
            raise RuntimeError(f"{group}: freeze Final before external evaluation")
    schedule = vlrb._order_schedule(records)
    all_rows = system.support.vlrb_rows(records)
    for group in GROUPS:
        branch = _branch(target, group)
        config = load_json(branch / "config.json")
        final = StructuredRubric.load_json(branch / "final.json")
        base.evaluate(config, branch, "vlrb/final", all_rows,
                      final, orders=schedule, attempts=attempts)
    for group in GROUPS:
        branch = _branch(target, group)
        config = load_json(branch / "config.json")
        ids = set(split["heldout_ids"][group])
        heldout = tuple(row for row in records if row["sample_id"] in ids)
        rows = system.support.vlrb_rows(heldout)
        for stage, rubric_path in (("r0", "r0/rubric.json"),
                                   ("s0", "init/rubric.json")):
            rubric = StructuredRubric.load_json(branch / rubric_path)
            base.evaluate(config, branch, f"vlrb_heldout/{stage}", rows,
                          rubric, orders=schedule, attempts=attempts)


def _metrics(records: tuple[dict, ...], votes: list[int | None],
             scopes: Mapping[str, Any]) -> dict:
    result = {}
    for scope in ("full", "heldout"):
        by_group = {group: transfer._subset(
            records, votes, scopes[f"{scope}_group"][group])
            for group in category.GROUPS}
        result[scope] = dict(
            overall=transfer._subset(records, votes, scopes[scope]),
            by_group=by_group,
            macro_strict_acc=sum(item["strict_acc"] for item in by_group.values()) / 3,
        )
    return result


def _training(branch: Path) -> dict:
    state = load_json(branch / "state.json")
    result = {}
    for stage, path in (("r0", "r0/system.json"),
                        ("s0", "init/system.json"),
                        ("final", f"{state['baseline']}.json")):
        metrics = system.load(branch / path)["metrics"]
        result[stage] = {key: metrics[key] for key in (
            "sample_count", "strict_accuracy", "coverage",
            "covered_accuracy", "technical_failure_count")}
    return result


def report(source: Path, hallucination: Path, target: Path,
           records: tuple[dict, ...], split: Mapping[str, Any]) -> None:
    schedule = vlrb._order_schedule(records)
    scopes = category._scopes(records, split)
    paths = {group: _branch(target, group) for group in GROUPS}
    paths["hallucination"] = hallucination / "gn"
    votes = {}
    result: dict[str, Any] = dict(
        protocol=PROTOCOL, seed=split["seed"],
        split=str(source / "split.json"),
        hall_reuse=str(paths["hallucination"] / "vlrb/final.json"),
        rubrics={}, training={}, matrix={"full": {}, "heldout": {}},
        systems={}, routed={}, stage_heldout={}, source_heldout={}, paired={},
        old_category_controls={}, historical_generated_five=None,
    )
    for group in category.GROUPS:
        branch = paths[group]
        rubric = StructuredRubric.load_json(branch / "final.json")
        _, votes[group] = previous._load_predictions(
            branch / "vlrb/final.json", rubric, records, schedule)
        r0 = StructuredRubric.load_json(branch / "r0/rubric.json")
        result["rubrics"][group] = dict(
            root_count=len(r0.root_ids), final_sha256=rubric.rubric_sha256,
            roots=[dict(root_id=root, name=r0.get_node(root).criterion.name,
                        description=r0.get_node(root).criterion.description)
                   for root in r0.root_ids],
            source="reused" if group == "hallucination" else "new",
        )
        result["systems"][group] = _metrics(records, votes[group], scopes)
        for scope in ("full", "heldout"):
            result["matrix"][scope][group] = result["systems"][group][scope]["by_group"]
        if group in GROUPS:
            result["training"][group] = _training(branch)
    routed_votes = [votes[str(row["group"])][i]
                    for i, row in enumerate(records)]
    result["routed"] = _metrics(records, routed_votes, scopes)
    old_votes = {}
    for group in GROUPS:
        branch = paths[group]
        ids = scopes["heldout_group"][group]
        heldout = tuple(row for row in records if row["sample_id"] in ids)
        stage_votes = {}
        for stage, rubric_path in (("r0", "r0/rubric.json"),
                                   ("s0", "init/rubric.json")):
            rubric = StructuredRubric.load_json(branch / rubric_path)
            _, stage_votes[stage] = previous._load_predictions(
                branch / f"vlrb_heldout/{stage}.json", rubric, heldout, schedule)
        stage_votes["final"] = [prediction for row, prediction in
                                zip(records, votes[group]) if row["sample_id"] in ids]
        result["stage_heldout"][group] = {
            stage: transfer._subset(heldout, prediction, ids)
            for stage, prediction in stage_votes.items()}
        source_ids = {}
        for row in heldout:
            origin = (vlrb._official_dataset(row["benchmark_id"])
                      if group == "general" else _origin(row, group))
            source_ids.setdefault(origin, set()).add(row["sample_id"])
        result["source_heldout"][group] = {
            origin: {stage: transfer._subset(heldout, prediction, subset)
                     for stage, prediction in stage_votes.items()}
            for origin, subset in source_ids.items()}
        for before, after in (("r0", "s0"), ("s0", "final"),
                              ("r0", "final")):
            result["paired"][f"{group}_{before}_to_{after}"] = category._paired_summary(
                heldout, stage_votes[before], stage_votes[after], ids)
        old_branch = source / group / "gn"
        old_config = load_json(old_branch / "config.json")
        if old_config["worker"] != load_json(branch / "config.json")["worker"]:
            raise ValueError(f"{group}: old and new Worker configurations differ")
        old_rubric = StructuredRubric.load_json(old_branch / "final.json")
        _, old_votes[group] = previous._load_predictions(
            old_branch / "vlrb/final.json", old_rubric, records, schedule)
        result["old_category_controls"][group] = dict(
            source=str(old_branch / "vlrb/final.json"),
            same_category=transfer._subset(records, old_votes[group], ids),
        )
        result["paired"][f"old_vs_new_{group}"] = category._paired_summary(
            records, old_votes[group], votes[group], ids)
    for group in category.GROUPS:
        result["paired"][f"routed_vs_{group}"] = category._paired_summary(
            records, votes[group], routed_votes, scopes["heldout"])
        for other in category.GROUPS:
            if group != other:
                result["paired"][f"{group}_on_{group}_vs_{other}"] = category._paired_summary(
                    records, votes[other], votes[group], scopes["heldout_group"][group])
    hybrid = [old_votes.get(str(row["group"]), votes["hallucination"])[i]
              for i, row in enumerate(records)]
    result["old_category_controls"]["hybrid_route"] = _metrics(records, hybrid, scopes)
    result["paired"]["hybrid_route_vs_new_route"] = category._paired_summary(
        records, hybrid, routed_votes, scopes["heldout"])
    hall_report = load_json(hallucination / "report.json")
    result["historical_generated_five"] = hall_report["metrics"][
        "historical_generated_5"]["heldout_998"]
    write(target / "report.json", result)
    heldout = result["routed"]["heldout"]["overall"]
    print(f"routed heldout: {heldout['correct']}/{heldout['total']} "
          f"({heldout['strict_acc']:.2%})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "evolve",
                                          "vlrb", "report", "all"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--hallucination-root", type=Path, default=HALLUCINATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt-limit", type=int, default=10)
    parser.add_argument("--manager-attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if min(args.attempt_limit, args.manager_attempt_limit) < 1:
        parser.error("attempt limits must be positive")
    source = (args.source_root / f"seed{args.seed}").resolve()
    hallucination = (args.hallucination_root / f"seed{args.seed}").resolve()
    target = (args.output_root / f"seed{args.seed}").resolve()
    split, records = prepare(source, hallucination, target, args.seed)
    stages = (("generate", "evolve", "vlrb", "report")
              if args.stage == "all" else (args.stage,))
    for stage in stages:
        if stage == "generate":
            generate(target, split, args.manager_attempt_limit)
        elif stage == "evolve":
            evolve(target, attempts=args.attempt_limit,
                   manager_attempts=args.manager_attempt_limit)
        elif stage == "vlrb":
            external(target, records, split, args.attempt_limit)
        elif stage == "report":
            report(source, hallucination, target, records, split)


if __name__ == "__main__":
    main()

"""Evolve one generated-root Rubric per VLRB category and cross-evaluate them.

The three Final Rubrics each run once on all 1,247 VLRB pairs. The full and
held-out 3x3 matrices are read-only slices of those three saved predictions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
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
from . import vlrb_hallucination_transfer as transfer
from . import vlrb_hallucination_generated_roots as previous
from .experiment_utils import atomic_write_json as write, load_json


PROTOCOL = "vlrb-category-specialized-rubrics-v1"
GROUPS = ("general", "hallucination", "reasoning")
TRAIN_COUNTS = {"general": 36, "hallucination": 150, "reasoning": 63}
FULL_COUNTS = {"general": 181, "hallucination": 749, "reasoning": 317}
DEFAULT_SOURCE = transfer.DEFAULT_OUTPUT
DEFAULT_OUTPUT = base.ROOT / "output/vlrb_category_specialized_rubrics"
DEFAULT_HISTORY = base.ROOT / "output/vlrb_hallucination100_generated_roots"


def _freeze_json(path: Path, value: Any) -> None:
    if path.is_file() and load_json(path) != value:
        raise ValueError(f"frozen artifact changed: {path}")
    write(path, value)


def _freeze_text(path: Path, value: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"frozen artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _branch(target: Path, group: str) -> Path:
    return target / group / "gn"


def _records(config: Mapping[str, Any], target: Path) -> tuple[dict, ...]:
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    records = vlrb._read_records(target / "dataset", parquet_path=parquet)
    if len(records) != 1247:
        raise RuntimeError("VLRB sample count changed")
    return records


def _split(records: Sequence[Mapping[str, Any]], old_train: Sequence[str],
           seed: int, parquet_sha256: str, old_split_sha256: str) -> dict:
    by_group = {group: sorted(str(row["sample_id"]) for row in records
                              if row["group"] == group) for group in GROUPS}
    if {group: len(ids) for group, ids in by_group.items()} != FULL_COUNTS:
        raise RuntimeError("VLRB category counts changed")
    old = set(old_train)
    if len(old_train) != 100 or len(old) != 100 or not old <= set(by_group["hallucination"]):
        raise RuntimeError("old Hallucination100 is not 100 distinct VLRB hallucination rows")
    train, heldout = {}, {}
    for group in GROUPS:
        candidates = [sid for sid in by_group[group] if sid not in old]
        quota = TRAIN_COUNTS[group] - (len(old) if group == "hallucination" else 0)
        chosen = set(random.Random(f"{seed}:{group}").sample(candidates, quota))
        if group == "hallucination":
            chosen |= old
        train[group] = [sid for sid in by_group[group] if sid in chosen]
        heldout[group] = [sid for sid in by_group[group] if sid not in chosen]
    if ([len(train[group]) for group in GROUPS] != [36, 150, 63]
            or sum(map(len, heldout.values())) != 998):
        raise RuntimeError("20/80 category split has unexpected counts")
    return dict(protocol=PROTOCOL, seed=seed, split_level="sample_id",
                near_duplicate_filter=False, parquet_sha256=parquet_sha256,
                old_hallucination100_split_sha256=old_split_sha256,
                train_ids=train, heldout_ids=heldout)


def _training_rows(records: Sequence[Mapping[str, Any]], ids: Sequence[str],
                   schedule: Mapping[str, Sequence[int]]) -> list[dict]:
    by_id = {str(row["sample_id"]): row for row in records}
    result = []
    for sid in ids:
        record = by_id[sid]
        order = int(schedule[sid][0])
        result.append(dict(
            sample_id=sid, image_path=record["image_path"],
            image_sha256=record["image_sha256"], question=record["question"],
            A=record["responses"][order], B=record["responses"][1 - order],
            answer=("A" if record["preferred_original_index"] == order else "B"),
            source=vlrb._official_dataset(record["benchmark_id"]),
            domain="visual",
        ))
    return result


def prepare(source_root: Path, target: Path, seed: int) -> tuple[dict, dict, tuple[dict, ...]]:
    source = source_root / "seed11"
    template = load_json(source / "run_config.json")
    old_split = load_json(source / "split.json")
    records = _records(template, target)
    parquet = Path(template["data_root"]) / template["datasets"]["vlrb"]
    split = _split(records, old_split["train_ids"], seed,
                   transfer.file_sha256(parquet),
                   transfer.file_sha256(source / "split.json"))
    _freeze_json(target / "split.json", split)
    schedule = vlrb._order_schedule(records)
    for group in GROUPS:
        branch = _branch(target, group)
        data_file = target / group / "discovery.jsonl"
        rows = _training_rows(records, split["train_ids"][group], schedule)
        _freeze_text(data_file, "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                         for row in rows))
        config = deepcopy(template)
        config["datasets"]["discovery"] = str(data_file.resolve())
        config["discovery_sample_count"] = len(rows)
        _freeze_json(branch / "config.json", config)
        _freeze_json(branch / "split.json", dict(
            protocol=PROTOCOL, group=group, seed=seed,
            train_ids=split["train_ids"][group],
            heldout_ids=split["heldout_ids"][group],
            parquet_sha256=split["parquet_sha256"],
        ))
        local.check(config, branch)
    print("prepared VLRB category split: train=36/150/63 heldout=145/599/254",
          flush=True)
    return template, split, records


def _config(target: Path, group: str) -> dict:
    return load_json(_branch(target, group) / "config.json")


def generate(target: Path, split: Mapping[str, Any], manager_attempts: int) -> None:
    for group in GROUPS:
        branch = _branch(target, group)
        config = _config(target, group)
        load_dotenv(config.get("env_file"), override=False)
        rows = base.load_rows(config, "discovery")
        rubrics = generated.generate_r0_pair(
            rows, seed=int(split["seed"]),
            manager_config=dict(config["manager"],
                                env_file=config.get("env_file", ".env")),
            output_dir=target / group, attempt_limit=manager_attempts,
            expected_count=len(split["train_ids"][group]), variants=("gn",),
            protocol=PROTOCOL,
        )
        print(f"{group}: generated {len(rubrics['gn'].root_ids)} roots", flush=True)


def evolve(target: Path, *, attempts: int, manager_attempts: int) -> None:
    infeasible = []
    for group in GROUPS:
        branch = _branch(target, group)
        config = _config(target, group)
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
            _freeze_json(branch / "init_feasibility.json", dict(
                status="infeasible", reason=str(exc),
                r0_rubric_sha256=r0.rubric_sha256))
            infeasible.append(f"{group}: {exc}")
            print(f"{group}: initial Split infeasible ({exc})", flush=True)
            continue
        state = load_json(branch / "state.json")
        if not state["completed"]:
            raise RuntimeError(f"{group}: evolution did not complete")
        _freeze_json(branch / "init_feasibility.json", dict(
            status="completed", r0_rubric_sha256=r0.rubric_sha256))
        print(f"{group}: roots={len(r0.root_ids)} epochs={state['epoch']} completed",
              flush=True)
    if infeasible:
        raise RuntimeError("; ".join(infeasible))


def external(target: Path, records: Sequence[Mapping[str, Any]],
             attempts: int) -> None:
    for group in GROUPS:
        state = load_json(_branch(target, group) / "state.json")
        if not state["completed"]:
            raise RuntimeError(f"{group}: freeze Final before reading VLRB results")
    rows = system.support.vlrb_rows(records)
    schedule = vlrb._order_schedule(records)
    for group in GROUPS:
        branch = _branch(target, group)
        rubric = StructuredRubric.load_json(branch / "final.json")
        base.evaluate(_config(target, group), branch, "vlrb/final", rows,
                      rubric, orders=schedule, attempts=attempts)


def _scopes(records: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> dict:
    full = {str(row["sample_id"]) for row in records}
    heldout = {sid for group in GROUPS for sid in split["heldout_ids"][group]}
    train = {sid for group in GROUPS for sid in split["train_ids"][group]}
    if len(full) != 1247 or len(heldout) != 998 or len(train) != 249 or full != heldout | train or heldout & train:
        raise RuntimeError("frozen 20/80 split differs from VLRB records")
    return dict(full=full, heldout=heldout, train=train,
                full_group={group: {str(row["sample_id"]) for row in records
                                   if row["group"] == group} for group in GROUPS},
                heldout_group={group: set(split["heldout_ids"][group])
                               for group in GROUPS})


def _paired_summary(records: tuple[dict, ...], before: list[int | None],
                    after: list[int | None], ids: set[str]) -> dict:
    paired = transfer._paired(records, before, after, ids)
    return dict(corrected=paired["corrected"], harmed=paired["harmed"],
                mcnemar_exact_two_sided_p=paired["mcnemar_exact_two_sided_p"],
                bootstrap=previous._paired_ci(records, before, after, ids))


def _historical(target: Path, history_root: Path, records: tuple[dict, ...],
                schedule: Mapping[str, Sequence[int]], scopes: Mapping[str, Any],
                worker: Mapping[str, Any], parquet_sha256: str) -> dict:
    seed = target.name
    sources = {
        "fixed_five": base.ROOT / "output/vlrb_hallucination100_fresh_init" / seed,
        "generated_five": history_root / seed / "g5",
        "generated_count_old_prompt": history_root / seed / "gn",
    }
    result = {}
    for name, source in sources.items():
        if name == "fixed_five":
            audit_path = history_root / seed / "f5/reuse_manifest.json"
            if not audit_path.is_file():
                result[name] = {"status": "omitted", "reason": "F5 reuse audit is absent"}
                continue
            audit = load_json(audit_path)
            if (audit["vlrb_parquet_sha256"] != parquet_sha256
                    or any(not (source / filename).is_file()
                           or transfer.file_sha256(source / filename) != digest
                           for filename, digest in audit["source_artifact_sha256"].items())):
                result[name] = {"status": "omitted", "reason": "F5 audited source changed"}
                continue
        split_path = source / "split.json"
        if (not split_path.is_file()
                or load_json(split_path).get("parquet_sha256") != parquet_sha256):
            result[name] = {"status": "omitted", "reason": "historical VLRB parquet differs"}
            continue
        config_path = source / "config.json"
        if not config_path.is_file() or load_json(config_path)["worker"] != worker:
            result[name] = {"status": "omitted", "reason": "Worker configuration differs or is absent"}
            continue
        stages = {}
        for stage, filename in (("s0", "initial.json"), ("final", "final.json")):
            path = source / "vlrb" / filename
            rubric_path = source / ("init/rubric.json" if stage == "s0" else "final.json")
            if not path.is_file() or not rubric_path.is_file():
                stages[stage] = {"status": "unavailable"}
                continue
            rubric = StructuredRubric.load_json(rubric_path)
            _, predictions = previous._load_predictions(path, rubric, records, schedule)
            stages[stage] = dict(
                status="reused", source=str(path),
                full=transfer._subset(records, predictions, scopes["full"]),
                heldout=transfer._subset(records, predictions, scopes["heldout"]),
                full_group={group: transfer._subset(records, predictions, scopes["full_group"][group])
                            for group in GROUPS},
                heldout_group={group: transfer._subset(records, predictions, scopes["heldout_group"][group])
                               for group in GROUPS},
            )
        result[name] = stages
    return result


def report(target: Path, records: tuple[dict, ...], split: Mapping[str, Any],
           history_root: Path) -> None:
    schedule = vlrb._order_schedule(records)
    scopes = _scopes(records, split)
    predictions: dict[str, list[int | None]] = {}
    result: dict[str, Any] = dict(
        protocol=PROTOCOL, seed=split["seed"], split=str(target / "split.json"),
        rubrics={}, training={}, matrix={"full": {}, "heldout": {}},
        systems={}, routed={}, paired={}, historical={}, costs={})
    for group in GROUPS:
        branch = _branch(target, group)
        r0 = StructuredRubric.load_json(branch / "r0/rubric.json")
        final = StructuredRubric.load_json(branch / "final.json")
        value, votes = previous._load_predictions(
            branch / "vlrb/final.json", final, records, schedule)
        predictions[group] = votes
        result["rubrics"][group] = dict(
            root_count=len(r0.root_ids), final_sha256=final.rubric_sha256,
            roots=[dict(root_id=root, name=r0.get_node(root).criterion.name,
                        description=r0.get_node(root).criterion.description)
                   for root in r0.root_ids])
        rows = base.load_rows(_config(target, group), "discovery")
        state = load_json(branch / "state.json")
        train_paths = dict(r0=branch / "r0/system.json",
                           s0=branch / "init/system.json",
                           final=branch / f"{state['baseline']}.json")
        stage_rubrics = dict(
            r0=r0,
            s0=StructuredRubric.load_json(branch / "init/rubric.json"),
            final=final,
        )
        result["training"][group] = {}
        for stage, path in train_paths.items():
            train_value = system.load(path)
            if ([sample["sample_id"] for sample in train_value["samples"]]
                    != [row["sample_id"] for row in rows]
                    or train_value["k"] != 1
                    or train_value["metrics"]["technical_failure_count"]):
                raise RuntimeError(f"{path}: training artifact differs")
            result["training"][group][stage] = dict(
                strict_acc=train_value["metrics"]["strict_accuracy"],
                coverage=train_value["metrics"]["coverage"],
                covered_acc=train_value["metrics"]["covered_accuracy"],
                roots=previous._training_root_metrics(
                    train_value, rows, stage_rubrics[stage]),
                total=len(rows), wall_seconds=train_value.get("wall_seconds"))
        result["systems"][group] = dict(
            full=transfer._subset(records, votes, scopes["full"]),
            heldout=transfer._subset(records, votes, scopes["heldout"]),
            external_wall_seconds=value.get("wall_seconds"))
        for scope in ("full", "heldout"):
            group_scope = scopes[f"{scope}_group"]
            result["matrix"][scope][group] = {
                category: transfer._subset(records, votes, group_scope[category])
                for category in GROUPS}
        result["costs"][group] = dict(
            external_logical_calls=1247 * 3 * (len(r0.root_ids) + 1),
            external_wall_seconds=value.get("wall_seconds"),
            manager=base.manager_cost([branch / "init", *sorted(branch.glob("e[0-9][0-9]"))]),
            generation=previous._generation_cost(target / group, "gn"),
            worker=previous._worker_cost(branch / "cache"))
    routed_votes = [predictions[str(row["group"])][i]
                    for i, row in enumerate(records)]
    for scope in ("full", "heldout"):
        result["routed"][scope] = dict(
            overall=transfer._subset(records, routed_votes, scopes[scope]),
            categories={group: transfer._subset(
                records, routed_votes, scopes[f"{scope}_group"][group])
                for group in GROUPS})
        result["routed"][scope]["macro_strict_acc"] = sum(
            item["strict_acc"] for item in result["routed"][scope]["categories"].values()) / 3
    for group in GROUPS:
        result["paired"][f"routed_vs_{group}"] = _paired_summary(
            records, predictions[group], routed_votes, scopes["heldout"])
        for other in GROUPS:
            if other == group:
                continue
            result["paired"][f"{group}_on_{group}_vs_{other}"] = _paired_summary(
                records, predictions[other], predictions[group],
                scopes["heldout_group"][group])
    result["historical"] = _historical(
        target, history_root, records, schedule, scopes,
        _config(target, GROUPS[0])["worker"], split["parquet_sha256"])
    write(target / "report.json", result)
    for scope in ("full", "heldout"):
        metric = result["routed"][scope]["overall"]
        print(f"routed {scope}: {metric['correct']}/{metric['total']} "
              f"({metric['strict_acc']:.2%})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "evolve", "vlrb",
                                          "report", "all"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt-limit", type=int, default=10)
    parser.add_argument("--manager-attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if min(args.attempt_limit, args.manager_attempt_limit) < 1:
        parser.error("attempt limits must be positive")
    target = (args.output_root / f"seed{args.seed}").resolve()
    source_root = args.source_root.resolve()
    history_root = args.history_root.resolve()
    _, split, records = prepare(source_root, target, args.seed)
    stages = (("generate", "evolve", "vlrb", "report")
              if args.stage == "all" else (args.stage,))
    for stage in stages:
        if stage == "generate":
            generate(target, split, args.manager_attempt_limit)
        elif stage == "evolve":
            evolve(target, attempts=args.attempt_limit,
                   manager_attempts=args.manager_attempt_limit)
        elif stage == "vlrb":
            external(target, records, args.attempt_limit)
        elif stage == "report":
            report(target, records, split, history_root)


if __name__ == "__main__":
    main()

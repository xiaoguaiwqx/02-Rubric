"""Run R0 generation, subtree evolution, and frozen external evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from structured_rubrics.structured.schema import StructuredRubric

from . import (generated_root_initialization, generated_roots_report, rubric_pipeline,
               subtree_local_reflection, vlrb_official)
from .model_call_support import file_sha256
from .experiment_utils import atomic_write_json, load_json
from .subtree_local_reflection_manager import make_manager


SPLIT_MANIFEST = (Path(__file__).resolve().parents[2] / "docs/experiments"
                  / "vlrb-hallucination100-generated-roots/seed11_split.json")


def prepare(config: dict, root: Path, seed: int) -> None:
    """Materialize the frozen Hallucination100 rows from their benchmark IDs."""
    split = load_json(SPLIT_MANIFEST)
    if seed != split["seed"]:
        raise ValueError(f"only frozen seed {split['seed']} is available")
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    if file_sha256(parquet) != split["parquet_sha256"]:
        raise ValueError("VL-RewardBench parquet differs from the frozen split")
    records = vlrb_official._read_records(root / "dataset", parquet_path=parquet)
    by_id = {item["sample_id"]: item for item in records}
    train_ids = split["train_ids"]
    heldout_ids = split["heldout_ids"]
    if (len(train_ids) != 100 or len(set(train_ids)) != 100
            or len(heldout_ids) != 648 or set(train_ids) & set(heldout_ids)
            or set(train_ids + heldout_ids) - set(by_id)):
        raise ValueError("frozen Hallucination100 ID split differs from benchmark")
    schedule = vlrb_official._order_schedule(records)
    rows = []
    for sample_id in train_ids:
        item = by_id[sample_id]
        if item["group"] != "hallucination":
            raise ValueError(f"{sample_id}: frozen train ID is not hallucination")
        order = schedule[sample_id][0]
        rows.append(dict(
            sample_id=sample_id, image_path=item["image_path"],
            image_sha256=item["image_sha256"], question=item["question"],
            A=item["responses"][order], B=item["responses"][1 - order],
            answer="A" if item["preferred_original_index"] == order else "B",
            source=vlrb_official._official_dataset(item["benchmark_id"]),
            domain="visual",
        ))
    discovery = root / "discovery_100.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if discovery.exists() and discovery.read_text(encoding="utf-8") != payload:
        raise ValueError(f"frozen discovery rows changed at {discovery}")
    discovery.parent.mkdir(parents=True, exist_ok=True)
    discovery.write_text(payload, encoding="utf-8")
    prepared = dict(config)
    prepared["datasets"] = dict(config["datasets"], discovery=str(discovery))
    target = root / "config.json"
    if target.exists() and load_json(target) != prepared:
        raise ValueError(f"prepared configuration changed at {target}")
    atomic_write_json(target, prepared)


def run_stage(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    load_dotenv(config.get("env_file"), override=False)
    root = args.output_root.resolve()
    target = root / args.variant

    if args.stage == "prepare":
        prepare(config, root, args.seed)
        print(f"prepare complete: {root / 'config.json'}", flush=True)
        return

    if args.stage == "roots":
        rows = rubric_pipeline.load_rows(config, "discovery")
        if args.variant == "f5":
            rubric = rubric_pipeline.build_multicrit_open_ended_init_rubric()
            path = target / "r0/rubric.json"
            if path.exists() and StructuredRubric.load_json(path).rubric_sha256 != rubric.rubric_sha256:
                raise ValueError(f"frozen R0 changed at {path}")
            atomic_write_json(path, rubric.to_dict())
            print(f"f5 root ready: {path}", flush=True)
        else:
            generated_root_initialization.generate_r0_pair(
                rows,
                seed=args.seed,
                manager_config=dict(config["manager"], env_file=config.get("env_file", ".env")),
                output_dir=root,
                attempt_limit=args.root_attempt_limit,
                expected_count=config.get("discovery_sample_count", 100),
                variants=(args.variant,),
                count_instructions=generated_root_initialization.LEGACY_COUNT_INSTRUCTIONS,
            )
            print(f"{args.variant} root ready: {target / 'r0/rubric.json'}", flush=True)
        return

    subtree_local_reflection.check(config, target)
    if args.stage == "evolve":
        r0 = StructuredRubric.load_json(target / "r0/rubric.json")
        manager = make_manager(
            dict(config["manager"], env_file=config.get("env_file", ".env")),
            args.attempt_limit,
            n_roots=len(r0.root_ids),
        )
        subtree_local_reflection.run(
            config, target, attempts=args.attempt_limit, manager=manager, r0=r0,
        )
    elif args.stage == "vlrb-r0":
        rubric = StructuredRubric.load_json(target / "r0/rubric.json")
        parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
        records = vlrb_official._read_records(target / "vlrb", parquet_path=parquet)
        rows = rubric_pipeline.system.support.vlrb_rows(records)
        value = rubric_pipeline.evaluate(
            config, target, "vlrb/r0", rows, rubric,
            orders=vlrb_official._order_schedule(records), attempts=args.attempt_limit,
            vlrb_records=records,
        )
        atomic_write_json(target / "vlrb/r0_report.json", dict(
            k=value["k"], rubric_sha256=rubric.rubric_sha256,
            runtime=value["metrics"],
            official=vlrb_official.official_system_metrics(records, vlrb_official._votes(value)),
        ))
    elif args.stage in {"dev", "vlrb"}:
        rubric_pipeline.external(config, target, args.stage, args.attempt_limit)
    elif args.stage == "report":
        subtree_local_reflection.report(config, target)
        generated_roots_report.report(config, root, args.variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "roots", "evolve", "vlrb-r0",
                                          "dev", "vlrb", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("f5", "g5", "gn"), default="g5")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--attempt-limit", type=int, default=10)
    parser.add_argument("--root-attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if args.attempt_limit < 1 or args.root_attempt_limit < 1:
        parser.error("attempt limits must be positive")
    run_stage(args)


if __name__ == "__main__":
    main()

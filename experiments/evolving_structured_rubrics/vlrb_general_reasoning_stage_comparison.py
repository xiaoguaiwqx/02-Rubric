"""Compare selected General/Reasoning stages with their Final rubrics on VLRB."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from critiq.structured.schema import StructuredRubric

from . import aligned_system_runtime as system
from . import framework_v6 as base
from . import vl_rewardbench as vlrb
from . import vlrb_category_specialized_rubrics as category
from . import vlrb_general_reasoning_warmup10 as experiment
from . import vlrb_hallucination_generated_roots as previous
from . import vlrb_hallucination_transfer as transfer
from .experiment_utils import atomic_write_json as write, load_json


STAGES = {"general": "e03", "reasoning": "r0"}


def run(seed: int, output_root: Path, attempts: int) -> None:
    target = (output_root / f"seed{seed}").resolve()
    source = experiment.SOURCE_ROOT / f"seed{seed}"
    if not (target / "report.json").is_file():
        raise RuntimeError("finish the main experiment and report before this comparison")
    split = load_json(source / "split.json")
    if split["seed"] != seed:
        raise ValueError("frozen split seed differs")
    config = load_json(target / "general/gn/config.json")
    records = category._records(config, source)
    scopes = category._scopes(records, split)
    schedule = vlrb._order_schedule(records)
    rows = system.support.vlrb_rows(records)
    result = dict(
        protocol="vlrb-general-reasoning-stage-comparison-v1",
        seed=seed,
        selection="General e03 is the training-set peak; Reasoning r0 is the unsplit-root control",
        main_report=str(target / "report.json"),
        comparisons={},
    )
    for group, stage in STAGES.items():
        branch = target / group / "gn"
        group_config = load_json(branch / "config.json")
        load_dotenv(group_config.get("env_file"), override=False)
        rubric = StructuredRubric.load_json(branch / stage / "rubric.json")
        final = StructuredRubric.load_json(branch / "final.json")
        base.evaluate(group_config, branch, f"vlrb/{stage}", rows, rubric,
                      orders=schedule, attempts=attempts)
        _, stage_votes = previous._load_predictions(
            branch / f"vlrb/{stage}.json", rubric, records, schedule)
        _, final_votes = previous._load_predictions(
            branch / "vlrb/final.json", final, records, schedule)
        scopes_to_compare = {
            "full_1247": scopes["full"],
            "heldout_998": scopes["heldout"],
            "own_full": scopes["full_group"][group],
            "own_heldout": scopes["heldout_group"][group],
        }
        comparison = dict(
            stage=stage,
            stage_rubric_sha256=rubric.rubric_sha256,
            final_rubric_sha256=final.rubric_sha256,
            stage_predictions=str(branch / f"vlrb/{stage}.json"),
            final_predictions=str(branch / "vlrb/final.json"),
            scopes={},
            by_group={"full": {}, "heldout": {}},
        )
        for name, ids in scopes_to_compare.items():
            before = transfer._subset(records, stage_votes, ids)
            after = transfer._subset(records, final_votes, ids)
            comparison["scopes"][name] = dict(
                stage=before, final=after,
                delta_correct=after["correct"] - before["correct"],
                delta_pp=100 * (after["strict_acc"] - before["strict_acc"]),
                paired=category._paired_summary(records, stage_votes, final_votes, ids),
            )
        for scope in ("full", "heldout"):
            for evaluated_group in category.GROUPS:
                ids = scopes[f"{scope}_group"][evaluated_group]
                before = transfer._subset(records, stage_votes, ids)
                after = transfer._subset(records, final_votes, ids)
                comparison["by_group"][scope][evaluated_group] = dict(
                    stage=before, final=after,
                    delta_correct=after["correct"] - before["correct"],
                )
        result["comparisons"][group] = comparison
        own = comparison["scopes"]["own_heldout"]
        print(f"{group} {stage} vs Final on own heldout: "
              f"{own['stage']['correct']}/{own['stage']['total']} -> "
              f"{own['final']['correct']}/{own['final']['total']}", flush=True)
    write(target / "stage_comparison.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-root", type=Path, default=experiment.DEFAULT_OUTPUT)
    parser.add_argument("--attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if args.attempt_limit < 1:
        parser.error("attempt limit must be positive")
    run(args.seed, args.output_root, args.attempt_limit)


if __name__ == "__main__":
    main()

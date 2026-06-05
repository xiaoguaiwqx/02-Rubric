"""Run final-eval ablations for the RLHF-V exp3 checkpoint.

真实运行方式：

1. 先检查本次 ablation 会使用哪些 criterion，不调用模型：
   python ablate_rlhfv_final_eval.py --dry-run

2. 正式运行 final ablation。第一次会对 best11 在 heldout500 上重新推理一次，
   并保存 raw prediction cache；随后 top3/top5/top7 等分组都会从同一份缓存离线重算：
   python ablate_rlhfv_final_eval.py

3. 如果已经有缓存，只想重新生成 ablation JSON/Markdown 报告，不调用模型：
   python ablate_rlhfv_final_eval.py --cache-policy cache-only

4. 如果想强制刷新 best11 raw prediction cache，再重新生成所有 ablation 结果：
   python ablate_rlhfv_final_eval.py --cache-policy run

默认输入：
- output/rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5/epoch_final.json
- data/RLHF-V/heldout_validation_500_pair.jsonl

默认输出：
- output/rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5/final_ablation_predictions.json
- output/rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5/final_ablation_results.json
- output/rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5/final_ablation_report.md

注意：正式运行模型推理时请使用运行 demo_rlhfv.py 的同一个 Python 环境。
项目要求 Python >= 3.10，并且需要可用的 worker VLM 服务/API 配置。

The script evaluates the best11 criteria once on heldout500, caches the raw
A/B/U votes, and then recomputes several criterion subsets offline from the
same prediction matrix. This keeps the ablation comparison free from extra
model-sampling noise across subsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_EXP_DIR = Path(
    "output/rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5"
)
HELDOUT_PAIR_DATA_PATH = Path("data/RLHF-V/heldout_validation_500_pair.jsonl")
BEST_THRESHOLD = 0.6
BROAD_ONLY = [
    "visual_grounding",
    "factual_consistency",
    "specificity",
    "completeness",
    "contextual_sensitivity",
    "adaptation_to_visual_quality",
    "interpretive_accuracy_under_ambiguous_context",
]
DROP_SENSITIVITY = "sensitivity_to_implied_meaning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final heldout500 ablation for RLHF-V exp3 criteria."
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=DEFAULT_EXP_DIR,
        help="Directory containing epoch_final.json.",
    )
    parser.add_argument(
        "--heldout-path",
        type=Path,
        default=HELDOUT_PAIR_DATA_PATH,
        help="Heldout pair JSONL file.",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("reuse-or-run", "run", "cache-only"),
        default="reuse-or-run",
        help=(
            "reuse-or-run reuses a valid raw prediction cache or runs best11; "
            "run always refreshes it; cache-only never calls the model."
        ),
    )
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=None,
        help="Path for cached best11 raw predictions.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for the ablation JSON result.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path for the Markdown ablation report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned criterion groups and exit without model calls.",
    )
    return parser.parse_args()


def default_output_paths(args: argparse.Namespace) -> None:
    if args.prediction_cache is None:
        args.prediction_cache = args.exp_dir / "final_ablation_predictions.json"
    if args.output_json is None:
        args.output_json = args.exp_dir / "final_ablation_results.json"
    if args.output_md is None:
        args.output_md = args.exp_dir / "final_ablation_report.md"


def load_checkpoint(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "epoch_final.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_rlhfv_pair_data(pair_path: Path) -> list[dict[str, Any]]:
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing heldout pair data: {pair_path}")

    pairs = load_jsonl(pair_path)
    if not pairs:
        raise ValueError(f"Empty heldout pair data: {pair_path}")

    dataset = []
    for index, pair in enumerate(pairs):
        answer = pair.get("answer")
        if answer not in ("A", "B"):
            raise ValueError(f"Invalid answer at row {index + 1}: {answer}")
        if not isinstance(pair.get("A"), str) or not isinstance(pair.get("B"), str):
            raise ValueError(f"A/B must be strings at row {index + 1}")
        if not isinstance(pair.get("question"), str) or not pair["question"].strip():
            raise ValueError(f"Missing question at row {index + 1}")
        if not isinstance(pair.get("image_path"), str) or not pair["image_path"].strip():
            raise ValueError(f"Missing image_path at row {index + 1}")

        item = {
            "sample_id": pair.get("sample_id"),
            "image_path": pair["image_path"],
            "question": pair["question"],
            "A": pair["A"],
            "B": pair["B"],
            "answer": answer,
        }
        dataset.append({key: value for key, value in item.items() if value is not None})

    return dataset


def load_all_criteria(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    return checkpoint["all_criteria"]


def best11_criteria(all_criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    criteria = [
        criterion
        for criterion in all_criteria
        if float(criterion.get("score", 0.0)) >= BEST_THRESHOLD
    ]
    if not criteria:
        raise ValueError(f"No criteria found with score >= {BEST_THRESHOLD}")
    return criteria


def build_groups(
    all_criteria: list[dict[str, Any]],
    best_criteria: list[dict[str, Any]],
) -> dict[str, list[str]]:
    ranked = sorted(
        all_criteria,
        key=lambda criterion: float(criterion.get("score", 0.0)),
        reverse=True,
    )
    best_names = [criterion["name"] for criterion in best_criteria]
    groups = {
        "best11": best_names,
        "top3_by_train": [criterion["name"] for criterion in ranked[:3]],
        "top5_by_train": [criterion["name"] for criterion in ranked[:5]],
        "top7_by_train": [criterion["name"] for criterion in ranked[:7]],
        "no_sensitivity_to_implied_meaning": [
            name for name in best_names if name != DROP_SENSITIVITY
        ],
        "broad_only": BROAD_ONLY[:],
    }
    validate_groups(groups, best_names)
    return groups


def validate_groups(groups: dict[str, list[str]], best_names: list[str]) -> None:
    best_set = set(best_names)
    for group_name, names in groups.items():
        missing = [name for name in names if name not in best_set]
        if missing:
            raise ValueError(
                f"Group {group_name!r} contains criteria outside best11: {missing}"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"Group {group_name!r} contains duplicate criteria")


def sample_identity(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "sample_id": item.get("sample_id"),
            "gold": item["answer"],
        }
        for index, item in enumerate(dataset)
    ]


def cache_is_valid(
    cache: dict[str, Any],
    dataset: list[dict[str, Any]],
    best_names: list[str],
) -> tuple[bool, str]:
    samples = sample_identity(dataset)
    cached_samples = cache.get("samples")
    if cached_samples != samples:
        return False, "cached sample ids/gold labels do not match heldout data"

    cached_names = [criterion["name"] for criterion in cache.get("criteria", [])]
    if set(cached_names) != set(best_names):
        return False, "cached criterion names do not match checkpoint best11"

    prediction = cache.get("prediction")
    if not isinstance(prediction, list) or len(prediction) != len(dataset):
        return False, "cached prediction length does not match heldout data"

    missing_rows = [
        index
        for index, row in enumerate(prediction)
        if not isinstance(row, dict) or any(name not in row for name in best_names)
    ]
    if missing_rows:
        return False, f"cached prediction rows miss best11 criteria: {missing_rows[:5]}"

    return True, "ok"


def load_valid_cache(
    cache_path: Path,
    dataset: list[dict[str, Any]],
    best_names: list[str],
) -> tuple[dict[str, Any] | None, str]:
    if not cache_path.exists():
        return None, "cache file does not exist"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    is_valid, reason = cache_is_valid(cache, dataset, best_names)
    if not is_valid:
        return None, reason
    return cache, "ok"


def run_best11_eval(
    dataset: list[dict[str, Any]],
    best_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "Running model evaluation requires Python >= 3.10 because the "
            "project package uses PEP 604 type syntax. Use the same environment "
            "that runs demo_rlhfv.py, or run with --cache-policy cache-only "
            "after a raw prediction cache exists."
        )

    from critiq import Criterion, MultiModalPairEvaluator
    from demo_rlhfv import (
        MAX_CONCURRENT,
        MAX_RETRIES,
        WORKER_ARGS,
        WORKER_PROMPT,
    )

    criteria = [Criterion.from_dict(criterion) for criterion in best_criteria]
    print(
        "Running best11 heldout evaluation. "
        f"samples={len(dataset)}, criteria={len(best_criteria)}"
    )
    evaluator = MultiModalPairEvaluator(
        WORKER_ARGS,
        dataset=dataset,
        max_concurrent=MAX_CONCURRENT,
        max_retries=MAX_RETRIES,
        worker_prompt=WORKER_PROMPT,
    )
    eval_output = evaluator.eval(criteria, update_score=False)
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "worker": {
            "model": WORKER_ARGS.get("model"),
            "base_url": WORKER_ARGS.get("base_url"),
            "max_concurrent": MAX_CONCURRENT,
            "max_retries": MAX_RETRIES,
        },
        "criteria": best_criteria,
        "samples": sample_identity(dataset),
        "prediction": eval_output.prediction,
        "best11_accuracy": eval_output.accuracy,
        "best11_is_correct": eval_output.is_correct,
        "per_criterion_acc": eval_output.per_criterion_acc,
    }


def load_or_create_prediction_cache(
    args: argparse.Namespace,
    dataset: list[dict[str, Any]],
    best_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    best_names = [criterion["name"] for criterion in best_criteria]
    if args.cache_policy != "run":
        cache, reason = load_valid_cache(args.prediction_cache, dataset, best_names)
        if cache is not None:
            print(f"Using raw prediction cache: {args.prediction_cache}")
            return cache
        if args.cache_policy == "cache-only":
            raise FileNotFoundError(
                f"No valid prediction cache at {args.prediction_cache}: {reason}"
            )
        print(f"No valid cache found ({reason}); best11 evaluation will run.")

    cache = run_best11_eval(dataset, best_criteria)
    args.prediction_cache.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_cache.write_text(
        json.dumps(cache, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved raw prediction cache: {args.prediction_cache}")
    return cache


def criterion_vote(vote_counts: dict[str, int]) -> str | None:
    if vote_counts.get("U", 0) > 0:
        return None
    if vote_counts.get("A", 0) > vote_counts.get("B", 0):
        return "A"
    if vote_counts.get("B", 0) > vote_counts.get("A", 0):
        return "B"
    return None


def ensemble_vote(votes: dict[str, int]) -> str | None:
    if votes["A"] > votes["B"]:
        return "A"
    if votes["B"] > votes["A"]:
        return "B"
    return None


def summarize_per_criterion(
    dataset: list[dict[str, Any]],
    prediction: list[dict[str, dict[str, int]]],
    criterion_names: list[str],
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    total = len(dataset)
    for name in criterion_names:
        correct = 0
        applicable = 0
        refuse = 0
        for item, row in zip(dataset, prediction):
            vote = criterion_vote(row[name])
            if vote is None:
                refuse += 1
                continue
            applicable += 1
            if vote == item["answer"]:
                correct += 1
        stats[name] = {
            "total": total,
            "refuse": refuse,
            "applicable": applicable,
            "correct": correct,
            "incorrect": applicable - correct,
            "coverage": applicable / total if total else 0.0,
            "accuracy": correct / applicable if applicable else 0.0,
        }
    return stats


def evaluate_group(
    group_name: str,
    criterion_names: list[str],
    dataset: list[dict[str, Any]],
    prediction: list[dict[str, dict[str, int]]],
) -> dict[str, Any]:
    sample_results = []
    vote_distribution: dict[str, int] = {}
    correct_count = 0
    none_final_count = 0
    none_vote_total = 0

    for index, (item, row) in enumerate(zip(dataset, prediction)):
        votes = {"A": 0, "B": 0, "None": 0}
        for name in criterion_names:
            vote = criterion_vote(row[name])
            if vote is None:
                votes["None"] += 1
            else:
                votes[vote] += 1

        final_vote = ensemble_vote(votes)
        correct = final_vote is not None and final_vote == item["answer"]
        correct_count += int(correct)
        none_final_count += int(final_vote is None)
        none_vote_total += votes["None"]

        dist_key = (
            f"A={votes['A']}|B={votes['B']}|None={votes['None']}|"
            f"final={final_vote}|correct={correct}"
        )
        vote_distribution[dist_key] = vote_distribution.get(dist_key, 0) + 1
        sample_results.append(
            {
                "index": index,
                "sample_id": item.get("sample_id"),
                "gold": item["answer"],
                "final_vote": final_vote,
                "correct": correct,
                "votes": votes,
            }
        )

    total = len(dataset)
    return {
        "group": group_name,
        "criterion_names": criterion_names,
        "n_criteria": len(criterion_names),
        "accuracy": correct_count / total if total else 0.0,
        "correct": correct_count,
        "total": total,
        "none_final_count": none_final_count,
        "avg_none_votes": none_vote_total / total if total else 0.0,
        "vote_count_distribution": [
            {"pattern": pattern, "count": count}
            for pattern, count in sorted(
                vote_distribution.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "per_criterion_stats": summarize_per_criterion(
            dataset, prediction, criterion_names
        ),
        "sample_results": sample_results,
    }


def attach_deltas(results: dict[str, dict[str, Any]]) -> None:
    baseline = results["best11"]
    baseline_by_index = {
        item["index"]: item for item in baseline["sample_results"]
    }
    baseline_correct = baseline["correct"]
    baseline_accuracy = baseline["accuracy"]

    for group_name, result in results.items():
        gained = []
        lost = []
        for sample in result["sample_results"]:
            base = baseline_by_index[sample["index"]]
            if sample["correct"] and not base["correct"]:
                gained.append(compare_sample(sample, base))
            elif base["correct"] and not sample["correct"]:
                lost.append(compare_sample(sample, base))

        result["delta_vs_best11_accuracy"] = result["accuracy"] - baseline_accuracy
        result["delta_vs_best11_correct"] = result["correct"] - baseline_correct
        result["gained_over_best11"] = gained
        result["lost_vs_best11"] = lost


def compare_sample(sample: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": sample["index"],
        "sample_id": sample.get("sample_id"),
        "gold": sample["gold"],
        "best11_vote": baseline["final_vote"],
        "ablation_vote": sample["final_vote"],
        "best11_votes": baseline["votes"],
        "ablation_votes": sample["votes"],
    }


def load_reference(exp_dir: Path) -> dict[str, Any] | None:
    path = exp_dir / "final_heldout_diagnostics.json"
    if not path.exists():
        return None
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    samples = diagnostics.get("sample_vote_counts", [])
    if not samples:
        return None
    correct = sum(1 for sample in samples if sample.get("final_correct"))
    total = len(samples)
    return {
        "path": str(path),
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def build_report(
    args: argparse.Namespace,
    groups: dict[str, list[str]],
    results: dict[str, dict[str, Any]],
    reference: dict[str, Any] | None,
) -> str:
    best11 = results["best11"]
    lines = [
        "# Final Ablation Report",
        "",
        f"- Experiment directory: `{args.exp_dir}`",
        f"- Prediction cache: `{args.prediction_cache}`",
        f"- Heldout samples: {best11['total']}",
        f"- Current cache best11: {best11['accuracy']:.3f} = {best11['correct']}/{best11['total']}",
    ]
    if reference is not None:
        lines.append(
            "- Exp3 final reference: "
            f"{reference['accuracy']:.3f} = {reference['correct']}/{reference['total']} "
            f"from `{reference['path']}`"
        )
        if (
            reference["correct"] != best11["correct"]
            or reference["total"] != best11["total"]
        ):
            lines.append(
                "- Note: current cache best11 differs from the original exp3 final; "
                "treat it as a separate inference pass."
            )
    lines.extend(["", "## Criterion Groups", ""])
    lines.append("| group | n | criteria |")
    lines.append("| --- | ---: | --- |")
    for group_name, names in groups.items():
        lines.append(f"| {group_name} | {len(names)} | {', '.join(names)} |")

    lines.extend(["", "## Results", ""])
    lines.append(
        "| group | n | accuracy | correct/total | delta vs best11 | tie/None final | avg None votes |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group_name, result in sorted(
        results.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        lines.append(
            f"| {group_name} | {result['n_criteria']} | {result['accuracy']:.3f} | "
            f"{result['correct']}/{result['total']} | "
            f"{result['delta_vs_best11_accuracy']:+.3f} "
            f"({result['delta_vs_best11_correct']:+d}) | "
            f"{result['none_final_count']} | {result['avg_none_votes']:.2f} |"
        )

    lines.extend(["", "## Gain/Loss Vs Best11", ""])
    for group_name, result in results.items():
        if group_name == "best11":
            continue
        gained = result["gained_over_best11"]
        lost = result["lost_vs_best11"]
        lines.append(
            f"### {group_name}: gained {len(gained)}, lost {len(lost)}"
        )
        lines.append("")
        lines.append(f"- gained sample ids: {format_sample_ids(gained)}")
        lines.append(f"- lost sample ids: {format_sample_ids(lost)}")
        lines.append("")

    lines.extend(["## Vote Distributions", ""])
    for group_name, result in results.items():
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append("| pattern | count |")
        lines.append("| --- | ---: |")
        for item in result["vote_count_distribution"][:40]:
            lines.append(f"| `{item['pattern']}` | {item['count']} |")
        if len(result["vote_count_distribution"]) > 40:
            lines.append(
                f"| ... {len(result['vote_count_distribution']) - 40} more patterns in JSON | |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_sample_ids(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "none"
    return ", ".join(
        str(sample.get("sample_id") if sample.get("sample_id") is not None else sample["index"])
        for sample in samples
    )


def print_dry_run(
    args: argparse.Namespace,
    all_criteria: list[dict[str, Any]],
    best_criteria: list[dict[str, Any]],
    groups: dict[str, list[str]],
) -> None:
    print("Dry run only. No model calls will be made.")
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Prediction cache: {args.prediction_cache}")
    print(f"All criteria: {len(all_criteria)}")
    print(f"Best criteria (score >= {BEST_THRESHOLD}): {len(best_criteria)}")
    print()
    for group_name, names in groups.items():
        print(f"{group_name} ({len(names)}):")
        for name in names:
            score = next(
                float(criterion.get("score", 0.0))
                for criterion in all_criteria
                if criterion["name"] == name
            )
            print(f"  - {name}: {score:.6f}")
        print()


def main() -> None:
    args = parse_args()
    default_output_paths(args)

    checkpoint = load_checkpoint(args.exp_dir)
    all_criteria = load_all_criteria(checkpoint)
    best_criteria = best11_criteria(all_criteria)
    groups = build_groups(all_criteria, best_criteria)

    if args.dry_run:
        print_dry_run(args, all_criteria, best_criteria, groups)
        return

    dataset = load_rlhfv_pair_data(args.heldout_path)
    cache = load_or_create_prediction_cache(args, dataset, best_criteria)
    prediction = cache["prediction"]

    results = {
        group_name: evaluate_group(group_name, names, dataset, prediction)
        for group_name, names in groups.items()
    }
    attach_deltas(results)
    reference = load_reference(args.exp_dir)

    output = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(),
            "exp_dir": str(args.exp_dir),
            "heldout_path": str(args.heldout_path),
            "prediction_cache": str(args.prediction_cache),
            "cache_policy": args.cache_policy,
            "best_threshold": BEST_THRESHOLD,
        },
        "reference": reference,
        "groups": groups,
        "cache_best11_accuracy": cache.get("best11_accuracy"),
        "cache_per_criterion_acc": cache.get("per_criterion_acc"),
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        build_report(args, groups, results, reference),
        encoding="utf-8",
    )

    print(f"Saved ablation JSON: {args.output_json}")
    print(f"Saved ablation report: {args.output_md}")
    for group_name, result in sorted(
        results.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        print(
            f"{group_name}: {result['accuracy']:.3f} "
            f"= {result['correct']}/{result['total']} "
            f"delta={result['delta_vs_best11_accuracy']:+.3f}"
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

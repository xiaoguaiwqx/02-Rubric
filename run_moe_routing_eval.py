"""Evaluate MoE soft routing for RLHF-V using cached votes or real workers.

Typical usage:

1. Inspect the planned replay setup without model calls:
   python run_moe_routing_eval.py --dry-run

2. Replay cached best11 worker votes with a cached router output:
   python run_moe_routing_eval.py --router-cache-policy cache-only

3. Build the router cache, then replay MoE variants:
   python run_moe_routing_eval.py --router-cache-policy reuse-or-run

4. Tune on one heldout split and verify on the other split:
   python run_moe_routing_eval.py --mode split-replay --router-cache-policy cache-only

5. Run the real MoE flow without rerunning majority workers:
   python run_moe_routing_eval.py --mode e2e-moe --routing-threshold 0.5 --tie-epsilon 0.0 --fallback-weight 0.3

6. Run a small real end-to-end subset after replay looks promising:
   python run_moe_routing_eval.py --mode e2e-subset --dry-run
   python run_moe_routing_eval.py --mode e2e-subset
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from critiq import Criterion, MoEPairEvaluator, MultiModalPairEvaluator
from critiq.evaluator import (
    CriterionPerformance,
    load_criterion_performance,
    static_weight,
)
from critiq.router import RouterManager, RoutingDecision
from critiq.utils import USE_TQDM


DEFAULT_EXP_DIR = Path(
    "output/rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5"
)
DEFAULT_HELDOUT_PATH = Path("data/RLHF-V/heldout_validation_500_pair.jsonl")
DEFAULT_FALLBACK_CRITERIA = ("visual_grounding", "factual_consistency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE soft-routing replay and e2e subset eval for RLHF-V."
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "split-replay", "e2e-moe", "e2e-subset"),
        default="replay",
        help=(
            "replay uses cached worker votes; split-replay tunes on one split "
            "and validates on another; e2e-moe calls router + active workers; "
            "e2e-subset calls majority and MoE workers on a subset."
        ),
    )
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--heldout-path", type=Path, default=DEFAULT_HELDOUT_PATH)
    parser.add_argument("--prediction-cache", type=Path, default=None)
    parser.add_argument("--router-cache", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--criterion-stats",
        type=Path,
        default=None,
        help="Optional stats JSON. Defaults to exp-dir/epoch_final.json.",
    )
    parser.add_argument(
        "--router-cache-policy",
        choices=("reuse-or-run", "run", "cache-only"),
        default="reuse-or-run",
        help="Controls router cache creation. cache-only never calls the router.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=100745534)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--image-field", type=str, default=None)
    parser.add_argument("--question-field", type=str, default=None)
    parser.add_argument(
        "--no-encode-local-image",
        action="store_true",
        help="Pass image_path as URL instead of encoding local files.",
    )
    parser.add_argument("--router-temperature", type=float, default=0.0)
    parser.add_argument("--worker-temperature", type=float, default=0.0)
    parser.add_argument("--static-alpha", type=float, default=0.7)
    parser.add_argument("--static-beta", type=float, default=0.3)
    parser.add_argument("--routing-threshold", type=float, default=0.2)
    parser.add_argument("--tie-epsilon", type=float, default=0.05)
    parser.add_argument("--fallback-weight", type=float, default=0.5)
    parser.add_argument(
        "--fallback-criteria",
        type=str,
        default=",".join(DEFAULT_FALLBACK_CRITERIA),
        help="Comma-separated broad fallback criteria.",
    )
    parser.add_argument(
        "--sweep-routing-thresholds",
        type=str,
        default="0.0,0.1,0.2,0.3,0.5",
    )
    parser.add_argument(
        "--sweep-tie-epsilons",
        type=str,
        default="0.0,0.03,0.05,0.1",
    )
    parser.add_argument(
        "--sweep-fallback-weights",
        type=str,
        default="0.3,0.5,0.7",
    )
    parser.add_argument(
        "--sweep-alpha-beta",
        type=str,
        default="1.0:0.0,0.7:0.3,0.5:0.5",
    )
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--subset-hard-count", type=int, default=50)
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.5,
        help="Fraction used as dev/tuning split in split-replay mode.",
    )
    parser.add_argument(
        "--subset-indices",
        type=str,
        default=None,
        help="Comma-separated explicit heldout indices for e2e-subset.",
    )
    return parser.parse_args()


def apply_default_paths(args: argparse.Namespace) -> None:
    output_dir = args.exp_dir / "moe_eval"
    if args.prediction_cache is None:
        args.prediction_cache = args.exp_dir / "final_ablation_predictions.json"
    if args.router_cache is None:
        args.router_cache = output_dir / "router_cache_raw.json"
    if args.output_json is None:
        suffix = {
            "replay": "replay",
            "split-replay": "split_replay",
            "e2e-moe": "e2e_moe",
            "e2e-subset": "e2e_subset",
        }[args.mode]
        args.output_json = output_dir / f"moe_{suffix}_results.json"
    if args.output_md is None:
        suffix = {
            "replay": "replay",
            "split-replay": "split_replay",
            "e2e-moe": "e2e_moe",
            "e2e-subset": "e2e_subset",
        }[args.mode]
        args.output_md = output_dir / f"moe_{suffix}_report.md"
    if args.criterion_stats is None:
        args.criterion_stats = args.exp_dir / "epoch_final.json"


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_alpha_beta_grid(text: str) -> list[tuple[float, float]]:
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        alpha, beta = item.split(":", 1)
        pairs.append((float(alpha), float(beta)))
    return pairs


def parse_names(text: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in text.split(",") if name.strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_checkpoint(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "epoch_final.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def sample_identity(dataset: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "sample_id": item.get("sample_id"),
            "gold": item["answer"],
        }
        for index, item in enumerate(dataset)
    ]


def load_prediction_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction cache: {path}")
    cache = json.loads(path.read_text(encoding="utf-8"))
    for key in ("criteria", "samples", "prediction"):
        if key not in cache:
            raise ValueError(f"Prediction cache missing {key!r}: {path}")
    return cache


def validate_prediction_cache(
    cache: dict[str, Any], dataset: Sequence[dict[str, Any]]
) -> None:
    expected = sample_identity(dataset)
    cached = cache["samples"]
    if len(cached) != len(expected):
        raise ValueError(
            f"Prediction cache sample count mismatch: {len(cached)} != {len(expected)}"
        )
    for expected_item, cached_item in zip(expected, cached):
        if (
            expected_item["sample_id"] != cached_item.get("sample_id")
            or expected_item["gold"] != cached_item.get("gold")
        ):
            raise ValueError(
                "Prediction cache sample identity mismatch at "
                f"index {expected_item['index']}"
            )


def cap_dataset_and_cache(
    dataset: list[dict[str, Any]], cache: dict[str, Any], limit: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if limit is None:
        return dataset, cache
    limited_cache = copy.deepcopy(cache)
    limited_cache["samples"] = limited_cache["samples"][:limit]
    limited_cache["prediction"] = limited_cache["prediction"][:limit]
    return dataset[:limit], limited_cache


def criteria_from_cache(cache: dict[str, Any]) -> list[Criterion]:
    return [Criterion.from_dict(item) for item in cache["criteria"]]


def with_temperature(args_dict: dict[str, Any], temperature: float) -> dict[str, Any]:
    copied = copy.deepcopy(args_dict)
    request_kwargs = copied.setdefault("request_kwargs", {})
    request_kwargs["temperature"] = temperature
    return copied


def resolve_runtime_options(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    evaluator_kwargs = checkpoint.get("evaluator_kwargs") or {}
    worker_args = with_temperature(checkpoint["worker_args"], args.worker_temperature)
    router_args = with_temperature(checkpoint["worker_args"], args.router_temperature)
    max_concurrent = args.max_concurrent or checkpoint.get("worker_max_concurrent", 40)
    image_field = args.image_field or evaluator_kwargs.get("image_field", "image_path")
    question_field = args.question_field or evaluator_kwargs.get(
        "question_field", "question"
    )
    encode_local_image = evaluator_kwargs.get("encode_local_image", True)
    if args.no_encode_local_image:
        encode_local_image = False
    return {
        "worker_args": worker_args,
        "router_args": router_args,
        "max_concurrent": max_concurrent,
        "image_field": image_field,
        "question_field": question_field,
        "encode_local_image": encode_local_image,
        "worker_prompt": checkpoint.get("worker_prompt"),
    }


def decision_to_dict(decision: RoutingDecision) -> dict[str, Any]:
    return {
        "scene_analysis": decision.scene_analysis,
        "routing_weights": decision.routing_weights,
        "source": decision.source,
    }


def dict_to_decision(data: dict[str, Any]) -> RoutingDecision:
    return RoutingDecision(
        scene_analysis=str(data.get("scene_analysis", "")),
        routing_weights={
            str(name): clamp_weight(value)
            for name, value in (data.get("routing_weights") or {}).items()
        },
        source=data.get("source", "router"),
    )


def trim_router_cache_to_dataset(
    cache: dict[str, Any], dataset: Sequence[dict[str, Any]], criterion_names: list[str]
) -> dict[str, Any]:
    """Allow a full router cache to serve a --limit prefix replay."""
    if cache.get("criteria_names") != criterion_names:
        raise ValueError("Router cache criterion names do not match prediction cache")
    expected_samples = sample_identity(dataset)
    cached_samples = cache.get("samples", [])
    cached_routing = cache.get("routing", [])
    if cached_samples[: len(expected_samples)] != expected_samples:
        raise ValueError("Router cache sample identity does not match heldout data")
    if len(cached_routing) < len(expected_samples):
        raise ValueError("Router cache routing length is shorter than heldout data")
    trimmed = copy.deepcopy(cache)
    trimmed["samples"] = cached_samples[: len(expected_samples)]
    trimmed["routing"] = cached_routing[: len(expected_samples)]
    return trimmed


def route_dataset(
    args: argparse.Namespace,
    dataset: Sequence[dict[str, Any]],
    criteria: Sequence[Criterion],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    router = RouterManager(
        router_args=runtime["router_args"],
        image_field=runtime["image_field"],
        question_field=runtime["question_field"],
        encode_local_image=runtime["encode_local_image"],
        routing_threshold=0.0,
        fallback_criteria=(),
        fallback_weight=0.0,
        max_retries=args.max_retries,
    )
    with ThreadPoolExecutor(max_workers=runtime["max_concurrent"]) as executor:
        futures = [executor.submit(router.route, item, criteria) for item in dataset]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            dynamic_ncols=True,
            disable=not USE_TQDM,
        ):
            pass
        decisions = [future.result() for future in futures]

    return {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(),
            "router_args": runtime["router_args"],
            "max_concurrent": runtime["max_concurrent"],
            "max_retries": args.max_retries,
            "note": "Raw router cache; broad fallback is applied during replay variants.",
        },
        "criteria_names": [criterion.name for criterion in criteria],
        "samples": sample_identity(dataset),
        "routing": [decision_to_dict(decision) for decision in decisions],
    }


def load_or_create_router_cache(
    args: argparse.Namespace,
    dataset: Sequence[dict[str, Any]],
    criteria: Sequence[Criterion],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    criterion_names = [criterion.name for criterion in criteria]
    if args.router_cache.exists() and args.router_cache_policy != "run":
        cache = json.loads(args.router_cache.read_text(encoding="utf-8"))
        return trim_router_cache_to_dataset(cache, dataset, criterion_names)

    if args.router_cache_policy == "cache-only":
        raise FileNotFoundError(f"Missing router cache: {args.router_cache}")

    cache = route_dataset(args, dataset, criteria, runtime)
    args.router_cache.parent.mkdir(parents=True, exist_ok=True)
    args.router_cache.write_text(
        json.dumps(cache, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    return cache


def clamp_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, weight))


def criterion_vote(vote_counts: dict[str, int]) -> str | None:
    if vote_counts.get("U", 0) > 0:
        return None
    if vote_counts.get("A", 0) > vote_counts.get("B", 0):
        return "A"
    if vote_counts.get("B", 0) > vote_counts.get("A", 0):
        return "B"
    return None


def majority_vote(votes: dict[str, int]) -> str | None:
    if votes["A"] > votes["B"]:
        return "A"
    if votes["B"] > votes["A"]:
        return "B"
    return None


def apply_broad_fallback(
    weights: dict[str, float],
    criterion_names: Sequence[str],
    fallback_criteria: Sequence[str],
    fallback_weight: float,
) -> dict[str, float]:
    result = {name: clamp_weight(weights.get(name, 0.0)) for name in criterion_names}
    available = set(criterion_names)
    for name in fallback_criteria:
        if name in available:
            result[name] = max(result.get(name, 0.0), fallback_weight)
    return result


def build_static_weights(
    criterion_names: Sequence[str],
    performances: dict[str, CriterionPerformance],
    alpha: float,
    beta: float,
    use_static: bool,
) -> dict[str, float]:
    if not use_static:
        return {name: 1.0 for name in criterion_names}
    weights = {}
    for name in criterion_names:
        stat = performances.get(name)
        weights[name] = 1.0 if stat is None else static_weight(stat, alpha, beta)
    return weights


def evaluate_majority(
    dataset: Sequence[dict[str, Any]],
    prediction: Sequence[dict[str, dict[str, int]]],
    criterion_names: Sequence[str],
) -> dict[str, Any]:
    sample_results = []
    correct = 0
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
        final_vote = majority_vote(votes)
        is_correct = final_vote is not None and final_vote == item["answer"]
        correct += int(is_correct)
        none_final_count += int(final_vote is None)
        none_vote_total += votes["None"]
        sample_results.append(
            {
                "index": index,
                "sample_id": item.get("sample_id"),
                "gold": item["answer"],
                "final_vote": final_vote,
                "correct": is_correct,
                "votes": votes,
                "active_criteria": list(criterion_names),
                "active_count": len(criterion_names),
            }
        )
    total = len(dataset)
    return {
        "system": "majority_best11",
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "none_final_count": none_final_count,
        "tie_count": none_final_count,
        "avg_none_votes": none_vote_total / total if total else 0.0,
        "estimated_worker_calls": total * len(criterion_names),
        "avg_active_criteria": len(criterion_names),
        "pruning_rate": 0.0,
        "sample_results": sample_results,
    }


def evaluate_weighted_replay(
    system_name: str,
    dataset: Sequence[dict[str, Any]],
    prediction: Sequence[dict[str, dict[str, int]]],
    criterion_names: Sequence[str],
    routing: Sequence[RoutingDecision],
    static_weights: dict[str, float],
    routing_threshold: float,
    tie_epsilon: float,
    use_router: bool,
    fallback_criteria: Sequence[str],
    fallback_weight: float,
) -> dict[str, Any]:
    sample_results = []
    correct = 0
    tie_count = 0
    none_vote_total = 0
    worker_calls = 0

    for index, (item, row, decision) in enumerate(zip(dataset, prediction, routing)):
        dynamic_weights = (
            apply_broad_fallback(
                decision.routing_weights,
                criterion_names,
                fallback_criteria,
                fallback_weight,
            )
            if use_router
            else {name: 1.0 for name in criterion_names}
        )
        votes = {"A": 0, "B": 0, "None": 0}
        active_criteria = [
            name
            for name in criterion_names
            if dynamic_weights.get(name, 0.0) >= routing_threshold
        ]
        worker_calls += len(active_criteria)

        score = {"A": 0.0, "B": 0.0}
        denominator = 0.0
        for name in active_criteria:
            vote = criterion_vote(row[name])
            if vote is None:
                votes["None"] += 1
                continue
            votes[vote] += 1
            weight = static_weights[name] * dynamic_weights.get(name, 0.0)
            if weight <= 0:
                continue
            score[vote] += weight
            denominator += weight

        if denominator <= 0:
            final_vote = "Tie"
            score_a = 0.0
            score_b = 0.0
        else:
            score_a = score["A"] / denominator
            score_b = score["B"] / denominator
            if abs(score_a - score_b) < tie_epsilon:
                final_vote = "Tie"
            elif score_a > score_b:
                final_vote = "A"
            else:
                final_vote = "B"

        is_correct = final_vote in ("A", "B") and final_vote == item["answer"]
        correct += int(is_correct)
        tie_count += int(final_vote == "Tie")
        none_vote_total += votes["None"]
        sample_results.append(
            {
                "index": index,
                "sample_id": item.get("sample_id"),
                "gold": item["answer"],
                "final_vote": final_vote,
                "correct": is_correct,
                "votes": votes,
                "score_a": score_a,
                "score_b": score_b,
                "denominator": denominator,
                "active_criteria": active_criteria,
                "active_count": len(active_criteria),
            }
        )

    total = len(dataset)
    max_calls = total * len(criterion_names)
    return {
        "system": system_name,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "none_final_count": tie_count,
        "tie_count": tie_count,
        "avg_none_votes": none_vote_total / total if total else 0.0,
        "estimated_worker_calls": worker_calls,
        "avg_active_criteria": worker_calls / total if total else 0.0,
        "pruning_rate": 1.0 - (worker_calls / max_calls) if max_calls else 0.0,
        "sample_results": sample_results,
    }


def compute_hard_indices(
    dataset: Sequence[dict[str, Any]],
    prediction: Sequence[dict[str, dict[str, int]]],
    criterion_names: Sequence[str],
    baseline: dict[str, Any],
) -> set[int]:
    baseline_by_index = {item["index"]: item for item in baseline["sample_results"]}
    hard_indices = set()
    for index, (item, row) in enumerate(zip(dataset, prediction)):
        if baseline_by_index[index]["correct"]:
            continue
        if any(criterion_vote(row[name]) == item["answer"] for name in criterion_names):
            hard_indices.add(index)
    return hard_indices


def attach_deltas_and_rescues(
    results: dict[str, dict[str, Any]], baseline_key: str, hard_indices: set[int]
) -> None:
    baseline = results[baseline_key]
    baseline_by_index = {
        item["index"]: item for item in baseline["sample_results"]
    }
    for name, result in results.items():
        result["delta_vs_majority_accuracy"] = result["accuracy"] - baseline["accuracy"]
        result["delta_vs_majority_correct"] = result["correct"] - baseline["correct"]
        gained = []
        lost = []
        rescued_hard = []
        for sample in result["sample_results"]:
            base = baseline_by_index[sample["index"]]
            if sample["correct"] and not base["correct"]:
                gained.append(compare_sample(sample, base))
                if sample["index"] in hard_indices:
                    rescued_hard.append(sample["index"])
            elif base["correct"] and not sample["correct"]:
                lost.append(compare_sample(sample, base))
        result["gained_over_majority"] = gained
        result["lost_vs_majority"] = lost
        result["hard_rescue_count"] = len(rescued_hard)
        result["hard_rescue_rate"] = (
            len(rescued_hard) / len(hard_indices) if hard_indices else 0.0
        )
        result["hard_rescued_indices"] = rescued_hard


def compare_sample(sample: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": sample["index"],
        "sample_id": sample.get("sample_id"),
        "gold": sample["gold"],
        "majority_vote": baseline["final_vote"],
        "variant_vote": sample["final_vote"],
        "majority_votes": baseline.get("votes"),
        "variant_votes": sample.get("votes"),
        "active_criteria": sample.get("active_criteria"),
    }


def summarize_routing(
    routing: Sequence[RoutingDecision],
    criterion_names: Sequence[str],
    fallback_criteria: Sequence[str],
    fallback_weight: float,
    routing_threshold: float,
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    total = len(routing)
    for name in criterion_names:
        raw_weights = [decision.routing_weights.get(name, 0.0) for decision in routing]
        fallback_weights = [
            apply_broad_fallback(
                decision.routing_weights,
                criterion_names,
                fallback_criteria,
                fallback_weight,
            ).get(name, 0.0)
            for decision in routing
        ]
        active = sum(weight >= routing_threshold for weight in fallback_weights)
        summary[name] = {
            "mean_raw_weight": sum(raw_weights) / total if total else 0.0,
            "mean_fallback_weight": (
                sum(fallback_weights) / total if total else 0.0
            ),
            "active_count": active,
            "active_rate": active / total if total else 0.0,
        }
    return summary


def run_sweep(
    dataset: Sequence[dict[str, Any]],
    prediction: Sequence[dict[str, dict[str, int]]],
    criterion_names: Sequence[str],
    routing: Sequence[RoutingDecision],
    performances: dict[str, CriterionPerformance],
    fallback_criteria: Sequence[str],
    thresholds: Sequence[float],
    tie_epsilons: Sequence[float],
    fallback_weights: Sequence[float],
    alpha_beta_grid: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        for tie_epsilon in tie_epsilons:
            for fallback_weight in fallback_weights:
                for alpha, beta in alpha_beta_grid:
                    static_weights = build_static_weights(
                        criterion_names,
                        performances,
                        alpha,
                        beta,
                        use_static=True,
                    )
                    result = evaluate_weighted_replay(
                        "sweep_full_moe",
                        dataset,
                        prediction,
                        criterion_names,
                        routing,
                        static_weights,
                        routing_threshold=threshold,
                        tie_epsilon=tie_epsilon,
                        use_router=True,
                        fallback_criteria=fallback_criteria,
                        fallback_weight=fallback_weight,
                    )
                    rows.append(
                        {
                            "routing_threshold": threshold,
                            "tie_epsilon": tie_epsilon,
                            "fallback_weight": fallback_weight,
                            "static_alpha": alpha,
                            "static_beta": beta,
                            "accuracy": result["accuracy"],
                            "correct": result["correct"],
                            "total": result["total"],
                            "tie_count": result["tie_count"],
                            "estimated_worker_calls": result[
                                "estimated_worker_calls"
                            ],
                            "avg_active_criteria": result["avg_active_criteria"],
                            "pruning_rate": result["pruning_rate"],
                        }
                    )
    return sorted(
        rows,
        key=lambda row: (
            row["accuracy"],
            -row["avg_active_criteria"],
            -row["tie_count"],
        ),
        reverse=True,
    )


def subset_by_indices(items: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [items[index] for index in indices]


def split_indices(
    n_items: int, dev_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("--dev-fraction must be between 0 and 1")
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    dev_size = max(1, min(n_items - 1, round(n_items * dev_fraction)))
    return indices[:dev_size], indices[dev_size:]


def evaluate_named_full_moe_config(
    system_name: str,
    config: dict[str, Any],
    dataset: Sequence[dict[str, Any]],
    prediction: Sequence[dict[str, dict[str, int]]],
    criterion_names: Sequence[str],
    routing: Sequence[RoutingDecision],
    performances: dict[str, CriterionPerformance],
    fallback_criteria: Sequence[str],
) -> dict[str, Any]:
    alpha = float(config["static_alpha"])
    beta = float(config["static_beta"])
    static_weights = build_static_weights(
        criterion_names,
        performances,
        alpha,
        beta,
        use_static=True,
    )
    return evaluate_weighted_replay(
        system_name,
        dataset,
        prediction,
        criterion_names,
        routing,
        static_weights,
        routing_threshold=float(config["routing_threshold"]),
        tie_epsilon=float(config["tie_epsilon"]),
        use_router=True,
        fallback_criteria=fallback_criteria,
        fallback_weight=float(config["fallback_weight"]),
    )


def format_sample_ids(samples: Sequence[dict[str, Any]], max_items: int = 40) -> str:
    if not samples:
        return "none"
    labels = [
        str(sample.get("sample_id") if sample.get("sample_id") is not None else sample["index"])
        for sample in samples[:max_items]
    ]
    if len(samples) > max_items:
        labels.append(f"... +{len(samples) - max_items} more")
    return ", ".join(labels)


def build_replay_report(output: dict[str, Any]) -> str:
    metadata = output["metadata"]
    results = output["results"]
    sweep = output["sweep"][:20]
    routing_summary = output["routing_summary"]
    lines = [
        "# MoE Routing Replay Report",
        "",
        f"- Created at: {metadata['created_at']}",
        f"- Experiment dir: `{metadata['exp_dir']}`",
        f"- Heldout path: `{metadata['heldout_path']}`",
        f"- Prediction cache: `{metadata['prediction_cache']}`",
        f"- Router cache: `{metadata['router_cache']}`",
        f"- Samples: {metadata['n_samples']}",
        f"- Criteria: {metadata['n_criteria']}",
        "",
        "## Main Results",
        "",
        "| system | accuracy | correct/total | delta vs majority | ties | avg active | pruning | hard rescue |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in sorted(
        results.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        lines.append(
            f"| {name} | {result['accuracy']:.3f} | "
            f"{result['correct']}/{result['total']} | "
            f"{result['delta_vs_majority_accuracy']:+.3f} "
            f"({result['delta_vs_majority_correct']:+d}) | "
            f"{result['tie_count']} | {result['avg_active_criteria']:.2f} | "
            f"{result['pruning_rate']:.1%} | "
            f"{result['hard_rescue_count']} ({result['hard_rescue_rate']:.1%}) |"
        )

    lines.extend(
        [
            "",
            "## Top Sweep Configs",
            "",
            "| rank | acc | correct | threshold | tie eps | fallback | alpha:beta | avg active | pruning | ties |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(sweep, start=1):
        lines.append(
            f"| {rank} | {row['accuracy']:.3f} | "
            f"{row['correct']}/{row['total']} | "
            f"{row['routing_threshold']:.2f} | {row['tie_epsilon']:.2f} | "
            f"{row['fallback_weight']:.2f} | "
            f"{row['static_alpha']:.1f}:{row['static_beta']:.1f} | "
            f"{row['avg_active_criteria']:.2f} | {row['pruning_rate']:.1%} | "
            f"{row['tie_count']} |"
        )

    lines.extend(["", "## Gain/Loss Vs Majority", ""])
    for name, result in results.items():
        if name == "majority_best11":
            continue
        lines.extend(
            [
                f"### {name}",
                "",
                f"- gained: {len(result['gained_over_majority'])} "
                f"({format_sample_ids(result['gained_over_majority'])})",
                f"- lost: {len(result['lost_vs_majority'])} "
                f"({format_sample_ids(result['lost_vs_majority'])})",
                "",
            ]
        )

    lines.extend(
        [
            "## Router Summary",
            "",
            "| criterion | mean raw | mean fallback | active rate | active count |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, stats in sorted(
        routing_summary.items(),
        key=lambda item: item[1]["active_rate"],
        reverse=True,
    ):
        lines.append(
            f"| {name} | {stats['mean_raw_weight']:.3f} | "
            f"{stats['mean_fallback_weight']:.3f} | "
            f"{stats['active_rate']:.1%} | {stats['active_count']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_split_replay_report(output: dict[str, Any]) -> str:
    metadata = output["metadata"]
    selected = output["selected_config"]
    dev_results = output["dev_results"]
    test_results = output["test_results"]
    lines = [
        "# MoE Split Replay Report",
        "",
        f"- Created at: {metadata['created_at']}",
        f"- Experiment dir: `{metadata['exp_dir']}`",
        f"- Prediction cache: `{metadata['prediction_cache']}`",
        f"- Router cache: `{metadata['router_cache']}`",
        f"- Dev/Test samples: {metadata['n_dev']} / {metadata['n_test']}",
        f"- Selected config from dev: threshold={selected['routing_threshold']}, "
        f"tie_epsilon={selected['tie_epsilon']}, "
        f"fallback_weight={selected['fallback_weight']}, "
        f"alpha:beta={selected['static_alpha']}:{selected['static_beta']}",
        "",
        "## Dev Selection",
        "",
        "| system | accuracy | correct/total | delta vs majority | ties | avg active | pruning |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in sorted(
        dev_results.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        lines.append(
            f"| {name} | {result['accuracy']:.3f} | "
            f"{result['correct']}/{result['total']} | "
            f"{result['delta_vs_majority_accuracy']:+.3f} "
            f"({result['delta_vs_majority_correct']:+d}) | "
            f"{result['tie_count']} | {result['avg_active_criteria']:.2f} | "
            f"{result['pruning_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Test Verification",
            "",
            "| system | accuracy | correct/total | delta vs majority | ties | avg active | pruning | hard rescue |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, result in sorted(
        test_results.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        lines.append(
            f"| {name} | {result['accuracy']:.3f} | "
            f"{result['correct']}/{result['total']} | "
            f"{result['delta_vs_majority_accuracy']:+.3f} "
            f"({result['delta_vs_majority_correct']:+d}) | "
            f"{result['tie_count']} | {result['avg_active_criteria']:.2f} | "
            f"{result['pruning_rate']:.1%} | "
            f"{result['hard_rescue_count']} ({result['hard_rescue_rate']:.1%}) |"
        )
    lines.extend(
        [
            "",
            "## Top Dev Sweep Configs",
            "",
            "| rank | acc | correct | threshold | tie eps | fallback | alpha:beta | avg active | pruning | ties |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(output["dev_sweep"][:20], start=1):
        lines.append(
            f"| {rank} | {row['accuracy']:.3f} | "
            f"{row['correct']}/{row['total']} | "
            f"{row['routing_threshold']:.2f} | {row['tie_epsilon']:.2f} | "
            f"{row['fallback_weight']:.2f} | "
            f"{row['static_alpha']:.1f}:{row['static_beta']:.1f} | "
            f"{row['avg_active_criteria']:.2f} | {row['pruning_rate']:.1%} | "
            f"{row['tie_count']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def print_replay_dry_run(
    args: argparse.Namespace,
    dataset: Sequence[dict[str, Any]],
    cache: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    print("Dry run only. No model calls will be made.")
    print(f"Mode: {args.mode}")
    print(f"Samples: {len(dataset)}")
    print(f"Criteria: {len(cache['criteria'])}")
    print(f"Prediction cache: {args.prediction_cache}")
    print(f"Router cache: {args.router_cache}")
    print(f"Router cache policy: {args.router_cache_policy}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output Markdown: {args.output_md}")
    print(f"Max concurrent: {runtime['max_concurrent']}")
    print(f"Image field: {runtime['image_field']}")
    print(f"Question field: {runtime['question_field']}")
    print(f"Encode local image: {runtime['encode_local_image']}")


def run_replay(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.exp_dir)
    runtime = resolve_runtime_options(args, checkpoint)
    dataset = load_rlhfv_pair_data(args.heldout_path)
    prediction_cache = load_prediction_cache(args.prediction_cache)
    validate_prediction_cache(prediction_cache, dataset)
    dataset, prediction_cache = cap_dataset_and_cache(
        dataset, prediction_cache, args.limit
    )
    criteria = criteria_from_cache(prediction_cache)
    criterion_names = [criterion.name for criterion in criteria]

    if args.dry_run:
        print_replay_dry_run(args, dataset, prediction_cache, runtime)
        return

    router_cache = load_or_create_router_cache(args, dataset, criteria, runtime)
    routing = [dict_to_decision(item) for item in router_cache["routing"]]
    performances = load_criterion_performance(args.criterion_stats)
    prediction = prediction_cache["prediction"]
    fallback_criteria = parse_names(args.fallback_criteria)

    results: dict[str, dict[str, Any]] = {
        "majority_best11": evaluate_majority(dataset, prediction, criterion_names)
    }
    static_weights = build_static_weights(
        criterion_names,
        performances,
        args.static_alpha,
        args.static_beta,
        use_static=True,
    )
    unit_weights = build_static_weights(
        criterion_names,
        performances,
        args.static_alpha,
        args.static_beta,
        use_static=False,
    )
    results["static_only_weighted"] = evaluate_weighted_replay(
        "static_only_weighted",
        dataset,
        prediction,
        criterion_names,
        routing,
        static_weights,
        routing_threshold=0.0,
        tie_epsilon=args.tie_epsilon,
        use_router=False,
        fallback_criteria=(),
        fallback_weight=0.0,
    )
    results["router_only_weighted"] = evaluate_weighted_replay(
        "router_only_weighted",
        dataset,
        prediction,
        criterion_names,
        routing,
        unit_weights,
        routing_threshold=args.routing_threshold,
        tie_epsilon=args.tie_epsilon,
        use_router=True,
        fallback_criteria=fallback_criteria,
        fallback_weight=args.fallback_weight,
    )
    results["full_moe"] = evaluate_weighted_replay(
        "full_moe",
        dataset,
        prediction,
        criterion_names,
        routing,
        static_weights,
        routing_threshold=args.routing_threshold,
        tie_epsilon=args.tie_epsilon,
        use_router=True,
        fallback_criteria=fallback_criteria,
        fallback_weight=args.fallback_weight,
    )
    results["full_moe_no_fallback"] = evaluate_weighted_replay(
        "full_moe_no_fallback",
        dataset,
        prediction,
        criterion_names,
        routing,
        static_weights,
        routing_threshold=args.routing_threshold,
        tie_epsilon=args.tie_epsilon,
        use_router=True,
        fallback_criteria=(),
        fallback_weight=0.0,
    )

    hard_indices = compute_hard_indices(
        dataset, prediction, criterion_names, results["majority_best11"]
    )
    attach_deltas_and_rescues(results, "majority_best11", hard_indices)
    sweep = run_sweep(
        dataset=dataset,
        prediction=prediction,
        criterion_names=criterion_names,
        routing=routing,
        performances=performances,
        fallback_criteria=fallback_criteria,
        thresholds=parse_float_list(args.sweep_routing_thresholds),
        tie_epsilons=parse_float_list(args.sweep_tie_epsilons),
        fallback_weights=parse_float_list(args.sweep_fallback_weights),
        alpha_beta_grid=parse_alpha_beta_grid(args.sweep_alpha_beta),
    )
    routing_summary = summarize_routing(
        routing,
        criterion_names,
        fallback_criteria,
        args.fallback_weight,
        args.routing_threshold,
    )

    output = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": args.mode,
            "exp_dir": str(args.exp_dir),
            "heldout_path": str(args.heldout_path),
            "prediction_cache": str(args.prediction_cache),
            "router_cache": str(args.router_cache),
            "criterion_stats": str(args.criterion_stats),
            "n_samples": len(dataset),
            "n_criteria": len(criteria),
            "routing_threshold": args.routing_threshold,
            "tie_epsilon": args.tie_epsilon,
            "fallback_criteria": list(fallback_criteria),
            "fallback_weight": args.fallback_weight,
            "static_alpha": args.static_alpha,
            "static_beta": args.static_beta,
            "hard_case_count": len(hard_indices),
        },
        "criteria": [criterion.to_dict() for criterion in criteria],
        "static_weights": static_weights,
        "routing_summary": routing_summary,
        "results": results,
        "sweep": sweep,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_replay_report(output), encoding="utf-8")

    print(f"Saved replay JSON: {args.output_json}")
    print(f"Saved replay report: {args.output_md}")
    for name, result in sorted(
        results.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        print(
            f"{name}: {result['accuracy']:.3f} = "
            f"{result['correct']}/{result['total']} "
            f"delta={result['delta_vs_majority_accuracy']:+.3f} "
            f"avg_active={result['avg_active_criteria']:.2f}"
        )


def print_split_replay_dry_run(
    args: argparse.Namespace,
    dataset: Sequence[dict[str, Any]],
    cache: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    dev_indices, test_indices = split_indices(len(dataset), args.dev_fraction, args.seed)
    print("Dry run only. No model calls will be made.")
    print(f"Mode: {args.mode}")
    print(f"Samples: {len(dataset)}")
    print(f"Dev/Test: {len(dev_indices)} / {len(test_indices)}")
    print(f"Criteria: {len(cache['criteria'])}")
    print(f"Prediction cache: {args.prediction_cache}")
    print(f"Router cache: {args.router_cache}")
    print(f"Router cache policy: {args.router_cache_policy}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output Markdown: {args.output_md}")
    print(f"Max concurrent: {runtime['max_concurrent']}")


def run_split_replay(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.exp_dir)
    runtime = resolve_runtime_options(args, checkpoint)
    dataset = load_rlhfv_pair_data(args.heldout_path)
    prediction_cache = load_prediction_cache(args.prediction_cache)
    validate_prediction_cache(prediction_cache, dataset)
    dataset, prediction_cache = cap_dataset_and_cache(
        dataset, prediction_cache, args.limit
    )
    criteria = criteria_from_cache(prediction_cache)
    criterion_names = [criterion.name for criterion in criteria]

    if args.dry_run:
        print_split_replay_dry_run(args, dataset, prediction_cache, runtime)
        return

    router_cache = load_or_create_router_cache(args, dataset, criteria, runtime)
    routing = [dict_to_decision(item) for item in router_cache["routing"]]
    performances = load_criterion_performance(args.criterion_stats)
    prediction = prediction_cache["prediction"]
    fallback_criteria = parse_names(args.fallback_criteria)
    dev_indices, test_indices = split_indices(
        len(dataset), args.dev_fraction, args.seed
    )

    dev_dataset = subset_by_indices(dataset, dev_indices)
    dev_prediction = subset_by_indices(prediction, dev_indices)
    dev_routing = subset_by_indices(routing, dev_indices)
    test_dataset = subset_by_indices(dataset, test_indices)
    test_prediction = subset_by_indices(prediction, test_indices)
    test_routing = subset_by_indices(routing, test_indices)

    dev_sweep = run_sweep(
        dataset=dev_dataset,
        prediction=dev_prediction,
        criterion_names=criterion_names,
        routing=dev_routing,
        performances=performances,
        fallback_criteria=fallback_criteria,
        thresholds=parse_float_list(args.sweep_routing_thresholds),
        tie_epsilons=parse_float_list(args.sweep_tie_epsilons),
        fallback_weights=parse_float_list(args.sweep_fallback_weights),
        alpha_beta_grid=parse_alpha_beta_grid(args.sweep_alpha_beta),
    )
    selected = dev_sweep[0]
    default_config = {
        "routing_threshold": args.routing_threshold,
        "tie_epsilon": args.tie_epsilon,
        "fallback_weight": args.fallback_weight,
        "static_alpha": args.static_alpha,
        "static_beta": args.static_beta,
    }

    dev_results = {
        "majority_best11": evaluate_majority(
            dev_dataset, dev_prediction, criterion_names
        ),
        "full_moe_default": evaluate_named_full_moe_config(
            "full_moe_default",
            default_config,
            dev_dataset,
            dev_prediction,
            criterion_names,
            dev_routing,
            performances,
            fallback_criteria,
        ),
        "full_moe_selected": evaluate_named_full_moe_config(
            "full_moe_selected",
            selected,
            dev_dataset,
            dev_prediction,
            criterion_names,
            dev_routing,
            performances,
            fallback_criteria,
        ),
    }
    test_results = {
        "majority_best11": evaluate_majority(
            test_dataset, test_prediction, criterion_names
        ),
        "full_moe_default": evaluate_named_full_moe_config(
            "full_moe_default",
            default_config,
            test_dataset,
            test_prediction,
            criterion_names,
            test_routing,
            performances,
            fallback_criteria,
        ),
        "full_moe_selected": evaluate_named_full_moe_config(
            "full_moe_selected",
            selected,
            test_dataset,
            test_prediction,
            criterion_names,
            test_routing,
            performances,
            fallback_criteria,
        ),
    }

    dev_hard = compute_hard_indices(
        dev_dataset,
        dev_prediction,
        criterion_names,
        dev_results["majority_best11"],
    )
    test_hard = compute_hard_indices(
        test_dataset,
        test_prediction,
        criterion_names,
        test_results["majority_best11"],
    )
    attach_deltas_and_rescues(dev_results, "majority_best11", dev_hard)
    attach_deltas_and_rescues(test_results, "majority_best11", test_hard)

    output = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": args.mode,
            "exp_dir": str(args.exp_dir),
            "heldout_path": str(args.heldout_path),
            "prediction_cache": str(args.prediction_cache),
            "router_cache": str(args.router_cache),
            "criterion_stats": str(args.criterion_stats),
            "seed": args.seed,
            "dev_fraction": args.dev_fraction,
            "n_dev": len(dev_dataset),
            "n_test": len(test_dataset),
            "n_criteria": len(criteria),
            "fallback_criteria": list(fallback_criteria),
            "default_config": default_config,
            "dev_hard_case_count": len(dev_hard),
            "test_hard_case_count": len(test_hard),
        },
        "criteria": [criterion.to_dict() for criterion in criteria],
        "selected_config": selected,
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "dev_sweep": dev_sweep,
        "dev_results": dev_results,
        "test_results": test_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_split_replay_report(output), encoding="utf-8")

    print(f"Saved split replay JSON: {args.output_json}")
    print(f"Saved split replay report: {args.output_md}")
    print(
        "selected:",
        f"threshold={selected['routing_threshold']}",
        f"tie_epsilon={selected['tie_epsilon']}",
        f"fallback={selected['fallback_weight']}",
        f"alpha:beta={selected['static_alpha']}:{selected['static_beta']}",
    )
    for name, result in sorted(
        test_results.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        print(
            f"test/{name}: {result['accuracy']:.3f} = "
            f"{result['correct']}/{result['total']} "
            f"delta={result['delta_vs_majority_accuracy']:+.3f} "
            f"avg_active={result['avg_active_criteria']:.2f}"
        )


def parse_subset_indices(text: str | None) -> list[int] | None:
    if text is None:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def choose_e2e_subset(
    args: argparse.Namespace,
    dataset: Sequence[dict[str, Any]],
    prediction: Sequence[dict[str, dict[str, int]]],
    criterion_names: Sequence[str],
) -> list[int]:
    explicit = parse_subset_indices(args.subset_indices)
    if explicit is not None:
        return explicit

    baseline = evaluate_majority(dataset, prediction, criterion_names)
    hard_indices = list(
        compute_hard_indices(dataset, prediction, criterion_names, baseline)
    )
    baseline_correct = [
        sample["index"] for sample in baseline["sample_results"] if sample["correct"]
    ]
    rng = random.Random(args.seed)
    rng.shuffle(hard_indices)
    rng.shuffle(baseline_correct)

    n_hard = min(args.subset_hard_count, args.subset_size, len(hard_indices))
    n_easy = max(0, min(args.subset_size - n_hard, len(baseline_correct)))
    selected = hard_indices[:n_hard] + baseline_correct[:n_easy]
    rng.shuffle(selected)
    return selected


def eval_output_to_dict(eval_output: Any) -> dict[str, Any]:
    return {
        "prediction": eval_output.prediction,
        "is_correct": eval_output.is_correct,
        "accuracy": eval_output.accuracy,
        "per_criterion_acc": eval_output.per_criterion_acc,
        "routing": getattr(eval_output, "routing", None),
    }


def build_e2e_report(output: dict[str, Any]) -> str:
    majority = output["majority"]
    moe = output["moe"]
    moe_calls = sum(
        len(item.get("active_criteria", [])) for item in (moe.get("routing") or [])
    )
    max_calls = output["metadata"]["n_samples"] * output["metadata"]["n_criteria"]
    lines = [
        "# MoE Routing E2E Subset Report",
        "",
        f"- Created at: {output['metadata']['created_at']}",
        f"- Samples: {output['metadata']['n_samples']}",
        f"- Criteria: {output['metadata']['n_criteria']}",
        f"- Majority accuracy: {majority['accuracy']:.3f}",
        f"- MoE accuracy: {moe['accuracy']:.3f}",
        f"- Estimated MoE worker calls: {moe_calls}/{max_calls}",
        "",
        "## Selected Samples",
        "",
        ", ".join(str(item['index']) for item in output["samples"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def select_e2e_moe_indices(
    args: argparse.Namespace, dataset: Sequence[dict[str, Any]]
) -> list[int]:
    explicit = parse_subset_indices(args.subset_indices)
    if explicit is not None:
        return explicit
    limit = args.limit if args.limit is not None else len(dataset)
    return list(range(min(limit, len(dataset))))


def build_e2e_moe_report(output: dict[str, Any]) -> str:
    metadata = output["metadata"]
    baseline = output["cached_majority"]
    moe = output["moe"]
    lines = [
        "# MoE Routing E2E Report",
        "",
        f"- Created at: {metadata['created_at']}",
        f"- Experiment dir: `{metadata['exp_dir']}`",
        f"- Heldout path: `{metadata['heldout_path']}`",
        f"- Samples: {metadata['n_samples']}",
        f"- Criteria: {metadata['n_criteria']}",
        f"- Config: threshold={metadata['routing_threshold']}, "
        f"tie_epsilon={metadata['tie_epsilon']}, "
        f"fallback_weight={metadata['fallback_weight']}, "
        f"alpha:beta={metadata['static_alpha']}:{metadata['static_beta']}",
        f"- Cached majority accuracy: {baseline['accuracy']:.3f} "
        f"= {baseline['correct']}/{baseline['total']}",
        f"- Real MoE accuracy: {moe['accuracy']:.3f} "
        f"= {sum(moe['is_correct'])}/{len(moe['is_correct'])}",
        f"- Delta vs cached majority: {output['delta_vs_cached_majority_accuracy']:+.3f} "
        f"({output['delta_vs_cached_majority_correct']:+d})",
        f"- Router calls: {metadata['router_calls']}",
        f"- Active worker calls: {metadata['active_worker_calls']}",
        f"- Full worker-call baseline: {metadata['full_worker_call_baseline']}",
        f"- Worker pruning: {metadata['worker_pruning_rate']:.1%}",
        f"- Avg active criteria: {metadata['avg_active_criteria']:.2f}",
        f"- Wall time seconds: {metadata['wall_time_seconds']:.1f}",
        "",
        "## Sample Indices",
        "",
        ", ".join(str(item["index"]) for item in output["samples"][:100]),
    ]
    if len(output["samples"]) > 100:
        lines.append(f"... +{len(output['samples']) - 100} more")
    return "\n".join(lines).rstrip() + "\n"


def run_e2e_moe(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.exp_dir)
    runtime = resolve_runtime_options(args, checkpoint)
    dataset = load_rlhfv_pair_data(args.heldout_path)
    prediction_cache = load_prediction_cache(args.prediction_cache)
    validate_prediction_cache(prediction_cache, dataset)
    criteria = criteria_from_cache(prediction_cache)
    criterion_names = [criterion.name for criterion in criteria]
    indices = select_e2e_moe_indices(args, dataset)
    selected_dataset = [dataset[index] for index in indices]
    selected_prediction = [prediction_cache["prediction"][index] for index in indices]
    selected_samples = [
        {
            "index": index,
            "sample_id": dataset[index].get("sample_id"),
            "gold": dataset[index]["answer"],
        }
        for index in indices
    ]
    cached_majority = evaluate_majority(
        selected_dataset, selected_prediction, criterion_names
    )

    if args.dry_run:
        print("Dry run only. No model calls will be made.")
        print(f"Mode: {args.mode}")
        print(f"Selected samples: {len(indices)}")
        print(f"Criteria: {len(criteria)}")
        print(f"Cached majority: {cached_majority['accuracy']:.3f}")
        print(f"Output JSON: {args.output_json}")
        print(f"Output Markdown: {args.output_md}")
        return

    if runtime["worker_prompt"] is None:
        raise ValueError("Checkpoint does not contain worker_prompt")

    moe_evaluator = MoEPairEvaluator(
        runtime["worker_args"],
        dataset=selected_dataset,
        max_concurrent=runtime["max_concurrent"],
        max_retries=args.max_retries,
        worker_prompt=runtime["worker_prompt"],
        image_field=runtime["image_field"],
        question_field=runtime["question_field"],
        encode_local_image=runtime["encode_local_image"],
        router_args=runtime["router_args"],
        criterion_stats=args.criterion_stats,
        static_alpha=args.static_alpha,
        static_beta=args.static_beta,
        routing_threshold=args.routing_threshold,
        fallback_criteria=parse_names(args.fallback_criteria),
        fallback_weight=args.fallback_weight,
        tie_epsilon=args.tie_epsilon,
    )
    started_at = time.perf_counter()
    moe_output = moe_evaluator.eval(criteria, update_score=False)
    wall_time_seconds = time.perf_counter() - started_at
    moe_dict = eval_output_to_dict(moe_output)
    routing = moe_dict.get("routing") or []
    active_worker_calls = sum(
        len(item.get("active_criteria", [])) for item in routing
    )
    full_worker_call_baseline = len(selected_dataset) * len(criteria)
    moe_correct = sum(moe_output.is_correct)
    cached_correct = cached_majority["correct"]

    output = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": args.mode,
            "exp_dir": str(args.exp_dir),
            "heldout_path": str(args.heldout_path),
            "prediction_cache": str(args.prediction_cache),
            "n_samples": len(selected_dataset),
            "n_criteria": len(criteria),
            "routing_threshold": args.routing_threshold,
            "tie_epsilon": args.tie_epsilon,
            "fallback_criteria": list(parse_names(args.fallback_criteria)),
            "fallback_weight": args.fallback_weight,
            "static_alpha": args.static_alpha,
            "static_beta": args.static_beta,
            "router_calls": len(selected_dataset),
            "active_worker_calls": active_worker_calls,
            "full_worker_call_baseline": full_worker_call_baseline,
            "worker_pruning_rate": (
                1.0 - active_worker_calls / full_worker_call_baseline
                if full_worker_call_baseline
                else 0.0
            ),
            "avg_active_criteria": (
                active_worker_calls / len(selected_dataset)
                if selected_dataset
                else 0.0
            ),
            "wall_time_seconds": wall_time_seconds,
        },
        "criteria": [criterion.to_dict() for criterion in criteria],
        "samples": selected_samples,
        "cached_majority": cached_majority,
        "moe": moe_dict,
        "delta_vs_cached_majority_accuracy": (
            moe_output.accuracy - cached_majority["accuracy"]
        ),
        "delta_vs_cached_majority_correct": moe_correct - cached_correct,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_e2e_moe_report(output), encoding="utf-8")
    print(f"Saved e2e MoE JSON: {args.output_json}")
    print(f"Saved e2e MoE report: {args.output_md}")
    print(
        f"cached_majority: {cached_majority['accuracy']:.3f} "
        f"= {cached_majority['correct']}/{cached_majority['total']}"
    )
    print(
        f"real_moe: {moe_output.accuracy:.3f} "
        f"= {moe_correct}/{len(moe_output.is_correct)}"
    )
    print(
        f"active_worker_calls={active_worker_calls}/{full_worker_call_baseline} "
        f"avg_active={output['metadata']['avg_active_criteria']:.2f} "
        f"pruning={output['metadata']['worker_pruning_rate']:.1%}"
    )


def run_e2e_subset(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.exp_dir)
    runtime = resolve_runtime_options(args, checkpoint)
    dataset = load_rlhfv_pair_data(args.heldout_path)
    prediction_cache = load_prediction_cache(args.prediction_cache)
    validate_prediction_cache(prediction_cache, dataset)
    criteria = criteria_from_cache(prediction_cache)
    criterion_names = [criterion.name for criterion in criteria]
    subset_indices = choose_e2e_subset(
        args, dataset, prediction_cache["prediction"], criterion_names
    )
    subset = [dataset[index] for index in subset_indices]
    subset_samples = [
        {
            "index": index,
            "sample_id": dataset[index].get("sample_id"),
            "gold": dataset[index]["answer"],
        }
        for index in subset_indices
    ]

    if args.dry_run:
        print("Dry run only. No model calls will be made.")
        print(f"Selected {len(subset_indices)} e2e samples")
        print(", ".join(str(index) for index in subset_indices))
        print(f"Output JSON: {args.output_json}")
        print(f"Output Markdown: {args.output_md}")
        return

    if runtime["worker_prompt"] is None:
        raise ValueError("Checkpoint does not contain worker_prompt")

    majority_evaluator = MultiModalPairEvaluator(
        runtime["worker_args"],
        dataset=subset,
        max_concurrent=runtime["max_concurrent"],
        max_retries=args.max_retries,
        worker_prompt=runtime["worker_prompt"],
        image_field=runtime["image_field"],
        question_field=runtime["question_field"],
        encode_local_image=runtime["encode_local_image"],
    )
    moe_evaluator = MoEPairEvaluator(
        runtime["worker_args"],
        dataset=subset,
        max_concurrent=runtime["max_concurrent"],
        max_retries=args.max_retries,
        worker_prompt=runtime["worker_prompt"],
        image_field=runtime["image_field"],
        question_field=runtime["question_field"],
        encode_local_image=runtime["encode_local_image"],
        router_args=runtime["router_args"],
        criterion_stats=args.criterion_stats,
        static_alpha=args.static_alpha,
        static_beta=args.static_beta,
        routing_threshold=args.routing_threshold,
        fallback_criteria=parse_names(args.fallback_criteria),
        fallback_weight=args.fallback_weight,
        tie_epsilon=args.tie_epsilon,
    )
    majority_output = majority_evaluator.eval(criteria, update_score=False)
    moe_output = moe_evaluator.eval(criteria, update_score=False)

    output = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": args.mode,
            "exp_dir": str(args.exp_dir),
            "heldout_path": str(args.heldout_path),
            "n_samples": len(subset),
            "n_criteria": len(criteria),
            "routing_threshold": args.routing_threshold,
            "tie_epsilon": args.tie_epsilon,
            "fallback_criteria": list(parse_names(args.fallback_criteria)),
            "fallback_weight": args.fallback_weight,
            "static_alpha": args.static_alpha,
            "static_beta": args.static_beta,
        },
        "criteria": [criterion.to_dict() for criterion in criteria],
        "samples": subset_samples,
        "majority": eval_output_to_dict(majority_output),
        "moe": eval_output_to_dict(moe_output),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_e2e_report(output), encoding="utf-8")
    print(f"Saved e2e JSON: {args.output_json}")
    print(f"Saved e2e report: {args.output_md}")
    print(f"majority: {majority_output.accuracy:.3f}")
    print(f"moe: {moe_output.accuracy:.3f}")


def main() -> None:
    args = parse_args()
    apply_default_paths(args)
    if args.mode == "replay":
        run_replay(args)
    elif args.mode == "split-replay":
        run_split_replay(args)
    elif args.mode == "e2e-moe":
        run_e2e_moe(args)
    elif args.mode == "e2e-subset":
        run_e2e_subset(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

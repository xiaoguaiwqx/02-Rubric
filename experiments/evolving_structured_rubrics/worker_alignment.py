"""Offline diagnostics for the exp4-to-Structured Worker semantic drift."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from critiq.structured import StructuredPredictionOutput, is_ab_swap_consistent


def _legacy_vote(value: dict[str, int]) -> str:
    for token in ("A", "B", "U"):
        if value.get(token) == 1:
            return token
    return "U"


def _majority(votes: list[str]) -> str:
    a, b = votes.count("A"), votes.count("B")
    return "A" if a > b else "B" if b > a else "U"


def build_worker_alignment_report(legacy_path: str | Path, structured_path: str | Path,
                                  swap_structured_path: str | Path | None = None) -> dict[str, Any]:
    """Compare the exact old vote matrix with Structured v1 on identical rows."""
    legacy = json.loads(Path(legacy_path).read_text(encoding="utf-8"))
    structured = StructuredPredictionOutput.load_json(structured_path)
    criteria = [item["name"] for item in legacy["criteria"]]
    if criteria != [item.name for item in structured.criteria]:
        raise ValueError("criterion order differs between artifacts")
    if len(legacy["prediction"]) != len(structured.node_outputs):
        raise ValueError("sample count differs between artifacts")
    gold = [sample["gold"] for sample in legacy["samples"]]
    if [sample["sample_id"] for sample in legacy["samples"]] != list(structured.sample_ids):
        raise ValueError("sample IDs/order differ between artifacts")

    transition: Counter[str] = Counter(); exact = 0; total = 0
    criterion_stats: dict[str, dict[str, Any]] = {}
    old_final, new_final = [], []
    parse_failures = consistency_failures = 0
    per_criterion: dict[str, Counter[str]] = defaultdict(Counter)
    for index, (old_row, new_row) in enumerate(zip(legacy["prediction"], structured.node_outputs)):
        old_votes, new_votes = [], []
        for name in criteria:
            old = _legacy_vote(old_row[name])
            decision = new_row[name].local_decision
            new = "A" if decision.vote.value == "A" else "B" if decision.vote.value == "B" else "U"
            transition[f"{old}->{new}"] += 1; per_criterion[name][f"{old}->{new}"] += 1
            exact += old == new; total += 1; old_votes.append(old); new_votes.append(new)
            parse_failures += not new_row[name].judgement.parse_ok
            consistency_failures += new_row[name].judgement.parse_ok and not new_row[name].judgement.consistency_ok
            per_criterion[name][f"applicable:{new_row[name].judgement.applicable.value}"] += 1
            if old in {"A", "B"}: per_criterion[name]["legacy_decisive_correct"] += old == gold[index]
            if new in {"A", "B"}: per_criterion[name]["structured_decisive_correct"] += new == gold[index]
        old_final.append(_majority(old_votes)); new_final.append(_majority(new_votes))

    def final_metrics(predictions: list[str]) -> dict[str, Any]:
        decisive = [value in {"A", "B"} for value in predictions]
        correct = [prediction == answer for prediction, answer in zip(predictions, gold)]
        by_gold = {}
        for side in ("A", "B"):
            indices = [index for index, answer in enumerate(gold) if answer == side]
            by_gold[side] = sum(correct[index] for index in indices) / len(indices)
        return {"accuracy": sum(correct) / len(correct), "coverage": sum(decisive) / len(decisive),
                "gold_a_accuracy": by_gold["A"], "gold_b_accuracy": by_gold["B"],
                "counts": dict(Counter(predictions))}

    for name in criteria:
        counts = per_criterion[name]
        old_decisive = sum(value for key, value in counts.items() if key[0] in "AB")
        new_decisive = sum(value for key, value in counts.items() if key[-1] in "AB")
        transition_counts = {key: value for key, value in counts.items() if "->" in key}
        criterion_stats[name] = {"transition": dict(sorted(transition_counts.items())),
                                 "legacy_coverage": old_decisive / len(gold),
                                 "structured_coverage": new_decisive / len(gold),
                                 "legacy_decisive_accuracy": counts["legacy_decisive_correct"] / old_decisive if old_decisive else 0.0,
                                 "structured_decisive_accuracy": counts["structured_decisive_correct"] / new_decisive if new_decisive else 0.0,
                                 "applicability": {token: counts[f"applicable:{token}"] / len(gold)
                                                   for token in ("yes", "no", "uncertain")}}
    report = {
        "artifact_type": "worker_alignment_report", "sample_count": len(gold), "criterion_count": len(criteria),
        "legacy": final_metrics(old_final), "structured": final_metrics(new_final),
        "node_exact_agreement": exact / total,
        "legacy_node_decisive_rate": sum(v for key, v in transition.items() if key[0] in "AB") / total,
        "structured_node_decisive_rate": sum(v for key, v in transition.items() if key[-1] in "AB") / total,
        "node_vote_transition": dict(sorted(transition.items())),
        "parse_failures": parse_failures, "consistency_failures": consistency_failures,
        "sample_transition": dict(Counter(f"{old == answer}->{new == answer}" for old, new, answer in zip(old_final, new_final, gold))),
        "per_criterion": criterion_stats,
    }
    if swap_structured_path is not None:
        swapped = StructuredPredictionOutput.load_json(swap_structured_path)
        original_index = {sample_id: index for index, sample_id in enumerate(structured.sample_ids)}
        by_criterion: dict[str, list[bool]] = {name: [] for name in criteria}
        for swapped_id, swapped_row in zip(swapped.sample_ids, swapped.node_outputs):
            source_id = swapped_id.removesuffix("::swap")
            if source_id not in original_index: raise ValueError("swap artifact sample is absent from original artifact")
            original_row = structured.node_outputs[original_index[source_id]]
            for name in criteria:
                by_criterion[name].append(is_ab_swap_consistent(original_row[name].judgement,
                                                                 swapped_row[name].judgement))
        report["structured_swap_consistency"] = {
            "sample_count": len(swapped.sample_ids),
            "mean_criterion_consistency": sum(sum(values) for values in by_criterion.values()) / sum(len(values) for values in by_criterion.values()),
            "per_criterion": {name: sum(values) / len(values) for name, values in by_criterion.items()},
        }
    return report

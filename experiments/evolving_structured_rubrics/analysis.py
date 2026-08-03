"""Pure Phase 4 metrics, bootstrap, path, and child diagnostics."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence

from critiq.structured import (
    FinalPreference,
    StructuredPredictionOutput,
    StructuredRubric,
    Vote,
    aggregate_child_subtrees,
    aggregate_weighted_root_votes,
    should_visit_child,
)
from critiq.structured.semantics import TraversalPolicy
from critiq.structured.trace import ExecutionTrace, StructuredExecutionOutput


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def preference_is_correct(preference: FinalPreference, gold: str) -> bool:
    return preference in {FinalPreference.A, FinalPreference.B} and preference.value == gold


def summarize_execution(
    execution: StructuredExecutionOutput,
    dataset: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(execution.traces) != len(dataset):
        raise ValueError("execution and dataset lengths differ")
    sample_ids = tuple(str(row.get("sample_id")) for row in dataset)
    if sample_ids != tuple(trace.sample_id for trace in execution.traces):
        raise ValueError("execution and dataset sample order differ")
    correct = [
        preference_is_correct(trace.final_preference, str(row["answer"]))
        for trace, row in zip(execution.traces, dataset)
    ]
    decisive = [
        trace.final_preference in {FinalPreference.A, FinalPreference.B}
        for trace in execution.traces
    ]
    counts = Counter(trace.final_preference.value for trace in execution.traces)
    covered_correct = sum(value and is_decisive for value, is_decisive in zip(correct, decisive))
    covered = sum(decisive)
    visited = [trace.counterfactual_metrics.visited_nodes for trace in execution.traces]
    avoided = [trace.counterfactual_metrics.avoided_nodes for trace in execution.traces]
    selected_roots = [trace.counterfactual_metrics.selected_roots for trace in execution.traces]
    return {
        "sample_count": len(dataset),
        "accuracy": sum(correct) / len(correct),
        "coverage": covered / len(decisive),
        "covered_accuracy": covered_correct / covered if covered else 0.0,
        "tie_rate": 1.0 - covered / len(decisive),
        "preference_counts": dict(sorted(counts.items())),
        "correct": correct,
        "mean_visited_nodes": mean(visited),
        "median_visited_nodes": median(visited),
        "p90_visited_nodes": _percentile(visited, 0.90),
        "mean_avoided_nodes": mean(avoided),
        "mean_selected_roots": mean(selected_roots),
    }


def paired_bootstrap_accuracy_difference(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
    *,
    samples: int = 10_000,
    seed: int = 20_260_731,
) -> dict[str, float | int]:
    if len(baseline_correct) != len(candidate_correct) or not baseline_correct:
        raise ValueError("paired correctness arrays must be non-empty and equal length")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    count = len(baseline_correct)
    observed = mean(candidate_correct) - mean(baseline_correct)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        differences.append(
            mean(candidate_correct[index] for index in indices)
            - mean(baseline_correct[index] for index in indices)
        )
    return {
        "seed": seed,
        "bootstrap_samples": samples,
        "observed_difference": observed,
        "ci95_low": _percentile(differences, 0.025),
        "ci95_high": _percentile(differences, 0.975),
    }


def path_statistics(
    rubric: StructuredRubric,
    execution: StructuredExecutionOutput,
) -> dict[str, Any]:
    node_activation = Counter()
    node_applicable = Counter()
    node_decisive = Counter()
    edge_matched = Counter()
    edge_seen = Counter()
    root_direct_stop = Counter()
    root_selected = Counter()
    path_signatures = Counter()
    conflict_count = 0
    sample_count = len(execution.traces)
    leaf_ids = {
        node_id for node_id in rubric.nodes if not rubric.child_edges(node_id)
    }
    leaf_visits = Counter()

    for trace in execution.traces:
        trace_nodes = {node.node_id: node for node in trace.nodes}
        for node in trace.nodes:
            node_activation[node.node_id] += 1
            if node.output.judgement.applicable.value == "yes":
                node_applicable[node.node_id] += 1
            if node.local_vote in {Vote.A, Vote.B}:
                node_decisive[node.node_id] += 1
            if node.node_id in leaf_ids:
                leaf_visits[node.node_id] += 1
            direct_votes = [node.local_vote]
            for edge in node.edges:
                key = f"{edge.parent_id}->{edge.child_id}"
                edge_seen[key] += 1
                if edge.matched:
                    edge_matched[key] += 1
                if edge.child_subtree_vote is not None:
                    direct_votes.append(edge.child_subtree_vote)
            if Vote.A in direct_votes and Vote.B in direct_votes:
                conflict_count += 1
        for root in trace.roots:
            if root.selected:
                root_selected[root.root_id] += 1
                root_node = trace_nodes.get(root.root_id)
                if root_node is not None and not any(edge.matched for edge in root_node.edges):
                    root_direct_stop[root.root_id] += 1
            signature = f"{root.root_id}:" + ">".join(root.visited_node_ids)
            path_signatures[signature] += 1

    def rates(counter: Counter[str]) -> dict[str, float]:
        return {
            key: counter[key] / sample_count
            for key in sorted(counter)
        }

    return {
        "node_activation_rate": rates(node_activation),
        "node_applicability_rate": rates(node_applicable),
        "node_decisive_rate": rates(node_decisive),
        "edge_match_rate": {
            key: edge_matched[key] / edge_seen[key]
            for key in sorted(edge_seen)
        },
        "root_selection_rate": rates(root_selected),
        "root_direct_stop_rate": {
            root_id: root_direct_stop[root_id] / root_selected[root_id]
            if root_selected[root_id]
            else 0.0
            for root_id in rubric.root_ids
        },
        "leaf_utilization_rate": {
            node_id: leaf_visits[node_id] / sample_count for node_id in sorted(leaf_ids)
        },
        "subtree_conflict_count": conflict_count,
        "path_signatures": dict(path_signatures.most_common()),
    }


def _conditional_subtree_vote(
    rubric: StructuredRubric,
    outputs: Mapping[str, Any],
    node_id: str,
) -> Vote:
    node = rubric.get_node(node_id)
    output = outputs[node.criterion.name]
    children = []
    for edge in rubric.child_edges(node_id):
        if should_visit_child(
            output.judgement,
            edge.condition,
            TraversalPolicy.CONDITIONAL,
        ):
            children.append(
                _conditional_subtree_vote(rubric, outputs, edge.child_id)
            )
    return aggregate_child_subtrees(output.local_decision.vote, children)


def child_diagnostics(
    rubric: StructuredRubric,
    execution: StructuredExecutionOutput,
    prediction: StructuredPredictionOutput,
    dataset: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(trace.sample_id for trace in execution.traces) != prediction.sample_ids:
        raise ValueError("execution and prediction sample IDs differ")
    if len(dataset) != len(execution.traces):
        raise ValueError("dataset length differs from execution")
    totals = Counter()
    by_edge: dict[str, Counter[str]] = defaultdict(Counter)

    for index, (trace, row) in enumerate(zip(execution.traces, dataset)):
        gold = str(row["answer"])
        all_outputs = prediction.node_outputs[index]
        for node in trace.nodes:
            matched_votes = [
                edge.child_subtree_vote
                for edge in node.edges
                if edge.matched and edge.child_subtree_vote is not None
            ]
            actual = aggregate_child_subtrees(node.local_vote, matched_votes)
            for edge in node.edges:
                edge_key = f"{edge.parent_id}->{edge.child_id}"
                if edge.matched:
                    if edge.child_subtree_vote is None:
                        raise ValueError("matched edge is missing child subtree vote")
                    remaining = [
                        other.child_subtree_vote
                        for other in node.edges
                        if other.matched
                        and other.child_id != edge.child_id
                        and other.child_subtree_vote is not None
                    ]
                    without = aggregate_child_subtrees(node.local_vote, remaining)
                    actual_correct = (
                        actual in {Vote.A, Vote.B} and actual.value == gold
                    )
                    without_correct = (
                        without in {Vote.A, Vote.B} and without.value == gold
                    )
                    if actual_correct and not without_correct:
                        label = "child_corrects_parent"
                    elif not actual_correct and without_correct:
                        label = "child_harms_parent"
                    else:
                        label = "child_neutral"
                    child_output = all_outputs[
                        rubric.get_node(edge.child_id).criterion.name
                    ]
                    if child_output.local_decision.vote is Vote.ABSTAIN:
                        totals["executed_but_inapplicable"] += 1
                        by_edge[edge_key]["executed_but_inapplicable"] += 1
                else:
                    counterfactual = _conditional_subtree_vote(
                        rubric,
                        all_outputs,
                        edge.child_id,
                    )
                    if counterfactual in {Vote.A, Vote.B}:
                        label = (
                            "blocked_correct_child"
                            if counterfactual.value == gold
                            else "blocked_wrong_child"
                        )
                    else:
                        label = "blocked_abstaining_child"
                totals[label] += 1
                by_edge[edge_key][label] += 1
    return {
        "totals": dict(sorted(totals.items())),
        "by_edge": {
            edge: dict(sorted(counts.items()))
            for edge, counts in sorted(by_edge.items())
        },
    }


def weighted_root_predictions(
    rubric: StructuredRubric,
    execution: StructuredExecutionOutput,
) -> tuple[FinalPreference, ...]:
    predictions: list[FinalPreference] = []
    for trace in execution.traces:
        selected = tuple(root.root_id for root in trace.roots if root.selected)
        votes = {
            root.root_id: root.subtree_vote
            for root in trace.roots
            if root.selected and root.subtree_vote is not None
        }
        if set(votes) != set(selected):
            raise ValueError("selected root trace is missing a subtree vote")
        weights = {
            root_id: rubric.get_node(root_id).criterion.score
            for root_id in selected
        }
        predictions.append(
            aggregate_weighted_root_votes(votes, selected, weights)
        )
    return tuple(predictions)
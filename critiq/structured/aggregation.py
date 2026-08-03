"""Pure aggregation functions for structured-rubric execution."""

from __future__ import annotations

import math

from collections.abc import Iterable, Mapping, Sequence

from .judgement import FinalPreference, Vote


def _require_vote(value: object, name: str) -> Vote:
    if not isinstance(value, Vote):
        raise TypeError(f"{name} must be Vote")
    return value


def aggregate_child_subtrees(
    parent_vote: Vote,
    child_votes: Iterable[Vote],
) -> Vote:
    """Aggregate direct child-subtree votes into one subtree vote.

    Abstaining children do not vote. A unique A/B majority wins; no decisive
    child or a child tie falls back exactly once to the parent local vote.
    """

    parent_vote = _require_vote(parent_vote, "parent_vote")
    count_a = 0
    count_b = 0
    for index, vote in enumerate(child_votes):
        vote = _require_vote(vote, f"child_votes[{index}]")
        if vote is Vote.A:
            count_a += 1
        elif vote is Vote.B:
            count_b += 1

    if count_a > count_b:
        return Vote.A
    if count_b > count_a:
        return Vote.B
    return parent_vote


def aggregate_selected_roots(
    root_votes: Mapping[str, Vote],
    selected_root_ids: Sequence[str],
) -> FinalPreference:
    """Uniformly aggregate at most one vote from each selected root subtree."""

    if isinstance(selected_root_ids, (str, bytes)):
        raise TypeError("selected_root_ids must be a sequence of root IDs")
    if not isinstance(selected_root_ids, Sequence):
        raise TypeError("selected_root_ids must be an ordered sequence of root IDs")

    selected = tuple(selected_root_ids)
    if not selected:
        raise ValueError("selected_root_ids must not be empty")
    if any(
        not isinstance(root_id, str) or not root_id.strip()
        for root_id in selected
    ):
        raise ValueError("every selected root ID must be a non-empty string")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_root_ids must not contain duplicates")

    count_a = 0
    count_b = 0
    for root_id in selected:
        if root_id not in root_votes:
            raise KeyError(f"missing vote for selected root: {root_id}")
        vote = _require_vote(root_votes[root_id], f"root_votes[{root_id!r}]")
        if vote is Vote.A:
            count_a += 1
        elif vote is Vote.B:
            count_b += 1

    if count_a > count_b:
        return FinalPreference.A
    if count_b > count_a:
        return FinalPreference.B
    return FinalPreference.TIE


def aggregate_flat_votes(votes: Iterable[Vote]) -> FinalPreference:
    """Uniformly aggregate decisive local votes without structural weighting."""

    count_a = 0
    count_b = 0
    for index, vote in enumerate(votes):
        vote = _require_vote(vote, f"votes[{index}]")
        if vote is Vote.A:
            count_a += 1
        elif vote is Vote.B:
            count_b += 1
    if count_a > count_b:
        return FinalPreference.A
    if count_b > count_a:
        return FinalPreference.B
    return FinalPreference.TIE


def aggregate_weighted_root_votes(
    root_votes: Mapping[str, Vote],
    selected_root_ids: Sequence[str],
    weights_by_root: Mapping[str, float],
) -> FinalPreference:
    """Aggregate selected root-subtree votes with frozen non-negative weights.

    This is a Phase 4 offline ablation. Abstaining roots contribute no weight;
    an exact weighted tie or no decisive vote returns ``Tie``.
    """

    if isinstance(selected_root_ids, (str, bytes)):
        raise TypeError("selected_root_ids must be a sequence of root IDs")
    if not isinstance(selected_root_ids, Sequence):
        raise TypeError("selected_root_ids must be an ordered sequence")
    selected = tuple(selected_root_ids)
    if not selected:
        raise ValueError("selected_root_ids must not be empty")
    if any(
        not isinstance(root_id, str) or not root_id.strip()
        for root_id in selected
    ):
        raise ValueError("every selected root ID must be a non-empty string")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_root_ids must not contain duplicates")
    if not isinstance(weights_by_root, Mapping):
        raise TypeError("weights_by_root must be a mapping")
    if set(weights_by_root) != set(selected):
        raise ValueError("weights_by_root keys must equal selected_root_ids")

    normalized_weights: dict[str, float] = {}
    for root_id in selected:
        weight = weights_by_root[root_id]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ValueError("root weights must be finite non-negative numbers")
        normalized_weights[root_id] = float(weight)
    if not any(weight > 0.0 for weight in normalized_weights.values()):
        raise ValueError("at least one selected root weight must be positive")

    weight_a = 0.0
    weight_b = 0.0
    for root_id in selected:
        if root_id not in root_votes:
            raise KeyError(f"missing vote for selected root: {root_id}")
        vote = _require_vote(root_votes[root_id], f"root_votes[{root_id!r}]")
        if vote is Vote.A:
            weight_a += normalized_weights[root_id]
        elif vote is Vote.B:
            weight_b += normalized_weights[root_id]
    if weight_a > weight_b:
        return FinalPreference.A
    if weight_b > weight_a:
        return FinalPreference.B
    return FinalPreference.TIE
"""Edge, traversal, and root-routing semantics for structured rubrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .judgement import (
    CriterionStatus,
    NodeJudgement,
    PairPreference,
    resolve_local_decision,
)


class EdgeCondition(str, Enum):
    ALWAYS = "always"
    PARENT_NONDECISIVE = "parent_nondecisive"
    PARENT_BOTH_PASS = "parent_both_pass"
    PARENT_BOTH_FAIL = "parent_both_fail"


class TraversalPolicy(str, Enum):
    CONDITIONAL = "conditional"
    ALL_NODES = "all_nodes"


def edge_condition_matches(
    judgement: NodeJudgement,
    condition: EdgeCondition,
) -> bool:
    """Evaluate an outgoing edge under Conditional traversal."""

    if not isinstance(condition, EdgeCondition):
        raise TypeError("condition must be EdgeCondition")

    decision = resolve_local_decision(judgement)
    if not decision.conditional_expandable:
        return False

    if condition is EdgeCondition.ALWAYS:
        return True
    if condition is EdgeCondition.PARENT_NONDECISIVE:
        return judgement.pair_preference in (
            PairPreference.TIE,
            PairPreference.UNCERTAIN,
        )
    if condition is EdgeCondition.PARENT_BOTH_PASS:
        return (
            judgement.status_a is CriterionStatus.PASS
            and judgement.status_b is CriterionStatus.PASS
        )
    return (
        judgement.status_a is CriterionStatus.FAIL
        and judgement.status_b is CriterionStatus.FAIL
    )


def should_visit_child(
    judgement: NodeJudgement,
    condition: EdgeCondition,
    policy: TraversalPolicy,
) -> bool:
    """Decide whether one child is visited under a traversal policy."""

    if not isinstance(condition, EdgeCondition):
        raise TypeError("condition must be EdgeCondition")
    if not isinstance(policy, TraversalPolicy):
        raise TypeError("policy must be TraversalPolicy")
    if policy is TraversalPolicy.ALL_NODES:
        return True
    return edge_condition_matches(judgement, condition)


class RoutingSource(str, Enum):
    ROUTER = "router"
    FALLBACK = "fallback"
    DISABLED = "disabled"


class RoutingOutcomeReason(str, Enum):
    VALID = "valid"
    DISABLED = "disabled"
    MISSING_DECISION = "missing_decision"
    PARSE_FAILURE = "parse_failure"
    CONSISTENCY_INVALID = "consistency_invalid"


@dataclass(frozen=True)
class RootRoutingDecision:
    """Normalized output of one joint multi-root router call."""

    selected_root_ids: tuple[str, ...]
    rationale_by_root: Mapping[str, str]
    parse_ok: bool = True


@dataclass(frozen=True)
class ResolvedRootRouting:
    """Validated root selection consumed by the future executor."""

    selected_root_ids: tuple[str, ...]
    source: RoutingSource
    outcome_reason: RoutingOutcomeReason
    consistency_errors: tuple[str, ...] = ()

    @property
    def fallback_to_all_roots(self) -> bool:
        return self.source is RoutingSource.FALLBACK


def _validate_root_ids(root_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(root_ids, (str, bytes)):
        raise TypeError("root_ids must be a sequence of root IDs")
    if not isinstance(root_ids, Sequence):
        raise TypeError("root_ids must be an ordered sequence of root IDs")
    normalized = tuple(root_ids)
    if not normalized:
        raise ValueError("root_ids must not be empty")
    if any(not isinstance(root_id, str) or not root_id.strip() for root_id in normalized):
        raise ValueError("every root ID must be a non-empty string")
    if len(set(normalized)) != len(normalized):
        raise ValueError("root_ids must not contain duplicates")
    return normalized


def root_routing_consistency_errors(
    decision: RootRoutingDecision,
    root_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return deterministic validation errors for a router decision."""

    roots = _validate_root_ids(root_ids)
    root_set = set(roots)
    errors: list[str] = []

    if not isinstance(decision.parse_ok, bool):
        errors.append("parse_ok must be bool")

    if not isinstance(decision.selected_root_ids, tuple):
        errors.append("selected_root_ids must be tuple")
        selected: tuple[object, ...] = ()
    else:
        selected = decision.selected_root_ids

    if not selected:
        errors.append("selected_root_ids must not be empty")
    selected_ids_are_strings = all(
        isinstance(root_id, str) and bool(root_id.strip()) for root_id in selected
    )
    if not selected_ids_are_strings:
        errors.append("every selected root ID must be a non-empty string")
    if selected_ids_are_strings:
        if len(set(selected)) != len(selected):
            errors.append("selected_root_ids must not contain duplicates")
        unknown = {root_id for root_id in selected if root_id not in root_set}
        if unknown:
            errors.append("selected_root_ids contains unknown roots")

    if not isinstance(decision.rationale_by_root, Mapping):
        errors.append("rationale_by_root must be a mapping")
    else:
        rationale_keys = set(decision.rationale_by_root)
        expected_rationale_keys = set(selected) if selected_ids_are_strings else set()
        if rationale_keys != expected_rationale_keys:
            errors.append("rationale_by_root keys must equal selected_root_ids")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in decision.rationale_by_root.values()
        ):
            errors.append("rationale values must be non-empty strings")

    return tuple(errors)


def resolve_root_routing(
    decision: RootRoutingDecision | None,
    root_ids: Sequence[str],
    *,
    enabled: bool,
) -> ResolvedRootRouting:
    """Resolve valid selection or traceably fall back to all roots."""

    roots = _validate_root_ids(root_ids)
    if not enabled:
        return ResolvedRootRouting(
            selected_root_ids=roots,
            source=RoutingSource.DISABLED,
            outcome_reason=RoutingOutcomeReason.DISABLED,
        )

    if decision is None:
        return ResolvedRootRouting(
            selected_root_ids=roots,
            source=RoutingSource.FALLBACK,
            outcome_reason=RoutingOutcomeReason.MISSING_DECISION,
        )

    if decision.parse_ok is False:
        return ResolvedRootRouting(
            selected_root_ids=roots,
            source=RoutingSource.FALLBACK,
            outcome_reason=RoutingOutcomeReason.PARSE_FAILURE,
        )

    errors = root_routing_consistency_errors(decision, roots)
    if errors:
        return ResolvedRootRouting(
            selected_root_ids=roots,
            source=RoutingSource.FALLBACK,
            outcome_reason=RoutingOutcomeReason.CONSISTENCY_INVALID,
            consistency_errors=errors,
        )

    selected_set = set(decision.selected_root_ids)
    selected_in_rubric_order = tuple(
        root_id for root_id in roots if root_id in selected_set
    )
    return ResolvedRootRouting(
        selected_root_ids=selected_in_rubric_order,
        source=RoutingSource.ROUTER,
        outcome_reason=RoutingOutcomeReason.VALID,
    )

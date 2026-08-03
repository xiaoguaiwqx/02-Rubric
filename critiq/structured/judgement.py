"""Deterministic node-judgement semantics for structured rubrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Applicability(str, Enum):
    """Whether a criterion applies to the current pair."""

    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class CriterionStatus(str, Enum):
    """Whether one candidate satisfies a criterion."""

    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class PairPreference(str, Enum):
    """Pairwise preference expressed by one criterion."""

    A = "A"
    B = "B"
    TIE = "tie"
    UNCERTAIN = "uncertain"


class Vote(str, Enum):
    """A node or subtree vote consumed by aggregation."""

    A = "A"
    B = "B"
    ABSTAIN = "abstain"


class FinalPreference(str, Enum):
    """Final preference after selected-root aggregation."""

    A = "A"
    B = "B"
    TIE = "Tie"


class OutcomeReason(str, Enum):
    """Why a node projects to its local vote."""

    DECISIVE = "decisive"
    TIE = "tie"
    PREFERENCE_UNCERTAIN = "preference_uncertain"
    INAPPLICABLE = "inapplicable"
    APPLICABILITY_UNCERTAIN = "applicability_uncertain"
    PARSE_FAILURE = "parse_failure"
    CONSISTENCY_INVALID = "consistency_invalid"


@dataclass(frozen=True)
class NodeJudgement:
    """Normalized structured output for one sample and one criterion.

    consistency_ok is intentionally derived from the semantic fields. It is
    never accepted as caller- or model-provided state.
    """

    applicable: Applicability
    status_a: CriterionStatus
    status_b: CriterionStatus
    pair_preference: PairPreference
    evidence_a: str = ""
    evidence_b: str = ""
    parse_ok: bool = True

    @property
    def consistency_errors(self) -> tuple[str, ...]:
        """Return deterministic cross-field validation errors."""

        errors: list[str] = []
        if not isinstance(self.parse_ok, bool):
            errors.append("parse_ok must be bool")

        enum_fields = (
            ("applicable", self.applicable, Applicability),
            ("status_a", self.status_a, CriterionStatus),
            ("status_b", self.status_b, CriterionStatus),
            ("pair_preference", self.pair_preference, PairPreference),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                errors.append(f"{name} must be {enum_type.__name__}")

        if errors:
            return tuple(errors)

        if self.applicable is not Applicability.YES:
            if self.status_a is not CriterionStatus.UNCERTAIN:
                errors.append("status_a must be uncertain when applicable is not yes")
            if self.status_b is not CriterionStatus.UNCERTAIN:
                errors.append("status_b must be uncertain when applicable is not yes")
            if self.pair_preference is not PairPreference.UNCERTAIN:
                errors.append(
                    "pair_preference must be uncertain when applicable is not yes"
                )
            return tuple(errors)

        if (
            self.status_a is CriterionStatus.PASS
            and self.status_b is CriterionStatus.FAIL
            and self.pair_preference is not PairPreference.A
        ):
            errors.append("pass/fail status requires pair_preference=A")
        elif (
            self.status_a is CriterionStatus.FAIL
            and self.status_b is CriterionStatus.PASS
            and self.pair_preference is not PairPreference.B
        ):
            errors.append("fail/pass status requires pair_preference=B")

        return tuple(errors)

    @property
    def consistency_ok(self) -> bool:
        """Whether all normalized fields satisfy the frozen semantics."""

        return not self.consistency_errors


@dataclass(frozen=True)
class LocalDecision:
    """Runtime projection consumed by aggregation and Conditional traversal."""

    vote: Vote
    outcome_reason: OutcomeReason
    conditional_expandable: bool


def resolve_local_decision(judgement: NodeJudgement) -> LocalDecision:
    """Compile a normalized judgement into deterministic executor state."""

    if judgement.parse_ok is False:
        return LocalDecision(
            vote=Vote.ABSTAIN,
            outcome_reason=OutcomeReason.PARSE_FAILURE,
            conditional_expandable=False,
        )

    if not judgement.consistency_ok:
        return LocalDecision(
            vote=Vote.ABSTAIN,
            outcome_reason=OutcomeReason.CONSISTENCY_INVALID,
            conditional_expandable=False,
        )

    if judgement.applicable is Applicability.NO:
        return LocalDecision(
            vote=Vote.ABSTAIN,
            outcome_reason=OutcomeReason.INAPPLICABLE,
            conditional_expandable=False,
        )

    if judgement.applicable is Applicability.UNCERTAIN:
        return LocalDecision(
            vote=Vote.ABSTAIN,
            outcome_reason=OutcomeReason.APPLICABILITY_UNCERTAIN,
            conditional_expandable=False,
        )

    if judgement.pair_preference is PairPreference.A:
        return LocalDecision(
            vote=Vote.A,
            outcome_reason=OutcomeReason.DECISIVE,
            conditional_expandable=True,
        )

    if judgement.pair_preference is PairPreference.B:
        return LocalDecision(
            vote=Vote.B,
            outcome_reason=OutcomeReason.DECISIVE,
            conditional_expandable=True,
        )

    if judgement.pair_preference is PairPreference.TIE:
        return LocalDecision(
            vote=Vote.ABSTAIN,
            outcome_reason=OutcomeReason.TIE,
            conditional_expandable=True,
        )

    return LocalDecision(
        vote=Vote.ABSTAIN,
        outcome_reason=OutcomeReason.PREFERENCE_UNCERTAIN,
        conditional_expandable=True,
    )

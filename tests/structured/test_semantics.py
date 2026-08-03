"""Phase 0 tests for deterministic structured-rubric semantics."""

from __future__ import annotations

import unittest

from critiq.structured import (
    Applicability,
    CriterionStatus,
    EdgeCondition,
    FinalPreference,
    LocalDecision,
    NodeJudgement,
    OutcomeReason,
    PairPreference,
    RootRoutingDecision,
    RoutingOutcomeReason,
    RoutingSource,
    STRUCTURED_SEMANTICS_VERSION,
    TraversalPolicy,
    Vote,
    aggregate_child_subtrees,
    aggregate_selected_roots,
    edge_condition_matches,
    resolve_local_decision,
    resolve_root_routing,
    should_visit_child,
)


def make_judgement(
    *,
    applicable: Applicability = Applicability.YES,
    status_a: CriterionStatus = CriterionStatus.PASS,
    status_b: CriterionStatus = CriterionStatus.FAIL,
    pair_preference: PairPreference = PairPreference.A,
    parse_ok: bool = True,
) -> NodeJudgement:
    return NodeJudgement(
        applicable=applicable,
        status_a=status_a,
        status_b=status_b,
        pair_preference=pair_preference,
        parse_ok=parse_ok,
    )


class NodeJudgementSemanticsTest(unittest.TestCase):
    def test_consistency_ok_is_computed_not_an_input_field(self) -> None:
        self.assertNotIn("consistency_ok", NodeJudgement.__dataclass_fields__)
        self.assertTrue(make_judgement().consistency_ok)

    def test_decisive_preferences_project_to_votes(self) -> None:
        cases = (
            (CriterionStatus.PASS, CriterionStatus.FAIL, PairPreference.A, Vote.A),
            (CriterionStatus.FAIL, CriterionStatus.PASS, PairPreference.B, Vote.B),
            (CriterionStatus.PASS, CriterionStatus.PASS, PairPreference.A, Vote.A),
            (CriterionStatus.FAIL, CriterionStatus.FAIL, PairPreference.B, Vote.B),
        )
        for status_a, status_b, preference, expected_vote in cases:
            with self.subTest(status_a=status_a, status_b=status_b, preference=preference):
                decision = resolve_local_decision(
                    make_judgement(
                        status_a=status_a,
                        status_b=status_b,
                        pair_preference=preference,
                    )
                )
                self.assertEqual(expected_vote, decision.vote)
                self.assertEqual(OutcomeReason.DECISIVE, decision.outcome_reason)
                self.assertTrue(decision.conditional_expandable)

    def test_tie_and_uncertain_preference_abstain_but_can_expand(self) -> None:
        cases = (
            (PairPreference.TIE, OutcomeReason.TIE),
            (PairPreference.UNCERTAIN, OutcomeReason.PREFERENCE_UNCERTAIN),
        )
        for preference, reason in cases:
            with self.subTest(preference=preference):
                decision = resolve_local_decision(
                    make_judgement(
                        status_a=CriterionStatus.PASS,
                        status_b=CriterionStatus.PASS,
                        pair_preference=preference,
                    )
                )
                self.assertEqual(Vote.ABSTAIN, decision.vote)
                self.assertEqual(reason, decision.outcome_reason)
                self.assertTrue(decision.conditional_expandable)

    def test_inapplicable_and_applicability_uncertain_are_distinct(self) -> None:
        cases = (
            (Applicability.NO, OutcomeReason.INAPPLICABLE),
            (Applicability.UNCERTAIN, OutcomeReason.APPLICABILITY_UNCERTAIN),
        )
        for applicable, reason in cases:
            with self.subTest(applicable=applicable):
                decision = resolve_local_decision(
                    make_judgement(
                        applicable=applicable,
                        status_a=CriterionStatus.UNCERTAIN,
                        status_b=CriterionStatus.UNCERTAIN,
                        pair_preference=PairPreference.UNCERTAIN,
                    )
                )
                self.assertEqual(Vote.ABSTAIN, decision.vote)
                self.assertEqual(reason, decision.outcome_reason)
                self.assertFalse(decision.conditional_expandable)

    def test_parse_failure_has_its_own_projection(self) -> None:
        judgement = make_judgement(parse_ok=False)
        decision = resolve_local_decision(judgement)
        self.assertEqual(Vote.ABSTAIN, decision.vote)
        self.assertEqual(OutcomeReason.PARSE_FAILURE, decision.outcome_reason)
        self.assertFalse(decision.conditional_expandable)

    def test_non_applicable_requires_all_uncertain_fields(self) -> None:
        invalid = make_judgement(
            applicable=Applicability.NO,
            status_a=CriterionStatus.PASS,
            status_b=CriterionStatus.UNCERTAIN,
            pair_preference=PairPreference.UNCERTAIN,
        )
        self.assertFalse(invalid.consistency_ok)
        decision = resolve_local_decision(invalid)
        self.assertEqual(OutcomeReason.CONSISTENCY_INVALID, decision.outcome_reason)
        self.assertEqual(Vote.ABSTAIN, decision.vote)
        self.assertFalse(decision.conditional_expandable)

    def test_asymmetric_status_preference_conflicts_are_invalid(self) -> None:
        cases = (
            (CriterionStatus.PASS, CriterionStatus.FAIL, PairPreference.B),
            (CriterionStatus.PASS, CriterionStatus.FAIL, PairPreference.TIE),
            (CriterionStatus.FAIL, CriterionStatus.PASS, PairPreference.A),
            (CriterionStatus.FAIL, CriterionStatus.PASS, PairPreference.UNCERTAIN),
        )
        for status_a, status_b, preference in cases:
            with self.subTest(status_a=status_a, status_b=status_b, preference=preference):
                judgement = make_judgement(
                    status_a=status_a,
                    status_b=status_b,
                    pair_preference=preference,
                )
                self.assertFalse(judgement.consistency_ok)
                self.assertEqual(
                    OutcomeReason.CONSISTENCY_INVALID,
                    resolve_local_decision(judgement).outcome_reason,
                )

    def test_all_applicability_status_preference_combinations_are_defined(self) -> None:
        for applicable in Applicability:
            for status_a in CriterionStatus:
                for status_b in CriterionStatus:
                    for preference in PairPreference:
                        with self.subTest(
                            applicable=applicable,
                            status_a=status_a,
                            status_b=status_b,
                            preference=preference,
                        ):
                            judgement = make_judgement(
                                applicable=applicable,
                                status_a=status_a,
                                status_b=status_b,
                                pair_preference=preference,
                            )
                            if applicable is not Applicability.YES:
                                expected_valid = (
                                    status_a is CriterionStatus.UNCERTAIN
                                    and status_b is CriterionStatus.UNCERTAIN
                                    and preference is PairPreference.UNCERTAIN
                                )
                            elif (
                                status_a is CriterionStatus.PASS
                                and status_b is CriterionStatus.FAIL
                            ):
                                expected_valid = preference is PairPreference.A
                            elif (
                                status_a is CriterionStatus.FAIL
                                and status_b is CriterionStatus.PASS
                            ):
                                expected_valid = preference is PairPreference.B
                            else:
                                expected_valid = True

                            self.assertEqual(expected_valid, judgement.consistency_ok)
                            decision = resolve_local_decision(judgement)
                            self.assertIsInstance(decision, LocalDecision)
                            if not expected_valid:
                                self.assertEqual(
                                    OutcomeReason.CONSISTENCY_INVALID,
                                    decision.outcome_reason,
                                )


class EdgeAndTraversalSemanticsTest(unittest.TestCase):
    def test_edge_condition_truth_table(self) -> None:
        both_pass_tie = make_judgement(
            status_a=CriterionStatus.PASS,
            status_b=CriterionStatus.PASS,
            pair_preference=PairPreference.TIE,
        )
        both_fail_uncertain = make_judgement(
            status_a=CriterionStatus.FAIL,
            status_b=CriterionStatus.FAIL,
            pair_preference=PairPreference.UNCERTAIN,
        )
        decisive = make_judgement()

        self.assertTrue(edge_condition_matches(both_pass_tie, EdgeCondition.ALWAYS))
        self.assertTrue(
            edge_condition_matches(both_pass_tie, EdgeCondition.PARENT_NONDECISIVE)
        )
        self.assertTrue(
            edge_condition_matches(both_pass_tie, EdgeCondition.PARENT_BOTH_PASS)
        )
        self.assertFalse(
            edge_condition_matches(both_pass_tie, EdgeCondition.PARENT_BOTH_FAIL)
        )

        self.assertTrue(
            edge_condition_matches(
                both_fail_uncertain, EdgeCondition.PARENT_NONDECISIVE
            )
        )
        self.assertTrue(
            edge_condition_matches(both_fail_uncertain, EdgeCondition.PARENT_BOTH_FAIL)
        )
        self.assertFalse(
            edge_condition_matches(decisive, EdgeCondition.PARENT_NONDECISIVE)
        )

    def test_nondecisive_depends_only_on_pair_preference(self) -> None:
        both_pass_a = make_judgement(
            status_a=CriterionStatus.PASS,
            status_b=CriterionStatus.PASS,
            pair_preference=PairPreference.A,
        )
        uncertain_status_tie = make_judgement(
            status_a=CriterionStatus.UNCERTAIN,
            status_b=CriterionStatus.PASS,
            pair_preference=PairPreference.TIE,
        )
        self.assertFalse(
            edge_condition_matches(both_pass_a, EdgeCondition.PARENT_NONDECISIVE)
        )
        self.assertTrue(
            edge_condition_matches(
                uncertain_status_tie, EdgeCondition.PARENT_NONDECISIVE
            )
        )

    def test_conditional_stops_but_all_nodes_ignores_ancestor_failure(self) -> None:
        judgements = (
            make_judgement(parse_ok=False),
            make_judgement(
                applicable=Applicability.NO,
                status_a=CriterionStatus.UNCERTAIN,
                status_b=CriterionStatus.UNCERTAIN,
                pair_preference=PairPreference.UNCERTAIN,
            ),
            make_judgement(
                applicable=Applicability.NO,
                status_a=CriterionStatus.PASS,
                status_b=CriterionStatus.UNCERTAIN,
                pair_preference=PairPreference.UNCERTAIN,
            ),
        )
        for judgement in judgements:
            for condition in EdgeCondition:
                with self.subTest(judgement=judgement, condition=condition):
                    self.assertFalse(
                        should_visit_child(
                            judgement, condition, TraversalPolicy.CONDITIONAL
                        )
                    )
                    self.assertTrue(
                        should_visit_child(
                            judgement, condition, TraversalPolicy.ALL_NODES
                        )
                    )


class AggregationSemanticsTest(unittest.TestCase):
    def test_child_subtree_unique_majority_wins(self) -> None:
        self.assertEqual(
            Vote.A,
            aggregate_child_subtrees(
                Vote.B, [Vote.A, Vote.A, Vote.B, Vote.ABSTAIN]
            ),
        )
        self.assertEqual(
            Vote.B,
            aggregate_child_subtrees(Vote.A, [Vote.B, Vote.B, Vote.A]),
        )

    def test_child_tie_or_no_decisive_vote_falls_back_to_parent(self) -> None:
        cases = (
            (Vote.A, [], Vote.A),
            (Vote.B, [Vote.ABSTAIN, Vote.ABSTAIN], Vote.B),
            (Vote.ABSTAIN, [Vote.A, Vote.B], Vote.ABSTAIN),
            (Vote.B, [Vote.A, Vote.B, Vote.ABSTAIN], Vote.B),
        )
        for parent, children, expected in cases:
            with self.subTest(parent=parent, children=children):
                self.assertEqual(
                    expected, aggregate_child_subtrees(parent, children)
                )

    def test_multiple_children_each_contribute_at_most_one_vote(self) -> None:
        child_subtree_votes = [Vote.A, Vote.B, Vote.A]
        self.assertEqual(
            Vote.A,
            aggregate_child_subtrees(
                parent_vote=Vote.B,
                child_votes=child_subtree_votes,
            ),
        )

    def test_selected_root_aggregation_is_uniform_and_ignores_unselected(self) -> None:
        root_votes = {
            "r1": Vote.A,
            "r2": Vote.B,
            "r3": Vote.B,
            "unselected": Vote.B,
        }
        self.assertEqual(
            FinalPreference.A,
            aggregate_selected_roots(root_votes, ["r1"]),
        )
        self.assertEqual(
            FinalPreference.B,
            aggregate_selected_roots(root_votes, ["r2", "r3"]),
        )
        self.assertEqual(
            FinalPreference.TIE,
            aggregate_selected_roots(root_votes, ["r1", "r2"]),
        )
        self.assertEqual(
            FinalPreference.TIE,
            aggregate_selected_roots({"r1": Vote.ABSTAIN}, ["r1"]),
        )

    def test_aggregation_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            aggregate_child_subtrees(Vote.A, ["A"])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            aggregate_selected_roots({"r1": Vote.A}, "r1")
        with self.assertRaises(TypeError):
            aggregate_selected_roots({"r1": Vote.A}, b"r1")
        with self.assertRaises(ValueError):
            aggregate_selected_roots({}, [])
        for selected in ([""], ["   "], ["r1", 1]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                aggregate_selected_roots(
                    {"r1": Vote.A, 1: Vote.B},  # type: ignore[dict-item]
                    selected,  # type: ignore[arg-type]
                )
        with self.assertRaises(ValueError):
            aggregate_selected_roots({"r1": Vote.A}, ["r1", "r1"])
        with self.assertRaises(KeyError):
            aggregate_selected_roots({}, ["r1"])


class RootRoutingSemanticsTest(unittest.TestCase):
    ROOTS = ("r1", "r2", "r3")

    @staticmethod
    def decision(
        selected: tuple[str, ...] = ("r1",),
        *,
        rationale: object | None = None,
        parse_ok: bool = True,
    ) -> RootRoutingDecision:
        if rationale is None:
            rationale = {root_id: "relevant" for root_id in selected}
        return RootRoutingDecision(
            selected_root_ids=selected,
            rationale_by_root=rationale,  # type: ignore[arg-type]
            parse_ok=parse_ok,
        )

    def test_router_v1_has_one_authoritative_selection_signal(self) -> None:
        self.assertNotIn(
            "confidence_by_root",
            RootRoutingDecision.__dataclass_fields__,
        )

    def test_valid_single_and_multi_selection_use_rubric_order(self) -> None:
        single = resolve_root_routing(
            self.decision(("r2",)), self.ROOTS, enabled=True
        )
        self.assertEqual(("r2",), single.selected_root_ids)
        self.assertEqual(RoutingSource.ROUTER, single.source)
        self.assertEqual(RoutingOutcomeReason.VALID, single.outcome_reason)

        multiple = resolve_root_routing(
            self.decision(("r3", "r1")), self.ROOTS, enabled=True
        )
        self.assertEqual(("r1", "r3"), multiple.selected_root_ids)
        self.assertEqual(RoutingSource.ROUTER, multiple.source)
        self.assertFalse(multiple.fallback_to_all_roots)

    def test_disabled_router_runs_all_roots(self) -> None:
        resolved = resolve_root_routing(None, self.ROOTS, enabled=False)
        self.assertEqual(self.ROOTS, resolved.selected_root_ids)
        self.assertEqual(RoutingSource.DISABLED, resolved.source)
        self.assertEqual(RoutingOutcomeReason.DISABLED, resolved.outcome_reason)

    def test_missing_and_parse_failure_fall_back_with_distinct_reasons(self) -> None:
        missing = resolve_root_routing(None, self.ROOTS, enabled=True)
        parse_failure = resolve_root_routing(
            self.decision(parse_ok=False), self.ROOTS, enabled=True
        )
        self.assertEqual(RoutingOutcomeReason.MISSING_DECISION, missing.outcome_reason)
        self.assertEqual(RoutingOutcomeReason.PARSE_FAILURE, parse_failure.outcome_reason)
        for resolved in (missing, parse_failure):
            self.assertEqual(self.ROOTS, resolved.selected_root_ids)
            self.assertEqual(RoutingSource.FALLBACK, resolved.source)
            self.assertTrue(resolved.fallback_to_all_roots)

    def test_all_invalid_router_decisions_fall_back_to_all_roots(self) -> None:
        invalid_decisions = (
            self.decision(()),
            self.decision(("r1", "r1")),
            self.decision(("unknown",)),
            RootRoutingDecision(  # type: ignore[arg-type]
                selected_root_ids=(["r1"],),
                rationale_by_root={},
            ),
            RootRoutingDecision(  # type: ignore[arg-type]
                selected_root_ids=["r1"],
                rationale_by_root={"r1": "relevant"},
            ),
            self.decision(rationale={}),
            self.decision(rationale={"r1": "   "}),
            self.decision(rationale={"r1": "ok", "r2": "extra"}),
            self.decision(parse_ok=1),  # type: ignore[arg-type]
        )
        for decision in invalid_decisions:
            with self.subTest(decision=decision):
                resolved = resolve_root_routing(decision, self.ROOTS, enabled=True)
                self.assertEqual(self.ROOTS, resolved.selected_root_ids)
                self.assertEqual(RoutingSource.FALLBACK, resolved.source)
                self.assertEqual(
                    RoutingOutcomeReason.CONSISTENCY_INVALID,
                    resolved.outcome_reason,
                )
                self.assertTrue(resolved.consistency_errors)

    def test_invalid_root_universe_is_rejected(self) -> None:
        for root_ids in ((), ("r1", "r1"), ("", "r2")):
            with self.subTest(root_ids=root_ids), self.assertRaises(ValueError):
                resolve_root_routing(None, root_ids, enabled=False)
        for root_ids in ("r1", {"r1", "r2"}):
            with self.subTest(root_ids=root_ids), self.assertRaises(TypeError):
                resolve_root_routing(None, root_ids, enabled=False)


class StructuredApiSemanticsTest(unittest.TestCase):
    def test_semantics_version_is_frozen_and_public(self) -> None:
        self.assertEqual("1.0.0", STRUCTURED_SEMANTICS_VERSION)


if __name__ == "__main__":
    unittest.main()

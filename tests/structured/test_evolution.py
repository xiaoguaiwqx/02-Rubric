"""Phase 5 evolution-core semantics and Init Rubric tests."""

from __future__ import annotations

import unittest

from critiq.structured import (
    CandidateAcceptancePolicy,
    DualWorkerRequestSpec,
    EdgeCondition,
    EditCandidate,
    EvolutionContext,
    EvolutionDecision,
    FinalPreference,
    OperatorKind,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    RubricPatch,
    StructuredCriterionSnapshot,
    StructuredRubric,
    Vote,
    apply_rubric_patch,
    diff_rubrics,
    evaluate_candidate,
    extract_rubric_feedback,
    plan_artifact_refresh,
    structured_input_fingerprint,
)
from experiments.evolving_structured_rubrics.rubric_factory import (
    MULTICRIT_OPEN_ENDED_CRITERIA,
    build_multicrit_open_ended_init_rubric,
)
from experiments.evolving_structured_rubrics.run_rubric_evolution import (
    PHASE5_EVOLUTION_POLICY_V2,
    _cluster_capacity,
    _model_rows,
)


def _valid(vote: Vote, thought: str = "t") -> PairwiseVoteOutput:
    return PairwiseVoteOutput(vote, True, "raw", None, 1, thought, True)


def _parse_invalid() -> PairwiseVoteOutput:
    return PairwiseVoteOutput(Vote.ABSTAIN, False, "bad", "parse", 1, None, False)


def _answer_invalid() -> PairwiseVoteOutput:
    return PairwiseVoteOutput(Vote.ABSTAIN, True, "raw", None, 1, "bad token", False)


class InitRubricTests(unittest.TestCase):
    def test_multicrit_init_is_exact_deterministic_five_root_forest(self):
        first = build_multicrit_open_ended_init_rubric()
        second = build_multicrit_open_ended_init_rubric()
        self.assertEqual(first.rubric_sha256, second.rubric_sha256)
        self.assertEqual((5, 5, 0), (len(first.nodes), len(first.root_ids), len(first.edges)))
        self.assertEqual(
            tuple(item[0] for item in MULTICRIT_OPEN_ENDED_CRITERIA),
            tuple(first.get_node(node_id).criterion.name for node_id in first.root_ids),
        )
        for node in first.nodes.values():
            self.assertEqual(node.criterion.score, 1.0)
            self.assertEqual(node.examples, ())
            self.assertEqual(node.lineage["task_family"], "open_ended_generation")

    def test_online_model_rows_never_expose_gold_answer(self):
        original = ({"sample_id": "s1", "question": "q", "A": "a", "B": "b", "answer": "A"},)
        model_rows = _model_rows(original)
        self.assertNotIn("answer", model_rows[0])
        self.assertEqual(original[0]["answer"], "A")

    def test_phase5_policy_and_cluster_capacity_are_frozen_to_five(self):
        thresholds = PHASE5_EVOLUTION_POLICY_V2["trigger_thresholds"]
        self.assertEqual(thresholds["max_children"], 5)
        self.assertEqual(PHASE5_EVOLUTION_POLICY_V2["p05_execution"]["replicates"], 1)
        self.assertNotIn("max_coverage_drop", PHASE5_EVOLUTION_POLICY_V2["candidate_acceptance"])
        capacity = _cluster_capacity({"n": {"wrong": 27}}, thresholds)["n"]
        self.assertEqual(capacity["max_equal_size_for_2_clusters"], 13)
        self.assertEqual(capacity["max_equal_size_for_5_clusters"], 5)
        self.assertEqual(capacity["max_supported_clusters"], 5)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.rubric = StructuredRubric(
            nodes={
                "n1": RubricNode("n1", RubricCriterionSnapshot("c1", "first criterion", 1.0)),
                "n2": RubricNode("n2", RubricCriterionSnapshot("c2", "second criterion", 1.0)),
            },
            edges=(),
            root_ids=("n1", "n2"),
        )
        self.rows = (
            {"sample_id": "s1", "answer": "A", "image_path": "https://example/s1.jpg", "question": "q1", "A": "a1", "B": "b1"},
            {"sample_id": "s2", "answer": "B", "image_path": "https://example/s2.jpg", "question": "q2", "A": "a2", "B": "b2"},
            {"sample_id": "s3", "answer": "A", "image_path": "https://example/s3.jpg", "question": "q3", "A": "a3", "B": "b3"},
            {"sample_id": "s4", "answer": "A", "image_path": "https://example/s4.jpg", "question": "q4", "A": "a4", "B": "b4"},
        )
        outputs = (
            {"c1": _valid(Vote.A), "c2": _valid(Vote.B)},
            {"c1": _valid(Vote.B), "c2": _valid(Vote.ABSTAIN)},
            {"c1": _valid(Vote.B), "c2": _parse_invalid()},
            {"c1": _valid(Vote.A), "c2": _answer_invalid()},
        )
        self.spec = DualWorkerRequestSpec("m", "b", "0" * 64, None, False,
                                          "image_path", "question", "sample_id", {"temperature": 0.5})
        fingerprints = tuple(structured_input_fingerprint(
            row, image_field="image_path", question_field="question",
            sample_id_field="sample_id", encode_local_image=False,
        ) for row in self.rows)
        self.prediction = PairwisePredictionOutput(
            tuple(row["sample_id"] for row in self.rows),
            fingerprints,
            (StructuredCriterionSnapshot("c1", "first criterion"),
             StructuredCriterionSnapshot("c2", "second criterion")),
            outputs,
            (FinalPreference.TIE, FinalPreference.B, FinalPreference.B, FinalPreference.A),
            self.spec,
        )

    def test_all_node_gap_and_failure_types_are_distinct(self):
        feedback = extract_rubric_feedback(
            self.rubric,
            self.prediction,
            self.rows,
            (FinalPreference.B, FinalPreference.B, FinalPreference.TIE, FinalPreference.A),
            expected_request_spec=self.spec,
        )
        n1 = feedback.nodes["n1"]
        self.assertEqual((4, 3, 1), (n1.support, n1.correct, n1.wrong))
        n2 = feedback.nodes["n2"]
        self.assertEqual((1, 1, 1, 1), (n2.wrong, n2.abstain, n2.parse_invalid, n2.answer_invalid))
        self.assertEqual(feedback.rubric_gap_sample_ids, ("s3",))
        self.assertEqual(feedback.cascade_failure_sample_ids, ("s1", "s3"))
        self.assertEqual(feedback.aggregation_conflict_sample_ids, ("s1",))


class PatchAndDecisionTests(unittest.TestCase):
    def setUp(self):
        self.before = StructuredRubric(
            nodes={"r": RubricNode("r", RubricCriterionSnapshot("root", "root text", 1.0))},
            edges=(),
            root_ids=("r",),
        )

    def test_description_refresh_and_stale_hash(self):
        replacement = RubricNode("r", RubricCriterionSnapshot("root", "refined text", 1.0))
        patch = RubricPatch(self.before.rubric_sha256, upsert_nodes=(replacement,))
        after = apply_rubric_patch(self.before, patch)
        self.assertEqual(self.before.get_node("r").criterion.description, "root text")
        diff = diff_rubrics(self.before, after)
        self.assertEqual(diff.modified_node_ids, ("r",))
        refresh = plan_artifact_refresh(self.before, after)
        self.assertEqual(refresh.refresh_pairwise_node_ids, ("r",))
        self.assertTrue(refresh.router_stale)
        with self.assertRaises(ValueError):
            apply_rubric_patch(after, patch)

    def test_always_child_refreshes_only_child_pairwise(self):
        child = RubricNode("c", RubricCriterionSnapshot("child", "child text", 1.0))
        patch = RubricPatch(
            self.before.rubric_sha256,
            upsert_nodes=(child,),
            add_edges=(RubricEdge("r", "c", EdgeCondition.ALWAYS),),
            root_ids=("r",),
        )
        after = apply_rubric_patch(self.before, patch)
        refresh = plan_artifact_refresh(self.before, after)
        self.assertEqual(refresh.refresh_pairwise_node_ids, ("c",))
        self.assertEqual(refresh.refresh_gate_node_ids, ())
        self.assertFalse(refresh.router_stale)
        self.assertEqual(self.before.edges, ())

    def test_noop_and_invalid_cycle(self):
        after = apply_rubric_patch(self.before, RubricPatch(self.before.rubric_sha256))
        self.assertTrue(diff_rubrics(self.before, after).is_noop)
        child = RubricNode("c", RubricCriterionSnapshot("child", "child text", 1.0))
        with self.assertRaises(Exception):
            apply_rubric_patch(
                self.before,
                RubricPatch(
                    self.before.rubric_sha256,
                    upsert_nodes=(child,),
                    add_edges=(
                        RubricEdge("r", "c", EdgeCondition.ALWAYS),
                        RubricEdge("c", "r", EdgeCondition.ALWAYS),
                    ),
                    root_ids=(),
                ),
            )

    def test_candidate_acceptance_requires_explicit_policy(self):
        policy = CandidateAcceptancePolicy(0.25, 0.95)
        accepted = evaluate_candidate(
            (FinalPreference.B, FinalPreference.A, FinalPreference.TIE, FinalPreference.A),
            (FinalPreference.A, FinalPreference.A, FinalPreference.B, FinalPreference.A),
            ("A", "A", "B", "A"),
            before_valid_rate=1.0,
            final_valid_rate=1.0,
            policy=policy,
        )
        self.assertIs(accepted.decision, EvolutionDecision.ACCEPT)
        rejected = evaluate_candidate(
            (FinalPreference.A,), (FinalPreference.B,), ("A",),
            before_valid_rate=1.0,
            final_valid_rate=1.0,
            policy=policy,
        )
        self.assertIs(rejected.decision, EvolutionDecision.REJECT)

    def test_evolution_core_json_round_trip_is_strict(self):
        replacement = RubricNode("r", RubricCriterionSnapshot("root", "refined text", 1.0))
        patch = RubricPatch(self.before.rubric_sha256, upsert_nodes=(replacement,))
        candidate = EditCandidate("candidate-1", OperatorKind.REFINE, "narrow the wording", patch)
        after = apply_rubric_patch(self.before, patch)
        diff = diff_rubrics(self.before, after)
        refresh = plan_artifact_refresh(self.before, after)
        policy = CandidateAcceptancePolicy(2 / 90, 0.95)
        evaluation = evaluate_candidate(
            (FinalPreference.B, FinalPreference.TIE),
            (FinalPreference.A, FinalPreference.A),
            ("A", "A"),
            before_valid_rate=1.0,
            final_valid_rate=1.0,
            policy=policy,
        )
        for value, cls in (
            (patch, RubricPatch),
            (candidate, EditCandidate),
            (diff, type(diff)),
            (refresh, type(refresh)),
            (policy, CandidateAcceptancePolicy),
            (evaluation, type(evaluation)),
        ):
            self.assertEqual(value, cls.from_dict(value.to_dict()))
            damaged = value.to_dict()
            damaged["schema_version"] = "0.0.0"
            with self.assertRaises(ValueError):
                cls.from_dict(damaged)

    def test_feedback_rejects_fingerprint_and_request_identity_mismatch(self):
        fixture = FeedbackTests()
        fixture.setUp()
        m1 = (FinalPreference.B, FinalPreference.B, FinalPreference.TIE, FinalPreference.A)
        feedback = extract_rubric_feedback(
            fixture.rubric, fixture.prediction, fixture.rows, m1,
            expected_request_spec=fixture.spec,
        )
        self.assertEqual(feedback, type(feedback).from_dict(feedback.to_dict()))
        context = EvolutionContext(fixture.rubric, feedback)
        self.assertEqual(context, EvolutionContext.from_dict(context.to_dict()))
        changed = list(dict(row) for row in fixture.rows)
        changed[0]["A"] = "changed candidate"
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            extract_rubric_feedback(
                fixture.rubric, fixture.prediction, changed, m1,
                expected_request_spec=fixture.spec,
            )
        wrong_spec = DualWorkerRequestSpec(
            "m", "other-backend", "0" * 64, None, False,
            "image_path", "question", "sample_id", {"temperature": 0.5},
        )
        with self.assertRaisesRegex(ValueError, "request identity"):
            extract_rubric_feedback(
                fixture.rubric, fixture.prediction, fixture.rows, m1,
                expected_request_spec=wrong_spec,
            )


if __name__ == "__main__":
    unittest.main()

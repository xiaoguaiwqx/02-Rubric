"""Tests for forest topology and Phase 1 artifact compatibility."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from critiq.structured import (
    Applicability,
    CriterionStatus,
    EdgeCondition,
    FinalPreference,
    NodeJudgement,
    PairPreference,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    RubricValidationError,
    StructuredCriterionSnapshot,
    StructuredNodeOutput,
    StructuredPredictionOutput,
    StructuredRubric,
    StructuredWorkerRequestSpec,
    assert_prediction_compatible,
)


def node(node_id: str, criterion_name: str | None = None, description: str | None = None):
    name = criterion_name or f"criterion_{node_id}"
    return RubricNode(
        node_id,
        RubricCriterionSnapshot(
            name,
            description or f"description for {name}",
            0.5,
        ),
    )


def rubric(nodes, edges=(), roots=("root",)) -> StructuredRubric:
    return StructuredRubric(
        nodes={value.node_id: value for value in nodes},
        edges=tuple(edges),
        root_ids=tuple(roots),
    )


def edge(parent: str, child: str, condition=EdgeCondition.ALWAYS) -> RubricEdge:
    return RubricEdge(parent, child, condition)


class ValidForestTest(unittest.TestCase):
    def test_single_node_chain_branching_and_multiple_roots(self) -> None:
        single = rubric([node("root")])
        self.assertEqual(("root",), single.preorder_node_ids())

        chain = rubric(
            [node("root"), node("middle"), node("leaf")],
            [edge("root", "middle"), edge("middle", "leaf")],
        )
        self.assertEqual(("root", "middle", "leaf"), chain.preorder_node_ids())

        branching = rubric(
            [node("root"), node("b"), node("a")],
            [edge("root", "b"), edge("root", "a")],
        )
        self.assertEqual(("root", "a", "b"), branching.preorder_node_ids())

        forest = rubric(
            [node("r1"), node("c1"), node("r2")],
            [edge("r1", "c1")],
            roots=("r2", "r1"),
        )
        self.assertEqual(("r2", "r1", "c1"), forest.preorder_node_ids())

    def test_node_id_and_criterion_name_are_independent_but_mapped(self) -> None:
        value = node("stable-node-id", "human_facing_criterion")
        structured = rubric([value], roots=("stable-node-id",))
        self.assertEqual(
            "human_facing_criterion",
            structured.criterion_name_for("stable-node-id"),
        )
        self.assertEqual(
            "stable-node-id",
            structured.node_id_for_criterion("human_facing_criterion"),
        )


class InvalidForestTest(unittest.TestCase):
    def assert_invalid(self, nodes, edges=(), roots=("root",)) -> None:
        with self.assertRaises(RubricValidationError):
            rubric(nodes, edges, roots)

    def test_empty_duplicate_identity_and_mapping_mismatch(self) -> None:
        self.assert_invalid([], roots=())
        with self.assertRaises(RubricValidationError):
            StructuredRubric({"wrong": node("actual")}, (), ("actual",))
        self.assert_invalid(
            [node("root", "same"), node("other", "same")],
            roots=("root", "other"),
        )
        self.assert_invalid([node("root")], roots=("root", "root"))

    def test_unknown_nodes_self_edges_and_duplicate_edges(self) -> None:
        self.assert_invalid([node("root")], [edge("missing", "root")])
        self.assert_invalid([node("root")], [edge("root", "missing")])
        self.assert_invalid([node("root")], [edge("root", "root")])
        self.assert_invalid(
            [node("root"), node("child")],
            [edge("root", "child"), edge("root", "child")],
        )
        self.assert_invalid([node("root")], roots=("missing",))

    def test_multiple_parents_and_cross_root_sharing_are_rejected(self) -> None:
        self.assert_invalid(
            [node("r1"), node("r2"), node("child")],
            [edge("r1", "child"), edge("r2", "child")],
            roots=("r1", "r2"),
        )
        self.assert_invalid(
            [node("root"), node("p2"), node("child")],
            [edge("root", "child"), edge("p2", "child")],
            roots=("root", "p2"),
        )

    def test_cycles_and_incorrect_root_declarations_are_rejected(self) -> None:
        self.assert_invalid(
            [node("a"), node("b")],
            [edge("a", "b"), edge("b", "a")],
            roots=("a",),
        )
        self.assert_invalid(
            [node("a"), node("b"), node("c")],
            [edge("a", "b"), edge("b", "c"), edge("c", "a")],
            roots=("a",),
        )
        self.assert_invalid(
            [node("root"), node("child")],
            [edge("root", "child")],
            roots=("root", "child"),
        )
        self.assert_invalid(
            [node("r1"), node("r2")],
            roots=("r1",),
        )


def decisive_output() -> StructuredNodeOutput:
    judgement = NodeJudgement(
        applicable=Applicability.YES,
        status_a=CriterionStatus.PASS,
        status_b=CriterionStatus.FAIL,
        pair_preference=PairPreference.A,
        evidence_a="supported",
        evidence_b="contradicted",
    )
    return StructuredNodeOutput(
        judgement=judgement,
        raw_response="valid response",
        parse_error=None,
        attempt_count=1,
    )


def prediction_for(criteria) -> StructuredPredictionOutput:
    snapshots = tuple(
        StructuredCriterionSnapshot(name=name, description=description)
        for name, description in criteria
    )
    outputs = {snapshot.name: decisive_output() for snapshot in snapshots}
    return StructuredPredictionOutput(
        sample_ids=("sample-1",),
        sample_fingerprints=("a" * 64,),
        criteria=snapshots,
        node_outputs=(outputs,),
        flat_answers=(FinalPreference.A,),
        request_spec=StructuredWorkerRequestSpec(
            model="fake-model",
            worker_backend_id="fake-backend@revision-1",
            prompt_sha256="b" * 64,
            max_data_chars=None,
            encode_local_image=False,
            image_field="image_path",
            question_field="question",
            sample_id_field="sample_id",
            decoding_config={"temperature": 0},
        ),
    )


class PredictionCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rubric = rubric(
            [
                node("node-a", "criterion_a", "description a"),
                node("node-b", "criterion_b", "description b"),
            ],
            roots=("node-a", "node-b"),
        )

    def test_exact_coverage_accepts_different_criterion_order(self) -> None:
        prediction = prediction_for(
            (("criterion_b", "description b"), ("criterion_a", "description a"))
        )
        self.assertIsNone(assert_prediction_compatible(self.rubric, prediction))
        self.assertEqual("node-a", self.rubric.node_id_for_criterion("criterion_a"))

    def test_missing_extra_and_description_mismatch_are_rejected(self) -> None:
        cases = (
            prediction_for((("criterion_a", "description a"),)),
            prediction_for(
                (
                    ("criterion_a", "description a"),
                    ("criterion_b", "description b"),
                    ("extra", "extra description"),
                )
            ),
            prediction_for(
                (
                    ("criterion_a", "changed"),
                    ("criterion_b", "description b"),
                )
            ),
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(
                RubricValidationError
            ):
                assert_prediction_compatible(self.rubric, value)

    def test_compatibility_check_never_creates_agent(self) -> None:
        prediction = prediction_for(
            (("criterion_a", "description a"), ("criterion_b", "description b"))
        )
        with patch(
            "critiq.evaluator.Agent",
            side_effect=AssertionError("compatibility check must not create Agent"),
        ):
            assert_prediction_compatible(self.rubric, prediction)

    def test_type_contracts_are_explicit(self) -> None:
        with self.assertRaises(TypeError):
            assert_prediction_compatible("not-rubric", object())
        with self.assertRaises(TypeError):
            assert_prediction_compatible(self.rubric, object())


if __name__ == "__main__":
    unittest.main()

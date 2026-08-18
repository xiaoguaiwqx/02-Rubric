"""Offline tests for the five-root Full-Rubric Child-Gate experiment."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from critiq.structured import (
    EdgeCondition,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredRubric,
)
from critiq.structured.judgement import Vote
from experiments.evolving_structured_rubrics import full_child_gate as full
from experiments.evolving_structured_rubrics import visual_gate as visual


def _description(label: str) -> str:
    return (
        f"Criterion focus: Focus {label}.\n\n"
        f"Applicable only when: Apply {label}.\n\n"
        f"Not applicable when: Exclude {label}.\n\n"
        f"Decision rule: Decide {label}."
    )


def _node(node_id: str, name: str, description: str) -> RubricNode:
    return RubricNode(
        node_id=node_id,
        criterion=RubricCriterionSnapshot(name, description, 1.0),
    )


def _rubric() -> StructuredRubric:
    roots = (
        _node("r1", "root_one", "Root one."),
        _node("r2", "root_two", "Root two."),
    )
    children = (
        _node("r1_c1", "r1_child_one", _description("r1 one")),
        _node("r1_c2", "r1_child_two", _description("r1 two")),
        _node("r2_c1", "r2_child_one", _description("r2 one")),
    )
    nodes = {item.node_id: item for item in (*roots, *children)}
    edges = (
        RubricEdge("r1", "r1_c1", EdgeCondition.ALWAYS),
        RubricEdge("r1", "r1_c2", EdgeCondition.ALWAYS),
        RubricEdge("r2", "r2_c1", EdgeCondition.ALWAYS),
    )
    return StructuredRubric(nodes=nodes, edges=edges, root_ids=("r1", "r2"))


class FullChildGateContractTest(unittest.TestCase):
    def test_visual_contract_builder_supports_arbitrary_root(self):
        rubric = _rubric()
        first = visual.build_routing_contract(
            rubric, parent_node_id="r1", expected_child_names=None)
        second = visual.build_routing_contract(
            rubric, parent_node_id="r2", expected_child_names=None)
        self.assertEqual(["r1_c1", "r1_c2"], [item["node_id"] for item in first["children"]])
        self.assertEqual(["r2_c1"], [item["node_id"] for item in second["children"]])

    def test_constant_selection_keeps_all_roots_present(self):
        rubric = _rubric()
        parent = full._constant_selection(rubric, 2, all_children=False)
        all_children = full._constant_selection(rubric, 2, all_children=True)
        self.assertEqual({"r1": (), "r2": ()}, parent[0])
        self.assertEqual(("r1_c1", "r1_c2"), all_children[0]["r1"])
        self.assertEqual(("r2_c1",), all_children[0]["r2"])

    def test_gate_selection_is_isolated_by_root(self):
        gate = {
            "sample_count": 1,
            "roots": {
                "r1": {"samples": [{"active_child_ids": ["r1_c2"]}]},
                "r2": {"samples": [{"active_child_ids": []}]},
            },
        }
        selected = full._selection_from_gate(gate, ("r1", "r2"))
        self.assertEqual({"r1": ("r1_c2",), "r2": ()}, selected[0])


class FullChildGateAggregationTest(unittest.TestCase):
    def setUp(self):
        self.rubric = _rubric()
        self.outputs = {
            "root_one": SimpleNamespace(vote=Vote.B),
            "r1_child_one": SimpleNamespace(vote=Vote.A),
            "r1_child_two": SimpleNamespace(vote=Vote.A),
            "root_two": SimpleNamespace(vote=Vote.A),
            "r2_child_one": SimpleNamespace(vote=Vote.B),
        }

    def test_no_active_child_falls_back_to_its_parent(self):
        self.assertEqual(Vote.B, full._root_vote(self.rubric, self.outputs, "r1", ()))
        self.assertEqual(Vote.A, full._root_vote(self.rubric, self.outputs, "r2", ()))

    def test_active_children_change_only_the_selected_root(self):
        self.assertEqual(
            Vote.A, full._root_vote(self.rubric, self.outputs, "r1", ("r1_c1", "r1_c2")))
        self.assertEqual(Vote.A, full._root_vote(self.rubric, self.outputs, "r2", ()))


class FullChildGatePromptTest(unittest.TestCase):
    def test_v4_user_prompt_always_contains_format_reminder(self):
        text = visual._user_prompt(
            {"question": "q", "A": "a", "B": "b"}, format_reminder=True)
        self.assertTrue(text.endswith("never output a bare status string."))


if __name__ == "__main__":
    unittest.main()

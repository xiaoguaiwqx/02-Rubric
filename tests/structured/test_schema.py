"""Tests for immutable structured-rubric schema and serialization."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

from critiq.structured import (
    EdgeCondition,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    RubricSchemaError,
    StructuredRubric,
)
from critiq.structured.version import STRUCTURED_RUBRIC_SCHEMA_VERSION
from critiq.utils import Criterion


def node(
    node_id: str,
    *,
    criterion_name: str | None = None,
    description: str | None = None,
    score: float = 0.5,
    examples=(),
    lineage=None,
) -> RubricNode:
    name = criterion_name or f"criterion_{node_id}"
    return RubricNode(
        node_id=node_id,
        criterion=RubricCriterionSnapshot(
            name=name,
            description=description or f"description for {name}",
            score=score,
        ),
        examples=examples,
        lineage=lineage,
    )


def branching_rubric(*, reverse_storage: bool = False) -> StructuredRubric:
    values = [node("root"), node("z-child"), node("a-child"), node("other")]
    edges = [
        RubricEdge("root", "z-child", EdgeCondition.PARENT_NONDECISIVE),
        RubricEdge("root", "a-child", EdgeCondition.ALWAYS),
    ]
    if reverse_storage:
        values.reverse()
        edges.reverse()
    return StructuredRubric(
        nodes={value.node_id: value for value in values},
        edges=tuple(edges),
        root_ids=("other", "root"),
    )


class RubricCriterionSchemaTest(unittest.TestCase):
    def test_criterion_snapshot_round_trip_returns_fresh_criterion(self) -> None:
        original = Criterion("grounding", "visible evidence", 0.75)
        snapshot = RubricCriterionSnapshot.from_criterion(original)
        restored = snapshot.to_criterion()

        self.assertEqual(original.to_dict(), restored.to_dict())
        self.assertIsNot(original, restored)
        original.description = "mutated"
        self.assertEqual("visible evidence", snapshot.description)
        restored.description = "also mutated"
        self.assertEqual("visible evidence", snapshot.description)
        self.assertEqual(
            {"name": "grounding", "description": "mutated", "score": 0.75},
            original.to_dict(),
        )
        self.assertEqual(
            Criterion("grounding", "visible evidence", 0.75),
            Criterion.from_dict(
                {"name": "grounding", "description": "visible evidence", "score": 0.75}
            ),
        )

    def test_criterion_validation_is_strict(self) -> None:
        invalid = (
            {"name": "", "description": "d", "score": 0.5},
            {"name": " x", "description": "d", "score": 0.5},
            {"name": "x", "description": " ", "score": 0.5},
            {"name": "x", "description": "d", "score": True},
            {"name": "x", "description": "d", "score": -0.1},
            {"name": "x", "description": "d", "score": 1.1},
            {"name": "x", "description": "d", "score": math.inf},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(RubricSchemaError):
                RubricCriterionSnapshot(**kwargs)


class RubricImmutabilityAndQueryTest(unittest.TestCase):
    def test_nested_metadata_and_containers_are_defensively_frozen(self) -> None:
        example = {"sample_id": "s1", "tags": ["a", {"weight": 1}]}
        lineage = {
            "unsupported_multi_parent_dependency": ["p1", "p2"],
            "metadata": {"round": 0},
        }
        mutable_nodes = {
            "root": node(
                "root",
                examples=[example],
                lineage=lineage,
            )
        }
        mutable_edges: list[RubricEdge] = []
        mutable_roots = ["root"]
        rubric = StructuredRubric(mutable_nodes, mutable_edges, mutable_roots)

        example["tags"].append("later")
        lineage["unsupported_multi_parent_dependency"].append("p3")
        mutable_nodes.clear()
        mutable_roots.clear()
        self.assertEqual(("root",), rubric.root_ids)
        self.assertEqual(("a", {"weight": 1}), rubric.get_node("root").examples[0]["tags"])
        self.assertEqual(
            ("p1", "p2"),
            rubric.get_node("root").lineage[
                "unsupported_multi_parent_dependency"
            ],
        )

        with self.assertRaises(TypeError):
            rubric.nodes["new"] = node("new")
        with self.assertRaises(TypeError):
            rubric.get_node("root").lineage["new"] = 1
        with self.assertRaises((AttributeError, TypeError)):
            rubric.get_node("root").examples[0]["tags"].append("x")

    def test_deterministic_queries_use_root_then_child_id_order(self) -> None:
        rubric = branching_rubric()
        self.assertEqual(("other", "root", "a-child", "z-child"), rubric.preorder_node_ids())
        self.assertEqual(
            ("a-child", "z-child"),
            tuple(edge.child_id for edge in rubric.child_edges("root")),
        )
        self.assertEqual(
            ("a-child", "z-child"),
            tuple(value.node_id for value in rubric.children("root")),
        )
        self.assertIsNone(rubric.parent_id("root"))
        self.assertEqual("root", rubric.parent_id("a-child"))
        self.assertEqual("root", rubric.root_id_for("z-child"))
        self.assertEqual("other", rubric.root_id_for("other"))
        self.assertEqual(0, rubric.depth("root"))
        self.assertEqual(1, rubric.depth("a-child"))
        self.assertEqual("criterion_a-child", rubric.criterion_name_for("a-child"))
        self.assertEqual("a-child", rubric.node_id_for_criterion("criterion_a-child"))

        criteria = rubric.criteria_in_execution_order()
        self.assertEqual(
            tuple(rubric.criterion_name_for(node_id) for node_id in rubric.preorder_node_ids()),
            tuple(value.name for value in criteria),
        )
        criteria[0].description = "changed"
        self.assertNotEqual("changed", rubric.get_node("other").criterion.description)

        for query in (
            rubric.get_node,
            rubric.parent_id,
            rubric.child_edges,
            rubric.root_id_for,
            rubric.depth,
            rubric.criterion_name_for,
        ):
            with self.subTest(query=query), self.assertRaises(KeyError):
                query("missing")
        with self.assertRaises(KeyError):
            rubric.node_id_for_criterion("missing")


class RubricSerializationTest(unittest.TestCase):
    def test_dict_and_json_round_trip_preserve_lineage(self) -> None:
        rubric = StructuredRubric(
            nodes={
                "independent": node(
                    "independent",
                    examples=[{"sample_id": "s1", "labels": ["A", "B"]}],
                    lineage={
                        "unsupported_multi_parent_dependency": [
                            "visual_grounding",
                            "factual_consistency",
                        ]
                    },
                )
            },
            edges=(),
            root_ids=("independent",),
        )
        serialized = rubric.to_dict()
        self.assertEqual(
            STRUCTURED_RUBRIC_SCHEMA_VERSION,
            serialized["schema_version"],
        )
        self.assertEqual(rubric.rubric_sha256, serialized["rubric_sha256"])
        self.assertEqual(rubric, StructuredRubric.from_dict(serialized))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rubric.json"
            rubric.save_json(path)
            self.assertEqual(rubric, StructuredRubric.load_json(path))

    def test_hash_is_canonical_for_node_and_edge_storage_order(self) -> None:
        first = branching_rubric()
        reversed_storage = branching_rubric(reverse_storage=True)
        self.assertEqual(first, reversed_storage)
        self.assertEqual(first.rubric_sha256, reversed_storage.rubric_sha256)
        self.assertEqual(first.to_dict(), reversed_storage.to_dict())

    def test_execution_and_metadata_changes_change_hash(self) -> None:
        base = branching_rubric()
        root_reordered = StructuredRubric(
            nodes=base.nodes,
            edges=base.edges,
            root_ids=("root", "other"),
        )
        description_changed_nodes = dict(base.nodes)
        description_changed_nodes["other"] = node(
            "other", description="changed description"
        )
        description_changed = StructuredRubric(
            description_changed_nodes,
            base.edges,
            base.root_ids,
        )
        condition_changed = StructuredRubric(
            base.nodes,
            tuple(
                RubricEdge(edge.parent_id, edge.child_id, EdgeCondition.PARENT_BOTH_PASS)
                if edge.child_id == "a-child"
                else edge
                for edge in base.edges
            ),
            base.root_ids,
        )
        metadata_nodes = dict(base.nodes)
        metadata_nodes["other"] = node("other", lineage={"origin": "manual"})
        metadata_changed = StructuredRubric(
            metadata_nodes,
            base.edges,
            base.root_ids,
        )
        score_nodes = dict(base.nodes)
        score_nodes["other"] = node("other", score=0.9)
        score_changed = StructuredRubric(score_nodes, base.edges, base.root_ids)

        for changed in (
            root_reordered,
            description_changed,
            condition_changed,
            metadata_changed,
            score_changed,
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(base.rubric_sha256, changed.rubric_sha256)

    def test_loader_rejects_schema_hash_and_shape_corruption(self) -> None:
        serialized = branching_rubric().to_dict()
        cases = []

        missing = copy.deepcopy(serialized)
        missing.pop("edges")
        cases.append(missing)
        extra = copy.deepcopy(serialized)
        extra["unknown"] = True
        cases.append(extra)
        version = copy.deepcopy(serialized)
        version["schema_version"] = "2.0.0"
        cases.append(version)
        semantics = copy.deepcopy(serialized)
        semantics["semantics_version"] = "2.0.0"
        cases.append(semantics)
        hash_changed = copy.deepcopy(serialized)
        hash_changed["nodes"][0]["criterion"]["description"] = "tampered"
        cases.append(hash_changed)
        bad_hash = copy.deepcopy(serialized)
        bad_hash["rubric_sha256"] = "not-a-hash"
        cases.append(bad_hash)
        duplicate = copy.deepcopy(serialized)
        duplicate["nodes"].append(copy.deepcopy(duplicate["nodes"][0]))
        cases.append(duplicate)
        wrong_nodes_type = copy.deepcopy(serialized)
        wrong_nodes_type["nodes"] = {}
        cases.append(wrong_nodes_type)

        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                StructuredRubric.from_dict(value)

    def test_metadata_and_edge_schema_reject_invalid_values(self) -> None:
        for invalid_metadata in (
            {"bad": {1, 2}},
            {1: "non-string-key"},
            {"bad": math.nan},
        ):
            with self.subTest(metadata=invalid_metadata), self.assertRaises(
                RubricSchemaError
            ):
                node("x", lineage=invalid_metadata)

        with self.assertRaises(RubricSchemaError):
            node("x", examples=["not-an-object"])
        with self.assertRaises(RubricSchemaError):
            RubricEdge("x", "y", "always")
        with self.assertRaises(RubricSchemaError):
            RubricEdge.from_dict(
                {"parent_id": "x", "child_id": "y", "condition": "unknown"}
            )


if __name__ == "__main__":
    unittest.main()

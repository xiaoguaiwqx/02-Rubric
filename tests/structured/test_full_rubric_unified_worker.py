"""Offline protocol tests for the Unified Full-Rubric Worker ablation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from critiq.structured import (
    EdgeCondition,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredRubric,
)
from experiments.evolving_structured_rubrics import (
    full_rubric_unified_worker as unified,
)


def _rubric() -> StructuredRubric:
    nodes = {
        "root": RubricNode(
            node_id="root",
            criterion=RubricCriterionSnapshot(
                name="root_focus", description="Judge the main requirement."),
        ),
        "child": RubricNode(
            node_id="child",
            criterion=RubricCriterionSnapshot(
                name="visual_detail", description="Check the decisive image detail."),
        ),
    }
    return StructuredRubric(
        nodes=nodes,
        edges=(RubricEdge(
            parent_id="root", child_id="child",
            condition=EdgeCondition.PARENT_NONDECISIVE),),
        root_ids=("root",),
    )


class TestFullRubricUnifiedWorker(unittest.TestCase):
    def test_example_config_matches_frozen_protocol(self) -> None:
        path = Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        settings = unified._settings(config)
        self.assertEqual(1, settings["internal_k"])
        self.assertEqual(3, settings["vl_rewardbench_k"])
        self.assertFalse(settings["internal_ab_swap"])
        self.assertEqual("unset", settings["generation_seed_policy"])
        self.assertEqual(["A", "B", "None"], settings["semantic_answer_space"])

    def test_serialization_preserves_hierarchy_and_all_nodes_once(self) -> None:
        rubric = _rubric()
        value = unified.full_rubric_serialization(rubric)
        self.assertEqual(["root"], value["root_ids"])
        root = value["roots"][0]
        self.assertEqual("root", root["node_id"])
        self.assertEqual("child", root["children"][0]["node"]["node_id"])
        self.assertEqual(
            EdgeCondition.PARENT_NONDECISIVE.value,
            root["children"][0]["edge_condition"],
        )
        rendered = unified.render_full_rubric(value)
        self.assertEqual(1, rendered.count("Judge the main requirement."))
        self.assertEqual(1, rendered.count("Check the decisive image detail."))

    def test_static_prompt_contains_rubric_but_user_prompt_does_not(self) -> None:
        rubric = _rubric()
        system = unified.full_rubric_system_prompt(rubric)
        user = unified.full_rubric_user_prompt({
            "question": "What is shown?", "A": "A cat", "B": "A dog"})
        self.assertIn("## Complete Structured Rubric", system)
        self.assertIn("root_focus", system)
        self.assertIn('"answer": "A / B"', system)
        self.assertNotIn("root_focus", user)
        self.assertIn("What is shown?", user)
        self.assertIn("A cat", user)
        self.assertIn("A dog", user)

    def test_parser_accepts_semantic_none_without_requiring_reasoning(self) -> None:
        for answer in ("A", "B", "None"):
            with self.subTest(answer=answer):
                value = unified.parse_full_rubric_response(
                    json.dumps({"answer": answer}))
                self.assertEqual(answer, value["answer"])
                self.assertEqual("", value["thought"])
        for answer in ("C", "", None):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                unified.parse_full_rubric_response(json.dumps({"answer": answer}))

    def test_vlrb_mapping_respects_each_replicate_order(self) -> None:
        result = {"samples": [{
            "orders": [0, 1, 0],
            "calls": {
                "0": {"parse_ok": True, "parsed": {"answer": "A"}},
                "1": {"parse_ok": True, "parsed": {"answer": "B"}},
                "2": {"parse_ok": True, "parsed": {"answer": "None"}},
            },
        }]}
        self.assertEqual([[0], [0], [None]], unified._vlrb_vote_matrix(result))

    def test_request_budget_and_execution_exclude_auxiliary_workers(self) -> None:
        self.assertEqual(750, sum(unified.INTERNAL_COUNTS.values()))
        self.assertEqual(3741, unified.VLRB_COUNT * unified.VLRB_K)
        self.assertEqual(4491, 750 + unified.VLRB_COUNT * unified.VLRB_K)
        self.assertEqual(7, len(unified.STAGES))
        self.assertTrue(all(
            stage.startswith("full-rubric-worker-") for stage in unified.STAGES))

    def test_scheduler_makes_exactly_one_call_per_sample_replicate(self) -> None:
        config_path = Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = tuple({
            "sample_id": f"sample-{index}", "image_path": "unused.png",
            "question": "q", "A": "a", "B": "b", "answer": "A",
        } for index in range(2))
        orders = {
            "sample-0": (0, 1, 0),
            "sample-1": (1, 0, 1),
        }

        def fake_call(*args, **kwargs):
            endpoint = args[1]
            return {
                "parse_ok": True,
                "parsed": {"answer": "A"},
                "model_generation_count": 1,
                "endpoint_id": endpoint.endpoint_id,
                "cache_hit": False,
                "metrics": {},
                "cache_key": "fake",
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(
                unified.support, "call_one", side_effect=fake_call) as mocked:
            target = Path(directory)
            result = unified._run_rows(
                config, target, "test", "test_k3", rows, orders,
                _rubric(), 1)
            self.assertEqual(6, mocked.call_count)
            self.assertEqual(2, len(result["samples"]))
            self.assertTrue((target / "progress" / "test_k3.json").is_file())
            request_ids = {
                call.kwargs["request_key"]["sample_id"]
                for call in mocked.call_args_list
            }
            self.assertEqual({"sample-0", "sample-1"}, request_ids)


if __name__ == "__main__":
    unittest.main()

"""Offline tests for the internal K=1 Global-Arbiter runner."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.evolving_structured_rubrics import internal_global_arbiter_k1 as k1


class TestInternalGlobalArbiterK1(unittest.TestCase):
    def test_protocol_is_exactly_one_original_order_inference(self) -> None:
        config = json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))
        settings = k1._settings(config)
        self.assertEqual(1, settings["k"])
        self.assertEqual(1, settings["replicate_count"])
        self.assertFalse(settings["ab_swap"])
        self.assertEqual("unset", settings["generation_seed_policy"])

    def test_subtree_parser_treats_none_as_semantic_success(self) -> None:
        value = k1.parse_subtree_response('{"answer":"None"}')
        self.assertEqual("None", value["answer"])
        self.assertEqual("", value["analysis_a"])
        for answer in ("C", "", None):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                k1.parse_subtree_response(json.dumps({"answer": answer}))

    def test_equal_root_ignores_none_and_requires_unique_plurality(self) -> None:
        def sample(values):
            return {"subtrees": {
                str(index): {"parse_ok": True, "parsed": {"answer": value}}
                for index, value in enumerate(values)}}
        self.assertEqual("A", k1._equal_root(sample(["A", "A", "B", "None", "None"])))
        self.assertEqual("B", k1._equal_root(sample(["A", "B", "B", "None", "None"])))
        self.assertIsNone(k1._equal_root(sample(["A", "B", "None", "None", "None"])))

    def test_metrics_separate_strict_coverage_and_overall(self) -> None:
        rows = [
            {"answer": "A", "domain": "x", "source": "s"},
            {"answer": "B", "domain": "x", "source": "s"},
            {"answer": "A", "domain": "y", "source": "t"},
        ]
        value = k1._metrics(rows, ["A", None, "B"])
        self.assertEqual(1 / 3, value["strict_accuracy"])
        self.assertEqual(2 / 3, value["coverage"])
        self.assertEqual(0.5, value["overall_acc"])
        self.assertEqual(1, value["none_count"])

    def test_request_budget_is_4500(self) -> None:
        self.assertEqual(4500, sum(k1.SPLIT_COUNTS.values()) * 6)
        self.assertEqual(6, len(k1.STAGES))


if __name__ == "__main__":
    unittest.main()

"""Deterministic tests for the cached K=3 coalition audit."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.evolving_structured_rubrics import (
    counterfactual_coalition_audit as audit,
)


class TestK3Aggregation(unittest.TestCase):
    def test_majority_requires_two_matching_decisive_votes(self) -> None:
        self.assertEqual("A", audit._majority(["A", "B", "A"]))
        self.assertEqual("B", audit._majority(["B", "None", "B"]))
        self.assertEqual("None", audit._majority(["A", "B", "None"]))
        self.assertEqual("None", audit._majority(
            ["A", "technical_failure", "None"]))

    def test_system_metrics_count_none_as_strict_error(self) -> None:
        rows = ({"answer": "A"}, {"answer": "B"}, {"answer": "A"})
        value = audit._system_metrics(("A", "None", "B"), rows)
        self.assertEqual(1, value["correct_count"])
        self.assertAlmostEqual(1 / 3, value["strict_accuracy"])
        self.assertAlmostEqual(2 / 3, value["coverage"])
        self.assertAlmostEqual(0.5, value["covered_accuracy"])


class TestCoalitionAnalysis(unittest.TestCase):
    def test_coalition_spec_uses_mask_order(self) -> None:
        entries = [
            {"epoch": 1, "root_id": f"r{index}", "candidate_id": f"c{index}"}
            for index in range(5)
        ]
        value = audit._coalition_spec(entries, 0b10101)
        self.assertEqual(["r0", "r2", "r4"], value["root_ids"])
        self.assertEqual(["c0", "c2", "c4"], value["candidate_ids"])
        self.assertEqual(3, value["size"])

    def test_exact_best_prefers_smaller_subset_then_mask(self) -> None:
        values = {mask: 0 for mask in range(8)}
        values[0b011] = 4
        values[0b100] = 4
        values[0b010] = 4
        self.assertEqual(0b010, audit._exact_best(values))

    def test_exact_shapley_recovers_additive_values(self) -> None:
        weights = (2.0, -1.0, 3.0, 0.5, -0.5)
        values = {
            mask: sum(weight for index, weight in enumerate(weights)
                      if mask & (1 << index))
            for mask in range(32)
        }
        actual = audit._shapley(values, 5)
        for expected, observed in zip(weights, actual):
            self.assertAlmostEqual(expected, observed)

    def test_pairwise_interaction_is_non_additive_residual(self) -> None:
        values = {mask: 0.0 for mask in range(4)}
        values[1], values[2], values[3] = 2.0, 3.0, 8.0
        self.assertEqual(3.0, audit._interaction(values, 0, 1))


class TestFrozenProtocol(unittest.TestCase):
    def test_config_matches_code(self) -> None:
        config = json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))
        self.assertEqual(audit.SETTINGS, audit._settings(config))

    def test_request_budget_and_no_subtree_regeneration(self) -> None:
        singleton_new = 26 * 100 * 2
        coalition_new = 2 * 26 * 100 * 3
        self.assertEqual(20800, singleton_new + coalition_new)
        self.assertEqual(0, audit.SETTINGS["subtree_regeneration_count"])
        self.assertEqual(3, audit.SETTINGS["k"])
        self.assertEqual([1, 5], audit.SETTINGS["coalition_epochs"])


if __name__ == "__main__":
    unittest.main()

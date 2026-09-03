"""Deterministic tests for the frozen Dev150 transfer audit."""

from __future__ import annotations

import unittest

from experiments.evolving_structured_rubrics import (
    counterfactual_dev150_transfer_audit as audit,
)


class TestCounterfactualDev150TransferAudit(unittest.TestCase):
    def test_frozen_systems_are_exact_and_not_a_search_space(self) -> None:
        expected = {
            "baseline": (None, 0b00000),
            "e1_completeness": (1, 0b00001),
            "e1_cvf": (1, 0b00111),
            "e1_positive_union": (1, 0b10011),
            "e1_all": (1, 0b11111),
            "e5_completeness": (5, 0b00001),
            "e5_clarity": (5, 0b10000),
            "e5_positive_union": (5, 0b10101),
            "e5_all": (5, 0b11111),
        }
        actual = {item["system_id"]: (item["epoch"], item["mask"])
                  for item in audit.FROZEN_SYSTEMS}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 9)
        self.assertTrue(audit.SETTINGS["selection_after_dev_forbidden"])

    def test_frozen_systems_require_ten_unique_candidate_roots(self) -> None:
        selected = {(item["epoch"], index) for item in audit.FROZEN_SYSTEMS
                    if item["epoch"] is not None
                    for index in range(5) if item["mask"] & (1 << index)}
        self.assertEqual(selected, {(epoch, index) for epoch in (1, 5)
                                    for index in range(5)})

    def test_k3_majority_uses_none_when_no_ab_majority(self) -> None:
        self.assertEqual(audit._majority(["A", "A", "B"]), "A")
        self.assertEqual(audit._majority(["B", "None", "B"]), "B")
        self.assertEqual(audit._majority(["A", "B", "None"]), "None")
        self.assertEqual(audit._majority(
            ["technical_failure", "A", "None"]), "None")

    def test_mcnemar_exact_is_symmetric(self) -> None:
        self.assertEqual(audit._mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(audit._mcnemar(3, 7), audit._mcnemar(7, 3))
        self.assertLess(audit._mcnemar(0, 10), 0.01)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        rows = (
            {"sample_id": "s1", "answer": "A"},
            {"sample_id": "s2", "answer": "B"},
            {"sample_id": "s3", "answer": "A"},
        )
        baseline = {"metrics": {"predictions": ["B", "B", "A"]}}
        candidate = {"metrics": {"predictions": ["A", "A", "A"]}}
        left = audit._paired_ci(baseline, candidate, rows, seed=42)
        right = audit._paired_ci(baseline, candidate, rows, seed=42)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 2)


if __name__ == "__main__":
    unittest.main()

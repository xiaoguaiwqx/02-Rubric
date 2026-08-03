"""Tests for deterministic static-rubric construction and reusable analysis."""

from __future__ import annotations

import unittest
from pathlib import Path

from critiq.structured import FinalPreference, Vote, aggregate_weighted_root_votes
from experiments.evolving_structured_rubrics.analysis import (
    paired_bootstrap_accuracy_difference,
)
from experiments.evolving_structured_rubrics.rubric_factory import (
    CRITERIA_SOURCE_SHA256,
    EXPECTED_CRITERION_NAMES,
    HELDOUT_DATASET_SHA256,
    PROJECTED_EDGES,
    PROJECTED_ROOT_NAMES,
    STRICT_EDGES,
    STRICT_ROOT_NAMES,
    build_static_rubrics,
    file_sha256,
    validate_heldout_dataset_source,
)


ROOT = Path(__file__).resolve().parents[2]
CRITERIA_PATH = (
    ROOT
    / "output/rlhfv_exp4_dis90_val100_n10_wp-final-heldout500_e10"
    / "final_heldout_raw_prediction.json"
)
DATASET_PATH = ROOT / "data/RLHF-V/heldout_validation_500_pair.jsonl"


class RubricFactoryTest(unittest.TestCase):
    def test_exact_sources_and_static_topologies(self):
        self.assertEqual(CRITERIA_SOURCE_SHA256, file_sha256(CRITERIA_PATH))
        self.assertEqual(HELDOUT_DATASET_SHA256, file_sha256(DATASET_PATH))
        validate_heldout_dataset_source(DATASET_PATH)

        strict, projected = build_static_rubrics(CRITERIA_PATH)
        self.assertEqual((17, 15, 2), (len(strict.nodes), len(strict.root_ids), len(strict.edges)))
        self.assertEqual((17, 9, 8), (len(projected.nodes), len(projected.root_ids), len(projected.edges)))
        self.assertEqual(
            tuple(
                f"c{index:02d}_{name}"
                for index, name in enumerate(EXPECTED_CRITERION_NAMES, 1)
            ),
            tuple(sorted(strict.nodes)),
        )
        self.assertEqual(
            tuple(
                f"c{EXPECTED_CRITERION_NAMES.index(name) + 1:02d}_{name}"
                for name in STRICT_ROOT_NAMES
            ),
            strict.root_ids,
        )
        self.assertEqual(
            tuple(
                f"c{EXPECTED_CRITERION_NAMES.index(name) + 1:02d}_{name}"
                for name in PROJECTED_ROOT_NAMES
            ),
            projected.root_ids,
        )
        self.assertEqual(len(STRICT_EDGES), len(strict.edges))
        self.assertEqual(len(PROJECTED_EDGES), len(projected.edges))

    def test_criteria_snapshots_and_lineage_are_deterministic(self):
        strict, projected = build_static_rubrics(CRITERIA_PATH)
        for name in EXPECTED_CRITERION_NAMES:
            strict_node = strict.get_node(strict.node_id_for_criterion(name))
            projected_node = projected.get_node(projected.node_id_for_criterion(name))
            self.assertEqual(strict_node.criterion, projected_node.criterion)

        completeness = projected.get_node(
            projected.node_id_for_criterion("completeness")
        )
        self.assertEqual(
            "factual_consistency",
            completeness.lineage["primary_parent_projection"],
        )
        self.assertEqual(
            strict.rubric_sha256,
            build_static_rubrics(CRITERIA_PATH)[0].rubric_sha256,
        )
        self.assertEqual(
            projected.rubric_sha256,
            build_static_rubrics(CRITERIA_PATH)[1].rubric_sha256,
        )


class AggregationAndAnalysisTest(unittest.TestCase):
    def test_weighted_root_voting_contract(self):
        self.assertEqual(
            FinalPreference.A,
            aggregate_weighted_root_votes(
                {"r1": Vote.A, "r2": Vote.B},
                ("r1", "r2"),
                {"r1": 0.8, "r2": 0.2},
            ),
        )
        self.assertEqual(
            FinalPreference.TIE,
            aggregate_weighted_root_votes(
                {"r1": Vote.ABSTAIN, "r2": Vote.B},
                ("r1", "r2"),
                {"r1": 0.8, "r2": 0.0},
            ),
        )
        with self.assertRaises(ValueError):
            aggregate_weighted_root_votes({"r": Vote.A}, ("r",), {"r": 0.0})
        with self.assertRaises(ValueError):
            aggregate_weighted_root_votes({"r": Vote.A}, ("r",), {"r": -1.0})
        with self.assertRaises(ValueError):
            aggregate_weighted_root_votes({"r": Vote.A}, ("r",), {})

    def test_paired_bootstrap_is_seeded(self):
        first = paired_bootstrap_accuracy_difference(
            [True, False], [True, True], samples=100, seed=7
        )
        second = paired_bootstrap_accuracy_difference(
            [True, False], [True, True], samples=100, seed=7
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

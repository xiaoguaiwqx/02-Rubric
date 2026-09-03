"""Deterministic tests for the Phase21 rejected-candidate audit."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from critiq.structured.schema import (
    RubricCriterionSnapshot,
    RubricNode,
    StructuredRubric,
)
from experiments.evolving_structured_rubrics import (
    rejected_candidate_counterfactual_arbiter_audit as audit,
)
from experiments.evolving_structured_rubrics.experiment_utils import atomic_write_json


def _call(answer: str, marker: str) -> dict:
    return {
        "parse_ok": True,
        "parsed": {
            "analysis_a": marker,
            "analysis_b": marker,
            "thought": marker,
            "answer": answer,
        },
        "model_generation_count": 1,
        "endpoint_id": "vllm-8000",
    }


def _root_artifact(root_id: str, marker: str) -> dict:
    return {
        "root_id": root_id,
        "samples": [
            {"sample_id": "s1", "order": 0, "call": _call("A", marker)},
            {"sample_id": "s2", "order": 0, "call": _call("B", marker)},
        ],
    }


class TestSelectiveUtility(unittest.TestCase):
    def test_none_is_zero_not_wrong(self) -> None:
        self.assertEqual(1, audit._utility("A", "A"))
        self.assertEqual(-1, audit._utility("B", "A"))
        self.assertEqual(0, audit._utility("None", "A"))

    def test_rank_correlations_handle_ties(self) -> None:
        xs = [0.0, 1.0, 1.0, 2.0]
        ys = [0.0, 2.0, 2.0, 4.0]
        self.assertAlmostEqual(1.0, audit._spearman(xs, ys))
        self.assertAlmostEqual(1.0, audit._pearson(xs, ys))
        self.assertAlmostEqual(1.0, audit._kendall_tau_b(xs, ys))

    def test_root_fixed_effect_removes_root_offsets(self) -> None:
        records = [
            {"root_id": "r1", "x": 0.0, "y": 10.0},
            {"root_id": "r1", "x": 1.0, "y": 11.0},
            {"root_id": "r2", "x": 0.0, "y": -10.0},
            {"root_id": "r2", "x": 1.0, "y": -9.0},
        ]
        value = audit._fixed_effect(records, "x", "y")
        self.assertAlmostEqual(1.0, value["pearson"])
        self.assertAlmostEqual(1.0, value["slope"])


class TestCounterfactualBundle(unittest.TestCase):
    def test_exactly_one_root_report_is_replaced(self) -> None:
        rubric = StructuredRubric({
            "r1": RubricNode(
                "r1", RubricCriterionSnapshot("root_one", "root one", 1.0)),
            "r2": RubricNode(
                "r2", RubricCriterionSnapshot("root_two", "root two", 1.0)),
        }, (), ("r1", "r2"))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            rubric.save_json(source / "candidate_rubric.json")
            atomic_write_json(source / "baseline_r1.json", _root_artifact("r1", "base-r1"))
            atomic_write_json(source / "baseline_r2.json", _root_artifact("r2", "base-r2"))
            atomic_write_json(source / "candidate_r1.json", _root_artifact("r1", "candidate-r1"))
            entry = {
                "root_id": "r1",
                "candidate_rubric_path": "candidate_rubric.json",
                "candidate_root_path": "candidate_r1.json",
                "baseline_root_paths": {
                    "r1": "baseline_r1.json", "r2": "baseline_r2.json"},
            }
            bundles = audit._report_bundle_index(entry, source, ("s1", "s2"))
        reports, _ = bundles["s1"]
        self.assertEqual("candidate-r1", reports[0]["report"]["thought"])
        self.assertEqual("base-r2", reports[1]["report"]["thought"])
        self.assertEqual(2, len(reports))


class TestFrozenProtocol(unittest.TestCase):
    def test_config_matches_code(self) -> None:
        config = json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))
        self.assertEqual(audit.SETTINGS, audit._settings(config))

    def test_stage1_selection_is_frozen_before_results(self) -> None:
        self.assertEqual(6, len(audit.STAGE1_KEYS))
        self.assertEqual(6, len(set(audit.STAGE1_KEYS)))
        self.assertEqual(
            ("positive", "positive", "borderline", "negative", "negative", "negative"),
            audit.STAGE1_ROLES)

    def test_protocol_forbids_new_subtree_requests(self) -> None:
        self.assertEqual(0, audit.SETTINGS["subtree_regeneration_count"])
        self.assertEqual("replace_exactly_one_root_report",
                         audit.SETTINGS["counterfactual_unit"])

    def test_endpoint_preflight_retries_transient_timeout(self) -> None:
        expected = [{"endpoint_id": "vllm-8000"},
                    {"endpoint_id": "vllm-8001"}]
        with patch.object(
                audit.phase21, "_endpoint_identities",
                side_effect=[urllib.error.URLError(TimeoutError()), expected]
        ) as inspect_endpoints, patch.object(audit.time, "sleep") as sleep:
            value = audit._endpoint_identities_with_retry({}, attempts=2)
        self.assertEqual(expected, value)
        self.assertEqual(2, inspect_endpoints.call_count)
        sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()

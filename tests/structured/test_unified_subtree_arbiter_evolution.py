"""Focused offline tests for Phase19 aligned execution and competition."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from critiq.structured.backend_pool import BackendEndpointSpec

from experiments.evolving_structured_rubrics import aligned_system_runtime as runtime
from experiments.evolving_structured_rubrics import run_rubric_evolution as base
from experiments.evolving_structured_rubrics import unified_subtree_arbiter_evolution as evolution
from experiments.evolving_structured_rubrics import vl_rewardbench_aligned_evolution as vlrb


def _call(answer: str, *, reused: bool = False):
    return {
        "parse_ok": True,
        "parsed": {"answer": answer},
        "model_generation_count": 1,
        "endpoint_id": "vllm-8000",
        "incremental_reuse": reused,
    }


class TestAlignedSystemRuntime(unittest.TestCase):
    def test_strict_metrics_count_none_as_wrong(self) -> None:
        value = {
            "k": 1,
            "samples": [
                {"sample_id": "s1", "replicates": {"0": {
                    "order": 0,
                    "subtrees": {"r": _call("A")},
                    "arbiter": _call("A")}}},
                {"sample_id": "s2", "replicates": {"0": {
                    "order": 0,
                    "subtrees": {"r": _call("None")},
                    "arbiter": _call("None")}}},
            ],
        }
        rows = [
            {"sample_id": "s1", "answer": "A"},
            {"sample_id": "s2", "answer": "B"},
        ]
        metrics = runtime.metrics(value, rows)
        self.assertEqual(0.5, metrics["strict_accuracy"])
        self.assertEqual(0.5, metrics["coverage"])
        self.assertEqual(1.0, metrics["covered_accuracy"])
        self.assertEqual(0, metrics["technical_failure_count"])

    def test_paired_uses_corrected_greater_than_harmed(self) -> None:
        rows = [
            {"sample_id": "s1", "answer": "A"},
            {"sample_id": "s2", "answer": "B"},
            {"sample_id": "s3", "answer": "A"},
        ]
        before = {"strict_accuracy": 1 / 3,
                  "predictions": ["B", "A", "A"]}
        after = {"strict_accuracy": 2 / 3,
                 "predictions": ["A", "B", "B"]}
        value = runtime.paired(before, after, rows)
        self.assertEqual(2, value["corrected"])
        self.assertEqual(1, value["harmed"])
        self.assertEqual(1, value["net_corrected"])

    def test_incremental_sample_regenerates_only_changed_root(self) -> None:
        rubric = base.build_multicrit_open_ended_init_rubric()
        changed = rubric.root_ids[1]
        old_calls = {root_id: _call("A") for root_id in rubric.root_ids}
        baseline = {"replicates": {"0": {"subtrees": old_calls}}}
        endpoint = BackendEndpointSpec(
            "vllm-8000", "http://example/v1", "checkpoint", 1)
        row = {"sample_id": "s1", "image_path": "unused",
               "question": "q", "A": "a", "B": "b", "answer": "A"}

        with patch.object(runtime, "_call_subtree", return_value=_call("B")) as subtree, \
                patch.object(runtime, "_call_arbiter", return_value=_call("A")):
            value = runtime._one_sample(
                {"model": "m"}, endpoint, Path("unused"), "split", row,
                (0,), rubric, baseline, frozenset({changed}), 1,
                runtime.RuntimeSettings(0.5, 2048, 10))

        self.assertEqual(1, subtree.call_count)
        item = value["replicates"]["0"]
        self.assertEqual([changed], item["regenerated_root_ids"])
        self.assertTrue(all(
            call["incremental_reuse"]
            for root_id, call in item["subtrees"].items()
            if root_id != changed))
        self.assertFalse(item["subtrees"][changed]["incremental_reuse"])


class TestAlignedEvolutionProtocol(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        records = [
            {"sample_id": "s1", "preferred_original_index": 0},
            {"sample_id": "s2", "preferred_original_index": 1},
            {"sample_id": "s3", "preferred_original_index": 0},
            {"sample_id": "s4", "preferred_original_index": 1},
        ]
        before = [1, 0, 0, 1]
        after = [0, 1, 1, 1]
        first = vlrb._paired_bootstrap_delta_ci(
            records, before, after, iterations=1_000)
        second = vlrb._paired_bootstrap_delta_ci(
            records, before, after, iterations=1_000)
        self.assertEqual(first, second)
        self.assertEqual(0.25, first["estimate"])

    def test_config_matches_frozen_settings(self) -> None:
        config = json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))
        self.assertEqual(evolution.SETTINGS, evolution._settings(config))
        self.assertEqual(vlrb.SETTINGS, vlrb._settings(config))

    def test_system_rejection_has_durable_attribution(self) -> None:
        result = {
            "root_id": "r1", "decision": "accepted",
            "attempt_dir": Path("output/test-aligned-attribution"),
            "candidate": None, "evaluation": None,
        }
        result["attempt_dir"].mkdir(parents=True, exist_ok=True)
        paired = {"corrected": 1, "harmed": 2, "net_corrected": -1,
                  "strict_accuracy_delta": -0.01,
                  "corrected_sample_ids": ["a"],
                  "harmed_sample_ids": ["b", "c"]}
        try:
            evolution._set_system_outcome(
                result, accepted=False, paired_value=paired,
                kind="subtree_evidence_regression")
            self.assertEqual("competition_rejected", result["decision"])
            self.assertIn("natural_language_attribution", result["history_payload"])
            self.assertTrue((result["attempt_dir"] /
                             "aligned_failure_attribution.json").is_file())
        finally:
            (result["attempt_dir"] /
             "aligned_failure_attribution.json").unlink(missing_ok=True)
            result["attempt_dir"].rmdir()

    def test_system_outcome_records_acceptance_status(self) -> None:
        result = {"node_id": "n1", "decision": "competition_rejected"}
        paired = {"corrected": 2, "harmed": 1, "net_corrected": 1,
                  "strict_accuracy_delta": 0.01}
        evolution._set_system_outcome(
            result, accepted=True, paired_value=paired,
            kind="independent_system_improvement")
        self.assertEqual("accepted", result["decision"])
        self.assertTrue(result["system_evaluation"]["accepted"])
        self.assertEqual(
            "independent_system_improvement",
            result["system_evaluation"]["failure_type"])

    def test_joint_report_composition_reuses_frozen_candidate_roots(self) -> None:
        rubric = base.build_multicrit_open_ended_init_rubric()
        roots = rubric.root_ids[:2]
        baseline_calls = {root_id: _call("A") for root_id in rubric.root_ids}
        baseline = {"samples": [{"sample_id": "s1", "replicates": {"0": {
            "subtrees": baseline_calls, "arbiter": _call("A"), "order": 0,
        }}}]}
        selected = []
        for index, root_id in enumerate(roots):
            calls = {key: dict(value) for key, value in baseline_calls.items()}
            calls[root_id] = _call("B")
            selected.append(("split", root_id, {
                "aligned_system_artifact": {"samples": [{
                    "sample_id": "s1", "replicates": {"0": {
                        "subtrees": calls, "arbiter": _call("A"), "order": 0,
                    }},
                }]},
            }))
        composed = evolution._compose_candidate_reports(
            baseline, rubric, selected)
        calls = composed["samples"][0]["replicates"]["0"]["subtrees"]
        self.assertEqual("B", calls[roots[0]]["parsed"]["answer"])
        self.assertEqual("B", calls[roots[1]]["parsed"]["answer"])
        self.assertEqual(
            "A", calls[rubric.root_ids[2]]["parsed"]["answer"])

    def test_stage_names_and_primary_acceptance_are_frozen(self) -> None:
        self.assertEqual(6, len(evolution.STAGES))
        self.assertEqual(6, len(vlrb.STAGES))
        self.assertEqual("system_strict_accuracy",
                         evolution.SETTINGS["acceptance_metric"])
        self.assertFalse(evolution.SETTINGS["subset_search"])
        self.assertFalse(evolution.SETTINGS["dev_used_for_selection"])


if __name__ == "__main__":
    unittest.main()

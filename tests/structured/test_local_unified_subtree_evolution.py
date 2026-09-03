"""Focused offline tests for Phase20 root-local competition semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments.evolving_structured_rubrics import aligned_system_runtime as runtime
from experiments.evolving_structured_rubrics import local_unified_subtree_evolution as evolution
from experiments.evolving_structured_rubrics import refine_evolution
from experiments.evolving_structured_rubrics import run_rubric_evolution as base
from experiments.evolving_structured_rubrics import split_evolution
from experiments.evolving_structured_rubrics import vl_rewardbench_local_unified_subtree_evolution as vlrb


def _call(answer: str):
    return {
        "parse_ok": True, "parsed": {"answer": answer},
        "model_generation_count": 1, "endpoint_id": "vllm-8000"}


def _root_artifact(root_id: str, answers: list[str], scope: list[str]):
    value = {
        "root_id": root_id,
        "samples": [
            {"sample_id": f"s{index + 1}", "order": 0, "call": _call(answer)}
            for index, answer in enumerate(answers)],
    }
    rows = [
        {"sample_id": "s1", "answer": "A"},
        {"sample_id": "s2", "answer": "B"},
        {"sample_id": "s3", "answer": "A"},
    ]
    value["metrics"] = runtime.root_metrics(value, rows, scope)
    return value, rows


class TestRootScopeRuntime(unittest.TestCase):
    def test_pool_workers_are_interleaved_across_endpoints(self) -> None:
        endpoints = (
            SimpleNamespace(endpoint_id="vllm-8000", max_concurrency=2),
            SimpleNamespace(endpoint_id="vllm-8001", max_concurrency=2),
        )
        workers = runtime._interleaved_workers(endpoints)
        self.assertEqual(
            ["vllm-8000", "vllm-8001", "vllm-8000", "vllm-8001"],
            [item.endpoint_id for item in workers])

    def test_none_is_wrong_and_outside_scope_does_not_change_denominator(self) -> None:
        value, _ = _root_artifact("r1", ["A", "None", "B"], ["s1", "s2"])
        metrics = value["metrics"]
        self.assertEqual(2, metrics["root_scope_support"])
        self.assertEqual(0.5, metrics["root_scope_strict_accuracy"])
        self.assertEqual(0.5, metrics["root_scope_coverage"])
        self.assertEqual(1.0, metrics["outside_scope_coverage"])

    def test_paired_root_uses_the_same_frozen_scope(self) -> None:
        before, rows = _root_artifact("r1", ["B", "B", "A"], ["s1", "s2"])
        after, _ = _root_artifact("r1", ["A", "A", "B"], ["s1", "s2"])
        value = runtime.paired_root(before, after, rows)
        self.assertEqual(["s1"], value["root_scope_corrected_sample_ids"])
        self.assertEqual(["s2"], value["root_scope_harmed_sample_ids"])
        self.assertEqual(0, value["root_scope_net_corrected"])

    def test_candidate_cannot_change_scope(self) -> None:
        before, rows = _root_artifact("r1", ["A", "B", "A"], ["s1", "s2"])
        after, _ = _root_artifact("r1", ["A", "B", "A"], ["s1", "s3"])
        with self.assertRaisesRegex(ValueError, "frozen evaluation scope"):
            runtime.paired_root(before, after, rows)


class TestLocalEvolutionProtocol(unittest.TestCase):
    def test_root_scope_comes_from_epoch_start_root_pairwise_votes(self) -> None:
        rubric = base.build_multicrit_open_ended_init_rubric()
        root_names = [rubric.get_node(root_id).criterion.name
                      for root_id in rubric.root_ids]
        rows = [{"sample_id": "s1"}, {"sample_id": "s2"}]
        outputs = []
        for index in range(2):
            outputs.append({name: SimpleNamespace(vote=SimpleNamespace(
                value="A" if index == 0 else "abstain")) for name in root_names})
        prediction = SimpleNamespace(node_outputs=outputs)
        scopes = evolution._root_scope_ids(prediction, rubric, rows)
        self.assertTrue(all(value == ("s1",) for value in scopes.values()))

    def test_config_and_acceptance_contract_are_frozen(self) -> None:
        config = json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))
        self.assertEqual(evolution.SETTINGS, evolution._settings(config))
        self.assertEqual(vlrb.SETTINGS, vlrb._settings(config))
        self.assertFalse(evolution.SETTINGS["same_root_winner"])
        self.assertEqual(
            "post_commit_diagnostic_only",
            evolution.SETTINGS["global_arbiter_role"])
        self.assertEqual(
            "root_or_node_local_errors_without_arbiter_filter",
            evolution.SETTINGS["error_signature_source"])

    def test_split_requires_positive_root_gain_but_refine_allows_tie(self) -> None:
        tie = {"technical_failure_count": 0, "root_scope_net_corrected": 0}
        gain = {"technical_failure_count": 0, "root_scope_net_corrected": 1}
        loss = {"technical_failure_count": 0, "root_scope_net_corrected": -1}
        self.assertFalse(evolution._split_passes_root_guard(tie))
        self.assertTrue(evolution._split_passes_root_guard(gain))
        self.assertTrue(evolution._refine_passes_gates("accepted", tie))
        self.assertFalse(evolution._refine_passes_gates("accepted", loss))
        self.assertFalse(evolution._refine_passes_gates(
            "competition_rejected", gain))

    def test_root_rejection_evidence_reaches_both_retry_managers(self) -> None:
        evidence = {"harmed_cases": [{"sample_id": "s1"}]}
        split_history = {"attempts": [{
            "root_id": "r1", "attempt": 1,
            "decision": split_evolution.COMPETITION_REJECTED,
            "history_payload": {}, "root_scope_evaluation": {"net": -1},
            "root_scope_evidence": evidence,
        }]}
        projected_split = split_evolution._prior(split_history, "r1")
        self.assertEqual(evidence, projected_split[0]["root_scope_evidence"])

        refine_history = {"refine_attempts": [{
            "node_id": "n1", "epoch": 1, "attempt": 1,
            "decision": "competition_rejected",
            "root_scope_evaluation": {"net": -1},
            "root_scope_evidence": evidence,
        }]}
        projected_refine = refine_evolution._refine_history_projection(
            refine_history, "n1")
        self.assertEqual(evidence, projected_refine[0]["root_scope_evidence"])


if __name__ == "__main__":
    unittest.main()

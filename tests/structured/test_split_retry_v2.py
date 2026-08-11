"""Offline contract tests for fixed-cluster Split-retry v2."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from critiq.structured import Vote
from experiments.evolving_structured_rubrics.split_evolution import (
    LOCKED_RETRY_V2_PROTOCOL,
    POLICY_V1,
    _v2_metrics,
    select_compatible_locked_children,
    validate_policy,
)


def _output(vote: Vote):
    return SimpleNamespace(parse_ok=True, answer_valid=True, vote=vote)


class SplitRetryV2Tests(unittest.TestCase):
    def _fixture(self):
        rows = tuple({"sample_id": f"s{i}", "answer": "A" if i % 4 == 1 else "B"}
                     for i in range(16))
        parent = tuple(Vote.B for _ in rows)
        strong = tuple(Vote.A if i % 4 == 1 else Vote.B for i in range(16))
        weak = tuple(Vote.B for _ in rows)
        combined = SimpleNamespace(node_outputs=tuple({
            "parent": _output(parent[i]), "strong": _output(strong[i]),
            "weak": _output(weak[i]),
        } for i in range(16)))
        diagnostics = {
            "children": [
                {"criterion_name": "strong", "cluster_id": "a", "support": 16,
                 "single_child_specialized_delta": .25, "net_corrected": 4,
                 "target": {"net_corrected": 1}},
                {"criterion_name": "weak", "cluster_id": "b", "support": 16,
                 "single_child_specialized_delta": 0.0, "net_corrected": 0,
                 "target": {"net_corrected": 0}},
            ]
        }
        return rows, combined, diagnostics

    def test_relative_gain_locks_strong_child_without_absolute_acc_gate(self):
        rows, combined, diagnostics = self._fixture()
        selected = select_compatible_locked_children(
            diagnostics, combined, rows, "parent")
        self.assertEqual(selected["locked_criterion_names"], ["strong"])
        self.assertEqual(selected["locked_metrics"]["corrected_sample_ids"],
                         ["s1", "s5", "s9", "s13"])
        self.assertGreater(selected["locked_metrics"]["specialized_accuracy"],
                           selected["locked_metrics"]["parent_accuracy"])

    def test_net_gain_boundary_does_not_lock_two_sample_improvement(self):
        rows, combined, diagnostics = self._fixture()
        diagnostics["children"][0]["net_corrected"] = 2
        selected = select_compatible_locked_children(
            diagnostics, combined, rows, "parent")
        self.assertEqual(selected["locked_criterion_names"], [])

    def test_empty_locked_set_replays_parent_vote(self):
        rows, combined, _ = self._fixture()
        parent = [row["parent"] for row in combined.node_outputs]
        children = {"strong": [row["strong"] for row in combined.node_outputs]}
        metric = _v2_metrics(parent, children, [], rows)
        self.assertEqual(metric["specialized_accuracy"], metric["parent_accuracy"])

    def test_v2_config_is_frozen(self):
        config = {
            "split_evolution": dict(POLICY_V1),
            "evolution_policy": {"trigger_thresholds": {
                "tau_split": .7, "tau_cov_high": .8}},
            "split_retry_v2_experiment": {
                "variant": "visual_grounding_locked_sample_v3",
                "root_id": "init_02_visual_grounding_and_details",
                "source_experiment_dir": "phase7_split_refine_evolution_v1",
                "max_repair_attempts": 2,
                "pairwise_endpoint": "vllm-8000",
                "retry_feedback_mode": "locked_sample_v3",
                "strong_child_min_support": 15,
                "strong_child_min_net_corrected": 3,
            },
        }
        self.assertEqual(validate_policy(config, LOCKED_RETRY_V2_PROTOCOL), POLICY_V1)


if __name__ == "__main__":
    unittest.main()

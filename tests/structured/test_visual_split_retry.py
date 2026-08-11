"""Offline tests for the Visual-Grounding Split retry feedback treatment."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from critiq.agent import AgentCallMetrics
from critiq.structured import ErrorSignature, RubricCriterionSnapshot, RubricNode, Vote
from critiq.structured.evolution.specialize_manager import SpecializeManager
from experiments.evolving_structured_rubrics.split_evolution import (
    VISUAL_RETRY_PROTOCOL,
    _history_projection,
    _retry_feedback,
    build_child_retry_diagnostics,
    validate_policy,
    POLICY_V1,
)


def _output(vote: Vote):
    return SimpleNamespace(parse_ok=True, answer_valid=True, vote=vote)


class VisualSplitRetryTests(unittest.TestCase):
    def test_control_manager_identity_is_unchanged_and_retry_is_versioned(self):
        class Pool:
            backend_id = "pool"

        control = SpecializeManager(model="model", backend_pool=Pool(),
                                    child_input_mode="text")
        memory = SpecializeManager(model="model", backend_pool=Pool(),
                                   child_input_mode="text",
                                   rubric_memory_mode="global_rubric_v1")
        retry = SpecializeManager(model="model", backend_pool=Pool(),
                                  child_input_mode="text",
                                  rubric_memory_mode="global_rubric_v1",
                                  retry_feedback_mode="child_diagnostic_v2")
        self.assertNotEqual(control.request_specs()["semantic_cluster"],
                            memory.request_specs()["semantic_cluster"])
        self.assertEqual(retry.request_specs()["semantic_cluster"].prompt_version,
                         "semantic-cluster-global-rubric-retry-v2")
        self.assertNotEqual(retry.request_specs()["child_generation"],
                            memory.request_specs()["child_generation"])
        self.assertNotEqual(retry.request_specs()["split_failure_attribution"],
                            memory.request_specs()["split_failure_attribution"])

    def test_retry_prompt_requires_and_contains_child_diagnostics(self):
        ids = tuple(f"s{i}" for i in range(5))
        response = json.dumps({"clusters": [{"cluster_id": "c", "label": "C",
            "shared_failure": "failure", "distinction": "boundary",
            "sample_ids": list(ids)}], "unclustered_sample_ids": []})

        class Pool:
            backend_id = "pool"

            def __init__(self):
                self.content = None

            def call(self, content, **kwargs):
                self.content = content
                return response, AgentCallMetrics(api_attempts=1)

        pool = Pool()
        manager = SpecializeManager(model="model", backend_pool=pool,
            child_input_mode="text", rubric_memory_mode="global_rubric_v1",
            retry_feedback_mode="child_diagnostic_v2")
        signatures = tuple(ErrorSignature(s, "task", "visual", "difference",
                                           "failure", "domain") for s in ids)
        memory = {"schema_version": "1.0.0", "root_ids": ["p"],
                  "nodes": [], "edges": []}
        with self.assertRaisesRegex(ValueError, "requires retry_feedback"):
            manager.cluster(signatures, criterion_name="parent", min_cluster_size=5,
                            max_clusters=1, rubric_memory=memory)
        feedback = {"latest": {"preservation_directives": [{
            "criterion_name": "strong_child", "action": "preserve_exact"}]}}
        manager.cluster(signatures, criterion_name="parent", min_cluster_size=5,
                        max_clusters=1, rubric_memory=memory,
                        retry_feedback=feedback)
        self.assertIn("strong_child", pool.content)
        self.assertIn("preserve_exact", pool.content)
        self.assertIn("leave-one-child-out", pool.content)

    def test_child_diagnostics_preserve_individually_strong_child(self):
        children = (
            SimpleNamespace(cluster_id="strong_cluster", criterion_name="strong_child",
                            description="Strong description"),
            SimpleNamespace(cluster_id="weak_cluster", criterion_name="weak_child",
                            description="Weak description"),
        )
        clusters = (
            SimpleNamespace(cluster_id="strong_cluster", sample_ids=("s0", "s1")),
            SimpleNamespace(cluster_id="weak_cluster", sample_ids=("s2", "s3")),
        )
        candidate = SimpleNamespace(children=children,
            cluster_proposal=SimpleNamespace(clusters=clusters))
        parent = (Vote.A, Vote.A, Vote.B, Vote.B)
        strong = (Vote.A, Vote.B, Vote.B, Vote.A)
        weak = (Vote.ABSTAIN,) * 4
        combined = SimpleNamespace(node_outputs=tuple({
            "parent": _output(parent[i]), "strong_child": _output(strong[i]),
            "weak_child": _output(weak[i])} for i in range(4)))
        rows = tuple({"sample_id": f"s{i}", "answer": gold}
                     for i, gold in enumerate(("A", "B", "B", "A")))
        evaluation = SimpleNamespace(parent_accuracy=.5, specialized_accuracy=1.0)
        result = build_child_retry_diagnostics(candidate=candidate, combined=combined,
            rows=rows, parent_node_id="parent", parent_name="parent",
            evaluation=evaluation)
        strong_result = result["children"][0]
        self.assertEqual(strong_result["support"], 4)
        # The production preservation gate requires support >= 15. Repeat the
        # synthetic pattern four times to exercise the actual strong-child gate.
        rows16 = tuple({"sample_id": f"s{i}", "answer": ("A", "B", "B", "A")[i % 4]}
                       for i in range(16))
        combined16 = SimpleNamespace(node_outputs=tuple({
            "parent": _output(parent[i % 4]), "strong_child": _output(strong[i % 4]),
            "weak_child": _output(weak[i % 4])} for i in range(16)))
        candidate.cluster_proposal.clusters[0].sample_ids = tuple(f"s{i}" for i in range(8))
        candidate.cluster_proposal.clusters[1].sample_ids = tuple(f"s{i}" for i in range(8, 16))
        result = build_child_retry_diagnostics(candidate=candidate, combined=combined16,
            rows=rows16, parent_node_id="parent", parent_name="parent",
            evaluation=evaluation)
        strong_result = result["children"][0]
        self.assertEqual(strong_result["recommended_action"], "preserve_exact")
        self.assertGreater(strong_result["accuracy_delta_same_support"], 0)
        self.assertGreater(strong_result["local_leave_one_child_out_specialized_delta"], 0)
        self.assertEqual(result["children"][1]["recommended_action"], "replace")

    def test_history_projection_and_config_are_treatment_specific(self):
        diagnostic = {"children": [{"criterion_name": "x"}],
                      "preservation_directives": [{"criterion_name": "x",
                                                   "action": "preserve_exact"}]}
        history = {"attempts": [{"root_id": "root", "attempt": 1,
            "decision": "competition_rejected", "history_payload": {
                "child_retry_diagnostics": diagnostic,
                "preservation_directives": diagnostic["preservation_directives"],
                "natural_language_attribution": {"summary": "siblings conflict"}}}]}
        control_projection = _history_projection(history, "root")
        self.assertNotIn("child_retry_diagnostics", control_projection[0])
        self.assertNotIn("preservation_directives", control_projection[0])
        projected = _history_projection(history, "root", True)
        self.assertEqual(projected[0]["child_retry_diagnostics"], diagnostic)
        feedback = _retry_feedback(projected)
        self.assertFalse(feedback["preservation_implementation"]
                         ["criterion_hash_identical_reuse"])
        config = {"split_evolution": dict(POLICY_V1),
                  "evolution_policy": {"trigger_thresholds": {
                      "tau_split": .7, "tau_cov_high": .8}},
                  "split_retry_experiment": {
                      "variant": "visual_grounding_child_diagnostic_v2",
                      "root_id": "init_02_visual_grounding_and_details",
                      "source_experiment_dir": "phase7_split_refine_evolution_v1",
                      "max_attempts": 3, "pairwise_endpoint": "vllm-8000",
                      "split_retry_feedback_mode": "child_diagnostic_v2"}}
        self.assertEqual(validate_policy(config, VISUAL_RETRY_PROTOCOL), POLICY_V1)


if __name__ == "__main__":
    unittest.main()

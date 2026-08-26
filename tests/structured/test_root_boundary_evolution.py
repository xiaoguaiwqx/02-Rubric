"""Offline contracts for Root-boundary pre-Refine evolution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from critiq.structured import (
    DualWorkerRequestSpec,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    RubricCriterionSnapshot,
    RubricNode,
    StructuredCriterionSnapshot,
    StructuredRubric,
    Vote,
    structured_input_fingerprint,
)
from critiq.structured.aggregation import aggregate_flat_votes
from experiments.evolving_structured_rubrics import root_boundary_evolution as root
from experiments.evolving_structured_rubrics import split_evolution as split


def _rubric(first="first old", second="second old"):
    nodes = {
        "r1": RubricNode("r1", RubricCriterionSnapshot("first", first, 1.0)),
        "r2": RubricNode("r2", RubricCriterionSnapshot("second", second, 1.0)),
    }
    return StructuredRubric(nodes, (), ("r1", "r2"))


def _vote(value: Vote) -> PairwiseVoteOutput:
    answer = value.value if value in {Vote.A, Vote.B} else "None"
    return PairwiseVoteOutput(
        value, True, json.dumps({"thought": "x", "answer": answer}),
        None, 1, "x", True)


def _prediction(rubric, rows, values):
    spec = DualWorkerRequestSpec(
        "worker", "vllm-8001", "a" * 64, None, False,
        "image_path", "question", "sample_id", {"temperature": .5})
    criteria = tuple(StructuredCriterionSnapshot(
        rubric.get_node(node_id).criterion.name,
        rubric.get_node(node_id).criterion.description)
        for node_id in rubric.preorder_node_ids())
    outputs = tuple({name: _vote(vote) for name, vote in item.items()}
                    for item in values)
    return PairwisePredictionOutput(
        tuple(str(item["sample_id"]) for item in rows),
        tuple(structured_input_fingerprint(
            item, image_field=spec.image_field,
            question_field=spec.question_field,
            sample_id_field=spec.sample_id_field,
            max_data_chars=spec.max_data_chars,
            encode_local_image=spec.encode_local_image) for item in rows),
        criteria, outputs,
        tuple(aggregate_flat_votes(item.vote for item in row.values())
              for row in outputs), spec)


class RootBoundaryEvolutionTests(unittest.TestCase):
    def setUp(self):
        self.rows = (
            {"sample_id": "s0", "image_path": "i0.jpg", "question": "q",
             "A": "a", "B": "b", "answer": "A"},
            {"sample_id": "s1", "image_path": "i1.jpg", "question": "q",
             "A": "a", "B": "b", "answer": "B"},
        )

    def test_protocol_keeps_gate_disabled_and_phase10_epoch_budget(self):
        self.assertFalse(root.PROTOCOL_V3["gate_worker"])
        self.assertEqual(3, root.PROTOCOL_V3["root_pre_refine_max_epochs"])
        self.assertEqual(3, root.PROTOCOL_V3["split_min_epochs"])
        self.assertEqual(5, root.PROTOCOL_V3["split_max_epochs"])
        self.assertEqual(2048, root.PROTOCOL_V3["worker_max_tokens"])
        self.assertEqual(
            ["vllm-8000", "vllm-8001"],
            root.PROTOCOL_V3["worker_endpoints"])

    def test_pairwise_pool_uses_both_available_slot_endpoints(self):
        config = {"backend_pool": {
            "pool_id": "dual",
            "common_checkpoint_id": "Qwen/Qwen3-VL-8B-Instruct",
            "global_request_concurrency": 40,
            "endpoints": [
                {"endpoint_id": "vllm-8000", "base_url": "http://10.102.137.255:8000/v1",
                 "checkpoint_root": "same", "max_concurrency": 20},
                {"endpoint_id": "vllm-8001", "base_url": "http://10.102.138.0:8000/v1",
                 "checkpoint_root": "same", "max_concurrency": 20},
            ],
        }}
        pool = root._pairwise_pool(config)
        self.assertEqual(40, pool["global_request_concurrency"])
        self.assertEqual(
            ["vllm-8000", "vllm-8001"],
            [item["endpoint_id"] for item in pool["endpoints"]])

    def test_nested_evolution_resolves_phase5_signature_cache_root(self):
        with tempfile.TemporaryDirectory() as directory:
            phase5 = Path(directory) / "rubric_evolution_phase5"
            nested = phase5 / "phase15" / "evolution"
            nested.mkdir(parents=True)
            self.assertEqual(
                phase5.resolve(), split._phase5_output_from_experiment(nested))

    def test_boundary_evidence_contains_only_peer_rubric_information(self):
        rubric = _rubric()
        evidence = root._boundary_evidence(rubric, "r1")["root_boundary_task"]
        self.assertEqual(["r2"], [item["node_id"] for item in evidence["peer_roots"]])
        serialized = json.dumps(evidence)
        self.assertNotIn("accuracy", serialized)
        self.assertNotIn("gold", serialized)
        self.assertIn("Applicable only when", serialized)

    def test_activation_diagnostic_counts_cross_root_conflict(self):
        rubric = _rubric()
        prediction = _prediction(rubric, self.rows, (
            {"first": Vote.A, "second": Vote.B},
            {"first": Vote.ABSTAIN, "second": Vote.B},
        ))
        diagnostic = root._root_activation_diagnostic(
            rubric, prediction, self.rows)
        self.assertEqual(1, diagnostic["conflict_sample_count"])
        self.assertEqual(1.5, diagnostic["mean_active_roots"])
        self.assertEqual(1, diagnostic["pairwise"][0]["joint_decisive"])
        self.assertEqual(1.0, diagnostic["pairwise"][0]["conflict_rate"])

    def test_change_diagnostic_distinguishes_safe_and_harmful_suppression(self):
        old_rubric = _rubric()
        new_rubric = _rubric(first="first refined")
        old = _prediction(old_rubric, self.rows, (
            {"first": Vote.A, "second": Vote.ABSTAIN},
            {"first": Vote.A, "second": Vote.ABSTAIN},
        ))
        new = _prediction(new_rubric, self.rows, (
            {"first": Vote.ABSTAIN, "second": Vote.ABSTAIN},
            {"first": Vote.ABSTAIN, "second": Vote.ABSTAIN},
        ))
        value = root._root_change_diagnostic(
            old_rubric, old, new_rubric, new, self.rows)
        first = value["roots"][0]
        self.assertEqual(1, first["suppressed_old_correct"])
        self.assertEqual(1, first["suppressed_old_wrong"])
        self.assertEqual(.5, first["suppression_precision"])
        self.assertTrue(first["description_changed"])


if __name__ == "__main__":
    unittest.main()

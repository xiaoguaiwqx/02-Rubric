"""Offline tests for the VL-RewardBench Full Child-Gate transfer runner."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.evolving_structured_rubrics import full_child_gate as full
from experiments.evolving_structured_rubrics import vl_rewardbench_child_gate as vl_gate
from experiments.evolving_structured_rubrics.run_rubric_evolution import _config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (ROOT / "experiments/evolving_structured_rubrics/configs"
          / "rubric_evolution_phase5.example.json")
OUTPUT = ROOT / "output/evolving_structured_rubrics/rubric_evolution_phase5"


class VLRBChildGateProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _config(CONFIG)

    def test_k3_replicates_use_distinct_frozen_seeds(self):
        protocol = vl_gate._protocol(self.config)
        self.assertEqual([42, 43, 44], protocol["replicate_seeds"])
        self.assertFalse(protocol["cross_replicate_cache_reuse"])

    def test_replicate_request_identities_differ(self):
        rubric = vl_gate._rubric(OUTPUT)
        contracts = full._contracts(rubric)
        spec = full.vg._gate_pool_spec(
            self.config, ("vllm-8000", "vllm-8001"))
        specs = vl_gate._replicate_request_specs(
            self.config, vl_gate._protocol(self.config), contracts, spec)
        seeds = [
            specs[f"replicate_{index:02d}"][rubric.root_ids[0]][
                "decoding_config"]["seed"]
            for index in (1, 2, 3)
        ]
        self.assertEqual([42, 43, 44], seeds)
        self.assertEqual(3, len({json.dumps(value, sort_keys=True)
                                for value in specs.values()}))

    def test_source_prediction_paths_are_replicate_specific(self):
        paths = [vl_gate._source_prediction_path(index) for index in range(3)]
        self.assertEqual(3, len(set(paths)))
        self.assertEqual("replicate_01", paths[0].parent.name)
        self.assertEqual("predictions.json", paths[0].name)
        self.assertEqual("replicate_03", paths[2].parent.name)


class VLRBChildGateMetricTest(unittest.TestCase):
    def test_display_votes_map_back_to_original_order(self):
        records = ({"sample_id": "s1"}, {"sample_id": "s2"})
        schedule = {"s1": (0, 1, 0), "s2": (1, 0, 1)}
        self.assertEqual(
            [0, 1],
            vl_gate._display_to_original(
                ["A", "A"], records, schedule, replicate=0))
        self.assertEqual(
            [0, 1],
            vl_gate._display_to_original(
                ["B", "B"], records, schedule, replicate=1))

    def test_routing_summary_aggregates_all_three_replicates(self):
        diagnostic = {
            "mean_active_children_total": 4.0,
            "roots": {
                "r": {
                    "mean_active_children": 4.0,
                    "empty_route_count": 0,
                    "all_children_sibling_conflict_count": 10,
                    "gated_sibling_conflict_count": 2,
                    "per_child": {
                        "c": {"active": 2, "decisive": 2, "correct": 1},
                    },
                }
            },
        }
        artifact = {
            "roots": {
                "r": {"samples": [
                    {"sample_id": "s1", "parse_ok": True, "parse_error": None},
                    {"sample_id": "s2", "parse_ok": True, "parse_error": None},
                ]}
            }
        }
        value = vl_gate._routing_summary(
            [diagnostic, diagnostic, diagnostic],
            [artifact, artifact, artifact], sample_count=2)
        self.assertEqual(1.0, value["parse_valid_rate"])
        self.assertEqual(4.0, value["mean_active_children_total"])
        self.assertEqual(30, value["all_children_sibling_conflict_count"])
        self.assertEqual(6, value["gated_sibling_conflict_count"])

    def test_gate_stability_distinguishes_independent_replicates(self):
        rubric = vl_gate._rubric(OUTPUT)
        root_id = rubric.root_ids[0]
        child_ids = [child.node_id for child in rubric.children(root_id)]

        def artifact(statuses):
            decisions = {
                child_id: {"status": status, "reason": "r"}
                for child_id, status in zip(child_ids, statuses)
            }
            active = [child_id for child_id, status in zip(child_ids, statuses)
                      if status in {"applicable", "uncertain"}]
            roots = {}
            for current_root in rubric.root_ids:
                current_children = [child.node_id for child in rubric.children(current_root)]
                if current_root == root_id:
                    sample = {"parse_ok": True, "decision": {"decisions": decisions},
                              "active_child_ids": active}
                else:
                    sample = {
                        "parse_ok": True,
                        "decision": {"decisions": {
                            child_id: {"status": "not_applicable", "reason": "r"}
                            for child_id in current_children}},
                        "active_child_ids": [],
                    }
                roots[current_root] = {"samples": [sample]}
            return {"roots": roots}

        first = artifact(["applicable", "not_applicable", "not_applicable"])
        second = artifact(["applicable", "not_applicable", "not_applicable"])
        third = artifact(["not_applicable", "not_applicable", "not_applicable"])
        value = vl_gate._gate_stability((first, second, third), rubric)
        root = value["roots"][root_id]
        self.assertEqual(
            0.0,
            root["replicate_01_vs_03_same_order_independent"]
                ["exact_status_vector_match_rate"])
        self.assertEqual(
            1.0,
            root["replicate_01_vs_02_swapped_order"]
                ["exact_status_vector_match_rate"])


if __name__ == "__main__":
    unittest.main()

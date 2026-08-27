"""Offline tests for the Qwen2.5 Clean S5-v2 transfer experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from critiq.structured import RubricCriterionSnapshot, RubricNode, StructuredRubric
from experiments.evolving_structured_rubrics import (
    global_arbiter_ab_only as arbiter,
    internal_global_arbiter_k1 as internal,
    qwen25_clean_s5_transfer as experiment,
)


def _rubric() -> StructuredRubric:
    return StructuredRubric(
        nodes={
            "root": RubricNode(
                node_id="root",
                criterion=RubricCriterionSnapshot(
                    name="root_focus", description="Judge the decisive evidence."),
            ),
        },
        edges=(),
        root_ids=("root",),
    )


class TestQwen25CleanS5Transfer(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json")
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def test_config_freezes_model_only_transfer_protocol(self) -> None:
        settings = experiment._settings(self.config)
        self.assertEqual("Qwen/Qwen2.5-VL-7B-Instruct", settings["worker_model"])
        self.assertEqual(1, settings["internal_k"])
        self.assertEqual(3, settings["vl_rewardbench_k"])
        self.assertEqual(2048, settings["max_tokens"])
        self.assertEqual("unset", settings["generation_seed_policy"])
        self.assertEqual(["A", "B", "None"], settings["semantic_answer_space"])

    def test_prompts_and_parsers_are_imported_from_clean_s5(self) -> None:
        self.assertIs(
            experiment.internal.UNIFIED_SUBTREE_SYSTEM_PROMPT,
            internal.UNIFIED_SUBTREE_SYSTEM_PROMPT,
        )
        self.assertIs(
            experiment.arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
            arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        )
        self.assertEqual(
            "None", experiment.internal.parse_subtree_response(
                '{"answer":"None"}')["answer"])
        self.assertEqual(
            "None", experiment.arbiter.parse_global_arbiter_ab_only_response(
                '{"answer":"None"}')["answer"])

    def test_request_budgets_are_exact(self) -> None:
        self.assertEqual(4500, sum(experiment.INTERNAL_COUNTS.values()) * 6)
        self.assertEqual(22446, experiment.VLRB_COUNT * experiment.VLRB_K * 6)
        self.assertEqual(26946, 4500 + 22446)

    def test_vlrb_mapping_respects_order(self) -> None:
        result = {"samples": [{
            "orders": [0, 1, 0],
            "replicates": {
                "0": {"order": 0, "arbiter": {
                    "parse_ok": True, "parsed": {"answer": "A"}}},
                "1": {"order": 1, "arbiter": {
                    "parse_ok": True, "parsed": {"answer": "B"}}},
                "2": {"order": 0, "arbiter": {
                    "parse_ok": True, "parsed": {"answer": "None"}}},
            },
        }]}
        self.assertEqual([[0], [0], [None]], experiment._vlrb_vote_matrix(result))

    def test_bundle_scheduler_uses_subtree_then_arbiter_per_replicate(self) -> None:
        rows = tuple({
            "sample_id": f"sample-{index}", "image_path": "unused.png",
            "question": "q", "A": "a", "B": "b", "answer": "A",
        } for index in range(2))
        orders = {"sample-0": (0, 1, 0), "sample-1": (1, 0, 1)}

        def fake_call(*args, **kwargs):
            endpoint = args[1]
            return {
                "parse_ok": True,
                "parsed": {"analysis_a": "a", "analysis_b": "b",
                           "thought": "t", "answer": "A"},
                "model_generation_count": 1,
                "endpoint_id": endpoint.endpoint_id,
                "cache_hit": False,
                "metrics": {},
                "cache_key": "fake",
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(
                experiment.support, "call_one", side_effect=fake_call) as mocked:
            result = experiment._run_rows(
                self.config, Path(directory), "test", "test_k3",
                rows, orders, _rubric(), 1)
        self.assertEqual(12, mocked.call_count)
        self.assertEqual(2, len(result["samples"]))
        kinds = Counter(
            call.kwargs["request_key"]["kind"]
            for call in mocked.call_args_list)
        self.assertEqual(6, kinds["qwen25_clean_s5_unified_subtree"])
        self.assertEqual(6, kinds["qwen25_clean_s5_global_arbiter"])

    def test_unresolved_subtree_blocks_only_corresponding_arbiter(self) -> None:
        calls = []

        def fake_call(*args, **kwargs):
            calls.append(kwargs["request_key"]["kind"])
            if kwargs["request_key"]["kind"].endswith("unified_subtree"):
                return {"parse_ok": False, "parsed": None,
                        "model_generation_count": 1,
                        "endpoint_id": args[1].endpoint_id,
                        "metrics": {}, "error": "bad json"}
            raise AssertionError("Arbiter must not run with an unresolved subtree")

        endpoint = experiment._pool(self.config).endpoints[0]
        row = {"sample_id": "s", "image_path": "unused.png",
               "question": "q", "A": "a", "B": "b"}
        with tempfile.TemporaryDirectory() as directory, patch.object(
                experiment.support, "call_one", side_effect=fake_call):
            result = experiment._process_sample(
                experiment._worker_config(self.config), Path(directory), endpoint,
                "test", row, (0,), _rubric(), 1, "test")
        self.assertEqual(["qwen25_clean_s5_unified_subtree"], calls)
        self.assertEqual(
            "blocked_by_unresolved_subtree",
            result["replicates"]["0"]["arbiter"]["error"],
        )


if __name__ == "__main__":
    unittest.main()

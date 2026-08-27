"""Offline tests for the ordered Global-Arbiter evidence/priority ablation."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from experiments.evolving_structured_rubrics import (
    global_arbiter_ab_only as formal,
)
from experiments.evolving_structured_rubrics import (
    global_arbiter_evidence_priority_ablation as ablation,
)


class TestGlobalArbiterEvidencePriorityAblation(unittest.TestCase):
    def test_neutral_prompt_removes_exactly_the_priority_paragraph(self) -> None:
        expected = formal.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.replace(
            ablation.FACTUALITY_FIRST_PARAGRAPH, "")
        self.assertEqual(expected, ablation.NEUTRAL_SYSTEM_PROMPT)
        self.assertNotIn(
            "Prioritize verifiable visual and factual correctness",
            ablation.NEUTRAL_SYSTEM_PROMPT,
        )
        for text in (
            "correlated evidence", "Do not decide by counting",
            "Do not abstain", '"answer": "A / B"',
        ):
            self.assertIn(text, ablation.NEUTRAL_SYSTEM_PROMPT)

    def test_label_only_and_full_report_share_pair_prefix_and_root_names(self) -> None:
        row = {"question": "q", "A": "a", "B": "b"}
        reports = [{
            "root_id": "r1",
            "criterion_name": "Visual Grounding",
            "report": {
                "analysis_a": "A evidence", "analysis_b": "B evidence",
                "thought": "comparison", "answer": "A",
            },
        }, {
            "root_id": "r2",
            "criterion_name": "Clarity",
            "report": {
                "analysis_a": "A clear", "analysis_b": "B unclear",
                "thought": "comparison", "answer": "None",
            },
        }]
        label = ablation._label_only_user_prompt(row, reports)
        full = ablation._full_report_user_prompt(row, reports)
        self.assertEqual(
            label.split("## Subtree Assessments")[0],
            full.split("## Subtree Assessments")[0],
        )
        for name in ("Visual Grounding", "Clarity"):
            self.assertIn(name, label)
            self.assertIn(name, full)
        for field in ('"analysis_a"', '"analysis_b"', '"thought"'):
            self.assertNotIn(field, label)
            self.assertIn(field, full)
        self.assertIn('{"answer":"A"}', label)
        self.assertIn('{"answer":"None"}', label)

    def test_formal_prompt_identity_matches_frozen_hash(self) -> None:
        value = hashlib.sha256(
            formal.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
        self.assertEqual(ablation.V2_SYSTEM_SHA256, value)

    def test_config_matches_exact_frozen_settings(self) -> None:
        path = Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        settings = ablation._settings(config)
        self.assertEqual(50, settings["smoke_sample_count"])
        self.assertEqual(3, settings["k"])
        self.assertEqual(["A", "B", "None"], settings[
            "semantic_answer_space"])
        self.assertFalse(settings["variants"][ablation.V0]["reuse"])
        self.assertFalse(settings["variants"][ablation.V1]["reuse"])
        self.assertTrue(settings["variants"][ablation.V2]["reuse"])

    def test_smoke_selector_has_exact_group_quotas_and_source_coverage(self) -> None:
        records = []
        specifications = {
            "general": (("wildvision-battle", 100), ("misc", 10)),
            "hallucination": (("hallucination_pair", 250),
                              ("RLAIF-V", 100), ("RLHF-V", 50)),
            "reasoning": (("mathverse", 200),),
        }
        for group, sources in specifications.items():
            for source, count in sources:
                records.extend({
                    "sample_id": f"{source}-{index}",
                    "benchmark_id": f"{source}-{index}",
                    "group": group,
                    "query_source": source,
                } for index in range(count))
        selected = ablation._smoke_records(records, 42)
        self.assertEqual(50, len(selected))
        self.assertEqual(
            {"general": 7, "hallucination": 30, "reasoning": 13},
            dict(Counter(str(item["group"]) for item in selected)),
        )
        self.assertEqual(
            {"wildvision-battle", "vlfeedback", "povid", "rlaif-v",
             "rlhf-v", "reasoning_tasks"},
            {ablation.vlrb._official_dataset(str(item["benchmark_id"]))
             for item in selected},
        )

    def test_paired_bootstrap_is_deterministic_and_strict(self) -> None:
        records = [
            {"preferred_original_index": index % 2,
             "group": ("general", "hallucination", "reasoning")[index % 3]}
            for index in range(30)
        ]
        before = [None] * 30
        after = [int(item["preferred_original_index"]) for item in records]
        left = ablation._paired_bootstrap_ci(
            records, before, after, iterations=100, seed=7)
        right = ablation._paired_bootstrap_ci(
            records, before, after, iterations=100, seed=7)
        self.assertEqual(left, right)
        self.assertEqual([1.0, 1.0], left)

    def test_holm_adjustment_for_two_registered_comparisons(self) -> None:
        value = ablation._holm_adjusted({"report": 0.01, "priority": 0.04})
        self.assertEqual(0.02, value["report"])
        self.assertEqual(0.04, value["priority"])

    def test_retry_merge_preserves_full_timing_and_replaces_only_failed_sample(
            self) -> None:
        base = {
            "samples": [
                {"sample_id": "a", "value": "initial-a"},
                {"sample_id": "b", "value": "initial-b"},
            ],
            "wall_seconds": 90.0,
            "endpoint_assignment": {"a": "vllm-8000", "b": "vllm-8001"},
        }
        patch = {
            "samples": [{"sample_id": "b", "value": "recovered-b"}],
            "wall_seconds": 7.0,
            "endpoint_assignment": {"b": "vllm-8001"},
        }
        value = ablation._merge_retry_prediction(
            base, patch, variant=ablation.V0, smoke_wall_seconds=12.0,
            initial_full_wall_seconds=90.0)
        self.assertEqual("initial-a", value["samples"][0]["value"])
        self.assertEqual("recovered-b", value["samples"][1]["value"])
        self.assertEqual(12.0, value["smoke_wall_seconds"])
        self.assertEqual(90.0, value["initial_full_wall_seconds"])
        self.assertEqual(7.0, value["retry_wall_seconds"])
        self.assertEqual(109.0, value["wall_seconds"])

    def test_repeated_retry_accumulates_only_retry_wall_time(self) -> None:
        base = {
            "samples": [{"sample_id": "a"}],
            "smoke_wall_seconds": 12.0,
            "initial_full_wall_seconds": 90.0,
            "retry_wall_seconds": 7.0,
            "wall_seconds": 109.0,
        }
        patch = {"samples": [{"sample_id": "a"}], "wall_seconds": 3.0}
        value = ablation._merge_retry_prediction(
            base, patch, variant=ablation.V1, smoke_wall_seconds=12.0,
            initial_full_wall_seconds=90.0)
        self.assertEqual(10.0, value["retry_wall_seconds"])
        self.assertEqual(112.0, value["wall_seconds"])

    def test_stage_namespace_is_complete(self) -> None:
        self.assertEqual(6, len(ablation.STAGES))
        self.assertTrue(all(stage.startswith("vlrb-arbiter-ablation-")
                            for stage in ablation.STAGES))


if __name__ == "__main__":
    unittest.main()

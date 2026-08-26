"""Focused offline tests for the A/B-only Global-Arbiter ablation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.evolving_structured_rubrics import global_arbiter_ab_only as ab_only


class TestGlobalArbiterABOnlyExperiment(unittest.TestCase):
    def test_prompt_preserves_evidence_synthesis_but_removes_abstention(self) -> None:
        prompt = ab_only.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT
        for key in ("analysis_a", "analysis_b", "thought", "answer"):
            self.assertIn(f'"{key}"', prompt)
        self.assertIn("correlated evidence", prompt)
        self.assertIn("Do not decide by counting", prompt)
        self.assertIn('"answer": "A / B"', prompt)
        self.assertIn("Do not abstain", prompt)
        self.assertNotIn("A / B / None", prompt)

    def test_parser_accepts_ab_and_semantic_none(self) -> None:
        self.assertEqual(
            {"answer": "A"},
            ab_only.parse_global_arbiter_ab_only_response(
                '{"analysis_a":{},"thought":null,"answer":"A"}'),
        )
        self.assertEqual(
            {"answer": "B"},
            ab_only.parse_global_arbiter_ab_only_response('{"answer":"B"}'),
        )
        self.assertEqual(
            {"answer": "None"},
            ab_only.parse_global_arbiter_ab_only_response('{"answer":"None"}'),
        )
        for answer in ("C", "", None):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                ab_only.parse_global_arbiter_ab_only_response(
                    json.dumps({"answer": answer}))

    def test_formal_s5_path_maps_each_arbiter_replicate_without_root_vote(self) -> None:
        result = {"samples": [{
            "sample_id": "s0",
            "orders": [0, 1, 0],
            "arbiter": {
                "0": {"parse_ok": True, "parsed": {"answer": "A"}},
                "1": {"parse_ok": True, "parsed": {"answer": "B"}},
                "2": {"parse_ok": True, "parsed": {"answer": "B"}},
            },
        }]}
        with patch.object(
                ab_only.support, "_five_root", side_effect=AssertionError):
            votes = ab_only._s5_vote_matrices(result)
        self.assertEqual([[0], [0], [1]], votes)

    def test_s5_rejects_unresolved_but_maps_semantic_none(self) -> None:
        unresolved = {"samples": [{
            "orders": [0, 1, 0],
            "arbiter": {
                "0": {"parse_ok": False, "parsed": None},
                "1": {"parse_ok": True, "parsed": {"answer": "A"}},
                "2": {"parse_ok": True, "parsed": {"answer": "B"}},
            },
        }]}
        with self.assertRaises(RuntimeError):
            ab_only._s5_vote_matrices(unresolved)

        with_none = json.loads(json.dumps(unresolved))
        with_none["samples"][0]["arbiter"]["0"] = {
            "parse_ok": True, "parsed": {"answer": "None"}}
        self.assertEqual(
            [[None], [1], [1]], ab_only._s5_vote_matrices(with_none))

    def test_former_none_subset_reports_required_recovery(self) -> None:
        records = [
            {"sample_id": f"s{index}", "preferred_original_index": 0,
             "group": group}
            for index, group in enumerate(
                ("general", "hallucination", "reasoning", "general"))
        ]
        value = ab_only._former_s4_none_subset(
            records,
            s0=[0, 0, 0, 1],
            s4=[0, None, None, 1],
            s5=[0, 0, 1, 0],
        )
        self.assertEqual(2, value["sample_count"])
        self.assertEqual(1, value["correct_count"])
        self.assertEqual(0.5, value["accuracy"])
        self.assertEqual(2, value["required_correct_to_match_s0"])
        self.assertEqual(3, value["required_correct_to_exceed_s0"])

    def test_transition_separates_none_recovery_from_harm(self) -> None:
        records = [
            {"preferred_original_index": 0},
            {"preferred_original_index": 0},
            {"preferred_original_index": 1},
        ]
        value = ab_only._transition(records, [None, 0, 1], [0, 1, 0])
        self.assertEqual(3, value["changed_prediction_count"])
        self.assertEqual(1, value["transitions"]["none->correct"])
        self.assertEqual(2, value["transitions"]["correct->wrong"])

    def test_example_config_matches_frozen_settings(self) -> None:
        config_path = Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        settings = ab_only._settings(config)
        self.assertEqual(["A", "B", "None"], settings["semantic_answer_space"])
        self.assertEqual(
            "s5_global_arbiter_ab_only_all_requests",
            settings["new_request_scope"],
        )

    def test_legacy_binary_manifest_is_parser_migration_compatible(self) -> None:
        legacy = {
            "settings": {"semantic_answer_space": ["A", "B"]},
            "settings_sha256": "legacy",
            "semantic_answer_space": ["A", "B"],
        }
        current = {
            "settings": {"semantic_answer_space": ["A", "B", "None"]},
            "settings_sha256": ab_only.canonical_sha256(
                {"semantic_answer_space": ["A", "B", "None"]}),
            "semantic_answer_space": ["A", "B", "None"],
        }
        self.assertEqual(
            ab_only._manifest_identity(current),
            ab_only._manifest_identity(legacy),
        )

    def test_stage_names_are_disjoint_from_s4_prefix_dispatch(self) -> None:
        self.assertEqual(6, len(ab_only.STAGES))
        self.assertTrue(all(
            stage.startswith("vlrb-global-arbiter-ab-only-")
            for stage in ab_only.STAGES))

    def test_retry_attempt_limit_extends_only_after_unresolved_failures(self) -> None:
        target = Path("output/test_global_arbiter_ab_only_retry")
        prediction = target / "predictions" / "arbiter_ab_only_full.json"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        try:
            prediction.write_text(json.dumps({"samples": [{
                "sample_id": "s0",
                "arbiter": {
                    "0": {"parse_ok": False, "model_generation_count": 11,
                          "error": "parse"},
                    "1": {"parse_ok": True, "model_generation_count": 1,
                          "parsed": {"answer": "A"}},
                },
            }]}), encoding="utf-8")
            limit, details = ab_only._retry_attempt_limit(target, 10)
            self.assertEqual(21, limit)
            self.assertEqual(1, details["previous_unresolved_technical_failures"])
            self.assertEqual(11, details["previous_failed_max_generation_count"])
            self.assertEqual(10, details["additional_retry_attempts_per_failed_call"])

            value = json.loads(prediction.read_text(encoding="utf-8"))
            value["samples"][0]["arbiter"]["0"][
                "model_generation_count"] = 21
            prediction.write_text(json.dumps(value), encoding="utf-8")
            limit, details = ab_only._retry_attempt_limit(target, 10)
            self.assertEqual(31, limit)
            self.assertEqual(10, details["additional_retry_attempts_per_failed_call"])
        finally:
            prediction.unlink(missing_ok=True)
            prediction.parent.rmdir()
            target.rmdir()

    def test_retry_migrates_legacy_tiebreak_calls_back_to_primary(self) -> None:
        target = Path("output/test_global_arbiter_legacy_tiebreak")
        prediction = target / "predictions" / "arbiter_ab_only_full.json"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        try:
            prediction.write_text(json.dumps({"samples": [{
                "sample_id": "s0",
                "arbiter": {"0": {
                    "parse_ok": True,
                    "parsed": {"answer": "A"},
                    "recovery_mode": "mandatory_ab_tiebreak",
                    "primary_model_generation_count": 21,
                }},
            }]}), encoding="utf-8")
            limit, details = ab_only._retry_attempt_limit(target, 10)
            self.assertEqual(31, limit)
            self.assertEqual(1, details["legacy_tiebreak_calls_to_remove"])
        finally:
            prediction.unlink(missing_ok=True)
            prediction.parent.rmdir()
            target.rmdir()


if __name__ == "__main__":
    unittest.main()

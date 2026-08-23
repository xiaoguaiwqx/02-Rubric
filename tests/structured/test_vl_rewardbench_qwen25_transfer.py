"""Offline tests for the Qwen2.5-VL Phase17 E4 transfer protocol."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.evolving_structured_rubrics import (
    vl_rewardbench_phase10_capped as capped,
)
from experiments.evolving_structured_rubrics import (
    vl_rewardbench_qwen25_transfer as transfer,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "experiments/evolving_structured_rubrics/configs/"
    "rubric_evolution_phase5.example.json"
)


class TestQwen25WorkerTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_matches_frozen_protocol(self) -> None:
        self.assertEqual(self.config[transfer.CONFIG_KEY], transfer.SETTINGS)
        self.assertEqual(transfer._protocol(self.config), transfer.SETTINGS)

    def test_worker_view_changes_only_experiment_local_model_identity(self) -> None:
        original_model = self.config["model"]
        original_pool = self.config["backend_pool"]["common_checkpoint_id"]
        worker = transfer._worker_config(self.config)
        self.assertEqual(worker["model"], transfer.WORKER_MODEL)
        self.assertEqual(
            worker["backend_pool"]["common_checkpoint_id"],
            transfer.WORKER_MODEL,
        )
        self.assertEqual(worker["worker_request_kwargs"], {
            "temperature": 0.5, "max_tokens": 2048})
        self.assertEqual(self.config["model"], original_model)
        self.assertEqual(
            self.config["backend_pool"]["common_checkpoint_id"], original_pool)

    def test_native_recovery_protocol_matches_historical_chain(self) -> None:
        settings = transfer.SETTINGS
        self.assertEqual(
            settings["native_regex_parser"],
            "vl_rewardbench_overall_judgment_regex_v1",
        )
        self.assertEqual(
            settings["native_fallback_model"], capped.NATIVE_FALLBACK_MODEL)
        self.assertEqual(settings["native_retry_max_attempts"], 10)
        self.assertEqual(settings["native_decoding"]["max_tokens"], 2048)

    def test_fallback_parser_extracts_only_explicit_answer(self) -> None:
        self.assertEqual(
            capped._parse_native_fallback('{"choice":"Answer 1"}'), 1)
        self.assertEqual(
            capped._parse_native_fallback('{"choice":"Answer 2"}'), 2)
        self.assertIsNone(
            capped._parse_native_fallback('{"choice":"Unclear"}'))

    def test_primary_systems_share_structured_prediction_set(self) -> None:
        self.assertEqual(transfer.SOURCE_EPOCH, 4)
        self.assertEqual(transfer.SOURCE_NODE_COUNT, 27)
        self.assertNotEqual(transfer.INITIAL_SYSTEM, transfer.E4_SYSTEM)

    def test_unknown_stage_fails_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            transfer.run_stage({}, Path("."), "not-a-stage")


if __name__ == "__main__":
    unittest.main()

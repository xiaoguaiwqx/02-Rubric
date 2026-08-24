"""Offline protocol tests for Qwen2.5-specific full evolution."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.evolving_structured_rubrics import (
    qwen25_full_evolution as evolution,
)
from experiments.evolving_structured_rubrics import (
    vl_rewardbench_qwen25_evolved as vlrb,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "experiments/evolving_structured_rubrics/configs/"
    "rubric_evolution_phase5.example.json"
)


class TestQwen25FullEvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_matches_both_frozen_protocols(self) -> None:
        self.assertEqual(
            self.config[evolution.CONFIG_KEY], evolution.SETTINGS)
        self.assertEqual(self.config[vlrb.CONFIG_KEY], vlrb.SETTINGS)

    def test_worker_adapter_is_local_and_qwen25(self) -> None:
        original_model = self.config["model"]
        original_pool = self.config["backend_pool"]["common_checkpoint_id"]
        worker = evolution._worker_config(self.config)
        self.assertEqual(worker["model"], evolution.WORKER_MODEL)
        self.assertEqual(
            worker["backend_pool"]["common_checkpoint_id"],
            evolution.WORKER_MODEL,
        )
        self.assertEqual(worker["worker_request_kwargs"], {
            "temperature": 0.5, "max_tokens": 2048})
        self.assertEqual(self.config["model"], original_model)
        self.assertEqual(
            self.config["backend_pool"]["common_checkpoint_id"],
            original_pool,
        )

    def test_evolution_forbids_qwen3_trajectory_reuse(self) -> None:
        self.assertFalse(evolution.PROTOCOL.read_only_control_signatures)
        self.assertFalse(evolution.PROTOCOL.allow_legacy_signature_reuse)
        self.assertIsNone(evolution.PROTOCOL.control_experiment_dir)
        self.assertFalse(
            evolution.SETTINGS["qwen3_evolution_artifact_reuse"])
        self.assertEqual(
            evolution.SETTINGS["error_signature_policy"],
            "fresh_qwen25_discovery100_only",
        )

    def test_dev_and_external_sets_do_not_select(self) -> None:
        self.assertEqual(
            evolution.SETTINGS["dev_policy"],
            "diagnostic_only_no_selection",
        )
        self.assertEqual(
            evolution.SETTINGS["heldout_access"], "final_stage_only")
        self.assertTrue(
            vlrb.SETTINGS["selection_after_benchmark_forbidden"])

    def test_vlrb_uses_existing_transfer_as_control(self) -> None:
        self.assertEqual(
            vlrb.SETTINGS["control_experiment"],
            "vl_rewardbench_qwen25_phase17_e4_transfer_v1",
        )
        self.assertFalse(vlrb.SETTINGS["native_rerun"])
        self.assertEqual(vlrb.SETTINGS["k"], 3)
        self.assertEqual(
            vlrb.SETTINGS["scheduler"],
            "sample_major_available_slot_dynamic",
        )

    def test_unknown_stages_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            evolution.run_stage({}, Path("."), "not-a-stage")
        with self.assertRaises(ValueError):
            vlrb.run_stage({}, Path("."), "not-a-stage")

    def test_run_stops_stage_chain_when_shared_runner_pauses(self) -> None:
        with TemporaryDirectory(dir=".") as directory:
            output = Path(directory)
            target = output / evolution.EXPERIMENT_DIR
            target.mkdir(parents=True)
            (target / "frozen_manifest.json").write_text(
                "{}", encoding="utf-8")
            (target / "stage_status.json").write_text(json.dumps({
                "run": {"status": "paused", "details": {
                    "stage": "pairwise_worker"}},
            }), encoding="utf-8")
            with (
                patch.object(evolution, "_activate"),
                patch.object(evolution, "_adapt_config", return_value={}),
                patch.object(evolution.shared, "_initialize_baseline"),
                patch.object(evolution, "_validate_baseline_identity"),
                patch.object(evolution.shared, "run"),
            ):
                with self.assertRaisesRegex(
                        RuntimeError, "did not complete"):
                    evolution.run({}, output)

    def test_vlrb_retry_reads_the_evolved_run_layout(self) -> None:
        with TemporaryDirectory(dir=".") as directory:
            target = Path(directory)
            report = target / "retry/structured/report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({
                "target_count": 0,
                "recovered_count": 0,
                "still_failed_count": 0,
                "new_model_requests": 0,
            }), encoding="utf-8")
            with (
                patch.object(vlrb, "_load_frozen", return_value=(
                    target, {}, (), (), object())),
                patch.object(vlrb, "_require"),
                patch.object(vlrb, "_verify_live"),
                patch.object(vlrb.transfer, "_retry_structured",
                             return_value=()) as retry_structured,
                patch.object(vlrb, "_status"),
            ):
                vlrb.retry({}, Path("."))
            self.assertEqual(
                retry_structured.call_args.kwargs["source_work"],
                target / "run",
            )


if __name__ == "__main__":
    unittest.main()

"""Offline protocol tests for the Phase22 VL-RewardBench wrapper."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.evolving_structured_rubrics import (
    all_sample_adaptive_recluster_evolution as phase22,
)
from experiments.evolving_structured_rubrics import (
    unified_subtree_bundle_evolution as phase21_engine,
)
from experiments.evolving_structured_rubrics import (
    vl_rewardbench_all_sample_adaptive_recluster_evolution as evolution,
)
from experiments.evolving_structured_rubrics import (
    vl_rewardbench_unified_subtree_bundle_evolution as phase21_vlrb,
)


class TestPhase22VLRewardBenchProtocol(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))

    def test_config_and_selection_contract_are_frozen(self) -> None:
        self.assertEqual(evolution.SETTINGS, evolution._settings(self._config()))
        self.assertEqual(3, evolution.SETTINGS["k"])
        self.assertTrue(
            evolution.SETTINGS["selection_after_benchmark_forbidden"])
        self.assertEqual(
            ["initial_five_root", "phase17_e4_control", "phase21_final",
             "phase22_final"],
            evolution.SETTINGS["systems"])

    def test_phase22_vlrb_engine_passes_both_protocols_explicitly(self) -> None:
        old_vlrb = (
            phase21_vlrb.EXPERIMENT_DIR, phase21_vlrb.PROTOCOL_VERSION,
            phase21_vlrb.STAGES, phase21_vlrb.SETTINGS,
            phase21_vlrb.FINAL_LABEL, phase21_vlrb.SOURCE_RUBRIC_KEY,
        )
        old_engine = (
            phase21_engine.EXPERIMENT_DIR, phase21_engine.PROTOCOL_VERSION,
            phase21_engine.SETTINGS,
        )
        config = self._config()
        with patch.object(phase21_vlrb, "run") as action:
            evolution.run(config, Path("unused"))
        action.assert_called_once_with(
            config, Path("unused"), protocol=evolution.PROTOCOL)
        self.assertEqual("phase22_final", evolution.PROTOCOL.final_label)
        self.assertIs(phase22.PROTOCOL, evolution.PROTOCOL.source)
        self.assertEqual(old_vlrb, (
            phase21_vlrb.EXPERIMENT_DIR, phase21_vlrb.PROTOCOL_VERSION,
            phase21_vlrb.STAGES, phase21_vlrb.SETTINGS,
            phase21_vlrb.FINAL_LABEL, phase21_vlrb.SOURCE_RUBRIC_KEY,
        ))
        self.assertEqual(old_engine, (
            phase21_engine.EXPERIMENT_DIR, phase21_engine.PROTOCOL_VERSION,
            phase21_engine.SETTINGS,
        ))

    def test_benchmark_stage_names_cannot_dispatch_evolution_selection(self) -> None:
        self.assertTrue(all(
            stage.startswith("vlrb-all-sample-adaptive-")
            for stage in evolution.STAGES))
        self.assertNotIn("run", evolution.STAGES)
        self.assertEqual("phase22_final", evolution.FINAL_LABEL)


if __name__ == "__main__":
    unittest.main()

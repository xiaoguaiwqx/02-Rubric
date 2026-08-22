"""Offline tests for the Phase17 VL-RewardBench checkpoint diagnostic."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from critiq.structured import (
    RubricCriterionSnapshot,
    RubricNode,
    StructuredRubric,
)

from experiments.evolving_structured_rubrics import (
    vl_rewardbench_phase16_checkpoints as shared,
)
from experiments.evolving_structured_rubrics import (
    vl_rewardbench_phase17_checkpoints as phase17,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "experiments/evolving_structured_rubrics/configs/"
    "rubric_evolution_phase5.example.json"
)


def _node(node_id: str, description: str) -> RubricNode:
    return RubricNode(
        node_id=node_id,
        criterion=RubricCriterionSnapshot(
            name=f"criterion_{node_id}", description=description, score=1.0),
    )


def _rubric(*nodes: RubricNode) -> StructuredRubric:
    return StructuredRubric(
        {node.node_id: node for node in nodes}, (),
        tuple(node.node_id for node in nodes),
    )


class TestPhase17CheckpointTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        names = (
            "EXPERIMENT_DIR", "PROTOCOL_VERSION", "CONFIG_KEY",
            "SOURCE_EVOLUTION_EXPERIMENT_DIR", "SOURCE_VLRB_EXPERIMENT_DIR",
            "SYSTEM_PREFIX", "REPORT_TITLE", "EPOCHS",
            "EXPECTED_VARIANT_DESCRIPTION_COUNT", "CHECKPOINT_ROLES",
            "STAGE_PREFIX", "STAGES",
        )
        self._shared_values = {name: getattr(shared, name) for name in names}

    def tearDown(self) -> None:
        for name, value in self._shared_values.items():
            setattr(shared, name, value)

    def test_config_matches_frozen_protocol(self) -> None:
        self.assertEqual(
            self.config[phase17.CONFIG_KEY], phase17.SETTINGS)
        self.assertEqual(phase17.EPOCHS, (2, 3, 4))
        self.assertEqual(phase17.EXPECTED_VARIANT_DESCRIPTION_COUNT, 18)

    def test_activation_configures_generic_runner(self) -> None:
        phase17._activate()
        self.assertEqual(shared.EXPERIMENT_DIR, phase17.EXPERIMENT_DIR)
        self.assertEqual(shared.EPOCHS, (2, 3, 4))
        self.assertEqual(shared.SYSTEM_PREFIX, "phase17")
        self.assertEqual(shared.STAGES, phase17.STAGES)
        self.assertEqual(shared._protocol(self.config), phase17.SETTINGS)

    def test_checkpoint_roles_are_preregistered(self) -> None:
        self.assertEqual(
            phase17.CHECKPOINT_ROLES,
            {
                2: "dev150_best_checkpoint",
                3: "discovery100_best_checkpoint",
                4: "first_checkpoint_with_all_five_roots_split",
            },
        )

    def test_changed_nodes_are_keyed_by_description(self) -> None:
        checkpoint = _rubric(_node("a", "same"), _node("b", "old"))
        final = _rubric(_node("a", "same"), _node("b", "new"))
        changed = shared._changed_nodes(checkpoint, final)
        self.assertEqual([node.node_id for node in changed], ["b"])
        self.assertNotEqual(
            shared._variant_key("b", "old"),
            shared._variant_key("b", "new"),
        )

    def test_stages_cover_frozen_run_order(self) -> None:
        self.assertEqual(
            phase17.STAGES,
            tuple(f"vlrb-phase17-checkpoint-{name}" for name in (
                "freeze", "audit", "smoke", "run", "retry", "report")),
        )


if __name__ == "__main__":
    unittest.main()

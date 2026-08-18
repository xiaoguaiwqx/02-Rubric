"""Offline contract tests for Prompt-v2-aligned evolution and transfer."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from critiq.dual_evaluator import (
    CacheOptimizedPairwiseVoteMultiModalEvaluator,
    PairwiseVoteMultiModalEvaluator,
)
from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    DualWorkerRequestSpec,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    StructuredCriterionSnapshot,
    Vote,
)
from critiq.structured.version import (
    PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    PAIRWISE_WORKER_PROMPT_VERSION,
)
from experiments.evolving_structured_rubrics import (
    prompt_v2_aligned_evolution as evolution,
    run_rubric_evolution as base,
    split_evolution as split,
    vl_rewardbench_prompt_v2_evolved as vlrb,
)
from experiments.evolving_structured_rubrics.rubric_factory import (
    build_multicrit_open_ended_init_rubric,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (ROOT / "experiments/evolving_structured_rubrics/configs/"
          "rubric_evolution_phase5.example.json")


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _row():
    return {"sample_id": "s0", "question": "q", "A": "a",
            "B": "b", "image_path": "image.jpg"}


def _prediction(prompt_version: str) -> PairwisePredictionOutput:
    rubric = build_multicrit_open_ended_init_rubric()
    criteria = tuple(StructuredCriterionSnapshot(
        rubric.get_node(node_id).criterion.name,
        rubric.get_node(node_id).criterion.description)
        for node_id in rubric.preorder_node_ids())
    outputs = {item.name: PairwiseVoteOutput(
        Vote.A, True, "raw", None, 1, "thought", True) for item in criteria}
    spec = DualWorkerRequestSpec(
        "model", "pool", "a" * 64, None, True, "image_path", "question",
        "sample_id", {"temperature": .5, "max_tokens": 2048})
    return PairwisePredictionOutput(
        ("s0",), ("f" * 64,), criteria, (outputs,),
        (base.aggregate_flat_votes(item.vote for item in outputs.values()),),
        spec, prompt_version=prompt_version)


class PromptV2AlignedEvolutionTests(unittest.TestCase):

    def test_frozen_config_blocks_match_implementation(self):
        config = _config()
        self.assertEqual(config["prompt_v2_aligned_evolution"], evolution.SETTINGS)
        self.assertEqual(
            config["vlrb_prompt_v2_evolved"]["protocol_version"],
            vlrb.PROTOCOL_VERSION)
        self.assertTrue(config["vlrb_prompt_v2_evolved"][
            "run_regardless_of_heldout_result"])

    def test_pairwise_evaluator_defaults_to_v1_and_v2_is_explicit(self):
        config = _config()
        pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(
            config["backend_pool"]))
        v1 = base._pairwise_evaluator(config, [_row()], pool)
        self.assertIsInstance(v1, PairwiseVoteMultiModalEvaluator)
        self.assertNotIsInstance(v1, CacheOptimizedPairwiseVoteMultiModalEvaluator)
        self.assertEqual(v1.prompt_version, PAIRWISE_WORKER_PROMPT_VERSION)

        treatment = dict(config); treatment["_pairwise_prompt_mode"] = "v2_cache"
        v2 = base._pairwise_evaluator(treatment, [_row()], pool)
        self.assertIsInstance(v2, CacheOptimizedPairwiseVoteMultiModalEvaluator)
        self.assertEqual(
            v2.prompt_version, PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION)

    def test_unknown_pairwise_prompt_mode_is_rejected(self):
        config = _config(); config["_pairwise_prompt_mode"] = "future"
        pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(
            config["backend_pool"]))
        with self.assertRaisesRegex(ValueError, "unsupported Pairwise prompt mode"):
            base._pairwise_evaluator(config, [_row()], pool)

    def test_split_merge_preserves_prompt_version(self):
        prediction = _prediction(PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION)
        rubric = build_multicrit_open_ended_init_rubric()
        merged = split.merge_predictions(prediction, (), rubric)
        self.assertEqual(merged.prompt_version, prediction.prompt_version)
        self.assertEqual(merged.request_spec, prediction.request_spec)

    def test_protocol_uses_fresh_signatures_and_global_memory(self):
        self.assertEqual(evolution.PROTOCOL.rubric_memory_mode, "global_rubric_v1")
        self.assertFalse(evolution.PROTOCOL.read_only_control_signatures)
        self.assertTrue(evolution.PROTOCOL.allow_configured_endpoint_pool)
        self.assertEqual(evolution.SETTINGS["error_signature_policy"],
                         "fresh_unless_exact_identity")

    def test_phase16_v2_broadens_only_runtime_split_threshold(self):
        config = _config()
        self.assertEqual(
            config["evolution_policy"]["trigger_thresholds"]["tau_split"], .70)
        runtime = evolution._v2_config(config)
        self.assertEqual(
            runtime["evolution_policy"]["trigger_thresholds"]["tau_split"], .75)
        self.assertEqual(
            runtime["evolution_policy"]["trigger_thresholds"]["tau_cov_high"], .80)
        self.assertEqual(
            config["evolution_policy"]["trigger_thresholds"]["tau_split"], .70)

    def test_unknown_stages_fail_explicitly(self):
        with self.assertRaises(ValueError):
            evolution.run_stage({}, Path("."), "not-a-stage")
        with self.assertRaises(ValueError):
            vlrb.run_stage({}, Path("."), "not-a-stage")

    def test_vlrb_prompt_v2_control_validator_does_not_require_native_system(self):
        records = [{"sample_id": "s0"}]
        logical = {
            "sample_ids": ["s0"],
            "k": 3,
            "systems": {
                vlrb.control.INITIAL_SYSTEM: {},
                vlrb.control.EQUAL_SYSTEM: {},
            },
        }
        vlrb._validate_prompt_v2_control_logical(records, logical)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            vlrb._validate_prompt_v2_control_logical(
                records, {**logical, "systems": {vlrb.control.EQUAL_SYSTEM: {}}})


if __name__ == "__main__":
    unittest.main()

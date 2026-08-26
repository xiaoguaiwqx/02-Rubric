"""Offline protocol tests for the clean Global-Arbiter v2 rerun."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from experiments.evolving_structured_rubrics import (
    global_arbiter_ab_only as legacy,
)
from experiments.evolving_structured_rubrics import (
    global_arbiter_ab_preferred_none_v2 as clean,
)


class TestGlobalArbiterABPreferredNoneV2(unittest.TestCase):
    def test_prompt_is_byte_identical_to_legacy_s5(self) -> None:
        self.assertIs(
            clean.GLOBAL_ARBITER_SYSTEM_PROMPT,
            legacy.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        )
        self.assertEqual(
            hashlib.sha256(clean.GLOBAL_ARBITER_SYSTEM_PROMPT.encode()).hexdigest(),
            hashlib.sha256(
                legacy.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode()).hexdigest(),
        )

    def test_parser_treats_none_as_semantic_output(self) -> None:
        self.assertEqual(
            {"answer": "None"},
            clean.parse_global_arbiter_response('{"answer":"None"}'),
        )

    def test_new_protocol_and_directory_prevent_legacy_cache_reuse(self) -> None:
        self.assertNotEqual(clean.PROTOCOL_VERSION, legacy.PROTOCOL_VERSION)
        self.assertNotEqual(clean.EXPERIMENT_DIR, legacy.EXPERIMENT_DIR)
        self.assertNotEqual(clean.REQUEST_KIND, legacy.REQUEST_KIND)
        self.assertEqual(clean.PROMPT_VERSION, legacy.PROMPT_VERSION)

    def test_frozen_config_matches_clean_profile(self) -> None:
        path = Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        settings = config[clean.CONFIG_KEY]
        self.assertEqual(clean.PROTOCOL_VERSION, settings["protocol_version"])
        self.assertEqual(clean.PROMPT_VERSION, settings["prompt_version"])
        self.assertEqual(["A", "B", "None"], settings["semantic_answer_space"])
        self.assertEqual(3, settings["k"])
        self.assertEqual(10, settings["max_parse_retries"])
        self.assertEqual("unset", settings["generation_seed_policy"])

    def test_stage_namespace_is_complete_and_disjoint(self) -> None:
        self.assertEqual(6, len(clean.STAGES))
        self.assertTrue(all(
            stage.startswith("vlrb-global-arbiter-ab-preferred-none-v2-")
            for stage in clean.STAGES
        ))
        self.assertTrue(set(clean.STAGES).isdisjoint(legacy.STAGES))


if __name__ == "__main__":
    unittest.main()


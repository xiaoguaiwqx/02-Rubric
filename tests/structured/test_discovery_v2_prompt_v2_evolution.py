import json
import unittest
from dataclasses import replace
from pathlib import Path

from critiq.structured import DualWorkerRequestSpec

from experiments.evolving_structured_rubrics import (
    discovery_v2_prompt_v2_evolution as evolution,
)
from experiments.evolving_structured_rubrics import (
    vl_rewardbench_discovery_v2 as vlrb,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "experiments/evolving_structured_rubrics/configs/"
    "rubric_evolution_phase5.example.json"
)


class TestDiscoveryV2PromptV2Evolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_config_matches_code(self):
        self.assertEqual(
            self.config["discovery_v2_prompt_v2_evolution"],
            evolution.SETTINGS,
        )
        self.assertEqual(self.config["vlrb_discovery_v2"], vlrb.SETTINGS)

    def test_runtime_uses_prompt_v2_and_independent_splits(self):
        runtime = evolution._runtime_config(self.config)
        self.assertEqual(runtime["_pairwise_prompt_mode"], "v2_cache")
        self.assertEqual(
            runtime["_experiment_dataset_paths"],
            {"discovery": evolution.DISCOVERY_PATH, "dev": evolution.DEV_PATH},
        )
        self.assertEqual(
            runtime["_experiment_dataset_counts"],
            {"discovery": 100, "dev": 150, "heldout": 500},
        )
        self.assertEqual(
            runtime["evolution_policy"]["trigger_thresholds"]["tau_split"],
            0.75,
        )

    def test_dev_is_diagnostic_only(self):
        settings = evolution.SETTINGS
        self.assertEqual(settings["dev_policy"], "diagnostic_only_no_selection")
        self.assertEqual(settings["error_signature_policy"],
                         "fresh_discovery100_only")
        self.assertNotIn("dev", settings["candidate_scope"])

    def test_overlap_audit_counts_strong_keys(self):
        left = [{
            "sample_id": "a", "image_sha256": "image-a",
            "question_sha256": "q-a", "unordered_pair_sha256": "p-a",
            "source": "s", "source_sample_id": "source-a",
        }]
        right = [{
            "sample_id": "b", "image_sha256": "image-a",
            "question_sha256": "q-b", "unordered_pair_sha256": "p-b",
            "source": "s", "source_sample_id": "source-b",
        }]
        value = evolution._cross_overlap(left, right)
        self.assertEqual(value["image_sha256"], 1)
        self.assertEqual(value["sample_id"], 0)
        self.assertEqual(value["source_sample_id"], 0)

    def test_heldout_merge_identity_ignores_only_backend_pool(self):
        spec = DualWorkerRequestSpec(
            model="model", backend_id="pool:old",
            prompt_sha256="1" * 64, max_data_chars=None,
            encode_local_image=True, image_field="image_path",
            question_field="question", sample_id_field="sample_id",
            decoding_config={"temperature": 0.5, "max_tokens": 2048},
        )

        class Prediction:
            semantics_version = "1.0.0"
            schema_version = "1.0.0"
            prompt_version = "1.1.0-cache-pilot"
            parser_version = "1.1.0"

            def __init__(self, request_spec):
                self.request_spec = request_spec

        source = Prediction(spec)
        compatible = Prediction(replace(spec, backend_id="pool:new"))
        incompatible = Prediction(replace(
            spec, backend_id="pool:new",
            decoding_config={"temperature": 0.5, "max_tokens": 1024},
        ))
        self.assertTrue(evolution._same_pairwise_scientific_request_identity(
            source, compatible))
        self.assertFalse(evolution._same_pairwise_scientific_request_identity(
            source, incompatible))

    def test_vlrb_wrapper_builds_shared_protocol_view(self):
        adapted = vlrb._adapt_config(self.config)
        generic = adapted["vlrb_prompt_v2_evolved"]
        self.assertEqual(generic["source_experiment"], evolution.EXPERIMENT_DIR)
        self.assertEqual(generic["scheduler"],
                         "sample_major_available_slot_dynamic")
        self.assertNotIn("primary_external_evaluation", generic)
        self.assertEqual(vlrb.TREATMENT_SYSTEM,
                         "phase17_discovery_v2_final_equal")

    def test_vlrb_wrapper_declares_phase16_direct_baseline(self):
        from experiments.evolving_structured_rubrics import (
            vl_rewardbench_prompt_v2_evolved as shared,
        )
        names = (
            "EXPERIMENT_DIR", "PROTOCOL_VERSION", "CONTROL_EXPERIMENT",
            "TREATMENT_SYSTEM", "SCHEDULER", "EXTRA_BASELINE_REPORTS",
            "STAGE_FREEZE", "STAGE_AUDIT", "STAGE_SMOKE", "STAGE_RUN",
            "STAGE_RETRY", "STAGE_REPORT", "evolution",
        )
        original = {name: getattr(shared, name) for name in names}
        try:
            vlrb._activate()
            self.assertIn("phase16_final_equal_prompt_v2",
                          shared.EXTRA_BASELINE_REPORTS)
            _, metric_name = shared.EXTRA_BASELINE_REPORTS[
                "phase16_final_equal_prompt_v2"]
            self.assertEqual(metric_name, "prompt_v2_evolved_final_equal")
        finally:
            for name, value in original.items():
                setattr(shared, name, value)


if __name__ == "__main__":
    unittest.main()

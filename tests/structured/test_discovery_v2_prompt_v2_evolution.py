import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from critiq.structured import DualWorkerRequestSpec

from experiments.evolving_structured_rubrics import (
    discovery_v2_prompt_v2_evolution as evolution,
)
from experiments.evolving_structured_rubrics import (
    run_rubric_evolution as runner,
)
from experiments.evolving_structured_rubrics import split_evolution as split
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

    def test_heldout_reference_output_defaults_to_current_output(self):
        output = Path("isolated-output")
        self.assertEqual(
            evolution._heldout_reference_output(self.config, output), output)

    def test_heldout_reference_output_accepts_historical_root(self):
        config = deepcopy(self.config)
        config["discovery_v2_heldout_reference_output"] = (
            "output/evolving_structured_rubrics/rubric_evolution_phase5")
        self.assertEqual(
            evolution._heldout_reference_output(config, Path("isolated")),
            runner.ROOT
            / "output/evolving_structured_rubrics/rubric_evolution_phase5",
        )

    def test_phase5_config_loader_allows_heldout_reference_output(self):
        config = deepcopy(self.config)
        config["discovery_v2_heldout_reference_output"] = (
            "output/evolving_structured_rubrics/rubric_evolution_phase5")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = runner._config(path)
        self.assertEqual(
            loaded["discovery_v2_heldout_reference_output"],
            config["discovery_v2_heldout_reference_output"],
        )

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
        self.assertEqual(
            runtime["evolution_policy"]["trigger_thresholds"]["N_min_cluster"],
            5,
        )
        self.assertFalse(runtime["_manager_compact_sample_ids"])

    def test_runtime_allows_experiment_local_min_cluster_override(self):
        config = deepcopy(self.config)
        config["discovery_v2_prompt_v2_evolution"][
            "split_min_cluster_size"] = 2
        evolution._config(config)
        runtime = evolution._runtime_config(config)
        self.assertEqual(
            runtime["evolution_policy"]["trigger_thresholds"]["N_min_cluster"],
            2,
        )
        self.assertEqual(
            config["evolution_policy"]["trigger_thresholds"]["N_min_cluster"],
            5,
        )

    def test_runtime_allows_compact_manager_sample_ids(self):
        config = deepcopy(self.config)
        config["discovery_v2_prompt_v2_evolution"][
            "compact_manager_sample_ids"] = True
        evolution._config(config)
        runtime = evolution._runtime_config(config)
        self.assertTrue(runtime["_manager_compact_sample_ids"])

    def test_frozen_manager_contract_uses_runtime_compact_id_setting(self):
        config = deepcopy(self.config)
        config["discovery_v2_prompt_v2_evolution"][
            "compact_manager_sample_ids"] = True
        for profile in config["specialize_managers"].values():
            profile["api_key_env"] = None
            profile["vllm_identity"] = None
        config["refine_manager"]["api_key_env"] = None

        runtime = evolution._runtime_config(config)
        protocol = evolution._protocol(runtime)
        _, _, expected_specs, _ = split._managers(runtime, protocol)

        contract = evolution._manager_contract(config)
        self.assertEqual(contract["specs"], expected_specs)

    def test_manager_runtime_uses_configured_retry_counts(self):
        config = deepcopy(self.config)
        config["structured_max_retries"] = 3
        config["api_retry_attempts"] = 7
        config["_manager_compact_sample_ids"] = True
        for profile in config["specialize_managers"].values():
            profile["api_key_env"] = None
            profile["vllm_identity"] = None
        manager, _, _, _ = runner._manager_runtime(config, "error_signature")
        self.assertEqual(manager.structured_max_retries, 3)
        self.assertEqual(manager.api_retry_attempts, 7)
        self.assertTrue(manager.compact_sample_ids)

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

    def test_vlrb_explicit_rubric_loads_without_evolution_reports(self):
        from experiments.evolving_structured_rubrics import (
            vl_rewardbench_prompt_v2_evolved as shared,
        )
        config = deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rubric.json"
            rubric = runner.build_multicrit_open_ended_init_rubric()
            rubric.save_json(path)
            output = Path(directory) / "evaluation"
            config["vlrb_discovery_v2"].update(
                rubric_path=str(path), output_dir=str(output))
            adapted = vlrb._adapt_config(config)
            self.assertEqual(shared._target(adapted), output)
            self.assertEqual(
                shared._rubric(Path(directory) / "no_evolution", adapted)
                .rubric_sha256, rubric.rubric_sha256)
        del config["vlrb_discovery_v2"]["output_dir"]
        with self.assertRaisesRegex(ValueError, "own output_dir"):
            vlrb._adapt_config(config)

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

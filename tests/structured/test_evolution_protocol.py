"""Offline isolation and compatibility contracts for shared Phase21/22 code."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.evolving_structured_rubrics import (
    all_sample_adaptive_recluster_evolution as phase22,
    unified_subtree_bundle_evolution as engine,
    vl_rewardbench_all_sample_adaptive_recluster_evolution as phase22_vlrb,
    vl_rewardbench_unified_subtree_bundle_evolution as benchmark,
)


class TestExplicitEvolutionProtocols(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))

    def test_settings_are_deeply_isolated_snapshots(self) -> None:
        protocol = phase22.PROTOCOL
        settings = protocol.settings
        settings["split_trigger"]["min_wrong"] = -1
        settings["recluster_failure_types"].clear()
        self.assertEqual(phase22.SETTINGS, protocol.settings)
        with self.assertRaises(FrozenInstanceError):
            protocol.experiment_dir = "changed"

    def test_configuration_cannot_silently_select_another_protocol(self) -> None:
        config = dict(self.config)
        config[engine.DEFAULT_PROTOCOL.config_key] = phase22.SETTINGS
        with self.assertRaisesRegex(RuntimeError, "drift"):
            engine._settings(config)
        self.assertEqual(phase22.SETTINGS, engine._settings(
            config, protocol=phase22.PROTOCOL))

    def test_concurrent_protocols_keep_paths_rules_and_settings_isolated(self) -> None:
        barrier = Barrier(2)

        def inspect(protocol):
            barrier.wait(timeout=5)
            return (
                engine._target(Path("runs"), protocol=protocol).name,
                engine._settings(self.config, protocol=protocol),
                engine._uses_all_sample_acceptance(protocol=protocol),
                engine.split_retry_action(
                    "cluster_or_decomposition_error", protocol=protocol),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(inspect, (
                engine.DEFAULT_PROTOCOL, phase22.PROTOCOL)))
        self.assertEqual((engine.EXPERIMENT_DIR, engine.SETTINGS, False,
                          "reuse_clusters_regenerate_complete_bundle"), results[0])
        self.assertEqual((phase22.EXPERIMENT_DIR, phase22.SETTINGS, True,
                          "recluster_regenerate_complete_bundle"), results[1])

    def test_wrapper_exception_does_not_mutate_config_or_default(self) -> None:
        before = json.dumps(self.config, sort_keys=True)
        with patch.object(engine, "run", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                phase22.run(self.config, Path("unused"))
        self.assertEqual(before, json.dumps(self.config, sort_keys=True))
        self.assertEqual(engine.SETTINGS, engine._settings(self.config))
        self.assertFalse(engine._uses_all_sample_acceptance())

    def test_existing_manifest_shape_loads_without_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for protocol in (engine.DEFAULT_PROTOCOL, phase22.PROTOCOL):
                manifest = {"protocol_version": protocol.version,
                            "settings": protocol.settings}
                engine._write(engine._target(output, protocol=protocol)
                              / "frozen_manifest.json", manifest)
                self.assertEqual(manifest, engine._load_manifest(
                    self.config, output, protocol=protocol))
            # Same filename, wrong protocol: still rejected, not migrated.
            engine._write(engine._target(output, protocol=phase22.PROTOCOL)
                          / "frozen_manifest.json", {
                              "protocol_version": engine.PROTOCOL_VERSION,
                              "settings": engine.SETTINGS})
            with self.assertRaisesRegex(RuntimeError, "protocol drift"):
                engine._load_manifest(self.config, output, protocol=phase22.PROTOCOL)

    def test_runtime_identity_does_not_include_protocol_routing(self) -> None:
        self.assertEqual(
            engine._unified_runtime_identity(self.config),
            engine._unified_runtime_identity(self.config, protocol=phase22.PROTOCOL))

    def test_benchmark_source_paths_and_labels_are_explicit(self) -> None:
        for protocol in (benchmark.DEFAULT_PROTOCOL, phase22_vlrb.PROTOCOL):
            with patch.object(benchmark.StructuredRubric, "load_json") as read:
                benchmark._rubrics(Path("runs"), protocol=protocol)
            self.assertEqual(
                Path("runs") / protocol.source.experiment_dir
                / "epochs" / "epoch_00" / "rubric_initial.json",
                read.call_args_list[0].args[0])
            self.assertEqual(protocol.experiment_dir, benchmark._target(
                Path("runs"), protocol=protocol).name)
        self.assertEqual("phase21_final", benchmark.FINAL_LABEL)

    def test_request_payload_and_hash_remain_identical(self) -> None:
        kwargs = dict(
            prompt_version="fixture-v1", system_prompt="system",
            user_text="sample", row={}, request_key={"replicate": 0},
            image_sha256="image-hash")
        endpoint = SimpleNamespace(checkpoint_root="fixture-model")
        before = benchmark._request_payload(self.config, endpoint, **kwargs)
        after = benchmark._request_payload(
            self.config, endpoint, protocol=phase22_vlrb.PROTOCOL, **kwargs)
        self.assertEqual(before, after)
        self.assertEqual(benchmark.base.canonical_sha256(before),
                         benchmark.base.canonical_sha256(after))

    def test_stage_dispatch_forwards_protocol_and_rejects_foreign_stage(self) -> None:
        with patch.object(engine, "freeze") as action:
            engine.run_stage(self.config, Path("unused"), phase22.STAGES[0],
                             protocol=phase22.PROTOCOL)
        action.assert_called_once_with(
            self.config, Path("unused"), protocol=phase22.PROTOCOL)
        with self.assertRaises(ValueError):
            engine.run_stage(self.config, Path("unused"), phase22.STAGES[0])
        with patch.object(benchmark, "run") as action:
            benchmark.run_stage(self.config, Path("unused"), phase22_vlrb.STAGES[3],
                                protocol=phase22_vlrb.PROTOCOL)
        action.assert_called_once_with(
            self.config, Path("unused"), protocol=phase22_vlrb.PROTOCOL)

    def test_report_success_is_not_published_before_wrapper_validation(self) -> None:
        metrics = {
            "strict_accuracy": 0.5, "covered_accuracy": 0.5,
            "macro_strict_accuracy": 0.5, "coverage": 1.0,
            "original_index_predictions": {},
        }
        paired = {"net_corrected": 0, "mcnemar_exact_two_sided_p": 1.0}
        for wrapper in (False, True):
            with self.subTest(wrapper=wrapper), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                protocol = (phase22_vlrb.PROTOCOL if wrapper
                            else benchmark.DEFAULT_PROTOCOL)
                target = benchmark._target(output, protocol=protocol)
                control_path = target / "control.json"
                engine._write(control_path, {})
                manifest = {
                    "initial_reuse": {"path": "fixture-initial.json"},
                    "phase21_vlrb_control": {
                        "prediction_path": str(control_path),
                        "prediction_sha256": "deliberately-mismatched-hash",
                    },
                }
                engine._write(target / "frozen_manifest.json", manifest)
                # Re-running an old passed report must also reset its status.
                engine._write(target / "stage_status.json", {
                    "retry": {"status": "passed"},
                    "report": {"status": "passed"}})
                observed = []
                original_write = benchmark.atomic_write_json

                def capture(path, value):
                    if Path(path).name == "stage_status.json":
                        observed.append(value["report"]["status"])
                    return original_write(path, value)

                with ExitStack() as stack:
                    stack.enter_context(patch.object(benchmark, "_load", return_value=(
                        target, manifest, [], [], {}, None, None)))
                    stack.enter_context(patch.object(benchmark.system, "load", return_value={}))
                    stack.enter_context(patch.object(benchmark, "_failure_count", return_value=0))
                    stack.enter_context(patch.object(benchmark.shared, "_votes", return_value=[]))
                    stack.enter_context(patch.object(benchmark.shared, "_control", return_value={
                        "metrics": metrics, "sha256": "control-hash"}))
                    stack.enter_context(patch.object(benchmark.vlrb_metrics, "_system_metrics", return_value=metrics))
                    stack.enter_context(patch.object(benchmark.vlrb, "_paired", return_value=paired))
                    stack.enter_context(patch.object(benchmark.shared, "_paired_bootstrap_delta_ci", return_value={}))
                    stack.enter_context(patch.object(benchmark, "atomic_write_json", side_effect=capture))
                    if wrapper:
                        with self.assertRaisesRegex(RuntimeError, "prediction control drift"):
                            phase22_vlrb.report(self.config, output)
                    else:
                        benchmark.report(self.config, output)
                status = json.loads((target / "stage_status.json").read_text())
                if wrapper:
                    self.assertNotIn("passed", observed)
                    self.assertEqual("pending_wrapper_comparison", status["report"]["status"])
                else:
                    self.assertEqual(["running", "passed"], observed)


if __name__ == "__main__":
    unittest.main()

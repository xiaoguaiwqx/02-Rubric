"""Focused offline tests for Phase22 all-sample competition semantics."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from critiq.structured.cache import canonical_sha256
from critiq.structured.schema import (
    EdgeCondition,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredRubric,
)
from experiments.evolving_structured_rubrics import aligned_system_runtime as runtime
from experiments.evolving_structured_rubrics import (
    all_sample_adaptive_recluster_evolution as evolution,
)
from experiments.evolving_structured_rubrics import (
    unified_subtree_bundle_evolution as phase21,
)
from experiments.evolving_structured_rubrics.subtree_bundle_manager import (
    bundle_refine_prompt_subtree,
)


def _call(answer: str, *, parse_ok: bool = True) -> dict:
    return {
        "parse_ok": parse_ok,
        "parsed": {"answer": answer} if parse_ok else None,
        "model_generation_count": 1,
        "endpoint_id": "vllm-8000",
    }


def _artifact(answers: list[str], *, parse_failures: tuple[int, ...] = ()) -> dict:
    return {
        "root_id": "r1",
        "samples": [
            {
                "sample_id": f"s{index + 1}",
                "order": 0,
                "call": _call(answer, parse_ok=index not in parse_failures),
            }
            for index, answer in enumerate(answers)
        ],
    }


class TestAllSamplePairedRoot(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"sample_id": "s1", "answer": "A"},
            {"sample_id": "s2", "answer": "B"},
            {"sample_id": "s3", "answer": "A"},
            {"sample_id": "s4", "answer": "B"},
        ]

    def test_plus_one_minus_one_zero_and_none_semantics(self) -> None:
        before = _artifact(["None", "B", "B", "A"])
        after = _artifact(["A", "None", "None", "A"])
        value = runtime.paired_root_all_samples(before, after, self.rows)
        self.assertEqual(["s1"], value["all_sample_corrected_sample_ids"])
        self.assertEqual(["s2"], value["all_sample_harmed_sample_ids"])
        self.assertEqual(["s3", "s4"], value["all_sample_unchanged_sample_ids"])
        self.assertEqual([1, -1, 0, 0], [
            item["gain"] for item in value["gain_ledger"]])
        self.assertEqual(0, value["all_sample_net_gain"])
        self.assertEqual(["s1"], value["none_to_correct_sample_ids"])
        self.assertEqual(["s2"], value["correct_to_none_sample_ids"])

    def test_support_is_all_rows_not_old_decisive_scope(self) -> None:
        value = runtime.paired_root_all_samples(
            _artifact(["A", "None", "B", "None"]),
            _artifact(["A", "B", "A", "None"]), self.rows)
        self.assertEqual(4, value["all_sample_support"])
        self.assertEqual(["s1", "s2", "s3", "s4"], value["all_sample_ids"])

    def test_sample_identity_drift_is_rejected(self) -> None:
        after = _artifact(["A", "B", "A", "B"])
        after["samples"][-1]["sample_id"] = "other"
        with self.assertRaisesRegex(ValueError, "sample identity"):
            runtime.paired_root_all_samples(
                _artifact(["A", "B", "A", "B"]), after, self.rows)

    def test_technical_failure_is_exposed_for_pause(self) -> None:
        value = runtime.paired_root_all_samples(
            _artifact(["A", "B", "A", "B"]),
            _artifact(["A", "B", "A", "B"], parse_failures=(2,)), self.rows)
        self.assertEqual(1, value["technical_failure_count"])
        self.assertEqual(["s3"], value["technical_failure_sample_ids"])
        self.assertEqual(1, value["technical_failure_count_after"])

    def test_ledger_digest_and_total_are_recomputable(self) -> None:
        value = runtime.paired_root_all_samples(
            _artifact(["B", "B", "A", "A"]),
            _artifact(["A", "A", "A", "B"]), self.rows)
        self.assertEqual(
            sum(item["gain"] for item in value["gain_ledger"]),
            value["all_sample_net_gain"])
        self.assertEqual(
            canonical_sha256(value["gain_ledger"]),
            value["gain_ledger_sha256"])


class TestPhase22Protocol(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))

    def test_config_matches_frozen_combined_protocol(self) -> None:
        config = self._config()
        self.assertEqual(evolution.SETTINGS, evolution._settings(config))
        self.assertEqual(
            phase21.SETTINGS["split_trigger"],
            evolution.SETTINGS["split_trigger"])
        self.assertEqual(
            phase21.SETTINGS["bundle_refine_trigger"],
            evolution.SETTINGS["bundle_refine_trigger"])
        self.assertEqual(
            phase21.SETTINGS["max_epochs"],
            evolution.SETTINGS["max_epochs"])
        self.assertFalse(evolution.SETTINGS["strong_child_locking"])
        self.assertFalse(evolution.SETTINGS["partial_acceptance"])
        unchanged_keys = (
            "source_protocol", "optimization_unit", "scope_source",
            "error_source", "strong_child_locking", "partial_acceptance",
            "specialized_acc_role", "candidate_commit", "global_arbiter_role",
            "split_trigger", "bundle_refine_trigger", "min_epochs",
            "max_epochs", "internal_k", "temperature", "max_tokens",
            "max_parse_retries", "generation_seed_policy",
            "pairwise_prompt_mode", "pairwise_prompt_version", "dev_policy",
            "heldout_access", "vl_rewardbench_required",
        )
        for key in unchanged_keys:
            self.assertEqual(
                phase21.SETTINGS[key], evolution.SETTINGS[key], msg=key)

    def test_failure_aware_retry_mapping(self) -> None:
        self.assertEqual(
            "recluster_regenerate_complete_bundle",
            evolution.split_retry_action("cluster_or_decomposition_error"))
        for failure_type in (
                "no_effect", "missed_correction", "coverage_loss_to_none",
                "mixed_or_inconclusive"):
            self.assertEqual(
                "reuse_clusters_regenerate_complete_bundle",
                evolution.split_retry_action(failure_type))

    def test_phase21_error_signature_ids_are_unchanged(self) -> None:
        rows = [
            {"sample_id": "s1", "answer": "A"},
            {"sample_id": "s2", "answer": "B"},
            {"sample_id": "s3", "answer": "A"},
        ]
        baseline = _artifact(["B", "None", "A"])
        baseline["metrics"] = runtime.root_metrics(
            baseline, rows, ["s1", "s3"])
        expected_scope = phase21._scope_ids(baseline)
        expected_errors = phase21._root_mismatches(baseline, rows)
        self.assertEqual(expected_scope, phase21._scope_ids(baseline))
        self.assertEqual(
            expected_errors, phase21._root_mismatches(baseline, rows))
        self.assertEqual(("s1", "s3"), expected_scope)
        self.assertEqual(("s1",), expected_errors)

    def test_recluster_discards_old_cluster_source(self) -> None:
        retry = {"action": "recluster_regenerate_complete_bundle",
                 "cluster_path": "old.json"}
        self.assertIsNone(evolution.split_retry_source(retry))
        retry["action"] = "reuse_clusters_regenerate_complete_bundle"
        self.assertIs(retry, evolution.split_retry_source(retry))

    def test_bound_engine_retry_mapper_matches_public_protocol(self) -> None:
        for failure_type in (
                "cluster_or_decomposition_error", "no_effect",
                "coverage_loss_to_none"):
            self.assertEqual(
                evolution.split_retry_action(failure_type),
                phase21.split_retry_action(failure_type, protocol=evolution.PROTOCOL))
        recluster = {
            "action": "recluster_regenerate_complete_bundle",
            "cluster_path": "old.json",
        }
        self.assertIsNone(phase21.split_retry_source(recluster))

    def test_bundle_refine_prompt_supplies_exact_child_description_hashes(self) -> None:
        description = (
            "Criterion focus: x\n\nApplicable only when: y\n\n"
            "Not applicable when: z\n\nDecision rule: None.")
        rubric = StructuredRubric(
            {
                "r1": RubricNode(
                    "r1", RubricCriterionSnapshot("root", "root", 1.0)),
                "c1": RubricNode(
                    "c1", RubricCriterionSnapshot("child", description, 1.0)),
            },
            (RubricEdge("r1", "c1", EdgeCondition.ALWAYS),),
            ("r1",),
        )
        prompt = bundle_refine_prompt_subtree(rubric, "r1")
        import hashlib
        expected = hashlib.sha256(description.encode("utf-8")).hexdigest()
        self.assertIn("Exact current child description SHA-256 values", prompt)
        self.assertIn(expected, prompt)

    def test_shared_engine_protocol_is_explicit(self) -> None:
        config = self._config()
        original_settings = dict(config)
        old = (phase21.EXPERIMENT_DIR, phase21.PROTOCOL_VERSION,
               phase21.STAGES, phase21.SETTINGS)
        with patch.object(phase21, "freeze") as action:
            evolution.freeze(config, Path("unused"))
        action.assert_called_once_with(
            config, Path("unused"), protocol=evolution.PROTOCOL)
        self.assertEqual(original_settings, config)
        self.assertEqual(old, (
            phase21.EXPERIMENT_DIR, phase21.PROTOCOL_VERSION,
            phase21.STAGES, phase21.SETTINGS))

    def test_split_and_refine_use_identical_strict_positive_gate(self) -> None:
        rows = [
            {"sample_id": "s1", "answer": "A"},
            {"sample_id": "s2", "answer": "B"},
        ]
        before = _artifact(["B", "B"])
        before["samples"] = before["samples"][:2]
        before["metrics"] = runtime.root_metrics(before, rows, ["s1", "s2"])
        config = self._config()
        for operator in ("split", "refine"):
            for answers, expected in (
                    (["B", "A"], "competition_rejected"),
                    (["A", "A"], "competition_rejected"),
                    (["A", "B"], "accepted")):
                after = _artifact(answers)
                after["samples"] = after["samples"][:2]
                after["metrics"] = runtime.root_metrics(
                    after, rows, ["s1", "s2"])
                with tempfile.TemporaryDirectory() as directory:
                    attempt_dir = Path(directory)
                    result = {
                        "operator": operator,
                        "root_id": "r1",
                        "attempt_dir": attempt_dir,
                        "after_rubric": object(),
                    }
                    with patch.object(
                            phase21.system, "evaluate_root", return_value=after):
                        value = phase21._evaluate_candidate(
                            self._config(), target=attempt_dir,
                            epoch_dir=attempt_dir, rows=rows,
                            baseline=before, scope=["s1", "s2"],
                            result=result, protocol=evolution.PROTOCOL)
                    self.assertEqual(expected, value["decision"])
                    ledger = json.loads((
                        attempt_dir / "all_sample_gain_ledger.json"
                    ).read_text(encoding="utf-8"))
                    self.assertEqual(
                        sum(item["gain"] for item in ledger["ledger"]),
                        ledger["net_gain"])

    def test_technical_failure_stops_before_scientific_decision(self) -> None:
        rows = [{"sample_id": "s1", "answer": "A"}]
        before = _artifact(["A"])
        before["samples"] = before["samples"][:1]
        before["metrics"] = runtime.root_metrics(before, rows, ["s1"])
        after = _artifact(["A"], parse_failures=(0,))
        after["samples"] = after["samples"][:1]
        after["metrics"] = runtime.root_metrics(after, rows, ["s1"])
        with tempfile.TemporaryDirectory() as directory:
            attempt_dir = Path(directory)
            with patch.object(
                    phase21.system, "evaluate_root", return_value=after):
                with self.assertRaisesRegex(RuntimeError, "technical failures"):
                    phase21._evaluate_candidate(
                        self._config(), target=attempt_dir,
                        epoch_dir=attempt_dir, rows=rows,
                        baseline=before, scope=["s1"],
                        result={"operator": "split", "root_id": "r1",
                                "attempt_dir": attempt_dir,
                                "after_rubric": object()}, protocol=evolution.PROTOCOL)
            self.assertFalse((attempt_dir / "retry_action.json").exists())

    def test_recluster_lineage_and_complete_bundle_are_auditable(self) -> None:
        old_cluster = {
            "clusters": [
                {"cluster_id": "old_a", "sample_ids": ["s1", "s2"]},
                {"cluster_id": "old_b", "sample_ids": ["s3"]},
            ],
            "metrics": {"logical_evaluations": 1},
        }
        new_cluster = {
            "clusters": [
                {"cluster_id": "new_a", "sample_ids": ["s1"]},
                {"cluster_id": "new_b", "sample_ids": ["s2", "s3"]},
            ],
            "metrics": {"logical_evaluations": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            old_path = root / "old_cluster.json"
            current_path = attempt / "cluster_proposal.json"
            signature_path = root / "signatures.json"
            attribution_path = root / "failure_attribution.json"
            phase21._write(old_path, old_cluster)
            phase21._write(current_path, new_cluster)
            phase21._write(signature_path, {"outputs": {"s1": {}}})
            phase21._write(attribution_path, {
                "attribution": {
                    "primary_failure_type": "cluster_or_decomposition_error"}})
            phase21._write(attempt / "children" / "child.json", {
                "criterion_name": "child"})
            result = {
                "attempt_dir": attempt,
                "cluster_path": str(current_path),
                "signature_path": str(signature_path),
                "generated_child_ids": ["child"],
            }
            retry = {
                "action": "recluster_regenerate_complete_bundle",
                "cluster_path": str(old_path),
                "failure_attribution_path": str(attribution_path),
            }
            phase21._record_split_lineage(result, retry, protocol=evolution.PROTOCOL)
            diff = json.loads((attempt / "cluster_diff.json").read_text(
                encoding="utf-8"))
            self.assertEqual(
                "recluster_regenerate_complete_bundle", diff["retry_action"])
            self.assertNotEqual(
                diff["old_cluster_sha256"], diff["new_cluster_sha256"])
            self.assertTrue(diff["complete_child_regeneration"])
            self.assertFalse(diff["partial_acceptance"])
            self.assertTrue((attempt / "new_cluster.json").is_file())
            self.assertFalse((attempt / "reused_cluster.json").exists())
            self.assertTrue((
                attempt / "complete_child_bundle" / "child.json").is_file())
            self.assertTrue((attempt / "source_error_signatures.json").is_file())

    def test_reuse_lineage_preserves_cluster_hash(self) -> None:
        cluster = {
            "clusters": [
                {"cluster_id": "a", "sample_ids": ["s1", "s2"]}],
            "metrics": {"logical_evaluations": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            old_path = root / "old.json"
            current_path = attempt / "cluster_proposal.json"
            signature_path = root / "signatures.json"
            attribution_path = root / "failure.json"
            phase21._write(old_path, cluster)
            phase21._write(current_path, cluster)
            phase21._write(signature_path, {"outputs": {}})
            phase21._write(attribution_path, {
                "attribution": {"primary_failure_type": "no_effect"}})
            phase21._write(attempt / "children" / "child.json", {
                "criterion_name": "child"})
            phase21._record_split_lineage({
                "attempt_dir": attempt,
                "cluster_path": str(current_path),
                "signature_path": str(signature_path),
                "generated_child_ids": ["child"],
            }, {
                "action": "reuse_clusters_regenerate_complete_bundle",
                "cluster_path": str(old_path),
                "failure_attribution_path": str(attribution_path),
            }, protocol=evolution.PROTOCOL)
            diff = json.loads((attempt / "cluster_diff.json").read_text(
                encoding="utf-8"))
            self.assertEqual(
                diff["old_cluster_sha256"], diff["new_cluster_sha256"])
            self.assertTrue((attempt / "reused_cluster.json").is_file())
            self.assertFalse((attempt / "new_cluster.json").exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from critiq.structured.cache import CacheMode, JsonPredictionCache
from critiq.structured.dual_worker import (
    DualWorkerRequestSpec,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
)
from critiq.structured.judgement import FinalPreference, Vote
from critiq.structured.schema import RubricCriterionSnapshot, RubricNode, StructuredRubric
from critiq.structured.telemetry import ModelCallMetrics
from critiq.structured.worker_output import StructuredCriterionSnapshot
from experiments.evolving_structured_rubrics.run_shared_output_pool import (
    _agreement,
    _generation_stage_status,
    _model_rows,
    _pairwise_pred_cached,
    _progress,
    _replace_pairwise_node_output,
    _reserve50,
    _system_metrics,
)


class SharedOutputDatasetAndReportTest(unittest.TestCase):
    def test_generation_stage_separates_artifact_and_telemetry(self):
        self.assertEqual("failed", _generation_stage_status(0.949, True))
        self.assertEqual("passed", _generation_stage_status(0.95, True))
        self.assertEqual(
            "passed_with_telemetry_gap",
            _generation_stage_status(1.0, False),
        )

    def test_pairwise_cache_quarantine_is_recoverable(self):
        spec = DualWorkerRequestSpec(
            model="fake-model", backend_id="fake-pool", prompt_sha256="a" * 64,
            max_data_chars=None, encode_local_image=True, image_field="image_path",
            question_field="question", sample_id_field="sample_id",
            decoding_config={"temperature": 0.0},
        )
        from critiq.structured.cache import pairwise_cache_key_payload
        payload = pairwise_cache_key_payload(
            sample_fingerprint="b" * 64,
            criterion_name="criterion",
            criterion_description="description",
            request_spec=spec,
        )
        output = PairwiseVoteOutput(
            Vote.ABSTAIN, False, None, "failed", 1, None, False
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            cache = JsonPredictionCache(Path(directory), CacheMode.READ_WRITE)
            cache.put_pairwise(payload, output, ModelCallMetrics())
            target = cache.quarantine("pairwise", payload)
            self.assertTrue(target.is_file())
            self.assertIn("quarantine", target.parts)
            self.assertIsNone(cache.get_pairwise(payload))

    def test_replacing_one_pairwise_node_rebuilds_only_its_flat_answer(self):
        spec = DualWorkerRequestSpec(
            model="fake-model", backend_id="fake-pool", prompt_sha256="a" * 64,
            max_data_chars=None, encode_local_image=True, image_field="image_path",
            question_field="question", sample_id_field="sample_id",
            decoding_config={"temperature": 0.0},
        )
        invalid = PairwiseVoteOutput(
            Vote.ABSTAIN, False, None, "failed", 1, None, False
        )
        valid = PairwiseVoteOutput(
            Vote.A, True, '{"thought":"ok","answer":"A"}', None, 1, "ok", True
        )
        prediction = PairwisePredictionOutput(
            ("sample-1",), ("b" * 64,),
            (StructuredCriterionSnapshot("criterion", "description"),),
            ({"criterion": invalid},), (FinalPreference.TIE,), spec,
        )
        repaired = _replace_pairwise_node_output(
            prediction, "sample-1", "criterion", valid
        )
        self.assertIs(repaired.node_outputs[0]["criterion"], valid)
        self.assertEqual((FinalPreference.A,), repaired.flat_answers)
        self.assertFalse(prediction.node_outputs[0]["criterion"].parse_ok)

    def test_pairwise_generation_resumes_from_per_node_cache(self):
        class FakeEvaluator:
            max_concurrent = 1
            sample_id_field = "sample_id"

            def __init__(self) -> None:
                self.calls = 0
                self._spec = DualWorkerRequestSpec(
                    model="fake-model",
                    backend_id="fake-pool",
                    prompt_sha256="a" * 64,
                    max_data_chars=None,
                    encode_local_image=True,
                    image_field="image_path",
                    question_field="question",
                    sample_id_field="sample_id",
                    decoding_config={"temperature": 0.5},
                )

            def request_spec(self):
                return self._spec

            def sample_fingerprint(self, data):
                del data
                return "b" * 64

            def infer_one(self, data, criterion):
                del data, criterion
                self.calls += 1
                return (
                    PairwiseVoteOutput(
                        vote=Vote.A,
                        parse_ok=True,
                        raw_response='{"thought":"ok","answer":"A"}',
                        parse_error=None,
                        attempt_count=1,
                        thought="ok",
                        answer_valid=True,
                    ),
                    ModelCallMetrics(
                        logical_evaluations=1,
                        api_attempts=1,
                        input_tokens=10,
                        output_tokens=2,
                        total_tokens=12,
                    ),
                )

        node = RubricNode(
            "root",
            RubricCriterionSnapshot("criterion", "description", 0.5),
        )
        rubric = StructuredRubric({"root": node}, (), ("root",))
        rows = ({"sample_id": "sample-1"},)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            cache = JsonPredictionCache(Path(directory), CacheMode.READ_WRITE)
            first = FakeEvaluator()
            first_prediction, first_hits = _pairwise_pred_cached(
                first, rows, rubric, cache, lambda *_: None
            )
            second = FakeEvaluator()
            second_prediction, second_hits = _pairwise_pred_cached(
                second, rows, rubric, cache, lambda *_: None
            )

        self.assertEqual(first.calls, 1)
        self.assertEqual(first_hits, 0)
        self.assertEqual(second.calls, 0)
        self.assertEqual(second_hits, 1)
        self.assertEqual(first_prediction.to_dict(), second_prediction.to_dict())

    def test_reserve50_is_balanced_unique_and_seeded(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); image_root = root / "images"; image_root.mkdir()
            source = root / "reserve.jsonl"; rows = []
            for index in range(60):
                task = "question_answering" if index < 30 else "detailed_description"
                image = image_root / f"{index}.jpg"; image.write_bytes(b"image")
                rows.append({"sample_id": f"r{index}", "image_path": image.name,
                    "question": "q", "chosen": "chosen", "rejected": "rejected",
                    "origin_split": {"type": task}})
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            import hashlib
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            config = {"reserve_pool": str(source), "reserve_pool_sha256": sha,
                      "image_root": str(image_root), "reserve_seed": 42}
            first = _reserve50(config, ()); second = _reserve50(config, ())
            self.assertEqual(first, second)
            self.assertEqual(len(first), 50)
            self.assertEqual(len({row["sample_id"] for row in first}), 50)
            self.assertEqual(len({row["image_path"] for row in first}), 50)
            self.assertEqual(sum(row["answer"] == "A" for row in first), 25)
            selected = {row["sample_id"] for row in first}
            self.assertEqual(sum(int(value[1:]) < 30 for value in selected), 25)

    def test_accuracy_metrics_are_a_pure_offline_calculation(self):
        rows = ({"answer": "A"}, {"answer": "B"}, {"answer": "A"})
        metrics = _system_metrics(
            (FinalPreference.A, FinalPreference.TIE, FinalPreference.B), rows)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["covered_accuracy"], 1 / 2)
        self.assertAlmostEqual(metrics["tie_rate"], 1 / 3)

    def test_online_model_rows_cannot_access_gold_answer(self):
        rows = _model_rows(({"sample_id": "s", "A": "a", "B": "b", "answer": "A"},))
        self.assertNotIn("answer", rows[0])

    def test_three_run_agreement(self):
        self.assertAlmostEqual(_agreement((("A", "B"), ("A", "A"), ("A", "B"))), 4 / 6)

    def test_progress_is_printable_and_persisted(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            callback = _progress(output, "pairwise-p05", 2)
            callback(0, "sample-1", ModelCallMetrics(
                logical_evaluations=1, api_attempts=17,
                input_tokens=10, output_tokens=2, total_tokens=12))
            progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["stage"], "pairwise-p05")
            self.assertEqual(progress["completed"], 1)
            self.assertEqual(progress["total"], 2)
            self.assertTrue((output / "logs/pairwise-p05.log").is_file())

    def test_progress_lock_failure_does_not_abort_inference_callback(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            callback = _progress(output, "pairwise-p05", 1)
            with patch.object(Path, "replace", side_effect=PermissionError("locked")), \
                    patch("experiments.evolving_structured_rubrics.experiment_utils.time.sleep"):
                callback(0, "sample-1", ModelCallMetrics(logical_evaluations=1))
            log = (output / "logs/pairwise-p05.log").read_text(encoding="utf-8")
            self.assertIn("progress.json update skipped", log)


if __name__ == "__main__":
    unittest.main()

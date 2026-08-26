import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from critiq.agent import Agent, AgentCallMetrics
from critiq.dual_evaluator import (
    CacheOptimizedPairwiseVoteMultiModalEvaluator,
    PairwiseVoteMultiModalEvaluator,
)
from critiq.dual_worker_prompts import (
    PAIRWISE_MULTIMODAL_WORKER_PROMPT,
    PAIRWISE_MULTIMODAL_WORKER_SYSTEM_PROMPT_V2_CACHE,
)
from critiq.structured import CacheMode, JsonPredictionCache
from critiq.structured.version import PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION
from experiments.evolving_structured_rubrics import pairwise_cache_ablation as cache_exp
from experiments.evolving_structured_rubrics.rubric_factory import (
    build_multicrit_open_ended_init_rubric,
)


class _FakeBackend:
    backend_id = "fake-backend"

    def __init__(self):
        self.calls = []

    def call(self, prompt, *, request_type, request_key, structured_attempt,
             agent_args):
        self.calls.append({
            "prompt": prompt,
            "request_type": request_type,
            "request_key": request_key,
            "structured_attempt": structured_attempt,
            "system": agent_args.get("system"),
        })
        raw = json.dumps({
            "analysis_a": "A",
            "analysis_b": "B",
            "thought": "criterion-only comparison",
            "answer": "A",
        })
        return raw, AgentCallMetrics(
            api_attempts=1, input_tokens=100, output_tokens=20,
            total_tokens=120, usage_complete=True, cached_input_tokens=80)


class PairwiseCachePromptTests(unittest.TestCase):
    def _row(self, root: Path, sample_id: str):
        image = root / f"{sample_id}.jpg"
        image.write_bytes(b"test-image")
        return {
            "sample_id": sample_id,
            "image_path": str(image),
            "question": "What is visible?",
            "A": "A cat is visible.",
            "B": "A dog is visible.",
        }

    @staticmethod
    def _args():
        return {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "api_keys": "EMPTY",
            "request_kwargs": {"temperature": 0.5, "max_tokens": 2048},
            "api_retry_attempts": 0,
        }

    def test_prompt_v2_is_dataset_neutral_and_image_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root, "s0")
            rubric = build_multicrit_open_ended_init_rubric()
            criterion = rubric.get_node(rubric.root_ids[0]).criterion.to_criterion()
            fake = _FakeBackend()
            v1 = PairwiseVoteMultiModalEvaluator(
                worker_args=self._args(), dataset=(row,), backend_id=fake.backend_id,
                max_retries=0, call_backend=fake)
            v2 = CacheOptimizedPairwiseVoteMultiModalEvaluator(
                worker_args=self._args(), dataset=(row,), backend_id=fake.backend_id,
                max_retries=0, call_backend=fake)

            self.assertEqual(v1.worker_prompt, PAIRWISE_MULTIMODAL_WORKER_PROMPT)
            self.assertNotEqual(v1.request_spec().prompt_sha256,
                                v2.request_spec().prompt_sha256)
            self.assertEqual(v2.worker_args["system"],
                             PAIRWISE_MULTIMODAL_WORKER_SYSTEM_PROMPT_V2_CACHE)
            content = v2._make_user_content(row, criterion)
            self.assertEqual([item["type"] for item in content],
                             ["image_url", "text"])
            text = content[1]["text"]
            self.assertLess(text.index("## Candidate B"), text.index("## Criterion"))
            self.assertNotIn("RLHF-V", v2.worker_args["system"])
            self.assertNotIn("VL-RewardBench", v2.worker_args["system"])

    def test_seed_completes_before_fanout_and_pilot_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = (self._row(root, "s0"), self._row(root, "s1"))
            rubric = build_multicrit_open_ended_init_rubric()
            fake = _FakeBackend()
            evaluator = CacheOptimizedPairwiseVoteMultiModalEvaluator(
                worker_args=self._args(), dataset=rows, backend_id=fake.backend_id,
                max_retries=0, call_backend=fake)
            cache = JsonPredictionCache(root / "cache", CacheMode.READ_WRITE)
            prediction, traces = cache_exp._affinity_prediction(
                evaluator, rows, rubric, cache, None, 1,
                PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION)

            self.assertTrue(all(item["seed_before_fanout"] for item in traces))
            seed_name = rubric.get_node(rubric.preorder_node_ids()[0]).criterion.name
            for sample_id in ("s0", "s1"):
                calls = [item["request_key"] for item in fake.calls
                         if item["request_key"].startswith(sample_id + "::")]
                self.assertTrue(calls)
                self.assertEqual(calls[0], f"{sample_id}::{seed_name}")
            path = root / "prediction.json"
            prediction.save_json(path)
            loaded = type(prediction).load_json(path)
            self.assertEqual(loaded.prompt_version,
                             PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION)
            self.assertEqual(loaded.to_dict(), prediction.to_dict())

    def test_sample_affinity_keeps_all_nodes_on_one_endpoint(self):
        rubric = build_multicrit_open_ended_init_rubric()
        rows = ({"sample_id": "s0"}, {"sample_id": "s1"})
        routes = cache_exp._route_map(rows, rubric)
        for row in rows:
            endpoints = {
                routes[f"{row['sample_id']}::{criterion.name}"]
                for criterion in rubric.criteria_in_execution_order()
            }
            self.assertEqual(len(endpoints), 1)

    def test_s3_uses_prompt_v2_with_original_dynamic_scheduler(self):
        self.assertTrue(cache_exp._uses_v2_prompt(
            cache_exp.VARIANT_PROMPT_V2_DYNAMIC))
        self.assertFalse(cache_exp._uses_affinity_scheduler(
            cache_exp.VARIANT_PROMPT_V2_DYNAMIC))
        self.assertTrue(cache_exp._uses_affinity_scheduler(
            cache_exp.VARIANT_PROMPT_V2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root, "s3")
            fake = _FakeBackend()
            evaluator = cache_exp._make_evaluator(
                {
                    "worker_request_kwargs": {
                        "temperature": 0.5, "max_tokens": 2048},
                    "model": "Qwen/Qwen3-VL-8B-Instruct",
                    "api_retry_attempts": 0,
                    "structured_max_retries": 0,
                    "backend_pool": {
                        "pool_id": "fake",
                        "common_checkpoint_id": "Qwen/Qwen3-VL-8B-Instruct",
                        "global_request_concurrency": 1,
                        "endpoints": [{
                            "endpoint_id": "vllm-8000",
                            "base_url": "http://10.102.137.255:8000/v1",
                            "checkpoint_root": "/fake/checkpoint",
                            "max_concurrency": 1,
                        }, {
                            "endpoint_id": "vllm-8001",
                            "base_url": "http://10.102.138.0:8000/v1",
                            "checkpoint_root": "/fake/checkpoint",
                            "max_concurrency": 1,
                        }],
                    },
                },
                (row,), fake, cache_exp.VARIANT_PROMPT_V2_DYNAMIC)
            self.assertIsInstance(
                evaluator, CacheOptimizedPairwiseVoteMultiModalEvaluator)

    def test_dynamic_scheduler_prompt_metadata_can_be_corrected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = (self._row(root, "s3-metadata"),)
            rubric = build_multicrit_open_ended_init_rubric()
            fake = _FakeBackend()
            evaluator = CacheOptimizedPairwiseVoteMultiModalEvaluator(
                worker_args=self._args(), dataset=rows, backend_id=fake.backend_id,
                max_retries=0, call_backend=fake)
            cache = JsonPredictionCache(root / "cache", CacheMode.READ_WRITE)
            prediction = cache_exp.base._pairwise_cached(
                evaluator, rows, rubric, cache, None)

            self.assertEqual(prediction.prompt_version, "1.0.0")
            corrected = cache_exp._with_prompt_version(
                prediction, PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION)
            self.assertEqual(
                corrected.prompt_version,
                PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION)
            self.assertEqual(corrected.sample_ids, prediction.sample_ids)
            self.assertEqual(corrected.flat_answers, prediction.flat_answers)
            self.assertEqual(corrected.request_spec, prediction.request_spec)
            self.assertEqual(corrected.node_outputs, prediction.node_outputs)

    def test_s3_heldout_compares_only_control_and_dynamic_prompt_v2(self):
        self.assertEqual(
            cache_exp.S3_HELDOUT_VARIANTS,
            (cache_exp.VARIANT_CONTROL,
             cache_exp.VARIANT_PROMPT_V2_DYNAMIC))
        self.assertTrue(cache_exp._uses_v2_prompt(
            cache_exp.S3_HELDOUT_VARIANTS[1]))
        self.assertFalse(cache_exp._uses_affinity_scheduler(
            cache_exp.S3_HELDOUT_VARIANTS[1]))

    def test_node_delta_summary_counts_direction(self):
        def metric(strict, covered, coverage):
            return {
                "strict_accuracy": strict,
                "covered_accuracy": covered,
                "coverage": coverage,
            }

        value = cache_exp._node_delta_summary(
            {"up": metric(0.5, 0.6, 0.8),
             "same": metric(0.7, 0.7, 0.9),
             "down": metric(0.8, 0.8, 1.0)},
            {"up": metric(0.6, 0.7, 0.9),
             "same": metric(0.7, 0.7, 0.9),
             "down": metric(0.7, 0.75, 0.9)})
        self.assertEqual(value["improved_strict_accuracy_count"], 1)
        self.assertEqual(value["unchanged_strict_accuracy_count"], 1)
        self.assertEqual(value["declined_strict_accuracy_count"], 1)

    def test_agent_records_cached_prompt_tokens_when_reported(self):
        agent = Agent(model="unused", api_keys="EMPTY")
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=75),
        )
        agent._reset_call_metrics()
        agent._record_usage(usage)
        self.assertEqual(agent._cached_input_tokens, 75)

    def test_prometheus_delta_reports_prefix_hit_and_ttft(self):
        before = {
            endpoint: {name: 0.0 for name in cache_exp.PROMETHEUS_METRICS}
            for endpoint in cache_exp.ENDPOINT_IDS
        }
        after = {
            endpoint: {name: 0.0 for name in cache_exp.PROMETHEUS_METRICS}
            for endpoint in cache_exp.ENDPOINT_IDS
        }
        after["vllm-8000"]["vllm:prefix_cache_queries_total"] = 100
        after["vllm-8000"]["vllm:prefix_cache_hits_total"] = 60
        after["vllm-8001"]["vllm:prefix_cache_queries_total"] = 100
        after["vllm-8001"]["vllm:prefix_cache_hits_total"] = 40
        after["vllm-8000"]["vllm:time_to_first_token_seconds_count"] = 2
        after["vllm-8000"]["vllm:time_to_first_token_seconds_sum"] = 1.0
        value = cache_exp._server_metrics_delta(before, after)
        self.assertEqual(value["prefix_cache_token_hit_rate"], 0.5)
        self.assertEqual(value["mean_ttft_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()

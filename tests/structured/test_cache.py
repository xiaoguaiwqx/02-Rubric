import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from critiq.structured import (
    Applicability,
    CacheCorruptionError,
    CacheMode,
    CriterionStatus,
    JsonPredictionCache,
    ModelCallMetrics,
    NodeJudgement,
    OnlineNodeBackend,
    OnlineRouterBackend,
    PairPreference,
    RootRouterOutput,
    RootRouterRequestSpec,
    RootRoutingDecision,
    RubricCriterionSnapshot,
    RubricNode,
    StructuredNodeOutput,
    StructuredRootRouter,
    StructuredRubric,
    StructuredWorkerRequestSpec,
    canonical_sha256,
    node_cache_key_payload,
    router_cache_key_payload,
)


def worker_spec():
    return StructuredWorkerRequestSpec(
        model="m", worker_backend_id="backend", prompt_sha256="a" * 64,
        max_data_chars=None, encode_local_image=False, image_field="image_path",
        question_field="question", sample_id_field="sample_id",
        decoding_config={"temperature": 0},
    )


def router_spec():
    return RootRouterRequestSpec(
        model="m", router_backend_id="router", prompt_sha256="b" * 64,
        rubric_sha256="c" * 64, roots_sha256="d" * 64, max_data_chars=None,
        encode_local_image=False, image_field="image_path", question_field="question",
        sample_id_field="sample_id", decoding_config={"temperature": 0},
    )


def node_output():
    return StructuredNodeOutput(
        judgement=NodeJudgement(
            Applicability.YES, CriterionStatus.PASS, CriterionStatus.FAIL,
            PairPreference.A, "supported", "wrong",
        ),
        raw_response="{}", parse_error=None, attempt_count=1,
    )


class JsonPredictionCacheTest(unittest.TestCase):
    def test_atomic_cache_temp_name_does_not_repeat_long_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ("long-cache-directory-" * 4)
            cache = JsonPredictionCache(root)
            payload = node_cache_key_payload(
                sample_fingerprint="e" * 64, criterion_name="c",
                criterion_description="desc", request_spec=worker_spec(),
            )
            cache.put_node(payload, node_output(), ModelCallMetrics())
            self.assertEqual(node_output(), cache.get_node(payload).output)
            self.assertFalse(tuple((root / "node").glob("*.tmp")))

    def test_node_and_router_round_trip_include_invalid_results(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonPredictionCache(directory)
            node_key = node_cache_key_payload(
                sample_fingerprint="e" * 64, criterion_name="c",
                criterion_description="desc", request_spec=worker_spec(),
            )
            metrics = ModelCallMetrics(logical_evaluations=1, api_attempts=1, input_tokens=2, output_tokens=1, total_tokens=3)
            cache.put_node(node_key, node_output(), metrics)
            restored = cache.get_node(node_key)
            self.assertEqual(node_output(), restored.output)
            self.assertEqual(metrics, restored.generation_metrics)

            router_key = router_cache_key_payload(sample_fingerprint="f" * 64, request_spec=router_spec())
            invalid = RootRouterOutput(
                decision=RootRoutingDecision((), {}, True), raw_response="{}",
                parse_error=None, attempt_count=2,
            )
            cache.put_router(router_key, invalid, metrics)
            self.assertEqual(invalid, cache.get_router(router_key).output)

    def test_refresh_and_disabled_do_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = node_cache_key_payload(
                sample_fingerprint="e" * 64, criterion_name="c",
                criterion_description="desc", request_spec=worker_spec(),
            )
            writer = JsonPredictionCache(directory)
            writer.put_node(payload, node_output(), ModelCallMetrics())
            self.assertIsNone(JsonPredictionCache(directory, CacheMode.REFRESH).get_node(payload))
            self.assertIsNone(JsonPredictionCache(Path(directory) / "disabled", CacheMode.DISABLED).get_node(payload))

    def test_key_is_sensitive_and_corruption_is_not_a_miss(self):
        first = node_cache_key_payload(
            sample_fingerprint="e" * 64, criterion_name="c",
            criterion_description="one", request_spec=worker_spec(),
        )
        second = node_cache_key_payload(
            sample_fingerprint="e" * 64, criterion_name="c",
            criterion_description="two", request_spec=worker_spec(),
        )
        self.assertNotEqual(canonical_sha256(first), canonical_sha256(second))
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonPredictionCache(directory)
            cache.put_node(first, node_output(), ModelCallMetrics())
            path = Path(directory) / "node" / f"{canonical_sha256(first)}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["key_sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(CacheCorruptionError):
                cache.get_node(first)

    def test_online_node_backend_cold_warm_and_refresh(self):
        class FakeEvaluator:
            sample_id_field = "sample_id"

            def __init__(self):
                self.calls = 0

            def sample_fingerprint(self, data):
                return "e" * 64

            def request_spec(self):
                return worker_spec()

            def infer_one(self, data, criterion):
                self.calls += 1
                metrics = ModelCallMetrics(logical_evaluations=1, api_attempts=1, input_tokens=2, output_tokens=1, total_tokens=3)
                return node_output(), metrics

        node = RubricNode("n", RubricCriterionSnapshot("c", "desc", 0.5))
        data = {"sample_id": "s"}
        with tempfile.TemporaryDirectory() as directory:
            evaluator = FakeEvaluator()
            backend = OnlineNodeBackend(evaluator, JsonPredictionCache(directory))
            cold = backend.evaluate(data, node)
            warm = backend.evaluate(data, node)
            self.assertEqual(1, evaluator.calls)
            self.assertEqual(1, cold.metrics.cache_misses)
            self.assertEqual(1, warm.metrics.cache_hits)
            self.assertEqual(0, warm.metrics.api_attempts)

            refresh = OnlineNodeBackend(evaluator, JsonPredictionCache(directory, CacheMode.REFRESH))
            refresh.evaluate(data, node)
            self.assertEqual(2, evaluator.calls)

    def test_online_router_backend_cold_then_warm_has_zero_api_calls(self):
        rubric = StructuredRubric(
            nodes={"r": RubricNode("r", RubricCriterionSnapshot("c", "desc", 0.5))},
            edges=(),
            root_ids=("r",),
        )
        data = {
            "sample_id": "s",
            "image_path": "https://example.invalid/image.jpg",
            "question": "q",
            "A": "a",
            "B": "b",
        }
        router = StructuredRootRouter(
            router_args={"model": "m", "request_kwargs": {"temperature": 0}},
            router_backend_id="router-checkpoint",
            max_retries=0,
            encode_local_image=False,
        )
        output = RootRouterOutput(
            RootRoutingDecision(("r",), {"r": "relevant"}, True),
            "{}",
            None,
            1,
        )
        generated = ModelCallMetrics(
            logical_evaluations=1,
            api_attempts=1,
            input_tokens=4,
            output_tokens=2,
            total_tokens=6,
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = OnlineRouterBackend(router, JsonPredictionCache(directory))
            with patch.object(
                router,
                "route_one_uncached",
                return_value=(output, generated),
            ) as call:
                cold = backend.route(data, rubric)
                warm = backend.route(data, rubric)
            self.assertEqual(1, call.call_count)
            self.assertEqual(1, cold.metrics.api_attempts)
            self.assertEqual(1, cold.metrics.cache_misses)
            self.assertEqual(0, warm.metrics.api_attempts)
            self.assertEqual(1, warm.metrics.cache_hits)
            self.assertEqual(generated, warm.generation_metrics)


if __name__ == "__main__":
    unittest.main()

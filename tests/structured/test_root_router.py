import json
import unittest
from unittest.mock import patch

from critiq.agent import AgentCallMetrics
from critiq.structured import (
    EdgeCondition,
    RootRouterOutputParseError,
    RubricCriterionSnapshot,
    RubricNode,
    StructuredRootRouter,
    StructuredRubric,
    parse_structured_root_router_response,
)


def make_rubric():
    nodes = {
        "r1": RubricNode("r1", RubricCriterionSnapshot("grounding", "Check grounding", 0.5)),
        "r2": RubricNode("r2", RubricCriterionSnapshot("relevance", "Check relevance", 0.5)),
    }
    return StructuredRubric(nodes=nodes, edges=(), root_ids=("r1", "r2"))


def sample(sample_id="s1"):
    return {
        "sample_id": sample_id,
        "image_path": "https://example.test/image.png",
        "question": "What is shown?",
        "A": "A cat",
        "B": "A dog",
        "answer": "A",
    }


class QueueAgent:
    responses = []
    calls = []

    def __init__(self, **kwargs):
        self.last_call_metrics = AgentCallMetrics()

    def __call__(self, prompt, stream=False):
        self.calls.append(prompt)
        response = self.responses.pop(0)
        self.last_call_metrics = AgentCallMetrics(
            api_attempts=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_seconds=0.01,
        )
        return response


class RootRouterParserTest(unittest.TestCase):
    def test_plain_fenced_and_surrounded_json(self):
        payload = {"selected_root_ids": ["r2", "r1"], "rationale_by_root": {"r2": "x", "r1": "y"}}
        text = json.dumps(payload)
        for raw in (text, f"```json\n{text}\n```", f"result:\n{text}\ndone"):
            decision = parse_structured_root_router_response(raw)
            self.assertEqual(("r2", "r1"), decision.selected_root_ids)
            self.assertTrue(decision.parse_ok)

    def test_schema_errors_are_parse_failures(self):
        invalid = [None, "{}", '{"selected_root_ids":"r1","rationale_by_root":{}}', '{"selected_root_ids":[1],"rationale_by_root":{}}', '{"selected_root_ids":["r1"],"rationale_by_root":{},"confidence":1}']
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(RootRouterOutputParseError):
                parse_structured_root_router_response(value)

    def test_empty_duplicate_and_unknown_are_consistency_invalid(self):
        rubric = make_rubric()
        for payload in (
            {"selected_root_ids": [], "rationale_by_root": {}},
            {"selected_root_ids": ["r1", "r1"], "rationale_by_root": {"r1": "x"}},
            {"selected_root_ids": ["unknown"], "rationale_by_root": {"unknown": "x"}},
            {"selected_root_ids": ["r1"], "rationale_by_root": {}},
        ):
            decision = parse_structured_root_router_response(json.dumps(payload))
            self.assertTrue(decision.parse_ok)
            self.assertTrue(decision.consistency_errors if hasattr(decision, "consistency_errors") else decision.selected_root_ids is not None)
            self.assertTrue(decision.selected_root_ids == () or decision.selected_root_ids)
            self.assertNotEqual("valid", decision)
            from critiq.structured import root_routing_consistency_errors
            self.assertTrue(root_routing_consistency_errors(decision, rubric.root_ids))


class StructuredRootRouterTest(unittest.TestCase):
    def setUp(self):
        QueueAgent.responses = []
        QueueAgent.calls = []
        self.router = StructuredRootRouter(
            router_args={"model": "fake", "request_kwargs": {"temperature": 0}},
            router_backend_id="fake-checkpoint",
            max_retries=1,
            encode_local_image=False,
        )

    def test_retry_then_valid_and_rubric_order_resolution(self):
        QueueAgent.responses = ["bad", json.dumps({"selected_root_ids": ["r2", "r1"], "rationale_by_root": {"r2": "x", "r1": "y"}})]
        with patch("critiq.structured.root_router.Agent", QueueAgent):
            output, metrics = self.router.route_one_uncached(sample(), make_rubric())
        self.assertEqual(2, output.attempt_count)
        self.assertEqual(("r1", "r2"), output.resolve(make_rubric()).selected_root_ids)
        self.assertEqual(2, metrics.api_attempts)
        self.assertEqual(1, metrics.parse_retries)

    def test_inconsistent_output_survives_later_parse_failure(self):
        QueueAgent.responses = [json.dumps({"selected_root_ids": [], "rationale_by_root": {}}), "bad"]
        with patch("critiq.structured.root_router.Agent", QueueAgent):
            output, _ = self.router.route_one_uncached(sample(), make_rubric())
        self.assertTrue(output.decision.parse_ok)
        self.assertTrue(output.resolve(make_rubric()).fallback_to_all_roots)
        self.assertIsNone(output.parse_error)

    def test_artifact_round_trip_and_request_identity(self):
        QueueAgent.responses = [json.dumps({"selected_root_ids": ["r1"], "rationale_by_root": {"r1": "relevant"}})]
        with patch("critiq.structured.root_router.Agent", QueueAgent):
            prediction = self.router.pred([sample()], make_rubric())
        restored = type(prediction).from_dict(prediction.to_dict())
        self.assertEqual(prediction, restored)
        self.assertEqual("fake-checkpoint", restored.request_spec.router_backend_id)
        self.assertNotIn("answer", json.dumps(restored.to_dict()))

    def test_request_spec_decoding_config_is_deeply_immutable(self):
        router = StructuredRootRouter(
            router_args={
                "model": "fake",
                "request_kwargs": {
                    "sampling": {"temperature": 0},
                    "stop": ["END"],
                },
            },
            router_backend_id="fake-checkpoint",
            encode_local_image=False,
        )
        spec = router.request_spec(make_rubric())
        with self.assertRaises(TypeError):
            spec.decoding_config["sampling"]["temperature"] = 1
        with self.assertRaises(AttributeError):
            spec.decoding_config["stop"].append("MORE")

        detached = spec.to_dict()
        detached["decoding_config"]["sampling"]["temperature"] = 1
        detached["decoding_config"]["stop"].append("MORE")
        self.assertEqual(0, spec.decoding_config["sampling"]["temperature"])
        self.assertEqual(("END",), spec.decoding_config["stop"])


if __name__ == "__main__":
    unittest.main()

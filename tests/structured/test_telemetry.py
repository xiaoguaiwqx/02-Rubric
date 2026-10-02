import types
import unittest
from unittest.mock import patch

from critiq import Agent
from critiq.structured.telemetry import ModelCallMetrics, TokenPricing, combine_model_call_metrics


class FakeCompletions:
    def __init__(self, usage):
        self.usage = usage

    def create(self, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))],
            usage=self.usage,
        )


class FakeOpenAI:
    usage = None

    def __init__(self, **kwargs):
        self.chat = types.SimpleNamespace(completions=FakeCompletions(self.usage))


class AgentTelemetryTest(unittest.TestCase):
    def test_agent_return_is_unchanged_and_usage_is_recorded(self):
        FakeOpenAI.usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        with patch("critiq.agent.OpenAI", FakeOpenAI):
            agent = Agent(model="fake")
            self.assertEqual("ok", agent("hello", stream=False))
        metrics = agent.last_call_metrics
        self.assertEqual(1, metrics.api_attempts)
        self.assertEqual(10, metrics.input_tokens)
        self.assertEqual(4, metrics.output_tokens)
        self.assertTrue(metrics.usage_complete)
        with self.assertRaises(AttributeError):
            agent.last_call_metrics = metrics
        with self.assertRaises(AttributeError):
            agent.last_call_metrics = metrics

    def test_missing_usage_is_not_estimated(self):
        FakeOpenAI.usage = None
        with patch("critiq.agent.OpenAI", FakeOpenAI):
            agent = Agent(model="fake")
            agent("hello", stream=False)
        self.assertFalse(agent.last_call_metrics.usage_complete)
        self.assertIsNone(agent.last_call_metrics.total_tokens)

    def test_successful_retry_keeps_usage_complete_and_counts_errors(self):
        class FlakyCompletions:
            calls = 0

            def create(self, **kwargs):
                del kwargs
                self.__class__.calls += 1
                if self.__class__.calls < 3:
                    raise RuntimeError("transient")
                usage = types.SimpleNamespace(
                    prompt_tokens=2,
                    completion_tokens=1,
                    total_tokens=3,
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(content="ok"))],
                    usage=usage,
                )

        class FlakyOpenAI:
            def __init__(self, **kwargs):
                del kwargs
                self.chat = types.SimpleNamespace(
                    completions=FlakyCompletions())

        FlakyCompletions.calls = 0
        with patch("critiq.agent.OpenAI", FlakyOpenAI), patch("critiq.agent.sleep"):
            agent = Agent(model="fake", api_retry_attempts=2)
            self.assertEqual("ok", agent("hello", stream=False))
        self.assertEqual(3, agent.last_call_metrics.api_attempts)
        self.assertEqual(2, agent.last_call_metrics.error_count)
        self.assertTrue(agent.last_call_metrics.usage_complete)
        self.assertEqual(3, agent.last_call_metrics.total_tokens)
        with self.assertRaises(ValueError):
            Agent(model="fake", api_retry_attempts=-1)

    def test_structured_metrics_cost_and_incomplete_aggregation(self):
        pricing = TokenPricing(2.0, 4.0)
        first = ModelCallMetrics(
            logical_evaluations=1, api_attempts=1, input_tokens=1_000_000,
            output_tokens=500_000, total_tokens=1_500_000, estimated_cost_usd=4.0,
        )
        second = ModelCallMetrics(
            logical_evaluations=1, api_attempts=1, input_tokens=None,
            output_tokens=None, total_tokens=None, usage_complete=False,
        )
        self.assertEqual(4.0, pricing.estimate(1_000_000, 500_000))
        combined = combine_model_call_metrics((first, second))
        self.assertFalse(combined.usage_complete)
        self.assertIsNone(combined.total_tokens)
        self.assertIsNone(combined.estimated_cost_usd)


if __name__ == "__main__":
    unittest.main()

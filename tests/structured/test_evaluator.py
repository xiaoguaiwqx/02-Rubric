"""Tests for StructuredMultiModalPairEvaluator without model or network calls."""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import patch

from critiq.evaluator import (
    MultiModalPairEvaluator,
    StructuredMultiModalPairEvaluator,
)
from critiq.structured import (
    FinalPreference,
    OutcomeReason,
    PairPreference,
    STRUCTURED_WORKER_PROMPT_VERSION,
    structured_worker_prompt_sha256,
)
from critiq.structured_prompts import (
    STRUCTURED_MULTIMODAL_WORKER_PROMPT,
    STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1,
    STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1_1,
    STRUCTURED_MULTIMODAL_WORKER_PROMPT_V2,
    STRUCTURED_MULTIMODAL_WORKER_PROMPT_V2_1,
    STRUCTURED_WORKER_PROMPT_POSTFIX,
    STRUCTURED_WORKER_PROMPT_POSTFIX_V1,
    STRUCTURED_WORKER_PROMPT_POSTFIX_V1_1,
    STRUCTURED_WORKER_PROMPT_POSTFIX_V2,
    STRUCTURED_WORKER_PROMPT_POSTFIX_V2_1,
)
from critiq.utils import Criterion


def response_for(
    preference: str,
    *,
    applicable: str = "yes",
) -> str:
    if applicable != "yes":
        status_a = "uncertain"
        status_b = "uncertain"
        preference = "uncertain"
    elif preference == "A":
        status_a, status_b = "pass", "fail"
    elif preference == "B":
        status_a, status_b = "fail", "pass"
    else:
        status_a, status_b = "pass", "pass"
    return json.dumps(
        {
            "applicable": applicable,
            "status_a": status_a,
            "status_b": status_b,
            "pair_preference": preference,
            "evidence_a": "evidence for A",
            "evidence_b": "evidence for B",
        }
    )


def sample(
    sample_id: str,
    *,
    answer: str = "A",
    candidate_a: str = "answer A",
    candidate_b: str = "answer B",
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "image_path": "https://example.invalid/image.jpg",
        "question": f"question for {sample_id}",
        "A": candidate_a,
        "B": candidate_b,
        "answer": answer,
    }


class QueueAgent:
    responses: list[object] = []
    calls: list[object] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def __call__(self, prompt: object, stream: bool = False) -> object:
        del stream
        self.__class__.calls.append(prompt)
        if not self.__class__.responses:
            raise AssertionError("QueueAgent response queue is empty")
        return self.__class__.responses.pop(0)

    @classmethod
    def reset(cls, responses: list[object]) -> None:
        cls.responses = list(responses)
        cls.calls = []


class PromptRoutingAgent:
    """Return prompt-dependent outputs with delays that reverse completion order."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def __call__(self, prompt: object, stream: bool = False) -> str:
        del stream
        text = prompt[0]["text"]
        cases = {
            ("question for s1", "**c1**"): (0.04, "A"),
            ("question for s1", "**c2**"): (0.03, "tie"),
            ("question for s2", "**c1**"): (0.02, "B"),
            ("question for s2", "**c2**"): (0.01, "A"),
        }
        for (question, criterion), (delay, preference) in cases.items():
            if question in text and criterion in text:
                time.sleep(delay)
                return response_for(preference)
        raise AssertionError("unexpected prompt")


class StructuredEvaluatorPromptTest(unittest.TestCase):
    def make_evaluator(
        self,
        dataset: list[dict[str, str]],
        *,
        max_retries: int = 0,
    ) -> StructuredMultiModalPairEvaluator:
        return StructuredMultiModalPairEvaluator(
            worker_args={"model": "fake-model", "request_kwargs": {"temperature": 0}},
            dataset=dataset,
            worker_backend_id="fake-backend@revision-1",
            max_concurrent=1,
            max_retries=max_retries,
            encode_local_image=False,
        )

    def test_multimodal_prompt_has_text_and_image_without_path_leak(self) -> None:
        evaluator = self.make_evaluator([sample("s1")])
        criterion = Criterion("visual_grounding", "Use visible evidence.")
        content = evaluator._make_user_content(evaluator.dataset[0], criterion)
        self.assertIsInstance(content, list)
        text = content[0]["text"]
        self.assertIn("question for s1", text)
        self.assertIn("visual_grounding", text)
        self.assertIn("answer A", text)
        self.assertIn("answer B", text)
        self.assertNotIn("https://example.invalid/image.jpg", text)
        self.assertEqual(
            "https://example.invalid/image.jpg",
            content[1]["image_url"]["url"],
        )
        self.assertNotIn('"yes / no / uncertain"', STRUCTURED_WORKER_PROMPT_POSTFIX)
        self.assertNotIn(
            "Candidate A is supported by visible evidence.",
            STRUCTURED_WORKER_PROMPT_POSTFIX,
        )

    def test_prompt_versions_and_active_v1_1_contract(self) -> None:
        self.assertEqual("1.1.0", STRUCTURED_WORKER_PROMPT_VERSION)
        self.assertIn(
            "First make one binding applicability decision",
            STRUCTURED_MULTIMODAL_WORKER_PROMPT,
        )
        self.assertIn(
            'Never output pass, fail, A, B, or tie when applicable is not "yes"',
            STRUCTURED_WORKER_PROMPT_POSTFIX,
        )
        self.assertIs(
            STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1_1,
            STRUCTURED_MULTIMODAL_WORKER_PROMPT,
        )
        self.assertIs(
            STRUCTURED_WORKER_PROMPT_POSTFIX_V1_1,
            STRUCTURED_WORKER_PROMPT_POSTFIX,
        )
        self.assertEqual(
            "9281382e8241d1eb4393d89753c2faea114cebc38b91c55cfa6867ee7e4a4ebc",
            structured_worker_prompt_sha256(
                STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1_1,
                STRUCTURED_WORKER_PROMPT_POSTFIX_V1_1,
            ),
        )
        self.assertEqual(
            "5f9bf7e6c42520d7e0e6273802cfe1511ec764cfe381d029a38a1d544656e23b",
            structured_worker_prompt_sha256(
                STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1,
                STRUCTURED_WORKER_PROMPT_POSTFIX_V1,
            ),
        )
        self.assertEqual(
            "7132e9aeb54a3cfb3cb12013fb5eb4b58b96812a56a603b6784381604c7a6588",
            structured_worker_prompt_sha256(
                STRUCTURED_MULTIMODAL_WORKER_PROMPT_V2,
                STRUCTURED_WORKER_PROMPT_POSTFIX_V2,
            ),
        )
        self.assertEqual(
            "ebf04dbba4e63719aca76214ba1c23b1cd8ca90816a87e29310285bac3624cbb",
            structured_worker_prompt_sha256(
                STRUCTURED_MULTIMODAL_WORKER_PROMPT_V2_1,
                STRUCTURED_WORKER_PROMPT_POSTFIX_V2_1,
            ),
        )
    def test_custom_prompt_and_request_controls_are_fingerprinted(self) -> None:
        custom_prompt = (
            "Q={question}\nC={criterion}: {description}\nA={A}\nB={B}"
        )
        evaluator = StructuredMultiModalPairEvaluator(
            worker_args={"model": "fake-model", "request_kwargs": {"temperature": 0}},
            dataset=[sample("s1", candidate_a="abcdef", candidate_b="uvwxyz")],
            worker_backend_id="fake-backend@revision-1",
            worker_prompt=custom_prompt,
            max_data_chars=3,
            encode_local_image=False,
        )
        QueueAgent.reset([response_for("A")])
        with patch("critiq.evaluator.Agent", QueueAgent):
            prediction = evaluator.pred([Criterion("c1", "criterion")])

        self.assertEqual(
            structured_worker_prompt_sha256(
                custom_prompt,
                STRUCTURED_WORKER_PROMPT_POSTFIX,
            ),
            prediction.request_spec.prompt_sha256,
        )
        self.assertEqual(3, prediction.request_spec.max_data_chars)
        self.assertFalse(prediction.request_spec.encode_local_image)
        prompt_text = QueueAgent.calls[0][0]["text"]
        self.assertIn("A=abc", prompt_text)
        self.assertIn("B=uvw", prompt_text)
        self.assertNotIn("abcdef", prompt_text)


    def test_parse_failure_retries_then_returns_valid_output(self) -> None:
        QueueAgent.reset(["not-json", response_for("A")])
        evaluator = self.make_evaluator([sample("s1")], max_retries=1)
        with patch("critiq.evaluator.Agent", QueueAgent):
            output = evaluator.pred([Criterion("c1", "criterion")])
        node = output.node_outputs[0]["c1"]
        self.assertEqual(2, node.attempt_count)
        self.assertTrue(node.judgement.consistency_ok)
        self.assertEqual(PairPreference.A, node.judgement.pair_preference)
        self.assertEqual(2, len(QueueAgent.calls))

    def test_inconsistent_output_survives_later_parse_failure(self) -> None:
        inconsistent = json.dumps(
            {
                "applicable": "yes",
                "status_a": "pass",
                "status_b": "fail",
                "pair_preference": "B",
                "evidence_a": "",
                "evidence_b": "",
            }
        )
        QueueAgent.reset([inconsistent, "not-json"])
        evaluator = self.make_evaluator([sample("s1")], max_retries=1)
        with patch("critiq.evaluator.Agent", QueueAgent):
            output = evaluator.pred([Criterion("c1", "criterion")])
        node = output.node_outputs[0]["c1"]
        self.assertTrue(node.judgement.parse_ok)
        self.assertFalse(node.judgement.consistency_ok)
        self.assertIsNone(node.parse_error)
        self.assertEqual(
            OutcomeReason.CONSISTENCY_INVALID,
            node.local_decision.outcome_reason,
        )
        self.assertEqual(2, node.attempt_count)

    def test_exhausted_parse_failures_are_not_plain_abstain(self) -> None:
        QueueAgent.reset([None, "bad", "{}"])
        evaluator = self.make_evaluator([sample("s1")], max_retries=2)
        with patch("critiq.evaluator.Agent", QueueAgent):
            output = evaluator.pred([Criterion("c1", "criterion")])
        node = output.node_outputs[0]["c1"]
        self.assertFalse(node.judgement.parse_ok)
        self.assertTrue(node.parse_error)
        self.assertEqual(3, node.attempt_count)
        self.assertEqual(OutcomeReason.PARSE_FAILURE, node.local_decision.outcome_reason)
        self.assertEqual(3, len(QueueAgent.calls))


class StructuredEvaluatorBatchTest(unittest.TestCase):
    def evaluator(
        self,
        dataset: list[dict[str, str]],
    ) -> StructuredMultiModalPairEvaluator:
        return StructuredMultiModalPairEvaluator(
            worker_args={"model": "fake-model", "request_kwargs": {"temperature": 0}},
            dataset=dataset,
            worker_backend_id="fake-backend@revision-1",
            max_concurrent=1,
            max_retries=0,
            encode_local_image=False,
        )

    def test_batch_order_flat_metrics_and_criterion_coverage(self) -> None:
        dataset = [sample("s1", answer="A"), sample("s2", answer="B")]
        criteria = [Criterion("c1", "first"), Criterion("c2", "second")]
        QueueAgent.reset(
            [
                response_for("A"),
                response_for("A"),
                response_for("B"),
                response_for("tie"),
            ]
        )
        with patch("critiq.evaluator.Agent", QueueAgent):
            evaluated = self.evaluator(dataset).eval(criteria)

        self.assertEqual(("s1", "s2"), evaluated.prediction.sample_ids)
        self.assertEqual(
            (FinalPreference.A, FinalPreference.B),
            evaluated.prediction.flat_answers,
        )
        self.assertEqual((True, True), evaluated.is_correct)
        self.assertEqual(1.0, evaluated.accuracy)
        self.assertEqual(1.0, evaluated.coverage)
        self.assertEqual({"c1": 1.0, "c2": 1.0}, evaluated.per_criterion_accuracy)
        self.assertEqual({"c1": 1.0, "c2": 0.5}, evaluated.per_criterion_coverage)
        self.assertEqual(
            evaluated.prediction.flat_answers,
            evaluated.prediction.replay_flat_answers(),
        )
        with patch(
            "critiq.evaluator.Agent",
            side_effect=AssertionError("replay must not create Agent"),
        ):
            self.assertEqual(
                evaluated.prediction.flat_answers,
                evaluated.prediction.replay_flat_answers(),
            )

    def test_concurrent_completion_does_not_change_matrix_order(self) -> None:
        evaluator = StructuredMultiModalPairEvaluator(
            worker_args={"model": "fake-model"},
            dataset=[sample("s1"), sample("s2")],
            worker_backend_id="fake-backend@revision-1",
            max_concurrent=4,
            max_retries=0,
            encode_local_image=False,
        )
        criteria = [Criterion("c1", "first"), Criterion("c2", "second")]
        with patch("critiq.evaluator.Agent", PromptRoutingAgent):
            prediction = evaluator.pred(criteria)
        self.assertEqual(
            PairPreference.A,
            prediction.node_outputs[0]["c1"].judgement.pair_preference,
        )
        self.assertEqual(
            PairPreference.TIE,
            prediction.node_outputs[0]["c2"].judgement.pair_preference,
        )
        self.assertEqual(
            PairPreference.B,
            prediction.node_outputs[1]["c1"].judgement.pair_preference,
        )
        self.assertEqual(
            PairPreference.A,
            prediction.node_outputs[1]["c2"].judgement.pair_preference,
        )

    def test_flat_tie_and_all_abstain_have_zero_coverage(self) -> None:
        dataset = [sample("s1"), sample("s2")]
        criteria = [Criterion("c1", "first"), Criterion("c2", "second")]
        QueueAgent.reset(
            [
                response_for("A"),
                response_for("B"),
                response_for("uncertain", applicable="no"),
                response_for("uncertain", applicable="no"),
            ]
        )
        with patch("critiq.evaluator.Agent", QueueAgent):
            evaluated = self.evaluator(dataset).eval(criteria)
        self.assertEqual(
            (FinalPreference.TIE, FinalPreference.TIE),
            evaluated.prediction.flat_answers,
        )
        self.assertEqual(0.0, evaluated.accuracy)
        self.assertEqual(0.0, evaluated.coverage)

    def test_criteria_are_copied_not_replaced_in_caller_container(self) -> None:
        dataset = [sample("s1")]
        criteria: list[object] = [{"name": "c1", "description": "criterion"}]
        QueueAgent.reset([response_for("A")])
        with patch("critiq.evaluator.Agent", QueueAgent):
            self.evaluator(dataset).pred(criteria)
        self.assertIsInstance(criteria[0], dict)

    def test_dataset_and_criterion_contracts_fail_early(self) -> None:
        worker_args = {"model": "fake"}
        with self.assertRaises(ValueError):
            StructuredMultiModalPairEvaluator(worker_args, [], "fake-backend")
        with self.assertRaises(ValueError):
            StructuredMultiModalPairEvaluator(
                worker_args,
                [sample("duplicate"), sample("duplicate")],
                "fake-backend",
                encode_local_image=False,
            )
        invalid = sample("s1")
        invalid["answer"] = "Tie"
        with self.assertRaises(ValueError):
            StructuredMultiModalPairEvaluator(
                worker_args,
                [invalid],
                "fake-backend",
                encode_local_image=False,
            )

        evaluator = self.evaluator([sample("s1")])
        with self.assertRaises(ValueError):
            evaluator.pred([])
        with self.assertRaises(ValueError):
            evaluator.pred(
                [Criterion("same", "first"), Criterion("same", "second")]
            )
        with self.assertRaises(ValueError):
            StructuredMultiModalPairEvaluator(
                worker_args,
                [sample("s1")],
                "fake-backend",
                worker_prompt="{question}",
                encode_local_image=False,
            )


class OriginalEvaluatorRegressionTest(unittest.TestCase):
    def test_original_multimodal_evaluator_keeps_abu_output(self) -> None:
        QueueAgent.reset(
            [json.dumps({"thought": "legacy thought", "answer": "A"})]
        )
        evaluator = MultiModalPairEvaluator(
            worker_args={"model": "fake-model"},
            dataset=[sample("s1")],
            max_concurrent=1,
            max_retries=0,
            encode_local_image=False,
        )
        with patch("critiq.evaluator.Agent", QueueAgent):
            output = evaluator.pred([Criterion("legacy", "legacy criterion")])
        self.assertEqual(
            [{"legacy": {"A": 1, "B": 0, "U": 0}}],
            output.prediction,
        )
        self.assertEqual(["A"], output.answer)
        self.assertEqual([{"legacy": "legacy thought"}], output.thoughts)


if __name__ == "__main__":
    unittest.main()

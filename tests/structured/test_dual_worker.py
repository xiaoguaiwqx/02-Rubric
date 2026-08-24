import json
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import demo_rlhfv

from critiq.dual_worker_prompts import (
    GATE_STATE_WORKER_PROMPT,
    GATE_STATE_WORKER_PROMPT_V2,
    GATE_STATE_WORKER_PROMPT_V2_1,
    PAIRWISE_MULTIMODAL_WORKER_PROMPT,
    PAIRWISE_WORKER_PROMPT_POSTFIX,
)
from critiq.i18n import ENGLISH_PROMPT
from critiq.agent import AgentCallMetrics
from critiq.dual_evaluator import GateStateMultiModalEvaluator, PairwiseVoteMultiModalEvaluator
from critiq.utils import Criterion
from critiq.structured import (
    Applicability,
    CriterionStatus,
    DualWorkerRequestSpec,
    GateJudgement,
    GateStateOutput,
    GatePredictionOutput,
    JsonPredictionCache,
    ModelCallMetrics,
    GateStateParseError,
    PairwiseVoteOutput,
    PairwisePredictionOutput,
    FinalPreference,
    GATE_WORKER_PROMPT_V2_1_PILOT_VERSION,
    GATE_WORKER_PROMPT_V2_PILOT_VERSION,
    StructuredCriterionSnapshot,
    PairwiseVoteParseError,
    Vote,
    parse_gate_state_response,
    parse_pairwise_vote_response,
    recover_cached_pairwise_vote_output,
    gate_cache_key_payload,
    pairwise_cache_key_payload,
    worker_prompt_sha256,
)


class DualWorkerProtocolTests(unittest.TestCase):
    def test_gate_v2_is_minimal_generic_and_versioned(self):
        self.assertIn("status is absolute", GATE_STATE_WORKER_PROMPT_V2)
        self.assertIn("must not depend on whether the candidate is shown", GATE_STATE_WORKER_PROMPT_V2)
        self.assertIn("Do not downgrade a candidate", GATE_STATE_WORKER_PROMPT_V2)
        self.assertNotIn("multimodal_alignment", GATE_STATE_WORKER_PROMPT_V2)
        self.assertNotEqual(GATE_STATE_WORKER_PROMPT, GATE_STATE_WORKER_PROMPT_V2)

        row = {"sample_id": "s", "image_path": "unused.jpg", "question": "q",
               "A": "a", "B": "b", "answer": "A"}
        criterion = Criterion("c", "d", 0.5)

        class FakeAgent:
            def __init__(self, **_kwargs):
                self.last_call_metrics = AgentCallMetrics(api_attempts=1)

            def __call__(self, _prompt, stream=False):
                return '{"applicable":"yes","status_a":"pass","status_b":"fail"}'

        with patch("critiq.dual_evaluator.Agent", FakeAgent):
            evaluator = GateStateMultiModalEvaluator(
                worker_args={"request_kwargs": {"temperature": 0}},
                dataset=[row],
                backend_id="b",
                worker_prompt=GATE_STATE_WORKER_PROMPT_V2,
                encode_local_image=False,
            )
            output = evaluator.pred((criterion,))
        self.assertEqual(output.prompt_version, GATE_WORKER_PROMPT_V2_PILOT_VERSION)
        self.assertEqual(GatePredictionOutput.from_dict(output.to_dict()), output)

    def test_gate_v2_1_adds_only_position_stable_decision_procedure(self):
        self.assertIn("separate judgment passes", GATE_STATE_WORKER_PROMPT_V2_1)
        self.assertIn("candidate's own claims", GATE_STATE_WORKER_PROMPT_V2_1)
        self.assertIn("clear violation", GATE_STATE_WORKER_PROMPT_V2_1)
        self.assertIn("do not revise either one merely to create a contrast", GATE_STATE_WORKER_PROMPT_V2_1)
        self.assertNotIn("multimodal_alignment", GATE_STATE_WORKER_PROMPT_V2_1)
        self.assertNotEqual(GATE_STATE_WORKER_PROMPT_V2, GATE_STATE_WORKER_PROMPT_V2_1)

        row = {"sample_id": "s", "image_path": "unused.jpg", "question": "q",
               "A": "a", "B": "b", "answer": "A"}
        criterion = Criterion("c", "d", 0.5)

        class FakeAgent:
            def __init__(self, **_kwargs):
                self.last_call_metrics = AgentCallMetrics(api_attempts=1)

            def __call__(self, _prompt, stream=False):
                return '{"applicable":"yes","status_a":"pass","status_b":"fail"}'

        with patch("critiq.dual_evaluator.Agent", FakeAgent):
            evaluator = GateStateMultiModalEvaluator(
                worker_args={"request_kwargs": {"temperature": 0}},
                dataset=[row],
                backend_id="b",
                worker_prompt=GATE_STATE_WORKER_PROMPT_V2_1,
                encode_local_image=False,
            )
            output = evaluator.pred((criterion,))
        self.assertEqual(output.prompt_version, GATE_WORKER_PROMPT_V2_1_PILOT_VERSION)
        self.assertEqual(GatePredictionOutput.from_dict(output.to_dict()), output)

    def test_pairwise_prompt_is_exact_exp4_contract(self):
        self.assertEqual(PAIRWISE_MULTIMODAL_WORKER_PROMPT, demo_rlhfv.WORKER_PROMPT)
        self.assertEqual(PAIRWISE_WORKER_PROMPT_POSTFIX, ENGLISH_PROMPT.PAIR_WORKER_PROMPT_POSTFIX)
        self.assertEqual(
            worker_prompt_sha256(PAIRWISE_MULTIMODAL_WORKER_PROMPT, PAIRWISE_WORKER_PROMPT_POSTFIX),
            "198dcc0a3086456c14da4cb090676853448829465302edc7fcd2f37c8e39a2cb",
        )

    def test_legacy_answer_parser(self):
        self.assertIs(parse_pairwise_vote_response('{"answer":"A","thought":"t"}').vote, Vote.A)
        self.assertIs(parse_pairwise_vote_response('{"answer":"B","thought":"t"}').vote, Vote.B)
        self.assertIs(parse_pairwise_vote_response('{"answer":"None","thought":"t"}').vote, Vote.ABSTAIN)
        self.assertIs(parse_pairwise_vote_response('{"answer":"UNSURE","thought":"t"}').vote, Vote.ABSTAIN)
        with self.assertRaises(PairwiseVoteParseError):
            parse_pairwise_vote_response('{"answer":"A"}')
        invalid = parse_pairwise_vote_response('{"answer":"X","thought":"legacy invalid"}')
        self.assertFalse(invalid.answer_valid)
        self.assertEqual(invalid.thought, "legacy invalid")
        with self.assertRaises(PairwiseVoteParseError):
            parse_pairwise_vote_response(None)

    def test_pairwise_parser_recovers_unambiguous_bare_none_and_latex(self):
        bare = parse_pairwise_vote_response("None\nNot applicable to this pair.")
        self.assertIs(bare.vote, Vote.ABSTAIN)
        self.assertTrue(bare.answer_valid)
        self.assertEqual(bare.thought, "Not applicable to this pair.")

        latex = parse_pairwise_vote_response(
            '{"analysis_a":"\\angle A = 90^\\circ",'
            '"analysis_b":"b","thought":"use \\angle A",'
            '"answer":"A"}'
        )
        self.assertIs(latex.vote, Vote.A)
        self.assertEqual(latex.thought, r"use \angle A")

    def test_cached_pairwise_technical_failure_is_reparsed_without_new_vote(self):
        failed = PairwiseVoteOutput(
            Vote.ABSTAIN, False, "None", "invalid JSON", 2, None, False)
        recovered = recover_cached_pairwise_vote_output(failed)
        self.assertTrue(recovered.parse_ok)
        self.assertTrue(recovered.answer_valid)
        self.assertIs(recovered.vote, Vote.ABSTAIN)
        self.assertEqual(recovered.attempt_count, 2)

        malformed = PairwiseVoteOutput(
            Vote.ABSTAIN, False, "not a decision", "invalid JSON", 2,
            None, False)
        self.assertIs(recover_cached_pairwise_vote_output(malformed), malformed)

    def test_parse_failure_is_distinct_from_model_abstain(self):
        abstain = PairwiseVoteOutput(Vote.ABSTAIN, True, '{"answer":"None"}', None, 1, "t", True)
        failure = PairwiseVoteOutput(Vote.ABSTAIN, False, "bad", "invalid JSON", 2, None, False)
        self.assertTrue(abstain.is_model_abstain)
        self.assertFalse(failure.is_model_abstain)

    def test_gate_schema_and_consistency_are_independent_of_preference(self):
        judgement = parse_gate_state_response(
            '{"applicable":"yes","status_a":"pass","status_b":"fail"}'
        )
        self.assertTrue(judgement.consistency_ok)
        self.assertIs(judgement.status_b, CriterionStatus.FAIL)
        inconsistent = parse_gate_state_response(
            '{"applicable":"no","status_a":"pass","status_b":"uncertain"}'
        )
        self.assertFalse(inconsistent.consistency_ok)
        with self.assertRaises(GateStateParseError):
            parse_gate_state_response(
                '{"applicable":"yes","status_a":"pass","status_b":"pass","answer":"A"}'
            )

    def test_request_spec_is_deeply_immutable(self):
        source = {"nested": {"temperature": 0}, "stops": ["x"]}
        spec = DualWorkerRequestSpec("m", "backend", "0" * 64, None, False,
                                     "image_path", "question", "sample_id", source)
        source["nested"]["temperature"] = 1
        self.assertEqual(spec.to_dict()["decoding_config"]["nested"]["temperature"], 0)
        with self.assertRaises(TypeError):
            spec.decoding_config["nested"]["temperature"] = 2

    def test_pairwise_and_gate_retry_keep_failure_types_separate(self):
        responses = iter(["not json", '{"answer":"B","thought":"choose B"}',
                          '{"applicable":"no","status_a":"pass","status_b":"uncertain"}',
                          '{"applicable":"yes","status_a":"pass","status_b":"fail"}'])
        class FakeAgent:
            def __init__(self, **_kwargs): self.last_call_metrics = AgentCallMetrics(api_attempts=1)
            def __call__(self, _prompt, stream=False): return next(responses)
        row = {"sample_id": "s", "image_path": "unused.jpg", "question": "q", "A": "a", "B": "b", "answer": "A"}
        criterion = Criterion("c", "d", 0.5)
        with patch("critiq.dual_evaluator.Agent", FakeAgent):
            pair_eval = PairwiseVoteMultiModalEvaluator(worker_args={}, dataset=[row], backend_id="b",
                                                        max_retries=1, encode_local_image=False)
            pair_output, _ = pair_eval.infer_one(row, criterion)
            self.assertTrue(pair_output.parse_ok); self.assertIs(pair_output.vote, Vote.B)
            self.assertEqual(pair_output.attempt_count, 2)
            self.assertEqual(pair_output.thought, "choose B")
            gate_eval = GateStateMultiModalEvaluator(worker_args={"request_kwargs": {"temperature": 0}}, dataset=[row], backend_id="b",
                                                     max_retries=1, encode_local_image=False)
            gate_output, _ = gate_eval.infer_one(row, criterion)
            self.assertTrue(gate_output.valid); self.assertEqual(gate_output.attempt_count, 2)

    def test_legacy_invalid_answer_does_not_retry(self):
        responses = iter(['{"answer":"X","thought":"parsed but invalid"}',
                          '{"answer":"A","thought":"must not be consumed"}'])
        class FakeAgent:
            def __init__(self, **_kwargs): self.last_call_metrics = AgentCallMetrics(api_attempts=1)
            def __call__(self, _prompt, stream=False): return next(responses)
        row = {"sample_id": "s", "image_path": "unused.jpg", "question": "q",
               "A": "a", "B": "b", "answer": "A"}
        with patch("critiq.dual_evaluator.Agent", FakeAgent):
            evaluator = PairwiseVoteMultiModalEvaluator(worker_args={}, dataset=[row], backend_id="b",
                                                        max_retries=1, encode_local_image=False)
            output, _ = evaluator.infer_one(row, Criterion("c", "d", 0.5))
        self.assertTrue(output.parse_ok)
        self.assertFalse(output.answer_valid)
        self.assertFalse(output.is_model_abstain)
        self.assertEqual(output.attempt_count, 1)

    def test_pairwise_and_gate_cache_namespaces_are_independent(self):
        spec = DualWorkerRequestSpec("m", "backend", "0" * 64, None, False,
                                     "image_path", "question", "sample_id", {"temperature": 0})
        pair_key = pairwise_cache_key_payload(sample_fingerprint="1" * 64, criterion_name="c",
                                              criterion_description="d", request_spec=spec)
        gate_key = gate_cache_key_payload(sample_fingerprint="1" * 64, criterion_name="c",
                                          criterion_description="d", request_spec=spec)
        pair_output = PairwiseVoteOutput(Vote.A, True, "raw", None, 1, "t", True)
        gate_output = GateStateOutput(GateJudgement(Applicability.YES, CriterionStatus.PASS,
                                                    CriterionStatus.FAIL), "raw", None, 1)
        with TemporaryDirectory(dir=".") as directory:
            cache = JsonPredictionCache(directory)
            generation = ModelCallMetrics(logical_evaluations=1, api_attempts=1,
                                          input_tokens=10, output_tokens=2, total_tokens=12)
            cache.put_pairwise(pair_key, pair_output, generation)
            cache.put_gate(gate_key, gate_output, generation)
            self.assertEqual(cache.get_pairwise(pair_key).output, pair_output)
            self.assertEqual(cache.get_gate(gate_key).output, gate_output)
            cumulative = cache.cumulative_generation_metrics(("pairwise", "gate"))
            self.assertEqual(cumulative.logical_evaluations, 2)
            self.assertEqual(cumulative.total_tokens, 24)

    def test_versioned_pairwise_and_sparse_gate_artifacts_round_trip(self):
        spec = DualWorkerRequestSpec("m", "backend", "0" * 64, None, False,
                                     "image_path", "question", "sample_id", {"temperature": 0})
        criteria = (StructuredCriterionSnapshot("c1", "d1"),
                    StructuredCriterionSnapshot("c2", "d2"))
        row = {"c1": PairwiseVoteOutput(Vote.A, True, "a", None, 1, "ta", True),
               "c2": PairwiseVoteOutput(Vote.ABSTAIN, True, "n", None, 1, "tn", True)}
        pairwise = PairwisePredictionOutput(("s",), ("1" * 64,), criteria, (row,),
                                            (FinalPreference.A,), spec)
        self.assertEqual(PairwisePredictionOutput.from_dict(pairwise.to_dict()), pairwise)
        gate_output = GateStateOutput(GateJudgement(Applicability.YES, CriterionStatus.PASS,
                                                    CriterionStatus.PASS), "g", None, 1)
        gate_artifact = GatePredictionOutput(("s",), ("1" * 64,), (criteria[0],),
                                             ({"c1": gate_output},), spec)
        self.assertEqual(GatePredictionOutput.from_dict(gate_artifact.to_dict()), gate_artifact)

    def test_batch_progress_callback_fires_once_per_completed_sample(self):
        rows = [
            {"sample_id": "s1", "image_path": "unused.jpg", "question": "q1",
             "A": "a1", "B": "b1", "answer": "A"},
            {"sample_id": "s2", "image_path": "unused.jpg", "question": "q2",
             "A": "a2", "B": "b2", "answer": "B"},
        ]
        criteria = (Criterion("c1", "d1", 0.5), Criterion("c2", "d2", 0.5))
        evaluator = PairwiseVoteMultiModalEvaluator(
            worker_args={}, dataset=rows, backend_id="b", max_concurrent=4,
            encode_local_image=False,
        )

        def infer_one(_row, criterion):
            vote = Vote.A if criterion.name == "c1" else Vote.B
            return (
                PairwiseVoteOutput(vote, True, "raw", None, 1, "thought", True),
                ModelCallMetrics(logical_evaluations=1, api_attempts=1),
            )

        callbacks = []
        with patch.object(evaluator, "infer_one", side_effect=infer_one):
            prediction = evaluator.pred(
                criteria,
                on_sample_complete=lambda index, sample_id, metrics: callbacks.append(
                    (index, sample_id, metrics)
                ),
            )

        self.assertEqual(prediction.sample_ids, ("s1", "s2"))
        self.assertEqual(tuple(tuple(row) for row in prediction.node_outputs),
                         (("c1", "c2"), ("c1", "c2")))
        self.assertEqual({item[1] for item in callbacks}, {"s1", "s2"})
        self.assertEqual(len(callbacks), 2)
        self.assertTrue(all(item[2].logical_evaluations == 2 for item in callbacks))
        self.assertTrue(all(item[2].api_attempts == 2 for item in callbacks))


if __name__ == "__main__":
    unittest.main()

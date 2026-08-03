"""Tests for structured worker parsing, artifacts, and swap diagnostics."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from critiq.structured import (
    Applicability,
    CriterionStatus,
    FinalPreference,
    NodeJudgement,
    OutcomeReason,
    PairPreference,
    StructuredCriterionSnapshot,
    StructuredNodeOutput,
    StructuredWorkerRequestSpec,
    StructuredOutputParseError,
    StructuredPredictionOutput,
    is_ab_swap_consistent,
    make_parse_failure_judgement,
    parse_structured_worker_response,
    structured_worker_prompt_sha256,
    structured_input_fingerprint,
)


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "applicable": "yes",
        "status_a": "pass",
        "status_b": "fail",
        "pair_preference": "A",
        "evidence_a": "A is grounded.",
        "evidence_b": "B contradicts the image.",
    }
    payload.update(overrides)
    return payload


def parsed_output(
    *,
    preference: PairPreference = PairPreference.A,
) -> StructuredNodeOutput:
    judgement = NodeJudgement(
        applicable=Applicability.YES,
        status_a=CriterionStatus.PASS,
        status_b=CriterionStatus.FAIL,
        pair_preference=preference,
        evidence_a="a",
        evidence_b="b",
    )
    return StructuredNodeOutput(
        judgement=judgement,
        raw_response=json.dumps(valid_payload()),
        parse_error=None,
        attempt_count=1,
    )


class StructuredWorkerParserTest(unittest.TestCase):
    def test_parses_plain_fenced_and_surrounded_json(self) -> None:
        raw_json = json.dumps(valid_payload())
        variants = (
            raw_json,
            f"```json\n{raw_json}\n```",
            f"Here is the result:\n{raw_json}\nDone.",
        )
        for raw in variants:
            with self.subTest(raw=raw):
                judgement = parse_structured_worker_response(raw)
                self.assertTrue(judgement.parse_ok)
                self.assertTrue(judgement.consistency_ok)
                self.assertEqual(PairPreference.A, judgement.pair_preference)

    def test_parser_strips_values_but_does_not_change_case(self) -> None:
        stripped = parse_structured_worker_response(
            json.dumps(valid_payload(applicable=" yes ", evidence_a="  evidence  "))
        )
        self.assertEqual(Applicability.YES, stripped.applicable)
        self.assertEqual("evidence", stripped.evidence_a)

        with self.assertRaises(StructuredOutputParseError):
            parse_structured_worker_response(
                json.dumps(valid_payload(pair_preference="a"))
            )

    def test_rejects_missing_extra_non_string_and_non_response(self) -> None:
        missing = valid_payload()
        missing.pop("evidence_b")
        extra = valid_payload(thought="not allowed")
        non_string = valid_payload(status_a=1)
        for value in (missing, extra, non_string):
            with self.subTest(value=value), self.assertRaises(
                StructuredOutputParseError
            ):
                parse_structured_worker_response(json.dumps(value))
        for value in (None, 1, {"applicable": "yes"}):
            with self.subTest(value=value), self.assertRaises(
                StructuredOutputParseError
            ):
                parse_structured_worker_response(value)

    def test_schema_valid_consistency_conflict_is_preserved(self) -> None:
        judgement = parse_structured_worker_response(
            json.dumps(valid_payload(pair_preference="B"))
        )
        self.assertTrue(judgement.parse_ok)
        self.assertFalse(judgement.consistency_ok)
        self.assertIn("pass/fail", judgement.consistency_errors[0])

    def test_non_applicable_strict_uncertain_rule_is_semantic_validation(self) -> None:
        valid = parse_structured_worker_response(
            json.dumps(
                valid_payload(
                    applicable="no",
                    status_a="uncertain",
                    status_b="uncertain",
                    pair_preference="uncertain",
                )
            )
        )
        invalid = parse_structured_worker_response(
            json.dumps(
                valid_payload(
                    applicable="no",
                    status_a="pass",
                    status_b="uncertain",
                    pair_preference="uncertain",
                )
            )
        )
        self.assertTrue(valid.consistency_ok)
        self.assertFalse(invalid.consistency_ok)

    def test_both_pass_and_both_fail_allow_all_preferences(self) -> None:
        for status in ("pass", "fail"):
            for preference in ("A", "B", "tie", "uncertain"):
                with self.subTest(status=status, preference=preference):
                    judgement = parse_structured_worker_response(
                        json.dumps(
                            valid_payload(
                                status_a=status,
                                status_b=status,
                                pair_preference=preference,
                            )
                        )
                    )
                    self.assertTrue(judgement.consistency_ok)


class StructuredArtifactTest(unittest.TestCase):
    def make_artifact(self) -> StructuredPredictionOutput:
        return StructuredPredictionOutput(
            sample_ids=("sample-1",),
            sample_fingerprints=("a" * 64,),
            criteria=(
                StructuredCriterionSnapshot(
                    name="visual_grounding",
                    description="Prefer answers grounded in visible evidence.",
                ),
            ),
            node_outputs=({"visual_grounding": parsed_output()},),
            flat_answers=(FinalPreference.A,),
            request_spec=StructuredWorkerRequestSpec(
                model="test-model",
                worker_backend_id="fake-backend@revision-1",
                prompt_sha256="b" * 64,
                max_data_chars=None,
                encode_local_image=False,
                image_field="image_path",
                question_field="question",
                sample_id_field="sample_id",
                decoding_config={
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            ),
        )

    def test_node_output_derives_decision_and_round_trips(self) -> None:
        output = parsed_output()
        self.assertEqual(OutcomeReason.DECISIVE, output.local_decision.outcome_reason)
        self.assertEqual(output, StructuredNodeOutput.from_dict(output.to_dict()))

    def test_parse_failure_requires_error_and_round_trips(self) -> None:
        output = StructuredNodeOutput(
            judgement=make_parse_failure_judgement(),
            raw_response="not json",
            parse_error="failed to parse",
            attempt_count=3,
        )
        self.assertEqual(OutcomeReason.PARSE_FAILURE, output.local_decision.outcome_reason)
        self.assertEqual(output, StructuredNodeOutput.from_dict(output.to_dict()))
        with self.assertRaises(ValueError):
            StructuredNodeOutput(
                judgement=make_parse_failure_judgement(),
                raw_response=None,
                parse_error=None,
                attempt_count=1,
            )

    def test_prediction_json_round_trip(self) -> None:
        artifact = self.make_artifact()
        self.assertEqual(artifact, StructuredPredictionOutput.from_dict(artifact.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            artifact.save_json(path)
            self.assertEqual(artifact, StructuredPredictionOutput.load_json(path))

    def test_loader_rejects_version_and_matrix_corruption(self) -> None:
        serialized = self.make_artifact().to_dict()
        bad_version = copy.deepcopy(serialized)
        bad_version["parser_version"] = "2.0.0"
        missing_node = copy.deepcopy(serialized)
        missing_node["samples"][0]["node_outputs"] = {}
        extra_node = copy.deepcopy(serialized)
        extra_node["samples"][0]["node_outputs"]["extra"] = copy.deepcopy(
            extra_node["samples"][0]["node_outputs"]["visual_grounding"]
        )
        duplicate_sample = copy.deepcopy(serialized)
        duplicate_sample["samples"].append(copy.deepcopy(duplicate_sample["samples"][0]))
        for value in (bad_version, missing_node, extra_node, duplicate_sample):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StructuredPredictionOutput.from_dict(value)

    def test_fingerprint_is_order_sensitive_and_ignores_gold(self) -> None:
        sample = {
            "sample_id": "s1",
            "image_path": "image.jpg",
            "question": "What is shown?",
            "A": "cat",
            "B": "dog",
            "answer": "A",
        }
        fingerprint = structured_input_fingerprint(
            sample,
            image_field="image_path",
            question_field="question",
            sample_id_field="sample_id",
            encode_local_image=False,
        )
        changed_gold = {**sample, "answer": "B"}
        swapped = {**sample, "A": sample["B"], "B": sample["A"], "answer": "B"}
        self.assertEqual(
            fingerprint,
            structured_input_fingerprint(
                changed_gold,
                image_field="image_path",
                question_field="question",
                sample_id_field="sample_id",
                encode_local_image=False,
            ),
        )
        self.assertNotEqual(
            fingerprint,
            structured_input_fingerprint(
                swapped,
                image_field="image_path",
                question_field="question",
                sample_id_field="sample_id",
                encode_local_image=False,
            ),
        )


    def test_artifact_is_deeply_immutable(self) -> None:
        artifact = self.make_artifact()
        with self.assertRaises(TypeError):
            artifact.node_outputs[0]["other"] = parsed_output()
        with self.assertRaises((AttributeError, TypeError)):
            artifact.node_outputs[0].clear()
        with self.assertRaises(TypeError):
            artifact.decoding_config["temperature"] = 1
        with self.assertRaises(TypeError):
            artifact.decoding_config["response_format"]["type"] = "text"

    def test_request_identity_compatibility_and_prompt_hash(self) -> None:
        artifact = self.make_artifact()
        same = StructuredWorkerRequestSpec.from_dict(
            artifact.request_spec.to_dict()
        )
        artifact.assert_request_compatible(same)

        changed = same.to_dict()
        changed["worker_backend_id"] = "fake-backend@revision-2"
        with self.assertRaisesRegex(ValueError, "worker_backend_id"):
            artifact.assert_request_compatible(
                StructuredWorkerRequestSpec.from_dict(changed)
            )

        default_hash = structured_worker_prompt_sha256("prompt", "postfix")
        custom_hash = structured_worker_prompt_sha256("custom prompt", "postfix")
        self.assertNotEqual(default_hash, custom_hash)

    def test_fingerprint_uses_truncated_text_and_local_image_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            image_path.write_bytes(b"first-image")
            sample = {
                "sample_id": "s1",
                "image_path": str(image_path),
                "question": "What is shown?",
                "A": "abc-tail-one",
                "B": "xyz-tail-one",
                "answer": "A",
            }
            fingerprint = structured_input_fingerprint(
                sample,
                image_field="image_path",
                question_field="question",
                sample_id_field="sample_id",
                max_data_chars=3,
                encode_local_image=True,
            )
            same_sent_text = {
                **sample,
                "A": "abc-different-tail",
                "B": "xyz-different-tail",
            }
            self.assertEqual(
                fingerprint,
                structured_input_fingerprint(
                    same_sent_text,
                    image_field="image_path",
                    question_field="question",
                    sample_id_field="sample_id",
                    max_data_chars=3,
                    encode_local_image=True,
                ),
            )

            image_path.write_bytes(b"second-image")
            self.assertNotEqual(
                fingerprint,
                structured_input_fingerprint(
                    sample,
                    image_field="image_path",
                    question_field="question",
                    sample_id_field="sample_id",
                    max_data_chars=3,
                    encode_local_image=True,
                ),
            )


class SwapConsistencyTest(unittest.TestCase):
    def judgement(
        self,
        status_a: CriterionStatus,
        status_b: CriterionStatus,
        preference: PairPreference,
    ) -> NodeJudgement:
        return NodeJudgement(
            applicable=Applicability.YES,
            status_a=status_a,
            status_b=status_b,
            pair_preference=preference,
        )

    def test_decisive_tie_and_uncertain_swap_rules(self) -> None:
        cases = (
            (
                self.judgement(
                    CriterionStatus.PASS,
                    CriterionStatus.FAIL,
                    PairPreference.A,
                ),
                self.judgement(
                    CriterionStatus.FAIL,
                    CriterionStatus.PASS,
                    PairPreference.B,
                ),
            ),
            (
                self.judgement(
                    CriterionStatus.PASS,
                    CriterionStatus.PASS,
                    PairPreference.TIE,
                ),
                self.judgement(
                    CriterionStatus.PASS,
                    CriterionStatus.PASS,
                    PairPreference.TIE,
                ),
            ),
            (
                self.judgement(
                    CriterionStatus.UNCERTAIN,
                    CriterionStatus.UNCERTAIN,
                    PairPreference.UNCERTAIN,
                ),
                self.judgement(
                    CriterionStatus.UNCERTAIN,
                    CriterionStatus.UNCERTAIN,
                    PairPreference.UNCERTAIN,
                ),
            ),
        )
        for original, swapped in cases:
            with self.subTest(original=original):
                self.assertTrue(is_ab_swap_consistent(original, swapped))

    def test_invalid_or_non_equivariant_outputs_are_inconsistent(self) -> None:
        original = self.judgement(
            CriterionStatus.PASS,
            CriterionStatus.FAIL,
            PairPreference.A,
        )
        same_order = self.judgement(
            CriterionStatus.PASS,
            CriterionStatus.FAIL,
            PairPreference.A,
        )
        parse_failure = make_parse_failure_judgement()
        self.assertFalse(is_ab_swap_consistent(original, same_order))
        self.assertFalse(is_ab_swap_consistent(original, parse_failure))


if __name__ == "__main__":
    unittest.main()

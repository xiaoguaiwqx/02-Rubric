import unittest
from types import SimpleNamespace

from critiq.structured import FinalPreference, ModelCallMetrics, PairwiseVoteOutput, Vote
from experiments.evolving_structured_rubrics import vl_rewardbench as vlrb
from experiments.evolving_structured_rubrics import vl_rewardbench_phase10 as phase10
from experiments.evolving_structured_rubrics import vl_rewardbench_phase10_capped as capped
from experiments.evolving_structured_rubrics.experiment_utils import canonical_sha256
from experiments.evolving_structured_rubrics.rubric_factory import (
    build_multicrit_open_ended_init_rubric,
)


def _record(sample_id, preferred=0, group="general"):
    return {
        "sample_id": sample_id,
        "benchmark_id": sample_id.split("__", 1)[0],
        "question": "question",
        "responses": ["preferred", "rejected"],
        "preferred_original_index": preferred,
        "image_path": "image.jpg",
        "group": group,
    }


class VLRewardBenchProtocolTests(unittest.TestCase):

    def test_k3_schedule_is_deterministic_and_balanced(self):
        records = tuple(_record(f"sample_{index}") for index in range(5))
        first = vlrb._order_schedule(records)
        second = vlrb._order_schedule(tuple(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {(0, 1, 0), (1, 0, 1)})
        self.assertEqual(sum(order[0] for order in first.values()), 2)
        self.assertEqual(sum(1 - order[0] for order in first.values()), 3)

    def test_duplicate_benchmark_rows_have_distinct_internal_ids(self):
        records = (_record("mathverse_1649__row_0780"),
                   _record("mathverse_1649__row_0781"))
        schedule = vlrb._order_schedule(records)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(
            {record["benchmark_id"] for record in records}, {"mathverse_1649"})

    def test_ordered_rows_maps_gold_to_display_position(self):
        records = (_record("s0", preferred=0), _record("s1", preferred=1))
        schedule = {"s0": (0, 1, 0), "s1": (1, 0, 1)}
        rows = vlrb._ordered_rows(records, schedule, 0)
        self.assertEqual((rows[0]["A"], rows[0]["answer"]), ("preferred", "A"))
        self.assertEqual((rows[1]["A"], rows[1]["answer"]), ("rejected", "A"))

    def test_majority_and_native_parser_are_position_normalized(self):
        self.assertEqual(vlrb._majority([0, 1, 0]), 0)
        self.assertEqual(vlrb._majority([None, None, None]), None)
        self.assertEqual(vlrb._parse_native(
            "Overall Judgment: Answer 2 is slightly better"), 2)
        self.assertEqual(vlrb._original_index(2, 1), 0)

    def test_metrics_use_strict_accuracy_and_record_abstentions(self):
        records = (_record("g", 0, "general"), _record("h", 1, "hallucination"),
                   _record("r", 0, "reasoning"))
        metrics = vlrb._system_metrics(records, ([0, 1, None], [0, 1, None], [1, 1, None]))
        self.assertEqual(metrics["correct_count"], 2)
        self.assertEqual(metrics["coverage_count"], 2)
        self.assertAlmostEqual(metrics["strict_accuracy"], 2 / 3)
        self.assertEqual(metrics["majority_tie_or_abstain_count"], 1)


class VLRewardBenchPhase10ProtocolTests(unittest.TestCase):

    def test_dual_endpoint_route_is_deterministic(self):
        key = phase10._route_key(
            phase10.FINAL_SYSTEM, 1, "sample-1", "criterion-hash")
        self.assertEqual(phase10._route_endpoint(key), phase10._route_endpoint(key))
        self.assertIn(phase10._route_endpoint(key), phase10.ENDPOINT_IDS)

    def test_endpoint_schedule_covers_native_and_every_criterion(self):
        rubric = build_multicrit_open_ended_init_rubric()
        records = tuple(_record(f"sample-{index}") for index in range(9))
        schedule = phase10._endpoint_schedule(records, rubric)
        phase10._validate_endpoint_schedule(schedule, records, rubric)
        self.assertEqual(
            schedule["route_count"],
            len(records) * vlrb.K * (1 + len(rubric.nodes)),
        )
        self.assertEqual(set(schedule["endpoint_counts"]), set(phase10.ENDPOINT_IDS))

    def test_root_reuse_requires_exact_initial_criteria(self):
        rubric = build_multicrit_open_ended_init_rubric()
        reuse = phase10._root_reuse_contract(rubric, rubric)
        self.assertEqual(len(reuse), 5)
        self.assertEqual(set(reuse.values()), set(rubric.root_ids))

    def test_weighted_answers_follow_frozen_root_weights(self):
        roots = tuple(
            SimpleNamespace(
                root_id=root_id,
                selected=True,
                subtree_vote=(Vote.A if root_id == phase10.VISUAL_ROOT_ID else Vote.B),
            )
            for root_id in phase10.ROOT_WEIGHTS
        )
        execution = SimpleNamespace(traces=(SimpleNamespace(roots=roots),))
        self.assertEqual(
            phase10._weighted_answers(execution, phase10.ROOT_WEIGHTS),
            (FinalPreference.B,),
        )

    def test_phase10_metrics_include_six_source_groups(self):
        ids = (
            "hallucination-1", "mathverse-1", "RLAIF-V-1", "RLHF-V-1",
            "other-1", "wildvision-1",
        )
        broad_groups = (
            "hallucination", "reasoning", "hallucination", "hallucination",
            "general", "general",
        )
        records = tuple(
            _record(sample_id, group=group)
            for sample_id, group in zip(ids, broad_groups)
        )
        votes = tuple([0] * len(records) for _ in range(vlrb.K))
        metrics = phase10._system_metrics(records, votes)
        self.assertEqual(len(metrics["source_groups"]), 6)
        self.assertIn("source_macro_strict_accuracy", metrics)
        self.assertEqual(metrics["overall_acc"], metrics["covered_accuracy"])
        self.assertAlmostEqual(
            metrics["macro_acc"],
            sum(item["covered_accuracy"] for item in metrics["groups"].values()) / 3,
        )


class VLRewardBenchPhase10CappedProtocolTests(unittest.TestCase):

    @staticmethod
    def _spec(max_tokens=None):
        decoding = {"temperature": 0.5}
        if max_tokens is not None:
            decoding["max_tokens"] = max_tokens
        return {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "backend_id": "pool:test",
            "prompt_sha256": "a" * 64,
            "max_data_chars": None,
            "encode_local_image": True,
            "image_field": "image_path",
            "question_field": "question",
            "sample_id_field": "sample_id",
            "decoding_config": decoding,
        }

    @classmethod
    def _entry(cls, *, output_tokens=300, parse_retries=0, attempt_count=1):
        old_spec = cls._spec()
        raw = '{"analysis_a":"a","analysis_b":"b","thought":"ok","answer":"A"}'
        output = PairwiseVoteOutput(
            Vote.A, True, raw, None, attempt_count, "ok", True)
        metrics = ModelCallMetrics(
            logical_evaluations=1,
            api_attempts=1,
            parse_retries=parse_retries,
            input_tokens=500,
            output_tokens=output_tokens,
            total_tokens=500 + output_tokens,
            usage_complete=True,
        )
        payload = {
            "kind": "pairwise",
            "sample_fingerprint": "f" * 64,
            "criterion_name": "criterion",
            "criterion_description": "description",
            "request_spec": old_spec,
        }
        return {
            "cache_schema_version": "1.0.0",
            "kind": "pairwise",
            "key_sha256": canonical_sha256(payload),
            "key_payload": payload,
            "output": output.to_dict(),
            "generation_metrics": metrics.to_dict(),
        }, old_spec

    def test_capped_config_adds_only_2048_limit(self):
        config = {"worker_request_kwargs": {"temperature": 0.5}}
        value = capped._capped_config(config)
        self.assertEqual(value["worker_request_kwargs"], {
            "temperature": 0.5, "max_tokens": 2048})
        self.assertEqual(config["worker_request_kwargs"], {"temperature": 0.5})

    def test_request_specs_differ_only_by_nonbinding_cap(self):
        self.assertTrue(capped._compatible_specs(
            self._spec(), self._spec(capped.MAX_TOKENS)))
        self.assertFalse(capped._compatible_specs(
            self._spec(), self._spec(1024)))

    def test_single_valid_completion_below_cap_is_promotable(self):
        entry, old_spec = self._entry()
        output, metrics, reason = capped._safe_pairwise_entry(entry, old_spec)
        self.assertIsNone(reason)
        self.assertEqual(output.vote, Vote.A)
        self.assertEqual(metrics.output_tokens, 300)

    def test_retry_or_cap_binding_completion_is_not_promotable(self):
        retried, old_spec = self._entry(parse_retries=1)
        self.assertEqual(
            capped._safe_pairwise_entry(retried, old_spec)[2],
            "generation_not_single_clean_attempt",
        )
        capped_entry, old_spec = self._entry(output_tokens=capped.MAX_TOKENS)
        self.assertEqual(
            capped._safe_pairwise_entry(capped_entry, old_spec)[2],
            "cap_may_bind",
        )

    def test_latex_backslash_repair_is_conservative_and_parseable(self):
        raw = (
            '{"analysis_a":"Use \\pi and \\(x\\)",'
            '"analysis_b":"line\\nvalid",'
            '"thought":"A is grounded","answer":"A"}'
        )
        repaired = capped._safe_latex_json_repair(raw)
        self.assertIsNotNone(repaired)
        normalized, output = repaired
        self.assertEqual(output.vote, Vote.A)
        value = __import__("json").loads(normalized)
        self.assertEqual(value["analysis_a"], "Use \\pi and \\(x\\)")
        self.assertEqual(value["analysis_b"], "line\nvalid")

    def test_latex_repair_does_not_guess_unescaped_quotes(self):
        malformed = (
            '{"analysis_a":"book "Happy Halloween"",'
            '"analysis_b":"b","thought":"t","answer":"A"}'
        )
        self.assertIsNone(capped._safe_latex_json_repair(malformed))

    def test_constrained_repair_schema_is_strict(self):
        response_format = capped._pairwise_json_schema()
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {
            "analysis_a", "analysis_b", "thought", "answer"})
        self.assertEqual(schema["properties"]["answer"]["enum"], [
            "A", "B", "None"])

    def test_native_fallback_parser_extracts_only_frozen_choices(self):
        self.assertEqual(capped._parse_native_fallback(
            '{"choice":"Answer 1"}'), 1)
        self.assertEqual(capped._parse_native_fallback(
            '```json\n{"choice": "Answer 2"}\n```'), 2)
        self.assertIsNone(capped._parse_native_fallback(
            '{"choice":"Unclear"}'))
        self.assertIsNone(capped._parse_native_fallback(
            "I think the first response is nicer"))

    def test_native_fallback_prompt_forbids_rejudging(self):
        prompt = capped._native_fallback_prompt("raw judgement")
        self.assertIn("parser, not a judge", prompt)
        self.assertIn("Do not inspect or reassess", prompt)
        self.assertIn("raw judgement", prompt)

    def test_native_retry_request_key_is_attempt_specific(self):
        item = {"key": "replicate_02:17"}
        self.assertEqual(
            capped._native_retry_request_key(item, 1),
            "replicate_02:17::native_retry_01",
        )
        self.assertEqual(
            capped._native_retry_request_key(item, 10),
            "replicate_02:17::native_retry_10",
        )

    def test_native_retry_protocol_keeps_technical_retry_bounded(self):
        self.assertEqual(capped.NATIVE_RETRY_MAX_ATTEMPTS, 10)
        self.assertEqual(capped.EXPECTED_NATIVE_FALLBACK_UNRESOLVED, 83)
        self.assertIn("same-prompt", capped.NATIVE_RETRY_PROTOCOL)

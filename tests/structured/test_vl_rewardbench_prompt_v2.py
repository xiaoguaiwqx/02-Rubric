import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from critiq.structured import (
    DualWorkerRequestSpec,
    FinalPreference,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    Vote,
    aggregate_flat_votes,
)
from critiq.structured.worker_output import (
    StructuredCriterionSnapshot,
    structured_input_fingerprint,
)
from experiments.evolving_structured_rubrics import vl_rewardbench as legacy
from experiments.evolving_structured_rubrics import vl_rewardbench_prompt_v2 as prompt_v2
from experiments.evolving_structured_rubrics.rubric_factory import (
    build_multicrit_open_ended_init_rubric,
)


def _record(sample_id, preferred=0, group="general"):
    return {
        "sample_id": sample_id,
        "benchmark_id": sample_id,
        "question": "question",
        "responses": ["first", "second"],
        "preferred_original_index": preferred,
        "image_path": "image.jpg",
        "group": group,
    }


def _output(vote=Vote.A, *, valid=True):
    return PairwiseVoteOutput(
        vote if valid else Vote.ABSTAIN, valid, "raw",
        None if valid else "invalid", 1, "thought" if valid else None, valid)


def _prediction(rubric, sample_ids, vote=Vote.A, rows=None):
    criteria = tuple(
        StructuredCriterionSnapshot(
            rubric.get_node(node_id).criterion.name,
            rubric.get_node(node_id).criterion.description,
        )
        for node_id in rubric.preorder_node_ids()
    )
    spec = DualWorkerRequestSpec(
        "model", "pool", "a" * 64, None, True,
        "image_path", "question", "sample_id",
        {"temperature": 0.5, "max_tokens": 2048},
    )
    fingerprints = (tuple(structured_input_fingerprint(
        row, image_field="image_path", question_field="question",
        sample_id_field="sample_id", encode_local_image=True) for row in rows)
        if rows is not None else tuple("f" * 64 for _ in sample_ids))
    output_rows = tuple({item.name: _output(vote) for item in criteria}
                        for _ in sample_ids)
    return PairwisePredictionOutput(
        tuple(sample_ids), fingerprints, criteria, output_rows,
        tuple(aggregate_flat_votes(item.vote for item in row.values())
              for row in output_rows),
        spec, prompt_version=prompt_v2.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
    )


class VLRewardBenchPromptV2Tests(unittest.TestCase):

    def test_protocol_requires_frozen_prompt_v2_fields(self):
        expected = {
            "protocol_version": prompt_v2.PROTOCOL_VERSION,
            "source_experiment": prompt_v2.SOURCE_EXPERIMENT,
            "source_rubric_sha256": prompt_v2.SOURCE_RUBRIC_SHA256,
            "source_node_count": prompt_v2.SOURCE_NODE_COUNT,
            "source_v1_experiment": prompt_v2.SOURCE_V1_EXPERIMENT,
            "source_v1_logical": prompt_v2.SOURCE_V1_LOGICAL,
            "endpoint_ids": list(prompt_v2.ENDPOINT_IDS),
            "scheduler": "available_slot_dynamic",
            "prompt_version": prompt_v2.PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION,
            "smoke_sample_count": prompt_v2.SMOKE_COUNT,
            "max_retry_attempts": prompt_v2.MAX_RETRY_ATTEMPTS,
            "k": legacy.K,
            "seed": legacy.SEED,
            "selection_after_benchmark_forbidden": True,
        }
        config = {
            "vlrb_prompt_v2_transfer": expected,
            "worker_request_kwargs": {"temperature": 0.5, "max_tokens": 2048},
        }
        self.assertEqual(prompt_v2._protocol(config), expected)
        config["vlrb_prompt_v2_transfer"] = {**expected, "k": 2}
        with self.assertRaises(RuntimeError):
            prompt_v2._protocol(config)

    def test_failure_manifest_targets_only_technical_invalid_outputs(self):
        prediction = SimpleNamespace(
            sample_ids=("s0",),
            node_outputs=({"good": _output(), "bad": _output(valid=False)},),
        )
        items = prompt_v2._failure_items((prediction,))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["criterion_name"], "bad")
        self.assertEqual(items[0]["sample_id"], "s0")

    def test_retry_overlay_recomputes_flat_answer_without_overwriting_source(self):
        rubric = build_multicrit_open_ended_init_rubric()
        source = _prediction(rubric, ("s0",), Vote.A)
        criterion = source.criteria[0].name
        replaced = prompt_v2._replace_prediction_outputs(
            source, {(0, criterion): _output(Vote.B)})
        self.assertEqual(source.node_outputs[0][criterion].vote, Vote.A)
        self.assertEqual(replaced.node_outputs[0][criterion].vote, Vote.B)
        self.assertEqual(replaced.prompt_version, source.prompt_version)

    def test_four_systems_are_derived_from_one_prediction_set(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image")
            rubric = build_multicrit_open_ended_init_rubric()
            records = (
                {**_record("s0", 0, "general"), "image_path": str(image)},
                {**_record("s1", 1, "hallucination"), "image_path": str(image)},
                {**_record("s2", 0, "reasoning"), "image_path": str(image)},
            )
            schedule = legacy._order_schedule(records)
            predictions = tuple(
                _prediction(
                    rubric, tuple(row["sample_id"] for row in records), Vote.A,
                    legacy._ordered_rows(records, schedule, replicate))
                for replicate in range(legacy.K)
            )
            logical = prompt_v2._logical_from_predictions(
                records, schedule, rubric, predictions)
        self.assertEqual(
            set(logical["systems"]),
            {prompt_v2.INITIAL_SYSTEM, prompt_v2.EQUAL_SYSTEM,
             prompt_v2.WEIGHTED_SYSTEM, prompt_v2.VISUAL_SYSTEM},
        )
        self.assertTrue(logical["prediction_reuse"][
            "all_four_systems_share_prompt_v2_predictions"])

    def test_position_metrics_separate_display_slots(self):
        records = (_record("s0", 0), _record("s1", 1))
        schedule = {"s0": (0, 1, 0), "s1": (1, 0, 1)}
        votes = ([0, 1], [0, 1], [0, 1])
        value = prompt_v2._position_metrics(records, schedule, votes)
        self.assertEqual(sum(value["display_prediction_counts"].values()), 6)
        self.assertEqual(value["by_gold_display"]["A"]["total"], 4)
        self.assertEqual(value["by_gold_display"]["B"]["total"], 2)
        self.assertEqual(value["all_three_valid_count"], 2)

    def test_smoke_metrics_allow_missing_official_groups(self):
        records = (_record("s0", 0, "general"), _record("s1", 1, "general"))
        value = prompt_v2._smoke_system_metrics(
            records, ([0, 1], [0, 1], [0, 1]))
        self.assertEqual(value["sample_count"], 2)
        self.assertEqual(value["strict_accuracy"], 1.0)
        self.assertEqual(value["coverage"], 1.0)

    def test_unknown_stage_fails_explicitly(self):
        with self.assertRaises(ValueError):
            prompt_v2.run_stage({}, None, "not-a-stage")


if __name__ == "__main__":
    unittest.main()

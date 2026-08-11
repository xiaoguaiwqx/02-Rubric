import unittest

from experiments.evolving_structured_rubrics import vl_rewardbench as vlrb


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

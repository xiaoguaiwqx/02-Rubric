"""Offline checks for the frozen generated-root reporting slices."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.evolving_structured_rubrics.experiment_utils import atomic_write_json, load_json

from experiments.evolving_structured_rubrics import generated_roots_report as report


class TestGeneratedRootsReport(unittest.TestCase):
    @staticmethod
    def fixture():
        records = []
        for group, count in (("general", 181), ("hallucination", 749),
                             ("reasoning", 317)):
            records.extend(dict(sample_id=f"{group}-{index}", group=group,
                                preferred_original_index=index % 2)
                           for index in range(count))
        train = [f"hallucination-{index}" for index in range(100)]
        heldout = [f"hallucination-{index}" for index in range(100, 748)]
        return records, dict(train_ids=train, heldout_ids=heldout)

    def test_frozen_slices_keep_near_duplicate_out_of_clean_set(self):
        records, split = self.fixture()
        slices = report.subsets(records, split)
        self.assertEqual({name: len(ids) for name, ids in slices.items()}, {
            "full": 1247, "train": 100, "nontrain_1147": 1147,
            "clean_1146": 1146, "heldout_hallucination": 648,
            "general": 181, "hallucination": 749, "reasoning": 317,
        })
        self.assertIn("hallucination-748", slices["nontrain_1147"])
        self.assertNotIn("hallucination-748", slices["clean_1146"])

    def test_subset_and_paired_metrics_preserve_old_keys_and_denominators(self):
        records, split = self.fixture()
        ids = report.subsets(records, split)["heldout_hallucination"]
        baseline = [None] * len(records)
        final = [None] * len(records)
        target = next(index for index, row in enumerate(records)
                      if row["sample_id"] == "hallucination-100")
        final[target] = records[target]["preferred_original_index"]
        value = report.subset_metrics(records, final, ids)
        self.assertEqual(value, dict(correct=1, total=648, strict_acc=1 / 648,
                                     covered=1, coverage=1 / 648, covered_acc=1.0))
        paired = report.paired_metrics(records, baseline, final, ids)
        self.assertEqual((paired["corrected"], paired["harmed"]), (1, 0))
        self.assertEqual(paired["corrected_ids"], ["hallucination-100"])
        self.assertEqual(paired["mcnemar_exact_two_sided_p"], 1.0)
        bootstrap = report.paired_bootstrap_ci(records, baseline, final, ids,
                                               iterations=100)
        self.assertEqual(bootstrap["estimate"], 1 / 648)
        self.assertEqual(bootstrap["seed"], 20260925)

    def test_root_vote_uses_original_index_after_ab_swap(self):
        sample = {"replicates": {
            "0": {"order": 0, "subtrees": {"r": {"parse_ok": True,
                                                     "parsed": {"answer": "A"}}}},
            "1": {"order": 1, "subtrees": {"r": {"parse_ok": True,
                                                     "parsed": {"answer": "B"}}}},
            "2": {"order": 0, "subtrees": {"r": {"parse_ok": True,
                                                     "parsed": {"answer": "None"}}}},
        }}
        self.assertEqual(report._root_predictions({"k": 3, "samples": [sample]}, "r"), [0])

    def test_combined_report_waits_for_all_variants_and_compares_final(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = dict(sample_id="s1", preferred_original_index=0)
            slices = {"full": {"s1"}, "heldout_hallucination": {"s1"}}
            payload = dict(root_count=5, root_catalog=[], root_generation=None,
                           systems={}, roots={}, costs={})
            for variant in ("f5", "g5"):
                atomic_write_json(root / variant / "report.json",
                                  {"generated_roots": payload})
            report.combined_report(root, [sample], slices)
            self.assertFalse((root / "report.json").exists())
            atomic_write_json(root / "gn/report.json", {"generated_roots": payload})
            atomic_write_json(root / "warmup/transcript.json",
                              {"request": {"samples": [dict(sample_id="s1",
                                                           source="povid", flipped=False)]}})
            predicted = {("f5", "r0"): None, ("f5", "s0"): 0,
                         ("f5", "final"): 0, ("g5", "r0"): None,
                         ("g5", "s0"): 1, ("g5", "final"): 0,
                         ("gn", "r0"): None, ("gn", "s0"): 1,
                         ("gn", "final"): 1}

            def validated(path, rubric, records, orders):
                stage = {"r0": "r0", "initial": "s0", "final": "final"}[path.stem]
                return {}, [predicted[path.parent.parent.name, stage]]

            with patch.object(report, "_rubrics", return_value=dict(r0=None, s0=None,
                                                                      final=None)), \
                    patch.object(report, "_validated_system", side_effect=validated):
                report.combined_report(root, [sample], slices)
            result = load_json(root / "report.json")
            self.assertEqual(result["systems"].keys(), {"f5", "g5", "gn"})
            self.assertEqual(result["paired"]["gn_final_vs_f5_final"]["full"]["harmed"], 1)
            self.assertEqual(result["paired"]["g5_final_vs_f5_final"]["full"]["corrected"], 0)
            self.assertIn("g5_final_vs_r0", result["paired"])


if __name__ == "__main__":
    unittest.main()

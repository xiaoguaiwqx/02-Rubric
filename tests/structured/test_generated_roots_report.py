"""Offline checks for the frozen generated-root reporting slices."""

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.evolving_structured_rubrics.experiment_utils import atomic_write_json, load_json

from experiments.evolving_structured_rubrics import generated_roots_report as report
from experiments.evolving_structured_rubrics import rubric_pipeline as pipeline
from tests.structured.core_fixtures import artifact, rows


class TestGeneratedRootsReport(unittest.TestCase):
    def test_initialization_report_compares_formal_votes_without_final(self):
        data = rows(6)
        prefixes = ("vlfeedback", "RLHF", "mathverse")
        groups = ("general", "hallucination", "reasoning")
        records = [dict(sample_id=row["sample_id"], benchmark_id=f"{prefixes[i % 3]}_{i}",
                        group=groups[i % 3], question=row["question"], image_path=row["image_path"],
                        responses=[row["A"], row["B"]], preferred_original_index=0)
                   for i, row in enumerate(data)]
        orders = report.vlrb_official._order_schedule(records)
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        old = pipeline.replace_groups(r0, {root: [
            dict(name="one", description="first"), dict(name="two", description="second")]
            for root in r0.root_ids})
        new = pipeline.replace_groups(old, {old.root_ids[0]: [
            dict(name="one", description="new first"), dict(name="two", description="new second")]})

        def full(rubric, correct):
            value = artifact(data, rubric, correct, k=3)
            for sample in value["samples"]:
                sample["orders"] = list(orders[sample["sample_id"]])
                for index, order in enumerate(sample["orders"]):
                    replicate = sample["replicates"][str(index)]
                    replicate["order"] = order
                    if order:
                        for call in [*replicate["subtrees"].values(), replicate["arbiter"]]:
                            answer = call["parsed"]["answer"]
                            call["parsed"]["answer"] = "B" if answer == "A" else "A"
            value["metrics"] = pipeline.system.metrics(value, data)
            return value

        values = dict(r0=full(r0, 3), old_s0=full(old, 4), new_s0=full(new, 5))
        for rep in ("1", "2"):
            values["new_s0"]["samples"][0]["replicates"][rep]["arbiter"]["parsed"]["answer"] = "None"
        values["new_s0"]["metrics"] = pipeline.system.metrics(values["new_s0"], data)
        ids = {row["sample_id"] for row in records}
        train = {data[0]["sample_id"]}
        slices = dict(full=ids, train=train, nontrain_1147=ids - train,
                      clean_1146=ids - train, heldout_hallucination={data[1]["sample_id"],
                                                                  data[4]["sample_id"]})
        with TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "target"
            for directory, rubric, label in ((source, old, "old_s0"), (target, new, "new_s0")):
                atomic_write_json(directory / "r0/rubric.json", r0.to_dict())
                atomic_write_json(directory / "init/rubric.json", rubric.to_dict())
                atomic_write_json(directory / "init/system.json", artifact(data[:1], rubric, 1))
                atomic_write_json(directory / "vlrb/initial.json", values[label])
            atomic_write_json(source / "vlrb/r0.json", values["r0"])
            atomic_write_json(target / "init/summary.json", dict(roots=[], wall_seconds=1))
            with patch.object(report.vlrb_official, "_read_records", return_value=records), \
                    patch.object(report, "subsets", return_value=slices), \
                    patch.object(pipeline, "load_rows", return_value=data[:1]):
                result = report.init_report(dict(data_root=".", datasets={"vlrb": "test.parquet"}),
                                            target, source)
            self.assertEqual(result["systems"]["new_s0"]["full"]["correct"], 4)
            self.assertEqual(result["systems"]["old_s0"]["full"]["correct"], 4)
            paired = result["paired"]["old_s0_to_new_s0"]["full"]
            self.assertEqual((paired["corrected"], paired["harmed"], paired["net_corrected"]),
                             (1, 1, 0))
            self.assertEqual(result["official"]["new_s0"]["majority_tie_or_abstain_count"], 1)
            self.assertIn("ci95", result["paired"]["old_s0_to_new_s0"]["clean_1146"]["bootstrap"])
            self.assertTrue((target / "init_comparison.json").exists())
            self.assertFalse((target / "final.json").exists())
            self.assertFalse((target / "state.json").exists())

    def test_artifact_cost_deduplicates_reused_calls(self):
        data = rows(1)
        rubric = pipeline.build_multicrit_open_ended_init_rubric()
        value = artifact(data, rubric, 1)
        replica = value["samples"][0]["replicates"]["0"]
        for i, call in enumerate([*replica["subtrees"].values(), replica["arbiter"]]):
            call.update(cache_key=f"key-{i}", model_generation_count=1,
                        metrics=dict(api_attempts=2, input_tokens=3, output_tokens=4,
                                     latency_seconds=5, usage_complete=True))
        cost = report.artifact_worker_cost([value, deepcopy(value)])
        self.assertEqual(cost["subtree"]["logical_calls"], 5)
        self.assertEqual(cost["subtree"]["input_tokens"], 15)
        self.assertEqual(cost["arbiter"]["api_attempts"], 2)
        self.assertEqual(cost["arbiter"]["missing_usage_calls"], 0)

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

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
    @staticmethod
    def small_fixture():
        data = rows(6)
        prefixes = ("vlfeedback", "RLHF", "mathverse")
        groups = ("general", "hallucination", "reasoning")
        records = [dict(sample_id=row["sample_id"], benchmark_id=f"{prefixes[i % 3]}_{i}",
                        group=groups[i % 3], question=row["question"], image_path=row["image_path"],
                        responses=[row["A"], row["B"]], preferred_original_index=0)
                   for i, row in enumerate(data)]
        ids = {row["sample_id"] for row in records}
        train = {data[0]["sample_id"]}
        slices = dict(full=ids, train=train, nontrain_1147=ids - train,
                      clean_1146=ids - train, heldout_hallucination={data[1]["sample_id"],
                                                                  data[4]["sample_id"]})
        return data, records, slices

    @staticmethod
    def scheduled_artifact(data, rubric, correct, records):
        value = artifact(data, rubric, correct, k=3)
        value["order_protocol"] = report.vlrb_official.ORDER_PROTOCOL
        orders = report.vlrb_official._order_schedule(records)
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

    def test_report_entry_inherits_r0_but_requires_local_s0_and_final(self):
        data, records, slices = self.small_fixture()
        config = dict(data_root=".", datasets={"vlrb": "test.parquet"})
        with TemporaryDirectory() as temporary:
            origin, source, root = (Path(temporary) / name for name in ("v1", "v2", "v3"))
            target = root / "g5"
            atomic_write_json(source / "g5/source.json", dict(source_run=str(origin / "g5")))
            atomic_write_json(target / "source.json", dict(source_run=str(source / "g5")))
            atomic_write_json(origin / "g5/vlrb/r0.json", {})
            for name in ("initial", "final"):
                atomic_write_json(origin / f"g5/vlrb/{name}.json", {})
                atomic_write_json(target / f"vlrb/{name}.json", {})
            split = Path(temporary) / "split.json"
            atomic_write_json(split, dict(train_ids=[data[0]["sample_id"]]))
            with patch.object(report, "SPLIT_MANIFEST", split), \
                    patch.object(pipeline, "load_rows", return_value=data[:1]), \
                    patch.object(report.vlrb_official, "_read_records", return_value=records), \
                    patch.object(report, "subsets", return_value=slices), \
                    patch.object(report, "variant_report") as variant, \
                    patch.object(report, "combined_report") as combined:
                report.report(config, root, "g5")
                variant.assert_called_once_with(config, root, "g5", records, slices)
                combined.assert_called_once_with(root, records, slices)
                for name in ("initial", "final"):
                    with self.subTest(missing_stage=name):
                        path = target / f"vlrb/{name}.json"
                        path.unlink()
                        variant.reset_mock()
                        combined.reset_mock()
                        with self.assertRaises(RuntimeError) as error:
                            report.report(config, root, "g5")
                        self.assertIn(repr(str(path)), str(error.exception))
                        variant.assert_not_called()
                        combined.assert_not_called()
                        atomic_write_json(path, {})
            self.assertFalse((target / "vlrb/r0.json").exists())

    def test_report_reads_untagged_historical_swaps_and_tagged_random_swaps(self):
        data, records, _ = self.small_fixture()
        rubric = pipeline.build_multicrit_open_ended_init_rubric()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.json"
            for protocol in (report.vlrb_official.LEGACY_ORDER_PROTOCOL,
                             report.vlrb_official.ORDER_PROTOCOL):
                with self.subTest(protocol=protocol):
                    value = artifact(data, rubric, len(data), k=3)
                    orders = report.vlrb_official._order_schedule(records, protocol=protocol)
                    for sample in value["samples"]:
                        sample["orders"] = list(orders[sample["sample_id"]])
                        for index, order in enumerate(sample["orders"]):
                            replicate = sample["replicates"][str(index)]
                            replicate["order"] = order
                            replicate["arbiter"]["parsed"]["answer"] = "B" if order else "A"
                    value["metrics"] = pipeline.system.metrics(value, data)
                    if protocol == report.vlrb_official.ORDER_PROTOCOL:
                        value["order_protocol"] = protocol
                        value["order_seed"] = report.vlrb_official.SEED
                    atomic_write_json(path, value)
                    saved_protocol, saved_orders = report._report_orders(records, path)
                    self.assertEqual(saved_protocol, protocol)
                    _, predictions = report._validated_system(path, rubric, records, saved_orders)
                    self.assertEqual(predictions, [0] * len(data))
                    other = (report.vlrb_official.ORDER_PROTOCOL
                             if protocol == report.vlrb_official.LEGACY_ORDER_PROTOCOL
                             else report.vlrb_official.LEGACY_ORDER_PROTOCOL)
                    with self.assertRaisesRegex(RuntimeError, "A/B schedule differs"):
                        report._validated_system(path, rubric, records,
                            report.vlrb_official._order_schedule(records, protocol=other))

    def test_variant_report_reuses_origin_r0_and_generation_without_copying(self):
        data, records, slices = self.small_fixture()
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        s0 = pipeline.replace_groups(r0, {root: [
            dict(name="one", description="first"), dict(name="two", description="second")]
            for root in r0.root_ids})
        final = pipeline.replace_groups(s0, {s0.root_ids[0]: [
            dict(name="one", description="final first"), dict(name="two", description="second")]})
        with TemporaryDirectory() as temporary:
            origin, source, root = (Path(temporary) / name for name in ("v1", "v2", "v3"))
            target = root / "g5"
            atomic_write_json(source / "g5/source.json", dict(source_run=str(origin / "g5")))
            atomic_write_json(target / "source.json", dict(source_run=str(source / "g5")))
            atomic_write_json(origin / "g5/vlrb/r0.json",
                              self.scheduled_artifact(data, r0, 2, records))
            atomic_write_json(target / "r0/rubric.json", r0.to_dict())
            atomic_write_json(target / "r0/system.json", artifact(data[:1], r0, 1))
            for rubric, name, training, correct in ((s0, "initial", "init", 4),
                                                   (final, "final", "e01", 5)):
                rubric_path = "init/rubric.json" if name == "initial" else "final.json"
                atomic_write_json(target / rubric_path, rubric.to_dict())
                atomic_write_json(target / f"{training}/system.json", artifact(data[:1], rubric, 1))
                atomic_write_json(target / f"vlrb/{name}.json",
                                  self.scheduled_artifact(data, rubric, correct, records))
            atomic_write_json(target / "state.json", dict(epoch=1, baseline="e01/system"))
            atomic_write_json(target / "e01/summary.json", dict(wall_seconds=2))
            atomic_write_json(target / "report.json", dict(existing_section={"kept": True}))
            generation = dict(parsed={"count_reason": "origin count"},
                              request={"warmup_history_sha256": "origin-warmup"},
                              attempts=[dict(metrics=dict(api_attempts=1, input_tokens=30,
                                                          output_tokens=31, latency_seconds=3))])
            warmup = dict(turns=[dict(response="warmup", attempts=[dict(metrics=dict(
                api_attempts=1, input_tokens=20, output_tokens=21, latency_seconds=2))])])
            atomic_write_json(origin / "g5/r0/generation.json", generation)
            atomic_write_json(origin / "warmup/transcript.json", warmup)
            for directory, tokens in ((target, 7), (origin / "g5", 700)):
                atomic_write_json(directory / "cache/init/subtree.json", dict(
                    cache_key="local-subtree", model_generation_count=1,
                    request={"request_key": {"kind": "aligned_unified_subtree"}},
                    metrics=dict(api_attempts=1, input_tokens=tokens, output_tokens=8,
                                 latency_seconds=1),
                    attempt_metrics=[dict(usage_complete=True, input_tokens=tokens, output_tokens=8)]))
                atomic_write_json(directory / "init/manager.json", dict(
                    request={}, attempts=[dict(wall_seconds=2,
                                              usage=dict(prompt_tokens=tokens, completion_tokens=12))]))
            source_files = {path: path.read_bytes() for directory in (origin, source)
                            for path in directory.rglob("*.json")}
            with patch.object(pipeline, "load_rows", return_value=data[:1]), \
                    patch.object(report, "_validated_system", wraps=report._validated_system) as validated:
                report.variant_report({}, root, "g5", records, slices)
            self.assertEqual([call.args[0] for call in validated.call_args_list], [
                origin / "g5/vlrb/r0.json", target / "vlrb/initial.json", target / "vlrb/final.json"])
            result = load_json(target / "report.json")
            self.assertEqual(result["existing_section"], {"kept": True})
            generated = result["generated_roots"]
            self.assertEqual({stage: generated["systems"][stage]["full"]["correct"]
                              for stage in ("r0", "s0", "final")}, {"r0": 2, "s0": 4, "final": 5})
            self.assertEqual(generated["root_count"], 5)
            self.assertEqual(generated["root_generation"], dict(
                count_reason="origin count", warmup_history_sha256="origin-warmup", attempt_count=1))
            costs = generated["costs"]
            self.assertTrue(costs["historical_cost_reused"])
            self.assertEqual(costs["reused_artifacts"], {
                "vlrb/r0.json": str(origin / "g5/vlrb/r0.json"),
                "r0/generation.json": str(origin / "g5/r0/generation.json"),
                "../warmup/transcript.json": str(origin / "warmup/transcript.json"),
            })
            self.assertEqual(costs["root_generation"]["shared_warmup"]["input_tokens"], 20)
            self.assertEqual(costs["root_generation"]["root_generation"]["input_tokens"], 30)
            self.assertEqual(costs["manager"]["input_tokens"], 7)
            self.assertEqual(costs["worker"]["subtree"]["input_tokens"], 7)
            self.assertEqual(costs["source_output"], str(target))
            for path, content in source_files.items():
                self.assertEqual(path.read_bytes(), content)
            for path in (target / "vlrb/r0.json", target / "r0/generation.json",
                         root / "warmup/transcript.json"):
                self.assertFalse(path.exists())
            atomic_write_json(target / "vlrb/r0.json",
                              self.scheduled_artifact(data, r0, 3, records))
            local_generation = deepcopy(generation)
            local_generation["parsed"]["count_reason"] = "local count"
            atomic_write_json(target / "r0/generation.json", local_generation)
            atomic_write_json(root / "warmup/transcript.json", warmup)
            with patch.object(pipeline, "load_rows", return_value=data[:1]):
                report.variant_report({}, root, "g5", records, slices)
            local = load_json(target / "report.json")["generated_roots"]
            self.assertEqual(local["systems"]["r0"]["full"]["correct"], 3)
            self.assertEqual(local["root_generation"]["count_reason"], "local count")
            self.assertFalse(local["costs"]["historical_cost_reused"])
            self.assertEqual(local["costs"]["reused_artifacts"], {})

    def test_r0_artifact_prefers_local_over_recorded_source(self):
        with TemporaryDirectory() as temporary:
            source, target = (Path(temporary) / name / "g5" for name in ("source", "target"))
            atomic_write_json(target / "source.json", dict(source_run=str(source)))
            for relative in ("vlrb/r0.json", "r0/generation.json", "../warmup/transcript.json"):
                with self.subTest(artifact=relative):
                    atomic_write_json(source / relative, {"value": "source"})
                    atomic_write_json(target / relative, {"value": "local"})
                    self.assertEqual(report._r0_artifact(target, relative), (target / relative).resolve())

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
            value["order_protocol"] = report.vlrb_official.ORDER_PROTOCOL
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

    def test_initialization_report_inherits_r0_without_changing_s0_baseline(self):
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
            value["order_protocol"] = report.vlrb_official.ORDER_PROTOCOL
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

        ids = {row["sample_id"] for row in records}
        train = {data[0]["sample_id"]}
        slices = dict(full=ids, train=train, nontrain_1147=ids - train,
                      clean_1146=ids - train, heldout_hallucination={data[1]["sample_id"],
                                                                  data[4]["sample_id"]})
        with TemporaryDirectory() as temp:
            origin, source, target = (Path(temp) / name for name in ("origin", "source", "target"))
            atomic_write_json(origin / "vlrb/r0.json", full(r0, 2))
            atomic_write_json(source / "source.json", dict(source_run=str(origin)))
            atomic_write_json(target / "source.json", dict(source_run=str(source), source_files={}))
            for directory, rubric, correct in ((source, old, 4), (target, new, 5)):
                atomic_write_json(directory / "r0/rubric.json", r0.to_dict())
                atomic_write_json(directory / "init/rubric.json", rubric.to_dict())
                atomic_write_json(directory / "init/system.json", artifact(data[:1], rubric, 1))
                atomic_write_json(directory / "vlrb/initial.json", full(rubric, correct))
            atomic_write_json(target / "init/summary.json", dict(roots=[], wall_seconds=1))
            self.assertFalse((source / "vlrb/r0.json").exists())
            with patch.object(report.vlrb_official, "_read_records", return_value=records), \
                    patch.object(report, "subsets", return_value=slices), \
                    patch.object(pipeline, "load_rows", return_value=data[:1]), \
                    patch.object(report, "_validated_system", wraps=report._validated_system) as validated:
                result = report.init_report(dict(data_root=".", datasets={"vlrb": "test.parquet"}),
                                            target, source)
            self.assertEqual([call.args[0] for call in validated.call_args_list], [
                origin / "vlrb/r0.json", source / "vlrb/initial.json", target / "vlrb/initial.json"])
            self.assertEqual(result["source_run"], str(source))
            self.assertEqual(result["systems"]["r0"]["full"]["correct"], 2)
            self.assertEqual(result["systems"]["old_s0"]["full"]["correct"], 4)
            paired = result["paired"]["old_s0_to_new_s0"]["full"]
            self.assertEqual((paired["corrected"], paired["harmed"], paired["net_corrected"]),
                             (1, 0, 1))

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
            root, source, origin = (Path(temporary) / name for name in ("v3", "v2", "v1"))
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
            for variant in ("f5", "g5", "gn"):
                atomic_write_json(root / variant / "source.json",
                                  dict(source_run=str(source / variant)))
                atomic_write_json(source / variant / "source.json",
                                  dict(source_run=str(origin / variant)))
                atomic_write_json(origin / variant / "vlrb/r0.json", {})
                atomic_write_json(root / variant / "vlrb/initial.json", {})
            atomic_write_json(origin / "warmup/transcript.json",
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
                    patch.object(report, "_validated_system", side_effect=validated) as validation:
                report.combined_report(root, [sample], slices)
            self.assertEqual([call.args[0] for call in validation.call_args_list], [
                directory / variant / f"vlrb/{name}.json"
                for variant in ("f5", "g5", "gn")
                for directory, name in ((origin, "r0"), (root, "initial"), (root, "final"))])
            result = load_json(root / "report.json")
            self.assertEqual(result["systems"].keys(), {"f5", "g5", "gn"})
            self.assertEqual(result["paired"]["gn_final_vs_f5_final"]["full"]["harmed"], 1)
            self.assertEqual(result["paired"]["g5_final_vs_f5_final"]["full"]["corrected"], 0)
            self.assertIn("g5_final_vs_r0", result["paired"])


if __name__ == "__main__":
    unittest.main()

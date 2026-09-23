"""Offline checks for local acceptance and isolated case feedback."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.evolving_structured_rubrics import subtree_local_reflection as local
from experiments.evolving_structured_rubrics.subtree_local_reflection_manager import validate_local
from tests.structured.test_framework_v6 import rows, report, artifact


def root_value(data, root, answers):
    return dict(root_id=root, samples=[dict(sample_id=row["sample_id"], order=0,
        call=dict(parse_ok=True, parsed=report(answer))) for row, answer in zip(data, answers)])


class TestLocalReflection(unittest.TestCase):
    def test_preservation_sampling_is_correct_unique_and_reproducible(self):
        data = rows(12)
        reports = root_value(data, "r", ["A"] * 10 + ["B", "None"])
        chosen = local.preservation_rows(data, reports, 1, "r", 5, 42)
        self.assertEqual(chosen, local.preservation_rows(data, reports, 1, "r", 5, 42))
        ids = {r["sample_id"] for r in chosen}
        self.assertEqual(len(ids), 5)
        self.assertTrue(ids <= {r["sample_id"] for r in data[:10]})
        self.assertEqual(len(local.preservation_rows(data, reports, 1, "r", 20, 42)), 10)
        self.assertEqual(local.preservation_rows(data, reports, 1, "r", 0, 42), [])

    def test_live_api_key_rotation(self):
        from experiments.evolving_structured_rubrics.framework_v6_manager import Manager
        import threading
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as temp:
            env = Path(temp) / ".env"
            env.write_text("GUIJI_API_KEY=first\nGUIJI_API_KEY=second\n", encoding="utf-8")
            manager = Manager.__new__(Manager)
            manager.config = {"api_key_env": "GUIJI_API_KEY", "env_file": str(env)}
            manager._key_lock = threading.Lock()
            manager._key_index = 0
            manager._injected_client = False
            class Client:
                def with_options(self, **kwargs):
                    return kwargs["api_key"]
            manager.client = Client()
            self.assertEqual(manager._request_client(), "first")
            self.assertEqual(manager._request_client(), "second")
            env.write_text('GUIJI_API_KEY=third,fourth\n', encoding="utf-8")
            with ThreadPoolExecutor(max_workers=6) as pool:
                values = list(pool.map(lambda _: manager._request_client(), range(20)))
            self.assertEqual(values.count("third"), 10)
            self.assertEqual(values.count("fourth"), 10)
            env.write_text('GUIJI_API_KEY=\'["fifth", "sixth"]\'\n', encoding="utf-8")
            self.assertEqual(manager._read_keys(), ["fifth", "sixth"])

    def test_stage_concurrency_limits(self):
        from experiments.evolving_structured_rubrics.subtree_local_reflection_manager import make_manager
        config = local.load_json(local.ROOT / "experiments/evolving_structured_rubrics/configs/subtree_local_reflection.example.json")["manager"]
        manager = make_manager(config, 4, client=object())
        for stage in ("signature", "case_reflection", "subtree_split"):
            semaphore = manager.stage_slots.get(stage, manager.slots)
            count = 6 if stage != "subtree_split" else 4
            for _ in range(count):
                self.assertTrue(semaphore.acquire(blocking=False))
            self.assertFalse(semaphore.acquire(blocking=False))
            for _ in range(count):
                semaphore.release()

    def setUp(self):
        self.rows = rows(4)
        r0 = local.base.build_multicrit_open_ended_init_rubric()
        self.rubric = local.base.replace_groups(r0, {r: [dict(name=f"rule{i}", description=f"check {i}")
                                                     for i in range(2)] for r in r0.root_ids})

    def test_covered_acceptance_not_strict_or_coverage_gate(self):
        a = root_value(self.rows, "r", ["A", "A", "B", "B"])
        b = root_value(self.rows, "r", ["A", "None", "None", "None"])
        decision = local.compare(a, b, self.rows)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["after"]["strict_acc"], .25)
        self.assertEqual(len(decision["transitions"]["ab_to_none"]), 3)
        self.assertFalse(local.compare(a, a, self.rows)["accepted"])

    def test_strict_acceptance_requires_more_correct_answers(self):
        a = root_value(self.rows, "r", ["A", "A", "B", "B"])
        abstain = root_value(self.rows, "r", ["A", "None", "None", "None"])
        improved = root_value(self.rows, "r", ["A", "A", "A", "None"])
        self.assertTrue(local.compare(a, abstain, self.rows)["accepted"])
        self.assertFalse(local.compare(a, abstain, self.rows, "strict_acc")["accepted"])
        self.assertTrue(local.compare(a, improved, self.rows, "strict_acc")["accepted"])
        self.assertFalse(local.compare(a, a, self.rows, "strict_acc")["accepted"])
        with self.assertRaises(ValueError):
            local.compare(a, a, self.rows, "unknown")

    def test_zero_coverage_and_technical_failure(self):
        empty = root_value(self.rows, "r", ["None"]*4)
        wrong = root_value(self.rows, "r", ["B"]*4)
        good = root_value(self.rows, "r", ["A"]*4)
        self.assertFalse(local.compare(good, empty, self.rows)["accepted"])
        self.assertFalse(local.compare(empty, wrong, self.rows)["accepted"])
        self.assertTrue(local.compare(empty, good, self.rows)["accepted"])
        good["samples"][0]["call"]["parse_ok"] = False
        with self.assertRaises(RuntimeError):
            local.compare(empty, good, self.rows)

    def test_payload_and_no_actionable_output(self):
        call = dict(order=0, call=dict(parse_ok=True, parsed=report("None")))
        value = local.local_case(self.rows[0], call)
        self.assertNotIn("arbiter", value)
        self.assertEqual(value["A"], self.rows[0]["A"])
        self.assertEqual(value["report"]["thought"], report()["thought"])
        validate_local("case_reflection", dict(analysis="valid local abstention", critique=""), {})
        call["order"] = 1
        with self.assertRaises(ValueError):
            local.local_case(self.rows[0], call)

    def test_frozen_context_mixed_acceptance_and_report_reuse(self):
        rubric, data = self.rubric, self.rows
        baseline = artifact(data, rubric, 2)
        baseline.update(split="init/system", root_subtree_sha256={
            r: local.system._root_subtree_sha256(rubric, r) for r in rubric.root_ids})
        calls = []

        class Manager:
            config = {"concurrency": 2}

            def call(self, stage, path, payload, rows=()):
                calls.append((stage, deepcopy(payload)))
                if stage == "case_reflection":
                    return dict(analysis="local failure", critique="Check scope explicitly")
                return dict(children=[dict(name=f"changed{i}", description=f"new check {i}")
                                      for i in range(2)], change_summary="clarify")

        def evaluate_root(config, target, name, rows, candidate, root, attempts):
            return root_value(data, root, ["A"]*4 if root == rubric.root_ids[0] else ["B"]*4)

        def evaluate(config, target, name, rows, updated, **kw):
            self.assertEqual(kw["changed"], [])
            for sample in kw["baseline"]["samples"]:
                rs = sample["replicates"]["0"]["subtrees"]
                self.assertEqual(rs[rubric.root_ids[0]]["parsed"]["answer"], "A")
                self.assertEqual(rs[rubric.root_ids[1]]["parsed"]["answer"], "B")
            return kw["baseline"]

        with TemporaryDirectory() as temp, patch.object(local, "root_evaluate", side_effect=evaluate_root), \
                patch.object(local.base, "evaluate", side_effect=evaluate):
            updated, _, summary = local.evolve_epoch({}, Path(temp), 1, data, rubric, baseline, Manager(), 4)
        self.assertEqual(summary["accepted_roots"], [rubric.root_ids[0]])
        self.assertEqual(sum(stage == "case_reflection" for stage, _ in calls), 20)
        self.assertEqual(sum(stage == "subtree_split" for stage, _ in calls), 5)
        for stage, payload in calls:
            self.assertEqual(payload["rubric"], local.base.project_rubric(rubric))
            self.assertNotIn("arbiter", str(payload))
            if stage == "subtree_split":
                self.assertNotIn("analysis", payload["critiques"][0])
                self.assertNotIn("case", payload)
        self.assertEqual(updated.get_node(rubric.root_ids[0]), rubric.get_node(rubric.root_ids[0]))
        self.assertEqual(updated.children(rubric.root_ids[1]), rubric.children(rubric.root_ids[1]))

    def test_completed_resume_makes_no_calls(self):
        with TemporaryDirectory() as temp:
            target = Path(temp)
            local.write(target / "state.json", dict(completed=True, rubric=self.rubric.to_dict()))
            with patch.object(local, "evolve_epoch") as evolve:
                local.run({}, target, manager=object(), rows=self.rows)
                evolve.assert_not_called()

    def test_single_endpoint_root_execution_never_calls_arbiter(self):
        config = local.load_json(local.ROOT / "experiments/evolving_structured_rubrics/configs/subtree_local_reflection.example.json")["worker"]
        config["backend_pool"]["global_request_concurrency"] = 2
        config["backend_pool"]["endpoints"][0]["max_concurrency"] = 2
        call = dict(parse_ok=True, parsed=report("A"), raw_response=json.dumps(report("A")))
        with TemporaryDirectory() as temp, \
                patch.object(local.system, "_call_subtree", return_value=call) as subtree, \
                patch.object(local.system, "_call_arbiter") as arbiter:
            value = local.system.evaluate_root(config, output_path=Path(temp) / "root.json",
                cache_dir=Path(temp) / "cache", split_name="test", rows=self.rows,
                rubric=self.rubric, root_id=self.rubric.root_ids[0],
                scope_sample_ids=[r["sample_id"] for r in self.rows],
                settings=local.system.RuntimeSettings(.5, 2048, 0),
                endpoint_ids=["qwen3vl8b-local"])
            self.assertEqual(subtree.call_count, 4)
            arbiter.assert_not_called()
            self.assertEqual(local.local_metrics(value, self.rows)["covered_acc"], 1.0)

    def test_empty_critiques_preserve_every_root_without_candidate_calls(self):
        class Manager:
            config = {"concurrency": 2}

            def call(self, stage, *args):
                if stage != "case_reflection":
                    raise AssertionError("Unexpected split")
                return dict(analysis="Valid local difference", critique="")

        baseline = artifact(self.rows, self.rubric, 2)
        baseline.update(split="init/system", root_subtree_sha256={
            r: local.system._root_subtree_sha256(self.rubric, r) for r in self.rubric.root_ids})
        with TemporaryDirectory() as temp, patch.object(local, "root_evaluate") as evaluate:
            updated, _, summary = local.evolve_epoch({}, Path(temp), 1, self.rows,
                self.rubric, baseline, Manager(), 4)
            evaluate.assert_not_called()
            self.assertTrue(summary["no_actionable_feedback"])
            self.assertEqual(updated.to_dict(), self.rubric.to_dict())


if __name__ == "__main__":
    unittest.main()

"""Offline scientific and recovery checks for the framework-v6 runner."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.evolving_structured_rubrics import framework_v6 as fw
from experiments.evolving_structured_rubrics import framework_v6_manager as fm
from experiments.evolving_structured_rubrics import global_arbiter_ab_only as arbiter


def rows(n=6):
    return [dict(sample_id=f"sample-{i}", question="Compare image content", A="candidate-a",
                 B="candidate-b", answer="A", image_path="image.png", source=f"source-{i%3}",
                 domain="visual", _signature_id=f"S{i+1:03d}") for i in range(n)]


def report(answer="A", thought="visible detail supports this choice"):
    return dict(analysis_a="A evidence", analysis_b="B evidence", thought=thought, answer=answer)


def artifact(data, rubric, correct, *, k=1):
    samples = []
    for i, row in enumerate(data):
        replicas = {}
        for rep in range(k):
            replicas[str(rep)] = dict(order=0, subtrees={
                r: dict(parse_ok=True, parsed=report("B"), raw_response=json.dumps(report("B")))
                for r in rubric.root_ids},
                arbiter=dict(parse_ok=True, parsed=report("A" if i < correct else "B"),
                             raw_response=json.dumps(report("A" if i < correct else "B"))))
        samples.append(dict(sample_id=row["sample_id"], replicates=replicas))
    value = dict(k=k, samples=samples, rubric_sha256=rubric.rubric_sha256,
                 protocol_version=fw.system.PROTOCOL_VERSION)
    value["metrics"] = fw.system.metrics(value, data)
    return value


class FakeManager:
    config = {"concurrency": 2}

    def __init__(self):
        self.calls = []

    def call(self, stage, path, payload, rows=()):
        self.calls.append((stage, path, deepcopy(payload)))
        if stage == "signature":
            value = dict(applicable=True, signature="Check local evidence within scope", basis="report detail")
        elif stage == "cluster":
            ids = [s["signature_id"] for s in payload["signatures"]]
            value = dict(clusters=[dict(pattern="one", signature_ids=ids[:2]),
                                   dict(pattern="two", signature_ids=ids[2:4])])
        elif stage == "children":
            epoch = next((p for p in path.parts if len(p) == 3 and p.startswith("e") and p[1:].isdigit()), "init")
            value = dict(children=[dict(name=f"rule_{i}", description=f"Verify detail {i} for {epoch}")
                                   for i in range(2)], change_summary="clarify visual checks")
        elif stage == "system":
            root = payload["rubric"][0]["root_id"]
            value = dict(root_feedback={root: [dict(sample_ids=[c["sample_id"] for c in payload["cases"]],
                       problem="scope", basis="report", direction="clarify scope")]}, system_observations=[])
        elif stage == "root":
            if "init_01" in payload["root"]["root_id"]:
                value = dict(action="revise", reason="scope confusion", revision_goal="clarify scope",
                             preserve_guidance="retain evidence checks", use_signature_ids=[
                                 s["signature_id"] for s in payload["signature_library"]])
            else:
                value = dict(action="preserve", reason="No supported deficiency")
        else:
            raise AssertionError(stage)
        value = fm.validate(stage, value, payload)
        fw.write(path, dict(parsed=value))
        return value


class TestFrameworkV6(unittest.TestCase):
    def setUp(self):
        self.r0 = fw.build_multicrit_open_ended_init_rubric()
        self.data = rows()
        self.groups = {r: [dict(name=f"rule_{i}", description=f"check {i}") for i in range(2)]
                       for r in self.r0.root_ids}

    def test_full_reason_parser_keeps_legacy_default(self):
        raw = json.dumps(report())
        self.assertEqual(arbiter.parse_global_arbiter_ab_only_response(raw), {"answer": "A"})
        self.assertEqual(arbiter.parse_global_arbiter_with_reason(raw), report())
        with self.assertRaisesRegex(ValueError, "analysis_a"):
            arbiter.parse_global_arbiter_with_reason('{"answer":"A"}')
        self.assertEqual(arbiter.parse_global_arbiter_with_reason(json.dumps(report("None")))["answer"], "None")

    def test_runtime_opts_in_to_reason_without_prompt_change(self):
        settings = fw.system.RuntimeSettings(.5, 2048, 3, retain_arbiter_reason=True)
        endpoint = SimpleNamespace(checkpoint_root="model", base_url="fake", endpoint_id="one")
        with patch.object(fw.system.support, "call_one", return_value={}) as call:
            fw.system._call_arbiter({}, endpoint, Path("cache"), "init", self.data[0], [], 0, 0, 4, settings)
            args = call.call_args.kwargs
            self.assertIs(args["response_parser"], arbiter.parse_global_arbiter_with_reason)
            self.assertEqual(args["system_prompt"], arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT)

    def test_projection_drops_examples_at_every_depth(self):
        rubric = fw.replace_groups(self.r0, self.groups)
        nodes = {key: replace(node, examples=({"SECRET_EXAMPLE": "gold and long text"},),
                             lineage={"SECRET_HISTORY": "large"}) for key, node in rubric.nodes.items()}
        rubric = replace(rubric, nodes=nodes)
        compact = json.dumps(fw.project_rubric(rubric))
        self.assertNotIn("SECRET", compact)
        self.assertIn("check 1", compact)

    def test_replace_whole_group_keeps_fixed_roots_and_other_children(self):
        s0 = fw.replace_groups(self.r0, self.groups)
        root = s0.root_ids[0]
        updated = fw.replace_groups(s0, {root: [dict(name="new1", description="new one"),
                                               dict(name="new2", description="new two")]})
        for r in s0.root_ids:
            self.assertEqual(s0.get_node(r), updated.get_node(r))
            if r != root:
                self.assertEqual(s0.children(r), updated.children(r))
        self.assertEqual(len(updated.children(root)), 2)
        self.assertEqual(len(updated.nodes), len(s0.nodes))
        # Preserving current child names must not repeatedly add root prefixes.
        same = fw.replace_groups(s0, {root: fw.project_rubric(s0, [root])[0]["children"]})
        self.assertEqual(same.rubric_sha256, s0.rubric_sha256)

    def test_selection_is_bounded_diverse_and_keeps_correct_cases(self):
        data = rows(100)
        current = artifact(data, self.r0, 50)
        previous = dict(comparison=dict(corrected_sample_ids=[f"sample-{i}" for i in range(4)],
                                       harmed_sample_ids=[f"sample-{i}" for i in range(50, 54)]))
        selected, coverage = fw.select_cases(data, current, previous)
        self.assertEqual(len(selected), 40)
        self.assertEqual(len({r['sample_id'] for r in selected}), 40)
        self.assertGreaterEqual(coverage["current_correct"], 8)
        self.assertEqual(len(coverage["source_counts"]), 3)
        self.assertEqual(fw.select_cases(data, current, previous), (selected, coverage))
        self.assertTrue({f"sample-{i}" for i in range(4)} <= set(coverage["sample_ids"]))

    def test_local_none_not_filtered_out_or_marked_wrong_by_code(self):
        current = artifact(self.data, self.r0, 2)
        rec = fw.system_records(current)[self.data[0]["sample_id"]]
        root = self.r0.root_ids[0]
        rec["subtrees"][root]["parsed"] = report("None")
        case = fw.case_payload(self.data[0], rec)
        self.assertEqual(case["current"]["reports"][root]["answer"], "None")
        self.assertNotIn("local_wrong", json.dumps(case))
        self.assertEqual(case["current"]["arbiter"]["thought"], rec["arbiter"]["parsed"]["thought"])

    def test_feedback_is_targeted_and_previous_baseline_is_preserved(self):
        current = artifact(self.data, self.r0, 4)
        before = artifact(self.data, self.r0, 2)
        manager = FakeManager()
        with TemporaryDirectory() as temp:
            reflections, packets, feedback = fw.reflect({}, manager, Path(temp), self.data, self.r0, current,
                {r: [] for r in self.r0.root_ids}, previous=current, previous_before=before)
            self.assertTrue(feedback[self.r0.root_ids[0]])
            self.assertFalse(feedback[self.r0.root_ids[1]])
            system_input = next(c[2] for c in manager.calls if c[0] == "system")
            case = next(c for c in system_input["cases"] if c["sample_id"] == "sample-2")
            self.assertEqual(case["previous_baseline"]["arbiter"]["answer"], "B")
            self.assertEqual(case["previous_candidate"]["arbiter"]["answer"], "A")
            self.assertLessEqual(len(packets[self.r0.root_ids[0]][2]), 20)

    def test_initialization_and_five_rounds_compare_current_then_resume(self):
        config = dict(max_epochs=5, manager={"concurrency": 2})
        manager = FakeManager()
        calls = []
        scores = {"r0/system": 1, "init/system": 2, "e01/candidate_system": 4,
                  "e02/candidate_system": 3, "e03/candidate_system": 5,
                  "e04/candidate_system": 5, "e05/candidate_system": 6}
        def evaluate(config, target, name, data, rubric, **kwargs):
            calls.append((name, kwargs))
            result = artifact(data, rubric, scores[name])
            fw.write(target / f"{name}.json", result)
            return result
        with TemporaryDirectory() as temp, patch.object(fw, "load_rows", return_value=self.data), \
                patch.object(fw, "evaluate", side_effect=evaluate):
            target = Path(temp)
            fw.run(config, target, manager=manager)
            s0 = fw.StructuredRubric.load_json(target / "init/rubric.json")
            self.assertTrue(all(len(s0.children(r)) == 2 for r in s0.root_ids))
            state = fw.load_json(target / "state.json")
            self.assertEqual(state["baseline"], "e05/candidate_system")
            self.assertEqual(state["epoch"], 5)
            results = [fw.load_json(target / f"e{i:02d}/summary.json") for i in range(1, 6)]
            self.assertEqual([r["accepted"] for r in results], [True, False, True, False, True])
            self.assertEqual(results[1]["baseline_system"], "e01/candidate_system")
            self.assertEqual(results[2]["baseline_system"], "e01/candidate_system")
            self.assertTrue(all(len(kwargs["changed"]) == 1 for name, kwargs in calls if name.startswith("e")))
            fw.report(config, target)
            self.assertTrue((target / "mechanism_examples.json").is_file())
            count = len(calls), len(manager.calls)
            fw.run(config, target, manager=manager)
            self.assertEqual(count, (len(calls), len(manager.calls)))

    def test_incomplete_initialization_never_creates_s0(self):
        manager = FakeManager()
        def evaluate(config, target, name, data, rubric, **kwargs):
            return artifact(data, rubric, 1)
        with TemporaryDirectory() as temp, patch.object(fw, "evaluate", side_effect=evaluate), \
                patch.object(fw, "generate_group", return_value=None):
            target = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "S0 not created"):
                fw.initialize({}, target, self.data, manager, 4)
            self.assertFalse((target / "state.json").exists())
            self.assertFalse((target / "init/rubric.json").exists())

    def test_technical_failure_is_not_a_competition_loss(self):
        a = artifact(self.data, self.r0, 2)
        b = artifact(self.data, self.r0, 4)
        b["metrics"]["technical_failure_count"] = 1
        with self.assertRaises(RuntimeError):
            fw.joint.joint_decision(a, b, self.data)

    def test_all_preserve_stops_without_candidate_inference(self):
        manager = FakeManager()
        original_call = manager.call
        def preserve(stage, path, payload, rows=()):
            if stage == "root":
                return dict(action="preserve", reason="no supported revision")
            return original_call(stage, path, payload, rows)
        manager.call = preserve
        def evaluate(config, target, name, data, rubric, **kwargs):
            self.assertIn(name, {"r0/system", "init/system"})
            value = artifact(data, rubric, 2)
            fw.write(target / f"{name}.json", value)
            return value
        with TemporaryDirectory() as temp, patch.object(fw, "load_rows", return_value=self.data), \
                patch.object(fw, "evaluate", side_effect=evaluate):
            target = Path(temp)
            fw.run(dict(max_epochs=5), target, manager=manager)
            state = fw.load_json(target / "state.json")
            self.assertEqual(state["stop_reason"], "all_roots_preserved")
            self.assertEqual(state["baseline"], "init/system")
            self.assertFalse((target / "e01/candidate_system.json").exists())
            fw.report({}, target)

    def test_external_uses_split_initial_and_counterbalanced_k3(self):
        from experiments.evolving_structured_rubrics import vl_rewardbench as vlrb
        initial = fw.replace_groups(self.r0, self.groups)
        root = initial.root_ids[0]
        final = fw.replace_groups(initial, {root: [dict(name=f"fixed{i}", description=f"new {i}") for i in range(2)]})
        records = [dict(sample_id=r["sample_id"], benchmark_id=f"{('vlfeedback', 'RLHF', 'mathverse')[i%3]}_{i}",
                        image_path=r["image_path"], question=r["question"], responses=[r["A"], r["B"]],
                        preferred_original_index=0, group=("general", "hallucination", "reasoning")[i%3], query_source="fixture")
                   for i, r in enumerate(self.data)]
        calls = []
        def evaluate(config, target, name, data, rubric, **kwargs):
            calls.append((name, rubric, kwargs))
            self.assertTrue(all(len(order) == 3 for order in kwargs["orders"].values()))
            return artifact(data, rubric, 4, k=3)
        with TemporaryDirectory() as temp, patch.object(vlrb, "_read_records", return_value=records), \
                patch.object(fw, "evaluate", side_effect=evaluate):
            target = Path(temp)
            fw.write(target / "init/rubric.json", initial.to_dict())
            fw.write(target / "state.json", dict(completed=True, rubric=final.to_dict()))
            fw.external(dict(data_root=".", datasets={"vlrb": "test.parquet"}), target, "vlrb", 4)
            self.assertEqual(calls[0][1].rubric_sha256, initial.rubric_sha256)
            self.assertEqual(calls[1][2]["changed"], [root])
            self.assertEqual(calls[0][2]["orders"], calls[1][2]["orders"])
            self.assertEqual(fw.load_json(target / "vlrb/report.json")["k"], 3)

    def test_identical_external_rubric_reuses_results(self):
        initial = fw.replace_groups(self.r0, self.groups)
        before = artifact(self.data, initial, 4)
        with TemporaryDirectory() as temp, patch.object(fw, "load_rows", return_value=self.data), \
                patch.object(fw, "evaluate", return_value=before) as evaluate:
            target = Path(temp)
            fw.write(target / "init/rubric.json", initial.to_dict())
            fw.write(target / "state.json", dict(completed=True, rubric=initial.to_dict()))
            fw.external({}, target, "dev", 4)
            self.assertEqual(evaluate.call_count, 1)
            self.assertTrue(fw.load_json(target / "dev/report.json")["identical_rubric_reuse"])

    def test_worker_and_arbiter_prompts_exclude_gold(self):
        row = dict(self.data[0], answer="SECRET_GOLD", labels="SECRET_LABEL")
        root = self.r0.root_ids[0]
        prompt = fw.system.unified.subtree_user_prompt(row, self.r0, root)
        prompt += fw.system.support.global_arbiter_user_prompt(row, [])
        self.assertNotIn("SECRET", prompt)
        self.assertIn("candidate-a", prompt)


class TestManagerBoundary(unittest.TestCase):
    def test_timeout_change_allows_resume_without_replacing_initial_snapshot(self):
        config = dict(manager=dict(timeout=300, model="fixed", request_kwargs={"temperature": .2}))
        with TemporaryDirectory() as temp:
            target = Path(temp)
            fw.freeze_config(config, target)
            updated = deepcopy(config)
            updated["manager"]["timeout"] = 500
            fw.freeze_config(updated, target)
            self.assertEqual(fw.load_json(target / "run_config.json"), config)
            updated["manager"]["model"] = "different"
            with self.assertRaises(ValueError):
                fw.freeze_config(updated, target)

    def test_rate_limit_waits_and_does_not_send_transport_error_to_model(self):
        config = dict(concurrency=4, api_key_env="TEST_KEY", base_url="http://fake/v1",
                      timeout=300, model="fake", request_kwargs={})
        error = RuntimeError("TPM limit reached")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "75"})
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(dict(applicable=False, basis="no local defect"))), finish_reason="stop")], usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock()))
        client.chat.completions.create.side_effect = [error, response]
        manager = fm.Manager(config, attempts=2, client=client)
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        with TemporaryDirectory() as temp, patch.object(fm.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(fm.time, "sleep", side_effect=sleep):
            path = Path(temp) / "call.json"
            self.assertFalse(manager.call("signature", path, {})["applicable"])
            self.assertEqual(clock[0], 75.0)
            self.assertEqual(fw.load_json(path)["attempts"][0]["cooldown_seconds"], 75.0)
            self.assertNotIn("Previous output validation failed", str(client.chat.completions.create.call_args))

    def test_rate_limit_cooldown_applies_to_other_logical_calls(self):
        config = dict(concurrency=4, api_key_env="TEST_KEY", model="fake", base_url="http://fake/v1",
                      timeout=300, request_kwargs={})
        manager = fm.Manager(config, client=Mock())
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(fm.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(fm.time, "sleep", side_effect=sleep):
            delay = manager._defer_rate_limit(RuntimeError("limit from root1"), 2)
            self.assertEqual(delay, 120.0)
            manager._wait_for_capacity("root2")
            self.assertEqual(clock[0], 120.0)

    def test_child_whitespace_is_normalized_before_caching(self):
        value = fm.validate("children", dict(children=[
            dict(name=" check_one ", description=" detail one "),
            dict(name="check_two", description="detail two")],
            change_summary="clarified"), {})
        self.assertEqual(value["children"][0]["name"], "check_one")
        rubric = fw.build_multicrit_open_ended_init_rubric()
        fw.replace_groups(rubric, {rubric.root_ids[0]: value["children"]})

    def test_preserve_extra_revision_fields_do_not_trigger_retry(self):
        value = fm.validate("root", dict(action="Preserve", reason="adequate",
                            revision_goal="incidental text", use_signature_ids=["unknown"]), {})
        self.assertEqual(value, dict(action="preserve", reason="adequate"))

    def test_cluster_unassigned_ids_are_not_failures(self):
        payload = dict(initial=True, signatures=[dict(signature_id=f"S{i}") for i in range(6)])
        value = dict(clusters=[dict(pattern="one", signature_ids=["S0", "S1"]),
                              dict(pattern="two", signature_ids=["S2", "S3"])])
        self.assertEqual(fm.validate("cluster", value, payload)["unassigned_ids"], ["S4", "S5"])
        self.assertEqual(fm.validate("cluster", {"clusters": []}, payload)["clusters"], [])
        value["clusters"][1]["signature_ids"] = ["S1", "S4"]
        with self.assertRaises(ValueError):
            fm.validate("cluster", value, payload)

    def test_model_cache_usage_retry_and_raw_reasoning(self):
        config = dict(concurrency=1, api_key_env="TEST_KEY", base_url="http://fake/v1",
                      timeout=300, model="fake", request_kwargs={"extra_body": {"enable_thinking": True}})
        def response(content):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content,
                reasoning_content="provider thinking"), finish_reason="stop")],
                usage=SimpleNamespace(model_dump=lambda: dict(prompt_tokens=10, completion_tokens=20)))
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock()))
        client.chat.completions.create.side_effect = [response('{}'), response(json.dumps(
            dict(applicable=False, basis="no supported local defect")))]
        manager = fm.Manager(config, attempts=2, client=client)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "signature.json"
            result = manager.call("signature", path, {"case": "fixture"})
            self.assertFalse(result["applicable"])
            self.assertEqual(client.chat.completions.create.call_count, 2)
            cached = manager.call("signature", path, {"case": "fixture"})
            self.assertEqual(cached, result)
            self.assertEqual(client.chat.completions.create.call_count, 2)
            record = fw.load_json(path)
            self.assertEqual(len(record["attempts"]), 2)
            self.assertEqual(record["attempts"][1]["reasoning_content"], "provider thinking")
            self.assertIn("Previous output validation failed", str(client.chat.completions.create.call_args))

    def test_exhausted_attempts_do_not_restart_on_resume(self):
        config = dict(concurrency=1, api_key_env="TEST_KEY", base_url="http://fake/v1",
                      timeout=300, model="fake", request_kwargs={})
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock()))
        client.chat.completions.create.side_effect = RuntimeError("test transport failure")
        manager = fm.Manager(config, attempts=1, client=client)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "call.json"
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    manager.call("signature", path, {})
            self.assertEqual(client.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()

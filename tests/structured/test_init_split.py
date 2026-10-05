"""Offline contracts for templated initialization and independent S0 evaluation."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.evolving_structured_rubrics import init_split as init
from experiments.evolving_structured_rubrics import rubric_pipeline as pipeline
from experiments.evolving_structured_rubrics import run_subtree_experiment as entry
from tests.structured.core_fixtures import artifact, rows
from tests.structured.test_manager_runtime import response


class TestInitSplit(unittest.TestCase):
    def test_templates_share_all_roots_and_preserve_original_case_text(self):
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        data = rows(1)
        data[0].update(question='公式 {x} 与 "quoted"\n第二行', A="A 的原文 {candidate_b}")
        root = r0.root_ids[2]
        value = artifact(data, r0, 1)
        value["samples"][0]["replicates"]["0"]["arbiter"] = {}
        case = init.case_payload(data[0], value["samples"][0]["replicates"]["0"], root)
        self.assertNotIn("arbiter", case)
        payload = dict(roots=pipeline.project_rubric(r0),
                       root=pipeline.project_rubric(r0, [root])[0], case=case)
        signatures = [dict(signature_id=f"S{i:03d}", signature=f"pattern {i}",
                           basis=f"evidence {i}") for i in range(1, 5)]
        for stage in ("signature", "cluster", "children"):
            with self.subTest(stage=stage):
                payload["signatures"] = signatures
                payload["clusters"] = dict(clusters=[
                    dict(pattern="shared cause", signature_ids=["S001", "S002"])],
                    unassigned_ids=["S003", "S004"])
                text = init.render_user_prompt(stage, payload)
                for root_id in r0.root_ids:
                    self.assertIn(root_id, text)
                    self.assertIn(r0.get_node(root_id).criterion.description, text)
                self.assertNotIn("arbiter", text.lower())
                if stage == "signature":
                    self.assertIn(data[0]["question"], text)
                    self.assertIn(data[0]["A"], text)
                    self.assertIn(case["worker"]["thought"], text)
                    self.assertIn("Local judgment:\nB", text)
                else:
                    self.assertIn("S001", text)
                    self.assertIn("evidence 4", text)
                    if stage == "children":
                        self.assertIn("shared cause", text)

    def test_rendered_user_text_is_sent_and_bound_to_cache_with_image(self):
        config = dict(model="offline", base_url="http://offline/v1", concurrency=1,
                      timeout=1, request_kwargs={})
        create = Mock(return_value=response(
            '{"applicable":true,"signature":"check","basis":"visible evidence"}'))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        manager = init.make_manager(config, client=client)
        image = dict(sample_id="sample-1", image_path="image.png")
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()), \
                patch.object(pipeline.system.support, "content",
                             return_value=[dict(type="image_url",
                                                image_url=dict(url="data:image/png;base64,AAAA"))]):
            path = Path(temp) / "signature.json"
            result = manager.call("signature", path, {}, [image], user_text="原文 {x}\n证据")
            saved = path.read_bytes()
            self.assertEqual(manager.call("signature", path, {}, [image],
                                          user_text="原文 {x}\n证据"), result)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(pipeline.load_json(path)["request"]["user_text"], "原文 {x}\n证据")
            with self.assertRaisesRegex(ValueError, "input changed"):
                manager.call("signature", path, {}, [image], user_text="另一个模板")
            with patch.dict(manager.prompts, signature="changed system"):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    manager.call("signature", path, {}, [image], user_text="原文 {x}\n证据")
        self.assertEqual(create.call_count, 1)
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], init.SIGNATURE_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"][0]["text"], "原文 {x}\n证据")
        self.assertEqual(messages[1]["content"][1]["type"], "image_url")

    def test_cluster_validation_keeps_supported_unassigned_and_distinct_ids(self):
        payload = dict(signatures=[dict(signature_id=f"S{i:03d}") for i in range(1, 6)])
        cluster = dict(pattern="cause", signature_ids=["S001", "S002"])
        result = init.validate("cluster", dict(clusters=[cluster]), payload)
        self.assertEqual(result["unassigned_ids"], ["S003", "S004", "S005"])
        self.assertEqual(init.validate("cluster", dict(clusters=[]), payload)["clusters"], [])
        for clusters in ([cluster, deepcopy(cluster)],
                         [dict(pattern="cause", signature_ids=["S001", "unknown"])],
                         [dict(pattern="cause", signature_ids=["S001", "S001"])]):
            with self.subTest(clusters=clusters), self.assertRaises(ValueError):
                init.validate("cluster", dict(clusters=clusters), payload)

    def test_changed_template_cannot_resume_an_existing_initialization(self):
        with TemporaryDirectory() as temp:
            target = Path(temp)
            pipeline.write(target / "init/protocol.json", dict(
                version=init.PROMPT_VERSION, system_prompts=init.PROMPTS,
                user_templates=init.USER_TEMPLATES))
            manager = SimpleNamespace()
            with patch.dict(init.USER_TEMPLATES, signature="changed"), \
                    patch.object(pipeline, "evaluate") as evaluate:
                with self.assertRaisesRegex(ValueError, "prompts changed"):
                    init.initialize({}, target, rows(), manager=manager)
                evaluate.assert_not_called()

    def test_reused_r0_reports_make_no_model_calls_and_keep_source_untouched(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            source, target = Path(temp) / "source", Path(temp) / "target"
            data = rows(3)
            image = Path(temp) / "image.png"
            image.write_bytes(b"same-image")
            for item in data:
                item["image_path"] = str(image)
            source_config = dict(worker=dict(model="worker", backend_pool=dict(endpoints=[
                dict(base_url="http://source-worker/v1")])), manager=dict(
                model="manager", base_url="http://offline/v1", request_kwargs={}))
            config = deepcopy(source_config)
            config["worker"]["backend_pool"]["endpoints"][0]["base_url"] = "http://target-worker/v1"
            r0 = pipeline.build_multicrit_open_ended_init_rubric()
            baseline = artifact(data, r0, 3)
            pipeline.write(source / "run_config.json", source_config)
            pipeline.write(source / "r0/rubric.json", r0.to_dict())
            pipeline.write(source / "r0/system.json", baseline)
            before = {p: p.read_bytes() for p in source.rglob("*.json")}
            with patch.object(pipeline, "load_rows", return_value=data), \
                    patch.object(init, "make_manager") as manager, \
                    patch.object(pipeline, "evaluate") as evaluate:
                reused = init.reuse_source(config, target, source)
                reused_again = init.reuse_source(config, target, source)
                manager.assert_not_called()
                evaluate.assert_not_called()
            self.assertEqual(reused.to_dict(), r0.to_dict())
            self.assertEqual(reused_again.to_dict(), r0.to_dict())
            self.assertEqual(pipeline.load_json(target / "r0/system.json"), baseline)
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_initialization_shares_previous_children_only_when_generating_children(self):
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        data = rows(4)
        calls, proposals = [], {}

        def call(stage, path, payload, *args, **kwargs):
            calls.append((stage, deepcopy(payload), kwargs["user_text"]))
            if stage == "signature":
                return dict(applicable=True, signature="supported preference pattern",
                            basis="specific case evidence")
            if stage == "cluster":
                ids = [item["signature_id"] for item in payload["signatures"]]
                return dict(clusters=[
                    dict(pattern="first pattern", signature_ids=ids[:2]),
                    dict(pattern="second pattern", signature_ids=ids[2:])], unassigned_ids=[])
            root = payload["root"]["root_id"]
            children = [dict(name=f"check_{i}", description=f"New guidance for {root}, check {i}")
                        for i in (1, 2)]
            proposals[root] = children
            return dict(children=children, change_summary="case-derived guidance")

        manager = SimpleNamespace(config=dict(concurrency=1), call=Mock(side_effect=call))

        def evaluate(config, target, stage, data, rubric, **kwargs):
            value = artifact(data, rubric, len(data))
            if stage == "r0/system":
                # Two primary failures require two additional cases before clustering.
                for sample in value["samples"][2:]:
                    record = sample["replicates"]["0"]
                    record["subtrees"] = {
                        root: deepcopy(record["arbiter"]) for root in rubric.root_ids}
                value["metrics"] = pipeline.system.metrics(value, data)
            return value

        with TemporaryDirectory() as temp, redirect_stdout(StringIO()), \
                patch.object(pipeline, "evaluate", side_effect=evaluate), \
                patch.object(pipeline, "manager_cost", return_value={}):
            result, _ = init.initialize({}, Path(temp), data, manager=manager, r0=r0)
            summary = pipeline.load_json(Path(temp) / "init/summary.json")

        self.assertEqual([item["expanded_count"] for item in summary["roots"]], [2] * 5)
        for stage, payload, text in calls:
            root = payload["root"]["root_id"]
            index = list(r0.root_ids).index(root)
            with self.subTest(root=root, stage=stage):
                if stage == "children":
                    previous = payload["previous_children"]
                    self.assertEqual([item["root_id"] for item in previous], list(r0.root_ids[:index]))
                    self.assertIn("## Previously generated child criteria", text)
                    for item in previous:
                        for child, proposal in zip(item["children"], proposals[item["root_id"]]):
                            self.assertEqual(child["description"], proposal["description"])
                            self.assertIn(child["name"], text)
                            self.assertIn(child["description"], text)
                    if not previous:
                        self.assertIn("No child criteria have been generated yet.", text)
                else:
                    self.assertNotIn("previous_children", payload)
                    self.assertNotIn("## Previously generated child criteria", text)
                    self.assertNotIn("New guidance for", text)
                self.assertEqual(payload["roots"], pipeline.project_rubric(r0))
                self.assertEqual(payload["root"], pipeline.project_rubric(r0, [root])[0])
        for root in r0.root_ids:
            root_calls = [(stage, payload) for stage, payload, _ in calls
                          if payload["root"]["root_id"] == root]
            self.assertEqual([stage for stage, _ in root_calls],
                             ["signature"] * 4 + ["cluster", "children"])
            self.assertEqual([payload["case"]["sample_id"] for stage, payload in root_calls
                              if stage == "signature"], [row["sample_id"] for row in data])
        self.assertEqual([len(result.children(root)) for root in r0.root_ids], [2] * 5)

    def test_reused_patterns_generate_only_children_in_root_order(self):
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        data = rows(4)
        calls = []
        libraries, clusters = {}, {}

        def call(stage, path, payload, *args, **kwargs):
            self.assertEqual(stage, "children")
            calls.append(deepcopy(payload))
            root = payload["root"]["root_id"]
            return dict(children=[
                dict(name=f"new_check_{i}", description=f"Fresh guidance for {root}, check {i}")
                for i in (1, 2)], change_summary="coordinate prior children")

        manager = SimpleNamespace(config=dict(concurrency=1), call=Mock(side_effect=call))
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            source, target = Path(temp) / "source", Path(temp) / "target"
            diagnostics = []
            for index, root in enumerate(r0.root_ids, 1):
                libraries[root] = [dict(signature_id=f"S{i:03d}", sample_id=row["sample_id"],
                                        applicable=True, signature=f"Saved pattern {root}, {i}",
                                        basis=f"Saved evidence {i}")
                                   for i, row in enumerate(data, 1)]
                clusters[root] = dict(clusters=[
                    dict(pattern=f"Saved cluster {root}, {i}", signature_ids=ids)
                    for i, ids in enumerate((["S001", "S002"], ["S003", "S004"]), 1)],
                    unassigned_ids=[])
                pipeline.write(source / f"init/r{index:02d}/library.json", libraries[root])
                diagnostics.append(dict(root_id=root, primary_count=4, expanded_count=0,
                                        clusters=clusters[root]))
            pipeline.write(source / "init/summary.json", dict(
                r0_sha256=r0.rubric_sha256, roots=diagnostics))
            before = {path: path.read_bytes() for path in source.rglob("*.json")}
            with patch.object(pipeline, "evaluate", side_effect=lambda config, target, stage,
                              data, rubric, **kwargs: artifact(data, rubric, 3)), \
                    patch.object(pipeline, "manager_cost", return_value={}):
                result, _ = init.initialize({}, target, data, manager=manager, r0=r0,
                                            reuse_patterns_from=source)
            self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
            for index, payload in enumerate(calls):
                root = r0.root_ids[index]
                with self.subTest(root=root):
                    self.assertEqual(payload["signatures"], libraries[root])
                    self.assertEqual(payload["clusters"], clusters[root])
                    previous = payload["previous_children"]
                    self.assertEqual([item["root_id"] for item in previous], list(r0.root_ids[:index]))
                    self.assertTrue(all(child["description"].startswith("Fresh guidance")
                                        for item in previous for child in item["children"]))
                    self.assertEqual(pipeline.load_json(target / f"init/r{index + 1:02d}/library.json"),
                                     libraries[root])
            self.assertEqual(len(calls), 5)
            self.assertEqual([len(result.children(root)) for root in r0.root_ids], [2] * 5)

    def test_init_entry_stops_before_evolution(self):
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        args = SimpleNamespace(config=Path("config.json"), output_root=Path("output"),
                               variant="g5", stage="init", source_run=Path("source"),
                               attempt_limit=4, reuse_init_patterns=False)
        with patch.object(entry, "load_json", return_value={"env_file": ".env"}), \
                patch.object(entry, "load_dotenv"), \
                patch.object(entry.subtree_local_reflection, "check"), \
                patch.object(init, "reuse_source", return_value=r0) as reuse, \
                patch.object(pipeline, "load_rows", return_value=rows(6)), \
                patch.object(init, "initialize") as initialize, \
                patch.object(entry.subtree_local_reflection, "run") as evolve:
            entry.run_stage(args)
        reuse.assert_called_once()
        self.assertEqual(initialize.call_args.kwargs, dict(attempts=4, r0=r0))
        evolve.assert_not_called()

    def test_init_entry_passes_pattern_source_without_running_evolution(self):
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        source = Path("source")
        args = SimpleNamespace(config=Path("config.json"), output_root=Path("output"),
                               variant="g5", stage="init", source_run=source,
                               attempt_limit=4, reuse_init_patterns=True)
        with patch.object(entry, "load_json", return_value={"env_file": ".env"}), \
                patch.object(entry, "load_dotenv"), \
                patch.object(entry.subtree_local_reflection, "check"), \
                patch.object(init, "reuse_source", return_value=r0), \
                patch.object(pipeline, "load_rows", return_value=rows(6)), \
                patch.object(init, "initialize") as initialize, \
                patch.object(entry.subtree_local_reflection, "run") as evolve:
            entry.run_stage(args)
        self.assertEqual(initialize.call_args.kwargs,
                         dict(attempts=4, r0=r0, reuse_patterns_from=source))
        evolve.assert_not_called()

    def test_s0_external_entry_uses_formal_k3_without_final_or_completed_state(self):
        data = rows(3)
        rubric = pipeline.build_multicrit_open_ended_init_rubric()
        value = artifact(data, rubric, 3, k=3)
        for rep in ("1", "2"):
            value["samples"][0]["replicates"][rep]["arbiter"]["parsed"]["answer"] = "None"
        value["metrics"] = pipeline.system.metrics(value, data)
        records = [dict(sample_id=row["sample_id"], benchmark_id=f"{prefix}_{i}",
                        group=group, preferred_original_index=0)
                   for i, (row, prefix, group) in enumerate(zip(
                       data, ("vlfeedback", "RLHF", "mathverse"),
                       ("general", "hallucination", "reasoning")))]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            pipeline.write(config_path, dict(env_file=".env"))
            target = root / "g5"
            pipeline.write(target / "init/rubric.json", rubric.to_dict())
            args = SimpleNamespace(config=config_path, output_root=root, variant="g5",
                                   stage="vlrb-s0", attempt_limit=4)
            with patch.object(entry, "load_dotenv"), \
                    patch.object(entry.subtree_local_reflection, "check"), \
                    patch.object(pipeline, "evaluate_external",
                                 return_value=(value, records, data)) as evaluate:
                entry.run_stage(args)
            result = pipeline.load_json(target / "vlrb/s0_report.json")
            self.assertEqual(result["official"]["correct_count"], 2)
            self.assertEqual(result["runtime"]["strict_accuracy"], 1.0)
            self.assertEqual(evaluate.call_args.args[2:4], ("vlrb", "initial"))
            self.assertFalse((target / "final.json").exists())
            self.assertFalse((target / "state.json").exists())


if __name__ == "__main__":
    unittest.main()

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
            config = dict(worker=dict(model="worker"), manager=dict(
                model="manager", base_url="http://offline/v1", request_kwargs={}))
            r0 = pipeline.build_multicrit_open_ended_init_rubric()
            baseline = artifact(data, r0, 3)
            pipeline.write(source / "run_config.json", config)
            pipeline.write(source / "r0/rubric.json", r0.to_dict())
            pipeline.write(source / "r0/system.json", baseline)
            before = {p: p.read_bytes() for p in source.rglob("*.json")}
            with patch.object(pipeline, "load_rows", return_value=data), \
                    patch.object(init, "make_manager") as manager, \
                    patch.object(pipeline, "evaluate") as evaluate:
                reused = init.reuse_source(config, target, source)
                init.reuse_source(config, target, source)
                manager.assert_not_called()
                evaluate.assert_not_called()
            self.assertEqual(reused.to_dict(), r0.to_dict())
            self.assertEqual(pipeline.load_json(target / "r0/system.json"), baseline)
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            changed = deepcopy(config)
            changed["worker"]["model"] = "other"
            with self.assertRaisesRegex(ValueError, "Worker configuration"):
                init.reuse_source(changed, target, source)

    def test_init_entry_stops_before_evolution(self):
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        args = SimpleNamespace(config=Path("config.json"), output_root=Path("output"),
                               variant="g5", stage="init", source_run=Path("source"),
                               attempt_limit=4)
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

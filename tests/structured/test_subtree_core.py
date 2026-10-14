"""Offline contracts for the current generated-root and subtree method."""

import hashlib
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from structured_rubrics.agent import AgentCallMetrics
from experiments.evolving_structured_rubrics import aligned_system_runtime as system
from experiments.evolving_structured_rubrics import rubric_pipeline as method
from experiments.evolving_structured_rubrics import aligned_prompts as prompts
from experiments.evolving_structured_rubrics import model_call_support as support
from experiments.evolving_structured_rubrics import vlrb_official as vlrb
from experiments.evolving_structured_rubrics import generated_root_initialization as generated
from experiments.evolving_structured_rubrics import run_subtree_experiment as entry
from tests.structured.core_fixtures import artifact, report, rows


class TestSubtreeCore(unittest.TestCase):
    def test_evolution_manager_uses_frozen_attempt_limit(self):
        config = {"manager": {}, "env_file": ".env"}
        r0 = SimpleNamespace(root_ids=("r1", "r2"))
        args = SimpleNamespace(
            config=Path("config.json"), output_root=Path("output"),
            variant="g5", stage="evolve", attempt_limit=4,
        )
        with patch.object(entry, "load_json", return_value=config), \
                patch.object(entry, "load_dotenv"), \
                patch.object(entry.subtree_local_reflection, "check"), \
                patch.object(entry.StructuredRubric, "load_json", return_value=r0), \
                patch.object(entry, "make_manager") as manager, \
                patch.object(entry.subtree_local_reflection, "run"):
            entry.run_stage(args)
        self.assertEqual(manager.call_args.args[1], 4)
        self.assertEqual(manager.call_args.kwargs["n_roots"], 2)

    def test_root_generation_uses_its_own_attempt_limit(self):
        config = {"manager": {}, "env_file": ".env"}
        args = SimpleNamespace(
            config=Path("config.json"), output_root=Path("output"),
            variant="g5", stage="roots", seed=11, root_attempt_limit=10,
            warmup_count=10,
        )
        with patch.object(entry, "load_json", return_value=config), \
                patch.object(entry, "load_dotenv"), \
                patch.object(entry.rubric_pipeline, "load_rows", return_value=[]), \
                patch.object(entry.generated_root_initialization, "generate_r0_pair") as generate:
            entry.run_stage(args)
        self.assertEqual(generate.call_args.kwargs["attempt_limit"], 10)
        self.assertEqual(generate.call_args.kwargs["warmup_count"], 10)

    def test_cli_warmup_count_defaults_to_five_and_accepts_ten(self):
        argv = ["run_subtree_experiment", "roots", "--config", "config.json",
                "--output-root", "output", "--variant", "g5"]
        for options, expected in (([], 5), (["--warmup-count", "10"], 10)):
            with self.subTest(expected=expected), patch("sys.argv", argv + options), \
                    patch.object(entry, "run_stage") as run_stage:
                entry.main()
                self.assertEqual(run_stage.call_args.args[0].warmup_count, expected)

    def test_frozen_prompts_and_reason_parser(self):
        self.assertEqual(hashlib.sha256(prompts.UNIFIED_SUBTREE_SYSTEM_PROMPT.encode()).hexdigest(),
                         "a5f07f753848287588479e276a824f3e585f4be90742a0dfbe46ad835c0dea1f")
        self.assertEqual(hashlib.sha256(prompts.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode()).hexdigest(),
                         "f6e20d219eaee04be33bbac2bddfe9f7c72c9cad3365c7aa897602911bfee3a2")
        raw = json.dumps(dict(answer="A", analysis_a="visible", analysis_b="unsupported",
                              thought="A has stronger image evidence"))
        self.assertEqual(prompts.parse_global_arbiter_with_reason(raw)["thought"],
                         "A has stronger image evidence")
        with self.assertRaises(ValueError):
            prompts.parse_global_arbiter_with_reason('{"answer":"A"}')

    def test_variable_root_arbiter_changes_only_count_phrase_and_version(self):
        endpoint = SimpleNamespace(endpoint_id="e1", checkpoint_root="checkpoint")
        row = dict(sample_id="s1", question="q", A="a", B="b")
        settings = system.RuntimeSettings(0.5, 2048, 3, retain_arbiter_reason=True)
        with patch.object(support, "call_one", return_value={}) as call:
            system._call_arbiter({}, endpoint, Path("cache"), "init", row, [],
                                 0, 0, 4, settings, 3)
        kwargs = call.call_args.kwargs
        self.assertEqual(kwargs["prompt_version"], prompts.ARBITER_PROMPT_VERSION + "-n3")
        self.assertIn("three subtree assessments produced", kwargs["system_prompt"])
        self.assertNotIn("five subtree assessments produced", kwargs["system_prompt"])
        self.assertIs(kwargs["response_parser"], prompts.parse_global_arbiter_with_reason)

    def test_fixed_r0_projection_and_group_replacement(self):
        r0 = method.build_multicrit_open_ended_init_rubric()
        self.assertEqual(r0.rubric_sha256,
                         "88f831f700967c31d22941b2cc660d9b48034bb3310706fe3124781ba53a09c8")
        root = r0.root_ids[0]
        children = [dict(name="evidence", description="Check image evidence"),
                    dict(name="scope", description="Check question scope")]
        result = method.replace_groups(r0, {root: children})
        self.assertEqual(result.root_ids, r0.root_ids)
        self.assertEqual(len(result.children(root)), 2)
        self.assertNotIn("lineage", str(method.project_rubric(result)))

    def test_generated_root_count_and_shared_history(self):
        roots = [dict(name=f"criterion_{i}", description="Judge visual evidence") for i in range(5)]
        parsed = generated._parse_roots(json.dumps(dict(count_reason="five duties", roots=roots)), "g5")
        self.assertEqual(len(parsed["roots"]), 5)
        with self.assertRaises(ValueError):
            generated._parse_roots(json.dumps(dict(count_reason="too few", roots=roots[:1])), "gn")
        warmup = dict(request=dict(sample_count=5, samples=[dict(sample_id="s1")]),
                      turns=[dict(sample_id="s1", prompt="image and question", response="A")])
        cfg = dict(model="manager", base_url="local", request_kwargs={"temperature": 0.2})
        g5 = generated._generation_request("g5", cfg, warmup, generated.LEGACY_COUNT_INSTRUCTIONS)
        gn = generated._generation_request("gn", cfg, warmup, generated.LEGACY_COUNT_INSTRUCTIONS)
        self.assertEqual(g5["warmup_history_sha256"], gn["warmup_history_sha256"])
        self.assertIn("five is allowed", gn["prompt"])

    def test_cached_call_preserves_request_identity_and_image_content(self):
        class FakeAgent:
            calls = 0

            def __init__(self, **kwargs):
                self.last_call_metrics = AgentCallMetrics()

            def __call__(self, content, stream=False):
                FakeAgent.calls += 1
                self.assert_content = content
                return '{"answer":"A"}'

        with TemporaryDirectory() as temporary, patch.object(support, "Agent", FakeAgent):
            path = Path(temporary) / "image.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            row = dict(sample_id="s1", image_path=str(path))
            endpoint = SimpleNamespace(endpoint_id="e1", checkpoint_root="checkpoint", base_url="http://local/v1")
            kwargs = dict(user_text="frozen prompt", row=row, request_key={"sample_id": "s1"},
                          total_attempt_limit=1, protocol_version="current-v1", prompt_version="frozen-v1",
                          system_prompt="system", response_parser=prompts.parse_global_arbiter_ab_only_response,
                          settings_loader=lambda _: dict(temperature=0.5, max_tokens=2048,
                                                         generation_seed_policy="unset"))
            config = dict(model="model", api_retry_attempts=0)
            first = support.call_one(config, endpoint, Path(temporary) / "cache", **kwargs)
            second = support.call_one(config, endpoint, Path(temporary) / "cache", **kwargs)
            self.assertEqual(FakeAgent.calls, 1)
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["request"]["image_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(support.content(row, "text")[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_incremental_runtime_reuses_reports_and_reruns_arbiter_with_same_ab_order(self):
        data = rows(1)
        rubric = method.build_multicrit_open_ended_init_rubric()
        endpoint = SimpleNamespace(endpoint_id="offline")
        settings = system.RuntimeSettings(0.5, 2048, 9)
        orders = (0, 1, 0)
        baseline = artifact(data, rubric, 1, k=3)["samples"][0]
        for replicate, order in enumerate(orders):
            baseline["replicates"][str(replicate)]["order"] = order
        frozen = json.dumps(baseline, sort_keys=True)

        def subtree(config, endpoint, cache, split, displayed, rubric, root,
                    replicate, order, attempts, settings):
            self.assertEqual((displayed["A"], displayed["B"]),
                             (data[0]["B"], data[0]["A"]) if order else
                             (data[0]["A"], data[0]["B"]))
            self.assertEqual(displayed["sample_id"], f"sample-0::k{replicate + 1}")
            self.assertEqual(attempts, 10)
            return dict(parse_ok=True, parsed=report("B" if order else "A"))

        def arbiter(config, endpoint, cache, split, displayed, reports,
                    replicate, order, attempts, settings, root_count):
            self.assertEqual(len(reports), root_count)
            self.assertEqual([item["root_id"] for item in reports], list(rubric.root_ids))
            self.assertTrue(all("analysis_a" in item["report"] for item in reports))
            self.assertEqual(displayed["answer"], "B" if order else "A")
            return dict(parse_ok=True, parsed=report("B" if order else "A"))

        for changed in (frozenset({rubric.root_ids[0]}), frozenset()):
            with self.subTest(changed=changed), \
                    patch.object(system, "_call_subtree", side_effect=subtree) as worker, \
                    patch.object(system, "_call_arbiter", side_effect=arbiter) as judge:
                sample = system._one_sample({}, endpoint, Path("cache"), "vlrb", data[0],
                                            orders, rubric, baseline, changed, 10, settings)
                self.assertEqual(worker.call_count, len(changed) * 3)
                self.assertEqual(judge.call_count, 3)
                for replicate in sample["replicates"].values():
                    self.assertEqual(replicate["regenerated_root_ids"], sorted(changed))
                    for root, call in replicate["subtrees"].items():
                        self.assertEqual(call["incremental_reuse"], root not in changed)
                        if root not in changed:
                            self.assertEqual(call["parsed"], report("B"))
                value = dict(k=3, samples=[sample])
                metric = system.metrics(value, data)
                self.assertEqual(metric["predictions_by_replicate"], [["A"], ["A"], ["A"]])
                self.assertEqual(metric["strict_accuracy"], 1.0)
                self.assertEqual(json.dumps(baseline, sort_keys=True), frozen)

    def test_runtime_distinguishes_unresolved_subtree_from_scientific_abstention(self):
        data = rows(1)
        rubric = method.build_multicrit_open_ended_init_rubric()
        endpoint = SimpleNamespace(endpoint_id="offline")
        settings = system.RuntimeSettings(0.5, 2048, 9)
        for parse_ok in (False, True):
            call = dict(parse_ok=parse_ok, parsed=report("None") if parse_ok else None)
            with self.subTest(parse_ok=parse_ok), \
                    patch.object(system, "_call_subtree", return_value=call), \
                    patch.object(system, "_call_arbiter", return_value=call) as judge:
                sample = system._one_sample({}, endpoint, Path("cache"), "dev", data[0],
                                            (1,), rubric, None, None, 10, settings)
                final = sample["replicates"]["0"]["arbiter"]
                self.assertEqual(judge.call_count, int(parse_ok))
                if not parse_ok:
                    self.assertEqual(final["error"], "blocked_by_unresolved_subtree")
                metric = system.metrics(dict(k=1, samples=[sample]), data)
                self.assertEqual(metric["predictions"], ["None" if parse_ok else "technical_failure"])
                self.assertEqual(metric["technical_failure_count"], 0 if parse_ok else 6)
                self.assertEqual(metric["strict_accuracy"], 0.0)

    def test_external_uses_frozen_k3_schedule_and_changed_root_only(self):
        data = rows()
        initial = method.replace_groups(method.build_multicrit_open_ended_init_rubric(), {
            root: [dict(name="one", description="Check one"), dict(name="two", description="Check two")]
            for root in method.build_multicrit_open_ended_init_rubric().root_ids})
        root = initial.root_ids[0]
        final = method.replace_groups(initial, {root: [dict(name="new_one", description="New one"),
                                                       dict(name="new_two", description="New two")]})
        prefixes = ("vlfeedback", "RLHF", "mathverse")
        groups = ("general", "hallucination", "reasoning")
        records = [dict(sample_id=item["sample_id"], benchmark_id=f"{prefixes[i % 3]}_{i}",
                        image_path=item["image_path"], question=item["question"],
                        responses=[item["A"], item["B"]], preferred_original_index=0,
                        group=groups[i % 3]) for i, item in enumerate(data)]
        calls = []

        def fake_evaluate(config, target, name, items, rubric, **kwargs):
            calls.append((rubric.rubric_sha256, kwargs))
            return artifact(items, rubric, 4, k=3)

        with TemporaryDirectory() as temporary, patch.object(vlrb, "_read_records", return_value=records), \
                patch.object(method, "evaluate", side_effect=fake_evaluate):
            target = Path(temporary)
            method.write(target / "init/rubric.json", initial.to_dict())
            method.write(target / "state.json", dict(completed=True, rubric=final.to_dict()))
            method.external(dict(data_root=".", datasets={"vlrb": "test.parquet"}), target, "vlrb", 4)
            self.assertEqual(calls[0][0], initial.rubric_sha256)
            self.assertEqual(calls[1][1]["changed"], [root])
            self.assertEqual(calls[0][1]["orders"], calls[1][1]["orders"])
            self.assertIs(calls[0][1]["vlrb_records"], records)
            self.assertIs(calls[1][1]["vlrb_records"], records)
            self.assertEqual(method.load_json(target / "vlrb/report.json")["k"], 3)

    def test_vlrb_log_uses_formal_majority_for_fresh_and_cached_results(self):
        data = rows(3)
        rubric = method.build_multicrit_open_ended_init_rubric()
        value = artifact(data, rubric, 3, k=3)
        for replicate in ("1", "2"):
            value["samples"][0]["replicates"][replicate]["arbiter"]["parsed"]["answer"] = "None"
        value["metrics"] = system.metrics(value, data)
        self.assertEqual(value["metrics"]["strict_accuracy"], 1.0)
        records = [dict(sample_id=row["sample_id"], benchmark_id=f"{prefix}_{index}",
                        group=group, preferred_original_index=0)
                   for index, (row, prefix, group) in enumerate(zip(
                       data, ("vlfeedback", "RLHF", "mathverse"),
                       ("general", "hallucination", "reasoning")))]
        config = dict(worker=dict(temperature=0.5, max_tokens=2048,
                                 backend_pool=dict(endpoints=[dict(endpoint_id="e1")])))
        for name in ("vlrb/r0", "vlrb/initial", "vlrb/final"):
            for cached in (False, True):
                with self.subTest(name=name, cached=cached), TemporaryDirectory() as temporary:
                    target = Path(temporary)
                    if cached:
                        method.write(target / f"{name}.json", value)
                    output = StringIO()
                    with patch.object(system, "evaluate", return_value=value) as evaluate, \
                            redirect_stdout(output):
                        result = method.evaluate(config, target, name, data, rubric,
                                                 vlrb_records=records)
                    self.assertIn(f"{name}: Strict ACC=66.67% (K=3)", output.getvalue())
                    self.assertNotIn("100.00%", output.getvalue())
                    self.assertEqual(result["metrics"], value["metrics"])
                    self.assertEqual(evaluate.call_count, 0 if cached else 1)
                    if not cached:
                        self.assertEqual(result["order_protocol"], vlrb.ORDER_PROTOCOL)
                        self.assertEqual(result["order_seed"], vlrb.SEED)
                        saved = method.load_json(target / f"{name}.json")
                        self.assertEqual(saved["order_protocol"], vlrb.ORDER_PROTOCOL)

    def test_vlrb_swaps_are_reproducible_independent_draws(self):
        records = [dict(sample_id=f"s{i:03d}") for i in range(100)]
        orders = vlrb._order_schedule(records)
        self.assertEqual(orders, vlrb._order_schedule(list(reversed(records))))
        self.assertEqual(set(orders.values()), set(product((0, 1), repeat=3)))

    def test_vlrb_votes_restore_original_answer_for_every_swap_sequence(self):
        data = rows(3)
        rubric = method.build_multicrit_open_ended_init_rubric()
        prefixes = ("vlfeedback", "RLHF", "mathverse")
        groups = ("general", "hallucination", "reasoning")
        for gold in (0, 1):
            for orders in product((0, 1), repeat=3):
                with self.subTest(gold=gold, orders=orders):
                    value = artifact(data, rubric, 3, k=3)
                    for sample in value["samples"]:
                        for index, order in enumerate(orders):
                            replicate = sample["replicates"][str(index)]
                            replicate["order"] = order
                            replicate["arbiter"]["parsed"]["answer"] = "A" if gold == order else "B"
                    value["metrics"] = system.metrics(value, data)
                    records = [dict(sample_id=row["sample_id"], benchmark_id=f"{prefixes[i]}_{i}",
                                    group=groups[i], preferred_original_index=gold)
                               for i, row in enumerate(data)]
                    result = vlrb.official_system_metrics(records, vlrb._votes(value))
                    self.assertEqual(result["original_index_predictions"], [gold] * 3)
                    self.assertEqual(result["correct_count"], 3)

    def test_vlrb_resume_does_not_reuse_a_different_swap_schedule(self):
        data = rows(3)
        rubric = method.build_multicrit_open_ended_init_rubric()
        value = artifact(data, rubric, 3, k=3)
        for sample in value["samples"]:
            sample["orders"] = [0, 0, 0]
        records = [dict(sample_id=row["sample_id"]) for row in data]
        orders = vlrb._order_schedule(records)
        with TemporaryDirectory() as temporary, patch.object(system, "evaluate") as evaluate:
            target = Path(temporary)
            method.write(target / "vlrb/initial.json", value)
            with self.assertRaisesRegex(ValueError, "swap schedule changed"):
                method.evaluate({}, target, "vlrb/initial", data, rubric,
                                orders=orders, vlrb_records=records)
            evaluate.assert_not_called()

    def test_formal_k3_majority_keeps_abstention(self):
        records = [dict(sample_id="a", benchmark_id="hallucination_a", group="hallucination",
                        preferred_original_index=0),
                   dict(sample_id="b", benchmark_id="mathverse_b", group="reasoning",
                        preferred_original_index=1),
                   dict(sample_id="c", benchmark_id="wildvision_c", group="general",
                        preferred_original_index=0)]
        result = vlrb.official_system_metrics(records, [[0, 1, 0], [None, 1, 0], [None, 1, None]])
        self.assertEqual(result["original_index_predictions"], [None, 1, 0])
        self.assertEqual(result["correct_count"], 2)
        self.assertEqual(result["coverage_count"], 2)
        self.assertEqual(result["overall_acc"], 1.0)

    def test_vlrb_images_use_shared_directory_for_new_runs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquet = root / "data" / "VL_RewardBench" / "data" / "test.parquet"
            target = root / "output" / "seed11" / "g5" / "vlrb"
            shared = root / "data" / "VL_RewardBench" / "dataset_images"
            self.assertEqual(vlrb._image_directory(target, parquet), shared)
            digest, image = vlrb._materialize_image(
                vlrb._image_directory(target, parquet), b"\x89PNG\r\n\x1a\nfixture")
            self.assertEqual(image.parent, shared)
            self.assertEqual(digest, hashlib.sha256(image.read_bytes()).hexdigest())
            legacy = target / "dataset_images"
            legacy.mkdir(parents=True)
            self.assertEqual(vlrb._image_directory(target, parquet), legacy)


if __name__ == "__main__":
    unittest.main()

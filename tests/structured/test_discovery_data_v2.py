import copy
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest import mock

from experiments.evolving_structured_rubrics import discovery_data_v2 as dv2
from experiments.evolving_structured_rubrics import discovery_data_v2_adjudication as adv2
from experiments.evolving_structured_rubrics import discovery_data_v2_demo as demo
from experiments.evolving_structured_rubrics import run_rubric_evolution as base
from experiments.evolving_structured_rubrics.experiment_utils import atomic_write_json, load_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/evolving_structured_rubrics/configs/rubric_evolution_phase5.example.json"


def source(adapter="rlhf_v", domain="visual"):
    return {
        "name": adapter, "source_family": adapter, "adapter": adapter,
        "repo_id": f"example/{adapter}", "split": "train", "revision": "main",
        "enabled": True, "domain": domain, "license": "test",
        "data_files": [], "image_archives": [],
    }


class FakeInfo:
    sha = "0123456789abcdef"
    siblings = ()


class FakeApi:
    def dataset_info(self, repo_id, revision, token=None):
        self.call = (repo_id, revision, token)
        return FakeInfo()


class DiscoveryDataV2Tests(unittest.TestCase):
    def test_discovery100_demo_selector_is_balanced_deterministic_and_disjoint(self):
        source_domains = {
            "rlhf_v": "visual", "mm_rlhf": "visual",
            "vilreward_73k": "reasoning", "mmpr_v1_2": "reasoning",
            "vision_arena_battle": "general", "mm_ifdpo": "general",
        }

        def row(source_name, role, index):
            identity = f"{role}-{source_name}-{index}"
            value = {"sample_id": identity, "source": source_name,
                "source_family": source_name, "source_sample_id": identity,
                "domain": source_domains[source_name], "subdomain": f"s{index % 4}",
                "question": f"question {identity}", "A": f"left {identity}",
                "B": f"right {identity}", "answer": "A",
                "image_path": f"images/{identity}.jpg",
                "image_sha256": f"image-{identity}",
                "question_sha256": f"question-{identity}",
                "unordered_pair_sha256": f"pair-{identity}"}
            if role == "hard":
                value["generic_screen"] = {"hardness": index % 4}
            return value

        coverage = [row(source_name, "coverage", index)
                    for source_name in source_domains for index in range(25)]
        hard = [row(source_name, "hard", index)
                for source_name in source_domains for index in range(15)]
        first = demo.select_demo_rows(coverage, hard, seed=42)
        second = demo.select_demo_rows(copy.deepcopy(coverage), copy.deepcopy(hard), seed=42)
        self.assertEqual([item["sample_id"] for item in first],
                         [item["sample_id"] for item in second])
        self.assertEqual(len(first), 100)
        self.assertEqual(Counter(item["demo_pool_role"] for item in first),
                         Counter({"coverage": 70, "hard": 30}))
        self.assertEqual(Counter(item["domain"] for item in first),
                         Counter(demo.EXPECTED_DOMAIN_COUNTS))
        self.assertEqual(Counter(item["preadjudication_assigned_order"]
                                 for item in first),
                         Counter({"original": 50, "swapped": 50}))
        self.assertFalse({item["sample_id"] for item in first
                          if item["demo_pool_role"] == "coverage"}
                         & {item["sample_id"] for item in first
                            if item["demo_pool_role"] == "hard"})
        for source_name in source_domains:
            chosen = [item["generic_screen"]["hardness"] for item in first
                      if item["source"] == source_name
                      and item["demo_pool_role"] == "hard"]
            self.assertEqual(chosen, sorted(chosen, reverse=True))

    def test_discovery100_single_order_mapping_and_reconciliation(self):
        parsed = {"answer": "B", "confidence": 4,
            "preference_rationale": "The displayed B is better.",
            "visual_evidence": ["visible"], "task_type": "visual_perception",
            "preference_dimensions": ["visual_grounding"],
            "evidence_type": "visual", "candidate_a_issues": ["wrong"],
            "candidate_b_issues": [], "ambiguity_flags": []}
        row = {"sample_id": "sample", "answer": "A",
               "preadjudication_assigned_order": "swapped"}
        merged = demo._merge_single_order(row, {
            "assigned_order": "swapped", "result": {"parsed": parsed}})
        pre = merged["preadjudication"]
        self.assertEqual(pre["suggested_answer"], "A")
        self.assertTrue(pre["source_answer_agreement"])
        self.assertFalse(pre["requires_reconciliation"])
        uncertain = copy.deepcopy(parsed)
        uncertain.update({"answer": "uncertain", "confidence": 2})
        pre = demo._merge_single_order(row, {
            "assigned_order": "swapped", "result": {"parsed": uncertain}
        })["preadjudication"]
        self.assertIsNone(pre["suggested_answer"])
        self.assertEqual(pre["reconciliation_reasons"], ["uncertain", "low_confidence"])

    def test_discovery100_source_label_export_balances_without_changing_preference(self):
        rows = []
        for index, answer in enumerate(("A", "A", "A", "B")):
            rows.append({"sample_id": f"sample-{index}", "image_sha256": f"i-{index}",
                "question": f"q-{index}", "A": f"a-{index}", "B": f"b-{index}",
                "answer": answer, "source": "rlhf_v", "source_family": "rlhf_v",
                "source_sample_id": f"source-{index}", "domain": "visual",
                "subdomain": "test", "question_sha256": f"qsha-{index}",
                "unordered_pair_sha256": f"pair-{index}"})
        image_paths = {f"i-{index}": f"data/images/i-{index}.png"
                       for index in range(4)}
        exported = demo._balanced_export_rows(
            rows, seed=42, split_name="discovery", image_paths=image_paths)
        self.assertEqual(Counter(row["answer"] for row in exported),
                         Counter({"A": 2, "B": 2}))
        original = {row["sample_id"]: row for row in rows}
        for row in exported:
            source_row = original[row["sample_id"]]
            preferred_text = source_row[source_row["answer"]]
            self.assertEqual(row[row["answer"]], preferred_text)
            self.assertEqual(row["label_origin"], "source_label_exploratory")

    def test_discovery100_final_orientation_maps_candidate_issues(self):
        row = {"sample_id": "sample", "image_path": "image.jpg", "question": "q",
            "A": "original A", "B": "original B", "domain": "visual",
            "subdomain": "test", "source": "rlhf_v", "source_family": "rlhf_v",
            "source_sample_id": "source", "demo_pool_role": "coverage",
            "image_sha256": "image", "question_sha256": "question",
            "unordered_pair_sha256": "pair"}
        review = {"confidence": 4, "preference_rationale": "clear",
            "visual_evidence": [], "task_type": "visual_perception",
            "preference_dimensions": ["visual_grounding"], "evidence_type": "visual",
            # Human saw swapped display: display A=original B, display B=original A.
            "candidate_a_issues": ["original B issue"],
            "candidate_b_issues": ["original A issue"], "ambiguity_flags": [],
            "decision": "accept", "reconciled": False}
        result = demo._orient_final(
            row, "A", target_answer="B", review=review, display_swapped=True)
        self.assertEqual((result["A"], result["B"], result["answer"]),
                         ("original B", "original A", "B"))
        self.assertEqual(result["candidate_a_issues"], ["original B issue"])
        self.assertEqual(result["candidate_b_issues"], ["original A issue"])

    def test_discovery100_finalize_requires_reviews_and_balances_gold(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            target = output / "discovery_data_v2"
            review_dir = target / demo.DIRECTORY / "review"
            frozen, queue, hidden = [], [], []
            for index in range(100):
                sample_id = f"sample-{index}"
                frozen.append({"sample_id": sample_id, "image_path": f"image-{index}.jpg",
                    "question": f"question {index}", "A": f"A {index}", "B": f"B {index}",
                    "answer": "A", "domain": "visual" if index < 34 else (
                        "reasoning" if index < 67 else "general"), "subdomain": "test",
                    "source": "rlhf_v", "source_family": "rlhf_v",
                    "source_sample_id": f"source-{index}",
                    "demo_pool_role": "coverage" if index < 70 else "hard",
                    "image_sha256": f"image-{index}",
                    "question_sha256": f"question-{index}",
                    "unordered_pair_sha256": f"pair-{index}"})
                swapped = index % 2 == 1
                review_id = f"review-{index}"
                queue.append({"review_id": review_id, "human_review": {
                    "answer": "B" if swapped else "A", "confidence": 4,
                    "preference_rationale": "The preferred response is clearly better.",
                    "visual_evidence": [], "task_type": "visual_perception",
                    "preference_dimensions": ["visual_grounding"],
                    "evidence_type": "visual", "candidate_a_issues": [],
                    "candidate_b_issues": [], "ambiguity_flags": [],
                    "decision": "accept", "reviewed": True, "reconciled": False,
                    "notes": ""}})
                hidden.append({"review_id": review_id, "sample_id": sample_id,
                    "display_swapped": swapped, "source_answer": "A",
                    "preadjudication": {"suggested_answer": "A",
                        "requires_reconciliation": False}})
            dv2._write_jsonl(review_dir / "blind_review_queue.jsonl", queue)
            dv2._write_jsonl(review_dir / "hidden_reference.jsonl", hidden)
            cfg = {"seed": 42}
            with mock.patch.object(demo, "_load_frozen",
                    return_value=(cfg, target, {}, frozen)), \
                    mock.patch.object(dv2, "_require"):
                demo.finalize({}, output)
            final = dv2._read_jsonl(
                target / demo.DIRECTORY / "final/discovery_100.jsonl")
            self.assertEqual(len(final), 100)
            self.assertEqual(Counter(row["answer"] for row in final),
                             Counter({"A": 50, "B": 50}))
            self.assertEqual(Counter(row["demo_pool_role"] for row in final),
                             Counter({"coverage": 70, "hard": 30}))

    def test_preadjudication_schema_is_criterion_agnostic_and_strict(self):
        value = {
            "answer": "A", "confidence": 4,
            "preference_rationale": "A is better grounded in the image.",
            "visual_evidence": ["A visible red object supports A."],
            "task_type": "visual_perception",
            "preference_dimensions": ["visual_grounding", "factual_correctness"],
            "evidence_type": "visual", "candidate_a_issues": [],
            "candidate_b_issues": ["Unsupported object."], "ambiguity_flags": [],
        }
        self.assertEqual(adv2.validate_response(value)["answer"], "A")
        uncertain = copy.deepcopy(value)
        uncertain.update({"answer": "uncertain", "confidence": 1,
                          "evidence_type": "insufficient",
                          "ambiguity_flags": ["image_unreadable"]})
        self.assertEqual(adv2.validate_response(uncertain)["answer"], "uncertain")
        root_aware = {**value, "applicable_root_ids": []}
        with self.assertRaisesRegex(ValueError, "fields"):
            adv2.validate_response(root_aware)
        invalid = {**value, "task_type": "factuality_root"}
        with self.assertRaisesRegex(ValueError, "task_type"):
            adv2.validate_response(invalid)

    def test_preadjudication_double_order_merge(self):
        base_value = {
            "answer": "A", "confidence": 4, "preference_rationale": "better",
            "visual_evidence": ["visible"], "task_type": "reasoning",
            "preference_dimensions": ["reasoning_validity"],
            "evidence_type": "mixed", "candidate_a_issues": [],
            "candidate_b_issues": ["wrong"], "ambiguity_flags": [],
        }
        swapped = copy.deepcopy(base_value)
        swapped.update({"answer": "B", "confidence": 3,
                        "candidate_a_issues": ["wrong"], "candidate_b_issues": []})
        row = {"sample_id": "sample", "answer": "A"}
        judgment = {"original": {"parsed": base_value},
                    "swapped": {"parsed": swapped}}
        merged = adv2.merge_orders(row, judgment)["preadjudication"]
        self.assertTrue(merged["answer_agreement"])
        self.assertEqual(merged["suggested_answer"], "A")
        self.assertEqual(merged["minimum_confidence"], 3)
        uncertain = copy.deepcopy(base_value)
        uncertain["answer"] = "uncertain"
        unresolved = adv2.merge_orders(row, {
            "original": {"parsed": uncertain},
            "swapped": {"parsed": uncertain}})["preadjudication"]
        self.assertTrue(unresolved["answer_agreement"])
        self.assertIsNone(unresolved["suggested_answer"])
        self.assertTrue(unresolved["requires_reconciliation"])

    def test_preadjudication_smoke_and_blind_display_are_balanced(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = config["discovery_data_v2"]
        source_names = [item["name"] for item in cfg["sources"]]
        rows = []
        for role in adv2.ROLE_PATHS:
            for source_name in source_names:
                rows.append({"sample_id": f"{role}-{source_name}",
                    "source": source_name, "_adjudication_role": role})
        smoke = adv2._smoke_rows(rows, cfg)
        self.assertEqual(len(smoke), 18)
        self.assertEqual(Counter(row["source"] for row in smoke),
                         Counter({name: 3 for name in source_names}))
        plan_rows = [{"sample_id": f"sample-{index}"} for index in range(390)]
        plan1 = adv2._display_plan(plan_rows, 42)
        plan2 = adv2._display_plan(copy.deepcopy(plan_rows), 42)
        self.assertEqual(plan1, plan2)
        self.assertEqual(sum(plan1.values()), 195)
        self.assertEqual(adv2.EFFECTIVE_REQUEST_CONCURRENCY, 5)

    def test_preadjudication_prompt_excludes_frozen_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            from PIL import Image

            image = Path(temporary) / "image.jpg"
            Image.new("RGB", (64, 64), "white").save(image)
            row = {"image_path": str(image), "question": "Question visible",
                   "A": "Answer left", "B": "Answer right",
                   "source": "SECRET_SOURCE", "answer": "B",
                   "_adjudication_role": "SECRET_ROLE",
                   "generic_screen": {"hardness": 3}}
            content = adv2._user_content(row, swapped=False, retry=False)
            serialized = json.dumps(content)
            self.assertIn("Question visible", serialized)
            self.assertNotIn("SECRET_SOURCE", serialized)
            self.assertNotIn("SECRET_ROLE", serialized)
            self.assertNotIn("hardness", serialized)

    def test_preadjudication_repairs_qwen3_vl_tiny_image_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            import base64
            import io

            from PIL import Image

            path = Path(temporary) / "banner.png"
            Image.new("RGB", (204, 22), "white").save(path)
            original = path.read_bytes()
            row = {"sample_id": "tiny", "image_path": str(path),
                   "image_sha256": "source-image-sha", "question": "q",
                   "A": "a", "B": "b"}
            data_url = adv2._qwen3_vl_image_data_url(row)
            payload = base64.b64decode(data_url.split(",", 1)[1])
            with Image.open(io.BytesIO(payload)) as repaired:
                self.assertGreaterEqual(min(repaired.size), 28)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(adv2._qwen3_vl_image_repair(row),
                             adv2.QWEN3_VL_IMAGE_REPAIR_VERSION)

            large = Path(temporary) / "large.png"
            Image.new("RGB", (64, 64), "white").save(large)
            large_row = {**row, "sample_id": "large", "image_path": str(large)}
            self.assertIsNone(adv2._qwen3_vl_image_repair(large_row))
            expected = dv2._sha({"sample_id": "large",
                "image_sha256": "source-image-sha", "question": "q",
                "A": "a", "B": "b", "swapped": False})
            self.assertEqual(adv2._input_fingerprint(large_row, swapped=False), expected)

    def test_revision_lock_and_visionarena_auth(self):
        api = FakeApi()
        value = dv2.resolve_source_revision(source(), api=api)
        self.assertEqual(value["resolved_revision"], FakeInfo.sha)
        arena = source("vision_arena", "general")
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            dv2.resolve_source_revision(arena, api=api, token=None)

    def test_source_adapters_and_invalid_paths(self):
        rlhf = dv2.normalize_source_row("rlhf_v", {
            "id": "x", "image": b"image", "text": {
                "question": "What?", "chosen": "Good", "rejected": "Bad"}},
            source=source())
        self.assertEqual((rlhf["A"], rlhf["B"], rlhf["answer"]), ("Good", "Bad", "A"))
        rlhf_string = dv2.normalize_source_row("rlhf_v", {
            "idx": 2, "image": b"image",
            "text": json.dumps({"question": "Q", "chosen": "C", "rejected": "R"})},
            source=source())
        self.assertEqual((rlhf_string["question"], rlhf_string["A"]), ("Q", "C"))

        arena_source = source("vision_arena", "general")
        arena = dv2.normalize_source_row("vision_arena", {
            "id": "a", "language": "English", "images": [b"image"],
            "conversation_a": [{"content": "Question"}, {"content": "Left"}],
            "conversation_b": [{"content": "Question"}, {"content": "Right"}],
            "winner": "model_b"}, source=arena_source)
        self.assertEqual((arena["question"], arena["answer"]), ("Question", "B"))
        with self.assertRaisesRegex(ValueError, "tie"):
            dv2.normalize_source_row("vision_arena", {
                "id": "a", "language": "English", "images": [b"image"],
                "conversation_a": ["q", "a"], "conversation_b": ["q", "b"],
                "winner": "tie"}, source=arena_source)

        vil_source = source("vilreward", "reasoning")
        rows = [dv2.normalize_source_row("vilreward", {
            "index": index, "image_path": "same.jpg", "question": "Solve",
            "process": process, "value": value}, source=vil_source)
            for index, process, value in ((1, "bad", .1), (2, "good", .9))]
        grouped = dv2.group_vilreward(rows, min_value_gap=.25)
        self.assertEqual((len(grouped), grouped[0]["A"], grouped[0]["B"]), (1, "good", "bad"))

        for adapter, domain in (("mm_rlhf", "visual"), ("mmpr", "reasoning"),
                                ("mm_ifdpo", "general")):
            normalized = dv2.normalize_source_row(adapter, {
                "id": adapter, "image": b"image", "question": "Question",
                "chosen": [{"content": "preferred"}],
                "rejected": [{"content": "rejected"}]},
                source=source(adapter, domain))
            self.assertEqual((normalized["A"], normalized["B"]),
                             ("preferred", "rejected"))
            with self.assertRaisesRegex(ValueError, "lacks"):
                dv2.normalize_source_row(adapter, {"id": "broken"},
                                         source=source(adapter, domain))

    def test_dev_partition_is_model_free_and_disjoint(self):
        cfg = {"seed": 42, "candidate_limits": {
            "dev_candidates_per_domain": 3, "screen_candidates_per_domain": 4}}
        rows = []
        for domain in dv2.DOMAINS:
            for index in range(12):
                rows.append({"domain": domain, "source_family": f"f{index % 2}",
                    "image_sha256": f"{domain}-i-{index}",
                    "question_sha256": f"{domain}-q-{index}"})
        dev, discovery = dv2._partition_dev(rows, cfg)
        self.assertEqual((len(dev), len(discovery)), (9, 12))
        self.assertFalse({id(row) for row in dev} & {id(row) for row in discovery})
        self.assertTrue(all(row["candidate_role"] == "dev" for row in dev))
        self.assertTrue(all("worker_screen" not in row for row in dev))
        self.assertFalse({row["image_sha256"] for row in dev}
                         & {row["image_sha256"] for row in discovery})
        self.assertFalse({row["question_sha256"] for row in dev}
                         & {row["question_sha256"] for row in discovery})

    def test_benchmark_exclusion_supports_query_response_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark.jsonl"
            path.write_text(json.dumps({"id": "bench-1", "query": "What?",
                "response": ["A response", "B response"]}) + "\n", encoding="utf-8")
            value = dv2._benchmark_exclusion(path)
            self.assertIn("bench-1", value["sample_ids"])
            self.assertIn(dv2._question_sha("What?"), value["question_sha256"])
            pair = dv2._pair_sha("A response", "B response")
            self.assertIn(pair, value["unordered_pair_sha256"])

    def test_dedup_keeps_different_images_with_same_question(self):
        rows = []
        for index, image in enumerate(("image-1", "image-2")):
            rows.append({"source": "rlhf_v", "source_family": "rlhf_v",
                "source_sample_id": f"sample-{index}", "image_sha256": image,
                "question_sha256": "same-question", "unordered_pair_sha256": f"pair-{index}",
                "A": f"A {index}", "B": f"B {index}"})
        accepted, stats = dv2._deduplicate(rows, {
            "sample_ids": set(), "image_sha256": set(),
            "image_question_sha256": set(), "question_sha256": set(),
            "unordered_pair_sha256": set()})
        self.assertEqual(len(accepted), 2)
        self.assertEqual(stats, {})

    def test_dedup_rejects_same_image_question(self):
        rows = [{"source": "rlhf_v", "source_family": "rlhf_v",
                 "source_sample_id": f"sample-{index}", "image_sha256": "image-1",
                 "question_sha256": "same-question", "unordered_pair_sha256": f"pair-{index}",
                 "A": "A response with visible evidence", "B": "B response"}
                for index in range(2)]
        accepted, stats = dv2._deduplicate(rows, {
            "sample_ids": set(), "image_sha256": set(),
            "image_question_sha256": set(), "question_sha256": set(),
            "unordered_pair_sha256": set()})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(stats["excluded_image_question_sha256"], 1)

    def test_dev_screen_artifact_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            path = target / "screen/predictions/example.json"
            path.parent.mkdir(parents=True)
            atomic_write_json(path, {"sample_ids": ["dev-1::swap"]})
            with self.assertRaisesRegex(RuntimeError, "leaked"):
                dv2._assert_dev_not_screened(target, [{"sample_id": "dev-1"}])

    def test_hardness_components_are_bounded(self):
        self.assertEqual(dv2._token_jaccard("red cat", "red cat"), 1.0)
        self.assertLess(dv2._token_jaccard("red cat", "blue dog"), .1)
        self.assertEqual(dv2._pair_sha("A text", "B text"),
                         dv2._pair_sha("B text", "A text"))

    def test_screen_vote_inversion_accepts_abstain(self):
        self.assertEqual(dv2._invert_screen_vote("A"), "B")
        self.assertEqual(dv2._invert_screen_vote("B"), "A")
        self.assertEqual(dv2._invert_screen_vote("abstain"), "abstain")
        with self.assertRaises(ValueError):
            dv2._invert_screen_vote("unexpected")

    def test_generic_screen_parser_requires_exact_schema(self):
        raw = json.dumps({"analysis_a": "A is grounded.",
                          "analysis_b": "B invents a detail.",
                          "thought": "A is better overall.", "answer": "A"})
        self.assertEqual(dv2._parse_generic_response(raw)["answer"], "A")
        none_raw = json.dumps({"analysis_a": "same", "analysis_b": "same",
                               "thought": "indistinguishable", "answer": "None"})
        self.assertEqual(dv2._parse_generic_response(none_raw)["answer"], "None")
        with self.assertRaisesRegex(ValueError, "fields"):
            dv2._parse_generic_response(json.dumps({"answer": "A"}))
        with self.assertRaisesRegex(ValueError, "A, B, or None"):
            dv2._parse_generic_response(json.dumps({
                "analysis_a": "a", "analysis_b": "b", "thought": "t",
                "answer": "tie"}))

    def test_generic_hardness_uses_both_orders_and_position_inconsistency(self):
        rows = [{"sample_id": "easy", "source": "s", "answer": "A"},
                {"sample_id": "hard", "source": "s", "answer": "A"},
                {"sample_id": "none", "source": "s", "answer": "B"}]

        def result(answer):
            return {"parse_ok": True, "answer": answer, "attempt_count": 1}

        judgments = [
            {"sample_id": "easy", "original": result("A"),
             "swapped": result("B")},
            {"sample_id": "hard", "original": result("B"),
             "swapped": result("A")},
            {"sample_id": "none", "original": result("None"),
             "swapped": result("None")},
        ]
        scored = {row["sample_id"]: row for row in
                  dv2._score_generic_judgments(rows, judgments)}
        self.assertEqual(scored["easy"]["generic_screen"]["hardness"], 0)
        self.assertEqual(scored["hard"]["generic_screen"]["hardness"], 2)
        self.assertEqual(scored["none"]["generic_screen"]["hardness"], 2)

        broken = copy.deepcopy(judgments)
        broken[0]["original"]["parse_ok"] = False
        with self.assertRaisesRegex(RuntimeError, "technical failure"):
            dv2._score_generic_judgments(rows, broken)

    def test_generic_hard_selector_is_source_balanced_and_excludes_coverage(self):
        sources = ["source_a", "source_b"]
        scored = []
        for source_name in sources:
            for index in range(25):
                scored.append({"sample_id": f"{source_name}-{index}",
                    "source": source_name,
                    "generic_screen": {"hardness": 3 if index < 20 else 0}})
        coverage_ids = {f"{source_name}-{index}"
                        for source_name in sources for index in range(5)}
        selected = dv2._select_hard_candidates(
            scored, coverage_ids=coverage_ids, source_names=sources,
            per_source=15, seed=42)
        self.assertEqual(len(selected), 30)
        self.assertEqual(Counter(row["source"] for row in selected),
                         Counter({"source_a": 15, "source_b": 15}))
        self.assertFalse(coverage_ids & {row["sample_id"] for row in selected})

    def test_adjudication_parser_requires_schema(self):
        value = dv2._extract_object("prefix```json\n{\"answer\":\"A\"}\n```suffix")
        self.assertEqual(value["answer"], "A")
        with self.assertRaisesRegex(ValueError, "schema"):
            dv2._validate_adjudication(value)

    def test_adjudication_item_cache_resumes_without_model_call(self):
        row = {"sample_id": "x", "image_sha256": "i", "question": "q",
               "A": "a", "B": "b"}
        record = {**row, "adjudication": {"model_disagreement": False}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "item.json"
            with mock.patch.object(dv2, "_adjudicate_one", return_value=record) as call:
                first, reused = dv2._adjudicate_cached(object(), {}, row, path)
                self.assertFalse(reused)
                second, reused = dv2._adjudicate_cached(object(), {}, row, path)
                self.assertTrue(reused)
                self.assertEqual(first, second)
                self.assertEqual(call.call_count, 1)

    def test_archive_member_lookup_and_safe_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("nested/example.jpg", b"image-payload")
            self.assertEqual(dv2._archive_image_bytes(archive, "nested/example.jpg"),
                             b"image-payload")
            extracted = dv2._safe_extract_zip(archive, root / "expanded")
            self.assertEqual(extracted[0].read_bytes(), b"image-payload")
            unsafe = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as handle:
                handle.writestr("../escape.txt", b"bad")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                dv2._safe_extract_zip(unsafe, root / "safe")

    def test_locked_hf_stream_retries_only_failed_parquet_shard(self):
        lock = {"resolved_revision": "pinned-sha", "siblings": [
            {"path": "data/train-00001.parquet"},
            {"path": "README.md"},
            {"path": "data/train-00000.parquet"},
        ]}
        calls = []
        failed_once = {"value": False}

        def load_dataset(repo_id, **kwargs):
            shard = kwargs["data_files"][0]
            calls.append((repo_id, shard, kwargs["revision"], kwargs["streaming"]))
            if shard.endswith("00001.parquet") and not failed_once["value"]:
                failed_once["value"] = True

                def broken():
                    yield {"id": "already-delivered"}
                    raise RuntimeError("Cannot send a request, as the client has been closed.")

                return broken()
            return iter([{"id": shard}])

        with mock.patch.object(dv2, "_reset_hf_http_client") as reset, \
                mock.patch.object(dv2.time, "sleep"):
            rows = list(dv2._iter_locked_hf_stream(
                source("vision_arena", "general"), lock,
                load_dataset=load_dataset, max_attempts=3))
        self.assertEqual([row["id"] for row in rows], [
            "data/train-00000.parquet", "already-delivered"])
        self.assertEqual([call[1] for call in calls], [
            "data/train-00000.parquet", "data/train-00001.parquet",
            "data/train-00001.parquet"])
        self.assertEqual(reset.call_count, 1)

    def test_phase5_hash_ignores_new_protocol_block(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        without = copy.deepcopy(config)
        without.pop("discovery_data_v2")
        self.assertEqual(base._phase5_config_view(config), base._phase5_config_view(without))

    def test_metadata_selector_is_source_balanced_and_worker_independent(self):
        sources = ["rlhf_v", "mm_rlhf", "vilreward_73k", "mmpr_v1_2",
                   "vision_arena_battle", "mm_ifdpo"]
        rows = []
        for source_name in sources:
            for index in range(30):
                rows.append({"sample_id": f"{source_name}-{index}",
                    "source": source_name, "source_family": source_name,
                    "source_sample_id": f"source-{source_name}-{index}",
                    "domain": dv2.DOMAINS[index % 3],
                    "subdomain": f"sub-{index % 4}",
                    "question": "question " + "x" * (index * 7),
                    "A": "answer A " + "a" * (index * 13),
                    "B": "answer B " + "b" * (index * 5),
                    "image_sha256": f"image-{source_name}-{index}",
                    "question_sha256": f"question-{source_name}-{index}",
                    "unordered_pair_sha256": f"pair-{source_name}-{index}"})
        first = dv2._select_metadata_balanced(
            rows, source_names=sources, seed=42, per_source=25)
        second = dv2._select_metadata_balanced(
            copy.deepcopy(rows), source_names=sources, seed=42, per_source=25)
        self.assertEqual([row["sample_id"] for row in first],
                         [row["sample_id"] for row in second])
        self.assertEqual(len(first), 150)
        self.assertEqual({source_name: sum(row["source"] == source_name for row in first)
                          for source_name in sources},
                         {source_name: 25 for source_name in sources})
        contaminated = copy.deepcopy(rows)
        contaminated[0]["worker_screen"] = {"hardness": 3}
        with self.assertRaisesRegex(RuntimeError, "forbidden model-derived"):
            dv2._select_metadata_balanced(
                contaminated, source_names=sources, seed=42, per_source=25)

    def test_selection_freeze_writes_deterministic_isolated_artifacts(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        sources = [source["name"] for source in config["discovery_data_v2"]["sources"]]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            target = output / "discovery_data_v2"
            candidates = target / "candidates"
            candidates.mkdir(parents=True)
            atomic_write_json(target / "stage_status.json", {
                "schema_version": "1.0.0",
                "discovery-v2-ingest": {"status": "passed"}})

            def build(role, source_name, index):
                identity = f"{role}-{source_name}-{index}"
                return {"sample_id": identity, "candidate_role": role,
                    "source": source_name, "source_family": source_name,
                    "source_sample_id": f"source-{identity}",
                    "domain": dv2.DOMAINS[sources.index(source_name) // 2],
                    "subdomain": f"sub-{index % 3}",
                    "question": f"question {identity}",
                    "A": f"answer A {identity}", "B": f"answer B {identity}",
                    "answer": "A", "image_path": f"images/{identity}.png",
                    "image_sha256": f"image-{identity}",
                    "question_sha256": f"question-{identity}",
                    "unordered_pair_sha256": f"pair-{identity}"}

            dev = [build("dev", source_name, index)
                   for source_name in sources for index in range(50)]
            discovery = [build("discovery", source_name, index)
                         for source_name in sources for index in range(150)]
            dv2._write_jsonl(candidates / "dev_candidates.jsonl", dev)
            dv2._write_jsonl(candidates / "discovery_candidates.jsonl", discovery)
            dv2.selection_freeze(config, output)
            manifest1 = load_json(target / "selection_v2/selection_manifest.json")
            dv2.selection_freeze(config, output)
            manifest2 = load_json(target / "selection_v2/selection_manifest.json")
            self.assertEqual(manifest1, manifest2)
            self.assertFalse(manifest1["worker_inputs_read"])
            self.assertEqual(manifest1["selection_model_calls"], 0)
            self.assertFalse(any(manifest1["cross_split_overlap"].values()))
            selected_dev = dv2._read_jsonl(target / "selection_v2/dev_150.jsonl")
            selected_coverage = dv2._read_jsonl(
                target / "selection_v2/coverage_candidates_150.jsonl")
            self.assertEqual((len(selected_dev), len(selected_coverage)), (150, 150))
            self.assertFalse(any(dv2._selection_overlap(
                selected_dev, selected_coverage).values()))

    def test_production_finalize_exact_quotas_and_determinism(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            target = output / "discovery_data_v2"
            (target / "review").mkdir(parents=True)
            atomic_write_json(target / "source_lock.json", {"lock_sha256": "lock",
                "sources": [{"name": "test", "repo_id": "test/repo",
                    "resolved_revision": "sha", "remote_license": "test",
                    "license": "test", "disabled": False}]})
            atomic_write_json(target / "stage_status.json", {
                "schema_version": "1.0.0",
                "discovery-v2-review-export": {"status": "passed"}})
            reference, decisions = {}, []
            sources = ["rlhf_v", "mm_rlhf", "vilreward_73k", "mmpr_v1_2",
                       "vision_arena_battle", "mm_ifdpo"]
            families = ["rlhf_v", "mm_rlhf", "vilreward", "mmpr",
                        "vision_arena", "mm_ifdpo"]
            index = 0

            def add(role, domain, boundary_type):
                nonlocal index
                source_index = index % len(sources)
                image = target / "images" / f"{index}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes((f"fake-image-{index}" * 8).encode())
                sample = f"sample-{index}"
                review_id = f"review-{index}"
                row = {"sample_id": sample, "image_path": str(image),
                    "question": f"question {index}", "A": f"answer A {index}",
                    "B": f"answer B {index}", "answer": "A",
                    "candidate_role": role, "domain": domain, "subdomain": "test",
                    "source": sources[source_index], "source_family": families[source_index],
                    "source_sample_id": f"source-{index}",
                    "image_sha256": f"image-{index}", "question_sha256": f"q-{index}",
                    "unordered_pair_sha256": f"pair-{index}",
                    "adjudication": {"model_disagreement": False,
                        "suggested_answer": "A"}}
                reference[review_id] = row
                decisions.append({"review_id": review_id, "answer": "A", "confidence": 4,
                    "preference_rationale": "Clear global preference.",
                    "visual_evidence": ["visible evidence"],
                    "primary_error_type": "content_error", "secondary_error_types": [],
                    "boundary_type": boundary_type, "applicable_root_ids": [],
                    "decision": "accept", "reviewed": True,
                    "reconciled": False, "reconciliation_notes": ""})
                index += 1

            for domain in dv2.DOMAINS:
                for _ in range(25):
                    add("discovery", domain, None)
            for boundary_type, count in dv2.DEFAULT_QUOTAS["boundary"].items():
                for offset in range(count):
                    add("discovery", dv2.DOMAINS[offset % 3], boundary_type)
            for domain in dv2.DOMAINS:
                for _ in range(40):
                    add("dev", domain, None)
                for offset in range(10):
                    add("dev", domain, dv2.BOUNDARY_TYPES[offset % 4])
            atomic_write_json(target / "review/hidden_reference.json", reference)
            dv2._write_jsonl(target / "review/human_review_decisions.jsonl", decisions)
            dv2.finalize(config, output)
            manifest1 = load_json(target / "final/dataset_manifest.json")
            dv2.finalize(config, output)
            manifest2 = load_json(target / "final/dataset_manifest.json")
            self.assertEqual(manifest1, manifest2)
            self.assertEqual(manifest1["artifacts"]["d_evolve_75.jsonl"]["count"], 75)
            self.assertEqual(manifest1["artifacts"]["d_boundary_25.jsonl"]["count"], 25)
            self.assertEqual(manifest1["artifacts"]["dev_150.jsonl"]["count"], 150)
            discovery = dv2._read_jsonl(target / "final/discovery_100.jsonl")
            dev = dv2._read_jsonl(target / "final/dev_150.jsonl")
            self.assertLessEqual(abs(sum(r["answer"] == "A" for r in discovery) - 50), 1)
            dv2._assert_isolation(discovery, dev)


if __name__ == "__main__":
    unittest.main()

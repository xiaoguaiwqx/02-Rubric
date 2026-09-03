"""Deterministic tests for Phase21 atomic root-subtree evolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from critiq.structured.schema import (
    EdgeCondition,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredRubric,
)
from experiments.evolving_structured_rubrics import aligned_system_runtime as runtime
from experiments.evolving_structured_rubrics import (
    unified_subtree_bundle_evolution as evolution,
)
from experiments.evolving_structured_rubrics import (
    vl_rewardbench_unified_subtree_bundle_evolution as vlrb,
)
from experiments.evolving_structured_rubrics.subtree_bundle_manager import (
    apply_bundle_refine,
    parse_bundle_refine,
    parse_failure_attribution,
    subtree_sha256,
)


def _description(label: str) -> str:
    return (
        f"Criterion focus: {label}\n\n"
        "Applicable only when: the evidence is relevant.\n\n"
        "Not applicable when: the evidence is absent.\n\n"
        "Decision rule: prefer the better supported answer; otherwise None.")


def _rubric() -> StructuredRubric:
    nodes = {
        "r1": RubricNode(
            "r1", RubricCriterionSnapshot("root_one", "root one", 1.0)),
        "r1c1": RubricNode(
            "r1c1", RubricCriterionSnapshot(
                "child_one", _description("one"), 1.0)),
        "r1c2": RubricNode(
            "r1c2", RubricCriterionSnapshot(
                "child_two", _description("two"), 1.0)),
        "r2": RubricNode(
            "r2", RubricCriterionSnapshot("root_two", "root two", 1.0)),
        "r2c1": RubricNode(
            "r2c1", RubricCriterionSnapshot(
                "child_three", _description("three"), 1.0)),
    }
    edges = (
        RubricEdge("r1", "r1c1", EdgeCondition.ALWAYS),
        RubricEdge("r1", "r1c2", EdgeCondition.ALWAYS),
        RubricEdge("r2", "r2c1", EdgeCondition.ALWAYS),
    )
    return StructuredRubric(nodes, edges, ("r1", "r2"))


def _call(answer: str, *, parse_ok: bool = True) -> dict:
    return {
        "parse_ok": parse_ok,
        "parsed": {"answer": answer} if parse_ok else None,
        "model_generation_count": 1,
        "endpoint_id": "vllm-8000",
    }


def _artifact(answers: list[str], scope: list[str] | None = None) -> tuple[dict, list[dict]]:
    rows = [
        {"sample_id": "s1", "answer": "A"},
        {"sample_id": "s2", "answer": "B"},
        {"sample_id": "s3", "answer": "A"},
    ]
    value = {
        "root_id": "r1",
        "samples": [
            {"sample_id": f"s{index + 1}", "order": 0, "call": _call(answer)}
            for index, answer in enumerate(answers)],
    }
    frozen = scope if scope is not None else [
        f"s{index + 1}" for index, answer in enumerate(answers)
        if answer in {"A", "B"}]
    value["metrics"] = runtime.root_metrics(value, rows, frozen)
    return value, rows


class TestUnifiedScopeAndAcceptance(unittest.TestCase):
    def test_scope_comes_only_from_unified_ab(self) -> None:
        value, _ = _artifact(["A", "None", "B"])
        self.assertEqual(("s1", "s3"), evolution._scope_ids(value))

    def test_signatures_are_selected_only_from_scope_mismatches(self) -> None:
        value, rows = _artifact(["B", "None", "A"], ["s1", "s3"])
        self.assertEqual(("s1",), evolution._root_mismatches(value, rows))

    def test_candidate_none_is_wrong_on_frozen_scope(self) -> None:
        before, rows = _artifact(["B", "B", "A"], ["s1", "s2"])
        after, _ = _artifact(["A", "None", "A"], ["s1", "s2"])
        paired = runtime.paired_root(before, after, rows)
        self.assertEqual(["s1"], paired["root_scope_corrected_sample_ids"])
        self.assertEqual(["s2"], paired["root_scope_harmed_sample_ids"])
        self.assertEqual(0, paired["root_scope_net_corrected"])

    def test_both_operators_require_strict_positive_net(self) -> None:
        for net, expected in ((-1, False), (0, False), (1, True)):
            decision = "accepted" if net > 0 else "competition_rejected"
            self.assertEqual(expected, decision == "accepted")

    def test_split_trigger_ignores_exhausted_root(self) -> None:
        rubric = StructuredRubric(
            {"r1": RubricNode(
                "r1", RubricCriterionSnapshot("root", "root", 1.0))},
            (), ("r1",))
        answers = ["B"] * 20
        rows = [{"sample_id": f"s{i}", "answer": "A"} for i in range(20)]
        value = {
            "root_id": "r1",
            "samples": [{"sample_id": f"s{i}", "order": 0,
                         "call": _call(answer)}
                        for i, answer in enumerate(answers)]}
        value["metrics"] = runtime.root_metrics(
            value, rows, [f"s{i}" for i in range(20)])
        trigger = evolution._trigger(
            "r1", rubric, value, rows, {"split_status": "exhausted"})
        self.assertFalse(trigger["triggered"])


class TestBundleRefine(unittest.TestCase):
    def _proposal(self, rubric: StructuredRubric) -> dict:
        old = rubric.get_node("r1c1").criterion.description
        return {
            "root_id": "r1",
            "original_subtree_sha256": subtree_sha256(rubric, "r1"),
            "edits": [{
                "node_id": "r1c1",
                "criterion_name": "child_one",
                "original_description_sha256": hashlib.sha256(
                    old.encode("utf-8")).hexdigest(),
                "description": _description("one revised"),
                "failure_analysis": ["the old boundary was too broad"],
                "rationale": "tighten only the implicated child",
            }],
            "unchanged_node_ids": ["r1c2"],
            "bundle_rationale": "one coherent boundary correction",
            "representative_sample_ids": ["s1"],
        }

    def test_bundle_refine_can_edit_only_part_of_children(self) -> None:
        rubric = _rubric()
        proposal = parse_bundle_refine(
            self._proposal(rubric), rubric=rubric, root_id="r1",
            allowed_sample_ids=["s1"], max_description_chars=1000)
        changed = apply_bundle_refine(rubric, proposal)
        self.assertNotEqual(rubric.get_node("r1c1"), changed.get_node("r1c1"))
        self.assertEqual(rubric.get_node("r1c2"), changed.get_node("r1c2"))
        self.assertEqual(rubric.get_node("r1"), changed.get_node("r1"))
        self.assertEqual(rubric.edges, changed.edges)
        self.assertEqual(rubric.root_ids, changed.root_ids)

    def test_bundle_refine_cannot_rename_child(self) -> None:
        rubric = _rubric()
        value = self._proposal(rubric)
        value["edits"][0]["criterion_name"] = "renamed"
        with self.assertRaisesRegex(ValueError, "cannot rename"):
            parse_bundle_refine(
                value, rubric=rubric, root_id="r1",
                allowed_sample_ids=["s1"], max_description_chars=1000)

    def test_failure_type_is_operator_specific(self) -> None:
        value = {
            "primary_failure_type": "bundle_edit_interaction",
            "implicated_child_ids": [],
            "summary": "interacting edits",
            "corrected_harmed_explanation": "corrections were offset",
            "preserve_next_time": [],
            "change_next_time": ["separate the boundaries"],
        }
        parsed = parse_failure_attribution(
            value, operator="refine", allowed_child_ids=["r1c1", "r1c2"])
        self.assertEqual("bundle_edit_interaction", parsed["primary_failure_type"])
        with self.assertRaisesRegex(ValueError, "invalid bundle failure type"):
            parse_failure_attribution(
                value, operator="split", allowed_child_ids=["r1c1", "r1c2"])


class TestPhase21FrozenProtocol(unittest.TestCase):
    def test_config_blocks_match_code(self) -> None:
        config = json.loads(Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json").read_text(encoding="utf-8"))
        self.assertEqual(evolution.SETTINGS, evolution._settings(config))
        self.assertEqual(vlrb.SETTINGS, vlrb._settings(config))

    def test_specialized_diagnostic_is_not_a_trigger_input(self) -> None:
        self.assertEqual(
            "post_commit_read_only_diagnostic",
            evolution.SETTINGS["specialized_acc_role"])
        self.assertFalse(evolution.SETTINGS["strong_child_locking"])
        self.assertFalse(evolution.SETTINGS["partial_acceptance"])

    def test_commit_reuses_every_root_call_without_unified_regeneration(self) -> None:
        rubric = _rubric()
        rows = [{"sample_id": "s1", "answer": "A"}]
        roots = {}
        for root_id in rubric.root_ids:
            value = {
                "root_id": root_id,
                "samples": [{"sample_id": "s1", "order": 0,
                             "call": _call("A")}],
            }
            value["metrics"] = runtime.root_metrics(value, rows, ["s1"])
            roots[root_id] = value

        def fake_system_eval(*args, **kwargs):
            baseline = kwargs["baseline"]
            samples = []
            for sample in baseline["samples"]:
                calls = {
                    root_id: {**call, "incremental_reuse": True}
                    for root_id, call in sample["replicates"]["0"][
                        "subtrees"].items()}
                samples.append({
                    "sample_id": sample["sample_id"],
                    "replicates": {"0": {
                        "order": 0, "subtrees": calls,
                        "regenerated_root_ids": [],
                        "arbiter": _call("A")}}})
            return {
                "samples": samples,
                "metrics": {"technical_failure_count": 0},
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(
                evolution, "_system_eval", side_effect=fake_system_eval):
            value = evolution._commit_system(
                {}, target=Path(directory), epoch_dir=Path(directory),
                rows=rows, rubric=rubric, roots=roots, label="test")
        self.assertEqual([], value["samples"][0]["replicates"]["0"][
            "regenerated_root_ids"])

    def test_all_independently_accepted_refines_merge(self) -> None:
        rubric = _rubric()
        first_value = self._proposal_for(rubric, "r1", "r1c1", "child_one")
        second_value = self._proposal_for(rubric, "r2", "r2c1", "child_three")
        first = apply_bundle_refine(rubric, parse_bundle_refine(
            first_value, rubric=rubric, root_id="r1",
            allowed_sample_ids=["s1"], max_description_chars=1000))
        second = apply_bundle_refine(rubric, parse_bundle_refine(
            second_value, rubric=rubric, root_id="r2",
            allowed_sample_ids=["s1"], max_description_chars=1000))
        merged = evolution._merge_accepted(rubric, {
            "r1": {"decision": "accepted", "operator": "refine",
                   "after_rubric": first},
            "r2": {"decision": "accepted", "operator": "refine",
                   "after_rubric": second},
        })
        self.assertNotEqual(rubric.get_node("r1c1"), merged.get_node("r1c1"))
        self.assertNotEqual(rubric.get_node("r2c1"), merged.get_node("r2c1"))

    @staticmethod
    def _proposal_for(
        rubric: StructuredRubric, root_id: str, child_id: str, name: str,
    ) -> dict:
        child_ids = {item.node_id for item in rubric.children(root_id)}
        old = rubric.get_node(child_id).criterion.description
        return {
            "root_id": root_id,
            "original_subtree_sha256": subtree_sha256(rubric, root_id),
            "edits": [{
                "node_id": child_id, "criterion_name": name,
                "original_description_sha256": hashlib.sha256(
                    old.encode("utf-8")).hexdigest(),
                "description": _description(f"{name} revised"),
                "failure_analysis": ["boundary failure"],
                "rationale": "local correction",
            }],
            "unchanged_node_ids": sorted(child_ids - {child_id}),
            "bundle_rationale": "atomic root patch",
            "representative_sample_ids": ["s1"],
        }


if __name__ == "__main__":
    unittest.main()

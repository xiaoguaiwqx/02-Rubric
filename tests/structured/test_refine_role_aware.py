"""Role-aware Refine v2 trigger and offline audit tests."""

from __future__ import annotations

import unittest

from critiq.structured import (
    EdgeCondition,
    ErrorSampleRef,
    EvolutionContext,
    NodeFeedback,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricFeedback,
    RubricNode,
    StructuredRubric,
    Vote,
    detect_refine_trigger,
)
from experiments.evolving_structured_rubrics.refine_evolution import (
    FIVE_ROOT_LOCKED_SPLIT_REFINE_V1,
    VISUAL_SPLIT_REFINE_V1,
    _checkpoint_description_map,
    _checkpoint_v2_description_map,
    _integrated_select_locked_child,
    _trigger_audit_rows,
    _visual_child_ids,
    _visual_schedule,
    _visual_variant_rubric,
)


THRESHOLDS = {
    "tau_acc": .55,
    "tau_refine": .80,
    "tau_cov_high": .80,
    "N_min_support": 15,
}


def _rubric() -> StructuredRubric:
    root = RubricNode("root", RubricCriterionSnapshot("root", "root", 1.0))
    child = RubricNode("child", RubricCriterionSnapshot("child", "child", 1.0))
    return StructuredRubric(
        {"root": root, "child": child},
        (RubricEdge("root", "child", EdgeCondition.ALWAYS),),
        ("root",),
    )


def _node_feedback(
    node_id: str,
    *,
    sample_count: int,
    correct: int,
    wrong: int,
    criterion_name: str | None = None,
) -> NodeFeedback:
    support = correct + wrong
    abstain = sample_count - support
    errors = tuple(
        ErrorSampleRef(
            f"{node_id}-wrong-{index}", node_id, Vote.B, "A", "wrong", "thought")
        for index in range(wrong)
    ) + tuple(
        ErrorSampleRef(
            f"{node_id}-abstain-{index}", node_id, Vote.ABSTAIN, "A",
            "abstain", "thought")
        for index in range(abstain)
    )
    return NodeFeedback(
        node_id, criterion_name or node_id, sample_count, support, correct, wrong, abstain, 0, 0,
        correct / support if support else 0.0, support / sample_count, 1.0,
        4, 0.0, errors,
    )


def _feedback(
    rubric: StructuredRubric,
    root_counts: tuple[int, int, int] = (20, 14, 6),
    child_counts: tuple[int, int, int] = (20, 11, 9),
) -> RubricFeedback:
    root = _node_feedback(
        "root", sample_count=root_counts[0], correct=root_counts[1],
        wrong=root_counts[2])
    child = _node_feedback(
        "child", sample_count=child_counts[0], correct=child_counts[1],
        wrong=child_counts[2])
    return RubricFeedback(
        rubric.rubric_sha256,
        tuple(f"s{index}" for index in range(max(root_counts[0], child_counts[0]))),
        {"root": root, "child": child}, {}, (), (), (),
    )


class RoleAwareRefineTriggerTests(unittest.TestCase):
    def test_uniform_v1_remains_default_and_root_rule_is_unchanged(self):
        rubric = _rubric()
        context = EvolutionContext(rubric, _feedback(rubric))
        default = detect_refine_trigger(context, "root", THRESHOLDS)
        explicit = detect_refine_trigger(
            context, "root", THRESHOLDS, trigger_mode="uniform_v1")
        role = detect_refine_trigger(
            context, "root", THRESHOLDS, trigger_mode="role_aware_v2")
        self.assertEqual(default, explicit)
        self.assertEqual(explicit, role)

    def test_child_ignores_coverage_but_requires_strict_accuracy_and_wrong(self):
        rubric = _rubric()
        context = EvolutionContext(rubric, _feedback(rubric))
        uniform = detect_refine_trigger(
            context, "child", THRESHOLDS, trigger_mode="uniform_v1")
        role = detect_refine_trigger(
            context, "child", THRESHOLDS, trigger_mode="role_aware_v2")
        self.assertFalse(uniform.triggered)
        self.assertIn("coverage_above_tau_cov_high", uniform.reasons)
        self.assertTrue(role.triggered)

        cases = (
            ((20, 10, 10), "accuracy_not_above_child_minimum"),  # 0.50
            ((20, 16, 4), "accuracy_not_below_child_maximum"),  # 0.80
            ((20, 12, 4), "wrong_below_child_minimum"),          # support 16
            ((20, 10, 4), "support_below_child_minimum"),        # support 14
        )
        for counts, reason in cases:
            with self.subTest(counts=counts):
                feedback = _feedback(rubric, child_counts=counts)
                decision = detect_refine_trigger(
                    EvolutionContext(rubric, feedback), "child", THRESHOLDS,
                    trigger_mode="role_aware_v2")
                self.assertFalse(decision.triggered)
                self.assertIn(reason, decision.reasons)

    def test_offline_audit_marks_only_role_aware_additions(self):
        rubric = _rubric()
        feedback = _feedback(rubric)
        rows = _trigger_audit_rows(
            rubric, feedback,
            {"evolution_policy": {"trigger_thresholds": THRESHOLDS}})
        by_id = {item["node_id"]: item for item in rows}
        self.assertFalse(by_id["root"]["newly_eligible"])
        self.assertEqual(by_id["root"]["role"], "root")
        self.assertTrue(by_id["child"]["newly_eligible"])
        self.assertEqual(by_id["child"]["role"], "child")

    def test_unknown_trigger_mode_is_rejected(self):
        rubric = _rubric()
        with self.assertRaisesRegex(ValueError, "unsupported Refine trigger mode"):
            detect_refine_trigger(
                EvolutionContext(rubric, _feedback(rubric)), "child", THRESHOLDS,
                trigger_mode="future")

    def test_checkpoint_v1_map_is_node_id_based_and_v2_keeps_versions_separate(self):
        source = _rubric()
        epoch_one = StructuredRubric(
            {"root": RubricNode("root", RubricCriterionSnapshot("root", "root-v1", 1.0)),
             "child": source.get_node("child")},
            source.edges, source.root_ids)
        epoch_three = StructuredRubric(
            {"root": epoch_one.get_node("root"),
             "child": RubricNode("child", RubricCriterionSnapshot("child", "child-v3", 1.0))},
            source.edges, source.root_ids)
        changes, generated = _checkpoint_description_map(
            {"source": source, "epoch_01": epoch_one, "epoch_03": epoch_three})
        self.assertEqual(changes["epoch_01"], ["root"])
        self.assertEqual(changes["epoch_03"], ["root", "child"])
        self.assertEqual(generated, ["root", "child"])

        v2_changes = _checkpoint_v2_description_map(
            {"source": source, "epoch_01": epoch_one, "epoch_03": epoch_three})
        self.assertEqual(v2_changes["epoch_01"], ["root"])
        self.assertEqual(v2_changes["epoch_03"], ["root", "child"])

    def test_checkpoint_v2_does_not_assume_repeated_node_has_same_description(self):
        source = _rubric()
        epoch_one = StructuredRubric(
            {"root": RubricNode("root", RubricCriterionSnapshot("root", "root-v1", 1.0)),
             "child": source.get_node("child")},
            source.edges, source.root_ids)
        epoch_three = StructuredRubric(
            {"root": RubricNode("root", RubricCriterionSnapshot("root", "root-v3", 1.0)),
             "child": source.get_node("child")},
            source.edges, source.root_ids)
        changes = _checkpoint_v2_description_map(
            {"source": source, "epoch_01": epoch_one, "epoch_03": epoch_three})
        self.assertEqual(changes["epoch_01"], ["root"])
        self.assertEqual(changes["epoch_03"], ["root"])
        self.assertNotEqual(
            epoch_one.get_node("root").criterion.description,
            epoch_three.get_node("root").criterion.description)

    def test_visual_local_protocol_keeps_split_disabled_and_uses_8000(self):
        self.assertFalse(VISUAL_SPLIT_REFINE_V1["split_enabled"])
        self.assertEqual(VISUAL_SPLIT_REFINE_V1["max_epochs"], 3)
        self.assertEqual(VISUAL_SPLIT_REFINE_V1["worker_endpoint"], "vllm-8000")

    def test_integrated_protocol_freezes_8001_and_no_partial_acceptance(self):
        self.assertEqual(
            FIVE_ROOT_LOCKED_SPLIT_REFINE_V1["worker_endpoint"], "vllm-8001")
        self.assertEqual(FIVE_ROOT_LOCKED_SPLIT_REFINE_V1["min_epochs"], 3)
        self.assertEqual(FIVE_ROOT_LOCKED_SPLIT_REFINE_V1["max_epochs"], 5)
        self.assertFalse(
            FIVE_ROOT_LOCKED_SPLIT_REFINE_V1["partial_split_acceptance"])

    def test_integrated_lock_selection_is_thresholded_and_deterministic(self):
        diagnostics = {"children": [
            {"criterion_name": "z", "cluster_id": "z", "criterion_sha256": "z",
             "support": 20, "net_corrected": 4,
             "single_child_specialized_accuracy": .70},
            {"criterion_name": "a", "cluster_id": "a", "criterion_sha256": "a",
             "support": 19, "net_corrected": 4,
             "single_child_specialized_accuracy": .75},
            {"criterion_name": "weak", "cluster_id": "weak", "criterion_sha256": "w",
             "support": 100, "net_corrected": 2,
             "single_child_specialized_accuracy": .99},
        ]}
        settings = {"strong_child_min_support": 15,
                    "strong_child_min_net_corrected": 3}
        selected = _integrated_select_locked_child(diagnostics, settings)
        self.assertEqual(selected["criterion_name"], "a")
        self.assertEqual(selected["cluster_id"], "a")
        self.assertIsNone(_integrated_select_locked_child(
            {"children": [diagnostics["children"][2]]}, settings))

    def test_visual_child_scope_is_exact_and_role_aware_schedules_locked_child(self):
        root = RubricNode("init_02_visual_grounding_and_details",
                          RubricCriterionSnapshot("visual", "root", 1.0))
        children = {
            f"child-{index}": RubricNode(
                f"child-{index}", RubricCriterionSnapshot(
                    f"criterion-{index}", f"description-{index}", 1.0))
            for index in range(4)
        }
        rubric = StructuredRubric(
            {root.node_id: root, **children},
            tuple(RubricEdge(root.node_id, node_id, EdgeCondition.ALWAYS)
                  for node_id in children),
            (root.node_id,))
        feedback_nodes = {
            root.node_id: _node_feedback(root.node_id, sample_count=90,
                                         correct=60, wrong=20, criterion_name="visual"),
            **{node_id: _node_feedback(node_id, sample_count=90,
                                       correct=47, wrong=25,
                                       criterion_name=rubric.get_node(node_id).criterion.name)
               for node_id in children},
        }
        feedback = RubricFeedback(
            rubric.rubric_sha256, tuple(f"s{index}" for index in range(90)),
            feedback_nodes, {}, (), (), ())
        child_ids = _visual_child_ids(rubric)
        self.assertEqual(child_ids, tuple(children))
        self.assertEqual(
            _visual_schedule(
                rubric, feedback,
                {"evolution_policy": {"trigger_thresholds": THRESHOLDS}}, child_ids),
            child_ids)

        base = StructuredRubric({root.node_id: root}, (), (root.node_id,))
        variant = _visual_variant_rubric(base, rubric, child_ids[:1])
        self.assertEqual(set(variant.nodes), {root.node_id, child_ids[0]})
        self.assertEqual(len(variant.edges), 1)


if __name__ == "__main__":
    unittest.main()

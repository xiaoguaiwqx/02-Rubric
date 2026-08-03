import unittest
import copy

from critiq.structured import (
    Applicability,
    BackendSource,
    CriterionStatus,
    DualCascadeExecutor,
    DualExecutionTrace,
    EdgeCondition,
    ExecutionConfig,
    GateBackendResult,
    GateJudgement,
    GateStateOutput,
    ModelCallMetrics,
    PairwiseBackendResult,
    PairwiseVoteOutput,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredRubric,
    StructuredSystemVariant,
    Vote,
    replay_dual_execution_trace,
)


def pair(vote, parse_ok=True):
    return PairwiseVoteOutput(vote, parse_ok, "raw", None if parse_ok else "bad", 1,
                              "thought" if parse_ok else None, parse_ok)


def gate(a, b, applicable=Applicability.YES):
    return GateStateOutput(GateJudgement(applicable, a, b), "raw", None, 1)


class FakePairBackend:
    def __init__(self, outputs): self.outputs = outputs; self.calls = []
    def sample_id(self, data): return data["sample_id"]
    def sample_fingerprint(self, data): return "0" * 64
    def evaluate(self, data, node):
        self.calls.append(node.node_id)
        return PairwiseBackendResult(self.outputs[node.node_id], BackendSource.OFFLINE_ARTIFACT,
                                     ModelCallMetrics(), ModelCallMetrics())


class FakeGateBackend:
    def __init__(self, outputs): self.outputs = outputs; self.calls = []
    def sample_id(self, data): return data["sample_id"]
    def sample_fingerprint(self, data): return "0" * 64
    def evaluate(self, data, node):
        self.calls.append(node.node_id)
        return GateBackendResult(self.outputs[node.node_id], BackendSource.OFFLINE_ARTIFACT,
                                 ModelCallMetrics(), ModelCallMetrics())


def rubric():
    nodes = {
        name: RubricNode(name, RubricCriterionSnapshot(name, name + " desc", 0.5))
        for name in ("root", "nondec", "status")
    }
    return StructuredRubric(nodes, (
        RubricEdge("root", "nondec", EdgeCondition.PARENT_NONDECISIVE),
        RubricEdge("root", "status", EdgeCondition.PARENT_BOTH_PASS),
    ), ("root",))


class DualExecutorTests(unittest.TestCase):
    def setUp(self): self.data = {"sample_id": "s", "A": "a", "B": "b"}

    def test_all_nodes_never_calls_gate(self):
        pair_backend = FakePairBackend({name: pair(Vote.A) for name in ("root", "nondec", "status")})
        gate_backend = FakeGateBackend({"root": gate(CriterionStatus.PASS, CriterionStatus.PASS)})
        trace = DualCascadeExecutor(rubric(), pair_backend, gate_backend).execute_one(
            self.data, ExecutionConfig(StructuredSystemVariant.H1_HIERARCHICAL_ONLY))
        self.assertEqual(set(pair_backend.calls), {"root", "nondec", "status"})
        self.assertEqual(gate_backend.calls, [])
        self.assertTrue(all(edge.condition_source == "all_nodes" for edge in trace.nodes[0].edges))

    def test_parse_failure_is_not_parent_nondecisive(self):
        pair_backend = FakePairBackend({"root": pair(Vote.ABSTAIN, False), "status": pair(Vote.B),
                                        "nondec": pair(Vote.A)})
        gate_backend = FakeGateBackend({"root": gate(CriterionStatus.PASS, CriterionStatus.PASS)})
        trace = DualCascadeExecutor(rubric(), pair_backend, gate_backend).execute_one(
            self.data, ExecutionConfig(StructuredSystemVariant.M1_ALL_ROOTS_CASCADE))
        self.assertEqual(pair_backend.calls, ["root", "status"])
        self.assertEqual(gate_backend.calls, ["root"])
        edges = {edge.child_id: edge for edge in trace.nodes[0].edges}
        self.assertFalse(edges["nondec"].matched)
        self.assertEqual(edges["nondec"].skip_reason, "pairwise_parse_failure")
        self.assertTrue(edges["status"].matched)

    def test_model_abstain_expands_nondecisive_and_gate_is_memoized(self):
        pair_backend = FakePairBackend({"root": pair(Vote.ABSTAIN), "nondec": pair(Vote.A),
                                        "status": pair(Vote.B)})
        gate_backend = FakeGateBackend({"root": gate(CriterionStatus.PASS, CriterionStatus.PASS)})
        trace = DualCascadeExecutor(rubric(), pair_backend, gate_backend).execute_one(
            self.data, ExecutionConfig(StructuredSystemVariant.G1_GATING_ONLY))
        self.assertEqual(set(pair_backend.calls), {"root", "nondec", "status"})
        self.assertEqual(gate_backend.calls, ["root"])
        self.assertEqual(trace.gate_metrics.logical_evaluations, 0)  # offline provenance

    def test_trace_roundtrip_and_replay_reject_tampering(self):
        pair_backend = FakePairBackend({name: pair(Vote.A) for name in ("root", "nondec", "status")})
        trace = DualCascadeExecutor(rubric(), pair_backend, None).execute_one(
            self.data, ExecutionConfig(StructuredSystemVariant.B1_FLAT))
        loaded = DualExecutionTrace.from_dict(trace.to_dict())
        self.assertEqual(replay_dual_execution_trace(loaded, rubric()), trace.final_preference)
        value = trace.to_dict(); value["final_preference"] = "B"
        tampered = DualExecutionTrace.from_dict(value)
        with self.assertRaises(ValueError): replay_dual_execution_trace(tampered, rubric())

    def test_replay_rejects_structural_and_statistical_tampering(self):
        pair_backend = FakePairBackend({name: pair(Vote.A) for name in ("root", "nondec", "status")})
        trace = DualCascadeExecutor(rubric(), pair_backend, None).execute_one(
            self.data, ExecutionConfig(StructuredSystemVariant.H1_HIERARCHICAL_ONLY))
        original = trace.to_dict()

        deleted_edge = copy.deepcopy(original)
        deleted_edge["nodes"][0]["edges"].pop()
        with self.assertRaisesRegex(ValueError, "edge count"):
            replay_dual_execution_trace(DualExecutionTrace.from_dict(deleted_edge), rubric())

        child_flag = copy.deepcopy(original)
        child_flag["nodes"][0]["edges"][0]["child_visited"] = False
        with self.assertRaisesRegex(ValueError, "child_visited"):
            replay_dual_execution_trace(DualExecutionTrace.from_dict(child_flag), rubric())

        root_sets = copy.deepcopy(original)
        root_sets["roots"][0]["visited_node_ids"].pop()
        with self.assertRaisesRegex(ValueError, "visited/avoided"):
            replay_dual_execution_trace(DualExecutionTrace.from_dict(root_sets), rubric())

        counterfactual = copy.deepcopy(original)
        counterfactual["counterfactual_metrics"]["visited_nodes"] -= 1
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            replay_dual_execution_trace(DualExecutionTrace.from_dict(counterfactual), rubric())

        resolved = copy.deepcopy(original)
        resolved["router"]["resolved"]["selected_root_ids"] = []
        with self.assertRaisesRegex(ValueError, "resolved roots"):
            replay_dual_execution_trace(DualExecutionTrace.from_dict(resolved), rubric())


if __name__ == "__main__": unittest.main()

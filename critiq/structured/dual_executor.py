"""Cascade executor whose aggregation vote and traversal state are decoupled."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

from critiq.utils import PairData

from .aggregation import aggregate_child_subtrees, aggregate_flat_votes, aggregate_selected_roots
from .backend import RouterBackend
from .dual_backend import GateStateBackend, PairwiseVoteBackend
from .dual_trace import (
    DualEdgeExecutionTrace,
    DualExecutionOutput,
    DualExecutionTrace,
    DualNodeExecutionTrace,
)
from .executor import ExecutionConfig, StructuredSystemVariant
from .judgement import Applicability, CriterionStatus, Vote
from .schema import StructuredRubric
from .semantics import EdgeCondition, TraversalPolicy, resolve_root_routing
from .telemetry import ModelCallMetrics, combine_model_call_metrics
from .trace import CounterfactualTraversalMetrics, RootExecutionTrace, RouterExecutionTrace


def _match_dual_edge(condition: EdgeCondition, pairwise_output, gate_output) -> tuple[bool, str, str | None]:
    if condition is EdgeCondition.ALWAYS:
        return True, "always", None
    if condition is EdgeCondition.PARENT_NONDECISIVE:
        if not pairwise_output.parse_ok:
            return False, "pairwise_vote", "pairwise_parse_failure"
        return (pairwise_output.is_model_abstain, "pairwise_vote",
                None if pairwise_output.is_model_abstain else "parent_pairwise_decisive")
    if gate_output is None:
        return False, "gate_state", "gate_backend_unavailable"
    judgement = gate_output.judgement
    if not gate_output.valid:
        return False, "gate_state", "gate_invalid"
    if judgement.applicable is not Applicability.YES:
        return False, "gate_state", "gate_inapplicable"
    if condition is EdgeCondition.PARENT_BOTH_PASS:
        matched = judgement.status_a is CriterionStatus.PASS and judgement.status_b is CriterionStatus.PASS
    elif condition is EdgeCondition.PARENT_BOTH_FAIL:
        matched = judgement.status_a is CriterionStatus.FAIL and judgement.status_b is CriterionStatus.FAIL
    else:
        raise ValueError(f"unsupported edge condition {condition}")
    return matched, "gate_state", None if matched else "parent_gate_not_matched"


class DualCascadeExecutor:
    def __init__(self, rubric: StructuredRubric, pairwise_backend: PairwiseVoteBackend,
                 gate_backend: GateStateBackend | None = None,
                 router_backend: RouterBackend | None = None) -> None:
        if not isinstance(rubric, StructuredRubric): raise TypeError("rubric must be StructuredRubric")
        self.rubric = rubric; self.pairwise_backend = pairwise_backend
        self.gate_backend = gate_backend; self.router_backend = router_backend

    def _subtree_ids(self, root_id: str) -> tuple[str, ...]:
        return tuple(node_id for node_id in self.rubric.preorder_node_ids() if self.rubric.root_id_for(node_id) == root_id)

    def execute_one(self, data: PairData, config: ExecutionConfig) -> DualExecutionTrace:
        semaphore = threading.BoundedSemaphore(config.request_concurrent_limit)
        return self._execute_one(data, config, semaphore)

    def _execute_one(self, data: PairData, config: ExecutionConfig,
                     semaphore: threading.BoundedSemaphore) -> DualExecutionTrace:
        started = time.perf_counter()
        sample_id = self.pairwise_backend.sample_id(data)
        fingerprint = self.pairwise_backend.sample_fingerprint(data)
        if config.router_enabled and self.router_backend is not None:
            with semaphore: router_result = self.router_backend.route(data, self.rubric)
            resolved = resolve_root_routing(router_result.output.decision, self.rubric.root_ids, enabled=True)
            router_trace = RouterExecutionTrace(True, router_result.output, resolved, router_result.source,
                router_result.metrics, router_result.generation_metrics, router_result.cache_key)
        elif config.router_enabled:
            resolved = resolve_root_routing(None, self.rubric.root_ids, enabled=True)
            router_trace = RouterExecutionTrace(True, None, resolved, None, ModelCallMetrics(), ModelCallMetrics(), None)
        else:
            resolved = resolve_root_routing(None, self.rubric.root_ids, enabled=False)
            router_trace = RouterExecutionTrace(False, None, resolved, None, ModelCallMetrics(), ModelCallMetrics(), None)

        selected = set(resolved.selected_root_ids)
        memo: dict[str, Vote] = {}; traces: dict[str, DualNodeExecutionTrace] = {}; lock = threading.Lock()

        def subtree(node_id: str) -> Vote:
            with lock:
                if node_id in memo: return memo[node_id]
            node = self.rubric.get_node(node_id)
            with semaphore: pairwise = self.pairwise_backend.evaluate(data, node)
            edges = self.rubric.child_edges(node_id)
            needs_gate = config.traversal_policy is TraversalPolicy.CONDITIONAL and any(
                edge.condition in {EdgeCondition.PARENT_BOTH_PASS, EdgeCondition.PARENT_BOTH_FAIL} for edge in edges)
            gate = None
            if needs_gate and self.gate_backend is not None:
                with semaphore: gate = self.gate_backend.evaluate(data, node)

            matches: list[tuple[bool, str, str | None]] = []
            for edge in edges:
                if config.traversal_policy is TraversalPolicy.ALL_NODES:
                    matches.append((True, "all_nodes", None))
                else:
                    matches.append(_match_dual_edge(edge.condition, pairwise.output, None if gate is None else gate.output))
            eligible = [edge for edge, match in zip(edges, matches) if match[0]]
            child_results: dict[str, Vote] = {}
            if len(eligible) > 1 and config.max_concurrent > 1:
                with ThreadPoolExecutor(max_workers=min(config.max_concurrent, len(eligible))) as pool:
                    futures = {edge.child_id: pool.submit(subtree, edge.child_id) for edge in eligible}
                    for edge in eligible: child_results[edge.child_id] = futures[edge.child_id].result()
            else:
                for edge in eligible: child_results[edge.child_id] = subtree(edge.child_id)

            edge_traces = tuple(DualEdgeExecutionTrace(edge.parent_id, edge.child_id, edge.condition.value,
                source, matched, matched, reason, child_results.get(edge.child_id))
                for edge, (matched, source, reason) in zip(edges, matches))
            child_votes = [child_results[edge.child_id] for edge in eligible]
            subtree_vote = aggregate_child_subtrees(pairwise.output.vote, child_votes)
            trace = DualNodeExecutionTrace(
                node_id, node.criterion.name, self.rubric.depth(node_id), pairwise.output, pairwise.source,
                pairwise.metrics, pairwise.generation_metrics, pairwise.cache_key,
                None if gate is None else gate.output, None if gate is None else gate.source,
                ModelCallMetrics() if gate is None else gate.metrics,
                ModelCallMetrics() if gate is None else gate.generation_metrics,
                None if gate is None else gate.cache_key, pairwise.output.vote, edge_traces, subtree_vote)
            with lock:
                if node_id in memo: raise RuntimeError(f"node evaluated more than once: {node_id}")
                memo[node_id] = subtree_vote; traces[node_id] = trace
            return subtree_vote

        roots = [root_id for root_id in self.rubric.root_ids if root_id in selected]
        root_votes: dict[str, Vote] = {}
        if len(roots) > 1 and config.max_concurrent > 1:
            with ThreadPoolExecutor(max_workers=min(config.max_concurrent, len(roots))) as pool:
                futures = {root_id: pool.submit(subtree, root_id) for root_id in roots}
                for root_id in roots: root_votes[root_id] = futures[root_id].result()
        else:
            for root_id in roots: root_votes[root_id] = subtree(root_id)

        ordered = tuple(traces[node_id] for node_id in self.rubric.preorder_node_ids() if node_id in traces)
        visited_ids = set(traces)
        root_traces = tuple(RootExecutionTrace(root_id, root_id in selected,
            None if root_id in selected else "root_not_selected",
            tuple(node_id for node_id in self._subtree_ids(root_id) if node_id in visited_ids),
            tuple(node_id for node_id in self._subtree_ids(root_id) if node_id not in visited_ids),
            root_votes.get(root_id)) for root_id in self.rubric.root_ids)
        if config.aggregation_mode.value == "flat":
            aggregation = {node.node_id: node.local_vote for node in ordered}
            final = aggregate_flat_votes(aggregation.values())
        else:
            aggregation = {root_id: root_votes[root_id] for root_id in roots}
            final = aggregate_selected_roots(aggregation, tuple(roots))
        matched = sum(edge.matched for node in ordered for edge in node.edges)
        skipped = sum(not edge.matched for node in ordered for edge in node.edges)
        counterfactual = CounterfactualTraversalMetrics(len(self.rubric.nodes), len(ordered),
            len(self.rubric.nodes) - len(ordered), matched, skipped,
            max((node.depth for node in ordered), default=0), len(roots))
        return DualExecutionTrace(sample_id, fingerprint, self.rubric.rubric_sha256, config.variant.value,
            config.traversal_policy, config.aggregation_mode, router_trace, root_traces, ordered,
            aggregation, final, counterfactual,
            combine_model_call_metrics(node.pairwise_metrics for node in ordered),
            combine_model_call_metrics(node.gate_metrics for node in ordered), time.perf_counter() - started)

    def execute_batch(self, dataset: Sequence[PairData], config: ExecutionConfig,
                      on_trace: Callable[[int, DualExecutionTrace], None] | None = None) -> DualExecutionOutput:
        if not dataset: raise ValueError("execution dataset must not be empty")
        semaphore = threading.BoundedSemaphore(config.request_concurrent_limit)
        ordered: list[DualExecutionTrace | None] = [None] * len(dataset)
        def run(index: int, row: PairData): return index, self._execute_one(row, config, semaphore)
        if config.sample_concurrent > 1:
            with ThreadPoolExecutor(max_workers=min(config.sample_concurrent, len(dataset))) as pool:
                futures = [pool.submit(run, index, row) for index, row in enumerate(dataset)]
                for future in as_completed(futures):
                    index, trace = future.result(); ordered[index] = trace
                    if on_trace: on_trace(index, trace)
        else:
            for index, row in enumerate(dataset):
                _, trace = run(index, row); ordered[index] = trace
                if on_trace: on_trace(index, trace)
        return DualExecutionOutput(config.variant.value, self.rubric.rubric_sha256,
                                   tuple(trace for trace in ordered if trace is not None))


def replay_dual_execution_trace(trace: DualExecutionTrace, rubric: StructuredRubric) -> FinalPreference:
    """Strictly reconstruct traversal and aggregation without Agent/cache access."""
    if trace.rubric_sha256 != rubric.rubric_sha256:
        raise ValueError("dual trace rubric mismatch")
    config = ExecutionConfig(StructuredSystemVariant(trace.variant))
    if trace.traversal_policy is not config.traversal_policy or trace.aggregation_mode is not config.aggregation_mode:
        raise ValueError("dual trace variant conflicts with traversal/aggregation mode")
    if trace.router.enabled != config.router_enabled:
        raise ValueError("dual trace router enabled flag conflicts with variant")
    router_decision = trace.router.output.decision if trace.router.output is not None else None
    expected_resolved = resolve_root_routing(router_decision, rubric.root_ids,
                                             enabled=config.router_enabled)
    if trace.router.resolved != expected_resolved:
        raise ValueError("dual trace resolved roots cannot be reconstructed")

    node_by_id = {node.node_id: node for node in trace.nodes}
    selected_roots = expected_resolved.selected_root_ids
    expected_visited: list[str] = []
    subtree_votes: dict[str, Vote] = {}

    def rebuild_subtree(node_id: str) -> Vote:
        if node_id not in node_by_id:
            raise ValueError(f"dual trace is missing visited node {node_id!r}")
        if node_id in subtree_votes:
            raise ValueError(f"dual trace reaches node more than once: {node_id!r}")
        expected_visited.append(node_id)
        node_trace = node_by_id[node_id]
        node = rubric.get_node(node_trace.node_id)
        if node.criterion.name != node_trace.criterion_name or node_trace.depth != rubric.depth(node_id):
            raise ValueError("dual trace criterion/depth mismatch")
        expected_edges = rubric.child_edges(node_id)
        if len(node_trace.edges) != len(expected_edges):
            raise ValueError("dual trace outgoing edge count mismatch")
        status_dependent = any(edge.condition in {EdgeCondition.PARENT_BOTH_PASS,
                                                   EdgeCondition.PARENT_BOTH_FAIL}
                               for edge in expected_edges)
        if config.traversal_policy is TraversalPolicy.ALL_NODES and node_trace.gate_output is not None:
            raise ValueError("all-nodes trace must not contain Gate output")
        if (config.traversal_policy is TraversalPolicy.CONDITIONAL
                and not status_dependent and node_trace.gate_output is not None):
            raise ValueError("trace contains an unnecessary Gate evaluation")

        child_votes: list[Vote] = []
        for edge_trace, edge in zip(node_trace.edges, expected_edges):
            if edge_trace.parent_id != edge.parent_id or edge_trace.child_id != edge.child_id or edge_trace.condition != edge.condition.value:
                raise ValueError("dual trace edge mismatch")
            if config.traversal_policy is TraversalPolicy.ALL_NODES:
                expected = (True, "all_nodes", None)
            else:
                expected = _match_dual_edge(edge.condition, node_trace.pairwise_output, node_trace.gate_output)
            if (edge_trace.matched, edge_trace.condition_source, edge_trace.skip_reason) != expected:
                raise ValueError("dual trace edge decision was tampered")
            if edge_trace.child_visited is not edge_trace.matched:
                raise ValueError("dual trace child_visited conflicts with edge match")
            if edge_trace.matched:
                child_vote = rebuild_subtree(edge.child_id)
                if edge_trace.child_subtree_vote is not child_vote:
                    raise ValueError("dual trace child subtree vote mismatch")
                child_votes.append(child_vote)
            elif edge_trace.child_subtree_vote is not None:
                raise ValueError("skipped edge must not contain a child subtree vote")
        expected_subtree = aggregate_child_subtrees(node_trace.local_vote, child_votes)
        if node_trace.subtree_vote is not expected_subtree:
            raise ValueError("dual trace subtree vote mismatch")
        subtree_votes[node_id] = expected_subtree
        return expected_subtree

    for root_id in selected_roots:
        rebuild_subtree(root_id)
    if set(expected_visited) != set(node_by_id):
        raise ValueError("dual trace contains nodes outside reconstructed traversal")
    canonical_visited = tuple(node_id for node_id in rubric.preorder_node_ids()
                              if node_id in node_by_id)
    if tuple(node.node_id for node in trace.nodes) != canonical_visited:
        raise ValueError("dual trace nodes are not in canonical execution order")

    if tuple(root.root_id for root in trace.roots) != rubric.root_ids:
        raise ValueError("dual trace roots are missing, extra, or out of order")
    selected_set = set(selected_roots)
    for root_trace in trace.roots:
        subtree_ids = tuple(node_id for node_id in rubric.preorder_node_ids()
                            if rubric.root_id_for(node_id) == root_trace.root_id)
        visited = tuple(node_id for node_id in subtree_ids if node_id in node_by_id)
        avoided = tuple(node_id for node_id in subtree_ids if node_id not in node_by_id)
        selected = root_trace.root_id in selected_set
        if root_trace.selected != selected:
            raise ValueError("dual trace root selected flag mismatch")
        if root_trace.skip_reason != (None if selected else "root_not_selected"):
            raise ValueError("dual trace root skip reason mismatch")
        if root_trace.visited_node_ids != visited or root_trace.avoided_node_ids != avoided:
            raise ValueError("dual trace root visited/avoided sets mismatch")
        expected_root_vote = subtree_votes[root_trace.root_id] if selected else None
        if root_trace.subtree_vote is not expected_root_vote:
            raise ValueError("dual trace root subtree vote mismatch")

    if config.aggregation_mode.value == "flat":
        expected_inputs = {node.node_id: node.local_vote for node in trace.nodes}
        final = aggregate_flat_votes(expected_inputs.values())
    else:
        expected_inputs = {root_id: subtree_votes[root_id] for root_id in selected_roots}
        final = aggregate_selected_roots(expected_inputs, selected_roots)
    if dict(trace.aggregation_input_votes) != expected_inputs or trace.final_preference is not final:
        raise ValueError("dual trace aggregation was tampered")

    expected_counterfactual = CounterfactualTraversalMetrics(
        total_nodes=len(rubric.nodes), visited_nodes=len(trace.nodes),
        avoided_nodes=len(rubric.nodes) - len(trace.nodes),
        matched_edges=sum(edge.matched for node in trace.nodes for edge in node.edges),
        skipped_edges=sum(not edge.matched for node in trace.nodes for edge in node.edges),
        max_visited_depth=max((node.depth for node in trace.nodes), default=0),
        selected_roots=len(selected_roots))
    if trace.counterfactual_metrics != expected_counterfactual:
        raise ValueError("dual trace counterfactual metrics mismatch")
    return final

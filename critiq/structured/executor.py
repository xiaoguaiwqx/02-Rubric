"""Deterministic offline/online executor for structured rubric forests."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from critiq.utils import PairData

from .aggregation import (
    aggregate_child_subtrees,
    aggregate_flat_votes,
    aggregate_selected_roots,
)
from .backend import NodeBackend, RouterBackend
from .judgement import FinalPreference, Vote
from .schema import StructuredRubric
from .semantics import TraversalPolicy, resolve_root_routing, should_visit_child
from .telemetry import ModelCallMetrics, combine_model_call_metrics
from .trace import (
    AggregationMode,
    CounterfactualTraversalMetrics,
    EdgeExecutionTrace,
    ExecutionTrace,
    NodeExecutionTrace,
    RootExecutionTrace,
    RouterExecutionTrace,
    StructuredExecutionEvaluation,
    StructuredExecutionOutput,
)


class StructuredSystemVariant(str, Enum):
    B1_FLAT = "B1_flat"
    H1_HIERARCHICAL_ONLY = "H1_hierarchical_only"
    G1_GATING_ONLY = "G1_gating_only"
    M1_ALL_ROOTS_CASCADE = "M1_all_roots_cascade"
    M2_ROUTED_ROOTS_CASCADE = "M2_routed_roots_cascade"


@dataclass(frozen=True)
class ExecutionConfig:
    variant: StructuredSystemVariant
    max_concurrent: int = 1
    sample_concurrent: int = 1
    global_request_concurrent: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variant, StructuredSystemVariant):
            raise TypeError("variant must be StructuredSystemVariant")
        for name, value in (
            ("max_concurrent", self.max_concurrent),
            ("sample_concurrent", self.sample_concurrent),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.global_request_concurrent is not None and (
            isinstance(self.global_request_concurrent, bool)
            or not isinstance(self.global_request_concurrent, int)
            or self.global_request_concurrent < 1
        ):
            raise ValueError(
                "global_request_concurrent must be None or a positive integer")

    @property
    def request_concurrent_limit(self) -> int:
        return self.global_request_concurrent or self.max_concurrent

    @property
    def traversal_policy(self) -> TraversalPolicy:
        if self.variant in {
            StructuredSystemVariant.B1_FLAT,
            StructuredSystemVariant.H1_HIERARCHICAL_ONLY,
        }:
            return TraversalPolicy.ALL_NODES
        return TraversalPolicy.CONDITIONAL

    @property
    def aggregation_mode(self) -> AggregationMode:
        if self.variant in {
            StructuredSystemVariant.B1_FLAT,
            StructuredSystemVariant.G1_GATING_ONLY,
        }:
            return AggregationMode.FLAT
        return AggregationMode.HIERARCHICAL

    @property
    def router_enabled(self) -> bool:
        return self.variant is StructuredSystemVariant.M2_ROUTED_ROOTS_CASCADE


class StructuredCascadeExecutor:
    def __init__(
        self,
        rubric: StructuredRubric,
        node_backend: NodeBackend,
        router_backend: RouterBackend | None = None,
    ) -> None:
        if not isinstance(rubric, StructuredRubric):
            raise TypeError("rubric must be StructuredRubric")
        self.rubric = rubric
        self.node_backend = node_backend
        self.router_backend = router_backend

    def _subtree_node_ids(self, root_id: str) -> tuple[str, ...]:
        return tuple(
            node_id
            for node_id in self.rubric.preorder_node_ids()
            if self.rubric.root_id_for(node_id) == root_id
        )

    def execute_one(self, data: PairData, config: ExecutionConfig) -> ExecutionTrace:
        if not isinstance(config, ExecutionConfig):
            raise TypeError("config must be ExecutionConfig")
        request_semaphore = threading.BoundedSemaphore(
            config.request_concurrent_limit)
        return self._execute_one(data, config, request_semaphore)

    def _execute_one(
        self,
        data: PairData,
        config: ExecutionConfig,
        request_semaphore: threading.BoundedSemaphore,
    ) -> ExecutionTrace:
        started_at = time.perf_counter()
        sample_id = self.node_backend.sample_id(data)
        sample_fingerprint = self.node_backend.sample_fingerprint(data)

        if config.router_enabled and self.router_backend is not None:
            with request_semaphore:
                router_result = self.router_backend.route(data, self.rubric)
            resolved = resolve_root_routing(
                router_result.output.decision,
                self.rubric.root_ids,
                enabled=True,
            )
            router_trace = RouterExecutionTrace(
                enabled=True,
                output=router_result.output,
                resolved=resolved,
                backend_source=router_result.source,
                metrics=router_result.metrics,
                generation_metrics=router_result.generation_metrics,
                cache_key=router_result.cache_key,
            )
        elif config.router_enabled:
            resolved = resolve_root_routing(None, self.rubric.root_ids, enabled=True)
            router_trace = RouterExecutionTrace(
                enabled=True,
                output=None,
                resolved=resolved,
                backend_source=None,
                metrics=ModelCallMetrics(),
                generation_metrics=ModelCallMetrics(),
                cache_key=None,
            )
        else:
            resolved = resolve_root_routing(None, self.rubric.root_ids, enabled=False)
            router_trace = RouterExecutionTrace(
                enabled=False,
                output=None,
                resolved=resolved,
                backend_source=None,
                metrics=ModelCallMetrics(),
                generation_metrics=ModelCallMetrics(),
                cache_key=None,
            )

        selected = set(resolved.selected_root_ids)
        memo: dict[str, Vote] = {}
        node_traces: dict[str, NodeExecutionTrace] = {}
        memo_lock = threading.Lock()

        def evaluate_subtree(node_id: str) -> Vote:
            with memo_lock:
                if node_id in memo:
                    return memo[node_id]
            node = self.rubric.get_node(node_id)
            # Recursive subtree scheduling can create nested worker pools. The
            # All samples in one batch share this request semaphore. It bounds
            # Router and Worker backend work without relaxing parent-child
            # dependencies inside any sample.
            with request_semaphore:
                result = self.node_backend.evaluate(data, node)
            decision = result.output.local_decision
            child_edges = self.rubric.child_edges(node_id)
            matches = [
                should_visit_child(
                    result.output.judgement,
                    edge.condition,
                    config.traversal_policy,
                )
                for edge in child_edges
            ]
            eligible_edges = [edge for edge, matched in zip(child_edges, matches) if matched]

            child_results: dict[str, Vote] = {}
            if eligible_edges and config.max_concurrent > 1:
                with ThreadPoolExecutor(max_workers=min(config.max_concurrent, len(eligible_edges))) as pool:
                    futures = {edge.child_id: pool.submit(evaluate_subtree, edge.child_id) for edge in eligible_edges}
                    for edge in eligible_edges:
                        child_results[edge.child_id] = futures[edge.child_id].result()
            else:
                for edge in eligible_edges:
                    child_results[edge.child_id] = evaluate_subtree(edge.child_id)

            edge_traces: list[EdgeExecutionTrace] = []
            child_votes: list[Vote] = []
            for edge, matched in zip(child_edges, matches):
                if matched:
                    child_vote = child_results[edge.child_id]
                    child_votes.append(child_vote)
                    edge_traces.append(
                        EdgeExecutionTrace(
                            parent_id=edge.parent_id,
                            child_id=edge.child_id,
                            condition=edge.condition.value,
                            matched=True,
                            child_visited=True,
                            skip_reason=None,
                            child_subtree_vote=child_vote,
                        )
                    )
                else:
                    edge_traces.append(
                        EdgeExecutionTrace(
                            parent_id=edge.parent_id,
                            child_id=edge.child_id,
                            condition=edge.condition.value,
                            matched=False,
                            child_visited=False,
                            skip_reason="parent_gate_not_matched",
                            child_subtree_vote=None,
                        )
                    )
            subtree_vote = aggregate_child_subtrees(decision.vote, child_votes)
            trace = NodeExecutionTrace(
                node_id=node_id,
                criterion_name=node.criterion.name,
                depth=self.rubric.depth(node_id),
                output=result.output,
                local_vote=decision.vote,
                outcome_reason=decision.outcome_reason,
                backend_source=result.source,
                metrics=result.metrics,
                generation_metrics=result.generation_metrics,
                cache_key=result.cache_key,
                edges=tuple(edge_traces),
                subtree_vote=subtree_vote,
            )
            with memo_lock:
                if node_id in memo:
                    raise RuntimeError(f"node evaluated more than once: {node_id}")
                memo[node_id] = subtree_vote
                node_traces[node_id] = trace
            return subtree_vote

        selected_root_votes: dict[str, Vote] = {}
        selected_roots = [root_id for root_id in self.rubric.root_ids if root_id in selected]
        if selected_roots and config.max_concurrent > 1:
            with ThreadPoolExecutor(max_workers=min(config.max_concurrent, len(selected_roots))) as pool:
                futures = {root_id: pool.submit(evaluate_subtree, root_id) for root_id in selected_roots}
                for root_id in selected_roots:
                    selected_root_votes[root_id] = futures[root_id].result()
        else:
            for root_id in selected_roots:
                selected_root_votes[root_id] = evaluate_subtree(root_id)

        ordered_node_traces = tuple(
            node_traces[node_id]
            for node_id in self.rubric.preorder_node_ids()
            if node_id in node_traces
        )
        root_traces: list[RootExecutionTrace] = []
        visited_ids = set(node_traces)
        for root_id in self.rubric.root_ids:
            subtree_ids = self._subtree_node_ids(root_id)
            visited = tuple(node_id for node_id in subtree_ids if node_id in visited_ids)
            avoided = tuple(node_id for node_id in subtree_ids if node_id not in visited_ids)
            root_traces.append(
                RootExecutionTrace(
                    root_id=root_id,
                    selected=root_id in selected,
                    skip_reason=None if root_id in selected else "root_not_selected",
                    visited_node_ids=visited,
                    avoided_node_ids=avoided,
                    subtree_vote=selected_root_votes.get(root_id),
                )
            )

        if config.aggregation_mode is AggregationMode.FLAT:
            aggregation_inputs = {
                trace.node_id: trace.local_vote for trace in ordered_node_traces
            }
            final_preference = aggregate_flat_votes(aggregation_inputs.values())
        else:
            aggregation_inputs = {
                root_id: selected_root_votes[root_id] for root_id in selected_roots
            }
            final_preference = aggregate_selected_roots(
                aggregation_inputs,
                tuple(selected_roots),
            )

        matched_edges = sum(edge.matched for node in ordered_node_traces for edge in node.edges)
        skipped_edges = sum(not edge.matched for node in ordered_node_traces for edge in node.edges)
        node_metrics = combine_model_call_metrics(trace.metrics for trace in ordered_node_traces)
        counterfactual = CounterfactualTraversalMetrics(
            total_nodes=len(self.rubric.nodes),
            visited_nodes=len(ordered_node_traces),
            avoided_nodes=len(self.rubric.nodes) - len(ordered_node_traces),
            matched_edges=matched_edges,
            skipped_edges=skipped_edges,
            max_visited_depth=max((trace.depth for trace in ordered_node_traces), default=0),
            selected_roots=len(selected_roots),
        )
        return ExecutionTrace(
            sample_id=sample_id,
            sample_fingerprint=sample_fingerprint,
            rubric_sha256=self.rubric.rubric_sha256,
            variant=config.variant.value,
            traversal_policy=config.traversal_policy,
            aggregation_mode=config.aggregation_mode,
            router=router_trace,
            roots=tuple(root_traces),
            nodes=ordered_node_traces,
            aggregation_input_votes=aggregation_inputs,
            final_preference=final_preference,
            counterfactual_metrics=counterfactual,
            node_metrics=node_metrics,
            wall_latency_seconds=time.perf_counter() - started_at,
        )

    def execute_batch(
        self,
        dataset: Sequence[PairData],
        config: ExecutionConfig,
        on_trace: Callable[[int, ExecutionTrace], None] | None = None,
    ) -> StructuredExecutionOutput:
        if not dataset:
            raise ValueError("execution dataset must not be empty")
        if not isinstance(config, ExecutionConfig):
            raise TypeError("config must be ExecutionConfig")
        if on_trace is not None and not callable(on_trace):
            raise TypeError("on_trace must be callable or None")

        request_semaphore = threading.BoundedSemaphore(
            config.request_concurrent_limit)
        ordered: list[ExecutionTrace | None] = [None] * len(dataset)

        def run(index: int, data: PairData) -> tuple[int, ExecutionTrace]:
            return index, self._execute_one(data, config, request_semaphore)

        if config.sample_concurrent > 1 and len(dataset) > 1:
            with ThreadPoolExecutor(
                max_workers=min(config.sample_concurrent, len(dataset))
            ) as pool:
                futures = [
                    pool.submit(run, index, data)
                    for index, data in enumerate(dataset)
                ]
                for future in as_completed(futures):
                    index, trace = future.result()
                    ordered[index] = trace
                    if on_trace is not None:
                        on_trace(index, trace)
        else:
            for index, data in enumerate(dataset):
                _, trace = run(index, data)
                ordered[index] = trace
                if on_trace is not None:
                    on_trace(index, trace)

        if any(trace is None for trace in ordered):
            raise RuntimeError("batch execution did not produce every trace")
        traces = tuple(trace for trace in ordered if trace is not None)
        return StructuredExecutionOutput(
            variant=config.variant.value,
            rubric_sha256=self.rubric.rubric_sha256,
            traces=traces,
        )

    def evaluate_batch(self, dataset: Sequence[PairData], config: ExecutionConfig) -> StructuredExecutionEvaluation:
        prediction = self.execute_batch(dataset, config)
        is_correct = tuple(
            trace.final_preference in {FinalPreference.A, FinalPreference.B}
            and trace.final_preference.value == data["answer"]
            for trace, data in zip(prediction.traces, dataset)
        )
        decisive = sum(trace.final_preference in {FinalPreference.A, FinalPreference.B} for trace in prediction.traces)
        ties = len(prediction.traces) - decisive
        return StructuredExecutionEvaluation(
            prediction=prediction,
            is_correct=is_correct,
            accuracy=sum(is_correct) / len(is_correct),
            coverage=decisive / len(prediction.traces),
            tie_rate=ties / len(prediction.traces),
        )


def replay_execution_trace(
    rubric: StructuredRubric,
    trace: ExecutionTrace,
) -> FinalPreference:
    """Rebuild every routing/traversal/aggregation decision without I/O."""

    if trace.rubric_sha256 != rubric.rubric_sha256:
        raise ValueError("trace rubric hash mismatch")
    try:
        variant = StructuredSystemVariant(trace.variant)
    except ValueError as exc:
        raise ValueError("trace variant is unknown") from exc
    config = ExecutionConfig(variant=variant)
    if trace.traversal_policy is not config.traversal_policy:
        raise ValueError("trace traversal policy conflicts with variant")
    if trace.aggregation_mode is not config.aggregation_mode:
        raise ValueError("trace aggregation mode conflicts with variant")
    expected_node_metrics = combine_model_call_metrics(
        node.metrics for node in trace.nodes
    )
    if trace.node_metrics != expected_node_metrics:
        raise ValueError("trace node_metrics cannot be reconstructed")

    if config.router_enabled:
        decision = trace.router.output.decision if trace.router.output is not None else None
        expected_routing = resolve_root_routing(decision, rubric.root_ids, enabled=True)
    else:
        if trace.router.output is not None or trace.router.enabled:
            raise ValueError("router-disabled trace contains router output")
        expected_routing = resolve_root_routing(None, rubric.root_ids, enabled=False)
    if trace.router.resolved != expected_routing:
        raise ValueError("trace resolved routing cannot be reconstructed")

    selected = set(expected_routing.selected_root_ids)
    if tuple(root.root_id for root in trace.roots) != rubric.root_ids:
        raise ValueError("trace root order mismatch")
    node_by_id = {node.node_id: node for node in trace.nodes}
    visited: set[str] = set()

    def replay_subtree(node_id: str) -> Vote:
        if node_id not in node_by_id:
            raise ValueError(f"trace is missing visited node {node_id}")
        if node_id in visited:
            raise ValueError(f"trace visits node more than once: {node_id}")
        visited.add(node_id)
        node_trace = node_by_id[node_id]
        rubric_node = rubric.get_node(node_id)
        if node_trace.criterion_name != rubric_node.criterion.name or node_trace.depth != rubric.depth(node_id):
            raise ValueError(f"trace node identity mismatch: {node_id}")
        decision = node_trace.output.local_decision
        if node_trace.local_vote is not decision.vote or node_trace.outcome_reason is not decision.outcome_reason:
            raise ValueError(f"trace local decision mismatch: {node_id}")
        rubric_edges = rubric.child_edges(node_id)
        if len(node_trace.edges) != len(rubric_edges):
            raise ValueError(f"trace edge count mismatch: {node_id}")
        child_votes: list[Vote] = []
        for rubric_edge, edge_trace in zip(rubric_edges, node_trace.edges):
            expected_match = should_visit_child(
                node_trace.output.judgement,
                rubric_edge.condition,
                config.traversal_policy,
            )
            if (
                edge_trace.parent_id != rubric_edge.parent_id
                or edge_trace.child_id != rubric_edge.child_id
                or edge_trace.condition != rubric_edge.condition.value
                or edge_trace.matched != expected_match
                or edge_trace.child_visited != expected_match
            ):
                raise ValueError(f"trace edge decision mismatch: {rubric_edge.parent_id}->{rubric_edge.child_id}")
            if expected_match:
                child_vote = replay_subtree(rubric_edge.child_id)
                if edge_trace.skip_reason is not None or edge_trace.child_subtree_vote is not child_vote:
                    raise ValueError("trace visited-edge payload mismatch")
                child_votes.append(child_vote)
            elif edge_trace.skip_reason != "parent_gate_not_matched" or edge_trace.child_subtree_vote is not None:
                raise ValueError("trace skipped-edge payload mismatch")
        subtree_vote = aggregate_child_subtrees(decision.vote, child_votes)
        if node_trace.subtree_vote is not subtree_vote:
            raise ValueError(f"trace subtree vote mismatch: {node_id}")
        return subtree_vote

    root_votes: dict[str, Vote] = {}
    for root_trace in trace.roots:
        subtree_ids = tuple(
            node_id
            for node_id in rubric.preorder_node_ids()
            if rubric.root_id_for(node_id) == root_trace.root_id
        )
        expected_selected = root_trace.root_id in selected
        if root_trace.selected != expected_selected:
            raise ValueError("trace selected-root flag mismatch")
        if expected_selected:
            vote = replay_subtree(root_trace.root_id)
            root_votes[root_trace.root_id] = vote
            if root_trace.skip_reason is not None or root_trace.subtree_vote is not vote:
                raise ValueError("trace selected root payload mismatch")
        elif root_trace.skip_reason != "root_not_selected" or root_trace.subtree_vote is not None:
            raise ValueError("trace unselected root payload mismatch")
        expected_visited = tuple(node_id for node_id in subtree_ids if node_id in visited)
        expected_avoided = tuple(node_id for node_id in subtree_ids if node_id not in visited)
        if root_trace.visited_node_ids != expected_visited or root_trace.avoided_node_ids != expected_avoided:
            raise ValueError("trace root visited/avoided nodes mismatch")

    if visited != set(node_by_id):
        raise ValueError("trace contains nodes outside reconstructed traversal")
    ordered_nodes = tuple(
        node_by_id[node_id] for node_id in rubric.preorder_node_ids() if node_id in node_by_id
    )
    selected_in_order = tuple(root_id for root_id in rubric.root_ids if root_id in selected)
    if config.aggregation_mode is AggregationMode.FLAT:
        aggregation_inputs = {node.node_id: node.local_vote for node in ordered_nodes}
        final = aggregate_flat_votes(aggregation_inputs.values())
    else:
        aggregation_inputs = {root_id: root_votes[root_id] for root_id in selected_in_order}
        final = aggregate_selected_roots(aggregation_inputs, selected_in_order)
    if dict(trace.aggregation_input_votes) != aggregation_inputs:
        raise ValueError("trace aggregation inputs mismatch")
    if trace.final_preference is not final:
        raise ValueError("trace final preference mismatch")

    edge_values = [edge for node in ordered_nodes for edge in node.edges]
    counterfactual = CounterfactualTraversalMetrics(
        total_nodes=len(rubric.nodes),
        visited_nodes=len(ordered_nodes),
        avoided_nodes=len(rubric.nodes) - len(ordered_nodes),
        matched_edges=sum(edge.matched for edge in edge_values),
        skipped_edges=sum(not edge.matched for edge in edge_values),
        max_visited_depth=max((node.depth for node in ordered_nodes), default=0),
        selected_roots=len(selected_in_order),
    )
    if trace.counterfactual_metrics != counterfactual:
        raise ValueError("trace counterfactual metrics mismatch")
    return final

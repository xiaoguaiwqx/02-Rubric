"""Immutable rubric patching, diffing and artifact impact planning."""

from __future__ import annotations

from ..semantics import EdgeCondition
from ..schema import RubricEdge, StructuredRubric
from .types import ArtifactRefreshPlan, RubricDiff, RubricPatch


_STATUS_CONDITIONS = {
    EdgeCondition.PARENT_BOTH_PASS,
    EdgeCondition.PARENT_BOTH_FAIL,
}


def apply_rubric_patch(rubric: StructuredRubric, patch: RubricPatch) -> StructuredRubric:
    if patch.base_rubric_sha256 != rubric.rubric_sha256:
        raise ValueError("stale RubricPatch base hash")
    nodes = dict(rubric.nodes)
    edges = list(rubric.edges)
    for node_id in patch.remove_node_ids:
        if node_id not in nodes:
            raise KeyError(node_id)
        del nodes[node_id]
        edges = [edge for edge in edges if edge.parent_id != node_id and edge.child_id != node_id]
    for node in patch.upsert_nodes:
        nodes[node.node_id] = node
    for edge in patch.remove_edges:
        if edge not in edges:
            raise ValueError(f"cannot remove unknown edge {edge}")
        edges.remove(edge)
    for edge in patch.add_edges:
        if edge in edges:
            raise ValueError(f"cannot add duplicate edge {edge}")
        edges.append(edge)
    root_ids = patch.root_ids if patch.root_ids is not None else tuple(
        root_id for root_id in rubric.root_ids if root_id in nodes
    )
    return StructuredRubric(nodes=nodes, edges=tuple(edges), root_ids=root_ids)


def diff_rubrics(before: StructuredRubric, after: StructuredRubric) -> RubricDiff:
    before_ids, after_ids = set(before.nodes), set(after.nodes)
    common = before_ids & after_ids
    return RubricDiff(
        added_node_ids=tuple(sorted(after_ids - before_ids)),
        removed_node_ids=tuple(sorted(before_ids - after_ids)),
        modified_node_ids=tuple(sorted(
            node_id for node_id in common if before.nodes[node_id] != after.nodes[node_id]
        )),
        added_edges=tuple(sorted(set(after.edges) - set(before.edges), key=lambda item: (
            item.parent_id, item.child_id, item.condition.value,
        ))),
        removed_edges=tuple(sorted(set(before.edges) - set(after.edges), key=lambda item: (
            item.parent_id, item.child_id, item.condition.value,
        ))),
        root_ids_changed=before.root_ids != after.root_ids,
    )


def _status_parents(rubric: StructuredRubric) -> set[str]:
    return {
        edge.parent_id for edge in rubric.edges if edge.condition in _STATUS_CONDITIONS
    }


def plan_artifact_refresh(
    before: StructuredRubric,
    after: StructuredRubric,
) -> ArtifactRefreshPlan:
    diff = diff_rubrics(before, after)
    refresh_pairwise = set(diff.added_node_ids)
    for node_id in diff.modified_node_ids:
        old, new = before.nodes[node_id], after.nodes[node_id]
        if old.criterion.name != new.criterion.name or old.criterion.description != new.criterion.description:
            refresh_pairwise.add(node_id)
    removed_pairwise = tuple(sorted(
        before.nodes[node_id].criterion.name for node_id in diff.removed_node_ids
    ))

    old_status, new_status = _status_parents(before), _status_parents(after)
    refresh_gate = set(new_status - old_status)
    for node_id in diff.modified_node_ids:
        if node_id in new_status:
            old, new = before.nodes[node_id], after.nodes[node_id]
            if old.criterion.name != new.criterion.name or old.criterion.description != new.criterion.description:
                refresh_gate.add(node_id)
    removed_gate = tuple(sorted(
        before.nodes[node_id].criterion.name
        for node_id in old_status
        if node_id not in new_status or node_id not in after.nodes
    ))

    before_roots = {
        node_id: before.nodes[node_id].criterion.description for node_id in before.root_ids
    }
    after_roots = {
        node_id: after.nodes[node_id].criterion.description for node_id in after.root_ids
    }
    return ArtifactRefreshPlan(
        refresh_pairwise_node_ids=tuple(sorted(refresh_pairwise)),
        remove_pairwise_criterion_names=removed_pairwise,
        refresh_gate_node_ids=tuple(sorted(refresh_gate)),
        remove_gate_criterion_names=removed_gate,
        router_stale=(before.root_ids != after.root_ids or before_roots != after_roots),
    )

"""Topology and cross-artifact validation for structured rubric forests."""

from __future__ import annotations

from .schema import RubricValidationError, StructuredRubric


def validate_structured_rubric(rubric: StructuredRubric) -> None:
    """Validate that a typed rubric is exactly a rooted single-parent forest."""

    if not rubric.nodes:
        raise RubricValidationError("structured rubric must contain at least one node")
    if not rubric.root_ids:
        raise RubricValidationError("structured rubric must contain at least one root")
    if len(set(rubric.root_ids)) != len(rubric.root_ids):
        raise RubricValidationError("root_ids must not contain duplicates")

    for mapping_id, node in rubric.nodes.items():
        if mapping_id != node.node_id:
            raise RubricValidationError(
                f"node mapping key {mapping_id!r} does not match node_id "
                f"{node.node_id!r}"
            )

    criterion_names = [node.criterion.name for node in rubric.nodes.values()]
    if len(set(criterion_names)) != len(criterion_names):
        raise RubricValidationError("criterion names must be globally unique")

    unknown_roots = [root_id for root_id in rubric.root_ids if root_id not in rubric.nodes]
    if unknown_roots:
        raise RubricValidationError(f"root_ids contains unknown nodes: {unknown_roots}")

    parent_by_id: dict[str, str] = {}
    child_ids_by_parent: dict[str, list[str]] = {
        node_id: [] for node_id in rubric.nodes
    }
    edge_pairs: set[tuple[str, str]] = set()
    for edge in rubric.edges:
        if edge.parent_id not in rubric.nodes:
            raise RubricValidationError(
                f"edge references unknown parent: {edge.parent_id}"
            )
        if edge.child_id not in rubric.nodes:
            raise RubricValidationError(
                f"edge references unknown child: {edge.child_id}"
            )
        if edge.parent_id == edge.child_id:
            raise RubricValidationError(f"self-edge is not allowed: {edge.parent_id}")
        pair = (edge.parent_id, edge.child_id)
        if pair in edge_pairs:
            raise RubricValidationError(
                f"duplicate parent-child edge: {edge.parent_id}->{edge.child_id}"
            )
        edge_pairs.add(pair)
        if edge.child_id in parent_by_id:
            raise RubricValidationError(
                f"node {edge.child_id!r} has multiple parents: "
                f"{parent_by_id[edge.child_id]!r} and {edge.parent_id!r}"
            )
        parent_by_id[edge.child_id] = edge.parent_id
        child_ids_by_parent[edge.parent_id].append(edge.child_id)

    expected_roots = {
        node_id for node_id in rubric.nodes if node_id not in parent_by_id
    }
    declared_roots = set(rubric.root_ids)
    if declared_roots != expected_roots:
        missing = sorted(expected_roots - declared_roots)
        extra = sorted(declared_roots - expected_roots)
        raise RubricValidationError(
            "root_ids must equal all indegree-zero nodes: "
            f"missing={missing}, extra={extra}"
        )

    visit_state: dict[str, int] = {node_id: 0 for node_id in rubric.nodes}

    def detect_cycle(node_id: str) -> None:
        if visit_state[node_id] == 1:
            raise RubricValidationError(f"cycle detected at node {node_id!r}")
        if visit_state[node_id] == 2:
            return
        visit_state[node_id] = 1
        for child_id in child_ids_by_parent[node_id]:
            detect_cycle(child_id)
        visit_state[node_id] = 2

    for node_id in rubric.nodes:
        detect_cycle(node_id)

    owner_by_id: dict[str, str] = {}

    def mark_owner(node_id: str, root_id: str) -> None:
        previous = owner_by_id.get(node_id)
        if previous is not None:
            raise RubricValidationError(
                f"node {node_id!r} is reachable from multiple roots: "
                f"{previous!r} and {root_id!r}"
            )
        owner_by_id[node_id] = root_id
        for child_id in child_ids_by_parent[node_id]:
            mark_owner(child_id, root_id)

    for root_id in rubric.root_ids:
        mark_owner(root_id, root_id)
    unreachable = sorted(set(rubric.nodes) - set(owner_by_id))
    if unreachable:
        raise RubricValidationError(f"nodes are unreachable from roots: {unreachable}")

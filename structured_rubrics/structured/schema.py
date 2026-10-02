"""Immutable, versioned tree/forest schema for structured rubrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, Union

from ..utils import Criterion
from .semantics import EdgeCondition
from .version import (
    STRUCTURED_RUBRIC_SCHEMA_VERSION,
    STRUCTURED_SEMANTICS_VERSION,
)


JSONValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    Mapping[str, "JSONValue"],
    tuple["JSONValue", ...],
    list["JSONValue"],
]


class RubricSchemaError(ValueError):
    """Raised when serialized rubric data violates the frozen wire schema."""


class RubricValidationError(ValueError):
    """Raised when rubric identities or forest topology are invalid."""


def _require_clean_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RubricSchemaError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise RubricSchemaError(f"{label} must not have surrounding whitespace")
    return value


def _freeze_json(value: Any, *, label: str) -> Any:
    """Defensively copy JSON-compatible data into immutable containers."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise RubricSchemaError(f"{label} must not contain non-finite floats")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RubricSchemaError(f"{label} keys must be strings")
            frozen[key] = _freeze_json(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise RubricSchemaError(f"{label} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise RubricSchemaError(
            f"{label} fields must match exactly: {', '.join(details)}"
        )


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RubricSchemaError(f"{label} must be a SHA-256 hex string")
    return value


@dataclass(frozen=True)
class RubricCriterionSnapshot:
    """Immutable criterion text and historical score stored by one node."""

    name: str
    description: str
    score: float = 0.0

    def __post_init__(self) -> None:
        _require_clean_identifier(self.name, label="criterion name")
        if not isinstance(self.description, str) or not self.description.strip():
            raise RubricSchemaError("criterion description must be a non-empty string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise RubricSchemaError("criterion score must be a number in [0, 1]")
        normalized_score = float(self.score)
        if not math.isfinite(normalized_score) or not 0.0 <= normalized_score <= 1.0:
            raise RubricSchemaError("criterion score must be a finite number in [0, 1]")
        object.__setattr__(self, "score", normalized_score)

    @classmethod
    def from_criterion(cls, criterion: Criterion) -> "RubricCriterionSnapshot":
        if not isinstance(criterion, Criterion):
            raise TypeError("criterion must be Criterion")
        return cls(
            name=criterion.name,
            description=criterion.description,
            score=criterion.score,
        )

    def to_criterion(self) -> Criterion:
        return Criterion(
            name=self.name,
            description=self.description,
            score=self.score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RubricCriterionSnapshot":
        if not isinstance(value, Mapping):
            raise RubricSchemaError("criterion snapshot must be an object")
        _require_exact_keys(
            value,
            frozenset({"name", "description", "score"}),
            label="criterion snapshot",
        )
        return cls(
            name=value["name"],
            description=value["description"],
            score=value["score"],
        )


@dataclass(frozen=True)
class RubricNode:
    """One executable criterion node plus opaque evolution metadata."""

    node_id: str
    criterion: RubricCriterionSnapshot
    examples: tuple[Mapping[str, JSONValue], ...] = ()
    lineage: Mapping[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        _require_clean_identifier(self.node_id, label="node_id")
        if not isinstance(self.criterion, RubricCriterionSnapshot):
            raise RubricSchemaError("criterion must be RubricCriterionSnapshot")
        if isinstance(self.examples, (str, bytes)) or not isinstance(
            self.examples, Sequence
        ):
            raise RubricSchemaError("examples must be an ordered sequence")
        frozen_examples: list[Mapping[str, JSONValue]] = []
        for index, example in enumerate(self.examples):
            if not isinstance(example, Mapping):
                raise RubricSchemaError(f"examples[{index}] must be an object")
            frozen_examples.append(
                _freeze_json(example, label=f"examples[{index}]")
            )
        object.__setattr__(self, "examples", tuple(frozen_examples))

        if self.lineage is not None:
            if not isinstance(self.lineage, Mapping):
                raise RubricSchemaError("lineage must be an object or None")
            object.__setattr__(
                self,
                "lineage",
                _freeze_json(self.lineage, label="lineage"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "criterion": self.criterion.to_dict(),
            "examples": [_thaw_json(example) for example in self.examples],
            "lineage": None if self.lineage is None else _thaw_json(self.lineage),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RubricNode":
        if not isinstance(value, Mapping):
            raise RubricSchemaError("rubric node must be an object")
        _require_exact_keys(
            value,
            frozenset({"node_id", "criterion", "examples", "lineage"}),
            label="rubric node",
        )
        examples = value["examples"]
        if not isinstance(examples, list):
            raise RubricSchemaError("rubric node examples must be a list")
        lineage = value["lineage"]
        if lineage is not None and not isinstance(lineage, Mapping):
            raise RubricSchemaError("rubric node lineage must be an object or null")
        return cls(
            node_id=value["node_id"],
            criterion=RubricCriterionSnapshot.from_dict(value["criterion"]),
            examples=tuple(examples),
            lineage=lineage,
        )


@dataclass(frozen=True)
class RubricEdge:
    """One directed, condition-bearing parent-child relationship."""

    parent_id: str
    child_id: str
    condition: EdgeCondition

    def __post_init__(self) -> None:
        _require_clean_identifier(self.parent_id, label="edge parent_id")
        _require_clean_identifier(self.child_id, label="edge child_id")
        if not isinstance(self.condition, EdgeCondition):
            raise RubricSchemaError("edge condition must be EdgeCondition")

    def to_dict(self) -> dict[str, str]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "condition": self.condition.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RubricEdge":
        if not isinstance(value, Mapping):
            raise RubricSchemaError("rubric edge must be an object")
        _require_exact_keys(
            value,
            frozenset({"parent_id", "child_id", "condition"}),
            label="rubric edge",
        )
        try:
            condition = EdgeCondition(value["condition"])
        except (TypeError, ValueError) as exc:
            raise RubricSchemaError("invalid edge condition") from exc
        return cls(
            parent_id=value["parent_id"],
            child_id=value["child_id"],
            condition=condition,
        )


@dataclass(frozen=True)
class StructuredRubric:
    """An immutable, validated forest of structured rubric nodes."""

    nodes: Mapping[str, RubricNode]
    edges: tuple[RubricEdge, ...]
    root_ids: tuple[str, ...]
    schema_version: str = STRUCTURED_RUBRIC_SCHEMA_VERSION
    semantics_version: str = STRUCTURED_SEMANTICS_VERSION
    _parent_by_id: Mapping[str, str] = field(init=False, repr=False, compare=False)
    _child_edges_by_id: Mapping[str, tuple[RubricEdge, ...]] = field(
        init=False, repr=False, compare=False
    )
    _root_by_id: Mapping[str, str] = field(init=False, repr=False, compare=False)
    _depth_by_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _node_id_by_criterion: Mapping[str, str] = field(
        init=False, repr=False, compare=False
    )
    _preorder: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != STRUCTURED_RUBRIC_SCHEMA_VERSION:
            raise RubricSchemaError(
                "schema_version mismatch: expected "
                f"{STRUCTURED_RUBRIC_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.semantics_version != STRUCTURED_SEMANTICS_VERSION:
            raise RubricSchemaError(
                "semantics_version mismatch: expected "
                f"{STRUCTURED_SEMANTICS_VERSION}, got {self.semantics_version}"
            )
        if not isinstance(self.nodes, Mapping):
            raise RubricSchemaError("nodes must be a mapping")
        copied_nodes: dict[str, RubricNode] = {}
        for node_id, node in self.nodes.items():
            if not isinstance(node_id, str):
                raise RubricSchemaError("node mapping keys must be strings")
            if not isinstance(node, RubricNode):
                raise RubricSchemaError("node mapping values must be RubricNode")
            copied_nodes[node_id] = node
        object.__setattr__(self, "nodes", MappingProxyType(copied_nodes))

        if isinstance(self.edges, (str, bytes)) or not isinstance(
            self.edges, Sequence
        ):
            raise RubricSchemaError("edges must be an ordered sequence")
        copied_edges = tuple(self.edges)
        if any(not isinstance(edge, RubricEdge) for edge in copied_edges):
            raise RubricSchemaError("every edge must be RubricEdge")
        object.__setattr__(
            self,
            "edges",
            tuple(
                sorted(
                    copied_edges,
                    key=lambda edge: (
                        edge.parent_id,
                        edge.child_id,
                        edge.condition.value,
                    ),
                )
            ),
        )

        if isinstance(self.root_ids, (str, bytes)) or not isinstance(
            self.root_ids, Sequence
        ):
            raise RubricSchemaError("root_ids must be an ordered sequence")
        copied_roots = tuple(self.root_ids)
        for root_id in copied_roots:
            _require_clean_identifier(root_id, label="root ID")
        object.__setattr__(self, "root_ids", copied_roots)

        from .validation import validate_structured_rubric

        validate_structured_rubric(self)
        self._build_indexes()

    def _build_indexes(self) -> None:
        parent_by_id = {edge.child_id: edge.parent_id for edge in self.edges}
        child_edges: dict[str, list[RubricEdge]] = {
            node_id: [] for node_id in self.nodes
        }
        for edge in self.edges:
            child_edges[edge.parent_id].append(edge)
        sorted_child_edges = {
            node_id: tuple(
                sorted(values, key=lambda edge: (edge.child_id, edge.condition.value))
            )
            for node_id, values in child_edges.items()
        }

        root_by_id: dict[str, str] = {}
        depth_by_id: dict[str, int] = {}
        preorder: list[str] = []

        def visit(node_id: str, root_id: str, depth: int) -> None:
            root_by_id[node_id] = root_id
            depth_by_id[node_id] = depth
            preorder.append(node_id)
            for edge in sorted_child_edges[node_id]:
                visit(edge.child_id, root_id, depth + 1)

        for root_id in self.root_ids:
            visit(root_id, root_id, 0)

        node_id_by_criterion = {
            node.criterion.name: node_id for node_id, node in self.nodes.items()
        }
        object.__setattr__(self, "_parent_by_id", MappingProxyType(parent_by_id))
        object.__setattr__(
            self,
            "_child_edges_by_id",
            MappingProxyType(sorted_child_edges),
        )
        object.__setattr__(self, "_root_by_id", MappingProxyType(root_by_id))
        object.__setattr__(self, "_depth_by_id", MappingProxyType(depth_by_id))
        object.__setattr__(
            self,
            "_node_id_by_criterion",
            MappingProxyType(node_id_by_criterion),
        )
        object.__setattr__(self, "_preorder", tuple(preorder))

    def get_node(self, node_id: str) -> RubricNode:
        return self.nodes[node_id]

    def parent_id(self, node_id: str) -> str | None:
        self.get_node(node_id)
        return self._parent_by_id.get(node_id)

    def child_edges(self, node_id: str) -> tuple[RubricEdge, ...]:
        self.get_node(node_id)
        return self._child_edges_by_id[node_id]

    def children(self, node_id: str) -> tuple[RubricNode, ...]:
        return tuple(self.nodes[edge.child_id] for edge in self.child_edges(node_id))

    def root_id_for(self, node_id: str) -> str:
        self.get_node(node_id)
        return self._root_by_id[node_id]

    def depth(self, node_id: str) -> int:
        self.get_node(node_id)
        return self._depth_by_id[node_id]

    def preorder_node_ids(self) -> tuple[str, ...]:
        return self._preorder

    def criteria_in_execution_order(self) -> tuple[Criterion, ...]:
        return tuple(
            self.nodes[node_id].criterion.to_criterion()
            for node_id in self._preorder
        )

    def criterion_name_for(self, node_id: str) -> str:
        return self.get_node(node_id).criterion.name

    def node_id_for_criterion(self, name: str) -> str:
        return self._node_id_by_criterion[name]

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantics_version": self.semantics_version,
            "root_ids": list(self.root_ids),
            "nodes": [self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)],
            "edges": [
                edge.to_dict()
                for edge in sorted(
                    self.edges,
                    key=lambda value: (
                        value.parent_id,
                        value.child_id,
                        value.condition.value,
                    ),
                )
            ],
        }

    @property
    def rubric_sha256(self) -> str:
        encoded = json.dumps(
            self._canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self._canonical_payload()
        return {
            "schema_version": payload["schema_version"],
            "semantics_version": payload["semantics_version"],
            "rubric_sha256": self.rubric_sha256,
            "root_ids": payload["root_ids"],
            "nodes": payload["nodes"],
            "edges": payload["edges"],
        }

    @classmethod
    def from_dict(cls, value: object) -> "StructuredRubric":
        if not isinstance(value, Mapping):
            raise RubricSchemaError("structured rubric must be an object")
        _require_exact_keys(
            value,
            frozenset(
                {
                    "schema_version",
                    "semantics_version",
                    "rubric_sha256",
                    "root_ids",
                    "nodes",
                    "edges",
                }
            ),
            label="structured rubric",
        )
        serialized_hash = _require_sha256(
            value["rubric_sha256"], label="rubric_sha256"
        )
        raw_nodes = value["nodes"]
        raw_edges = value["edges"]
        raw_roots = value["root_ids"]
        if not isinstance(raw_nodes, list):
            raise RubricSchemaError("structured rubric nodes must be a list")
        if not isinstance(raw_edges, list):
            raise RubricSchemaError("structured rubric edges must be a list")
        if not isinstance(raw_roots, list):
            raise RubricSchemaError("structured rubric root_ids must be a list")

        nodes_list = [RubricNode.from_dict(item) for item in raw_nodes]
        node_ids = [node.node_id for node in nodes_list]
        if len(set(node_ids)) != len(node_ids):
            raise RubricValidationError("serialized nodes contain duplicate node IDs")
        rubric = cls(
            nodes={node.node_id: node for node in nodes_list},
            edges=tuple(RubricEdge.from_dict(item) for item in raw_edges),
            root_ids=tuple(raw_roots),
            schema_version=value["schema_version"],
            semantics_version=value["semantics_version"],
        )
        if rubric.rubric_sha256 != serialized_hash:
            raise RubricSchemaError("rubric_sha256 does not match rubric content")
        return rubric

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "StructuredRubric":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RubricSchemaError(f"failed to load structured rubric: {exc}") from exc
        return cls.from_dict(value)

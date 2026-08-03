"""Frozen Phase 4 criteria import and manual forest construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.structured import (
    EdgeCondition,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredRubric,
)

CRITERIA_SOURCE_SHA256 = (
    "20fa11ab4516cbc510e2ae65d5605137d086676f7821d6a65ec20c1355c7f0ed"
)
HELDOUT_DATASET_SHA256 = (
    "fa57ca429469d3c2c014e7aef42801de43befad3fc600738da7c9fb12fd312c7"
)
EXPECTED_CRITERIA_COUNT = 17

NODE_IDS = (
    "c01_visual_grounding",
    "c02_factual_consistency",
    "c03_avoidance_of_hallucination",
    "c04_completeness",
    "c05_specificity",
    "c06_calibrated_uncertainty",
    "c07_biological_or_realistic_plausibility",
    "c08_logical_consistency",
    "c09_multimodal_alignment",
    "c10_action_or_state_accuracy",
    "c11_temporal_coherence",
    "c12_visual_coherence",
    "c13_semantic_specificity",
    "c14_contextual_fidelity",
    "c15_multimodal_consistency",
    "c16_visual_ambiguity_resolution",
    "c17_temporal_consistency",
)

EXPECTED_CRITERION_NAMES = tuple(node_id.split("_", 1)[1] for node_id in NODE_IDS)
NODE_ID_BY_CRITERION = dict(zip(EXPECTED_CRITERION_NAMES, NODE_IDS))

STRICT_CHILDREN = {
    "visual_coherence",
    "semantic_specificity",
}
STRICT_ROOT_NAMES = tuple(
    name for name in EXPECTED_CRITERION_NAMES if name not in STRICT_CHILDREN
)
STRICT_EDGES = (
    ("multimodal_alignment", "visual_coherence"),
    ("multimodal_alignment", "semantic_specificity"),
)

PROJECTED_EDGES = (
    ("visual_grounding", "contextual_fidelity"),
    ("factual_consistency", "completeness"),
    ("factual_consistency", "biological_or_realistic_plausibility"),
    ("multimodal_alignment", "temporal_coherence"),
    ("multimodal_alignment", "visual_coherence"),
    ("multimodal_alignment", "semantic_specificity"),
    ("multimodal_alignment", "visual_ambiguity_resolution"),
    ("multimodal_alignment", "temporal_consistency"),
)
PROJECTED_CHILDREN = {child for _, child in PROJECTED_EDGES}
PROJECTED_ROOT_NAMES = tuple(
    name for name in EXPECTED_CRITERION_NAMES if name not in PROJECTED_CHILDREN
)

MULTI_PARENT_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "completeness": ("visual_grounding", "factual_consistency"),
    "biological_or_realistic_plausibility": (
        "visual_grounding",
        "factual_consistency",
    ),
    "temporal_coherence": ("visual_grounding", "multimodal_alignment"),
    "contextual_fidelity": ("visual_grounding", "factual_consistency"),
    "visual_ambiguity_resolution": (
        "visual_grounding",
        "multimodal_alignment",
    ),
    "temporal_consistency": ("visual_grounding", "multimodal_alignment"),
}

PROJECTED_PRIMARY_PARENT = {
    child: parent for parent, child in PROJECTED_EDGES
}


class Phase4InputError(ValueError):
    """Raised when a frozen Phase 4 source or topology has drifted."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_source(path: Path, expected_sha256: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise Phase4InputError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def load_frozen_criteria(path: str | Path) -> tuple[RubricCriterionSnapshot, ...]:
    source = Path(path)
    _require_exact_source(source, CRITERIA_SOURCE_SHA256, "criteria source")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase4InputError(f"failed to load criteria source: {exc}") from exc
    if not isinstance(payload, dict) or "criteria" not in payload:
        raise Phase4InputError("criteria source must contain top-level criteria")
    raw = payload["criteria"]
    if not isinstance(raw, list) or len(raw) != EXPECTED_CRITERIA_COUNT:
        raise Phase4InputError("criteria source must contain exactly 17 criteria")
    snapshots: list[RubricCriterionSnapshot] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"name", "description", "score"}:
            raise Phase4InputError(
                f"criteria[{index}] must contain exactly name/description/score"
            )
        snapshots.append(RubricCriterionSnapshot.from_dict(item))
    names = tuple(snapshot.name for snapshot in snapshots)
    if names != EXPECTED_CRITERION_NAMES:
        raise Phase4InputError(
            "criteria names/order drifted from the frozen Phase 4 node mapping"
        )
    return tuple(snapshots)


def _lineage_for(name: str, *, projected: bool) -> dict[str, Any]:
    lineage: dict[str, Any] = {
        "phase4_source": "exp4_final_heldout_raw_prediction",
        "manual_forest_version": "projected_v0" if projected else "strict_v0",
    }
    dependencies = MULTI_PARENT_DEPENDENCIES.get(name)
    if not dependencies:
        return lineage
    if projected and name in PROJECTED_PRIMARY_PARENT:
        primary = PROJECTED_PRIMARY_PARENT[name]
        lineage["primary_parent_projection"] = primary
        lineage["unrepresented_dependencies"] = [
            dependency for dependency in dependencies if dependency != primary
        ]
    else:
        lineage["unsupported_multi_parent_dependency"] = list(dependencies)
    return lineage


def _build_rubric(
    criteria: Sequence[RubricCriterionSnapshot],
    *,
    projected: bool,
) -> StructuredRubric:
    nodes = {
        NODE_ID_BY_CRITERION[criterion.name]: RubricNode(
            node_id=NODE_ID_BY_CRITERION[criterion.name],
            criterion=criterion,
            lineage=_lineage_for(criterion.name, projected=projected),
        )
        for criterion in criteria
    }
    named_edges = PROJECTED_EDGES if projected else STRICT_EDGES
    edges = tuple(
        RubricEdge(
            parent_id=NODE_ID_BY_CRITERION[parent],
            child_id=NODE_ID_BY_CRITERION[child],
            condition=EdgeCondition.PARENT_BOTH_PASS,
        )
        for parent, child in named_edges
    )
    root_names = PROJECTED_ROOT_NAMES if projected else STRICT_ROOT_NAMES
    return StructuredRubric(
        nodes=nodes,
        edges=edges,
        root_ids=tuple(NODE_ID_BY_CRITERION[name] for name in root_names),
    )


def build_static_rubrics(
    criteria_source: str | Path,
) -> tuple[StructuredRubric, StructuredRubric]:
    criteria = load_frozen_criteria(criteria_source)
    strict = _build_rubric(criteria, projected=False)
    projected = _build_rubric(criteria, projected=True)
    if (len(strict.nodes), len(strict.root_ids), len(strict.edges)) != (17, 15, 2):
        raise AssertionError("strict_v0 topology drifted")
    if (len(projected.nodes), len(projected.root_ids), len(projected.edges)) != (
        17,
        9,
        8,
    ):
        raise AssertionError("projected_v0 topology drifted")
    strict_criteria = {
        node.criterion.name: node.criterion for node in strict.nodes.values()
    }
    projected_criteria = {
        node.criterion.name: node.criterion for node in projected.nodes.values()
    }
    if strict_criteria != projected_criteria:
        raise AssertionError("strict/projected criteria snapshots differ")
    return strict, projected


def validate_heldout_dataset_source(path: str | Path) -> None:
    _require_exact_source(Path(path), HELDOUT_DATASET_SHA256, "heldout dataset")
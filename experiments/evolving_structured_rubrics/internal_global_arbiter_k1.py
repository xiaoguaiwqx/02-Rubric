"""K=1 single-prompt Global-Arbiter evaluation on internal datasets."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from critiq.structured import PairwisePredictionOutput, execute_offline_m1
from critiq.structured.backend_pool import BackendEndpointSpec, BackendPoolSpec
from critiq.utils import parse_json

from . import _global_arbiter_ab_only_support as support
from . import discovery_v2_prompt_v2_evolution as phase17
from . import global_arbiter_ab_only as arbiter
from . import run_rubric_evolution as base
from .experiment_utils import atomic_write_json, canonical_sha256, load_json
from .rubric_factory import file_sha256


SCHEMA_VERSION = "1.0.0"
PROTOCOL_VERSION = "internal-global-arbiter-k1-v1"
SUBTREE_PROMPT_VERSION = "implicit-unified-subtree-direct-judge-v1"
ARBITER_PROMPT_VERSION = arbiter.PROMPT_VERSION
EXPERIMENT_DIR = "internal_global_arbiter_k1_v1"
SOURCE_EXPERIMENT = phase17.EXPERIMENT_DIR
SOURCE_EPOCH = 4
ENDPOINT_IDS = ("vllm-8000", "vllm-8001")
SPLIT_COUNTS = {"discovery100": 100, "dev150": 150, "heldout500": 500}

STAGES = (
    "internal-global-arbiter-freeze",
    "internal-global-arbiter-audit",
    "internal-global-arbiter-smoke",
    "internal-global-arbiter-run",
    "internal-global-arbiter-retry",
    "internal-global-arbiter-report",
)

UNIFIED_SUBTREE_SYSTEM_PROMPT = """## Instruction

You are judging a multimodal image-text preference pair under one structured rubric subtree. You are given the image, the source instruction or question, two candidate responses, and the rubric subtree.

The root defines the broad criterion, while its children provide specialized guidance for particular cases. Treat the entire subtree as one unified decision policy, not as a set of independent votes.

Use the image when the rubric depends on visual evidence. If the rubric subtree as a whole is not applicable to this pair, answer None.

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A based on the given criterion.",
    "analysis_b": "Analyze B based on the given criterion.",
    "thought": "Compare A and B.",
    "answer": "A / B / None"
}
```

Return None if any of the following conditions are met:
- The rubric subtree as a whole is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
"""


def _target(output: Path) -> Path:
    return output.parent / EXPERIMENT_DIR


def _source_target(output: Path) -> Path:
    return output / SOURCE_EXPERIMENT


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_epoch": SOURCE_EPOCH,
        "rubric_sha256": "007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d",
        "datasets": dict(SPLIT_COUNTS),
        "k": 1,
        "replicate_count": 1,
        "ab_swap": False,
        "generation_seed_policy": "unset",
        "temperature": 0.5,
        "max_tokens": 2048,
        "max_parse_retries": 10,
        "smoke_samples_per_split": 2,
        "endpoint_ids": list(ENDPOINT_IDS),
        "scheduler": "sample_major_available_slot_affinity",
        "semantic_answer_space": ["A", "B", "None"],
        "selection_after_dev_or_heldout_forbidden": True,
    }
    value = config.get("internal_global_arbiter_k1_experiment")
    if value != expected:
        raise RuntimeError(
            "internal_global_arbiter_k1_experiment does not match the frozen protocol")
    return dict(value)


def _rubric(output: Path):
    path = (_source_target(output) / "epochs" / f"epoch_{SOURCE_EPOCH:02d}"
            / "rubric_committed.json")
    rubric = support.StructuredRubric.load_json(path)
    if len(rubric.root_ids) != 5 or len(rubric.nodes) != 27:
        raise RuntimeError("internal Global Arbiter requires Phase17 E4 5-root rubric")
    return rubric


def _rows(config: Mapping[str, Any], split: str) -> tuple[dict[str, Any], ...]:
    aliases = {
        "discovery100": "discovery",
        "dev150": "dev",
        "heldout500": "heldout",
    }
    rows = tuple(dict(item) for item in phase17._rows(config, aliases[split]))
    if len(rows) != SPLIT_COUNTS[split]:
        raise RuntimeError(f"{split} count drift")
    return rows


def _dataset_path(config: Mapping[str, Any], split: str) -> Path:
    if split == "discovery100":
        return base._path(phase17.DISCOVERY_PATH)
    if split == "dev150":
        return base._path(phase17.DEV_PATH)
    return base._path(config["heldout_dataset"])


def parse_subtree_response(raw: object) -> dict[str, str]:
    if not isinstance(raw, str):
        raise ValueError("Unified-Subtree response must be text")
    payload = parse_json(raw, allow_invalid_escapes=True)
    if not isinstance(payload, dict):
        raise ValueError("Unified-Subtree response must be one JSON object")
    answer = payload.get("answer")
    if answer not in {"A", "B", "None"}:
        raise ValueError("Unified-Subtree answer must be A, B, or None")
    return {
        "analysis_a": payload.get("analysis_a")
        if isinstance(payload.get("analysis_a"), str) else "",
        "analysis_b": payload.get("analysis_b")
        if isinstance(payload.get("analysis_b"), str) else "",
        "thought": payload.get("thought")
        if isinstance(payload.get("thought"), str) else "",
        "answer": str(answer),
    }


def subtree_user_prompt(row: Mapping[str, Any], rubric: Any, root_id: str) -> str:
    root = rubric.get_node(root_id)
    lines = [
        "## Question", str(row["question"]), "",
        "## Candidate A", str(row["A"]), "",
        "## Candidate B", str(row["B"]), "",
        "## Structured Rubric Subtree", "",
        "### Root Criterion",
        f"**{root.criterion.name}**: {root.criterion.description}", "",
        "### Specialized Child Criteria",
    ]
    children = rubric.children(root_id)
    if children:
        for index, child in enumerate(children, start=1):
            lines.extend([
                f"{index}. **{child.criterion.name}**",
                child.criterion.description,
            ])
    else:
        lines.append("None.")
    lines.extend([
        "", "Which candidate better follows this structured rubric subtree and is "
        "more likely to align with human preference?",
    ])
    return "\n".join(lines)


def _manifest(config: Mapping[str, Any], output: Path, include_endpoints: bool) -> dict[str, Any]:
    settings = _settings(config)
    rubric = _rubric(output)
    source = _source_target(output)
    datasets = {
        split: {
            "count": SPLIT_COUNTS[split],
            "path": str(_dataset_path(config, split).resolve()),
            "sha256": file_sha256(_dataset_path(config, split)),
            "semantic_sha256": canonical_sha256([{
                key: row.get(key) for key in ("sample_id", "question", "A", "B", "answer")
            } for row in _rows(config, split)]),
        }
        for split in SPLIT_COUNTS
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "settings": settings,
        "settings_sha256": canonical_sha256(settings),
        "rubric_sha256": rubric.rubric_sha256,
        "root_ids": list(rubric.root_ids),
        "datasets": datasets,
        "subtree_system_prompt_sha256": hashlib.sha256(
            UNIFIED_SUBTREE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "arbiter_system_prompt_sha256": hashlib.sha256(
            arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "source_artifacts": {
            "rubric": file_sha256(
                source / "epochs" / f"epoch_{SOURCE_EPOCH:02d}" / "rubric_committed.json"),
            "discovery_prediction": file_sha256(
                source / "epochs" / f"epoch_{SOURCE_EPOCH:02d}" / "discovery_pairwise.json"),
            "dev_prediction": file_sha256(
                source / "epochs" / f"epoch_{SOURCE_EPOCH:02d}" / "dev150" / "pairwise.json"),
            "heldout_prediction": file_sha256(source / "heldout500" / "combined_pairwise.json"),
        },
        "request_count": sum(SPLIT_COUNTS.values()) * 6,
        "k": 1,
        "ab_swap": False,
    }
    if include_endpoints:
        value["endpoint_identities"] = base._inspect_endpoints(
            config, BackendPoolSpec.from_dict(config["backend_pool"]))
    return value


def _status(target: Path, stage: str, details: Mapping[str, Any]) -> None:
    support.status(target, stage, details)


def _require(target: Path, stage: str) -> None:
    support.require(target, stage)


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target(output)
    manifest = _manifest(config, output, True)
    path = target / "frozen_manifest.json"
    if path.is_file() and load_json(path) != manifest:
        raise RuntimeError("internal Global-Arbiter frozen manifest drift")
    atomic_write_json(path, manifest)
    atomic_write_json(target / "prompt_spec.json", {
        "schema_version": SCHEMA_VERSION,
        "subtree_prompt_version": SUBTREE_PROMPT_VERSION,
        "subtree_system_prompt": UNIFIED_SUBTREE_SYSTEM_PROMPT,
        "arbiter_prompt_version": ARBITER_PROMPT_VERSION,
        "arbiter_system_prompt": arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        "subtree_parser": "A_or_B_or_None_with_optional_reasoning_fields",
        "arbiter_parser": "A_or_B_or_None",
    })
    details = {
        "datasets": dict(SPLIT_COUNTS),
        "rubric_sha256": manifest["rubric_sha256"],
        "logical_request_count": manifest["request_count"],
    }
    _status(target, STAGES[0], details)
    print(json.dumps(details, indent=2))


def _load_frozen(config: Mapping[str, Any], output: Path):
    target = _target(output)
    _require(target, STAGES[0])
    manifest = load_json(target / "frozen_manifest.json")
    expected = _manifest(config, output, False)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"internal Global-Arbiter manifest drift: {key}")
    return target, manifest, _rubric(output)


def _verify_live(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    actual = base._inspect_endpoints(
        config, BackendPoolSpec.from_dict(config["backend_pool"]))
    if actual != manifest.get("endpoint_identities"):
        raise RuntimeError("internal Global-Arbiter endpoint identity drift")


def audit(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    row = _rows(config, "discovery100")[0]
    root_id = rubric.root_ids[0]
    subtree_prompt = subtree_user_prompt(row, rubric, root_id)
    reports = [{
        "criterion_name": rubric.get_node(item).criterion.name,
        "report": {"analysis_a": "a", "analysis_b": "b", "thought": "t", "answer": "None"},
    } for item in rubric.root_ids]
    arbiter_prompt = support.global_arbiter_user_prompt(row, reports)
    checks = {
        "k_is_one": manifest["k"] == 1,
        "single_replicate": manifest["settings"]["replicate_count"] == 1,
        "no_swap": manifest["ab_swap"] is False,
        "five_roots": len(rubric.root_ids) == 5,
        "parser_accepts_subtree_none": parse_subtree_response(
            '{"answer":"None"}')["answer"] == "None",
        "parser_accepts_arbiter_none": arbiter.parse_global_arbiter_ab_only_response(
            '{"answer":"None"}')["answer"] == "None",
        "subtree_prompt_has_original_pair": str(row["A"]) in subtree_prompt
            and str(row["B"]) in subtree_prompt,
        "arbiter_prompt_has_five_reports": sum(
            f"### {item['criterion_name']}" in arbiter_prompt for item in reports) == 5,
        "gold_absent_from_prompts": "gold" not in (
            subtree_prompt + arbiter_prompt).lower(),
        "all_datasets_frozen": set(manifest["datasets"]) == set(SPLIT_COUNTS),
    }
    if not all(checks.values()):
        raise RuntimeError(f"internal Global-Arbiter audit failed: {checks}")
    atomic_write_json(target / "offline_audit.json", {
        "schema_version": SCHEMA_VERSION, "offline_only": True, "checks": checks})
    _status(target, STAGES[1], checks)
    print(json.dumps(checks, indent=2))


def _call_subtree(
    config: Mapping[str, Any], target: Path, endpoint: BackendEndpointSpec,
    split: str, row: Mapping[str, Any], rubric: Any, root_id: str,
    total_attempt_limit: int,
) -> dict[str, Any]:
    return support.call_one(
        config, endpoint, target / "cache" / split / "subtree",
        user_text=subtree_user_prompt(row, rubric, root_id), row=row,
        request_key={
            "kind": "internal_unified_subtree", "split": split,
            "sample_id": str(row["sample_id"]), "root_id": root_id,
            "replicate": 0, "order": 0,
        },
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=SUBTREE_PROMPT_VERSION,
        system_prompt=UNIFIED_SUBTREE_SYSTEM_PROMPT,
        response_parser=parse_subtree_response,
        settings_loader=_settings,
    )


def _call_arbiter(
    config: Mapping[str, Any], target: Path, endpoint: BackendEndpointSpec,
    split: str, row: Mapping[str, Any], reports: Sequence[Mapping[str, Any]],
    total_attempt_limit: int,
) -> dict[str, Any]:
    bundle_sha = canonical_sha256(reports)
    return support.call_one(
        config, endpoint, target / "cache" / split / "arbiter",
        user_text=support.global_arbiter_user_prompt(row, reports), row=row,
        request_key={
            "kind": "internal_global_arbiter", "split": split,
            "sample_id": str(row["sample_id"]), "replicate": 0,
            "order": 0, "source_report_bundle_sha256": bundle_sha,
        },
        total_attempt_limit=total_attempt_limit,
        protocol_version=PROTOCOL_VERSION,
        prompt_version=ARBITER_PROMPT_VERSION,
        system_prompt=arbiter.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT,
        response_parser=arbiter.parse_global_arbiter_ab_only_response,
        settings_loader=_settings,
    )


def _bundle(
    config: Mapping[str, Any], target: Path, endpoint: BackendEndpointSpec,
    split: str, row: Mapping[str, Any], rubric: Any, total_attempt_limit: int,
) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []
    for root_id in rubric.root_ids:
        call = _call_subtree(
            config, target, endpoint, split, row, rubric, root_id,
            total_attempt_limit)
        compact = support.compact_call(call)
        roots[root_id] = compact
        if compact.get("parse_ok"):
            reports.append({
                "root_id": root_id,
                "criterion_name": rubric.get_node(root_id).criterion.name,
                "report": compact["parsed"],
            })
    result = {
        "sample_id": str(row["sample_id"]),
        "endpoint_id": endpoint.endpoint_id,
        "order": 0,
        "subtrees": roots,
        "report_bundle_sha256": canonical_sha256(reports),
    }
    if len(reports) == len(rubric.root_ids):
        result["arbiter"] = support.compact_call(_call_arbiter(
            config, target, endpoint, split, row, reports,
            total_attempt_limit))
    else:
        result["arbiter"] = {
            "parse_ok": False,
            "parsed": None,
            "model_generation_count": 0,
            "endpoint_id": endpoint.endpoint_id,
            "error": "blocked_by_unresolved_subtree",
        }
    atomic_write_json(
        target / "bundles" / split / f"{canonical_sha256(row['sample_id'])}.json",
        result)
    return result


def _run_split(
    config: Mapping[str, Any], target: Path, split: str,
    rows: Sequence[Mapping[str, Any]], rubric: Any, total_attempt_limit: int,
) -> dict[str, Any]:
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    endpoints = tuple(item for item in pool.endpoints if item.endpoint_id in ENDPOINT_IDS)
    if tuple(item.endpoint_id for item in endpoints) != ENDPOINT_IDS:
        raise RuntimeError("internal Global Arbiter requires both frozen endpoints")
    assignment_path = target / "endpoint_assignment.json"
    assignments = load_json(assignment_path) if assignment_path.is_file() else {}
    pending: queue.Queue[Mapping[str, Any]] = queue.Queue()
    assigned = {endpoint.endpoint_id: queue.Queue() for endpoint in endpoints}
    for row in rows:
        key = f"{split}::{row['sample_id']}"
        endpoint_id = assignments.get(key)
        (assigned[endpoint_id] if endpoint_id in assigned else pending).put(row)
    lock = threading.Lock()
    completed = 0
    values: dict[str, Any] = {}
    started = time.perf_counter()
    print(f"internal_arbiter_{split}: 0/{len(rows)} sample bundles started", flush=True)

    def worker(endpoint: BackendEndpointSpec) -> None:
        nonlocal completed
        while True:
            try:
                row = assigned[endpoint.endpoint_id].get_nowait()
            except queue.Empty:
                try:
                    row = pending.get_nowait()
                except queue.Empty:
                    return
                with lock:
                    assignments[f"{split}::{row['sample_id']}"] = endpoint.endpoint_id
                    atomic_write_json(assignment_path, assignments)
            value = _bundle(
                config, target, endpoint, split, row, rubric,
                total_attempt_limit)
            with lock:
                values[str(row["sample_id"])] = value
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(rows) - completed)
                print(
                    f"internal_arbiter_{split}: {completed}/{len(rows)} "
                    f"sample={row['sample_id']} endpoint={endpoint.endpoint_id} "
                    f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
                    flush=True)

    workers = [endpoint for endpoint in endpoints
               for _ in range(endpoint.max_concurrency)]
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        list(executor.map(worker, workers))
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "k": 1,
        "ab_swap": False,
        "samples": [values[str(row["sample_id"])] for row in rows],
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(target / "predictions" / f"{split}.json", result)
    return result


def _failures(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sample in result["samples"]:
        for root_id, call in sample["subtrees"].items():
            if not call.get("parse_ok"):
                rows.append({
                    "sample_id": sample["sample_id"], "kind": "subtree",
                    "root_id": root_id, "error": call.get("error"),
                })
        call = sample["arbiter"]
        if not call.get("parse_ok"):
            rows.append({
                "sample_id": sample["sample_id"], "kind": "arbiter",
                "root_id": None, "error": call.get("error"),
            })
    return rows


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[1])
    _verify_live(config, manifest)
    count = _settings(config)["smoke_samples_per_split"]
    values = {}
    for split in SPLIT_COUNTS:
        result = _run_split(
            # Use the formal cache so smoke requests are not generated a second
            # time by the full run. The full prediction file is overwritten
            # with the complete split while the request identities stay exact.
            config, target, split, _rows(config, split)[:count],
            rubric, 1 + _settings(config)["max_parse_retries"])
        values[split] = {
            "sample_count": count,
            "failure_count": len(_failures(result)),
        }
    if any(item["failure_count"] for item in values.values()):
        raise RuntimeError(f"internal Global-Arbiter smoke failed: {values}")
    _status(target, STAGES[2], values)
    print(json.dumps(values, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[2])
    _verify_live(config, manifest)
    details = {}
    failures = {}
    for split in SPLIT_COUNTS:
        result = _run_split(config, target, split, _rows(config, split), rubric, 1)
        current = _failures(result)
        failures[split] = current
        details[split] = {
            "sample_count": len(result["samples"]),
            "technical_failure_count": len(current),
            "wall_seconds": result["wall_seconds"],
        }
    atomic_write_json(target / "parse_failures.json", failures)
    _status(target, STAGES[3], details)
    print(json.dumps(details, indent=2))


def retry(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, rubric = _load_frozen(config, output)
    _require(target, STAGES[3])
    _verify_live(config, manifest)
    limit = 1 + _settings(config)["max_parse_retries"]
    details = {}
    failures = {}
    for split in SPLIT_COUNTS:
        result = _run_split(
            config, target, split, _rows(config, split), rubric, limit)
        current = _failures(result)
        failures[split] = current
        details[split] = {
            "technical_failure_count": len(current),
            "parse_valid_rate": 1.0 - len(current) / (len(result["samples"]) * 6),
        }
    atomic_write_json(target / "parse_failures.json", failures)
    atomic_write_json(target / "retry_summary.json", {
        "schema_version": SCHEMA_VERSION, "splits": details})
    _status(target, STAGES[4], details)
    print(json.dumps(details, indent=2))


def _normalize(value: str | None) -> str | None:
    return value if value in {"A", "B"} else None


def _equal_root(sample: Mapping[str, Any]) -> str | None:
    answers = [
        (call.get("parsed") or {}).get("answer")
        for call in sample["subtrees"].values()
        if call.get("parse_ok")
    ]
    a_count, b_count = answers.count("A"), answers.count("B")
    return "A" if a_count > b_count else "B" if b_count > a_count else None


def _control_predictions(
    output: Path, split: str, rubric: Any,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str | None], dict[str, Any]]:
    source = _source_target(output)
    epoch = source / "epochs" / f"epoch_{SOURCE_EPOCH:02d}"
    paths = {
        "discovery100": epoch / "discovery_pairwise.json",
        "dev150": epoch / "dev150" / "pairwise.json",
        "heldout500": source / "heldout500" / "combined_pairwise.json",
    }
    prediction = PairwisePredictionOutput.load_json(paths[split])
    control_rubric = rubric
    checkpoint = f"epoch_{SOURCE_EPOCH:02d}"
    exact_same_rubric_as_treatment = True
    if split == "heldout500":
        # Phase17 generated heldout only for the final E5 checkpoint. Six node
        # descriptions changed from E4 to E5, so executing that artifact with
        # E4 would violate OfflinePairwiseVoteBackend identity checks. Reuse it
        # only as a clearly labelled E5 reference; never present it as E4.
        control_rubric = support.StructuredRubric.load_json(
            source / "epochs" / "epoch_05" / "rubric_committed.json")
        checkpoint = "epoch_05_reference"
        exact_same_rubric_as_treatment = False
    _, answers = execute_offline_m1(control_rubric, prediction, rows)
    return [_normalize(item.value) for item in answers], {
        "checkpoint": checkpoint,
        "rubric_sha256": control_rubric.rubric_sha256,
        "prediction_path": str(paths[split].resolve()),
        "exact_same_rubric_as_treatment": exact_same_rubric_as_treatment,
    }


def _metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[str | None],
             technical_failures: Sequence[bool] | None = None) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("metric rows/predictions length mismatch")
    if technical_failures is None:
        technical_failures = [False] * len(rows)
    if len(rows) != len(technical_failures):
        raise ValueError("metric rows/technical failures length mismatch")
    correct = [prediction == row["answer"] for prediction, row in zip(predictions, rows)]
    covered = [prediction in {"A", "B"} for prediction in predictions]
    correct_count = sum(correct)
    coverage_count = sum(covered)

    def grouped(field: str) -> dict[str, Any]:
        names = sorted({str(row[field]) for row in rows if row.get(field) is not None})
        result = {}
        for name in names:
            indices = [index for index, row in enumerate(rows) if str(row.get(field)) == name]
            group_correct = sum(correct[index] for index in indices)
            group_covered = sum(covered[index] for index in indices)
            result[name] = {
                "sample_count": len(indices),
                "strict_accuracy": group_correct / len(indices),
                "coverage": group_covered / len(indices),
                "overall_acc": group_correct / group_covered if group_covered else 0.0,
            }
        return result

    domains = grouped("domain")
    sources = grouped("source")
    return {
        "sample_count": len(rows),
        "correct_count": correct_count,
        "strict_accuracy": correct_count / len(rows),
        "accuracy_ci95_wilson": base._wilson_interval(correct_count, len(rows)),
        "coverage_count": coverage_count,
        "coverage": coverage_count / len(rows),
        "overall_acc": correct_count / coverage_count if coverage_count else 0.0,
        "none_count": len(rows) - coverage_count,
        "semantic_none_count": sum(
            prediction is None and not failed
            for prediction, failed in zip(predictions, technical_failures)),
        "technical_failure_count": sum(technical_failures),
        "macro_acc": (sum(item["strict_accuracy"] for item in domains.values())
                      / len(domains) if domains else None),
        "domains": domains,
        "sources": sources,
        "prediction_distribution": dict(Counter(
            prediction or "None" for prediction in predictions)),
        "predictions": list(predictions),
    }


def _paired(rows: Sequence[Mapping[str, Any]], before: Sequence[str | None],
            after: Sequence[str | None]) -> dict[str, Any]:
    corrected, harmed = [], []
    for row, left, right in zip(rows, before, after):
        gold = row["answer"]
        if right == gold and left != gold:
            corrected.append(str(row["sample_id"]))
        elif left == gold and right != gold:
            harmed.append(str(row["sample_id"]))
    return {
        "corrected_count": len(corrected),
        "harmed_count": len(harmed),
        "net_corrected": len(corrected) - len(harmed),
        "corrected_sample_ids": corrected,
        "harmed_sample_ids": harmed,
    }


def report(config: Mapping[str, Any], output: Path) -> None:
    target, _, rubric = _load_frozen(config, output)
    _require(target, STAGES[4])
    failures = load_json(target / "parse_failures.json")
    reports = {}
    for split in SPLIT_COUNTS:
        rows = _rows(config, split)
        result = load_json(target / "predictions" / f"{split}.json")
        explicit, explicit_provenance = _control_predictions(
            output, split, rubric, rows)
        explicit_name = (
            "phase17_e4_explicit_recursive"
            if explicit_provenance["exact_same_rubric_as_treatment"]
            else "phase17_e5_explicit_recursive_reference")
        equal_root = [_equal_root(sample) for sample in result["samples"]]
        final = [_normalize((sample["arbiter"].get("parsed") or {}).get("answer"))
                 for sample in result["samples"]]
        technical = [
            (not sample["arbiter"].get("parse_ok"))
            or any(not call.get("parse_ok") for call in sample["subtrees"].values())
            for sample in result["samples"]
        ]
        systems = {
            explicit_name: _metrics(rows, explicit),
            "implicit_subtree_equal_root": _metrics(rows, equal_root, technical),
            "single_prompt_global_arbiter": _metrics(rows, final, technical),
        }
        value = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "systems": systems,
            "explicit_control_provenance": explicit_provenance,
            "paired": {
                "arbiter_vs_explicit_reference": _paired(rows, explicit, final),
                "arbiter_vs_equal_root": _paired(rows, equal_root, final),
            },
            "root_response_distribution": {
                root_id: dict(Counter(
                    ((sample["subtrees"][root_id].get("parsed") or {}).get("answer")
                     or "technical_failure") for sample in result["samples"]))
                for root_id in rubric.root_ids
            },
            "unresolved_technical_failures": failures.get(split, []),
            "selection_from_this_split_forbidden": split != "discovery100",
        }
        atomic_write_json(target / "reports" / f"{split}.json", value)
        reports[split] = value
    final = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "k": 1,
        "replicate_count": 1,
        "ab_swap": False,
        "splits": reports,
        "unresolved_technical_failure_count": sum(
            len(failures.get(split, [])) for split in SPLIT_COUNTS),
        "selection_after_dev_or_heldout_forbidden": True,
    }
    atomic_write_json(target / "final_report.json", final)
    lines = [
        "# Internal K=1 Single-Prompt Global Arbiter", "",
        "One original-order inference per sample; no A/B swap and no replicate ensemble.", "",
        "| Split | System | Strict ACC | OverallAcc | MacroAcc | Coverage | Semantic None | Technical failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in reports.items():
        for name, metrics in value["systems"].items():
            macro = "-" if metrics["macro_acc"] is None else f"{metrics['macro_acc']:.2%}"
            lines.append(
                f"| {split} | {name} | {metrics['strict_accuracy']:.2%} | "
                f"{metrics['overall_acc']:.2%} | {macro} | "
                f"{metrics['coverage']:.2%} | {metrics['semantic_none_count']} | "
                f"{metrics['technical_failure_count']} |")
    (target / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        split: {
            name: {
                key: metrics[key] for key in (
                    "strict_accuracy", "overall_acc", "macro_acc", "coverage", "none_count",
                    "semantic_none_count", "technical_failure_count")
            } for name, metrics in value["systems"].items()
        } for split, value in reports.items()
    }
    _status(target, STAGES[5], summary)
    print(json.dumps(summary, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze,
        STAGES[1]: audit,
        STAGES[2]: smoke,
        STAGES[3]: run,
        STAGES[4]: retry,
        STAGES[5]: report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported internal Global-Arbiter stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

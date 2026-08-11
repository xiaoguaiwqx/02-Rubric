"""Counterbalanced K=3 external transfer evaluation on VL-RewardBench.

The runner intentionally keeps the benchmark's native prompt baseline separate
from CritiQ's frozen structured Pairwise Worker. It never calls a Manager and
never sends a benchmark preference label to either model prompt.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from critiq.structured import (
    AvailableSlotBackendPool,
    BackendPoolSpec,
    FinalPreference,
    ModelCallMetrics,
    PairwisePredictionOutput,
    StructuredRubric,
    combine_model_call_metrics,
)

from . import run_rubric_evolution as base
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, make_progress_callback
from .rubric_factory import build_multicrit_open_ended_init_rubric, file_sha256


EXPERIMENT_DIR = "vl_rewardbench_external_transfer_v1"
ENDPOINT_ID = "vllm-8000"
K = 3
SEED = 42
EXPECTED_COUNT = 1247
SOURCE_EXPERIMENT = "phase8_refine_role_aware_v2"
NATIVE_DECODING = {"temperature": 0.2, "top_p": 0.2, "max_tokens": 2048}
NATIVE_PATTERN = re.compile(
    r"(?:Overall Judgment|Therefore)\s*.*\s*-*\s*Answer\s*(\d+)\s*"
    r"is\s*(?:the\s*)?(?:slightly\s*)?better",
    re.IGNORECASE,
)


def _target() -> Path:
    return base.ROOT / "output/evolving_structured_rubrics" / EXPERIMENT_DIR


def _parquet_path() -> Path:
    return base.ROOT / "data/VL_RewardBench/data/test-00000-of-00001.parquet"


def _prompt_path() -> Path:
    return base.ROOT / "data/VL_RewardBench/prompt.py"


def _source_epoch_one_path(output: Path) -> Path:
    return (output / SOURCE_EXPERIMENT / "epochs/epoch_01/rubric_committed.json").resolve()


def _status_path(target: Path) -> Path:
    return target / "stage_status.json"


def _status(target: Path, stage: str, details: Mapping[str, Any] | None = None) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    value[stage] = {"status": "passed", "details": dict(details or {})}
    atomic_write_json(_status_path(target), value)


def _require(target: Path, stage: str) -> None:
    value = load_json(_status_path(target)) if _status_path(target).exists() else {}
    if value.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _image_suffix(value: bytes) -> str:
    if value.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if value.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if value.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if value.startswith(b"RIFF") and value[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("VL-RewardBench image bytes have an unsupported format")


def _materialize_image(target: Path, image_bytes: bytes) -> tuple[str, Path]:
    digest = hashlib.sha256(image_bytes).hexdigest()
    path = target / "dataset_images" / f"{digest}{_image_suffix(image_bytes)}"
    if path.exists():
        if file_sha256(path) != digest:
            raise RuntimeError(f"materialized image hash mismatch: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_bytes(image_bytes)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return digest, path


def _read_records(target: Path) -> tuple[dict[str, Any], ...]:
    """Load parquet records and losslessly materialize their image bytes."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("pandas with parquet support is required for VL-RewardBench") from exc
    path = _parquet_path()
    if not path.is_file():
        raise RuntimeError(f"VL-RewardBench parquet is missing: {path}")
    frame = pd.read_parquet(path)
    required = {"id", "query", "response", "image", "human_ranking", "query_source"}
    if set(frame.columns) < required or len(frame) != EXPECTED_COUNT:
        raise RuntimeError("VL-RewardBench parquet schema or row count changed")
    raw_items = frame.to_dict(orient="records")
    raw_id_counts = Counter(str(item["id"]) for item in raw_items)
    records: list[dict[str, Any]] = []
    for row_index, item in enumerate(raw_items):
        benchmark_id = str(item["id"])
        # Nine released rows share an original ID.  They remain distinct
        # preference pairs, so only the runner-internal ID receives a stable
        # row suffix.  The untouched benchmark ID is retained for reporting.
        sample_id = (benchmark_id if raw_id_counts[benchmark_id] == 1
                     else f"{benchmark_id}__row_{row_index:04d}")
        responses = tuple(str(value) for value in item["response"])
        ranking = tuple(int(value) for value in item["human_ranking"])
        image = item["image"]
        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if (not benchmark_id or len(responses) != 2
                or set(ranking) != {0, 1} or not isinstance(image_bytes, bytes)):
            raise RuntimeError(f"invalid VL-RewardBench record: {benchmark_id!r}")
        digest, image_path = _materialize_image(target, image_bytes)
        records.append({
            "sample_id": sample_id,
            "benchmark_id": benchmark_id,
            "question": str(item["query"]),
            "responses": list(responses),
            "preferred_original_index": ranking.index(0),
            "image_sha256": digest,
            "image_path": str(image_path),
            "query_source": str(item["query_source"]),
            "group": _official_group(benchmark_id),
        })
    return tuple(records)


def _official_dataset(sample_id: str) -> str:
    split = min(
        (index for index in (sample_id.find("_"), sample_id.find("-")) if index >= 0),
        default=len(sample_id),
    )
    prefix = sample_id[:split]
    if prefix == "RLAIF":
        return "rlaif-v"
    if prefix == "RLHF":
        return "rlhf-v"
    if prefix in {"mathverse", "mmmu"}:
        return "reasoning_tasks"
    if prefix == "wildvision":
        return "wildvision-battle"
    if prefix == "hallucination":
        return "povid"
    return "vlfeedback"


def _official_group(sample_id: str) -> str:
    mapping = {
        "vlfeedback": "general", "povid": "hallucination",
        "reasoning_tasks": "reasoning", "rlhf-v": "hallucination",
        "rlaif-v": "hallucination", "wildvision-battle": "general",
    }
    return mapping[_official_dataset(sample_id)]


def _order_schedule(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int, int]]:
    """Return an exactly balanced A/B/A vs B/A/B schedule."""

    ordered_ids = sorted(
        (str(row["sample_id"]) for row in records),
        key=lambda sample_id: (hashlib.sha256(
            f"vlrb-k3-v1|{SEED}|{sample_id}".encode("utf-8")).hexdigest(), sample_id),
    )
    return {sample_id: (index % 2, 1 - (index % 2), index % 2)
            for index, sample_id in enumerate(ordered_ids)}


def _ordered_rows(records: Sequence[Mapping[str, Any]], schedule: Mapping[str, Sequence[int]],
                  replicate: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for record in records:
        order = int(schedule[str(record["sample_id"])][replicate])
        preferred = int(record["preferred_original_index"])
        rows.append({
            "sample_id": str(record["sample_id"]),
            "image_path": str(record["image_path"]),
            "question": str(record["question"]),
            "A": str(record["responses"][order]),
            "B": str(record["responses"][1 - order]),
            "answer": "A" if preferred == order else "B",
        })
    return tuple(rows)


@lru_cache(maxsize=1)
def _load_native_prompt() -> Callable[[Mapping[str, Any], int], str]:
    path = _prompt_path()
    spec = importlib.util.spec_from_file_location("vl_rewardbench_native_prompt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load VL-RewardBench prompt module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "prompt", None)
    if not callable(function):
        raise RuntimeError("VL-RewardBench prompt module has no prompt function")
    return function


def _native_content(record: Mapping[str, Any], order: int) -> list[dict[str, Any]]:
    prompt = _load_native_prompt()({
        "query": record["question"], "response": record["responses"]}, order)
    prompt = str(prompt).replace("<image>\n", "")
    image_bytes = Path(str(record["image_path"])).read_bytes()
    suffix = Path(str(record["image_path"])).suffix.lower()
    mime = {".jpg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp"}[suffix]
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"}},
    ]


def _parse_native(raw: object) -> int | None:
    if not isinstance(raw, str):
        return None
    match = NATIVE_PATTERN.search(raw.replace("\n", "").replace("*", ""))
    if match is None:
        return None
    value = int(match.group(1))
    return value if value in {1, 2} else None


def _native_spec(config: Mapping[str, Any], prompt_sha256: str) -> dict[str, Any]:
    return {
        "model": config["model"], "endpoint_id": ENDPOINT_ID,
        "prompt_sha256": prompt_sha256, "decoding": NATIVE_DECODING,
        "parser": "vl_rewardbench_overall_judgment_regex_v1",
        "k": K, "seed": SEED,
    }


def _inspect_vlrb_endpoint(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Capture the live external-evaluation server identity.

    The existing Phase-5 configuration intentionally pins the historical
    RLHF-V deployment's checkpoint path and context length.  VL-RewardBench is
    a new experiment, so it freezes the current 8000 deployment instead of
    treating those historical deployment fields as a precondition.  Model name
    and vLLM version remain validated, while root and context length are
    recorded and must remain unchanged after freeze.
    """

    pool = BackendPoolSpec.from_dict(base._single_endpoint_execution_pool(config, ENDPOINT_ID))
    if len(pool.endpoints) != 1 or pool.endpoints[0].endpoint_id != ENDPOINT_ID:
        raise RuntimeError("VL-RewardBench must use only vllm-8000")
    endpoint = pool.endpoints[0]
    base_url = endpoint.base_url.rstrip("/")
    version = base.request_json(base_url[:-3] + "/version")
    models = base.request_json(base_url + "/models")
    matches = [item for item in models.get("data", []) if item.get("id") == config["model"]]
    if len(matches) != 1:
        raise RuntimeError("vllm-8000 does not expose the configured model exactly once")
    actual = {
        "endpoint_id": ENDPOINT_ID,
        "vllm_version": version.get("version"),
        "model": matches[0].get("id"),
        "checkpoint_root": matches[0].get("root"),
        "max_model_len": matches[0].get("max_model_len"),
    }
    if actual["vllm_version"] != config["vllm_version"] or actual["model"] != config["model"]:
        raise RuntimeError(f"VL-RewardBench server model/version mismatch: {actual}")
    return [actual]


def _manifest(config: Mapping[str, Any], output: Path, target: Path,
              records: Sequence[Mapping[str, Any]], schedule: Mapping[str, Sequence[int]],
              *, include_endpoint: bool = True) -> dict[str, Any]:
    initial = build_multicrit_open_ended_init_rubric()
    epoch_one_path = _source_epoch_one_path(output)
    if not epoch_one_path.is_file():
        raise RuntimeError(f"Role-aware Epoch-1 rubric is missing: {epoch_one_path}")
    epoch_one = StructuredRubric.load_json(epoch_one_path)
    if len(initial.nodes) != 5 or len(epoch_one.nodes) != 17:
        raise RuntimeError("frozen VL-RewardBench rubric node count changed")
    endpoint_identities = _inspect_vlrb_endpoint(config) if include_endpoint else None
    rows = _ordered_rows(records, schedule, 0)
    structured_spec = base._expected_pairwise_request_spec(config, rows).to_dict()
    prompt_sha = file_sha256(_prompt_path())
    manifest = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_DIR,
        "exploratory": True,
        "overlap_audit": "skipped_by_protocol",
        "dataset": {
            "path": str(_parquet_path()), "sha256": file_sha256(_parquet_path()),
            "count": len(records), "record_sha256": canonical_sha256(list(records)),
        },
        "order_schedule_sha256": canonical_sha256(schedule),
        "counterbalance": {"k": K, "seed": SEED, "protocol": "balanced_b_1minusb_b"},
        "native": _native_spec(config, prompt_sha),
        "structured_worker_request_spec": structured_spec,
        "systems": {
            "initial_five_root_m1": {
                "rubric_path": "deterministic:build_multicrit_open_ended_init_rubric",
                "rubric_sha256": initial.rubric_sha256, "node_count": len(initial.nodes),
            },
            "role_aware_refine_epoch_01": {
                "rubric_path": str(epoch_one_path), "file_sha256": file_sha256(epoch_one_path),
                "rubric_sha256": epoch_one.rubric_sha256, "node_count": len(epoch_one.nodes),
            },
        },
        "selection_after_benchmark_forbidden": True,
    }
    if include_endpoint:
        manifest["endpoint"] = {"endpoint_id": ENDPOINT_ID, "identities": endpoint_identities}
    return manifest


def _load_frozen(config: Mapping[str, Any], output: Path) -> tuple[Path, dict[str, Any], tuple[dict[str, Any], ...], dict[str, tuple[int, int, int]]]:
    target = _target()
    _require(target, "vlrb-freeze")
    records = _read_records(target)
    schedule = _order_schedule(records)
    manifest = _manifest(config, output, target, records, schedule, include_endpoint=False)
    stored = load_json(target / "frozen_manifest.json")
    stored_without_endpoint = {key: value for key, value in stored.items() if key != "endpoint"}
    if stored_without_endpoint != manifest:
        raise RuntimeError("VL-RewardBench frozen manifest drift")
    return target, manifest, records, schedule


def _verify_live_endpoint(config: Mapping[str, Any], stored_manifest: Mapping[str, Any]) -> None:
    """Require the live 8000 identity captured by freeze before inference."""

    live = _inspect_vlrb_endpoint(config)
    expected = stored_manifest.get("endpoint")
    if expected != {"endpoint_id": ENDPOINT_ID, "identities": live}:
        raise RuntimeError("VL-RewardBench port-8000 endpoint identity drift")


def freeze(config: Mapping[str, Any], output: Path) -> None:
    target = _target()
    records = _read_records(target)
    schedule = _order_schedule(records)
    manifest = _manifest(config, output, target, records, schedule)
    manifest_path = target / "frozen_manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        status = load_json(_status_path(target)) if _status_path(target).exists() else {}
        downstream = ("vlrb-smoke", "vlrb-run", "vlrb-report")
        if any(status.get(stage, {}).get("status") == "passed" for stage in downstream):
            raise RuntimeError("VL-RewardBench frozen manifest drift after inference")
        # A pre-inference freeze is recoverable: no model output exists whose
        # provenance could be invalidated.  This also repairs the manifest
        # emitted by the short-lived pre-release runner.
        atomic_write_json(target / "freeze_repair.json", {
            "reason": "pre_inference_manifest_rebuild",
            "replaced_manifest_sha256": canonical_sha256(load_json(manifest_path)),
            "replacement_manifest_sha256": canonical_sha256(manifest),
        })
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(target / "dataset_manifest.json", manifest["dataset"])
    atomic_write_json(target / "order_schedule.json", schedule)
    _status(target, "vlrb-freeze", {"dataset_count": len(records), "k": K,
                                     "target": str(target)})
    print(json.dumps({"dataset_count": len(records), "k": K, "target": str(target)},
                     indent=2, ensure_ascii=False))


def _load_rubric(label: str, output: Path) -> StructuredRubric:
    if label == "initial_five_root_m1":
        return build_multicrit_open_ended_init_rubric()
    if label == "role_aware_refine_epoch_01":
        return StructuredRubric.load_json(_source_epoch_one_path(output))
    raise ValueError(f"unknown structured system {label}")


def _validate_prediction(prediction: PairwisePredictionOutput, rubric: StructuredRubric,
                         rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    if tuple(prediction.sample_ids) != tuple(row["sample_id"] for row in rows):
        raise RuntimeError("structured VL-RewardBench sample order drift")
    if tuple(item.name for item in prediction.criteria) != tuple(
            rubric.get_node(node_id).criterion.name for node_id in rubric.preorder_node_ids()):
        raise RuntimeError("structured VL-RewardBench criterion order drift")
    if prediction.request_spec.to_dict() != manifest["structured_worker_request_spec"]:
        raise RuntimeError("structured VL-RewardBench request identity drift")


def _structured_prediction(config: Mapping[str, Any], output: Path, target: Path,
                           manifest: Mapping[str, Any], label: str,
                           rubric: StructuredRubric, rows: Sequence[Mapping[str, Any]],
                           replicate: int) -> PairwisePredictionOutput:
    run_dir = target / "structured" / label / f"replicate_{replicate + 1:02d}"
    artifact = run_dir / "predictions" / f"{label}.json"
    if artifact.exists():
        result = PairwisePredictionOutput.load_json(artifact)
    else:
        result, _, _ = base._generate_pairwise(
            config, run_dir, rubric, rows, label,
            execution_backend_pool=base._single_endpoint_execution_pool(config, ENDPOINT_ID),
            request_backend_id=manifest["structured_worker_request_spec"]["backend_id"],
            request_level_progress=True)
    _validate_prediction(result, rubric, rows, manifest)
    return result


def _native_cache_path(target: Path, sample_id: str, replicate: int) -> Path:
    digest = hashlib.sha256(f"native-v1|{sample_id}|{replicate}".encode("utf-8")).hexdigest()
    return target / "native" / "cache" / f"{digest}.json"


def _native_result(config: Mapping[str, Any], target: Path, manifest: Mapping[str, Any],
                   records: Sequence[Mapping[str, Any]], schedule: Mapping[str, Sequence[int]],
                   replicate: int) -> list[dict[str, Any]]:
    output_path = target / "native" / f"replicate_{replicate + 1:02d}.json"
    if output_path.exists():
        value = load_json(output_path)
        expected_ids = [str(record["sample_id"]) for record in records]
        expected_orders = [int(schedule[sample_id][replicate]) for sample_id in expected_ids]
        if (isinstance(value, list) and len(value) == len(records)
                and [item.get("sample_id") for item in value] == expected_ids
                and [item.get("order") for item in value] == expected_orders
                and all(item.get("request_identity") == canonical_sha256(manifest["native"])
                        for item in value)):
            return value
        raise RuntimeError("native VL-RewardBench output identity drift")
    pool = AvailableSlotBackendPool(BackendPoolSpec.from_dict(
        base._single_endpoint_execution_pool(config, ENDPOINT_ID)))
    callback = make_progress_callback(target, f"native_replicate_{replicate + 1:02d}",
                                      len(records), pool)
    request_identity = canonical_sha256(manifest["native"])

    def one(index: int, record: Mapping[str, Any]) -> tuple[int, dict[str, Any], ModelCallMetrics]:
        sample_id = str(record["sample_id"])
        order = int(schedule[sample_id][replicate])
        cache_path = _native_cache_path(target, sample_id, replicate)
        if cache_path.exists():
            cached = load_json(cache_path)
            if (cached.get("sample_id") == sample_id and cached.get("order") == order
                    and cached.get("request_identity") == request_identity):
                metrics = ModelCallMetrics.from_agent_calls((), cache_hit=True)
                return index, cached, metrics
            raise RuntimeError("native VL-RewardBench cache identity drift")
        raw, metrics = pool.call(
            _native_content(record, order), request_type="vlrb_native",
            request_key=f"{sample_id}::replicate_{replicate + 1:02d}", structured_attempt=1,
            agent_args={"model": config["model"], "api_keys": "EMPTY",
                        "request_kwargs": NATIVE_DECODING,
                        "api_retry_attempts": config["api_retry_attempts"]})
        display_choice = _parse_native(raw)
        normalized_metrics = ModelCallMetrics.from_agent_calls((metrics,))
        result = {
            "sample_id": sample_id, "order": order, "request_identity": request_identity,
            "display_choice": display_choice, "parse_ok": display_choice is not None,
            "raw_response": raw, "metrics": normalized_metrics.to_dict(),
        }
        atomic_write_json(cache_path, result)
        return index, result, metrics

    results: list[dict[str, Any] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=pool.spec.global_request_concurrency) as executor:
        futures = {executor.submit(one, index, record): (index, record)
                   for index, record in enumerate(records)}
        for future in as_completed(futures):
            index, record = futures[future]
            result_index, result, metrics = future.result()
            results[result_index] = result
            callback(index, str(record["sample_id"]), metrics)
    completed = [item for item in results if item is not None]
    atomic_write_json(output_path, completed)
    atomic_write_json(target / "native" / f"provenance_replicate_{replicate + 1:02d}.json",
                      pool.provenance_dict())
    return completed


def _original_index(display_choice: object, order: int) -> int | None:
    if display_choice == 1:
        return order
    if display_choice == 2:
        return 1 - order
    if display_choice == "A":
        return order
    if display_choice == "B":
        return 1 - order
    return None


def _majority(votes: Sequence[int | None]) -> int | None:
    return 0 if votes.count(0) >= 2 else 1 if votes.count(1) >= 2 else None


def _system_metrics(records: Sequence[Mapping[str, Any]], votes_by_replicate: Sequence[Sequence[int | None]]) -> dict[str, Any]:
    if len(votes_by_replicate) != K or any(len(item) != len(records) for item in votes_by_replicate):
        raise ValueError("K=3 vote matrix shape mismatch")
    decisions = [_majority([votes[index] for votes in votes_by_replicate])
                 for index in range(len(records))]
    correct = [vote == int(record["preferred_original_index"])
               for vote, record in zip(decisions, records)]
    covered = [vote is not None for vote in decisions]
    groups: dict[str, dict[str, Any]] = {}
    for group in ("general", "hallucination", "reasoning"):
        indices = [index for index, record in enumerate(records) if record["group"] == group]
        group_correct = sum(correct[index] for index in indices)
        group_covered = sum(covered[index] for index in indices)
        groups[group] = {
            "sample_count": len(indices), "correct_count": group_correct,
            "strict_accuracy": group_correct / len(indices),
            "coverage": group_covered / len(indices),
            "covered_accuracy": (sum(correct[index] and covered[index] for index in indices)
                                 / group_covered if group_covered else 0.0),
        }
    per_order = []
    for replicate, votes in enumerate(votes_by_replicate):
        valid = [vote is not None for vote in votes]
        per_order.append({
            "replicate": replicate + 1,
            "strict_accuracy": sum(vote == int(record["preferred_original_index"])
                                   for vote, record in zip(votes, records)) / len(records),
            "coverage": sum(valid) / len(records),
        })
    disagreement = sum(len(set(vote for vote in (items[index] for items in votes_by_replicate)
                                 if vote is not None)) > 1 for index in range(len(records)))
    correct_count = sum(correct)
    coverage_count = sum(covered)
    return {
        "sample_count": len(records), "correct_count": correct_count,
        "strict_accuracy": correct_count / len(records),
        "accuracy_ci95_wilson": base._wilson_interval(correct_count, len(records)),
        "coverage_count": coverage_count, "coverage": coverage_count / len(records),
        "covered_accuracy": sum(ok and active for ok, active in zip(correct, covered)) / coverage_count if coverage_count else 0.0,
        "macro_strict_accuracy": sum(item["strict_accuracy"] for item in groups.values()) / len(groups),
        "groups": groups, "per_order": per_order,
        "order_disagreement_count": disagreement,
        "majority_tie_or_abstain_count": len(records) - coverage_count,
        "original_index_predictions": decisions,
    }


def _paired(records: Sequence[Mapping[str, Any]], baseline: Sequence[int | None],
            treatment: Sequence[int | None]) -> dict[str, Any]:
    corrected, harmed = [], []
    for record, before, after in zip(records, baseline, treatment):
        target = int(record["preferred_original_index"])
        if after == target and before != target:
            corrected.append(str(record["sample_id"]))
        elif before == target and after != target:
            harmed.append(str(record["sample_id"]))
    total = len(corrected) + len(harmed)
    tail = (sum(math.comb(total, index) for index in range(min(len(corrected), len(harmed)) + 1))
            / (2 ** total) if total else 0.5)
    return {"corrected_count": len(corrected), "harmed_count": len(harmed),
            "net_corrected": len(corrected) - len(harmed),
            "mcnemar_exact_two_sided_p": min(1.0, 2.0 * tail),
            "corrected_sample_ids": corrected, "harmed_sample_ids": harmed}


def _run(config: Mapping[str, Any], output: Path, *, records: Sequence[Mapping[str, Any]],
         schedule: Mapping[str, Sequence[int]], target: Path, manifest: Mapping[str, Any],
         scope: int | None, label: str) -> None:
    selected = tuple(records[:scope] if scope is not None else records)
    systems = ("initial_five_root_m1", "role_aware_refine_epoch_01")
    native_by_rep: list[list[dict[str, Any]]] = []
    structured_by_system: dict[str, list[PairwisePredictionOutput]] = {name: [] for name in systems}
    for replicate in range(K):
        native_by_rep.append(_native_result(config, target / label, manifest, selected, schedule, replicate))
        rows = _ordered_rows(selected, schedule, replicate)
        for system in systems:
            rubric = _load_rubric(system, output)
            structured_by_system[system].append(
                _structured_prediction(config, output, target / label, manifest, system, rubric, rows, replicate))
    result = {"sample_ids": [row["sample_id"] for row in selected], "k": K, "systems": {}}
    native_votes = [
        [_original_index(item["display_choice"], int(item["order"])) for item in replicate]
        for replicate in native_by_rep]
    result["systems"]["native_vlrb_prompt"] = {"votes_by_replicate": native_votes}
    for system, predictions in structured_by_system.items():
        votes = []
        for replicate, prediction in enumerate(predictions):
            rows = _ordered_rows(selected, schedule, replicate)
            _, answers = base.execute_offline_m1(
                _load_rubric(system, output), prediction, rows)
            votes.append([_original_index(answer.value, int(schedule[row["sample_id"]][replicate]))
                          for answer, row in zip(answers, rows)])
        result["systems"][system] = {"votes_by_replicate": votes}
    atomic_write_json(target / label / "combined" / "logical_votes.json", result)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output)
    _verify_live_endpoint(config, load_json(target / "frozen_manifest.json"))
    _run(config, output, records=records, schedule=schedule, target=target, manifest=manifest,
         scope=20, label="smoke")
    _status(target, "vlrb-smoke", {"sample_count": 20, "request_count": 20 * K * 23})
    print(json.dumps({"sample_count": 20, "request_count": 20 * K * 23}, indent=2))


def run(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output)
    _require(target, "vlrb-smoke")
    _verify_live_endpoint(config, load_json(target / "frozen_manifest.json"))
    _run(config, output, records=records, schedule=schedule, target=target, manifest=manifest,
         scope=None, label="run")
    _status(target, "vlrb-run", {"sample_count": len(records), "request_count": len(records) * K * 23})
    print(json.dumps({"sample_count": len(records), "request_count": len(records) * K * 23}, indent=2))


def report(config: Mapping[str, Any], output: Path) -> None:
    target, manifest, records, schedule = _load_frozen(config, output)
    del schedule
    _require(target, "vlrb-run")
    combined = load_json(target / "run" / "combined" / "logical_votes.json")
    if combined.get("sample_ids") != [record["sample_id"] for record in records]:
        raise RuntimeError("VL-RewardBench report sample order drift")
    systems = combined["systems"]
    metrics = {name: _system_metrics(records, value["votes_by_replicate"])
               for name, value in systems.items()}
    initial = metrics["initial_five_root_m1"]["original_index_predictions"]
    evolved = metrics["role_aware_refine_epoch_01"]["original_index_predictions"]
    native = metrics["native_vlrb_prompt"]["original_index_predictions"]
    report_value = {
        "schema_version": "1.0.0", "experiment": EXPERIMENT_DIR, "exploratory": True,
        "frozen_manifest_sha256": file_sha256(target / "frozen_manifest.json"),
        "selection_after_benchmark_forbidden": True,
        "metrics": metrics,
        "paired": {
            "initial_to_evolved": _paired(records, initial, evolved),
            "native_to_evolved": _paired(records, native, evolved),
            "native_to_initial": _paired(records, native, initial),
        },
    }
    atomic_write_json(target / "report.json", report_value)
    lines = ["# VL-RewardBench External Transfer Report", "",
             "Exploratory K=3 counterbalanced evaluation.", "",
             "| System | Strict ACC | Macro ACC | Coverage |", "|---|---:|---:|---:|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value['strict_accuracy']:.4f} | "
                     f"{value['macro_strict_accuracy']:.4f} | {value['coverage']:.4f} |")
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _status(target, "vlrb-report", {"report": str(target / "report.json")})
    print(json.dumps({name: value["strict_accuracy"] for name, value in metrics.items()}, indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions = {"vlrb-freeze": freeze, "vlrb-smoke": smoke, "vlrb-run": run, "vlrb-report": report}
    try:
        actions[stage](config, output)
    except KeyError as exc:
        raise ValueError(f"unsupported VL-RewardBench stage: {stage}") from exc

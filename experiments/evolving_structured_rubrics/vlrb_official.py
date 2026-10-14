"""VL-RewardBench data, independent random swaps, and K=3 metrics."""
from __future__ import annotations
import hashlib
import math
import random
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence
from .model_call_support import file_sha256

K = 3
SEED = 42
EXPECTED_COUNT = 1247
ORDER_PROTOCOL = "independent-random-v1"
LEGACY_ORDER_PROTOCOL = "alternating-v1"


def _parquet_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data/VL_RewardBench/data/test-00000-of-00001.parquet"


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


def _image_directory(target: Path, parquet_path: Path) -> Path:
    """Reuse old run-local images; place new runs beside the VLRB data."""
    legacy = target / "dataset_images"
    if legacy.is_dir():
        return legacy
    dataset_root = (parquet_path.parent.parent if parquet_path.parent.name == "data"
                    else parquet_path.parent)
    return dataset_root / "dataset_images"


def _materialize_image(image_dir: Path, image_bytes: bytes) -> tuple[str, Path]:
    digest = hashlib.sha256(image_bytes).hexdigest()
    path = image_dir / f"{digest}{_image_suffix(image_bytes)}"
    if path.exists():
        if file_sha256(path) != digest:
            raise RuntimeError(f"materialized image hash mismatch: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=f".{digest}.",
                                    suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(image_bytes)
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return digest, path


def _read_records(target: Path, *, parquet_path: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Load parquet records and losslessly materialize their image bytes."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("pandas with parquet support is required for VL-RewardBench") from exc
    path = (_parquet_path() if parquet_path is None else parquet_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"VL-RewardBench parquet is missing: {path}")
    frame = pd.read_parquet(path)
    required = {"id", "query", "response", "image", "human_ranking", "query_source"}
    if set(frame.columns) < required or len(frame) != EXPECTED_COUNT:
        raise RuntimeError("VL-RewardBench parquet schema or row count changed")
    image_dir = _image_directory(target, path)
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
        digest, image_path = _materialize_image(image_dir, image_bytes)
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


def _order_schedule(records: Sequence[Mapping[str, Any]], *,
                    protocol: str = ORDER_PROTOCOL) -> dict[str, tuple[int, int, int]]:
    """Draw each swap independently; retain the historical training/report schedule."""

    if protocol == ORDER_PROTOCOL:
        generator = random.Random(SEED)
        return {sample_id: tuple(generator.choice([0, 1]) for _ in range(K))
                for sample_id in sorted(str(row["sample_id"]) for row in records)}
    if protocol != LEGACY_ORDER_PROTOCOL:
        raise ValueError(f"unknown VLRB order protocol: {protocol}")

    ordered_ids = sorted(
        (str(row["sample_id"]) for row in records),
        key=lambda sample_id: (hashlib.sha256(
            f"vlrb-k3-v1|{SEED}|{sample_id}".encode("utf-8")).hexdigest(), sample_id),
    )
    return {sample_id: (index % 2, 1 - (index % 2), index % 2)
            for index, sample_id in enumerate(ordered_ids)}


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
        "accuracy_ci95_wilson": _wilson_interval(correct_count, len(records)),
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


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid Wilson interval counts")
    z = 1.959963984540054
    p = successes / total
    scale = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / scale
    margin = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / scale
    return [max(0.0, center - margin), min(1.0, center + margin)]


def official_system_metrics(records: Sequence[Mapping[str, Any]], votes_by_replicate):
    value = _system_metrics(records, votes_by_replicate)
    # Match VL-RewardBench ``cal.py::distribution_dataset``: unmatched/other
    # decisions are excluded from OverallAcc, and MacroAcc is the unweighted
    # mean of General, Hallucination, and Reasoning covered accuracies.
    value["overall_acc"] = value["covered_accuracy"]
    value["macro_acc"] = sum(
        item["covered_accuracy"] for item in value["groups"].values()) / 3
    value["official_metric_semantics"] = (
        "VL-RewardBench agree/reject accuracy; tie/abstain excluded")
    decisions = value["original_index_predictions"]
    source_groups = {}
    for source in ("povid", "reasoning_tasks", "rlaif-v", "rlhf-v",
                   "vlfeedback", "wildvision-battle"):
        indices = [index for index, record in enumerate(records)
                   if _official_dataset(str(record["benchmark_id"])) == source]
        correct = sum(decisions[index] == int(records[index]["preferred_original_index"])
                      for index in indices)
        covered = sum(decisions[index] is not None for index in indices)
        source_groups[source] = {
            "sample_count": len(indices),
            "correct_count": correct,
            "strict_accuracy": correct / len(indices) if indices else 0.0,
            "coverage": covered / len(indices) if indices else 0.0,
            "covered_accuracy": correct / covered if covered else 0.0,
        }
    nonempty = [item for item in source_groups.values() if item["sample_count"]]
    value["source_groups"] = source_groups
    value["source_macro_strict_accuracy"] = (
        sum(item["strict_accuracy"] for item in nonempty) / len(nonempty))
    return value


def _votes(value: Mapping[str, Any]) -> list[list[int | None]]:
    result = []
    for predictions in value["metrics"]["predictions_by_replicate"]:
        result.append([0 if item == "A" else 1 if item == "B" else None
                       for item in predictions])
    return result

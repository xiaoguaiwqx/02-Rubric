"""Prepare RLHF-V datasets for multimodal Bradley-Terry reward modeling.

The generated files keep image paths relative to the RLHF-V image root. The
remote training script should join each `image_path` with its own server-side
image root.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEED = 100745534
DEFAULT_HOLDOUT_AB_SEED = 1008
CORE_FIELDS = (
    "sample_id",
    "prompt_id",
    "image_path",
    "question",
    "chosen",
    "rejected",
    "origin_dataset",
    "origin_split",
    "source_idx",
    "source_image_path",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build BT reward-model JSONL splits for RLHF-V."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir,
        help="Directory containing RLHF-V JSONL source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "bt_reward",
        help="Directory for generated BT reward-model files.",
    )
    parser.add_argument(
        "--train90",
        type=str,
        default="discovery_train_90.jsonl",
        help="90-sample discovery training split.",
    )
    parser.add_argument(
        "--reserve",
        type=str,
        default="reserve_pool.jsonl",
        help="Reserve pool split excluding train90 and holdout500.",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default="heldout_validation_500.jsonl",
        help="Holdout500 source split with chosen/rejected fields.",
    )
    parser.add_argument(
        "--existing-holdout-pair",
        type=str,
        default="heldout_validation_500_pair.jsonl",
        help="Optional existing A/B holdout file used to validate ordering.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--holdout-ab-seed",
        type=int,
        default=DEFAULT_HOLDOUT_AB_SEED,
        help="Seed for deterministic A/B ordering in holdout eval.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def normalize_image_path(value: object) -> str:
    path = "" if value is None else str(value).replace("\\", "/")
    marker = "/processed/images/"
    if marker in path:
        return path.split(marker, 1)[1]
    return path


def require_fields(row: dict[str, Any], fields: Iterable[str], source: Path, idx: int) -> None:
    missing = [field for field in fields if not row.get(field)]
    if missing:
        raise ValueError(f"{source}:{idx + 1} missing fields: {', '.join(missing)}")


def to_bt_record(row: dict[str, Any], source_split: str, idx: int) -> dict[str, Any]:
    require_fields(
        row,
        ("sample_id", "image_path", "question", "chosen", "rejected"),
        Path(source_split),
        idx,
    )
    record = {field: row[field] for field in CORE_FIELDS if field in row}
    record["image_path"] = normalize_image_path(
        row.get("source_image_path") or row.get("image_path")
    )
    record["source_split"] = source_split
    return record


def to_bt_records(rows: list[dict[str, Any]], source_split: str) -> list[dict[str, Any]]:
    return [to_bt_record(row, source_split, idx) for idx, row in enumerate(rows)]


def split_train_val(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, math.ceil(len(rows) * val_fraction))
    val_indices = set(indices[:n_val])
    train = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val = [row for idx, row in enumerate(rows) if idx in val_indices]
    return train, val


def make_holdout_eval(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output = []
    for idx, row in enumerate(rows):
        record = to_bt_record(row, "heldout_validation_500", idx)
        if rng.choice([True, False]):
            record.update({"A": record["rejected"], "B": record["chosen"], "answer": "B"})
        else:
            record.update({"A": record["chosen"], "B": record["rejected"], "answer": "A"})
        output.append(record)
    return output


def overlap_count(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    right_ids = {row["sample_id"] for row in right}
    return sum(1 for row in left if row["sample_id"] in right_ids)


def duplicate_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)
    return sorted(duplicates)


def validate_against_existing_pair(
    generated: list[dict[str, Any]], existing_path: Path
) -> dict[str, Any]:
    if not existing_path.exists():
        return {"path": str(existing_path), "exists": False}
    existing = read_jsonl(existing_path)
    mismatches = []
    for idx, (generated_row, existing_row) in enumerate(zip(generated, existing)):
        same = (
            generated_row.get("sample_id") == existing_row.get("sample_id")
            and generated_row.get("A") == existing_row.get("A")
            and generated_row.get("B") == existing_row.get("B")
            and generated_row.get("answer") == existing_row.get("answer")
        )
        if not same:
            mismatches.append(
                {
                    "index": idx,
                    "sample_id": generated_row.get("sample_id"),
                    "existing_sample_id": existing_row.get("sample_id"),
                }
            )
            if len(mismatches) >= 10:
                break
    return {
        "path": str(existing_path),
        "exists": True,
        "count": len(existing),
        "matches": not mismatches and len(existing) == len(generated),
        "mismatch_count_shown": len(mismatches),
        "first_mismatches": mismatches,
    }


def write_split_bundle(
    output_dir: Path,
    prefix: str,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> dict[str, Any]:
    train, val = split_train_val(rows, seed=seed, val_fraction=val_fraction)
    counts = {
        f"{prefix}_all.jsonl": write_jsonl(output_dir / f"{prefix}_all.jsonl", rows),
        f"{prefix}_train.jsonl": write_jsonl(output_dir / f"{prefix}_train.jsonl", train),
        f"{prefix}_val.jsonl": write_jsonl(output_dir / f"{prefix}_val.jsonl", val),
    }
    return {
        "all": len(rows),
        "train": len(train),
        "val": len(val),
        "files": counts,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    train90_path = data_dir / args.train90
    reserve_path = data_dir / args.reserve
    holdout_path = data_dir / args.holdout
    existing_holdout_pair_path = data_dir / args.existing_holdout_pair

    train90 = to_bt_records(read_jsonl(train90_path), "discovery_train_90")
    reserve = to_bt_records(read_jsonl(reserve_path), "reserve_pool")
    holdout_source = read_jsonl(holdout_path)
    holdout_eval = make_holdout_eval(holdout_source, seed=args.holdout_ab_seed)
    full_train = train90 + reserve

    duplicates = {
        "train90": duplicate_ids(train90),
        "reserve": duplicate_ids(reserve),
        "holdout500": duplicate_ids(holdout_eval),
        "full_train": duplicate_ids(full_train),
    }
    if any(duplicates.values()):
        raise ValueError(f"Duplicate sample IDs found: {duplicates}")

    overlaps = {
        "train90_reserve": overlap_count(train90, reserve),
        "train90_holdout500": overlap_count(train90, holdout_eval),
        "reserve_holdout500": overlap_count(reserve, holdout_eval),
    }
    if any(overlaps.values()):
        raise ValueError(f"Unexpected split overlap: {overlaps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        "rm90": write_split_bundle(
            output_dir,
            "rm90",
            train90,
            seed=args.seed,
            val_fraction=args.val_fraction,
        ),
        "reserve": write_split_bundle(
            output_dir,
            "reserve",
            reserve,
            seed=args.seed,
            val_fraction=args.val_fraction,
        ),
        "full": write_split_bundle(
            output_dir,
            "full",
            full_train,
            seed=args.seed,
            val_fraction=args.val_fraction,
        ),
        "holdout500_eval": {
            "all": write_jsonl(output_dir / "holdout500_eval.jsonl", holdout_eval)
        },
    }

    validation = {
        "existing_holdout_pair": validate_against_existing_pair(
            holdout_eval, existing_holdout_pair_path
        )
    }
    manifest = {
        "created_by": Path(__file__).name,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "holdout_ab_seed": args.holdout_ab_seed,
        "sources": {
            "train90": str(train90_path),
            "reserve": str(reserve_path),
            "holdout500": str(holdout_path),
        },
        "schema": {
            "train": [
                "sample_id",
                "image_path",
                "question",
                "chosen",
                "rejected",
                "source_split",
            ],
            "eval": [
                "sample_id",
                "image_path",
                "question",
                "A",
                "B",
                "answer",
                "chosen",
                "rejected",
                "source_split",
            ],
            "image_path_note": "Relative to the remote RLHF-V processed/images root.",
        },
        "counts": split_counts,
        "overlaps": overlaps,
        "validation": validation,
    }
    write_json(output_dir / "manifest.json", manifest)

    print(json.dumps(manifest, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()

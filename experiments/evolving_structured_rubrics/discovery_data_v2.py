"""Build the isolated Discovery-v2 preference dataset.

This runner deliberately keeps data acquisition, active Worker screening,
397B pre-adjudication, human review, and finalization as separate resumable
stages.  In particular, dev candidates are frozen before Worker screening and
are never submitted to that stage.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import mimetypes
import os
import random
import re
import shutil
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from critiq.structured.backend_pool import AvailableSlotBackendPool, BackendPoolSpec
from critiq.structured.telemetry import ModelCallMetrics, combine_model_call_metrics
from . import run_rubric_evolution as base
from .experiment_utils import atomic_write_json, load_json, make_progress_callback
from .rubric_factory import build_multicrit_open_ended_init_rubric, file_sha256


PROTOCOL_VERSION = "discovery-data-v2-v1"
SCHEMA_VERSION = "1.0.0"
DEDUP_POLICY_VERSION = "image-question-v2-benchmark-schema-v2"
SELECTION_POLICY_VERSION = "metadata-source-balanced-v1"
SELECTION_PER_SOURCE = 25
GENERIC_SCREEN_PROMPT_VERSION = "generic-multimodal-preference-v1"
PREADJUDICATION_PROTOCOL_VERSION = "criterion-agnostic-double-order-v2"
PREADJUDICATION_PROMPT_VERSION = "multimodal-preference-preadjudication-v2"
PREADJUDICATION_POOL_COUNTS = {
    "coverage_candidate": 150,
    "hard_candidate": 90,
    "dev": 150,
}
PREADJUDICATION_TASK_TYPES = {
    "visual_perception", "reasoning", "instruction_following", "generation", "other",
}
PREADJUDICATION_DIMENSIONS = {
    "visual_grounding", "factual_correctness", "reasoning_validity",
    "instruction_following", "relevance", "completeness", "clarity",
    "conciseness", "style", "safety",
}
PREADJUDICATION_EVIDENCE_TYPES = {
    "visual", "textual", "reasoning", "mixed", "insufficient",
}
PREADJUDICATION_AMBIGUITY_FLAGS = {
    "image_unreadable", "question_underspecified", "both_good", "both_bad",
    "preference_close", "source_label_suspect", "other",
}
GENERIC_SCREEN_SYSTEM_PROMPT = """## Instruction

You are judging the overall human preference between two candidate responses to a multimodal image-based instruction or question.

Use the image and source instruction. Consider factual correctness, visual grounding, reasoning validity, instruction following, relevance, clarity, and whether unsupported claims are introduced. Choose the response that is better overall. Do not assume that either response is preferred because of its position, length, style, dataset, or source.

Return exactly one JSON object:
```json
{
  "analysis_a": "Briefly assess Candidate A.",
  "analysis_b": "Briefly assess Candidate B.",
  "thought": "Briefly compare the two candidates.",
  "answer": "A / B / None"
}
```

Return None only when the two responses are genuinely indistinguishable in overall quality or the available evidence is insufficient for a reliable preference.
"""
GENERIC_SCREEN_USER_PROMPT = """## Source Instruction or Question
{question}

## Candidate A
{A}

## Candidate B
{B}

Which candidate is better overall and more likely to align with human preference? Return the required JSON object only."""
PREADJUDICATION_SYSTEM_PROMPT = """## Role

You are independently pre-adjudicating the overall human preference between two candidate responses to a multimodal image-based instruction or question. You are providing structured evidence for a later blind human review; you are not defining the final gold label.

Use the image and source instruction. Consider factual correctness, visual grounding, reasoning validity, instruction following, relevance, completeness, clarity, conciseness, style when relevant, safety, and unsupported claims. Do not assume either response is preferred because of position, length, wording, dataset, or source. Use uncertain when the evidence does not support a reliable A/B preference.

Return exactly one JSON object with these keys and no additional keys:
```json
{
  "answer": "A | B | uncertain",
  "confidence": 1,
  "preference_rationale": "Brief overall preference rationale.",
  "visual_evidence": ["Concrete visible evidence used in the decision."],
  "task_type": "visual_perception | reasoning | instruction_following | generation | other",
  "preference_dimensions": ["factual_correctness"],
  "evidence_type": "visual | textual | reasoning | mixed | insufficient",
  "candidate_a_issues": ["Issue in A, if any."],
  "candidate_b_issues": ["Issue in B, if any."],
  "ambiguity_flags": []
}
```

confidence must be an integer from 1 to 4. preference_dimensions may only use: visual_grounding, factual_correctness, reasoning_validity, instruction_following, relevance, completeness, clarity, conciseness, style, safety. ambiguity_flags may only use: image_unreadable, question_underspecified, both_good, both_bad, preference_close, source_label_suspect, other.
"""
PREADJUDICATION_USER_PROMPT = """## Source Instruction or Question
{question}

## Candidate A
{A}

## Candidate B
{B}

Independently determine the overall preference and return the required JSON object only."""
STAGES = (
    "discovery-v2-source-lock",
    "discovery-v2-ingest",
    "discovery-v2-selection-freeze",
    "discovery-v2-generic-screen-smoke",
    "discovery-v2-generic-screen",
    "discovery-v2-adjudication-freeze",
    "discovery-v2-adjudication-smoke",
    "discovery-v2-adjudication-run",
    "discovery-v2-adjudication-report",
    "discovery-v2-review-v2-export",
    "discovery-v2-demo-freeze",
    "discovery-v2-demo-smoke",
    "discovery-v2-demo-adjudicate",
    "discovery-v2-demo-report",
    "discovery-v2-demo-review-export",
    "discovery-v2-demo-finalize",
    "discovery-v2-demo-export",
    "discovery-v2-screen-smoke",
    "discovery-v2-screen",
    "discovery-v2-adjudicate",
    "discovery-v2-review-export",
    "discovery-v2-finalize",
    "discovery-v2-report",
)
DOMAINS = ("visual", "reasoning", "general")
BOUNDARY_TYPES = (
    "cross_root_non_applicable",
    "criterion_conflict",
    "insufficient_visual_evidence",
    "target_equal_other_dimension",
)
DEFAULT_QUOTAS = {
    "evolve": {"visual": 25, "reasoning": 25, "general": 25},
    "boundary": {
        "cross_root_non_applicable": 10,
        "criterion_conflict": 8,
        "insufficient_visual_evidence": 4,
        "target_equal_other_dimension": 3,
    },
    "dev": {
        "visual": {"regular": 40, "boundary": 10},
        "reasoning": {"regular": 40, "boundary": 10},
        "general": {"regular": 40, "boundary": 10},
    },
}
SOURCE_ADAPTERS = {
    "rlhf_v", "mm_rlhf", "vilreward", "mmpr", "vision_arena", "mm_ifdpo",
}
ADAPTER_INPUT_SCHEMA = {
    "rlhf_v": "image + text.{question,chosen,rejected}",
    "mm_rlhf": "image + question/prompt + chosen/rejected",
    "vilreward": "image_path + question + process + numeric value; grouped by image/question",
    "mmpr": "recoverable image + question/prompt + chosen/rejected",
    "vision_arena": "one image + English single-turn conversation_a/b + non-tie winner",
    "mm_ifdpo": "recoverable image + question/prompt + chosen/rejected",
}
REQUIRED_FINAL_FIELDS = {
    "sample_id", "image_path", "question", "A", "B", "answer",
    "split_role", "domain", "subdomain", "boundary_type", "source",
    "source_family", "source_sample_id", "label_origin",
    "preference_confidence", "preference_rationale", "visual_evidence",
    "primary_error_type", "secondary_error_types", "applicable_root_ids",
    "image_sha256", "question_sha256", "unordered_pair_sha256", "human_review",
}
_ARCHIVE_INDEX: dict[str, tuple[dict[str, str], dict[str, str | None]]] = {}
# VisionArena stores image payloads inside remote Parquet rows. Keeping the
# configured 8,000-row generic reservoir for this source would retain hundreds
# of megabytes of decoded image bytes before normalization. The final quotas
# need at most 400 general candidates, so a deterministic 1,200-row reservoir
# gives ample screening headroom without making ingest memory-bound.
REMOTE_IMAGE_RESERVOIR_CAP = 1200
SELECTION_FORBIDDEN_FIELDS = frozenset({
    "worker_screen", "m1_prediction", "m1_wrong", "root_votes",
    "swapped_root_votes", "swap_inconsistency", "root_conflict",
    "hardness", "prediction", "criterion", "node", "routing",
    "adjudication",
})


def _target(output: Path) -> Path:
    return output / "discovery_data_v2"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must be an object")
                result.append(value)
    return result


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _normalized(value: object) -> str:
    return re.sub(r"\W+", " ", _text(value).casefold(), flags=re.UNICODE).strip()


def _question_sha(question: str) -> str:
    return hashlib.sha256(_normalized(question).encode("utf-8")).hexdigest()


def _pair_sha(a: str, b: str) -> str:
    values = sorted((_normalized(a), _normalized(b)))
    return hashlib.sha256("\n---\n".join(values).encode("utf-8")).hexdigest()


def _image_question_sha(image_sha256: str, question_sha256: str) -> str:
    """Fingerprint one visual question group without collapsing all same-text questions."""

    return _sha([image_sha256, question_sha256])


def _token_jaccard(left: str, right: str) -> float:
    a, b = set(_normalized(left).split()), set(_normalized(right).split())
    return len(a & b) / len(a | b) if a or b else 1.0


def _config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("discovery_data_v2")
    if not isinstance(value, dict):
        raise ValueError("discovery_data_v2 config block is required")
    required = {"protocol_version", "seed", "sources", "candidate_limits",
                "quotas", "screening", "generic_screening", "adjudicator",
                "vl_rewardbench_path"}
    if not required.issubset(value):
        raise ValueError(f"discovery_data_v2 fields missing: {sorted(required-set(value))}")
    if value["protocol_version"] != PROTOCOL_VERSION or value["seed"] != 42:
        raise ValueError("Discovery-v2 freezes protocol version and seed=42")
    if value.get("dedup_policy_version") != DEDUP_POLICY_VERSION:
        raise ValueError("Discovery-v2 deduplication policy drift")
    if not isinstance(value["sources"], list) or not value["sources"]:
        raise ValueError("Discovery-v2 sources must be a non-empty list")
    names = []
    for source in value["sources"]:
        fields = {"name", "source_family", "adapter", "repo_id", "split",
                  "revision", "enabled", "domain", "license", "data_files",
                  "image_archives"}
        if not isinstance(source, dict) or set(source) != fields:
            raise ValueError("each Discovery-v2 source must use the frozen source schema")
        if source["adapter"] not in SOURCE_ADAPTERS or source["domain"] not in DOMAINS:
            raise ValueError("invalid Discovery-v2 source adapter/domain")
        if (not isinstance(source["data_files"], list)
                or not isinstance(source["image_archives"], list)):
            raise ValueError("source data_files/image_archives must be lists")
        names.append(source["name"])
    if len(names) != len(set(names)):
        raise ValueError("Discovery-v2 source names must be unique")
    if value["quotas"] != DEFAULT_QUOTAS:
        raise ValueError("Discovery-v2 production quotas drift")
    screening = value["screening"]
    if (screening.get("prompt_mode") != "v2_cache"
            or screening.get("max_tokens") != 2048
            or screening.get("temperature") != .5
            or screening.get("swap_orders") != 2
            or screening.get("initial_root_count") != 5):
        raise ValueError("Discovery-v2 Worker screening protocol drift")
    generic = value["generic_screening"]
    if (not isinstance(generic, dict)
            or set(generic) != {"prompt_version", "temperature", "max_tokens",
                                "seed", "max_attempts", "smoke_sample_count",
                                "hard_candidates_per_source",
                                "exclude_coverage_candidates"}
            or generic["prompt_version"] != GENERIC_SCREEN_PROMPT_VERSION
            or generic["temperature"] != .5
            or generic["max_tokens"] != 2048
            or generic["seed"] != 42
            or generic["max_attempts"] != 5
            or generic["smoke_sample_count"] != 20
            or generic["hard_candidates_per_source"] != 15
            or generic["exclude_coverage_candidates"] is not True):
        raise ValueError("Discovery-v2 Generic screening protocol drift")
    adjudicator = value["adjudicator"]
    required_adjudicator = {
        "protocol_version", "prompt_version", "structured_max_attempts",
        "smoke_samples_per_source_role", "allow_uncertain", "backend_pool",
        "api_key_env", "api_retry_attempts", "request_kwargs",
    }
    if (not isinstance(adjudicator, dict)
            or set(adjudicator) != required_adjudicator
            or adjudicator["protocol_version"] != PREADJUDICATION_PROTOCOL_VERSION
            or adjudicator["prompt_version"] != PREADJUDICATION_PROMPT_VERSION
            or adjudicator["structured_max_attempts"] != 5
            or adjudicator["smoke_samples_per_source_role"] != 1
            or adjudicator["allow_uncertain"] is not True
            or adjudicator["api_retry_attempts"] != 3
            or adjudicator["request_kwargs"] != {
                "temperature": .2, "max_tokens": 4096, "seed": 42,
                "extra_body": {"enable_thinking": True, "thinking_budget": 2048},
            }):
        raise ValueError("Discovery-v2 pre-adjudication protocol drift")
    adjudicator_pool = BackendPoolSpec.from_dict(adjudicator["backend_pool"])
    if adjudicator_pool.common_checkpoint_id != "Qwen/Qwen3.5-397B-A17B":
        raise ValueError("Discovery-v2 pre-adjudicator model drift")
    pool = BackendPoolSpec.from_dict(config["backend_pool"])
    if {item.endpoint_id for item in pool.endpoints} != {"vllm-8000", "vllm-8001"}:
        raise ValueError("Discovery-v2 screening requires vllm-8000 and vllm-8001")
    return dict(value)


def _set_status(target: Path, stage: str, status: str,
                details: Mapping[str, Any] | None = None) -> None:
    path = target / "stage_status.json"
    value = load_json(path) if path.is_file() else {"schema_version": SCHEMA_VERSION}
    value[stage] = {"status": status, "updated_at_unix": time.time(),
                    "details": dict(details or {})}
    atomic_write_json(path, value)


def _invalidate_downstream(target: Path, *, cause: str) -> list[str]:
    """Prevent artifacts built from a previous candidate pool from being reused."""

    path = target / "stage_status.json"
    if not path.is_file():
        return []
    status = load_json(path)
    invalidated = []
    for stage in STAGES[2:]:
        entry = status.get(stage)
        if isinstance(entry, dict) and entry.get("status") == "passed":
            entry["status"] = "stale"
            entry["stale_reason"] = cause
            entry["updated_at_unix"] = time.time()
            invalidated.append(stage)
    if invalidated:
        atomic_write_json(path, status)
    return invalidated


def _require(target: Path, stage: str) -> None:
    path = target / "stage_status.json"
    status = load_json(path) if path.is_file() else {}
    if status.get(stage, {}).get("status") != "passed":
        raise RuntimeError(f"run {stage} first")


def _source_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in (
        "name", "source_family", "adapter", "repo_id", "split", "revision",
        "enabled", "domain", "license", "data_files", "image_archives")}


def resolve_source_revision(source: Mapping[str, Any], *, api: Any | None = None,
                            token: str | None = None) -> dict[str, Any]:
    """Resolve a mutable HF revision once; injectable for offline tests."""

    if not source["enabled"]:
        return {**_source_identity(source), "resolved_revision": None,
                "disabled": True, "siblings": []}
    if source["adapter"] == "vision_arena" and not token:
        raise RuntimeError(
            "VisionArena requires HF_TOKEN and prior acceptance of its data agreement")
    if api is None:
        try:
            from huggingface_hub import HfApi  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install datasets and huggingface_hub for source-lock") from exc
        api = HfApi(token=token)
    if source["adapter"] == "vision_arena" and hasattr(api, "auth_check"):
        try:
            api.auth_check(source["repo_id"], repo_type="dataset", token=token)
        except Exception as exc:
            raise RuntimeError(
                "HF_TOKEN cannot access VisionArena; accept its data agreement first") from exc
    try:
        info = api.dataset_info(source["repo_id"], revision=source["revision"],
                                token=token, files_metadata=True)
    except TypeError:
        # Lightweight offline test doubles and older huggingface_hub releases
        # may not expose files_metadata; immutable revision locking still works.
        info = api.dataset_info(source["repo_id"], revision=source["revision"], token=token)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or len(sha) < 7:
        raise RuntimeError(f"HF did not return an immutable revision for {source['name']}")
    siblings = []
    for item in getattr(info, "siblings", ()) or ():
        name = getattr(item, "rfilename", None)
        if isinstance(name, str):
            lfs = getattr(item, "lfs", None)
            lfs_sha = (lfs.get("sha256") if isinstance(lfs, Mapping)
                       else getattr(lfs, "sha256", None))
            siblings.append({"path": name, "blob_id": getattr(item, "blob_id", None),
                "lfs_sha256": lfs_sha,
                "size": getattr(item, "size", None)})
    card = getattr(info, "card_data", None)
    remote_license = (card.get("license") if isinstance(card, Mapping)
                      else getattr(card, "license", None)) if card is not None else None
    return {**_source_identity(source), "resolved_revision": sha,
            "disabled": False, "remote_license": remote_license,
            "gated": getattr(info, "gated", None),
            "siblings": sorted(siblings, key=lambda item: item["path"])}


def source_lock(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    target.mkdir(parents=True, exist_ok=True)
    lock_path = target / "source_lock.json"
    identities = [_source_identity(item) for item in cfg["sources"]]
    if lock_path.is_file():
        lock = load_json(lock_path)
        if (lock.get("protocol_version") != PROTOCOL_VERSION
                or lock.get("configured_sources") != identities
                or any(item.get("adapter_input_schema") != ADAPTER_INPUT_SCHEMA[item["adapter"]]
                       for item in lock.get("sources", []))):
            raise RuntimeError("Discovery-v2 source configuration drift after lock")
        _set_status(target, "discovery-v2-source-lock", "passed", {"reused": True})
        print(json.dumps({"source_count": len(lock["sources"]), "reused": True}, indent=2))
        return
    token = os.environ.get("HF_TOKEN")
    sources = [resolve_source_revision(item, token=token) for item in cfg["sources"]]
    for item in sources:
        item["adapter_input_schema"] = ADAPTER_INPUT_SCHEMA[item["adapter"]]
    enabled = [item for item in sources if not item["disabled"]]
    for domain in DOMAINS:
        families = {item["source_family"] for item in enabled if item["domain"] == domain}
        if len(families) < 2:
            raise RuntimeError(f"domain {domain} requires at least two enabled source families")
    lock = {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION,
            "configured_sources": identities, "sources": sources,
            "resolved_at_unix": time.time()}
    lock["lock_sha256"] = _sha(lock)
    atomic_write_json(lock_path, lock)
    _set_status(target, "discovery-v2-source-lock", "passed",
                {"enabled_sources": len(enabled), "lock_sha256": lock["lock_sha256"]})
    print(json.dumps({"source_count": len(enabled), "lock_sha256": lock["lock_sha256"]}, indent=2))


def _conversation_text(value: object) -> str:
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, Mapping):
        for key in ("content", "text", "value", "answer", "response"):
            if key in value:
                return _conversation_text(value[key])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [_conversation_text(item) for item in value]
        return _text("\n".join(item for item in parts if item))
    return ""


def normalize_source_row(adapter: str, row: Mapping[str, Any], *,
                         source: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the six supported source schemas to one internal pair."""

    if adapter not in SOURCE_ADAPTERS:
        raise ValueError(f"unsupported source adapter: {adapter}")
    sample_id = _text(row.get("id") or row.get("index") or row.get("sample_id")
                      or row.get("question_id"))
    image: object = next((row[key] for key in ("image", "images", "image_path")
                          if key in row and row[key] is not None), None)
    raw_question = next((row[key] for key in ("question", "prompt", "instruction")
                         if key in row and row[key] is not None), None)
    question = _conversation_text(raw_question)
    a = b = ""
    answer = ""
    subdomain = _text(row.get("category") or row.get("source") or row.get("type") or "unknown")

    if adapter == "rlhf_v":
        payload = row.get("text")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError("RLHF-V text is not valid JSON") from exc
        payload = payload if isinstance(payload, Mapping) else row
        question = _text(payload.get("question") or question)
        a, b = _text(payload.get("chosen")), _text(payload.get("rejected"))
        answer = "A"
    elif adapter in {"mm_rlhf", "mmpr", "mm_ifdpo"}:
        a = _conversation_text(row.get("chosen") or row.get("response_chosen")
                               or row.get("chosen_response") or row.get("pos"))
        b = _conversation_text(row.get("rejected") or row.get("response_rejected")
                               or row.get("rejected_response") or row.get("neg"))
        answer = "A"
        if not question:
            question = _conversation_text(row.get("prompt") or row.get("conversations"))
    elif adapter == "vision_arena":
        language = _text(row.get("language") or row.get("lang") or "English").casefold()
        images = row.get("images")
        if language not in {"english", "en"}:
            raise ValueError("VisionArena row is not English")
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            if len(images) != 1:
                raise ValueError("VisionArena row must contain exactly one image")
            image = images[0]
        ca, cb = row.get("conversation_a"), row.get("conversation_b")
        if (isinstance(ca, Sequence) and not isinstance(ca, (str, bytes)) and len(ca) != 2):
            raise ValueError("VisionArena row must be single-turn")
        question = question or _conversation_text(ca[0] if isinstance(ca, list) and ca else "")
        a = _conversation_text(ca[-1] if isinstance(ca, list) and ca else row.get("response_a"))
        b = _conversation_text(cb[-1] if isinstance(cb, list) and cb else row.get("response_b"))
        winner = _text(row.get("winner")).casefold()
        if winner in {"model_a", "a", "winner_a"}:
            answer = "A"
        elif winner in {"model_b", "b", "winner_b"}:
            answer = "B"
        else:
            raise ValueError("VisionArena tie/unknown winner is excluded")
    elif adapter == "vilreward":
        # ViLReward is grouped later; one row is a scored process, not a pair.
        process = _text(row.get("process"))
        value = row.get("value")
        if not process or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("ViLReward row lacks process/value")
        return {"_vilreward_process": process, "_vilreward_value": float(value),
                "source_sample_id": sample_id, "image": image, "question": question,
                "subdomain": subdomain, "source": source["name"],
                "source_family": source["source_family"], "domain": source["domain"]}
    if not sample_id:
        sample_id = _sha([source["name"], question, a, b])[:20]
    if not question or not a or not b or answer not in {"A", "B"} or image is None:
        raise ValueError(f"{adapter} row lacks image/question/pair/gold")
    return {"source_sample_id": sample_id, "image": image, "question": question,
            "A": a, "B": b, "answer": answer, "subdomain": subdomain,
            "source": source["name"], "source_family": source["source_family"],
            "domain": source["domain"]}


def group_vilreward(rows: Sequence[Mapping[str, Any]], *, min_value_gap: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(_normalized(row["question"]), _text(row["image"]))].append(row)
    result = []
    for values in groups.values():
        values = sorted(values, key=lambda item: (item["_vilreward_value"], item["_vilreward_process"]))
        if len(values) < 2 or values[-1]["_vilreward_value"] - values[0]["_vilreward_value"] < min_value_gap:
            continue
        low, high = values[0], values[-1]
        result.append({key: high[key] for key in (
            "source_sample_id", "image", "question", "subdomain", "source",
            "source_family", "domain")})
        result[-1].update({"A": high["_vilreward_process"], "B": low["_vilreward_process"],
                           "answer": "A", "source_sample_id":
                           f"{high['source_sample_id']}::{low['source_sample_id']}"})
    return result


def _hf_download(source: Mapping[str, Any], filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub for Discovery-v2 ingest") from exc
    return Path(hf_hub_download(repo_id=source["repo_id"], repo_type="dataset",
        filename=filename, revision=source["_resolved_revision"],
        token=os.environ.get("HF_TOKEN")))


def _safe_extract_zip(path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with archive.open(member) as source_handle, target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)
            extracted.append(target)
    return extracted


def _annotation_files(source: Mapping[str, Any]) -> list[Path]:
    """Download pinned annotation files and expand annotation-only archives."""

    result = []
    destination = Path(source["_raw_source_dir"]) / "annotations"
    for filename in source["data_files"]:
        path = _hf_download(source, filename)
        if path.suffix.casefold() == ".zip":
            result.extend(_safe_extract_zip(path, destination / Path(filename).stem))
        else:
            result.append(path)
    return [path for path in result if path.suffix.casefold() in {".json", ".jsonl", ".parquet"}]


def _iter_annotation_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        yield from _read_jsonl(path)
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{index + 1} must be an object")
                yield item
            return
        if isinstance(value, dict):
            yield value
            return
        raise ValueError(f"{path} must contain a JSON object or array")
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pandas is required for Discovery-v2 Parquet ingest") from exc
        for record in pd.read_parquet(path).to_dict(orient="records"):
            yield dict(record)
        return
    raise ValueError(f"unsupported annotation file: {path}")


def _ensure_image_archives(source: Mapping[str, Any], image_reference: str) -> list[Path]:
    configured = list(source.get("image_archives", []))
    reference = image_reference.replace("\\", "/").casefold()
    # For MM-RLHF, select short.zip/long.zip/etc. from the path prefix. For
    # multipart archives (MMPR images.zip_aa...), every part is required.
    multipart = [name for name in configured if re.search(r"\.zip_[a-z]+$", name)]
    if multipart:
        selected = sorted(multipart)
    else:
        matched = [name for name in configured
                   if Path(name).stem.casefold() in reference.split("/")]
        if len(configured) > 1 and not matched:
            raise ValueError(
                f"image archive cannot be routed from reference: {image_reference}")
        selected = matched or configured
    route_key = "|".join(selected)
    cache = source.get("_image_archive_paths_by_route", {})
    if route_key in cache:
        return [Path(path) for path in cache[route_key]]
    downloaded = [_hf_download(source, name) for name in selected]
    paths: list[Path]
    if multipart:
        destination = Path(source["_raw_source_dir"]) / "archives"
        destination.mkdir(parents=True, exist_ok=True)
        combined = destination / "images.multipart.zip"
        if not combined.exists():
            temporary = combined.with_name(f".{combined.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as target_handle:
                for part in downloaded:
                    with part.open("rb") as source_handle:
                        shutil.copyfileobj(source_handle, target_handle, 8 * 1024 * 1024)
            os.replace(temporary, combined)
        paths = [combined]
    else:
        paths = downloaded
    if isinstance(source, dict):
        cache = dict(source.get("_image_archive_paths_by_route", {}))
        cache[route_key] = [str(path) for path in paths]
        source["_image_archive_paths_by_route"] = cache
    return paths


def _archive_image_bytes(path: Path, reference: str) -> bytes | None:
    normalized = reference.replace("\\", "/").lstrip("./")
    if zipfile.is_zipfile(path):
        cache_key = str(path.resolve())
        try:
            with zipfile.ZipFile(path) as archive:
                if cache_key not in _ARCHIVE_INDEX:
                    exact: dict[str, str] = {}
                    basename: dict[str, str | None] = {}
                    for name in archive.namelist():
                        exact[name.lstrip("./")] = name
                        key = Path(name).name
                        basename[key] = name if key not in basename else None
                    _ARCHIVE_INDEX[cache_key] = exact, basename
                exact, basename = _ARCHIVE_INDEX[cache_key]
                member = exact.get(normalized) or basename.get(Path(normalized).name)
                if member is not None:
                    return archive.read(member)
        except (OSError, zipfile.BadZipFile):
            return None
    elif tarfile.is_tarfile(path):
        cache_key = str(path.resolve())
        try:
            with tarfile.open(path) as archive:
                if cache_key not in _ARCHIVE_INDEX:
                    exact = {}
                    basename = {}
                    for member in archive.getmembers():
                        if member.isfile():
                            exact[member.name.lstrip("./")] = member.name
                            key = Path(member.name).name
                            basename[key] = member.name if key not in basename else None
                    _ARCHIVE_INDEX[cache_key] = exact, basename
                exact, basename = _ARCHIVE_INDEX[cache_key]
                member_name = exact.get(normalized) or basename.get(Path(normalized).name)
                if member_name is not None:
                    handle = archive.extractfile(member_name)
                    return None if handle is None else handle.read()
        except (OSError, tarfile.TarError):
            return None
    return None


def _image_bytes(value: object, *, source: Mapping[str, Any]) -> tuple[bytes, str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            raise ValueError("only single-image preference records are supported")
        value = value[0]
    if isinstance(value, Mapping):
        if isinstance(value.get("bytes"), bytes):
            return value["bytes"], "image/png"
        value = value.get("path") or value.get("url")
    if hasattr(value, "save"):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="JPEG")
        return buffer.getvalue(), "image/jpeg"
    if isinstance(value, bytes):
        return value, "image/png"
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            with urllib.request.urlopen(value, timeout=30) as response:
                return response.read(), response.headers.get_content_type()
        path = Path(value)
        if not path.is_absolute() and source.get("local_root"):
            path = Path(source["local_root"]) / path
        if path.is_file():
            return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/png"
        if source.get("image_archives"):
            for archive in _ensure_image_archives(source, value):
                payload = _archive_image_bytes(archive, value)
                if payload is not None:
                    return payload, mimetypes.guess_type(value)[0] or "image/png"
        if (not source.get("image_archives") and source.get("repo_id")
                and source.get("_resolved_revision")):
            try:
                downloaded_path = _hf_download(source, value)
                return (downloaded_path.read_bytes(),
                        mimetypes.guess_type(downloaded_path.name)[0] or "image/png")
            except (OSError, ValueError):
                pass
    raise ValueError("image cannot be recovered")


def _store_image(value: object, store: Path, *, source: Mapping[str, Any]) -> tuple[str, str]:
    payload, mime = _image_bytes(value, source=source)
    if len(payload) < 64:
        raise ValueError("image payload is empty/corrupt")
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install Pillow before Discovery-v2 ingest") from exc
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError("image payload cannot be decoded") from exc
    digest = hashlib.sha256(payload).hexdigest()
    suffix = mimetypes.guess_extension(mime) or ".img"
    path = store / digest[:2] / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return str(path.resolve()), digest


def _valid_pair(row: Mapping[str, Any], limits: Mapping[str, Any]) -> tuple[bool, str | None]:
    a, b = _text(row.get("A")), _text(row.get("B"))
    if not a or not b:
        return False, "empty_answer"
    if len(a) > limits["max_answer_chars"] or len(b) > limits["max_answer_chars"]:
        return False, "truncated_or_long_answer"
    ratio = max(len(a), len(b)) / max(1, min(len(a), len(b)))
    if ratio > limits["max_length_ratio"]:
        return False, "extreme_length_ratio"
    if _token_jaccard(a, b) >= limits["near_identical_jaccard"]:
        return False, "near_identical_answers"
    return True, None


def _load_hf_source(source: Mapping[str, Any], lock: Mapping[str, Any], *,
                    cap: int) -> list[dict[str, Any]]:
    runtime_source = {**source, "_resolved_revision": lock["resolved_revision"],
                      "_raw_source_dir": source.get("_raw_source_dir", ".")}
    if source["data_files"]:
        files = _annotation_files(runtime_source)
        if not files:
            raise RuntimeError(f"no readable annotation files for {source['name']}")
        dataset = (record for path in files for record in _iter_annotation_records(path))
    else:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install datasets for Discovery-v2 ingest") from exc
        dataset = _iter_locked_hf_stream(source, lock, load_dataset=load_dataset)
    remote_embedded_images = not bool(source["data_files"])
    effective_cap = min(cap, REMOTE_IMAGE_RESERVOIR_CAP) \
        if remote_embedded_images else cap
    scan_multiplier = 4 if remote_embedded_images else 40
    if remote_embedded_images:
        print(f"discovery-v2 source={source['name']} "
              f"raw_reservoir_cap={effective_cap} scan_limit="
              f"{effective_cap * scan_multiplier}", flush=True)
    rng = random.Random(42 + int(_sha(source["name"])[:8], 16))
    reservoir: list[dict[str, Any]] = []
    for index, value in enumerate(dataset):
        if index < effective_cap:
            reservoir.append(dict(value))
        else:
            selected = rng.randint(0, index)
            if selected < effective_cap:
                reservoir[selected] = dict(value)
        if index + 1 >= effective_cap * scan_multiplier:
            break
    return reservoir


def _locked_parquet_shards(lock: Mapping[str, Any]) -> list[str]:
    """Return the Parquet files frozen by source-lock in dataset order."""

    paths = []
    for sibling in lock.get("siblings", []):
        path = sibling.get("path") if isinstance(sibling, Mapping) else sibling
        if isinstance(path, str) and path.casefold().endswith(".parquet"):
            paths.append(path)
    return sorted(set(paths))


def _reset_hf_http_client() -> None:
    """Discard a poisoned huggingface_hub client after an SSL transport error."""

    try:
        from huggingface_hub.utils._http import close_session  # type: ignore
        close_session()
    except (ImportError, AttributeError, RuntimeError):
        # Older huggingface_hub releases do not expose close_session. Rebuilding
        # the Dataset iterator is still useful in that case.
        return


def _is_retryable_hf_stream_error(exc: Exception) -> bool:
    """Keep transport and Arrow allocation failures inside shard recovery."""

    return isinstance(exc, (OSError, RuntimeError, ConnectionError, MemoryError)) \
        or type(exc).__name__ == "ArrowMemoryError"


def _iter_locked_hf_stream(source: Mapping[str, Any], lock: Mapping[str, Any], *,
                           load_dataset: Callable[..., Iterable[Mapping[str, Any]]],
                           max_attempts: int = 8) -> Iterable[dict[str, Any]]:
    """Read a pinned HF dataset one Parquet shard at a time with recovery.

    Hugging Face's range reader can close its shared httpx client after an SSL
    EOF and then fail its own retry with ``client has been closed``.  Rebuilding
    the iterator at the shard boundary both resets that client and prevents a
    late transport failure from replaying every earlier shard.  Rows from a
    failed attempt are buffered and are yielded only after the entire shard has
    completed, so retries cannot introduce duplicate records.
    """

    token = os.environ.get("HF_TOKEN")
    shards: list[str | None] = list(_locked_parquet_shards(lock)) or [None]
    for shard_index, shard in enumerate(shards, start=1):
        last_error: Exception | None = None
        resume_row = 0
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"discovery-v2 stream source={source['name']} "
                      f"shard={shard_index}/{len(shards)} attempt={attempt} started",
                      flush=True)
                kwargs: dict[str, Any] = {
                    "split": source["split"],
                    "revision": lock["resolved_revision"],
                    "streaming": True,
                    "token": token,
                }
                if shard is not None:
                    kwargs["data_files"] = [shard]
                dataset = load_dataset(source["repo_id"], **kwargs)
                rows_read = 0
                row_index = -1
                for row_index, value in enumerate(dataset):
                    rows_read = row_index + 1
                    # A failed attempt may already have yielded rows to the
                    # reservoir in _load_hf_source. The HF shard order is
                    # deterministic at a pinned revision, so resume at the
                    # first row not delivered by the previous attempt.
                    if row_index < resume_row:
                        continue
                    resume_row = row_index + 1
                    yield dict(value)
                print(f"discovery-v2 stream source={source['name']} "
                      f"shard={shard_index}/{len(shards)} rows={rows_read} completed",
                      flush=True)
                if attempt > 1:
                    print(f"discovery-v2 recovered source={source['name']} "
                          f"shard={shard_index}/{len(shards)} attempt={attempt}",
                          flush=True)
                last_error = None
                break
            except Exception as exc:
                if not _is_retryable_hf_stream_error(exc):
                    raise
                last_error = exc
                # row_index is only assigned once iteration starts; preserve
                # the greatest delivered position when a transport/allocation
                # error interrupts the Arrow iterator.
                resume_row = max(resume_row, row_index + 1)
                _reset_hf_http_client()
                if attempt == max_attempts:
                    break
                delay = min(30.0, float(2 ** (attempt - 1)))
                print(f"discovery-v2 retry source={source['name']} "
                      f"shard={shard_index}/{len(shards)} attempt={attempt}/"
                      f"{max_attempts} delay={delay:.0f}s error={type(exc).__name__}: "
                      f"{exc}", flush=True)
                time.sleep(delay)
        if last_error is not None:
            raise RuntimeError(
                f"failed to stream pinned source {source['name']} shard "
                f"{shard_index}/{len(shards)} after {max_attempts} attempts: "
                f"{last_error}") from last_error


def _benchmark_exclusion(path: Path) -> dict[str, set[str]]:
    result = {"sample_ids": set(), "image_sha256": set(),
              "image_question_sha256": set(), "question_sha256": set(),
              "unordered_pair_sha256": set()}
    if not path.exists():
        raise RuntimeError(f"VL-RewardBench checkout is missing: {path}")
    files = [path] if path.is_file() else list(path.rglob("*.parquet")) + list(path.rglob("*.jsonl"))
    for file in files:
        if file.suffix == ".jsonl":
            values = _read_jsonl(file)
        else:
            try:
                import pandas as pd  # type: ignore
                import pyarrow.parquet as pq  # type: ignore
            except ImportError as exc:
                raise RuntimeError("pandas/pyarrow are required for VL-RewardBench exclusion") from exc
            allowed = {"sample_id", "id", "question", "query", "prompt", "A", "B",
                       "response", "response_a", "response_b", "chosen", "rejected",
                       "image", "images", "image_path", "image_sha256"}
            columns = [name for name in pq.read_schema(file).names if name in allowed]
            values = pd.read_parquet(file, columns=columns).to_dict(orient="records")
        recognized_rows = 0
        question_rows = 0
        pair_rows = 0
        for row in values:
            if any(row.get(key) is not None for key in
                   ("sample_id", "id", "question", "query", "prompt", "response",
                    "A", "B", "response_a", "response_b", "chosen", "rejected")):
                recognized_rows += 1
            sample_id = _text(row.get("sample_id") or row.get("id"))
            if sample_id:
                result["sample_ids"].add(sample_id)
            question = _text(row.get("question") or row.get("prompt") or row.get("query"))
            if question:
                question_rows += 1
                result["question_sha256"].add(_question_sha(question))
            responses = row.get("response")
            if hasattr(responses, "tolist"):
                responses = responses.tolist()
            if (isinstance(responses, Sequence)
                    and not isinstance(responses, (str, bytes, bytearray))
                    and len(responses) == 2):
                a, b = (_conversation_text(responses[0]), _conversation_text(responses[1]))
            else:
                a = _conversation_text(row.get("A") or row.get("response_a")
                                       or row.get("chosen"))
                b = _conversation_text(row.get("B") or row.get("response_b")
                                       or row.get("rejected"))
            if a and b:
                pair_rows += 1
                result["unordered_pair_sha256"].add(_pair_sha(a, b))
            image_hash = _text(row.get("image_sha256"))
            if image_hash:
                result["image_sha256"].add(image_hash)
            elif row.get("image") is not None:
                try:
                    payload, _ = _image_bytes(row["image"], source={})
                    result["image_sha256"].add(hashlib.sha256(payload).hexdigest())
                except (OSError, ValueError):
                    pass
            if question and image_hash:
                result["image_question_sha256"].add(
                    _image_question_sha(image_hash, _question_sha(question)))
            elif question and row.get("image") is not None:
                try:
                    payload, _ = _image_bytes(row["image"], source={})
                    image_digest = hashlib.sha256(payload).hexdigest()
                    result["image_question_sha256"].add(
                        _image_question_sha(image_digest, _question_sha(question)))
                except (OSError, ValueError):
                    pass
        if recognized_rows and (question_rows == 0 or pair_rows == 0):
            raise RuntimeError(
                f"benchmark fingerprint extraction incomplete for {file}: "
                f"recognized_rows={recognized_rows}, question_rows={question_rows}, "
                f"pair_rows={pair_rows}")
    return result


def _deduplicate(rows: Sequence[dict[str, Any]], exclusion: Mapping[str, set[str]]) -> tuple[list[dict[str, Any]], Counter]:
    benchmark_sample_ids = set(exclusion.get("sample_ids", set()))
    benchmark_images = set(exclusion.get("image_sha256", set()))
    benchmark_image_questions = set(exclusion.get("image_question_sha256", set()))
    benchmark_pairs = set(exclusion.get("unordered_pair_sha256", set()))
    seen_source_ids: set[str] = set()
    seen_image_questions: set[str] = set()
    seen_pairs: set[str] = set()
    accepted, stats = [], Counter()
    near_by_image_question: dict[str, list[str]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["source_family"], item["source"], item["source_sample_id"])):
        source_id = f"{row['source']}\x00{row['source_sample_id']}"
        image_question = _image_question_sha(row["image_sha256"], row["question_sha256"])
        pair_sha = row["unordered_pair_sha256"]
        if row["source_sample_id"] in benchmark_sample_ids:
            stats["excluded_sample_ids"] += 1
            continue
        if row["image_sha256"] in benchmark_images:
            stats["excluded_image_sha256"] += 1
            continue
        if image_question in benchmark_image_questions:
            stats["excluded_image_question_sha256"] += 1
            continue
        if pair_sha in benchmark_pairs:
            stats["excluded_unordered_pair_sha256"] += 1
            continue
        if source_id in seen_source_ids:
            stats["excluded_source_sample_ids"] += 1
            continue
        if image_question in seen_image_questions:
            stats["excluded_image_question_sha256"] += 1
            continue
        if pair_sha in seen_pairs:
            stats["excluded_unordered_pair_sha256"] += 1
            continue
        pair_text = row["A"] + "\n" + row["B"]
        if any(_token_jaccard(pair_text, old) >= .92
               for old in near_by_image_question[image_question]):
            stats["excluded_near_duplicate_pair"] += 1
            continue
        accepted.append(row)
        seen_source_ids.add(source_id)
        seen_image_questions.add(image_question)
        seen_pairs.add(pair_sha)
        near_by_image_question[image_question].append(pair_text)
    return accepted, stats


def _partition_dev(rows: Sequence[dict[str, Any]], cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze dev groups before screening, without using Worker behavior."""

    rng = random.Random(cfg["seed"])
    wanted = int(cfg["candidate_limits"]["dev_candidates_per_domain"])
    screen_wanted = int(cfg["candidate_limits"]["screen_candidates_per_domain"])
    dev, discovery = [], []
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    for domain in DOMAINS:
        values = sorted(by_domain[domain], key=lambda row: _sha([
            cfg["seed"], row["source_family"], row["image_sha256"], row["question_sha256"]]))
        # Prefer families not needed by Discovery, but never select on Worker correctness.
        family_counts = Counter()
        chosen = []
        for row in values:
            if len(chosen) >= wanted:
                break
            if family_counts[row["source_family"]] >= math.ceil(wanted / max(2, len({v['source_family'] for v in values}))):
                continue
            chosen.append(row); family_counts[row["source_family"]] += 1
        if len(chosen) < wanted:
            remaining = [row for row in values if row not in chosen]
            rng.shuffle(remaining); chosen.extend(remaining[:wanted-len(chosen)])
        chosen_ids = {id(row) for row in chosen}
        dev.extend({**row, "candidate_role": "dev"} for row in chosen)
        remaining = [row for row in values if id(row) not in chosen_ids]
        dev_images = {row.get("image_sha256") for row in chosen if row.get("image_sha256")}
        dev_questions = {row.get("question_sha256") for row in chosen if row.get("question_sha256")}
        dev_pairs = {row.get("unordered_pair_sha256") for row in chosen
                     if row.get("unordered_pair_sha256")}
        dev_source_ids = {(row.get("source"), row.get("source_sample_id"))
                          for row in chosen if row.get("source_sample_id")}
        remaining = [row for row in remaining
                     if row.get("image_sha256") not in dev_images
                     and row.get("question_sha256") not in dev_questions
                     and row.get("unordered_pair_sha256") not in dev_pairs
                     and (row.get("source"), row.get("source_sample_id")) not in dev_source_ids]
        # Deterministic source-family round robin gives every enabled family a
        # chance to contribute before any Worker behavior is observed.
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in remaining:
            by_family[row["source_family"]].append(row)
        selected_discovery = []
        while len(selected_discovery) < screen_wanted:
            changed = False
            for family in sorted(by_family):
                if by_family[family] and len(selected_discovery) < screen_wanted:
                    selected_discovery.append(by_family[family].pop(0)); changed = True
            if not changed:
                break
        discovery.extend({**row, "candidate_role": "discovery"}
                         for row in selected_discovery)
    return dev, discovery


def _selection_length_bin(length: int, *, question: bool = False) -> str:
    boundaries = (48, 160) if question else (160, 640)
    if length <= boundaries[0]:
        return "short"
    if length <= boundaries[1]:
        return "medium"
    return "long"


def _selection_ratio_bin(a: str, b: str) -> str:
    shorter, longer = sorted((max(1, len(a)), max(1, len(b))))
    ratio = longer / shorter
    if ratio <= 1.5:
        return "balanced"
    if ratio <= 3.0:
        return "moderate"
    return "asymmetric"


def _selection_stratum(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    question = _text(row.get("question"))
    a, b = _text(row.get("A")), _text(row.get("B"))
    mean_answer_length = (len(a) + len(b)) // 2
    return (
        _text(row.get("subdomain")) or "unknown",
        _selection_length_bin(len(question), question=True),
        _selection_length_bin(mean_answer_length),
        _selection_ratio_bin(a, b),
    )


def _forbidden_selection_paths(value: object, path: str = "row") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in SELECTION_FORBIDDEN_FIELDS:
                found.append(child_path)
            found.extend(_forbidden_selection_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden_selection_paths(child, f"{path}[{index}]"))
    return found


def _metadata_selection_order(rows: Sequence[dict[str, Any]], *, source: str,
                              seed: int) -> list[dict[str, Any]]:
    """Return a deterministic metadata-only round-robin order for one source."""

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source") != source:
            raise ValueError("metadata selector received a row from another source")
        forbidden = _forbidden_selection_paths(row)
        if forbidden:
            raise RuntimeError(
                "metadata selector candidate contains forbidden model-derived fields: "
                + ", ".join(forbidden[:5]))
        groups[_selection_stratum(row)].append(row)

    for stratum, values in groups.items():
        values.sort(key=lambda row: _sha([
            SELECTION_POLICY_VERSION, seed, source, stratum,
            row.get("sample_id"), row.get("image_sha256"),
            row.get("question_sha256"), row.get("unordered_pair_sha256"),
        ]))
    strata = sorted(groups, key=lambda stratum: _sha([
        SELECTION_POLICY_VERSION, seed, source, stratum]))
    ordered = []
    while len(ordered) < len(rows):
        changed = False
        for stratum in strata:
            values = groups[stratum]
            if values:
                ordered.append(values.pop(0))
                changed = True
        if not changed:
            break
    if len(ordered) != len(rows):
        raise RuntimeError("metadata selector did not produce a complete source order")
    return ordered


def _select_metadata_balanced(rows: Sequence[dict[str, Any]], *,
                              source_names: Sequence[str], seed: int,
                              per_source: int = SELECTION_PER_SOURCE) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source"))].append(row)
    unexpected = sorted(set(by_source) - set(source_names))
    if unexpected:
        raise RuntimeError(f"metadata selector found unexpected sources: {unexpected}")
    selected = []
    for source in source_names:
        values = by_source.get(source, [])
        if len(values) < per_source:
            raise RuntimeError(
                f"metadata selector source {source} has {len(values)} rows; "
                f"requires {per_source}")
        selected.extend(_metadata_selection_order(
            values, source=source, seed=seed)[:per_source])
    return selected


def _selection_overlap(left: Sequence[Mapping[str, Any]],
                       right: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fields = ("sample_id", "image_sha256", "question_sha256",
              "unordered_pair_sha256")
    result = {}
    for field in fields:
        left_values = {row.get(field) for row in left if row.get(field)}
        right_values = {row.get(field) for row in right if row.get(field)}
        result[field] = len(left_values & right_values)
    left_source_ids = {(row.get("source"), row.get("source_sample_id"))
                       for row in left if row.get("source_sample_id")}
    right_source_ids = {(row.get("source"), row.get("source_sample_id"))
                        for row in right if row.get("source_sample_id")}
    result["source+source_sample_id"] = len(left_source_ids & right_source_ids)
    return result


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _selection_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strata = Counter("/".join(_selection_stratum(row)) for row in rows)
    return {
        "count": len(rows),
        "sources": dict(sorted(Counter(str(row["source"]) for row in rows).items())),
        "domains": dict(sorted(Counter(str(row["domain"]) for row in rows).items())),
        "subdomains": dict(sorted(Counter(str(row.get("subdomain") or "unknown")
                                           for row in rows).items())),
        "metadata_strata": dict(sorted(strata.items())),
    }


def selection_freeze(config: Mapping[str, Any], output: Path) -> None:
    """Freeze source-balanced Dev150 and Coverage150 without model behavior."""

    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-ingest")
    candidate_dir = target / "candidates"
    dev_path = candidate_dir / "dev_candidates.jsonl"
    discovery_path = candidate_dir / "discovery_candidates.jsonl"
    dev_candidates = _read_jsonl(dev_path)
    discovery_candidates = _read_jsonl(discovery_path)
    source_names = [source["name"] for source in cfg["sources"] if source["enabled"]]
    if len(source_names) != 6:
        raise RuntimeError("metadata selector requires the six frozen sources")

    dev = _select_metadata_balanced(
        dev_candidates, source_names=source_names, seed=cfg["seed"])
    coverage = _select_metadata_balanced(
        discovery_candidates, source_names=source_names, seed=cfg["seed"])
    if len(dev) != 150 or len(coverage) != 150:
        raise RuntimeError("metadata selector output counts must both equal 150")
    overlaps = _selection_overlap(dev, coverage)
    if any(overlaps.values()):
        raise RuntimeError(f"Dev150/Coverage150 isolation failed: {overlaps}")

    selection_dir = target / "selection_v2"
    dev_output = selection_dir / "dev_150.jsonl"
    coverage_output = selection_dir / "coverage_candidates_150.jsonl"
    diagnostic_paths = [
        target / "screen/report.json",
        target / "candidates/discovery_screened_ranked.jsonl",
    ]
    stage_status = load_json(target / "stage_status.json")
    diagnostic = {
        "diagnostic_only": True,
        "used_for_selection": False,
        "stage_status": stage_status.get("discovery-v2-screen", {}).get("status", "missing"),
        "artifacts": [{
            "path": str(path.relative_to(target)).replace("\\", "/"),
            "exists": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else None,
        } for path in diagnostic_paths],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "seed": cfg["seed"],
        "selection_model_calls": 0,
        "worker_inputs_read": False,
        "selection_inputs": [
            "source", "subdomain", "question_length", "mean_answer_length",
            "answer_length_ratio", "sample_and_content_hashes",
        ],
        "forbidden_inputs": sorted(SELECTION_FORBIDDEN_FIELDS),
        "source_order": source_names,
        "per_source": SELECTION_PER_SOURCE,
        "input_artifacts": {
            "dev_candidates.jsonl": {
                "count": len(dev_candidates), "sha256": file_sha256(dev_path)},
            "discovery_candidates.jsonl": {
                "count": len(discovery_candidates), "sha256": file_sha256(discovery_path)},
        },
        "output_artifacts": {
            "dev_150.jsonl": {"count": len(dev), "sha256": _jsonl_sha256(dev)},
            "coverage_candidates_150.jsonl": {
                "count": len(coverage), "sha256": _jsonl_sha256(coverage)},
        },
        "cross_split_overlap": overlaps,
        "five_root_screening": diagnostic,
    }
    report_value = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "dev_150": _selection_distribution(dev),
        "coverage_candidates_150": _selection_distribution(coverage),
        "cross_split_overlap": overlaps,
        "scientific_isolation": {
            "worker_inputs_read": False,
            "selection_model_calls": 0,
            "five_root_screening_diagnostic_only": True,
        },
    }

    manifest_path = selection_dir / "selection_manifest.json"
    report_path = selection_dir / "selection_report.json"
    if manifest_path.is_file():
        if load_json(manifest_path) != manifest:
            raise RuntimeError("Discovery-v2 selection manifest drift")
        for path, expected in (
                (dev_output, manifest["output_artifacts"]["dev_150.jsonl"]),
                (coverage_output, manifest["output_artifacts"]["coverage_candidates_150.jsonl"])):
            if not path.is_file() or file_sha256(path) != expected["sha256"]:
                raise RuntimeError(f"Discovery-v2 frozen selection artifact drift: {path}")
        if not report_path.is_file() or load_json(report_path) != report_value:
            raise RuntimeError("Discovery-v2 selection report drift")
    else:
        _write_jsonl(dev_output, dev)
        _write_jsonl(coverage_output, coverage)
        if file_sha256(dev_output) != manifest["output_artifacts"]["dev_150.jsonl"]["sha256"]:
            raise RuntimeError("Dev150 write hash mismatch")
        if file_sha256(coverage_output) != manifest["output_artifacts"]["coverage_candidates_150.jsonl"]["sha256"]:
            raise RuntimeError("Coverage150 write hash mismatch")
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(report_path, report_value)

    _set_status(target, "discovery-v2-selection-freeze", "passed", {
        "dev_count": len(dev), "coverage_count": len(coverage),
        "per_source": SELECTION_PER_SOURCE, "selection_model_calls": 0,
        "worker_inputs_read": False,
    })
    print(json.dumps({
        "dev_count": len(dev), "coverage_count": len(coverage),
        "per_source": SELECTION_PER_SOURCE,
        "worker_inputs_read": False,
        "cross_split_overlap": overlaps,
    }, indent=2))


def ingest(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-source-lock")
    lock = load_json(target / "source_lock.json")
    source_locks = {item["name"]: item for item in lock["sources"]}
    store = target / "image_store"
    raw_normalized, rejected = [], Counter()
    cap = int(cfg["candidate_limits"]["raw_rows_per_source"])
    for source in cfg["sources"]:
        if not source["enabled"]:
            continue
        source_lock = source_locks[source["name"]]
        runtime_source = {**source, "_resolved_revision": source_lock["resolved_revision"],
                          "_raw_source_dir": str(target / "raw_sources" / source["name"])}
        shard_dir = target / "ingest/source_shards"
        shard_path = shard_dir / f"{source['name']}.jsonl"
        shard_meta_path = shard_dir / f"{source['name']}.meta.json"
        shard_fingerprint = _sha({"source_lock": source_lock,
            "candidate_limits": cfg["candidate_limits"]})
        if shard_path.is_file() and shard_meta_path.is_file():
            shard_meta = load_json(shard_meta_path)
            if shard_meta.get("input_fingerprint") != shard_fingerprint:
                raise RuntimeError(f"ingest source shard drift for {source['name']}")
            raw_normalized.extend(_read_jsonl(shard_path))
            rejected.update(shard_meta.get("rejected", {}))
            continue
        rows = _load_hf_source(runtime_source, source_lock, cap=cap)
        normalized = []
        source_rejected = Counter()
        for row in rows:
            try:
                normalized.append(normalize_source_row(
                    source["adapter"], row, source=runtime_source))
            except ValueError as exc:
                source_rejected[f"schema:{str(exc)}"] += 1
        if source["adapter"] == "vilreward":
            normalized = group_vilreward(normalized, min_value_gap=float(
                cfg["candidate_limits"]["vilreward_min_value_gap"]))
        for item in normalized:
            valid, reason = _valid_pair(item, cfg["candidate_limits"])
            if not valid:
                source_rejected[reason] += 1
                continue
            try:
                image_path, image_sha = _store_image(
                    item.pop("image"), store, source=runtime_source)
            except (OSError, ValueError) as exc:
                source_rejected[f"image:{type(exc).__name__}"] += 1
                continue
            item.update({"image_path": image_path, "image_sha256": image_sha,
                         "question_sha256": _question_sha(item["question"]),
                         "unordered_pair_sha256": _pair_sha(item["A"], item["B"])})
            item["sample_id"] = f"dv2-{item['source']}-{_sha([item['source_sample_id'], image_sha])[:16]}"
            raw_normalized.append(item)
        source_rows = [row for row in raw_normalized if row["source"] == source["name"]]
        _write_jsonl(shard_path, source_rows)
        atomic_write_json(shard_meta_path, {"schema_version": SCHEMA_VERSION,
            "input_fingerprint": shard_fingerprint, "accepted": len(source_rows),
            "rejected": dict(source_rejected)})
        rejected.update(source_rejected)
    benchmark = _benchmark_exclusion(base._path(cfg["vl_rewardbench_path"]))
    unique, dedup_stats = _deduplicate(raw_normalized, benchmark)
    dev, discovery = _partition_dev(unique, cfg)
    expected_dev = int(cfg["candidate_limits"]["dev_candidates_per_domain"])
    expected_screen = int(cfg["candidate_limits"]["screen_candidates_per_domain"])
    for domain in DOMAINS:
        dev_count = sum(row["domain"] == domain for row in dev)
        discovery_count = sum(row["domain"] == domain for row in discovery)
        if dev_count != expected_dev or discovery_count != expected_screen:
            raise RuntimeError(
                f"insufficient post-dedup candidates for {domain}: "
                f"dev={dev_count}/{expected_dev}, discovery={discovery_count}/{expected_screen}")
    _write_jsonl(target / "candidates/dev_candidates.jsonl", dev)
    _write_jsonl(target / "candidates/discovery_candidates.jsonl", discovery)
    invalidated = _invalidate_downstream(
        target, cause=f"candidate pool rebuilt under {DEDUP_POLICY_VERSION}")
    report = {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION,
              "dedup_policy_version": DEDUP_POLICY_VERSION,
              "normalized": len(raw_normalized),
              "deduplicated": len(unique), "dev_candidates": len(dev),
              "discovery_candidates": len(discovery), "rejected": dict(rejected),
              "deduplication": dict(dedup_stats), "dev_frozen_before_screening": True,
              "dev_selection_model_calls": 0,
              "benchmark_fingerprint_counts": {
                  key: len(value) for key, value in benchmark.items()},
              "invalidated_downstream_stages": invalidated,
              "deduplication_policy": {
                  "internal_exact_keys": ["source+source_sample_id",
                                           "image_sha256+question_sha256",
                                           "unordered_pair_sha256"],
                  "internal_near_duplicate_scope": "image_sha256+question_sha256",
                  "benchmark_exact_keys": ["sample_id", "image_sha256",
                                            "image_sha256+question_sha256",
                                            "unordered_pair_sha256"],
                  "benchmark_question_only_exclusion": False,
                  "dev_discovery_group_isolation": ["image_sha256", "question_sha256",
                                                      "source+source_sample_id",
                                                      "unordered_pair_sha256"]},
              "normalized_schema": sorted({key for row in unique for key in row})}
    atomic_write_json(target / "ingest_report.json", report)
    _set_status(target, "discovery-v2-ingest", "passed", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _swap(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value["sample_id"] = f"{row['sample_id']}::swap"
    value["A"], value["B"] = row["B"], row["A"]
    value["answer"] = "B" if row["answer"] == "A" else "A"
    return value


def _invert_screen_vote(value: str) -> str:
    """Map a swapped A/B vote back to original position semantics."""

    if value == "A":
        return "B"
    if value == "B":
        return "A"
    # Current Vote.ABSTAIN serializes as ``abstain``. Keep the legacy None
    # spelling accepted for cached artifacts created by older runners.
    if value in {"abstain", "None", "none"}:
        return value
    raise ValueError(f"unsupported pairwise vote value: {value!r}")


def _screen_rows(config: Mapping[str, Any], target: Path,
                 rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    run_config = dict(config)
    run_config["_pairwise_prompt_mode"] = "v2_cache"
    request = dict(run_config["worker_request_kwargs"])
    request.update({"temperature": .5, "max_tokens": 2048})
    run_config["worker_request_kwargs"] = request
    rubric = build_multicrit_open_ended_init_rubric()
    original, _, valid_original = base._generate_pairwise(
        run_config, target / "screen", rubric, rows, f"{label}_original",
        request_level_progress=True)
    swapped_rows = [_swap(row) for row in rows]
    swapped, _, valid_swapped = base._generate_pairwise(
        run_config, target / "screen", rubric, swapped_rows, f"{label}_swapped",
        request_level_progress=True)
    scored = []
    for index, row in enumerate(rows):
        original_votes = [value.vote.value for value in original.node_outputs[index].values()]
        swapped_votes = [value.vote.value for value in swapped.node_outputs[index].values()]
        inverted = [_invert_screen_vote(value) for value in swapped_votes]
        swap_inconsistency = sum(a != b for a, b in zip(original_votes, inverted)) / len(original_votes)
        counts = Counter(value for value in original_votes if value in {"A", "B"})
        root_conflict = (min(counts["A"], counts["B"]) / max(1, counts["A"] + counts["B"])) * 2
        # PairwisePredictionOutput stores the flat M1 result in
        # ``flat_answers``. ``answers`` was the field name used by an older
        # prediction artifact and is not part of the current interface.
        final = original.flat_answers[index].value
        wrong = final != row["answer"]
        hardness = 3 * int(wrong) + 2 * swap_inconsistency + root_conflict
        scored.append({"sample_id": row["sample_id"], "m1_prediction": final,
                       "m1_wrong": wrong, "root_votes": original_votes,
                       "swapped_root_votes": swapped_votes,
                       "swap_inconsistency": swap_inconsistency,
                       "root_conflict": root_conflict, "hardness": hardness})
    return {"rows": scored, "original_valid_rate": valid_original,
            "swapped_valid_rate": valid_swapped, "selection_model_calls": len(rows) * 10,
            "hardness_definition": "3*I[M1_wrong]+2*root_swap_inconsistency+normalized_root_conflict"}


def screen_smoke(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-ingest")
    candidates = _read_jsonl(target / "candidates/discovery_candidates.jsonl")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[row["source"]].append(row)
    enabled = [item["name"] for item in cfg["sources"] if item["enabled"]]
    missing = [name for name in enabled if not by_source[name]]
    if missing:
        raise RuntimeError(f"screen smoke has no normalized candidate from sources: {missing}")
    rows = [by_source[name].pop(0) for name in enabled]
    remainder = [row for name in sorted(by_source) for row in by_source[name]]
    rows.extend(remainder[:20-len(rows)])
    if len(rows) != 20:
        raise RuntimeError("screen smoke requires at least 20 discovery candidates")
    value = _screen_rows(config, target, rows, "smoke20")
    value.update({"sample_count": 20, "dev_screened": False,
                  "sources_covered": enabled})
    atomic_write_json(target / "screen/smoke_report.json", value)
    _set_status(target, "discovery-v2-screen-smoke", "passed", value)
    print(json.dumps({key: value[key] for key in (
        "sample_count", "original_valid_rate", "swapped_valid_rate")}, indent=2))


def _stratified_screen_selection(rows: Sequence[dict[str, Any]], scored: Sequence[Mapping[str, Any]],
                                 cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    merged = [{**by_id[item["sample_id"]], "worker_screen": dict(item)} for item in scored]
    cap = int(cfg["candidate_limits"]["adjudication_discovery_per_domain"])
    selected = []
    for domain in DOMAINS:
        values = [item for item in merged if item["domain"] == domain]
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in values:
            signature = "wrong" if item["worker_screen"]["m1_wrong"] else (
                "inconsistent" if item["worker_screen"]["swap_inconsistency"] > 0 else "conflict")
            strata[(item["source_family"], signature)].append(item)
        for group in strata.values():
            group.sort(key=lambda item: (-item["worker_screen"]["hardness"], item["sample_id"]))
        while len([item for item in selected if item["domain"] == domain]) < cap:
            changed = False
            for key in sorted(strata):
                if strata[key] and len([item for item in selected if item["domain"] == domain]) < cap:
                    selected.append(strata[key].pop(0)); changed = True
            if not changed:
                break
    return selected


def screen(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-screen-smoke")
    rows = _read_jsonl(target / "candidates/discovery_candidates.jsonl")
    value = _screen_rows(config, target, rows, "discovery_candidates")
    selected = _stratified_screen_selection(rows, value["rows"], cfg)
    _write_jsonl(target / "candidates/discovery_screened_ranked.jsonl", selected)
    report = {key: value[key] for key in value if key != "rows"}
    report.update({"screened_count": len(rows), "selected_for_adjudication": len(selected),
                   "dev_screened": False, "dev_selection_model_calls": 0})
    atomic_write_json(target / "screen/report.json", report)
    _set_status(target, "discovery-v2-screen", "passed", report)
    print(json.dumps(report, indent=2))


def _image_data_url(path: str) -> str:
    payload = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _extract_object(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("model returned no text")
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def _generic_request_spec(config: Mapping[str, Any],
                          cfg: Mapping[str, Any]) -> dict[str, Any]:
    profile = cfg["generic_screening"]
    pool_spec = BackendPoolSpec.from_dict(config["backend_pool"])
    if pool_spec.common_checkpoint_id != config["model"]:
        raise RuntimeError("Generic screening pool/model identity mismatch")
    if {endpoint.endpoint_id for endpoint in pool_spec.endpoints} != {
            "vllm-8000", "vllm-8001"}:
        raise RuntimeError("Generic screening requires vllm-8000 and vllm-8001")
    decoding = {"temperature": profile["temperature"],
                "max_tokens": profile["max_tokens"], "seed": profile["seed"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": GENERIC_SCREEN_PROMPT_VERSION,
        "model": config["model"],
        "backend_id": pool_spec.backend_id,
        "endpoint_ids": [endpoint.endpoint_id for endpoint in pool_spec.endpoints],
        "system_prompt_sha256": hashlib.sha256(
            GENERIC_SCREEN_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(
            GENERIC_SCREEN_USER_PROMPT.encode("utf-8")).hexdigest(),
        "content_order": "image_then_sample_text",
        "decoding_config": decoding,
        "max_attempts": profile["max_attempts"],
        "swap_orders": 2,
    }


def _generic_user_content(row: Mapping[str, Any], *, swapped: bool,
                          retry: bool) -> list[dict[str, Any]]:
    a, b = (row["B"], row["A"]) if swapped else (row["A"], row["B"])
    text = GENERIC_SCREEN_USER_PROMPT.format(
        question=row["question"], A=a, B=b)
    if retry:
        text += ("\n\nFormatting reminder: return exactly one JSON object with "
                 "analysis_a, analysis_b, thought, and answer. answer must be "
                 "A, B, or None.")
    return [
        {"type": "image_url", "image_url": {
            "url": _image_data_url(str(row["image_path"]))}},
        {"type": "text", "text": text},
    ]


def _parse_generic_response(raw: object) -> dict[str, str]:
    value = _extract_object(raw)
    fields = {"analysis_a", "analysis_b", "thought", "answer"}
    if set(value) != fields:
        raise ValueError("Generic judge response fields are invalid")
    if not all(isinstance(value[key], str) for key in fields):
        raise ValueError("Generic judge response values must be strings")
    answer = value["answer"].strip()
    normalized = {"a": "A", "b": "B", "none": "None",
                  "abstain": "None"}.get(answer.casefold())
    if normalized is None:
        raise ValueError("Generic judge answer must be A, B, or None")
    return {"analysis_a": value["analysis_a"], "analysis_b": value["analysis_b"],
            "thought": value["thought"], "answer": normalized}


def _generic_input_fingerprint(row: Mapping[str, Any], *, swapped: bool) -> str:
    return _sha({"sample_id": row["sample_id"],
                 "image_sha256": row["image_sha256"],
                 "question": row["question"], "A": row["A"], "B": row["B"],
                 "swapped": swapped})


def _generic_judge_cached(config: Mapping[str, Any], cfg: Mapping[str, Any],
                          request_spec: Mapping[str, Any],
                          row: Mapping[str, Any], *, swapped: bool,
                          pool: AvailableSlotBackendPool,
                          cache_dir: Path) -> tuple[dict[str, Any], ModelCallMetrics]:
    fingerprint = _generic_input_fingerprint(row, swapped=swapped)
    key = _sha({"request_spec": request_spec, "input_fingerprint": fingerprint})
    path = cache_dir / f"{key}.json"
    if path.is_file():
        cached = load_json(path)
        if (cached.get("request_spec") != request_spec
                or cached.get("input_fingerprint") != fingerprint):
            raise RuntimeError("Generic screening cache identity drift")
        if cached.get("result", {}).get("parse_ok"):
            result = dict(cached["result"])
            result["cache_hit"] = True
            return result, ModelCallMetrics.from_agent_calls((), cache_hit=True)

    profile = cfg["generic_screening"]
    agent_args = {
        "model": config["model"], "api_keys": "EMPTY",
        "system": GENERIC_SCREEN_SYSTEM_PROMPT,
        "request_kwargs": dict(request_spec["decoding_config"]),
        "api_retry_attempts": int(config["api_retry_attempts"]),
    }
    calls, raw_outputs = [], []
    parsed = None
    last_error = "Generic judge did not return parseable JSON"
    order = "swapped" if swapped else "original"
    for attempt in range(1, int(profile["max_attempts"]) + 1):
        raw, metrics = pool.call(
            _generic_user_content(row, swapped=swapped, retry=attempt > 1),
            request_type="discovery_v2_generic_screen",
            request_key=f"{row['sample_id']}::{order}",
            structured_attempt=attempt, agent_args=agent_args)
        calls.append(metrics)
        raw_outputs.append(raw if isinstance(raw, str) else None)
        try:
            parsed = _parse_generic_response(raw)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    metrics = ModelCallMetrics.from_agent_calls(
        calls, logical_evaluations=1, parse_retries=max(0, len(calls) - 1))
    result = {
        "sample_id": str(row["sample_id"]), "order": order,
        "parse_ok": parsed is not None,
        "answer": parsed["answer"] if parsed is not None else None,
        "parsed": parsed, "attempt_count": len(calls),
        "parse_error": None if parsed is not None else last_error,
        "raw_responses": raw_outputs, "cache_hit": False,
        "generation_metrics": metrics.to_dict(),
    }
    atomic_write_json(path, {"schema_version": SCHEMA_VERSION,
        "request_spec": dict(request_spec), "input_fingerprint": fingerprint,
        "result": result})
    return result, metrics


def _run_generic_orders(config: Mapping[str, Any], cfg: Mapping[str, Any],
                        target: Path, rows: Sequence[Mapping[str, Any]], *,
                        label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_spec = _generic_request_spec(config, cfg)
    pool_spec = BackendPoolSpec.from_dict(config["backend_pool"])
    pool = AvailableSlotBackendPool(pool_spec)
    total = len(rows) * 2
    work = target / "selection_v2/generic_screen" / label
    callback = make_progress_callback(work, label, total, pool)
    ordered_results: list[dict[str, Any]] = [
        {"sample_id": row["sample_id"], "source": row["source"],
         "gold": row["answer"], "original": None, "swapped": None}
        for row in rows]
    metrics_by_call: list[ModelCallMetrics] = []
    cache_dir = target / "selection_v2/generic_screen/cache"
    with ThreadPoolExecutor(max_workers=pool_spec.global_request_concurrency) as executor:
        futures = {}
        for index, row in enumerate(rows):
            for swapped in (False, True):
                future = executor.submit(
                    _generic_judge_cached, config, cfg, request_spec, row,
                    swapped=swapped, pool=pool, cache_dir=cache_dir)
                futures[future] = (index, swapped)
        for future in as_completed(futures):
            index, swapped = futures[future]
            result, metrics = future.result()
            key = "swapped" if swapped else "original"
            ordered_results[index][key] = result
            metrics_by_call.append(metrics)
            callback(index, f"{rows[index]['sample_id']}::{key}", metrics)
    if any(item["original"] is None or item["swapped"] is None
           for item in ordered_results):
        raise RuntimeError("Generic screening lost an order result")
    combined = combine_model_call_metrics(metrics_by_call)
    summary = {
        "request_spec": request_spec,
        "logical_order_count": total,
        "parse_valid_count": sum(
            int(item[order]["parse_ok"])
            for item in ordered_results for order in ("original", "swapped")),
        "parse_failure_sample_orders": [
            f"{item['sample_id']}::{order}"
            for item in ordered_results for order in ("original", "swapped")
            if not item[order]["parse_ok"]],
        "current_run_metrics": combined.to_dict(),
        "endpoint_call_counts": dict(pool.records_by_endpoint()),
    }
    summary["parse_valid_rate"] = summary["parse_valid_count"] / total
    atomic_write_json(work / "provenance.json", pool.provenance_dict())
    _write_jsonl(work / "judgments.jsonl", ordered_results)
    atomic_write_json(work / "summary.json", summary)
    return ordered_results, summary


def _generic_smoke_rows(rows: Sequence[dict[str, Any]], *,
                        source_names: Sequence[str], count: int,
                        seed: int) -> list[dict[str, Any]]:
    by_source = {source: _metadata_selection_order(
        [row for row in rows if row["source"] == source],
        source=source, seed=seed) for source in source_names}
    selected = []
    while len(selected) < count:
        changed = False
        for source in source_names:
            if by_source[source] and len(selected) < count:
                selected.append(by_source[source].pop(0))
                changed = True
        if not changed:
            break
    if len(selected) != count:
        raise RuntimeError("Generic screening smoke has insufficient candidates")
    return selected


def generic_screen_smoke(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-selection-freeze")
    rows = _read_jsonl(target / "candidates/discovery_candidates.jsonl")
    source_names = [source["name"] for source in cfg["sources"] if source["enabled"]]
    selected = _generic_smoke_rows(
        rows, source_names=source_names,
        count=int(cfg["generic_screening"]["smoke_sample_count"]),
        seed=int(cfg["seed"]))
    judgments, summary = _run_generic_orders(
        config, cfg, target, selected, label="generic_screen_smoke")
    failures = summary["parse_failure_sample_orders"]
    report_value = {**summary, "sample_count": len(selected),
                    "sources": dict(Counter(row["source"] for row in selected)),
                    "selection_model_calls": len(selected) * 2,
                    "criterion_or_rubric_in_prompt": False}
    atomic_write_json(target / "selection_v2/generic_screen/smoke_report.json",
                      report_value)
    if failures:
        raise RuntimeError(
            f"Generic screening smoke has {len(failures)} unresolved parse failures")
    _set_status(target, "discovery-v2-generic-screen-smoke", "passed", {
        "sample_count": len(selected), "order_count": len(judgments) * 2,
        "parse_valid_rate": summary["parse_valid_rate"]})
    print(json.dumps({"sample_count": len(selected),
                      "order_count": len(judgments) * 2,
                      "parse_valid_rate": summary["parse_valid_rate"],
                      "endpoint_call_counts": summary["endpoint_call_counts"]}, indent=2))


def _score_generic_judgments(
        rows: Sequence[dict[str, Any]],
        judgments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    scored = []
    for item in judgments:
        row = by_id.get(item["sample_id"])
        if row is None:
            raise RuntimeError("Generic judgment sample is not in candidate pool")
        original, swapped = item["original"], item["swapped"]
        if not original["parse_ok"] or not swapped["parse_ok"]:
            raise RuntimeError("technical failure cannot be scored as a hard sample")
        original_answer = original["answer"]
        swapped_mapped = _invert_screen_vote(swapped["answer"])
        if swapped_mapped in {"none", "abstain"}:
            swapped_mapped = "None"
        original_wrong = original_answer != row["answer"]
        swapped_wrong = swapped_mapped != row["answer"]
        inconsistent = original_answer != swapped_mapped
        hardness = int(original_wrong) + int(swapped_wrong) + int(inconsistent)
        scored.append({**row, "generic_screen": {
            "prompt_version": GENERIC_SCREEN_PROMPT_VERSION,
            "gold": row["answer"], "original_answer": original_answer,
            "swapped_answer_raw": swapped["answer"],
            "swapped_answer_mapped": swapped_mapped,
            "original_wrong": original_wrong, "swapped_wrong": swapped_wrong,
            "position_inconsistent": inconsistent, "hardness": hardness,
            "original_attempt_count": original["attempt_count"],
            "swapped_attempt_count": swapped["attempt_count"],
        }})
    return scored


def _select_hard_candidates(scored: Sequence[dict[str, Any]], *,
                            coverage_ids: set[str], source_names: Sequence[str],
                            per_source: int, seed: int) -> list[dict[str, Any]]:
    selected = []
    for source in source_names:
        values = [row for row in scored
                  if row["source"] == source and row["sample_id"] not in coverage_ids]
        values.sort(key=lambda row: (
            -int(row["generic_screen"]["hardness"]),
            _sha([GENERIC_SCREEN_PROMPT_VERSION, seed, source, row["sample_id"]])))
        if len(values) < per_source:
            raise RuntimeError(
                f"Generic Hard selection source {source} has only {len(values)} rows")
        selected.extend(values[:per_source])
    return selected


def generic_screen(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-selection-freeze")
    _require(target, "discovery-v2-generic-screen-smoke")
    rows_path = target / "candidates/discovery_candidates.jsonl"
    coverage_path = target / "selection_v2/coverage_candidates_150.jsonl"
    rows = _read_jsonl(rows_path)
    coverage = _read_jsonl(coverage_path)
    source_names = [source["name"] for source in cfg["sources"] if source["enabled"]]
    request_spec = _generic_request_spec(config, cfg)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": GENERIC_SCREEN_PROMPT_VERSION,
        "request_spec": request_spec,
        "discovery_candidates": {"count": len(rows),
                                  "sha256": file_sha256(rows_path)},
        "coverage_candidates": {"count": len(coverage),
                                 "sha256": file_sha256(coverage_path)},
        "source_names": source_names,
        "hard_candidates_per_source": cfg["generic_screening"]["hard_candidates_per_source"],
        "coverage_excluded_from_hard": True,
        "five_root_screening_used": False,
    }
    manifest_path = target / "selection_v2/generic_screen/frozen_manifest.json"
    if manifest_path.is_file() and load_json(manifest_path) != manifest:
        raise RuntimeError("Generic screening frozen manifest drift")
    if not manifest_path.is_file():
        atomic_write_json(manifest_path, manifest)

    judgments, summary = _run_generic_orders(
        config, cfg, target, rows, label="generic_screen_full")
    failures = summary["parse_failure_sample_orders"]
    if failures:
        atomic_write_json(target / "selection_v2/generic_screen/incomplete_report.json", {
            "schema_version": SCHEMA_VERSION, "unresolved_count": len(failures),
            "unresolved_sample_orders": failures,
            "rerun_stage": "discovery-v2-generic-screen"})
        raise RuntimeError(
            f"Generic screening has {len(failures)} unresolved technical failures; "
            "rerun the same stage to retry only failed orders")

    scored = _score_generic_judgments(rows, judgments)
    coverage_ids = {row["sample_id"] for row in coverage}
    per_source = int(cfg["generic_screening"]["hard_candidates_per_source"])
    hard = _select_hard_candidates(
        scored, coverage_ids=coverage_ids, source_names=source_names,
        per_source=per_source, seed=int(cfg["seed"]))
    if len(hard) != per_source * len(source_names):
        raise RuntimeError("Generic Hard candidate count mismatch")
    if coverage_ids & {row["sample_id"] for row in hard}:
        raise RuntimeError("Coverage and Generic Hard candidates overlap")
    hard_path = target / "selection_v2/hard_candidates_90.jsonl"
    _write_jsonl(hard_path, hard)
    score_histogram = Counter(
        str(row["generic_screen"]["hardness"]) for row in scored)
    hard_histogram = Counter(
        str(row["generic_screen"]["hardness"]) for row in hard)
    report_value = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": GENERIC_SCREEN_PROMPT_VERSION,
        "screened_samples": len(rows), "logical_order_count": len(rows) * 2,
        "parse_valid_rate": summary["parse_valid_rate"],
        "original_accuracy": sum(
            row["generic_screen"]["original_answer"] == row["answer"]
            for row in scored) / len(scored),
        "swapped_mapped_accuracy": sum(
            row["generic_screen"]["swapped_answer_mapped"] == row["answer"]
            for row in scored) / len(scored),
        "position_consistency_rate": sum(
            not row["generic_screen"]["position_inconsistent"]
            for row in scored) / len(scored),
        "score_histogram": dict(sorted(score_histogram.items())),
        "hard_candidate_count": len(hard),
        "hard_candidates_by_source": dict(sorted(
            Counter(row["source"] for row in hard).items())),
        "hard_score_histogram": dict(sorted(hard_histogram.items())),
        "coverage_hard_overlap": 0,
        "five_root_screening_used": False,
        "technical_failures_used_as_hardness": False,
        "current_run_metrics": summary["current_run_metrics"],
        "endpoint_call_counts": summary["endpoint_call_counts"],
        "artifacts": {"hard_candidates_90.jsonl": {
            "count": len(hard), "sha256": file_sha256(hard_path)}},
    }
    atomic_write_json(target / "selection_v2/generic_screen/report.json", report_value)
    _set_status(target, "discovery-v2-generic-screen", "passed", {
        "screened_samples": len(rows), "logical_order_count": len(rows) * 2,
        "parse_valid_rate": summary["parse_valid_rate"],
        "hard_candidate_count": len(hard), "coverage_hard_overlap": 0})
    print(json.dumps({key: report_value[key] for key in (
        "screened_samples", "logical_order_count", "parse_valid_rate",
        "original_accuracy", "swapped_mapped_accuracy",
        "position_consistency_rate", "hard_candidate_count",
        "hard_candidates_by_source", "coverage_hard_overlap")}, indent=2))


def _adjudication_prompt(row: Mapping[str, Any], swapped: bool) -> list[dict[str, Any]]:
    a, b = (row["B"], row["A"]) if swapped else (row["A"], row["B"])
    text = f"""Independently pre-adjudicate this multimodal preference pair. Use the image.
Question: {row['question']}
Candidate A: {a}
Candidate B: {b}

Return one JSON object with exactly these keys:
{{"answer":"A or B","confidence":1,"preference_rationale":"...","visual_evidence":["..."],"primary_error_type":"...","secondary_error_types":[],"boundary_type":null,"applicable_root_ids":[]}}
confidence is 1-4. boundary_type is null or one of {list(BOUNDARY_TYPES)}. Do not output a tie."""
    return [{"type": "image_url", "image_url": {"url": _image_data_url(row["image_path"])}},
            {"type": "text", "text": text}]


def _validate_adjudication(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"answer", "confidence", "preference_rationale", "visual_evidence",
              "primary_error_type", "secondary_error_types", "boundary_type",
              "applicable_root_ids"}
    if set(value) != fields or value["answer"] not in {"A", "B"}:
        raise ValueError("adjudication schema/answer invalid")
    if value["boundary_type"] not in {None, *BOUNDARY_TYPES}:
        raise ValueError("adjudication boundary_type invalid")
    if (isinstance(value["confidence"], bool) or not isinstance(value["confidence"], int)
            or not 1 <= value["confidence"] <= 4):
        raise ValueError("adjudication confidence invalid")
    for key in ("visual_evidence", "secondary_error_types", "applicable_root_ids"):
        if not isinstance(value[key], list):
            raise ValueError(f"adjudication {key} must be a list")
    return dict(value)


def _adjudicator_pool(config: Mapping[str, Any], cfg: Mapping[str, Any]):
    profile = cfg["adjudicator"]
    spec = BackendPoolSpec.from_dict(profile["backend_pool"])
    if spec.common_checkpoint_id != "Qwen/Qwen3.5-397B-A17B":
        raise RuntimeError("Discovery-v2 adjudicator must be Qwen/Qwen3.5-397B-A17B")
    key_env = profile["api_key_env"]
    api_keys = os.environ.get(key_env, "") if key_env else "EMPTY"
    if not api_keys:
        raise RuntimeError(f"environment variable {key_env!r} is required")
    return AvailableSlotBackendPool(spec), {
        "model": spec.common_checkpoint_id, "api_keys": api_keys,
        "request_kwargs": dict(profile["request_kwargs"]),
        "api_retry_attempts": int(profile["api_retry_attempts"]),
    }


def _adjudicate_one(pool, agent_args: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    outputs, raw_outputs = [], []
    max_parse_retries = 2
    for swapped in (False, True):
        parsed = None
        for attempt in range(1, max_parse_retries + 2):
            raw, _ = pool.call(_adjudication_prompt(row, swapped),
                request_type="discovery_v2_adjudication",
                request_key=f"{row['sample_id']}::{'swap' if swapped else 'original'}",
                structured_attempt=attempt, agent_args=agent_args)
            raw_outputs.append(raw)
            try:
                parsed = _validate_adjudication(_extract_object(raw))
                break
            except (ValueError, json.JSONDecodeError):
                continue
        if parsed is None:
            raise RuntimeError(f"397B adjudication parse failed for {row['sample_id']}")
        if swapped:
            parsed["answer"] = "B" if parsed["answer"] == "A" else "A"
        outputs.append(parsed)
    disagreement = outputs[0]["answer"] != outputs[1]["answer"]
    primary = outputs[0]
    return {**dict(row), "adjudication": {"original": outputs[0], "swapped": outputs[1],
        "suggested_answer": primary["answer"] if not disagreement else None,
        "model_disagreement": disagreement, "raw_responses": raw_outputs}}


def _adjudication_fingerprint(row: Mapping[str, Any]) -> str:
    return _sha({key: row[key] for key in (
        "sample_id", "image_sha256", "question", "A", "B")})


def _adjudicate_cached(pool, agent_args: Mapping[str, Any], row: Mapping[str, Any],
                       item_path: Path) -> tuple[dict[str, Any], bool]:
    fingerprint = _adjudication_fingerprint(row)
    if item_path.is_file():
        cached = load_json(item_path)
        if cached.get("input_fingerprint") != fingerprint:
            raise RuntimeError(f"adjudication cache drift for {row['sample_id']}")
        return dict(cached["record"]), True
    result = _adjudicate_one(pool, agent_args, row)
    atomic_write_json(item_path, {"schema_version": SCHEMA_VERSION,
        "input_fingerprint": fingerprint, "record": result})
    return result, False


def adjudicate(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-screen")
    discovery = _read_jsonl(target / "candidates/discovery_screened_ranked.jsonl")
    dev = _read_jsonl(target / "candidates/dev_candidates.jsonl")
    pool, agent_args = _adjudicator_pool(config, cfg)
    combined = [("discovery", row) for row in discovery] + [("dev", row) for row in dev]
    results: list[tuple[str, dict[str, Any]] | None] = [None] * len(combined)
    item_dir = target / "adjudication/items"
    item_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=pool.spec.global_request_concurrency) as executor:
        futures = {executor.submit(_adjudicate_cached, pool, agent_args, row,
                    item_dir / f"{_sha([role, row['sample_id']])}.json"): (index, role)
                   for index, (role, row) in enumerate(combined)}
        completed = 0
        reused = 0
        for future in as_completed(futures):
            index, role = futures[future]
            record, was_reused = future.result()
            results[index] = (role, record)
            reused += int(was_reused)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(results):
                print(f"discovery-v2 adjudicate {completed}/{len(results)}")
    discovery_out = [item[1] for item in results if item and item[0] == "discovery"]
    dev_out = [item[1] for item in results if item and item[0] == "dev"]
    _write_jsonl(target / "adjudication/discovery.jsonl", discovery_out)
    _write_jsonl(target / "adjudication/dev.jsonl", dev_out)
    report = {"discovery_count": len(discovery_out), "dev_count": len(dev_out),
              "dev_worker_screen_calls": 0, "dev_selection_model_calls": 0,
              "dev_adjudication_calls": len(dev_out) * 2,
              "generated_items": len(results) - reused, "reused_items": reused,
              "model_disagreement_count": sum(item["adjudication"]["model_disagreement"]
                                                for item in discovery_out + dev_out),
              "provenance": pool.provenance_dict()}
    atomic_write_json(target / "adjudication/report.json", report)
    _set_status(target, "discovery-v2-adjudicate", "passed", {
        key: report[key] for key in ("discovery_count", "dev_count", "model_disagreement_count")})
    print(json.dumps({key: report[key] for key in (
        "discovery_count", "dev_count", "model_disagreement_count")}, indent=2))


def _blind_item(row: Mapping[str, Any], queue_rank: int) -> dict[str, Any]:
    return {"review_id": _sha([row["sample_id"], "blind-review-v1"])[:24],
            "queue_rank": queue_rank, "image_path": row["image_path"],
            "question": row["question"], "A": row["A"], "B": row["B"],
            "domain": row["domain"], "candidate_role": row["candidate_role"]}


def _review_template(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"review_id": item["review_id"], "answer": None, "confidence": None,
            "preference_rationale": "", "visual_evidence": [],
            "primary_error_type": "", "secondary_error_types": [],
            "boundary_type": None, "applicable_root_ids": [],
            "decision": None, "reviewed": False, "reconciled": False,
            "reconciliation_notes": ""}


def review_export(config: Mapping[str, Any], output: Path) -> None:
    _config(config); target = _target(output)
    _require(target, "discovery-v2-adjudicate")
    discovery = _read_jsonl(target / "adjudication/discovery.jsonl")
    dev = _read_jsonl(target / "adjudication/dev.jsonl")
    # Stable rank: screened difficulty only affects Discovery, never Dev.
    discovery.sort(key=lambda row: (-row.get("worker_screen", {}).get("hardness", 0), row["sample_id"]))
    dev.sort(key=lambda row: _sha([42, row["source_family"], row["sample_id"]]))
    rows = discovery + dev
    blind = [_blind_item(row, index + 1) for index, row in enumerate(rows)]
    reference = {item["review_id"]: row for item, row in zip(blind, rows)}
    _write_jsonl(target / "review/blind_queue.jsonl", blind)
    reference_path = target / "review/hidden_reference.json"
    atomic_write_json(reference_path, reference)
    decisions_path = target / "review/human_review_decisions.jsonl"
    if not decisions_path.exists():
        _write_jsonl(decisions_path, (_review_template(item) for item in blind))
    rubric = build_multicrit_open_ended_init_rubric()
    guide_lines = ["# Discovery-v2 Blind Review Guide", "",
        "First pass: inspect only `blind_queue.jsonl`, then fill the decision template. "
        "Do not open `hidden_reference.json`.", "",
        "Use `decision=accept` for a usable first-pass record or `decision=reject` "
        "for an unusable record. The finalizer derives `corrected` when the reconciled "
        "human A/B label differs from the hidden source label.", "",
        "A boundary record still needs a global A/B preference, while one or more "
        "roots may be non-applicable and should abstain.", "", "## Initial roots", ""]
    for root_id in rubric.root_ids:
        criterion = rubric.get_node(root_id).criterion
        guide_lines.extend([f"- `{root_id}` / **{criterion.name}**: {criterion.description}"])
    guide_lines.extend(["", "Second pass: rerun `discovery-v2-review-export`, review only "
        "`reconciliation_queue.jsonl`, then set `reconciled=true` and document the decision."])
    (target / "review/REVIEW_GUIDE.md").write_text("\n".join(guide_lines) + "\n", encoding="utf-8")
    decisions = _read_jsonl(decisions_path)
    by_review = {item["review_id"]: item for item in decisions}
    reconciliation = []
    for item in blind:
        human = by_review.get(item["review_id"], {})
        if not human.get("reviewed"):
            continue
        row = reference[item["review_id"]]
        ai = row["adjudication"]
        if (ai["model_disagreement"] or human.get("answer") != ai.get("suggested_answer")
                or human.get("answer") != row.get("answer")):
            reconciliation.append({**item, "human": human, "source_answer": row["answer"],
                "adjudication": ai, "requires_reconciliation": True})
    _write_jsonl(target / "review/reconciliation_queue.jsonl", reconciliation)
    report = {"blind_count": len(blind), "completed_reviews": sum(
        bool(item.get("reviewed")) for item in decisions),
        "reconciliation_count": len(reconciliation),
        "blind_fields_hide_source_and_models": True}
    atomic_write_json(target / "review/export_report.json", report)
    _set_status(target, "discovery-v2-review-export", "passed", report)
    print(json.dumps(report, indent=2))


def _validated_human(value: Mapping[str, Any]) -> bool:
    return (value.get("reviewed") is True and value.get("decision") in {"accept", "corrected", "reject"}
            and value.get("answer") in {"A", "B"}
            and isinstance(value.get("confidence"), int) and value["confidence"] >= 3
            and bool(_text(value.get("preference_rationale")))
            and isinstance(value.get("visual_evidence"), list))


def _select_quota(rows: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str],
                  quotas: Mapping[str, int], *, label: str) -> list[dict[str, Any]]:
    selected = []
    counts = Counter()
    for row in rows:
        cell = key(row)
        if cell in quotas and counts[cell] < quotas[cell]:
            selected.append(row); counts[cell] += 1
    missing = {cell: count-counts[cell] for cell, count in quotas.items() if counts[cell] < count}
    if missing:
        raise RuntimeError(f"insufficient reviewed records for {label}: {missing}")
    return selected


def _final_record(row: Mapping[str, Any], review: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {"sample_id": row["sample_id"], "image_path": row["image_path"],
        "question": row["question"], "A": row["A"], "B": row["B"],
        "answer": review["answer"], "split_role": role, "domain": row["domain"],
        "subdomain": row["subdomain"], "boundary_type": review.get("boundary_type"),
        "source": row["source"], "source_family": row["source_family"],
        "source_sample_id": row["source_sample_id"], "label_origin": "human_reviewed",
        "preference_confidence": review["confidence"],
        "preference_rationale": review["preference_rationale"],
        "visual_evidence": list(review["visual_evidence"]),
        "primary_error_type": review["primary_error_type"],
        "secondary_error_types": list(review.get("secondary_error_types", [])),
        "applicable_root_ids": list(review.get("applicable_root_ids", [])),
        "image_sha256": row["image_sha256"], "question_sha256": row["question_sha256"],
        "unordered_pair_sha256": row["unordered_pair_sha256"],
        "human_review": {"decision": (
            "corrected" if review["answer"] != row["answer"] else "accept"),
            "reviewed": True}}


def _deterministic_ab(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(sorted(rows, key=lambda item: item["sample_id"])):
        value = dict(row)
        # Exactly balanced when even; one-position difference when odd.
        want_a = index % 2 == 0
        if (value["answer"] == "A") != want_a:
            value["A"], value["B"] = value["B"], value["A"]
            value["answer"] = "B" if value["answer"] == "A" else "A"
        value["unordered_pair_sha256"] = _pair_sha(value["A"], value["B"])
        result.append(value)
    random.Random(seed).shuffle(result)
    return result


def _assert_isolation(discovery: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]]) -> None:
    keys = ("image_sha256", "question_sha256", "source_sample_id", "unordered_pair_sha256")
    for key in keys:
        overlap = {row[key] for row in discovery} & {row[key] for row in dev}
        if overlap:
            raise RuntimeError(f"Discovery/Dev overlap on {key}: {len(overlap)}")


def _assert_dev_not_screened(target: Path, dev_rows: Sequence[Mapping[str, Any]]) -> None:
    dev_ids = {row["sample_id"] for row in dev_rows}
    for path in (target / "screen/predictions").glob("*.json"):
        value = load_json(path)
        sample_ids = value.get("sample_ids", []) if isinstance(value, dict) else []
        normalized = {str(item).removesuffix("::swap") for item in sample_ids}
        overlap = dev_ids & normalized
        if overlap:
            raise RuntimeError(f"D_dev leaked into Worker screening: {len(overlap)} records")


def finalize(config: Mapping[str, Any], output: Path) -> None:
    cfg, target = _config(config), _target(output)
    _require(target, "discovery-v2-review-export")
    reference = load_json(target / "review/hidden_reference.json")
    decisions = _read_jsonl(target / "review/human_review_decisions.jsonl")
    reviewed = []
    for decision in decisions:
        if not _validated_human(decision):
            continue
        if decision["decision"] == "reject":
            continue
        row = reference.get(decision["review_id"])
        if row is not None:
            ai = row["adjudication"]
            needs_reconciliation = (ai["model_disagreement"]
                or decision["answer"] != ai.get("suggested_answer")
                or decision["answer"] != row.get("answer"))
            if needs_reconciliation and decision.get("reconciled") is not True:
                raise RuntimeError(
                    f"review {decision['review_id']} requires completed reconciliation")
            reviewed.append((row, decision))
    discovery_pairs = [(row, decision) for row, decision in reviewed
                       if row["candidate_role"] == "discovery"]
    dev_pairs = [(row, decision) for row, decision in reviewed if row["candidate_role"] == "dev"]
    evolve_candidates = [_final_record(row, decision, "evolve") for row, decision in discovery_pairs
                         if decision.get("boundary_type") is None]
    boundary_candidates = [_final_record(row, decision, "boundary") for row, decision in discovery_pairs
                           if decision.get("boundary_type") in BOUNDARY_TYPES]
    evolve = _select_quota(evolve_candidates, lambda row: row["domain"],
                           cfg["quotas"]["evolve"], label="D_evolve")
    boundary = _select_quota(boundary_candidates, lambda row: row["boundary_type"],
                             cfg["quotas"]["boundary"], label="D_boundary")
    dev_candidates = [_final_record(row, decision, "dev") for row, decision in dev_pairs]
    dev_quotas = {f"{domain}:{kind}": count for domain, values in cfg["quotas"]["dev"].items()
                  for kind, count in values.items()}
    def dev_cell(row: dict[str, Any]) -> str:
        return f"{row['domain']}:{'boundary' if row['boundary_type'] else 'regular'}"
    dev = _select_quota(dev_candidates, dev_cell, dev_quotas, label="D_dev")
    evolve, boundary, dev = (_deterministic_ab(evolve, 42),
                             _deterministic_ab(boundary, 43), _deterministic_ab(dev, 44))
    discovery = evolve + boundary
    _assert_isolation(discovery, dev)
    _assert_dev_not_screened(target, dev)
    for row in discovery + dev:
        if set(row) != REQUIRED_FINAL_FIELDS or row["answer"] not in {"A", "B"}:
            raise RuntimeError("final record schema drift")
        if not Path(row["image_path"]).is_file():
            raise RuntimeError(f"final image is unreadable: {row['image_path']}")
    source_counts = Counter(row["source"] for row in discovery)
    if any(count > 35 for count in source_counts.values()):
        raise RuntimeError(f"Discovery source cap exceeded: {source_counts}")
    if sum(1 for row in discovery if row["source_family"] == "rlhf_v") > 25:
        raise RuntimeError("RLHF-V cap exceeded")
    for domain in DOMAINS:
        families = {row["source_family"] for row in discovery if row["domain"] == domain}
        if len(families) < 2:
            raise RuntimeError(f"Discovery domain {domain} needs at least two source families")
    # Validate benchmark isolation again on the exact finalized records rather
    # than trusting only the larger ingest pool.
    exclusion = _benchmark_exclusion(base._path(cfg["vl_rewardbench_path"]))
    for row in discovery + dev:
        if row["source_sample_id"] in exclusion["sample_ids"]:
            raise RuntimeError("final VL-RewardBench overlap on sample_id")
        if row["image_sha256"] in exclusion["image_sha256"]:
            raise RuntimeError("final VL-RewardBench overlap on image_sha256")
        image_question = _image_question_sha(row["image_sha256"], row["question_sha256"])
        if image_question in exclusion["image_question_sha256"]:
            raise RuntimeError("final VL-RewardBench overlap on image_question_sha256")
        if row["unordered_pair_sha256"] in exclusion["unordered_pair_sha256"]:
            raise RuntimeError("final VL-RewardBench overlap on unordered_pair_sha256")
    final_dir = target / "final"
    _write_jsonl(final_dir / "d_evolve_75.jsonl", evolve)
    _write_jsonl(final_dir / "d_boundary_25.jsonl", boundary)
    _write_jsonl(final_dir / "discovery_100.jsonl", discovery)
    _write_jsonl(final_dir / "dev_150.jsonl", dev)
    artifacts = {path.name: {"count": len(_read_jsonl(path)), "sha256": file_sha256(path)}
                 for path in final_dir.glob("*.jsonl")}
    source_lock_value = load_json(target / "source_lock.json")
    manifest = {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION,
                "seed": 42, "artifacts": artifacts, "source_lock_sha256":
                source_lock_value["lock_sha256"],
                "dev_selection_model_calls": 0, "manager_visible": {
                    "d_evolve": True, "d_boundary": True, "d_dev": False},
                "checkpoint_selection_uses_dev": False}
    manifest["manifest_sha256"] = _sha(manifest)
    atomic_write_json(final_dir / "dataset_manifest.json", manifest)
    source_lines = []
    for item in source_lock_value["sources"]:
        if not item["disabled"]:
            source_lines.append(
                f"| {item['name']} | `{item['repo_id']}` | `{item['resolved_revision']}` | "
                f"{item.get('remote_license') or item['license']} |")
    card = f"""# Discovery-v2 Dataset Card

Discovery-v2 contains 75 evolution pairs, 25 boundary pairs, and 150 independent
development pairs. All labels were fully reviewed by a human. D_dev was frozen
before Worker screening and is excluded from Manager context, active selection,
early stopping, and checkpoint selection.

Protocol: `{PROTOCOL_VERSION}`
Manifest SHA-256: `{manifest['manifest_sha256']}`

The JSONL files contain local content-addressed image paths and are intended for
the authorized local research environment. Review source licenses before reuse.

## Locked sources

| Source | Repository | Immutable revision | License |
|---|---|---|---|
{chr(10).join(source_lines)}
"""
    (final_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    _set_status(target, "discovery-v2-finalize", "passed", {
        "evolve": len(evolve), "boundary": len(boundary), "dev": len(dev),
        "manifest_sha256": manifest["manifest_sha256"]})
    print(json.dumps({"evolve": len(evolve), "boundary": len(boundary),
                      "dev": len(dev), "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(_text(row.get(field)) or "null" for row in rows).items()))


def report(config: Mapping[str, Any], output: Path) -> None:
    _config(config); target = _target(output)
    _require(target, "discovery-v2-finalize")
    evolve = _read_jsonl(target / "final/d_evolve_75.jsonl")
    boundary = _read_jsonl(target / "final/d_boundary_25.jsonl")
    dev = _read_jsonl(target / "final/dev_150.jsonl")
    all_rows = evolve + boundary + dev
    adjudicated = (_read_jsonl(target / "adjudication/discovery.jsonl")
                   + _read_jsonl(target / "adjudication/dev.jsonl"))
    decisions = _read_jsonl(target / "review/human_review_decisions.jsonl")
    final_ids = {row["sample_id"] for row in all_rows}
    reviewed_final = [row for row in all_rows if row["human_review"]["reviewed"]]
    corrected = sum(row["human_review"]["decision"] == "corrected"
                    for row in reviewed_final)
    lengths = [len(row[side]) for row in all_rows for side in ("A", "B")]
    report_value = {"schema_version": SCHEMA_VERSION,
        "counts": {"evolve": len(evolve), "boundary": len(boundary), "dev": len(dev)},
        "domain": {"evolve": _distribution(evolve, "domain"),
                   "boundary": _distribution(boundary, "domain"),
                   "dev": _distribution(dev, "domain")},
        "sources": _distribution(all_rows, "source"),
        "source_families": _distribution(all_rows, "source_family"),
        "boundary_types": _distribution(boundary, "boundary_type"),
        "gold_position": _distribution(all_rows, "answer"),
        "answer_length_chars": {"min": min(lengths), "max": max(lengths),
            "mean": sum(lengths) / len(lengths)},
        "human_correction_rate": corrected / len(reviewed_final) if reviewed_final else 0,
        "model_disagreement_rate": sum(row["adjudication"]["model_disagreement"]
            for row in adjudicated if row["sample_id"] in final_ids) / len(all_rows),
        "deduplication": load_json(target / "ingest_report.json")["deduplication"],
        "quality": {"human_reviewed_rate": 1.0, "missing_images": 0,
            "exact_duplicates": 0, "benchmark_exact_overlap": 0},
        "scientific_isolation": {"dev_selection_model_calls": 0,
            "dev_in_manager_context": False, "dev_used_for_checkpoint_selection": False}}
    atomic_write_json(target / "final/report.json", report_value)
    lines = ["# Discovery-v2 Collection Report", "",
             f"- D_evolve: {len(evolve)}", f"- D_boundary: {len(boundary)}",
             f"- D_dev: {len(dev)}", "",
             "D_dev was not screened by the Worker and is not visible to Managers.", "",
             "## Domain counts", "", "```json",
             json.dumps(report_value["domain"], indent=2, ensure_ascii=False), "```", "",
             "## Boundary counts", "", "```json",
             json.dumps(report_value["boundary_types"], indent=2, ensure_ascii=False), "```"]
    (target / "final/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _set_status(target, "discovery-v2-report", "passed", report_value["counts"])
    print(json.dumps(report_value["counts"], indent=2))


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    from . import discovery_data_v2_adjudication as adjudication_v2
    from . import discovery_data_v2_demo as demo

    actions = {"discovery-v2-source-lock": source_lock,
               "discovery-v2-ingest": ingest,
               "discovery-v2-selection-freeze": selection_freeze,
               "discovery-v2-generic-screen-smoke": generic_screen_smoke,
               "discovery-v2-generic-screen": generic_screen,
               "discovery-v2-adjudication-freeze": adjudication_v2.freeze,
               "discovery-v2-adjudication-smoke": adjudication_v2.smoke,
               "discovery-v2-adjudication-run": adjudication_v2.run,
               "discovery-v2-adjudication-report": adjudication_v2.report,
               "discovery-v2-review-v2-export": adjudication_v2.review_export,
               "discovery-v2-demo-freeze": demo.freeze,
               "discovery-v2-demo-smoke": demo.smoke,
               "discovery-v2-demo-adjudicate": demo.adjudicate,
               "discovery-v2-demo-report": demo.report,
               "discovery-v2-demo-review-export": demo.review_export,
               "discovery-v2-demo-finalize": demo.finalize,
               "discovery-v2-demo-export": demo.export_data,
               "discovery-v2-screen-smoke": screen_smoke,
               "discovery-v2-screen": screen,
               "discovery-v2-adjudicate": adjudicate,
               "discovery-v2-review-export": review_export,
               "discovery-v2-finalize": finalize,
               "discovery-v2-report": report}
    if stage not in actions:
        raise ValueError(f"unsupported Discovery-v2 stage: {stage}")
    started = time.monotonic()
    try:
        actions[stage](config, output)
    except Exception as exc:
        _target(output).mkdir(parents=True, exist_ok=True)
        _set_status(_target(output), stage, "failed", {"error": str(exc),
                    "error_type": type(exc).__name__})
        raise
    print(f"{stage} elapsed_seconds={time.monotonic()-started:.1f}")

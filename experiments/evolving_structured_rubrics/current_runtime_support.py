"""Multimodal model calls shared by the current Worker, Arbiter, and Manager."""
from __future__ import annotations
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from critiq.agent import Agent, AgentCallMetrics
from critiq.structured.backend_pool import BackendEndpointSpec
from .experiment_utils import atomic_write_json, canonical_sha256, load_json


SCHEMA_VERSION = "1.0.0"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/jpeg"
    with path.open("rb") as handle:
        image_b64 = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{image_b64}"


def vlrb_rows(items: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple({
        "sample_id": str(item["sample_id"]),
        "image_path": str(item["image_path"]),
        "question": str(item["question"]),
        "A": str(item["responses"][0]),
        "B": str(item["responses"][1]),
        "answer": "A" if int(item["preferred_original_index"]) == 0 else "B",
    } for item in items)


def ordered_row(row: Mapping[str, Any], order: int, replicate: int) -> dict[str, Any]:
    result = dict(row)
    result["sample_id"] = f"{row['sample_id']}::k{replicate + 1}"
    if order:
        result["A"], result["B"] = str(row["B"]), str(row["A"])
        result["answer"] = "B" if row["answer"] == "A" else "A"
    return result


def content(row: Mapping[str, Any], user_text: str) -> list[dict[str, Any]]:
    image_url = _image_path_to_data_url(str(row["image_path"]))
    return [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": user_text},
    ]


def metric_dict(metrics: AgentCallMetrics | None) -> dict[str, Any]:
    return dict((metrics or AgentCallMetrics()).__dict__)


def sum_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        key: sum(float(item.get(key) or 0) for item in items)
        for key in ("api_attempts", "error_count", "input_tokens", "output_tokens",
                    "latency_seconds")
    }
    for key in ("api_attempts", "error_count", "input_tokens", "output_tokens"):
        result[key] = int(result[key])
    return result


def compact_call(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "parse_ok", "parsed", "raw_response", "model_generation_count",
        "endpoint_id", "cache_hit", "metrics", "error", "cache_key",
    )
    return {key: value[key] for key in keys if key in value}


def global_arbiter_user_prompt(
    row: Mapping[str, Any], reports: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "## Source Instruction or Question", str(row["question"]), "",
        "## Candidate A", str(row["A"]), "",
        "## Candidate B", str(row["B"]), "",
        "## Subtree Assessments",
    ]
    for item in reports:
        lines.extend([
            "", f"### {item['criterion_name']}",
            json.dumps(item["report"], ensure_ascii=False, separators=(",", ":")),
        ])
    lines.extend([
        "", "Integrate the image evidence and all subtree assessments into one final preference.",
    ])
    return "\n".join(lines)


def call_one(
    config: Mapping[str, Any], endpoint: BackendEndpointSpec, cache_dir: Path,
    *, user_text: str, row: Mapping[str, Any], request_key: Mapping[str, Any],
    total_attempt_limit: int, protocol_version: str, prompt_version: str,
    system_prompt: str, response_parser: Callable[[object], dict[str, str]],
    settings_loader: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    settings = settings_loader(config)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": protocol_version,
        "prompt_version": prompt_version,
        "model": config["model"],
        "endpoint_checkpoint": endpoint.checkpoint_root,
        "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
        "image_sha256": file_sha256(Path(str(row["image_path"]))),
        "request": {
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "generation_seed_policy": settings["generation_seed_policy"],
        },
        "request_key": dict(request_key),
    }
    key = canonical_sha256(payload)
    path = cache_dir / key[:2] / f"{key}.json"
    cached = None
    if path.is_file():
        candidate = load_json(path)
        if candidate.get("request") == payload:
            cached = candidate
            if candidate.get("parse_ok"):
                candidate["cache_hit"] = True
                return candidate
            try:
                parsed = response_parser(candidate.get("raw_response"))
            except Exception:
                pass
            else:
                candidate.update({
                    "parse_ok": True, "parsed": parsed, "cache_hit": True,
                    "parser_recovered": True,
                })
                candidate.pop("error", None)
                atomic_write_json(path, candidate)
                return candidate

    attempts = list((cached or {}).get("attempts", []))
    generations = int((cached or {}).get("model_generation_count", len(attempts)))
    metric_items = list((cached or {}).get("attempt_metrics", []))
    last_raw = (cached or {}).get("raw_response")
    last_error = (cached or {}).get("error")
    for _ in range(max(0, total_attempt_limit - generations)):
        attempt_number = generations + 1
        agent = Agent(
            system=system_prompt,
            model=config["model"],
            base_url=endpoint.base_url,
            api_keys="EMPTY",
            request_kwargs={
                "temperature": settings["temperature"],
                "max_tokens": settings["max_tokens"],
            },
            api_retry_attempts=config["api_retry_attempts"],
        )
        raw = None
        try:
            raw = agent(content(row, user_text), stream=False)
            parsed = response_parser(raw)
        except Exception as exc:
            last_raw = raw
            last_error = f"{type(exc).__name__}: {exc}"
            metric_items.append(metric_dict(agent.last_call_metrics))
            attempts.append({
                "attempt": attempt_number, "error": last_error,
                "raw_response": last_raw,
            })
            generations += 1
            continue
        metric_items.append(metric_dict(agent.last_call_metrics))
        result = {
            "request": payload, "parse_ok": True, "parsed": parsed,
            "raw_response": raw, "model_generation_count": attempt_number,
            "endpoint_id": endpoint.endpoint_id, "cache_hit": False,
            "metrics": sum_metrics(metric_items), "attempt_metrics": metric_items,
            "attempts": attempts, "cache_key": key,
        }
        atomic_write_json(path, result)
        return result
    result = {
        "request": payload, "parse_ok": False, "parsed": None,
        "raw_response": last_raw, "model_generation_count": generations,
        "endpoint_id": endpoint.endpoint_id, "cache_hit": bool(cached),
        "metrics": sum_metrics(metric_items), "attempt_metrics": metric_items,
        "attempts": attempts, "error": last_error or "parse failure",
        "cache_key": key,
    }
    atomic_write_json(path, result)
    return result

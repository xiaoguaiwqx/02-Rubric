"""Generate Hallucination100 R0 roots from five stateful CritiQ-style examples.

The five warm-up replies are shared by G5 and GN. Only their final generation
request differs. Checkpoints contain text and image hashes, never image bytes.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Any, Mapping, Sequence

from critiq.agent import API_REQUEST_TIMEOUT_SECONDS, Agent
from critiq.structured.schema import (
    RubricCriterionSnapshot,
    RubricNode,
    StructuredRubric,
)
from critiq.utils import parse_json, random_reverse

from . import _global_arbiter_ab_only_support as image_support
from .experiment_utils import atomic_write_json, load_json
from .framework_v6_manager import Manager


PROTOCOL = "hallucination100-generated-roots-v1"
REVERSE_SEED = 100745534
WARMUP_COUNT = 5
MIN_ROOTS = 2
MAX_ROOTS = 7

WARMUP_PROMPT = """Question: {question}
Response A: {A}
Response B: {B}
Human preference: {answer}
Why might a human prefer that response for this image and question?
Explain the evidence you can verify; say when the preference is uncertain."""

GENERATION_PROMPT = """From the {warmup_count} comparisons above, propose high-level root responsibilities for
judging new image/question/response pairs. A root must be reusable, distinct
from the others, and broad enough to support more specific child criteria later.
Describe what it checks and how it informs a relative A/B/None judgment.
Do not copy sample-specific facts, human labels, or a fixed preference for A/B.
Return JSON only: {"count_reason": "...", "roots":
[{"name": "short_unique_name", "description": "..."}]}.
"""

COUNT_INSTRUCTIONS = {
    "g5": "Return exactly five roots.",
    "gn": "Choose the number of roots from two through seven according to the distinct recurring responsibilities in these examples.",
}

# Preserve the prompt used by the already-running Hallucination100 comparison.
LEGACY_COUNT_INSTRUCTIONS = dict(
    COUNT_INSTRUCTIONS,
    gn="Choose the number of roots from two through seven according to the distinct recurring responsibilities in these examples; five is allowed.",
)


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _warmup_rows(rows: Sequence[Mapping[str, Any]], seed: int,
                 expected_count: int = 100,
                 warmup_examples: Sequence[Mapping[str, Any]] | None = None,
                 ) -> list[dict[str, Any]]:
    """Preserve CritiQ's random_reverse result without changing global RNG state."""
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} frozen training rows, got {len(rows)}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("training sample IDs are not unique")
    if warmup_examples is None:
        if expected_count < WARMUP_COUNT:
            raise ValueError(f"expected at least {WARMUP_COUNT} training rows")
        chosen = random.Random(seed).sample(list(rows), WARMUP_COUNT)
    else:
        selected_ids = [str(row["sample_id"]) for row in warmup_examples]
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("warmup sample IDs must be nonempty and unique")
        by_id = {str(row["sample_id"]): row for row in rows}
        if any(sample_id not in by_id for sample_id in selected_ids):
            raise ValueError("warmup samples must belong to the frozen training rows")
        chosen = [by_id[sample_id] for sample_id in selected_ids]
    state = random.getstate()
    try:
        shuffled = random_reverse(chosen, seed=REVERSE_SEED)
    finally:
        random.setstate(state)
    originals = {str(row["sample_id"]): row for row in chosen}
    prepared = []
    for row in shuffled:
        sample_id = str(row["sample_id"])
        original = originals[sample_id]
        if row.get("answer") not in {"A", "B"}:
            raise ValueError(f"{sample_id}: expected human A/B preference")
        image_path = Path(str(row["image_path"]))
        image_sha256 = _file_sha256(image_path)
        expected_sha256 = row.get("image_sha256")
        if expected_sha256 and expected_sha256.lower() != image_sha256:
            raise ValueError(f"{sample_id}: image differs from frozen dataset hash")
        entry = dict(row)
        entry["sample_id"] = sample_id
        entry["image_path"] = str(image_path.resolve())
        entry["image_sha256"] = image_sha256
        entry["flipped"] = row["answer"] != original["answer"]
        entry["original_answer"] = original["answer"]
        prepared.append(entry)
    return prepared


def _warmup_prompt(row: Mapping[str, Any]) -> str:
    return WARMUP_PROMPT.format(
        question=row["question"], A=row["A"], B=row["B"], answer=row["answer"]
    )


def _warmup_request(rows: Sequence[Mapping[str, Any]], seed: int,
                    manager_config: Mapping[str, Any],
                    protocol: str = PROTOCOL) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "seed": seed,
        "reverse_seed": REVERSE_SEED,
        "sample_count": len(rows),
        "model": manager_config["model"],
        "base_url": manager_config["base_url"],
        "request_kwargs": manager_config["request_kwargs"],
        "agent_timeout_seconds": API_REQUEST_TIMEOUT_SECONDS,
        "agent_api_retry_attempts": 0,
        "sdk_max_retries": 0,
        "warmup_prompt_sha256": sha256(WARMUP_PROMPT.encode("utf-8")).hexdigest(),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "source": row.get("source"),
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "question": row["question"],
                "A": row["A"],
                "B": row["B"],
                "human_preference": row["answer"],
                "original_preference": row["original_answer"],
                "flipped": row["flipped"],
                "prompt": _warmup_prompt(row),
            }
            for row in rows
        ],
    }


def _agent(manager_config: Mapping[str, Any], keys: Sequence[str]) -> Agent:
    # Agent chooses an API key with global random. Keep that implementation
    # detail from perturbing subsequent experiment sampling.
    state = random.getstate()
    try:
        agent = Agent(
            system=None,
            model=str(manager_config["model"]),
            base_url=str(manager_config["base_url"]),
            api_keys=list(keys),
            request_kwargs=dict(manager_config["request_kwargs"]),
            api_retry_attempts=0,
        )
    finally:
        random.setstate(state)
    agent.client = agent.client.with_options(max_retries=0)
    return agent


def _fork(agent: Agent) -> Agent:
    state = random.getstate()
    try:
        branch = agent.fork()
    finally:
        random.setstate(state)
    # Agent.fork currently restores the default 50 internal retries.
    branch.api_retry_attempts = 0
    branch.client = branch.client.with_options(max_retries=0)
    return branch


def _warmup_history(agent: Agent, rows: Sequence[Mapping[str, Any]],
                    turns: Sequence[Mapping[str, Any]]) -> None:
    for row, turn in zip(rows, turns):
        if "response" not in turn:
            break
        content = image_support.content(row, turn["prompt"])
        agent.history.extend((
            {"role": "user", "content": content},
            {"role": "assistant", "content": turn["response"]},
        ))


def _run_warmup(rows: Sequence[Mapping[str, Any]], seed: int,
                manager_config: Mapping[str, Any], keys: Sequence[str],
                output_dir: Path, attempt_limit: int,
                protocol: str = PROTOCOL) -> tuple[Agent, dict[str, Any]]:
    path = output_dir / "warmup/transcript.json"
    request = _warmup_request(rows, seed, manager_config, protocol)
    record = load_json(path) if path.exists() else {"request": request, "turns": []}
    if record.get("request") != request:
        raise ValueError(f"warmup input changed at {path}; use a new run directory")
    if len(record["turns"]) > len(rows):
        raise ValueError(f"too many warmup turns in {path}")
    incomplete_seen = False
    for turn in record["turns"]:
        if "response" not in turn:
            incomplete_seen = True
        elif incomplete_seen:
            raise ValueError(f"warmup has a successful turn after an incomplete turn: {path}")
    agent = _agent(manager_config, keys)
    _warmup_history(agent, rows, record["turns"])
    for index, row in enumerate(rows):
        prompt = _warmup_prompt(row)
        if index < len(record["turns"]):
            turn = record["turns"][index]
            if turn["sample_id"] != row["sample_id"] or turn["prompt"] != prompt:
                raise ValueError(f"warmup turn changed at {path}: {index + 1}")
            if "response" in turn:
                continue
        else:
            turn = {"sample_id": row["sample_id"], "prompt": prompt, "attempts": []}
            record["turns"].append(turn)
            atomic_write_json(path, record)
        for attempt in range(len(turn["attempts"]) + 1, attempt_limit + 1):
            response = agent(image_support.content(row, prompt), stream=False)
            result = {"attempt": attempt, "metrics": asdict(agent.last_call_metrics)}
            if response is None:
                result["error"] = "Agent request failed; no turn was added to history"
            elif not response.strip():
                agent.forget_last_turn()
                result["error"] = "Agent returned an empty warmup reply"
            else:
                result["response"] = response
                turn["response"] = response
            turn["attempts"].append(result)
            atomic_write_json(path, record)
            if "response" in turn:
                break
        if "response" not in turn:
            raise RuntimeError(f"warmup sample {row['sample_id']} failed after {attempt_limit} attempts")
    return agent, record


def _normalized_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", name).casefold()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def _parse_roots(raw: str, variant: str) -> dict[str, Any]:
    result = parse_json(raw, allow_invalid_escapes=True)
    if not isinstance(result, dict):
        raise ValueError("root generation must return an object")
    reason = result.get("count_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("count_reason must be nonempty text")
    roots = result.get("roots")
    if not isinstance(roots, list):
        raise ValueError("roots must be a list")
    if variant == "g5" and len(roots) != 5:
        raise ValueError("G5 requires exactly five roots")
    if variant == "gn" and not MIN_ROOTS <= len(roots) <= MAX_ROOTS:
        raise ValueError("GN requires two through seven roots")
    normalized = set()
    clean = []
    for index, item in enumerate(roots, 1):
        if not isinstance(item, dict):
            raise ValueError(f"roots[{index}] must be an object")
        name, description = item.get("name"), item.get("description")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"roots[{index}].name must be nonempty text")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"roots[{index}].description must be nonempty text")
        name, description = name.strip(), description.strip()
        key = _normalized_name(name)
        if not key or key in normalized:
            raise ValueError(f"roots[{index}].name duplicates another normalized name")
        normalized.add(key)
        clean.append({"name": name, "description": description})
    return {"count_reason": reason.strip(), "roots": clean}


def _generation_request(variant: str, manager_config: Mapping[str, Any],
                        warmup: Mapping[str, Any],
                        count_instructions: Mapping[str, str],
                        protocol: str = PROTOCOL) -> dict[str, Any]:
    history = [
        {"sample_id": turn["sample_id"], "prompt": turn["prompt"],
         "response": turn["response"]}
        for turn in warmup["turns"]
    ]
    return {
        "protocol": protocol,
        "variant": variant,
        "warmup_request_sha256": _digest(warmup["request"]),
        "warmup_history_sha256": _digest(history),
        "model": manager_config["model"],
        "base_url": manager_config["base_url"],
        "request_kwargs": manager_config["request_kwargs"],
        "agent_timeout_seconds": API_REQUEST_TIMEOUT_SECONDS,
        "agent_api_retry_attempts": 0,
        "sdk_max_retries": 0,
        "prompt": GENERATION_PROMPT.replace(
            "{warmup_count}",
            {5: "five", 10: "ten"}.get(warmup["request"]["sample_count"],
                                      str(warmup["request"]["sample_count"])),
        ) + count_instructions[variant],
    }


def _rubric(parsed: Mapping[str, Any], variant: str,
            warmup: Mapping[str, Any],
            protocol: str = PROTOCOL) -> StructuredRubric:
    sample_ids = [sample["sample_id"] for sample in warmup["request"]["samples"]]
    nodes = {}
    root_ids = []
    for index, item in enumerate(parsed["roots"], 1):
        node_id = f"generated_root_{index:02d}"
        root_ids.append(node_id)
        nodes[node_id] = RubricNode(
            node_id=node_id,
            criterion=RubricCriterionSnapshot(item["name"], item["description"]),
            examples=(),
            lineage={
                "initialization": protocol,
                "variant": variant,
                "source_sample_ids": sample_ids,
                "source_order": index,
                "warmup_history_sha256": _digest([
                    {"sample_id": turn["sample_id"], "prompt": turn["prompt"],
                     "response": turn["response"]}
                    for turn in warmup["turns"]
                ]),
            },
        )
    return StructuredRubric(nodes=nodes, edges=(), root_ids=tuple(root_ids))


def _run_generation(base_agent: Agent, variant: str,
                    manager_config: Mapping[str, Any], warmup: Mapping[str, Any],
                    output_dir: Path, attempt_limit: int,
                    count_instructions: Mapping[str, str],
                    protocol: str = PROTOCOL) -> StructuredRubric:
    path = output_dir / variant / "r0/generation.json"
    request = _generation_request(variant, manager_config, warmup,
                                  count_instructions, protocol)
    record = load_json(path) if path.exists() else {"request": request, "attempts": []}
    if record.get("request") != request:
        raise ValueError(f"root generation input changed at {path}; use a new run directory")
    agent = _fork(base_agent)
    for previous in record["attempts"]:
        if "response" in previous:
            agent.history.extend((
                {"role": "user", "content": previous["prompt"]},
                {"role": "assistant", "content": previous["response"]},
            ))
    if "parsed" not in record:
        for attempt in range(len(record["attempts"]) + 1, attempt_limit + 1):
            previous_invalid = next(
                (item for item in reversed(record["attempts"])
                 if item.get("validation_error")), None
            )
            if previous_invalid is not None:
                prompt = (
                    "Your previous output did not match the required JSON structure: "
                    + previous_invalid["validation_error"]
                    + ("\nReturn a corrected JSON object with exactly five roots."
                       if variant == "g5" else
                       "\nReturn a corrected JSON object; choose any number "
                       "of roots from two through seven based on the examples.")
                )
            else:
                prompt = request["prompt"]
            response = agent(prompt, stream=False)
            result = {"attempt": attempt, "prompt": prompt,
                      "metrics": asdict(agent.last_call_metrics)}
            if response is None:
                result["error"] = "Agent request failed; no turn was added to history"
            else:
                result["response"] = response
                try:
                    parsed = _parse_roots(response, variant)
                    # Structural rubric validation is part of the format check.
                    _rubric(parsed, variant, warmup, protocol)
                except (ValueError, TypeError) as exc:
                    result["validation_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    record["parsed"] = parsed
            record["attempts"].append(result)
            atomic_write_json(path, record)
            if "parsed" in record:
                break
    if "parsed" not in record:
        raise RuntimeError(f"{variant} root generation failed after {attempt_limit} attempts: {path}")
    rubric = _rubric(record["parsed"], variant, warmup, protocol)
    rubric_path = output_dir / variant / "r0/rubric.json"
    if rubric_path.exists() and load_json(rubric_path) != rubric.to_dict():
        raise ValueError(f"generated R0 changed at {rubric_path}; use a new run directory")
    atomic_write_json(rubric_path, rubric.to_dict())
    return rubric


def generate_r0_pair(rows: Sequence[Mapping[str, Any]], *, seed: int,
                     manager_config: Mapping[str, Any], output_dir: Path,
                     attempt_limit: int = 10,
                     count_instructions: Mapping[str, str] | None = None,
                     expected_count: int = 100,
                     warmup_examples: Sequence[Mapping[str, Any]] | None = None,
                     variants: Sequence[str] = ("g5", "gn"),
                     protocol: str = PROTOCOL,
                     ) -> dict[str, StructuredRubric]:
    """Generate selected R0 variants from one frozen warmup conversation.

    ``rows`` must be original-order discovery rows with absolute image paths.
    Defaults retain the historical 100-row G5/GN experiment. Checkpointing
    resumes individual calls without re-running successful responses.
    """
    if attempt_limit < 1:
        raise ValueError("attempt_limit must be positive")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(COUNT_INSTRUCTIONS):
        raise ValueError("variants must be unique selections from g5 and gn")
    if count_instructions is None:
        count_instructions = COUNT_INSTRUCTIONS
    output_dir = Path(output_dir)
    prepared = _warmup_rows(rows, seed, expected_count, warmup_examples)
    keys = Manager(dict(manager_config), attempts=attempt_limit)._read_keys()
    agent, warmup = _run_warmup(
        prepared, seed, manager_config, keys, output_dir, attempt_limit,
        protocol
    )
    errors = {}
    rubrics = {}
    for variant in variants:
        try:
            rubrics[variant] = _run_generation(
                agent, variant, manager_config, warmup, output_dir,
                attempt_limit, count_instructions, protocol
            )
        except Exception as exc:
            errors[variant] = exc
    if errors:
        details = "; ".join(f"{variant}: {exc}" for variant, exc in errors.items())
        raise RuntimeError(f"root generation incomplete: {details}") from next(iter(errors.values()))
    return rubrics

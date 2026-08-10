"""Model-backed Manager for Refine v1, isolated behind strict parsers."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.agent import AgentCallMetrics
from critiq.refine_prompts import REFINE_FAILURE_ATTRIBUTION_PROMPT, REFINE_GENERATION_PROMPT

from ..backend_pool import AvailableSlotBackendPool
from ..schema import RubricNode
from ..telemetry import ModelCallMetrics
from ..version import (REFINE_FAILURE_ATTRIBUTION_PROMPT_VERSION,
                       REFINE_GENERATION_PROMPT_VERSION, REFINE_PARSER_VERSION)
from .refine import (RefineParseError, parse_refine_failure_attribution_response,
                     parse_refine_proposal_response)
from .refine_types import RefineEvaluation, RefineProposal
from .specialize_types import SpecializeManagerRequestSpec


class RefineManagerFailure(RuntimeError):
    def __init__(self, stage: str, raw_response: str | None, parse_error: str,
                 attempt_count: int, metrics: ModelCallMetrics) -> None:
        super().__init__(f"{stage} failed after {attempt_count} attempts: {parse_error}")
        self.stage = stage; self.raw_response = raw_response
        self.parse_error = parse_error; self.attempt_count = attempt_count
        self.metrics = metrics

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "raw_response": self.raw_response,
                "parse_error": self.parse_error, "attempt_count": self.attempt_count,
                "metrics": self.metrics.to_dict()}


class RefineManager:
    """Stateless Refine Manager calls using the shared available-slot pool."""

    def __init__(self, *, model: str, backend_pool: AvailableSlotBackendPool,
                 api_keys: str | list[str] | None = "EMPTY",
                 api_retry_attempts: int = 3, structured_max_retries: int = 1,
                 generation_request_kwargs: Mapping[str, Any] | None = None,
                 attribution_request_kwargs: Mapping[str, Any] | None = None,
                 image_field: str = "image_path",
                 rubric_memory_mode: str = "global_rubric_v1") -> None:
        self.model = model; self.backend_pool = backend_pool; self.api_keys = api_keys
        self.api_retry_attempts = api_retry_attempts
        self.structured_max_retries = structured_max_retries
        self.generation_request_kwargs = dict(generation_request_kwargs or
                                              {"temperature": 0.7, "seed": 42})
        self.attribution_request_kwargs = dict(attribution_request_kwargs or
                                               {"temperature": 0.2, "seed": 42})
        self.image_field = image_field
        if rubric_memory_mode not in {"none", "global_rubric_v1"}:
            raise ValueError("rubric_memory_mode must be none or global_rubric_v1")
        self.rubric_memory_mode = rubric_memory_mode
        for label, request in (("generation", self.generation_request_kwargs),
                               ("attribution", self.attribution_request_kwargs)):
            temperature = request.get("temperature")
            if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                    or not 0 <= temperature <= 2):
                raise ValueError(f"Refine {label} temperature must be in [0, 2]")
            if "max_tokens" in request:
                raise ValueError("Refine Manager must not set client max_tokens")

    def _spec(self, prompt, request_kwargs, version):
        return SpecializeManagerRequestSpec.from_prompt(
            model=self.model, backend_id=self.backend_pool.backend_id, prompt=prompt,
            decoding_config=request_kwargs, prompt_version=version,
            parser_version=REFINE_PARSER_VERSION)

    def request_specs(self) -> dict[str, SpecializeManagerRequestSpec]:
        suffix = f"\n[rubric_memory_mode={self.rubric_memory_mode}]"
        return {
            "refine_generation": self._spec(REFINE_GENERATION_PROMPT + suffix,
                                              self.generation_request_kwargs,
                                              REFINE_GENERATION_PROMPT_VERSION),
            "refine_failure_attribution": self._spec(
                REFINE_FAILURE_ATTRIBUTION_PROMPT + suffix,
                self.attribution_request_kwargs,
                REFINE_FAILURE_ATTRIBUTION_PROMPT_VERSION),
        }

    def _call(self, content, *, request_type, request_key, structured_attempt,
              request_kwargs):
        return self.backend_pool.call(
            content, request_type=request_type, request_key=request_key,
            structured_attempt=structured_attempt,
            agent_args={"model": self.model, "api_keys": self.api_keys,
                        "request_kwargs": dict(request_kwargs),
                        "api_retry_attempts": self.api_retry_attempts})

    @staticmethod
    def _metrics(calls: Sequence[AgentCallMetrics]) -> ModelCallMetrics:
        if not calls or any(not isinstance(item, AgentCallMetrics) for item in calls):
            raise TypeError("Refine Manager backend must return AgentCallMetrics")
        return ModelCallMetrics.from_agent_calls(
            calls, logical_evaluations=1, parse_retries=max(0, len(calls) - 1))

    def _multimodal(self, prompt: str, rows: Sequence[Mapping[str, Any]]):
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for row in rows:
            image = row.get(self.image_field)
            if not isinstance(image, str) or not image.strip():
                raise ValueError("Refine representative sample is missing an image")
            path = Path(image); mime_type, _ = mimetypes.guess_type(str(path))
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "text", "text":
                            f"Image for sample_id={row.get('sample_id', '<unknown>')}"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{mime_type or 'image/jpeg'};base64,{encoded}"}})
        return content

    @staticmethod
    def _repair_prompt(base_prompt: str, raw_response: str | None,
                       parse_error: str, max_description_chars: int,
                       allowed_representative_ids: Sequence[str]) -> str:
        target = max(400, min(max_description_chars - 200, 1400))
        previous = json.dumps(raw_response, ensure_ascii=False)
        allowed_ids = json.dumps(list(allowed_representative_ids), ensure_ascii=False)
        return f"""{base_prompt}

## Structured retry correction
The previous response was rejected by the deterministic parser:
{parse_error}

Return a corrected replacement JSON object. Preserve the same criterion name,
the four required description sections, and the evidence-grounded semantic
content. Do not add fields. If the description was too long, compress repeated
conditions and examples; target at most {target} characters and never exceed
the hard limit of {max_description_chars} characters. Do not mechanically cut
off a section. For representative_sample_ids, copy one to six values only from
this exact allowlist (these are the multimodal representative samples):
{allowed_ids}
Do not select other sample IDs mentioned elsewhere in the textual evidence.

Previous rejected response, encoded as a JSON string:
{previous}
"""

    def generate(self, *, node: RubricNode, evidence: Mapping[str, Any],
                 representative_rows: Sequence[Mapping[str, Any]],
                 prior_failures: Sequence[Mapping[str, Any]],
                 rubric_memory: Mapping[str, Any] | None,
                 max_description_chars: int) -> RefineProposal:
        if self.rubric_memory_mode == "global_rubric_v1" and rubric_memory is None:
            raise ValueError("global_rubric_v1 Refine requires rubric_memory")
        allowed = evidence.get("representative_sample_ids")
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("Refine evidence must contain representative_sample_ids")
        prompt = REFINE_GENERATION_PROMPT.format(
            node_id=node.node_id, criterion_name=node.criterion.name,
            criterion_description=node.criterion.description,
            evidence_json=json.dumps(dict(evidence), indent=2, ensure_ascii=False),
            refine_failure_history_json=json.dumps(list(prior_failures), indent=2,
                                                   ensure_ascii=False),
            rubric_memory_json=json.dumps(rubric_memory, indent=2, ensure_ascii=False),
            max_description_chars=max_description_chars)
        spec = self.request_specs()["refine_generation"]
        calls = []; last_raw = None; last_error = "invalid Refine proposal"
        total = self.structured_max_retries + 1
        for attempt in range(1, total + 1):
            attempt_prompt = (prompt if attempt == 1 else self._repair_prompt(
                prompt, last_raw, last_error, max_description_chars, allowed))
            raw, metrics = self._call(
                self._multimodal(attempt_prompt, representative_rows),
                request_type="refine_generation", request_key=node.node_id,
                structured_attempt=attempt,
                request_kwargs=self.generation_request_kwargs)
            calls.append(metrics); last_raw = raw if isinstance(raw, str) else None
            try:
                return parse_refine_proposal_response(
                    raw, node=node, allowed_representative_ids=allowed,
                    max_description_chars=max_description_chars,
                    attempt_count=attempt, metrics=self._metrics(calls), request_spec=spec)
            except RefineParseError as exc:
                last_error = str(exc)
        raise RefineManagerFailure("refine_generation", last_raw, last_error, total,
                                   self._metrics(calls))

    def attribute_failure(self, *, original_node: RubricNode,
                          proposal: RefineProposal, evaluation: RefineEvaluation,
                          changed_predictions: Sequence[Mapping[str, Any]],
                          rubric_memory: Mapping[str, Any] | None) -> dict[str, Any]:
        if self.rubric_memory_mode == "global_rubric_v1" and rubric_memory is None:
            raise ValueError("global_rubric_v1 Refine attribution requires rubric_memory")
        prompt = REFINE_FAILURE_ATTRIBUTION_PROMPT.format(
            original_criterion_json=json.dumps(original_node.to_dict(), indent=2,
                                               ensure_ascii=False),
            proposal_json=json.dumps(proposal.to_dict(), indent=2, ensure_ascii=False),
            evaluation_json=json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False),
            changed_predictions_json=json.dumps(list(changed_predictions), indent=2,
                                                ensure_ascii=False),
            rubric_memory_json=json.dumps(rubric_memory, indent=2, ensure_ascii=False))
        spec = self.request_specs()["refine_failure_attribution"]
        calls = []; last_raw = None; last_error = "invalid Refine failure attribution"
        total = self.structured_max_retries + 1
        for attempt in range(1, total + 1):
            raw, metrics = self._call(
                prompt, request_type="refine_failure_attribution",
                request_key=original_node.node_id, structured_attempt=attempt,
                request_kwargs=self.attribution_request_kwargs)
            calls.append(metrics); last_raw = raw if isinstance(raw, str) else None
            try:
                attribution = parse_refine_failure_attribution_response(raw)
                return {"schema_version": "1.0.0", "attribution": attribution,
                        "raw_response": last_raw, "attempt_count": attempt,
                        "metrics": self._metrics(calls).to_dict(),
                        "request_spec": spec.to_dict()}
            except RefineParseError as exc:
                last_error = str(exc)
        raise RefineManagerFailure("refine_failure_attribution", last_raw, last_error,
                                   total, self._metrics(calls))

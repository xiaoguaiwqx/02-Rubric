"""Model-backed Manager stages for Specialize, isolated behind strict parsers."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.agent import AgentCallMetrics
from critiq.specialize_prompts import (
    CHILD_GENERATION_PROMPT,
    ERROR_SIGNATURE_PROMPT,
    SEMANTIC_CLUSTER_PROMPT,
)

from ..backend_pool import AvailableSlotBackendPool
from ..schema import RubricNode
from ..telemetry import ModelCallMetrics
from ..version import (
    CHILD_GENERATION_PROMPT_VERSION,
    ERROR_SIGNATURE_PROMPT_VERSION,
    SEMANTIC_CLUSTER_PROMPT_VERSION,
    SEMANTIC_CLUSTER_PARSER_VERSION,
    SPECIALIZE_PARSER_VERSION,
)
from .specialize import (
    SpecializeParseError,
    parse_child_proposal_response,
    parse_cluster_proposal_response,
    parse_error_signature_response,
)
from .specialize_types import (
    ChildCriterionProposal,
    ClusterProposal,
    ErrorSignature,
    ErrorSignatureOutput,
    SemanticCluster,
    SpecializeManagerRequestSpec,
)
from .types import ErrorSampleRef


class SpecializeManagerFailure(RuntimeError):
    def __init__(self, stage: str, raw_response: str | None, parse_error: str,
                 attempt_count: int, metrics: ModelCallMetrics) -> None:
        super().__init__(f"{stage} failed after {attempt_count} attempts: {parse_error}")
        self.stage = stage
        self.raw_response = raw_response
        self.parse_error = parse_error
        self.attempt_count = attempt_count
        self.metrics = metrics

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "raw_response": self.raw_response,
                "parse_error": self.parse_error, "attempt_count": self.attempt_count,
                "metrics": self.metrics.to_dict()}


class SpecializeManager:
    """Stateless Manager calls using the shared available-slot backend pool."""

    def __init__(self, *, model: str, backend_pool: AvailableSlotBackendPool,
                 api_keys: str | list[str] | None = "EMPTY",
                 api_retry_attempts: int = 3, structured_max_retries: int = 1,
                 analysis_request_kwargs: Mapping[str, Any] | None = None,
                 clustering_request_kwargs: Mapping[str, Any] | None = None,
                 generation_request_kwargs: Mapping[str, Any] | None = None,
                 child_input_mode: str = "multimodal",
                 image_field: str = "image_path") -> None:
        self.model = model
        self.backend_pool = backend_pool
        self.api_keys = api_keys
        self.api_retry_attempts = api_retry_attempts
        self.structured_max_retries = structured_max_retries
        self.analysis_request_kwargs = dict(analysis_request_kwargs or {"temperature": 0, "seed": 42})
        self.clustering_request_kwargs = dict(
            clustering_request_kwargs or {"temperature": 0, "seed": 42})
        self.generation_request_kwargs = dict(generation_request_kwargs or {"temperature": 0.7, "seed": 42})
        if child_input_mode not in {"multimodal", "text"}:
            raise ValueError("child_input_mode must be multimodal or text")
        self.child_input_mode = child_input_mode
        self.image_field = image_field
        for stage, request in (
                ("error-signature", self.analysis_request_kwargs),
                ("semantic-clustering", self.clustering_request_kwargs),
                ("child-generation", self.generation_request_kwargs)):
            temperature = request.get("temperature")
            if (isinstance(temperature, bool)
                    or not isinstance(temperature, (int, float))
                    or not 0 <= temperature <= 2):
                raise ValueError(f"{stage} temperature must be a number in [0, 2]")
        if ("max_tokens" in self.analysis_request_kwargs
                or "max_tokens" in self.clustering_request_kwargs
                or "max_tokens" in self.generation_request_kwargs):
            raise ValueError("Specialize Manager must not set client max_tokens")

    def _spec(self, prompt: str, request_kwargs: Mapping[str, Any], version: str,
              *, parser_version: str = SPECIALIZE_PARSER_VERSION):
        return SpecializeManagerRequestSpec.from_prompt(
            model=self.model, backend_id=self.backend_pool.backend_id, prompt=prompt,
            decoding_config=request_kwargs, prompt_version=version,
            parser_version=parser_version)

    def request_specs(self) -> dict[str, SpecializeManagerRequestSpec]:
        return {
            "error_signature": self._spec(ERROR_SIGNATURE_PROMPT,
                                           self.analysis_request_kwargs,
                                           ERROR_SIGNATURE_PROMPT_VERSION),
            "semantic_cluster": self._spec(
                                            SEMANTIC_CLUSTER_PROMPT,
                                            self.clustering_request_kwargs,
                                            SEMANTIC_CLUSTER_PROMPT_VERSION,
                                            parser_version=SEMANTIC_CLUSTER_PARSER_VERSION),
            "child_generation": self._spec(
                                            CHILD_GENERATION_PROMPT
                                            + f"\n[child_input_mode={self.child_input_mode}]",
                                            self.generation_request_kwargs,
                                            CHILD_GENERATION_PROMPT_VERSION),
        }

    def _call(self, content: object, *, request_type: str, request_key: str,
              structured_attempt: int, request_kwargs: Mapping[str, Any]):
        return self.backend_pool.call(
            content, request_type=request_type, request_key=request_key,
            structured_attempt=structured_attempt,
            agent_args={"model": self.model, "api_keys": self.api_keys,
                        "request_kwargs": dict(request_kwargs),
                        "api_retry_attempts": self.api_retry_attempts})

    @staticmethod
    def _metrics(calls: Sequence[AgentCallMetrics]) -> ModelCallMetrics:
        """Convert Agent-level attempts into one logical Manager evaluation."""

        if not calls or any(not isinstance(item, AgentCallMetrics) for item in calls):
            raise TypeError("Specialize Manager backend must return AgentCallMetrics")
        return ModelCallMetrics.from_agent_calls(
            calls,
            logical_evaluations=1,
            parse_retries=max(0, len(calls) - 1),
        )

    def _multimodal(self, prompt: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for row in rows:
            image = row.get(self.image_field)
            if not isinstance(image, str) or not image.strip():
                raise ValueError("Manager sample is missing an image")
            path = Path(image)
            mime_type, _ = mimetypes.guess_type(str(path))
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "text", "text":
                            f"Image for sample_id={row.get('sample_id', '<unknown>')}"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{mime_type or 'image/jpeg'};base64,{encoded}"}})
        return content

    def infer_signature(self, row: Mapping[str, Any], parent: RubricNode,
                        error: ErrorSampleRef) -> ErrorSignatureOutput:
        if row.get("sample_id") != error.sample_id or error.outcome != "wrong":
            raise ValueError("signature input must match a decisive-wrong reference")
        prompt = ERROR_SIGNATURE_PROMPT.format(
            criterion_name=parent.criterion.name,
            criterion_description=parent.criterion.description,
            sample_id=error.sample_id, question=row["question"], A=row["A"], B=row["B"],
            gold=error.gold, parent_vote=error.vote.value,
            thought=error.thought or "No thought was returned.")
        spec = self.request_specs()["error_signature"]
        calls = []
        last_raw = None
        last_error = "Manager did not return a valid error signature"
        total = self.structured_max_retries + 1
        for attempt in range(1, total + 1):
            raw, metrics = self._call(self._multimodal(prompt, (row,)),
                request_type="specialize_error_signature", request_key=error.sample_id,
                structured_attempt=attempt, request_kwargs=self.analysis_request_kwargs)
            calls.append(metrics); last_raw = raw if isinstance(raw, str) else None
            try:
                signature = parse_error_signature_response(raw, expected_sample_id=error.sample_id)
                return ErrorSignatureOutput(signature, last_raw, None, attempt,
                    self._metrics(calls), spec)
            except SpecializeParseError as exc:
                last_error = str(exc)
        return ErrorSignatureOutput(None, last_raw, last_error, total,
                                    self._metrics(calls), spec)

    def cluster(self, signatures: Sequence[ErrorSignature], *, criterion_name: str,
                min_cluster_size: int, max_clusters: int,
                prior_failures: Sequence[Mapping[str, Any]] = ()) -> ClusterProposal:
        if not signatures:
            raise ValueError("signatures must not be empty")
        prompt = SEMANTIC_CLUSTER_PROMPT.format(
            criterion_name=criterion_name, min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
            signatures_json=json.dumps([item.to_dict() for item in signatures],
                                       indent=2, ensure_ascii=False),
            split_failure_history_json=json.dumps(list(prior_failures), indent=2,
                                                  ensure_ascii=False))
        spec = self.request_specs()["semantic_cluster"]
        calls = []; last_raw = None; last_error = "invalid cluster proposal"
        total = self.structured_max_retries + 1
        for attempt in range(1, total + 1):
            raw, metrics = self._call(prompt, request_type="specialize_cluster",
                request_key=criterion_name, structured_attempt=attempt,
                request_kwargs=self.clustering_request_kwargs)
            calls.append(metrics); last_raw = raw if isinstance(raw, str) else None
            combined = self._metrics(calls)
            try:
                return parse_cluster_proposal_response(
                    raw, expected_sample_ids=tuple(item.sample_id for item in signatures),
                    min_cluster_size=min_cluster_size, max_clusters=max_clusters,
                    attempt_count=attempt, metrics=combined, request_spec=spec)
            except SpecializeParseError as exc:
                last_error = str(exc)
        raise SpecializeManagerFailure("semantic_cluster", last_raw, last_error, total,
                                       self._metrics(calls))

    def generate_child(self, *, parent: RubricNode, cluster: SemanticCluster,
                       signatures: Sequence[ErrorSignature], representative_rows: Sequence[Mapping[str, Any]],
                       siblings: Sequence[ChildCriterionProposal],
                       prior_failures: Sequence[Mapping[str, Any]] = ()) -> ChildCriterionProposal:
        representative_ids = tuple(str(row["sample_id"]) for row in representative_rows)
        representative_samples = [
            {"sample_id": str(row["sample_id"]), "question": row["question"],
             "A": row["A"], "B": row["B"], "gold": row.get("answer")}
            for row in representative_rows
        ]
        prompt = CHILD_GENERATION_PROMPT.format(
            criterion_name=parent.criterion.name,
            criterion_description=parent.criterion.description,
            cluster_json=json.dumps(cluster.to_dict(), indent=2, ensure_ascii=False),
            signatures_json=json.dumps([item.to_dict() for item in signatures], indent=2,
                                       ensure_ascii=False),
            siblings_json=json.dumps([
                {"criterion_name": item.criterion_name, "description": item.description}
                for item in siblings], indent=2, ensure_ascii=False),
            split_failure_history_json=json.dumps(list(prior_failures), indent=2,
                                                  ensure_ascii=False),
            representative_sample_ids=json.dumps(representative_ids, ensure_ascii=False),
            representative_samples_json=json.dumps(
                representative_samples, indent=2, ensure_ascii=False))
        spec = self.request_specs()["child_generation"]
        calls = []; last_raw = None; last_error = "invalid child proposal"
        total = self.structured_max_retries + 1
        for attempt in range(1, total + 1):
            content = (self._multimodal(prompt, representative_rows)
                       if self.child_input_mode == "multimodal" else prompt)
            raw, metrics = self._call(content,
                request_type="specialize_child", request_key=f"{parent.node_id}::{cluster.cluster_id}",
                structured_attempt=attempt, request_kwargs=self.generation_request_kwargs)
            calls.append(metrics); last_raw = raw if isinstance(raw, str) else None
            combined = self._metrics(calls)
            try:
                return parse_child_proposal_response(raw, cluster=cluster,
                    attempt_count=attempt, metrics=combined, request_spec=spec)
            except SpecializeParseError as exc:
                last_error = str(exc)
        raise SpecializeManagerFailure("child_generation", last_raw, last_error, total,
                                       self._metrics(calls))

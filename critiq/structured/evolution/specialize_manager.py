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
    CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC,
    CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC_LOCKED_RETRY_V3,
    CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC_RETRY_V2,
    ERROR_SIGNATURE_PROMPT,
    ERROR_SIGNATURE_PROMPT_COMPACT_IDS,
    SEMANTIC_CLUSTER_PROMPT,
    SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS,
    SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS_GLOBAL_RUBRIC,
    SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS_GLOBAL_RUBRIC_RETRY_V2,
    SEMANTIC_CLUSTER_PROMPT_GLOBAL_RUBRIC,
    SEMANTIC_CLUSTER_PROMPT_GLOBAL_RUBRIC_RETRY_V2,
    SPLIT_FAILURE_ATTRIBUTION_PROMPT,
    SPLIT_FAILURE_ATTRIBUTION_PROMPT_LOCKED_RETRY_V3,
    SPLIT_FAILURE_ATTRIBUTION_PROMPT_RETRY_V2,
)

from ..backend_pool import AvailableSlotBackendPool
from ..schema import RubricNode
from ..telemetry import ModelCallMetrics
from ..version import (
    CHILD_GENERATION_GLOBAL_RUBRIC_PROMPT_VERSION,
    CHILD_GENERATION_PROMPT_VERSION,
    ERROR_SIGNATURE_COMPACT_IDS_PROMPT_VERSION,
    ERROR_SIGNATURE_PROMPT_VERSION,
    SEMANTIC_CLUSTER_COMPACT_IDS_GLOBAL_RUBRIC_PROMPT_VERSION,
    SEMANTIC_CLUSTER_COMPACT_IDS_PROMPT_VERSION,
    SEMANTIC_CLUSTER_GLOBAL_RUBRIC_PROMPT_VERSION,
    SEMANTIC_CLUSTER_PROMPT_VERSION,
    SEMANTIC_CLUSTER_PARSER_VERSION,
    SPLIT_FAILURE_ATTRIBUTION_PROMPT_VERSION,
    SPECIALIZE_PARSER_VERSION,
)
from .specialize import (
    SpecializeParseError,
    parse_child_proposal_response,
    parse_cluster_proposal_response,
    parse_error_signature_response,
    parse_split_failure_attribution_response,
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
                 image_field: str = "image_path",
                 rubric_memory_mode: str = "none",
                 retry_feedback_mode: str = "aggregate_v1",
                 compact_sample_ids: bool = False) -> None:
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
        if not isinstance(compact_sample_ids, bool):
            raise TypeError("compact_sample_ids must be bool")
        self.compact_sample_ids = compact_sample_ids
        if rubric_memory_mode not in {"none", "global_rubric_v1"}:
            raise ValueError("rubric_memory_mode must be none or global_rubric_v1")
        self.rubric_memory_mode = rubric_memory_mode
        if retry_feedback_mode not in {"aggregate_v1", "child_diagnostic_v2",
                                       "locked_sample_v3"}:
            raise ValueError(
                "retry_feedback_mode must be aggregate_v1, child_diagnostic_v2, "
                "or locked_sample_v3")
        if (retry_feedback_mode in {"child_diagnostic_v2", "locked_sample_v3"}
                and rubric_memory_mode != "global_rubric_v1"):
            raise ValueError(f"{retry_feedback_mode} requires global_rubric_v1")
        self.retry_feedback_mode = retry_feedback_mode
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
        retry_v2 = self.retry_feedback_mode == "child_diagnostic_v2"
        retry_v3 = self.retry_feedback_mode == "locked_sample_v3"
        signature_prompt = (ERROR_SIGNATURE_PROMPT_COMPACT_IDS
                            if self.compact_sample_ids
                            else ERROR_SIGNATURE_PROMPT)
        signature_version = (ERROR_SIGNATURE_COMPACT_IDS_PROMPT_VERSION
                             if self.compact_sample_ids
                             else ERROR_SIGNATURE_PROMPT_VERSION)
        cluster_prompt = (
                          SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS_GLOBAL_RUBRIC_RETRY_V2
                          if self.compact_sample_ids and retry_v2 else
                          SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS_GLOBAL_RUBRIC
                          if self.compact_sample_ids and self.rubric_memory_mode == "global_rubric_v1" else
                          SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS
                          if self.compact_sample_ids else
                          SEMANTIC_CLUSTER_PROMPT_GLOBAL_RUBRIC_RETRY_V2
                          if retry_v2 else SEMANTIC_CLUSTER_PROMPT_GLOBAL_RUBRIC
                          if self.rubric_memory_mode == "global_rubric_v1"
                          else SEMANTIC_CLUSTER_PROMPT)
        cluster_version = ("semantic-cluster-compact-ids-global-rubric-retry-v2"
                           if self.compact_sample_ids and retry_v2 else
                           SEMANTIC_CLUSTER_COMPACT_IDS_GLOBAL_RUBRIC_PROMPT_VERSION
                           if self.compact_sample_ids and self.rubric_memory_mode == "global_rubric_v1" else
                           SEMANTIC_CLUSTER_COMPACT_IDS_PROMPT_VERSION
                           if self.compact_sample_ids else
                           "semantic-cluster-global-rubric-retry-v2"
                           if retry_v2 else SEMANTIC_CLUSTER_GLOBAL_RUBRIC_PROMPT_VERSION
                           if self.rubric_memory_mode == "global_rubric_v1"
                           else SEMANTIC_CLUSTER_PROMPT_VERSION)
        child_prompt = (CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC_LOCKED_RETRY_V3
                        if retry_v3 else CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC_RETRY_V2
                        if retry_v2 else CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC
                        if self.rubric_memory_mode == "global_rubric_v1"
                        else CHILD_GENERATION_PROMPT)
        child_version = ("child-generation-global-rubric-locked-retry-v3"
                         if retry_v3 else "child-generation-global-rubric-retry-v2"
                         if retry_v2 else CHILD_GENERATION_GLOBAL_RUBRIC_PROMPT_VERSION
                         if self.rubric_memory_mode == "global_rubric_v1"
                         else CHILD_GENERATION_PROMPT_VERSION)
        attribution_prompt = (SPLIT_FAILURE_ATTRIBUTION_PROMPT_LOCKED_RETRY_V3
                              if retry_v3 else SPLIT_FAILURE_ATTRIBUTION_PROMPT_RETRY_V2
                              if retry_v2 else SPLIT_FAILURE_ATTRIBUTION_PROMPT)
        attribution_version = ("split-failure-attribution-locked-retry-v3"
                               if retry_v3 else "split-failure-attribution-retry-v2"
                               if retry_v2 else SPLIT_FAILURE_ATTRIBUTION_PROMPT_VERSION)
        return {
            "error_signature": self._spec(signature_prompt,
                                           self.analysis_request_kwargs,
                                           signature_version),
            "semantic_cluster": self._spec(
                                            cluster_prompt,
                                            self.clustering_request_kwargs,
                                            cluster_version,
                                            parser_version=SEMANTIC_CLUSTER_PARSER_VERSION),
            "child_generation": self._spec(
                                            child_prompt
                                            + f"\n[child_input_mode={self.child_input_mode}]",
                                            self.generation_request_kwargs,
                                            child_version),
            "split_failure_attribution": self._spec(
                                            attribution_prompt,
                                            self.clustering_request_kwargs,
                                            attribution_version),
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
        prompt_template = (ERROR_SIGNATURE_PROMPT_COMPACT_IDS
                           if self.compact_sample_ids
                           else ERROR_SIGNATURE_PROMPT)
        prompt = prompt_template.format(
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
                signature = parse_error_signature_response(
                    raw, expected_sample_id=error.sample_id,
                    compact_ids=self.compact_sample_ids)
                return ErrorSignatureOutput(signature, last_raw, None, attempt,
                    self._metrics(calls), spec)
            except SpecializeParseError as exc:
                last_error = str(exc)
        return ErrorSignatureOutput(None, last_raw, last_error, total,
                                    self._metrics(calls), spec)

    def cluster(self, signatures: Sequence[ErrorSignature], *, criterion_name: str,
                min_cluster_size: int, max_clusters: int,
                prior_failures: Sequence[Mapping[str, Any]] = (),
                rubric_memory: Mapping[str, Any] | None = None,
                retry_feedback: Mapping[str, Any] | None = None) -> ClusterProposal:
        if not signatures:
            raise ValueError("signatures must not be empty")
        if self.rubric_memory_mode == "global_rubric_v1" and rubric_memory is None:
            raise ValueError("global_rubric_v1 clustering requires rubric_memory")
        if self.retry_feedback_mode in {"child_diagnostic_v2", "locked_sample_v3"} and retry_feedback is None:
            raise ValueError(f"{self.retry_feedback_mode} clustering requires retry_feedback")
        prompt_template = (
                           SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS_GLOBAL_RUBRIC_RETRY_V2
                           if self.compact_sample_ids and self.retry_feedback_mode == "child_diagnostic_v2"
                           else SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS_GLOBAL_RUBRIC
                           if self.compact_sample_ids and self.rubric_memory_mode == "global_rubric_v1"
                           else SEMANTIC_CLUSTER_PROMPT_COMPACT_IDS
                           if self.compact_sample_ids
                           else SEMANTIC_CLUSTER_PROMPT_GLOBAL_RUBRIC_RETRY_V2
                           if self.retry_feedback_mode == "child_diagnostic_v2"
                           else SEMANTIC_CLUSTER_PROMPT_GLOBAL_RUBRIC
                           if self.rubric_memory_mode == "global_rubric_v1"
                           else SEMANTIC_CLUSTER_PROMPT)
        if self.compact_sample_ids:
            key_to_sample_id = {
                f"S{index:03d}": item.sample_id
                for index, item in enumerate(signatures, start=1)
            }
            signature_payload = []
            for signature_key, item in zip(key_to_sample_id, signatures):
                value = item.to_dict()
                value.pop("sample_id")
                signature_payload.append({"signature_key": signature_key, **value})
        else:
            key_to_sample_id = {item.sample_id: item.sample_id
                                for item in signatures}
            signature_payload = [item.to_dict() for item in signatures]
        prompt = prompt_template.format(
            criterion_name=criterion_name, min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
            signatures_json=json.dumps(signature_payload, indent=2,
                                       ensure_ascii=False),
            split_failure_history_json=json.dumps(list(prior_failures), indent=2,
                                                  ensure_ascii=False),
            rubric_memory_json=json.dumps(rubric_memory, indent=2,
                                          ensure_ascii=False),
            retry_feedback_json=json.dumps(retry_feedback or {}, indent=2,
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
                proposal = parse_cluster_proposal_response(
                    raw, expected_sample_ids=tuple(key_to_sample_id),
                    min_cluster_size=min_cluster_size, max_clusters=max_clusters,
                    attempt_count=attempt, metrics=combined, request_spec=spec)
                if not self.compact_sample_ids:
                    return proposal
                return ClusterProposal(
                    tuple(SemanticCluster(
                        cluster.cluster_id, cluster.label,
                        cluster.shared_failure, cluster.distinction,
                        tuple(key_to_sample_id[key] for key in cluster.sample_ids),
                    ) for cluster in proposal.clusters),
                    tuple(key_to_sample_id[key]
                          for key in proposal.unclustered_sample_ids),
                    proposal.raw_response, proposal.attempt_count,
                    proposal.metrics, proposal.request_spec,
                )
            except SpecializeParseError as exc:
                last_error = str(exc)
        raise SpecializeManagerFailure("semantic_cluster", last_raw, last_error, total,
                                       self._metrics(calls))

    def generate_child(self, *, parent: RubricNode, cluster: SemanticCluster,
                       signatures: Sequence[ErrorSignature], representative_rows: Sequence[Mapping[str, Any]],
                       siblings: Sequence[ChildCriterionProposal],
                       prior_failures: Sequence[Mapping[str, Any]] = (),
                       rubric_memory: Mapping[str, Any] | None = None,
                       retry_feedback: Mapping[str, Any] | None = None,
                       repair_context: Mapping[str, Any] | None = None,
                       supplemental_rows: Sequence[Mapping[str, Any]] = ()) -> ChildCriterionProposal:
        if self.rubric_memory_mode == "global_rubric_v1" and rubric_memory is None:
            raise ValueError("global_rubric_v1 child generation requires rubric_memory")
        representative_ids = tuple(str(row["sample_id"]) for row in representative_rows)
        representative_samples = [
            {"sample_id": str(row["sample_id"]), "question": row["question"],
             "A": row["A"], "B": row["B"], "gold": row.get("answer")}
            for row in representative_rows
        ]
        if self.retry_feedback_mode in {"child_diagnostic_v2", "locked_sample_v3"} and retry_feedback is None:
            raise ValueError(f"{self.retry_feedback_mode} child generation requires retry_feedback")
        if self.retry_feedback_mode == "locked_sample_v3" and repair_context is None:
            raise ValueError("locked_sample_v3 child generation requires repair_context")
        prompt_template = (CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC_LOCKED_RETRY_V3
                           if self.retry_feedback_mode == "locked_sample_v3"
                           else CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC_RETRY_V2
                           if self.retry_feedback_mode == "child_diagnostic_v2"
                           else CHILD_GENERATION_PROMPT_GLOBAL_RUBRIC
                           if self.rubric_memory_mode == "global_rubric_v1"
                           else CHILD_GENERATION_PROMPT)
        prompt = prompt_template.format(
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
                representative_samples, indent=2, ensure_ascii=False),
            rubric_memory_json=json.dumps(rubric_memory, indent=2,
                                          ensure_ascii=False),
            retry_feedback_json=json.dumps(retry_feedback or {}, indent=2,
                                           ensure_ascii=False),
            repair_context_json=json.dumps(repair_context or {}, indent=2,
                                           ensure_ascii=False))
        spec = self.request_specs()["child_generation"]
        calls = []; last_raw = None; last_error = "invalid child proposal"
        total = self.structured_max_retries + 1
        for attempt in range(1, total + 1):
            visual_rows = tuple(representative_rows) + tuple(
                row for row in supplemental_rows
                if str(row.get("sample_id")) not in set(representative_ids))
            content = (self._multimodal(prompt, visual_rows)
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

    def attribute_split_failure(
        self,
        *,
        parent: RubricNode,
        signatures: Sequence[ErrorSignature],
        cluster_proposal: Mapping[str, Any],
        children: Sequence[ChildCriterionProposal],
        local_metrics: Mapping[str, Any],
        changed_predictions: Sequence[Mapping[str, Any]],
        retry_feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured natural-language diagnosis for a rejected Split."""

        if self.retry_feedback_mode in {"child_diagnostic_v2", "locked_sample_v3"} and retry_feedback is None:
            raise ValueError(f"{self.retry_feedback_mode} attribution requires retry_feedback")
        prompt_template = (SPLIT_FAILURE_ATTRIBUTION_PROMPT_LOCKED_RETRY_V3
                           if self.retry_feedback_mode == "locked_sample_v3"
                           else SPLIT_FAILURE_ATTRIBUTION_PROMPT_RETRY_V2
                           if self.retry_feedback_mode == "child_diagnostic_v2"
                           else SPLIT_FAILURE_ATTRIBUTION_PROMPT)
        prompt = prompt_template.format(
            criterion_name=parent.criterion.name,
            criterion_description=parent.criterion.description,
            signatures_json=json.dumps([item.to_dict() for item in signatures], indent=2,
                                       ensure_ascii=False),
            cluster_json=json.dumps(cluster_proposal, indent=2, ensure_ascii=False),
            children_json=json.dumps([
                {"cluster_id": item.cluster_id,
                 "criterion_name": item.criterion_name,
                 "description": item.description,
                 "rationale": item.rationale}
                for item in children], indent=2, ensure_ascii=False),
            metrics_json=json.dumps(dict(local_metrics), indent=2, ensure_ascii=False),
            changed_predictions_json=json.dumps(list(changed_predictions), indent=2,
                                                ensure_ascii=False),
            retry_feedback_json=json.dumps(retry_feedback or {}, indent=2,
                                           ensure_ascii=False),
            repair_context_json=json.dumps({}, indent=2, ensure_ascii=False))
        spec = self.request_specs()["split_failure_attribution"]
        calls = []
        last_raw = None
        last_error = "invalid split failure attribution"
        total = self.structured_max_retries + 1
        for attempt in range(1, total + 1):
            raw, metrics = self._call(
                prompt, request_type="split_failure_attribution",
                request_key=parent.node_id, structured_attempt=attempt,
                request_kwargs=self.clustering_request_kwargs)
            calls.append(metrics)
            last_raw = raw if isinstance(raw, str) else None
            try:
                attribution = parse_split_failure_attribution_response(raw)
                return {
                    "schema_version": "1.0.0",
                    "attribution": attribution,
                    "raw_response": last_raw,
                    "attempt_count": attempt,
                    "metrics": self._metrics(calls).to_dict(),
                    "request_spec": spec.to_dict(),
                }
            except SpecializeParseError as exc:
                last_error = str(exc)
        raise SpecializeManagerFailure(
            "split_failure_attribution", last_raw, last_error, total,
            self._metrics(calls))

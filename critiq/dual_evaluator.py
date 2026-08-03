"""Multimodal evaluators for the decoupled vote and gate channels."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Sequence

from .agent import Agent, AgentCallMetrics
from .dual_worker_prompts import (
    GATE_STATE_WORKER_PROMPT,
    GATE_STATE_WORKER_PROMPT_V2,
    GATE_STATE_WORKER_PROMPT_V2_1,
    GATE_STATE_WORKER_PROMPT_POSTFIX,
    PAIRWISE_MULTIMODAL_WORKER_PROMPT,
    PAIRWISE_WORKER_PROMPT_POSTFIX,
)
from .evaluator import MultiModalPairEvaluator
from .structured.dual_worker import (
    DualWorkerRequestSpec,
    GateJudgement,
    GatePredictionOutput,
    GateStateOutput,
    GateStateParseError,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    PairwiseVoteParseError,
    make_gate_parse_failure,
    parse_gate_state_response,
    parse_pairwise_vote_response,
    worker_prompt_sha256,
)
from .structured.version import (
    GATE_WORKER_PROMPT_V2_1_PILOT_VERSION,
    GATE_WORKER_PROMPT_V2_PILOT_VERSION,
    GATE_WORKER_PROMPT_VERSION,
)
from .structured.aggregation import aggregate_flat_votes
from .structured.judgement import FinalPreference, Vote
from .structured.telemetry import (
    ModelCallMetrics,
    TokenPricing,
    combine_model_call_metrics,
)
from .structured.worker_output import StructuredCriterionSnapshot, structured_input_fingerprint
from .utils import Criterion, PairData


class _DualMultimodalEvaluator(MultiModalPairEvaluator):
    def __init__(self, *, worker_args: dict[str, Any], dataset: Sequence[PairData],
                 backend_id: str, worker_prompt: str, worker_postfix: str,
                 max_concurrent: int = 1, max_retries: int = 1,
                 max_data_chars: int | None = None, image_field: str = "image_path",
                 question_field: str = "question", sample_id_field: str = "sample_id",
                 encode_local_image: bool = True, pricing: TokenPricing | None = None,
                 call_backend: Any | None = None) -> None:
        if not dataset:
            raise ValueError("dual worker dataset must not be empty")
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("backend_id must be non-empty")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        sample_ids: list[str] = []
        for index, row in enumerate(dataset):
            sample_id = row.get(sample_id_field)
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"missing sample ID at row {index + 1}")
            if not isinstance(row.get("A"), str) or not isinstance(row.get("B"), str):
                raise ValueError(f"A/B must be strings at row {index + 1}")
            sample_ids.append(sample_id)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique")
        self.backend_id = backend_id
        self.sample_id_field = sample_id_field
        self.pricing = pricing
        self.call_backend = call_backend
        self.worker_prompt_postfix = worker_postfix
        super().__init__(worker_args=worker_args, dataset=dataset, max_concurrent=max_concurrent,
                         max_retries=max_retries, worker_prompt=worker_prompt,
                         max_data_chars=max_data_chars, image_field=image_field,
                         question_field=question_field, encode_local_image=encode_local_image)

    @staticmethod
    def _agent_metrics(worker: Agent) -> AgentCallMetrics:
        value = worker.last_call_metrics
        return value if isinstance(value, AgentCallMetrics) else AgentCallMetrics(
            api_attempts=1, input_tokens=None, output_tokens=None, total_tokens=None,
            usage_complete=False)

    def sample_fingerprint(self, data: PairData) -> str:
        return structured_input_fingerprint(data, image_field=self.image_field,
            question_field=self.question_field, sample_id_field=self.sample_id_field,
            max_data_chars=self.max_data_chars, encode_local_image=self.encode_local_image)

    def request_spec(self) -> DualWorkerRequestSpec:
        decoding = self.worker_args.get("request_kwargs") or {}
        decoding = json.loads(json.dumps(decoding, ensure_ascii=False, allow_nan=False))
        return DualWorkerRequestSpec(
            model=str(self.worker_args.get("model", "gpt-4o-mini")), backend_id=self.backend_id,
            prompt_sha256=worker_prompt_sha256(self.worker_prompt, self.worker_prompt_postfix),
            max_data_chars=self.max_data_chars, encode_local_image=self.encode_local_image,
            image_field=self.image_field, question_field=self.question_field,
            sample_id_field=self.sample_id_field, decoding_config=decoding)

    @staticmethod
    def _criteria(criteria: Sequence[Criterion]) -> tuple[Criterion, ...]:
        result = tuple(criteria)
        if not result or any(not isinstance(item, Criterion) for item in result):
            raise TypeError("criteria must be a non-empty Criterion sequence")
        names = [item.name for item in result]
        if len(set(names)) != len(names):
            raise ValueError("criterion names must be unique")
        return result

    def _call_model(self, content: object, *, request_type: str, request_key: str,
                    structured_attempt: int) -> tuple[object, AgentCallMetrics]:
        if self.call_backend is not None:
            return self.call_backend.call(content, request_type=request_type,
                request_key=request_key, structured_attempt=structured_attempt,
                agent_args=self.worker_args)
        worker = Agent(**self.worker_args)
        raw = worker(content, stream=False)
        return raw, self._agent_metrics(worker)


class PairwiseVoteMultiModalEvaluator(_DualMultimodalEvaluator):
    """Exact exp4 request semantics with auditable output and telemetry."""

    def __init__(self, *, worker_args: dict[str, Any], dataset: Sequence[PairData], backend_id: str,
                 max_concurrent: int = 1, max_retries: int = 1, worker_prompt: str | None = None,
                 **kwargs: Any) -> None:
        if worker_prompt is not None and worker_prompt != PAIRWISE_MULTIMODAL_WORKER_PROMPT:
            raise ValueError("pairwise worker prompt is frozen to the exact exp4 prompt")
        super().__init__(worker_args=worker_args, dataset=dataset, backend_id=backend_id,
            worker_prompt=worker_prompt or PAIRWISE_MULTIMODAL_WORKER_PROMPT,
            worker_postfix=PAIRWISE_WORKER_PROMPT_POSTFIX, max_concurrent=max_concurrent,
            max_retries=max_retries, **kwargs)

    def infer_one(self, data: PairData, criterion: Criterion) -> tuple[PairwiseVoteOutput, ModelCallMetrics]:
        agent_calls: list[AgentCallMetrics] = []
        last_raw: str | None = None
        last_error = "pairwise worker did not return a parseable response"
        total = self.max_retries + 1
        for attempt in range(1, total + 1):
            raw, call_metrics = self._call_model(
                self._make_user_content(data, criterion), request_type="pairwise",
                request_key=f"{data[self.sample_id_field]}::{criterion.name}",
                structured_attempt=attempt)
            agent_calls.append(call_metrics)
            last_raw = raw if isinstance(raw, str) else None
            try:
                parsed = parse_pairwise_vote_response(raw)
                output = PairwiseVoteOutput(parsed.vote, True, raw, None, attempt,
                                            parsed.thought, parsed.answer_valid)
                break
            except PairwiseVoteParseError as exc:
                last_error = str(exc)
        else:
            output = PairwiseVoteOutput(Vote.ABSTAIN, False, last_raw, last_error, total,
                                        None, False)
        metrics = ModelCallMetrics.from_agent_calls(agent_calls, logical_evaluations=1,
            parse_retries=max(0, len(agent_calls) - 1), pricing=self.pricing)
        return output, metrics

    def pred(
        self,
        criteria: Sequence[Criterion],
        *,
        on_sample_complete: Callable[[int, str, ModelCallMetrics], None] | None = None,
    ) -> PairwisePredictionOutput:
        criteria = self._criteria(criteria)
        outputs: list[list[PairwiseVoteOutput | None]] = [
            [None] * len(criteria) for _ in self.dataset
        ]
        metrics_by_sample: list[list[ModelCallMetrics]] = [
            [] for _ in self.dataset
        ]
        remaining = [len(criteria)] * len(self.dataset)
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {
                pool.submit(self.infer_one, row, criterion): (sample_index, criterion_index)
                for sample_index, row in enumerate(self.dataset)
                for criterion_index, criterion in enumerate(criteria)
            }
            for future in as_completed(futures):
                sample_index, criterion_index = futures[future]
                output, metrics = future.result()
                outputs[sample_index][criterion_index] = output
                metrics_by_sample[sample_index].append(metrics)
                remaining[sample_index] -= 1
                if remaining[sample_index] == 0 and on_sample_complete is not None:
                    on_sample_complete(
                        sample_index,
                        str(self.dataset[sample_index][self.sample_id_field]),
                        combine_model_call_metrics(metrics_by_sample[sample_index]),
                    )
        rows = tuple(
            {
                criterion.name: outputs[sample_index][criterion_index]
                for criterion_index, criterion in enumerate(criteria)
            }
            for sample_index in range(len(self.dataset))
        )
        answers = tuple(aggregate_flat_votes(item.vote for item in row.values()) for row in rows)
        return PairwisePredictionOutput(
            tuple(str(row[self.sample_id_field]) for row in self.dataset),
            tuple(self.sample_fingerprint(row) for row in self.dataset),
            tuple(StructuredCriterionSnapshot(item.name, item.description) for item in criteria),
            rows, answers, self.request_spec())


class GateStateMultiModalEvaluator(_DualMultimodalEvaluator):
    """Joint A/B status judge with no pairwise preference or evidence task."""

    def __init__(self, *, worker_args: dict[str, Any], dataset: Sequence[PairData], backend_id: str,
                 max_concurrent: int = 1, max_retries: int = 1, worker_prompt: str | None = None,
                 **kwargs: Any) -> None:
        selected_prompt = worker_prompt or GATE_STATE_WORKER_PROMPT
        prompt_versions = {
            GATE_STATE_WORKER_PROMPT: GATE_WORKER_PROMPT_VERSION,
            GATE_STATE_WORKER_PROMPT_V2: GATE_WORKER_PROMPT_V2_PILOT_VERSION,
            GATE_STATE_WORKER_PROMPT_V2_1: GATE_WORKER_PROMPT_V2_1_PILOT_VERSION,
        }
        if selected_prompt not in prompt_versions:
            raise ValueError("gate worker prompt must be a known versioned prompt")
        self.prompt_version = prompt_versions[selected_prompt]
        request_kwargs = worker_args.get("request_kwargs") or {}
        if request_kwargs.get("temperature") != 0:
            raise ValueError("gate worker requires request_kwargs.temperature=0")
        super().__init__(worker_args=worker_args, dataset=dataset, backend_id=backend_id,
            worker_prompt=selected_prompt,
            worker_postfix=GATE_STATE_WORKER_PROMPT_POSTFIX, max_concurrent=max_concurrent,
            max_retries=max_retries, **kwargs)

    def infer_one(self, data: PairData, criterion: Criterion) -> tuple[GateStateOutput, ModelCallMetrics]:
        agent_calls: list[AgentCallMetrics] = []
        last_raw: str | None = None
        last_error = "gate worker did not return a parseable response"
        last_inconsistent: GateStateOutput | None = None
        total = self.max_retries + 1
        for attempt in range(1, total + 1):
            raw, call_metrics = self._call_model(
                self._make_user_content(data, criterion), request_type="gate",
                request_key=f"{data[self.sample_id_field]}::{criterion.name}",
                structured_attempt=attempt)
            agent_calls.append(call_metrics); last_raw = raw if isinstance(raw, str) else None
            try:
                judgement = parse_gate_state_response(raw)
            except GateStateParseError as exc:
                last_error = str(exc); continue
            candidate = GateStateOutput(judgement, raw, None, attempt)
            if judgement.consistency_ok:
                output = candidate; break
            last_inconsistent = candidate
        else:
            output = (GateStateOutput(last_inconsistent.judgement, last_inconsistent.raw_response,
                                      None, total) if last_inconsistent is not None else
                      GateStateOutput(make_gate_parse_failure(), last_raw, last_error, total))
        metrics = ModelCallMetrics.from_agent_calls(agent_calls, logical_evaluations=1,
            parse_retries=max(0, len(agent_calls) - 1), pricing=self.pricing)
        return output, metrics

    def pred(
        self,
        criteria: Sequence[Criterion],
        *,
        on_sample_complete: Callable[[int, str, ModelCallMetrics], None] | None = None,
    ) -> GatePredictionOutput:
        criteria = self._criteria(criteria)
        outputs: list[list[GateStateOutput | None]] = [
            [None] * len(criteria) for _ in self.dataset
        ]
        metrics_by_sample: list[list[ModelCallMetrics]] = [
            [] for _ in self.dataset
        ]
        remaining = [len(criteria)] * len(self.dataset)
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {
                pool.submit(self.infer_one, row, criterion): (sample_index, criterion_index)
                for sample_index, row in enumerate(self.dataset)
                for criterion_index, criterion in enumerate(criteria)
            }
            for future in as_completed(futures):
                sample_index, criterion_index = futures[future]
                output, metrics = future.result()
                outputs[sample_index][criterion_index] = output
                metrics_by_sample[sample_index].append(metrics)
                remaining[sample_index] -= 1
                if remaining[sample_index] == 0 and on_sample_complete is not None:
                    on_sample_complete(
                        sample_index,
                        str(self.dataset[sample_index][self.sample_id_field]),
                        combine_model_call_metrics(metrics_by_sample[sample_index]),
                    )
        rows = tuple(
            {
                criterion.name: outputs[sample_index][criterion_index]
                for criterion_index, criterion in enumerate(criteria)
            }
            for sample_index in range(len(self.dataset))
        )
        return GatePredictionOutput(
            tuple(str(row[self.sample_id_field]) for row in self.dataset),
            tuple(self.sample_fingerprint(row) for row in self.dataset),
            tuple(StructuredCriterionSnapshot(item.name, item.description) for item in criteria),
            rows, self.request_spec(), prompt_version=self.prompt_version)

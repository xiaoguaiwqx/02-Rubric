"""Cost telemetry shared by structured router and worker backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from structured_rubrics.agent import AgentCallMetrics


def _require_non_negative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


@dataclass(frozen=True)
class TokenPricing:
    """Optional USD prices per one million input/output tokens."""

    input_usd_per_million: float
    output_usd_per_million: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_usd_per_million",
            _require_non_negative_number(
                self.input_usd_per_million, "input_usd_per_million"
            ),
        )
        object.__setattr__(
            self,
            "output_usd_per_million",
            _require_non_negative_number(
                self.output_usd_per_million, "output_usd_per_million"
            ),
        )

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_million
            + output_tokens * self.output_usd_per_million
        ) / 1_000_000

    def to_dict(self) -> dict[str, float]:
        return {
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
        }

    @classmethod
    def from_dict(cls, value: object) -> "TokenPricing":
        if not isinstance(value, dict) or set(value) != {
            "input_usd_per_million",
            "output_usd_per_million",
        }:
            raise ValueError("token pricing fields are invalid")
        return cls(**value)


@dataclass(frozen=True)
class ModelCallMetrics:
    """Actual cost observed in the current run, excluding counterfactual calls."""

    logical_evaluations: int = 0
    api_attempts: int = 0
    parse_retries: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0
    usage_complete: bool = True
    call_latency_seconds: float = 0.0
    estimated_cost_usd: float | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "logical_evaluations",
            "api_attempts",
            "parse_retries",
            "cache_hits",
            "cache_misses",
            "error_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.usage_complete, bool):
            raise TypeError("usage_complete must be bool")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if self.usage_complete:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
            elif value is not None:
                raise ValueError(f"{name} must be None when usage is incomplete")
        object.__setattr__(
            self,
            "call_latency_seconds",
            _require_non_negative_number(
                self.call_latency_seconds, "call_latency_seconds"
            ),
        )
        if self.estimated_cost_usd is not None:
            object.__setattr__(
                self,
                "estimated_cost_usd",
                _require_non_negative_number(
                    self.estimated_cost_usd, "estimated_cost_usd"
                ),
            )

    @classmethod
    def from_agent_calls(
        cls,
        calls: Iterable[AgentCallMetrics],
        *,
        logical_evaluations: int = 1,
        parse_retries: int = 0,
        cache_hit: bool = False,
        pricing: TokenPricing | None = None,
    ) -> "ModelCallMetrics":
        call_tuple = tuple(calls)
        if cache_hit:
            if call_tuple:
                raise ValueError("cache hit must not include Agent calls")
            return cls(logical_evaluations=logical_evaluations, cache_hits=1)

        usage_complete = all(call.usage_complete for call in call_tuple)
        input_tokens = (
            sum(call.input_tokens or 0 for call in call_tuple)
            if usage_complete
            else None
        )
        output_tokens = (
            sum(call.output_tokens or 0 for call in call_tuple)
            if usage_complete
            else None
        )
        total_tokens = (
            sum(call.total_tokens or 0 for call in call_tuple)
            if usage_complete
            else None
        )
        estimated_cost = (
            pricing.estimate(input_tokens, output_tokens)
            if pricing is not None and usage_complete
            else None
        )
        return cls(
            logical_evaluations=logical_evaluations,
            api_attempts=sum(call.api_attempts for call in call_tuple),
            parse_retries=parse_retries,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_complete=usage_complete,
            call_latency_seconds=sum(call.latency_seconds for call in call_tuple),
            estimated_cost_usd=estimated_cost,
            cache_misses=0,
            error_count=sum(call.error_count for call in call_tuple),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_evaluations": self.logical_evaluations,
            "api_attempts": self.api_attempts,
            "parse_retries": self.parse_retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usage_complete": self.usage_complete,
            "call_latency_seconds": self.call_latency_seconds,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ModelCallMetrics":
        if not isinstance(value, dict):
            raise ValueError("model call metrics must be an object")
        expected = {
            "logical_evaluations", "api_attempts", "parse_retries",
            "input_tokens", "output_tokens", "total_tokens", "usage_complete",
            "call_latency_seconds", "estimated_cost_usd", "cache_hits",
            "cache_misses", "error_count",
        }
        if set(value) != expected:
            raise ValueError("model call metrics fields are invalid")
        return cls(**value)


def combine_model_call_metrics(
    metrics: Iterable[ModelCallMetrics],
) -> ModelCallMetrics:
    """Combine current-run costs while preserving missing-usage semantics."""

    values = tuple(metrics)
    usage_complete = all(value.usage_complete for value in values)
    costs = [value.estimated_cost_usd for value in values]
    cost_complete = bool(values) and all(cost is not None for cost in costs)
    return ModelCallMetrics(
        logical_evaluations=sum(value.logical_evaluations for value in values),
        api_attempts=sum(value.api_attempts for value in values),
        parse_retries=sum(value.parse_retries for value in values),
        input_tokens=(sum(value.input_tokens or 0 for value in values) if usage_complete else None),
        output_tokens=(sum(value.output_tokens or 0 for value in values) if usage_complete else None),
        total_tokens=(sum(value.total_tokens or 0 for value in values) if usage_complete else None),
        usage_complete=usage_complete,
        call_latency_seconds=sum(value.call_latency_seconds for value in values),
        estimated_cost_usd=(sum(cost or 0.0 for cost in costs) if cost_complete else None),
        cache_hits=sum(value.cache_hits for value in values),
        cache_misses=sum(value.cache_misses for value in values),
        error_count=sum(value.error_count for value in values),
    )

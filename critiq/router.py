"""Dynamic soft routing for multimodal criterion evaluation."""

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .agent import Agent
from .utils import Criterion, PairData, parse_json, print_debug


@dataclass(frozen=True)
class RoutingDecision:
    """Router output for one multimodal pair sample."""

    scene_analysis: str
    routing_weights: dict[str, float]
    source: Literal["router", "fallback"]


class RouterManager:
    """Use a VLM to choose which criteria should evaluate a sample.

    The router sees only the image, question, and criterion descriptions. Candidate
    answers A/B are intentionally omitted to avoid leaking answer-specific bias into
    the gating stage.
    """

    def __init__(
        self,
        router_args: dict,
        image_field: str = "image_path",
        question_field: str = "question",
        encode_local_image: bool = True,
        routing_threshold: float = 0.2,
        fallback_criteria: Sequence[str] = (
            "visual_grounding",
            "factual_consistency",
        ),
        fallback_weight: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        self.router_args = router_args
        self.image_field = image_field
        self.question_field = question_field
        self.encode_local_image = encode_local_image
        self.routing_threshold = routing_threshold
        self.fallback_criteria = tuple(fallback_criteria)
        self.fallback_weight = fallback_weight
        self.max_retries = max_retries

    @staticmethod
    def _clamp_weight(value: object) -> float:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, weight))

    @staticmethod
    def _image_path_to_data_url(image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "image/jpeg"

        with path.open("rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{image_b64}"

    def _fallback_decision(
        self, criteria: Sequence[Criterion], reason: str | None = None
    ) -> RoutingDecision:
        if reason:
            print_debug(reason)
        return RoutingDecision(
            scene_analysis="Router failed; all criteria are routed by fallback.",
            routing_weights={criterion.name: 1.0 for criterion in criteria},
            source="fallback",
        )

    def _apply_broad_fallback(
        self, weights: dict[str, float], criteria: Sequence[Criterion]
    ) -> dict[str, float]:
        available = {criterion.name for criterion in criteria}
        for name in self.fallback_criteria:
            if name in available:
                weights[name] = max(weights.get(name, 0.0), self.fallback_weight)
        return weights

    def _make_prompt(self, data: PairData, criteria: Sequence[Criterion]) -> str:
        question = data.get(self.question_field, "")
        question = "" if question is None else str(question)
        criteria_text = "\n".join(
            f"- {criterion.name}: {criterion.description}" for criterion in criteria
        )
        names_text = ", ".join(criterion.name for criterion in criteria)
        return f"""You are a multimodal router for RLHF-V visual QA preference evaluation.

You will receive the image and the source question. You will NOT receive the two candidate answers. Your task is to decide which evaluation criteria are relevant for judging answers to this question and image.

# Question
{question}

# Available Criteria
{criteria_text}

Return a JSON object with:
- "scene_analysis": a concise analysis of the visual/question dimensions that matter.
- "routing_weights": an object mapping every criterion name to a relevance weight from 0.0 to 1.0.

Use only these criterion names: {names_text}
Use 0.0 for irrelevant criteria, around 0.5 for partially relevant criteria, and close to 1.0 for core criteria.
"""

    def _make_user_content(self, data: PairData, criteria: Sequence[Criterion]):
        prompt = self._make_prompt(data, criteria)
        image_path = data.get(self.image_field)
        image_path = "" if image_path is None else str(image_path)
        if not image_path:
            return prompt

        image_url = (
            self._image_path_to_data_url(image_path)
            if self.encode_local_image
            else str(image_path)
        )
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    def _route_once(
        self, data: PairData, criteria: Sequence[Criterion]
    ) -> RoutingDecision:
        router = Agent(**self.router_args)
        response = router(self._make_user_content(data, criteria), stream=False)
        parsed = parse_json(response)

        raw_weights = parsed.get("routing_weights", {})
        if not isinstance(raw_weights, dict):
            raise ValueError("Router response has no routing_weights object")

        weights = {
            criterion.name: self._clamp_weight(raw_weights.get(criterion.name, 0.0))
            for criterion in criteria
        }
        weights = self._apply_broad_fallback(weights, criteria)
        scene_analysis = parsed.get("scene_analysis", "")
        scene_analysis = "" if scene_analysis is None else str(scene_analysis)
        return RoutingDecision(
            scene_analysis=scene_analysis,
            routing_weights=weights,
            source="router",
        )

    def route(self, data: PairData, criteria: Sequence[Criterion]) -> RoutingDecision:
        """Route one sample, falling back to all criteria on any router failure."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._route_once(data, criteria)
            except Exception as e:  # pylint: disable=W0718:broad-exception-caught
                last_error = e
                print_debug(f"Failed to parse router response, retrying {attempt=}", e)

        return self._fallback_decision(criteria, reason=f"Router fallback: {last_error}")


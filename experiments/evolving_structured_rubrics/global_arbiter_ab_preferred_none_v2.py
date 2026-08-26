"""Clean rerun profile for the A/B-preferred, None-tolerant Global Arbiter.

The model prompt is intentionally byte-identical to the original A/B-only
experiment.  Only parser semantics and the cache/protocol namespace differ:
``None`` is a valid abstention, while malformed outputs are retried with the
same prompt.  No tie-break rescue prompt is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import global_arbiter_ab_only as implementation


PROTOCOL_VERSION = "global-arbiter-ab-preferred-none-tolerant-v2-vlrb-only"
PROMPT_VERSION = implementation.PROMPT_VERSION
EXPERIMENT_DIR = "vl_rewardbench_global_arbiter_ab_preferred_none_v2"
CONFIG_KEY = "global_arbiter_ab_preferred_none_v2_experiment"
SYSTEM_NAME = "s5_v2_global_arbiter_ab_preferred_none_tolerant"
REQUEST_SCOPE = "s5_v2_global_arbiter_ab_preferred_none_all_requests"
REQUEST_KIND = "global_arbiter_ab_preferred_none_v2"
LEGACY_S5_EXPERIMENT_DIR = implementation.EXPERIMENT_DIR
REPORT_TITLE = "VL-RewardBench Single-Prompt Global Arbiter v2 (A/B-preferred, None-tolerant)"

STAGES = (
    "vlrb-global-arbiter-ab-preferred-none-v2-freeze",
    "vlrb-global-arbiter-ab-preferred-none-v2-audit",
    "vlrb-global-arbiter-ab-preferred-none-v2-smoke",
    "vlrb-global-arbiter-ab-preferred-none-v2-run",
    "vlrb-global-arbiter-ab-preferred-none-v2-retry",
    "vlrb-global-arbiter-ab-preferred-none-v2-report",
)

# Alias instead of redefining the text: this is the scientific guarantee that
# the clean rerun changes no model instruction.
GLOBAL_ARBITER_SYSTEM_PROMPT = (
    implementation.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT)
parse_global_arbiter_response = (
    implementation.parse_global_arbiter_ab_only_response)


def _activate_profile() -> None:
    """Apply the v2 identity to the shared single-prompt implementation."""

    implementation.PROTOCOL_VERSION = PROTOCOL_VERSION
    implementation.EXPERIMENT_DIR = EXPERIMENT_DIR
    implementation.CONFIG_KEY = CONFIG_KEY
    implementation.SYSTEM_NAME = SYSTEM_NAME
    implementation.REQUEST_SCOPE = REQUEST_SCOPE
    implementation.REQUEST_KIND = REQUEST_KIND
    implementation.LEGACY_S5_EXPERIMENT_DIR = LEGACY_S5_EXPERIMENT_DIR
    implementation.REPORT_TITLE = REPORT_TITLE
    implementation.STAGES = STAGES


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    _activate_profile()
    implementation.run_stage(config, output, stage)

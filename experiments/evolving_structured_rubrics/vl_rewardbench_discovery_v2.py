"""VL-RewardBench transfer evaluation for the Phase17 Discovery-v2 Rubric.

The inference/retry/report machinery is shared with the previously validated
Phase16 Prompt-v2 runner.  This wrapper changes only the frozen source Rubric,
experiment directory, stage names, and protocol identity.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import discovery_v2_prompt_v2_evolution as evolution
from . import run_rubric_evolution as base
from . import vl_rewardbench as legacy
from . import vl_rewardbench_prompt_v2 as control
from . import vl_rewardbench_prompt_v2_evolved as shared


EXPERIMENT_DIR = "vl_rewardbench_phase17_discovery_v2_prompt_v2_v1"
PROTOCOL_VERSION = "vlrb-discovery-v2-prompt-v2-v1"
TREATMENT_SYSTEM = "phase17_discovery_v2_final_equal"

STAGE_FREEZE = "vlrb-discovery-v2-freeze"
STAGE_AUDIT = "vlrb-discovery-v2-audit"
STAGE_SMOKE = "vlrb-discovery-v2-smoke"
STAGE_RUN = "vlrb-discovery-v2-run"
STAGE_RETRY = "vlrb-discovery-v2-retry"
STAGE_REPORT = "vlrb-discovery-v2-report"

SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": evolution.EXPERIMENT_DIR,
    "control_experiment": control.EXPERIMENT_DIR,
    "endpoint_ids": list(control.ENDPOINT_IDS),
    "scheduler": "sample_major_available_slot_dynamic",
    "prompt_version": "1.1.0-cache-pilot",
    "smoke_sample_count": 20,
    "max_retry_attempts": 10,
    "k": legacy.K,
    "seed": legacy.SEED,
    "run_regardless_of_heldout_result": True,
    "selection_after_benchmark_forbidden": True,
    "primary_external_evaluation": True,
}


def _activate() -> None:
    """Configure the generic Prompt-v2 transfer implementation for Phase17."""

    shared.EXPERIMENT_DIR = EXPERIMENT_DIR
    shared.PROTOCOL_VERSION = PROTOCOL_VERSION
    shared.CONTROL_EXPERIMENT = control.EXPERIMENT_DIR
    shared.TREATMENT_SYSTEM = TREATMENT_SYSTEM
    shared.SCHEDULER = "sample_major_available_slot_dynamic"
    shared.EXTRA_BASELINE_REPORTS = {
        "phase16_final_equal_prompt_v2": (
            base.ROOT / "output/evolving_structured_rubrics"
            / "vl_rewardbench_phase16_prompt_v2_evolved_v2"
            / "final_report.json",
            "prompt_v2_evolved_final_equal",
        )
    }
    shared.STAGE_FREEZE = STAGE_FREEZE
    shared.STAGE_AUDIT = STAGE_AUDIT
    shared.STAGE_SMOKE = STAGE_SMOKE
    shared.STAGE_RUN = STAGE_RUN
    shared.STAGE_RETRY = STAGE_RETRY
    shared.STAGE_REPORT = STAGE_REPORT
    shared.evolution = evolution


def _adapt_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("vlrb_discovery_v2")
    if value != SETTINGS:
        raise RuntimeError("vlrb_discovery_v2 must match the frozen v1 protocol")
    adapted = deepcopy(dict(config))
    # The shared implementation validates this internal generic protocol view.
    adapted["vlrb_prompt_v2_evolved"] = {
        key: item for key, item in SETTINGS.items()
        if key not in {"primary_external_evaluation"}
    }
    return adapted


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    _activate()
    shared.run_stage(_adapt_config(config), output, stage)

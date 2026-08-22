"""VL-RewardBench checkpoint diagnostic for Phase17 epochs 2, 3, and 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import discovery_v2_prompt_v2_evolution as evolution
from . import vl_rewardbench_discovery_v2 as source_vlrb
from . import vl_rewardbench_phase16_checkpoints as shared


EXPERIMENT_DIR = "vl_rewardbench_phase17_checkpoint_transfer_v1"
PROTOCOL_VERSION = "vlrb-phase17-checkpoint-transfer-v1"
CONFIG_KEY = "vlrb_phase17_checkpoint_transfer"
EPOCHS = (2, 3, 4)
EXPECTED_VARIANT_DESCRIPTION_COUNT = 18
STAGE_PREFIX = "vlrb-phase17-checkpoint"
STAGES = tuple(f"{STAGE_PREFIX}-{name}" for name in (
    "freeze", "audit", "smoke", "run", "retry", "report"))

SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_experiment": evolution.EXPERIMENT_DIR,
    "source_vlrb_experiment": source_vlrb.EXPERIMENT_DIR,
    "epochs": list(EPOCHS),
    "endpoint_ids": ["vllm-8000", "vllm-8001"],
    "scheduler": "sample_major_available_slot_dynamic",
    "prompt_version": "1.1.0-cache-pilot",
    "smoke_sample_count": 20,
    "max_retry_attempts": 10,
    "k": 3,
    "seed": 42,
    "selection_after_benchmark_forbidden": True,
    "expected_unique_variant_description_count": (
        EXPECTED_VARIANT_DESCRIPTION_COUNT),
}

CHECKPOINT_ROLES = {
    2: "dev150_best_checkpoint",
    3: "discovery100_best_checkpoint",
    4: "first_checkpoint_with_all_five_roots_split",
}


def _activate() -> None:
    """Activate the generic checkpoint runner with the frozen Phase17 view."""

    shared.EXPERIMENT_DIR = EXPERIMENT_DIR
    shared.PROTOCOL_VERSION = PROTOCOL_VERSION
    shared.CONFIG_KEY = CONFIG_KEY
    shared.SOURCE_EVOLUTION_EXPERIMENT_DIR = evolution.EXPERIMENT_DIR
    shared.SOURCE_VLRB_EXPERIMENT_DIR = source_vlrb.EXPERIMENT_DIR
    shared.SYSTEM_PREFIX = "phase17"
    shared.REPORT_TITLE = (
        "Phase17 checkpoint VL-RewardBench transfer diagnostic")
    shared.EPOCHS = EPOCHS
    shared.EXPECTED_VARIANT_DESCRIPTION_COUNT = (
        EXPECTED_VARIANT_DESCRIPTION_COUNT)
    shared.CHECKPOINT_ROLES = CHECKPOINT_ROLES
    shared.STAGE_PREFIX = STAGE_PREFIX
    shared.STAGES = STAGES


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    _activate()
    shared.run_stage(config, output, stage)

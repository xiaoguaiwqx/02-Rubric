"""Explicit, immutable identities for shared evolution and evaluation engines.

Protocol settings use a JSON snapshot so callers cannot mutate nested trigger
thresholds through a shared dictionary. Runtime settings remain ordinary JSON
values; existing manifests and request identities keep their original shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class EvolutionProtocol:
    experiment_dir: str
    version: str
    config_key: str
    stages: tuple[str, ...]
    _settings_json: str = field(repr=False)

    @classmethod
    def from_settings(
        cls, *, experiment_dir: str, version: str, config_key: str,
        stages: tuple[str, ...], settings: Mapping[str, Any],
    ) -> EvolutionProtocol:
        if settings.get("protocol_version") != version:
            raise ValueError("protocol/settings version mismatch")
        return cls(experiment_dir, version, config_key, stages,
                   json.dumps(dict(settings), ensure_ascii=False))

    @property
    def settings(self) -> dict[str, Any]:
        return json.loads(self._settings_json)

    def validate(self, config: Mapping[str, Any]) -> dict[str, Any]:
        expected = self.settings
        if config.get(self.config_key) != expected:
            raise RuntimeError(f"{self.config_key} drift")
        return expected


@dataclass(frozen=True)
class BenchmarkProtocol:
    identity: EvolutionProtocol
    source: EvolutionProtocol
    final_label: str
    source_rubric_key: str
    report_title: str

    @property
    def experiment_dir(self) -> str:
        return self.identity.experiment_dir

    @property
    def version(self) -> str:
        return self.identity.version

    @property
    def stages(self) -> tuple[str, ...]:
        return self.identity.stages

    @property
    def settings(self) -> dict[str, Any]:
        return self.identity.settings

    def validate(self, config: Mapping[str, Any]) -> dict[str, Any]:
        return self.identity.validate(config)

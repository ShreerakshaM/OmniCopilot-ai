"""Ablation study runner — systematically disable components and measure impact."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omnicopilot.evaluation.metrics import (
    CommunicationMetrics,
    PerceptionMetrics,
    SafetyMetrics,
)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    ablation_name: str  # e.g., "no_uncertainty", "no_prioritization".
    component_removed: str
    perception: PerceptionMetrics
    communication: CommunicationMetrics
    safety: SafetyMetrics
    delta_map_vs_full: float = 0.0  # Δ mAP compared to full system.


@dataclass
class AblationSuiteResult:
    """Complete ablation study results."""

    full_system: AblationResult
    ablations: list[AblationResult] = field(default_factory=list)
    most_important_component: str = ""
    importance_ranking: list[tuple[str, float]] = field(default_factory=list)


def run_ablation_suite(
    base_config_path: Path,
    ablation_configs: list[Path],
    output_dir: Path,
) -> AblationSuiteResult:
    """Run a complete ablation study.

    Args:
        base_config_path: Path to the full-system config.
        ablation_configs: Paths to configs with one component removed each.
        output_dir: Where to save results.

    Returns:
        Complete ablation results with rankings.
    """
    # TODO: Implement — run each config, collect metrics, rank components.
    raise NotImplementedError

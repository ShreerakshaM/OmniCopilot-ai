"""Typed configuration dataclasses for Hydra.

These define the structure of experiment configurations.
Hydra composes them from YAML files and validates types.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerceptionConfig:
    """Perception model configuration."""

    model_name: str = "pointpillars"
    checkpoint_path: str = ""
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.5
    max_detections: int = 100


@dataclass
class FusionConfig:
    """Fusion algorithm configuration."""

    algorithm: str = "weighted_average"  # "weighted_average", "bayesian", "learned"
    association_threshold_m: float = 3.0
    trust_weight: float = 0.5
    conflict_threshold: float = 0.6


@dataclass
class CommunicationConfig:
    """Communication configuration."""

    policy: str = "hand_designed"  # "none", "hand_designed", "learned"
    bandwidth_bps: float = 6e6
    latency_ms: float = 20.0
    packet_loss_rate: float = 0.05
    budget_bytes_per_tick: int = 4096
    # Prioritization weights (hand-designed policy).
    safety_weight: float = 0.35
    novelty_weight: float = 0.20
    confidence_weight: float = 0.15
    time_sensitivity_weight: float = 0.15
    receiver_benefit_weight: float = 0.15


@dataclass
class ActiveAcquisitionConfig:
    """Active information acquisition configuration."""

    enabled: bool = True
    max_requests_per_tick: int = 3
    request_timeout_s: float = 1.0
    min_value_threshold: float = 0.2


@dataclass
class TrustConfig:
    """Agent trust/reliability configuration."""

    enabled: bool = True
    initial_trust: float = 0.7
    ema_alpha: float = 0.05
    min_trust: float = 0.05


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    # Experiment metadata.
    name: str = "default"
    seed: int = 42
    description: str = ""

    # Scenario.
    dataset: str = "opv2v"
    dataset_path: str = "data/raw/opv2v"
    split: str = "test"
    num_agents: int = 3
    duration_s: float = 30.0
    tick_rate_hz: float = 10.0

    # Components.
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    active_acquisition: ActiveAcquisitionConfig = field(
        default_factory=ActiveAcquisitionConfig
    )
    trust: TrustConfig = field(default_factory=TrustConfig)

    # Output.
    output_dir: str = "data/results"
    wandb_project: str = "omnicopilot"
    wandb_enabled: bool = True

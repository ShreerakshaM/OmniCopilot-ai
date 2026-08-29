"""Cooperative perception models (V2VNet, Where2comm, etc.).

Wraps OpenCOOD cooperative perception models for multi-agent perception.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class CooperativeInput:
    """Input for cooperative perception — data from multiple agents."""

    # Per-agent sensor data.
    agent_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-agent poses (4x4 transformation matrices).
    agent_poses: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class CooperativeOutput:
    """Output from cooperative perception model."""

    # Fused detections (in ego vehicle frame).
    detections: list[Any] = field(default_factory=list)
    # Per-agent feature maps (for communication analysis).
    agent_features: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)
    # Communication cost (bytes transmitted).
    communication_bytes: int = 0


class BaseCooperativeModel(ABC):
    """Abstract base for cooperative perception models."""

    @abstractmethod
    def forward(self, inputs: CooperativeInput) -> CooperativeOutput:
        """Run cooperative perception.

        Args:
            inputs: Multi-agent sensor data and poses.

        Returns:
            Fused cooperative perception output.
        """

    @abstractmethod
    def get_communication_cost(self) -> int:
        """Return bytes required for inter-agent communication."""


class V2VNetModel(BaseCooperativeModel):
    """V2VNet cooperative perception model.

    Reference: Wang et al., "V2VNet: Vehicle-to-Vehicle Communication for
    Joint Perception and Prediction", ECCV 2020.
    """

    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        """Initialize V2VNet."""
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path

    def forward(self, inputs: CooperativeInput) -> CooperativeOutput:
        """Run V2VNet cooperative perception."""
        # TODO: Implement via OpenCOOD.
        raise NotImplementedError

    def get_communication_cost(self) -> int:
        """Return communication cost for V2VNet feature sharing."""
        # TODO: Measure from model.
        raise NotImplementedError

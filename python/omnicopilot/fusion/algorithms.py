"""Fusion algorithms — combine observations from multiple agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class FusedEstimate:
    """Result of fusing multiple observations of the same entity."""

    position: npt.NDArray[np.float64]  # shape (3,)
    velocity: npt.NDArray[np.float64]  # shape (3,)
    confidence: float
    source_count: int
    class_id: int
    class_confidence: float


class BaseFusionAlgorithm(ABC):
    """Abstract base for fusion algorithms."""

    @abstractmethod
    def fuse(
        self,
        positions: npt.NDArray[np.float64],
        confidences: npt.NDArray[np.float64],
        trusts: npt.NDArray[np.float64],
    ) -> FusedEstimate:
        """Fuse multiple observations into a single estimate.

        Args:
            positions: Observation positions, shape (N, 3).
            confidences: Per-observation confidence, shape (N,).
            trusts: Per-agent trust scores, shape (N,).

        Returns:
            Fused position and confidence estimate.
        """


class WeightedAverageFusion(BaseFusionAlgorithm):
    """Simple confidence-and-trust weighted averaging."""

    def fuse(
        self,
        positions: npt.NDArray[np.float64],
        confidences: npt.NDArray[np.float64],
        trusts: npt.NDArray[np.float64],
    ) -> FusedEstimate:
        """Fuse by weighted average of positions."""
        weights = confidences * trusts
        weight_sum = weights.sum()
        if weight_sum < 1e-10:
            # Fallback to uniform.
            fused_pos = positions.mean(axis=0)
            fused_conf = 0.0
        else:
            fused_pos = (positions * weights[:, np.newaxis]).sum(axis=0) / weight_sum
            fused_conf = float(1.0 - np.prod(1.0 - confidences * trusts))

        return FusedEstimate(
            position=fused_pos,
            velocity=np.zeros(3),
            confidence=fused_conf,
            source_count=len(positions),
            class_id=0,
            class_confidence=0.0,
        )


class BayesianFusion(BaseFusionAlgorithm):
    """Bayesian confidence update treating observations as independent evidence."""

    def fuse(
        self,
        positions: npt.NDArray[np.float64],
        confidences: npt.NDArray[np.float64],
        trusts: npt.NDArray[np.float64],
    ) -> FusedEstimate:
        """Bayesian fusion of position estimates."""
        # TODO: Implement proper Bayesian update with Gaussian likelihoods.
        raise NotImplementedError

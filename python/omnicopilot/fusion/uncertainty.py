"""Uncertainty estimation methods for perception models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a detection."""

    aleatoric: float  # Data uncertainty (irreducible).
    epistemic: float  # Model uncertainty (reducible with more data).
    total: float  # Combined uncertainty.
    calibrated_confidence: float  # Calibrated confidence score.


class BaseUncertaintyEstimator(ABC):
    """Abstract base for uncertainty estimation."""

    @abstractmethod
    def estimate(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]
    ) -> list[UncertaintyEstimate]:
        """Estimate uncertainty for each detection."""


class MCDropoutEstimator(BaseUncertaintyEstimator):
    """Monte Carlo Dropout uncertainty estimation.

    Runs multiple forward passes with dropout enabled and measures variance.
    """

    def __init__(self, num_samples: int = 10) -> None:
        """Initialize MC Dropout estimator.

        Args:
            num_samples: Number of stochastic forward passes.
        """
        self._num_samples = num_samples

    def estimate(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]
    ) -> list[UncertaintyEstimate]:
        """Estimate uncertainty via MC Dropout."""
        # TODO: Implement — enable dropout, run N passes, compute variance.
        raise NotImplementedError


class EnsembleEstimator(BaseUncertaintyEstimator):
    """Deep ensemble uncertainty estimation.

    Uses disagreement between independently trained models.
    """

    def __init__(self, models: list[torch.nn.Module]) -> None:
        """Initialize ensemble estimator.

        Args:
            models: List of independently trained models.
        """
        self._models = models

    def estimate(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]
    ) -> list[UncertaintyEstimate]:
        """Estimate uncertainty from ensemble disagreement."""
        # TODO: Implement — run all models, compute variance across predictions.
        raise NotImplementedError

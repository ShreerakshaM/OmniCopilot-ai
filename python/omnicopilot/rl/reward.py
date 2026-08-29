"""Reward function design for communication policy RL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class RewardComponents:
    """Breakdown of reward into interpretable components."""

    perception_improvement: float  # Δ mAP at receivers.
    uncertainty_reduction: float  # Reduction in receiver uncertainty.
    bandwidth_penalty: float  # Cost for bytes transmitted.
    novelty_bonus: float  # Bonus for genuinely new information.
    total: float  # Weighted sum.


class BaseReward(ABC):
    """Abstract base for reward functions."""

    @abstractmethod
    def compute(
        self,
        receiver_map_before: float,
        receiver_map_after: float,
        bytes_transmitted: int,
        budget_bytes: int,
        novelty_fraction: float,
    ) -> RewardComponents:
        """Compute reward for a communication action.

        Args:
            receiver_map_before: Receivers' collective mAP before message.
            receiver_map_after: Receivers' collective mAP after message.
            bytes_transmitted: Bytes used for transmission.
            budget_bytes: Total budget available.
            novelty_fraction: Fraction of transmitted info that was genuinely new.

        Returns:
            Reward breakdown.
        """


class PerceptionImprovementReward(BaseReward):
    """Reward based on improvement in receiver perception accuracy."""

    def __init__(
        self,
        perception_weight: float = 1.0,
        bandwidth_weight: float = 0.1,
        novelty_weight: float = 0.2,
    ) -> None:
        """Initialize reward weights."""
        self._perception_w = perception_weight
        self._bandwidth_w = bandwidth_weight
        self._novelty_w = novelty_weight

    def compute(
        self,
        receiver_map_before: float,
        receiver_map_after: float,
        bytes_transmitted: int,
        budget_bytes: int,
        novelty_fraction: float,
    ) -> RewardComponents:
        """Compute perception-improvement-based reward."""
        delta_map = receiver_map_after - receiver_map_before
        bandwidth_cost = bytes_transmitted / max(budget_bytes, 1)
        novelty_bonus = novelty_fraction

        total = (
            self._perception_w * delta_map
            - self._bandwidth_w * bandwidth_cost
            + self._novelty_w * novelty_bonus
        )

        return RewardComponents(
            perception_improvement=delta_map,
            uncertainty_reduction=0.0,
            bandwidth_penalty=-bandwidth_cost * self._bandwidth_w,
            novelty_bonus=novelty_bonus * self._novelty_w,
            total=total,
        )

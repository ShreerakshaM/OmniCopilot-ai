"""Gymnasium environment for learning communication policies.

The agent decides which observations to transmit under bandwidth constraints.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces


class CommunicationEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Communication policy environment.

    Observation space:
        - Agent's local detections (positions, confidences).
        - Estimated receiver belief states.
        - Network conditions (bandwidth, latency, loss).

    Action space:
        - Per-observation transmit score (continuous [0, 1]).
        - Top-K observations above threshold are transmitted.

    Reward:
        - Improvement in collective perception at receiving agents.
        - Penalty for bandwidth usage.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        num_observations_max: int = 50,
        num_receivers: int = 4,
        bandwidth_budget_bytes: int = 4096,
        observation_bytes: int = 256,
    ) -> None:
        """Initialize communication environment.

        Args:
            num_observations_max: Maximum observations per step.
            num_receivers: Number of other agents (receivers).
            bandwidth_budget_bytes: Bandwidth budget per step.
            observation_bytes: Estimated bytes per observation.
        """
        super().__init__()

        self._num_obs_max = num_observations_max
        self._num_receivers = num_receivers
        self._budget_bytes = bandwidth_budget_bytes
        self._obs_bytes = observation_bytes
        self._max_transmit = bandwidth_budget_bytes // observation_bytes

        # Observation space: flattened state vector.
        obs_dim = (
            num_observations_max * 7  # Per-obs: x, y, z, vx, vy, conf, class
            + num_receivers * 3  # Per-receiver: x, y, estimated_knowledge
            + 4  # Network state: bandwidth, latency, loss, queue
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: per-observation transmit probability.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_observations_max,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        # TODO: Load scenario, initialize agents.
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one step — transmit selected observations.

        Args:
            action: Per-observation transmit scores.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # TODO: Implement communication simulation and reward computation.
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

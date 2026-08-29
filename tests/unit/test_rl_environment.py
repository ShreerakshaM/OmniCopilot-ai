"""Unit tests for RL communication environment."""

from __future__ import annotations

import numpy as np

from omnicopilot.rl.environment import CommunicationEnv


class TestCommunicationEnv:
    """Tests for the Gymnasium environment."""

    def test_env_creation(self) -> None:
        """Environment can be created without errors."""
        env = CommunicationEnv()
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_reset_returns_valid_observation(self) -> None:
        """Reset returns observation matching the observation space."""
        env = CommunicationEnv()
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_returns_valid_shape(self) -> None:
        """Step returns correctly shaped outputs."""
        env = CommunicationEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_action_space_bounds(self) -> None:
        """Action space is bounded [0, 1]."""
        env = CommunicationEnv(num_observations_max=20)
        assert env.action_space.shape == (20,)
        assert env.action_space.low.min() == 0.0
        assert env.action_space.high.max() == 1.0

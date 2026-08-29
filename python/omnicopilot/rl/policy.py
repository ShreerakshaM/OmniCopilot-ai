"""Communication policy network architectures."""

from __future__ import annotations

import torch
import torch.nn as nn


class CommunicationPolicyNetwork(nn.Module):
    """MLP-based communication policy network.

    Input: state (local observations + receiver beliefs + network state).
    Output: per-observation transmit scores.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
    ) -> None:
        """Initialize policy network.

        Args:
            obs_dim: Observation space dimension.
            action_dim: Action space dimension (num observations).
            hidden_dims: Hidden layer sizes.
        """
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1].

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass — produce transmit scores.

        Args:
            obs: Observation tensor, shape (batch, obs_dim).

        Returns:
            Transmit scores, shape (batch, action_dim).
        """
        return self.network(obs)

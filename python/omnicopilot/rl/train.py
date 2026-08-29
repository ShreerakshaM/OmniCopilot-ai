"""PPO training loop for communication policy (CleanRL-style)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Training configuration."""

    # Environment.
    num_envs: int = 4
    num_steps: int = 2048
    # PPO.
    learning_rate: float = 3e-4
    num_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Training.
    total_timesteps: int = 1_000_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    # Paths.
    checkpoint_dir: Path = Path("data/models/communication_policy")
    wandb_project: str = "omnicopilot-rl"


def train(config: TrainConfig) -> None:
    """Run PPO training for communication policy.

    Args:
        config: Training configuration.
    """
    # TODO: Implement CleanRL-style PPO training loop.
    # 1. Create vectorized environments.
    # 2. Initialize policy and value networks.
    # 3. Collect rollouts.
    # 4. Compute advantages (GAE).
    # 5. PPO update.
    # 6. Log to W&B.
    raise NotImplementedError


def evaluate(checkpoint_path: Path, num_episodes: int = 100) -> dict[str, float]:
    """Evaluate a trained policy.

    Args:
        checkpoint_path: Path to saved policy checkpoint.
        num_episodes: Number of evaluation episodes.

    Returns:
        Dictionary of evaluation metrics.
    """
    # TODO: Implement evaluation loop.
    raise NotImplementedError

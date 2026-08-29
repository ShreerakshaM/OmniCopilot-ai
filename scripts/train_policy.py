"""Train RL communication policy.

Usage:
    python scripts/train_policy.py
    python scripts/train_policy.py --total-timesteps 2000000 --lr 1e-4
"""

from __future__ import annotations

from pathlib import Path

import typer

from omnicopilot.rl.train import TrainConfig, train

app = typer.Typer()


@app.command()
def main(
    total_timesteps: int = 1_000_000,
    lr: float = 3e-4,
    num_envs: int = 4,
    checkpoint_dir: str = "data/models/communication_policy",
    wandb_project: str = "omnicopilot-rl",
) -> None:
    """Train PPO communication policy."""
    config = TrainConfig(
        total_timesteps=total_timesteps,
        learning_rate=lr,
        num_envs=num_envs,
        checkpoint_dir=Path(checkpoint_dir),
        wandb_project=wandb_project,
    )
    train(config)


if __name__ == "__main__":
    app()

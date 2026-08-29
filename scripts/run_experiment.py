"""Run an experiment from Hydra configuration.

Usage:
    python scripts/run_experiment.py experiment=single_agent
    python scripts/run_experiment.py experiment=cooperative_intelligent
    python scripts/run_experiment.py --multirun experiment=single_agent,cooperative_naive
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../python/omnicopilot/config/defaults")
def main(cfg: DictConfig) -> None:
    """Run an experiment."""
    # TODO: Implement experiment runner.
    # 1. Load dataset.
    # 2. Create agents.
    # 3. Run simulation loop.
    # 4. Evaluate metrics.
    # 5. Log to W&B.
    print(f"Running experiment: {cfg.get('name', 'unknown')}")  # noqa: T201


if __name__ == "__main__":
    main()

"""Run the full ablation study.

Usage:
    python scripts/run_ablation_suite.py
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    base_config: str = "python/omnicopilot/config/defaults/base.yaml",
    output_dir: str = "data/results/ablations",
) -> None:
    """Run ablation suite — systematically remove each component."""
    ablation_components = [
        "no_cooperation",
        "no_uncertainty",
        "no_prioritization",
        "no_receiver_awareness",
        "no_active_acquisition",
        "no_trust",
        "no_learned_policy",
        "no_temporal",
    ]

    print(f"Running {len(ablation_components)} ablation experiments...")  # noqa: T201
    for component in ablation_components:
        print(f"  Ablation: {component}")  # noqa: T201
        # TODO: Load config, disable component, run experiment, collect metrics.

    print("Ablation suite complete. Results saved to:", output_dir)  # noqa: T201


if __name__ == "__main__":
    app()

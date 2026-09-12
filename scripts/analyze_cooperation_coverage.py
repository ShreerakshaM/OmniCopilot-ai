"""Cooperation coverage analysis — the first quantitative result (pre-Kill-Gate-A).

Measures, using GROUND-TRUTH objects, how many more objects the fleet sees collectively
versus any single agent alone. This is the CEILING of cooperative benefit available in
the data (no detection model involved — that comes in Phase 2-3).

For each frame in each scenario:
    single_agent_count = objects visible to the ego agent alone
    collective_count   = unique objects visible to ALL agents combined
                         (matched across agents by world-frame proximity, deduplicated)
    gain               = collective_count / single_agent_count

Aggregated across all frames → the headline: "cooperation reveals X% more objects."

This directly informs Kill Gate A:
    strong gain (e.g. collective >> single)  -> premise holds, proceed
    weak gain  (collective ~= single)        -> investigate before building further

Usage:
    python scripts/analyze_cooperation_coverage.py \
        --data-root /kaggle/input/datasets/<user>/opv2v-test-001/test \
        --out data/results/cooperation_coverage \
        [--max-scenarios N] [--match-tolerance 3.0]

Runs on CPU only. No GPU, no detection model, no OpenCOOD install required.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from omnicopilot.data.opv2v import OPV2VDataset


@dataclass
class FrameCoverage:
    """Coverage stats for a single frame."""

    scenario_id: str
    frame_idx: int
    num_agents: int
    # Per-agent GT object counts.
    per_agent_counts: dict[str, int]
    # Objects seen by the "ego" agent (the one that sees the FEWEST — worst case)
    # and the one that sees the MOST (best single agent).
    min_single_count: int
    max_single_count: int
    mean_single_count: float
    # Unique objects across all agents (matched + deduplicated).
    collective_count: int
    # Gains.
    gain_vs_worst_single: float  # collective / min_single
    gain_vs_best_single: float  # collective / max_single


@dataclass
class CoverageSummary:
    """Aggregate summary across all analyzed frames."""

    data_root: str
    num_scenarios: int
    num_frames: int
    match_tolerance_m: float

    mean_agents_per_frame: float
    mean_single_count: float
    mean_collective_count: float

    # THE headline numbers.
    mean_gain_vs_worst_single: float
    mean_gain_vs_best_single: float
    median_gain_vs_best_single: float

    # Distribution of extra objects revealed by cooperation.
    mean_extra_objects_vs_best_single: float

    # Best cooperation scenes (biggest gain) — demo candidates.
    top_cooperation_scenes: list[dict] = field(default_factory=list)


def analyze_frame(
    dataset: OPV2VDataset,
    scenario_id: str,
    frame_idx: int,
    match_tolerance_m: float,
) -> FrameCoverage | None:
    """Compute coverage stats for one frame."""
    frames = dataset.get_all_agent_frames(scenario_id, frame_idx, load_lidar=False)
    if len(frames) < 2:
        return None  # cooperation is undefined with <2 agents

    per_agent_counts = {aid: len(f.gt_objects) for aid, f in frames.items()}
    counts = list(per_agent_counts.values())
    if max(counts) == 0:
        return None  # empty frame

    # Collective unique objects = number of match groups across agents.
    groups = OPV2VDataset.match_objects_across_agents(frames, tolerance_m=match_tolerance_m)
    collective_count = len(groups)

    min_single = min(counts)
    max_single = max(counts)

    return FrameCoverage(
        scenario_id=scenario_id,
        frame_idx=frame_idx,
        num_agents=len(frames),
        per_agent_counts=per_agent_counts,
        min_single_count=min_single,
        max_single_count=max_single,
        mean_single_count=float(statistics.mean(counts)),
        collective_count=collective_count,
        gain_vs_worst_single=(collective_count / min_single) if min_single > 0 else float("inf"),
        gain_vs_best_single=(collective_count / max_single) if max_single > 0 else float("inf"),
    )


def run_analysis(
    data_root: Path,
    out_dir: Path,
    max_scenarios: int | None,
    match_tolerance_m: float,
) -> CoverageSummary:
    """Run the full cooperation coverage analysis."""
    dataset = OPV2VDataset(data_root)
    dataset.load()

    scenario_ids = dataset.scenario_ids()
    if max_scenarios is not None:
        scenario_ids = scenario_ids[:max_scenarios]

    if not scenario_ids:
        msg = f"No scenarios found under {data_root}"
        raise RuntimeError(msg)

    all_frames: list[FrameCoverage] = []

    for sid in scenario_ids:
        scenario = dataset.get_scenario(sid)
        for frame_idx in scenario.frame_indices:
            fc = analyze_frame(dataset, sid, frame_idx, match_tolerance_m)
            if fc is not None:
                all_frames.append(fc)

    if not all_frames:
        msg = "No valid multi-agent frames found (need >=2 agents with objects)."
        raise RuntimeError(msg)

    # Aggregate (exclude inf gains from empty-single edge cases).
    finite = [f for f in all_frames if np.isfinite(f.gain_vs_best_single)]

    gains_best = [f.gain_vs_best_single for f in finite]
    gains_worst = [f.gain_vs_worst_single for f in finite if np.isfinite(f.gain_vs_worst_single)]
    extra = [f.collective_count - f.max_single_count for f in finite]

    # Top cooperation scenes (biggest gain over the BEST single agent).
    top = sorted(finite, key=lambda f: f.gain_vs_best_single, reverse=True)[:10]
    top_scenes = [
        {
            "scenario_id": f.scenario_id,
            "frame_idx": f.frame_idx,
            "num_agents": f.num_agents,
            "best_single_count": f.max_single_count,
            "collective_count": f.collective_count,
            "gain": round(f.gain_vs_best_single, 3),
            "extra_objects": f.collective_count - f.max_single_count,
        }
        for f in top
    ]

    summary = CoverageSummary(
        data_root=str(data_root),
        num_scenarios=len(scenario_ids),
        num_frames=len(all_frames),
        match_tolerance_m=match_tolerance_m,
        mean_agents_per_frame=float(statistics.mean(f.num_agents for f in all_frames)),
        mean_single_count=float(statistics.mean(f.max_single_count for f in all_frames)),
        mean_collective_count=float(statistics.mean(f.collective_count for f in all_frames)),
        mean_gain_vs_worst_single=float(statistics.mean(gains_worst)) if gains_worst else 0.0,
        mean_gain_vs_best_single=float(statistics.mean(gains_best)) if gains_best else 0.0,
        median_gain_vs_best_single=float(statistics.median(gains_best)) if gains_best else 0.0,
        mean_extra_objects_vs_best_single=float(statistics.mean(extra)) if extra else 0.0,
        top_cooperation_scenes=top_scenes,
    )

    # Persist results.
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "coverage_summary.json", "w") as f:
        json.dump(asdict(summary), f, indent=2)
    with open(out_dir / "per_frame_coverage.json", "w") as f:
        json.dump([asdict(fc) for fc in all_frames], f, indent=2)

    # Optional plot (skip gracefully if matplotlib unavailable).
    try:
        _plot(all_frames, out_dir)
    except ImportError:
        pass

    return summary


def _plot(frames: list[FrameCoverage], out_dir: Path) -> None:
    """Plot the distribution of cooperative gain."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = [f.gain_vs_best_single for f in frames if np.isfinite(f.gain_vs_best_single)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(gains, bins=30, edgecolor="black")
    ax[0].axvline(1.0, color="red", linestyle="--", label="no gain")
    ax[0].set_xlabel("collective / best-single object count")
    ax[0].set_ylabel("frames")
    ax[0].set_title("Cooperative coverage gain (vs best single agent)")
    ax[0].legend()

    single = [f.max_single_count for f in frames]
    collective = [f.collective_count for f in frames]
    ax[1].scatter(single, collective, alpha=0.3, s=10)
    lim = max(max(single, default=1), max(collective, default=1))
    ax[1].plot([0, lim], [0, lim], "r--", label="y=x (no gain)")
    ax[1].set_xlabel("best single-agent object count")
    ax[1].set_ylabel("collective object count")
    ax[1].set_title("Single vs. collective coverage")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "cooperation_coverage.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to OPV2V split dir (e.g. .../opv2v-test-001/test)")
    parser.add_argument("--out", type=Path, default=Path("data/results/cooperation_coverage"))
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit number of scenarios (for a quick run)")
    parser.add_argument("--match-tolerance", type=float, default=3.0,
                        help="World-frame distance (m) to consider two detections the same object")
    args = parser.parse_args()

    summary = run_analysis(args.data_root, args.out, args.max_scenarios, args.match_tolerance)

    # Print the headline.
    print("=" * 70)  # noqa: T201
    print("COOPERATION COVERAGE ANALYSIS — first quantitative result")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print(f"Scenarios analyzed:        {summary.num_scenarios}")  # noqa: T201
    print(f"Frames analyzed:           {summary.num_frames}")  # noqa: T201
    print(f"Mean agents per frame:     {summary.mean_agents_per_frame:.2f}")  # noqa: T201
    print(f"Mean objects (best single):{summary.mean_single_count:.2f}")  # noqa: T201
    print(f"Mean objects (collective): {summary.mean_collective_count:.2f}")  # noqa: T201
    print("-" * 70)  # noqa: T201
    print(f"HEADLINE — mean coverage gain vs best single agent: "  # noqa: T201
          f"{summary.mean_gain_vs_best_single:.2f}x "
          f"(+{summary.mean_extra_objects_vs_best_single:.1f} objects/frame)")
    print(f"          median gain: {summary.median_gain_vs_best_single:.2f}x")  # noqa: T201
    print(f"          gain vs WORST single agent: "  # noqa: T201
          f"{summary.mean_gain_vs_worst_single:.2f}x")
    print("-" * 70)  # noqa: T201
    print("Kill Gate A read: a gain well above 1.0x means cooperation has strong")  # noqa: T201
    print("headroom on this data. A gain near 1.0x means investigate before building.")  # noqa: T201
    print(f"\nResults written to: {args.out}")  # noqa: T201
    print("Top cooperation scenes (demo candidates):")  # noqa: T201
    for s in summary.top_cooperation_scenes[:5]:
        print(f"  {s['scenario_id']} frame {s['frame_idx']}: "  # noqa: T201
              f"{s['best_single_count']} -> {s['collective_count']} "
              f"({s['gain']}x, +{s['extra_objects']} objects)")


if __name__ == "__main__":
    main()

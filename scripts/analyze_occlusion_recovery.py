"""Occluded-object recovery — the honest, paper-aligned cooperation metric.

> NOTE ON A DEGENERATE RESULT: run on ground-truth objects, this metric returns
> ~100% recovery by construction. "Missed" is defined as "an object some OTHER agent
> sees but this ego does not" -- so every missed object is, by definition, seen by the
> fleet, making recovered/missed always ~1.0. That is the trivial GT ceiling and is NOT
> a meaningful measure of cooperation. The USEFUL quantity it does produce is
> `mean_missed_per_ego` (~3.4 objects) -- how many blind-spot objects an agent has for
> cooperation to address. For the real cooperative result under detection uncertainty,
> see scripts/analyze_detection_cooperation.py and docs/results/03_detection_cooperation.md
> (recall 0.45 -> 0.70 through fusion). This script is kept for the blind-spot-count
> statistic and as the basis for an ACHIEVABLE recovery metric once real/simulated
> detections replace GT.

Raw object-count gain (Result 01) is dominated by cars every agent can already
see, so it understates cooperation's real value and is sensitive to how counts are
deduplicated. This metric isolates what cooperation actually fixes: BLIND SPOTS.

For each candidate ego agent in a frame:
  missed    = distinct fleet objects this ego does NOT see on its own (occluded /
              out of range / out of FOV), established from the multi-agent GT.
  recovered = of those missed objects, how many the fleet (cooperation) provides.
  recovery_rate = recovered / missed.

This is the OPV2V paper's core argument (cooperation resolves occlusion), measured
directly. It compares like-with-like (blind-spot objects), avoiding the raw-count
ratio pitfall where a deduplicated collective is compared to a non-deduplicated
single-agent list.

Usage:
    python scripts/analyze_occlusion_recovery.py \
        --data-root /kaggle/input/.../opv2v-2/test \
        --tolerance 2.5 --frame-stride 10 --min-agents 2 \
        --out /kaggle/working/results/occlusion_recovery.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if _PY_ROOT.is_dir() and str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))


def run(
    data_root: Path,
    tolerance_m: float,
    frame_stride: int,
    min_agents: int,
    max_scenarios: int | None,
) -> dict:
    """Aggregate occluded-object recovery across sampled multi-agent frames."""
    from omnicopilot.data.opv2v import OPV2VDataset
    from omnicopilot.perception.occlusion import frame_recovery

    ds = OPV2VDataset(data_root)
    ds.load()
    sids = ds.scenario_ids()
    if max_scenarios is not None:
        sids = sids[:max_scenarios]

    per_agentcount: dict[int, list[float]] = defaultdict(list)
    all_rates: list[float] = []
    all_missed: list[float] = []
    all_recovered: list[float] = []
    n_frames = 0

    for sid in sids:
        sc = ds.get_scenario(sid)
        if len(sc.agent_ids) < min_agents:
            continue
        for fi in sc.frame_indices[::frame_stride]:
            frames = ds.get_all_agent_frames(sid, fi, load_lidar=False)
            centers = {aid: f.gt_locations_world() for aid, f in frames.items()}
            stats = frame_recovery(centers, tolerance_m=tolerance_m)
            if stats.num_agents < min_agents or stats.mean_missed == 0.0:
                # Skip frames where a single agent already sees everything (no blind
                # spots to recover) -- they carry no information about recovery.
                if stats.num_agents >= min_agents:
                    per_agentcount[stats.num_agents].append(
                        stats.recovery_rate
                    )  # rate is 0 with 0 missed; keep for honesty
                continue
            all_rates.append(stats.recovery_rate)
            all_missed.append(stats.mean_missed)
            all_recovered.append(stats.mean_recovered)
            per_agentcount[stats.num_agents].append(stats.recovery_rate)
            n_frames += 1

    def _mean(xs: list[float]) -> float:
        return round(statistics.mean(xs), 3) if xs else 0.0

    by_agents = {
        str(k): {"mean_recovery_rate": _mean(v), "frames": len(v)}
        for k, v in sorted(per_agentcount.items())
    }

    summary = {
        "data_root": str(data_root),
        "tolerance_m": tolerance_m,
        "frames_with_blind_spots": n_frames,
        "mean_recovery_rate": _mean(all_rates),
        "median_recovery_rate": round(statistics.median(all_rates), 3) if all_rates else 0.0,
        "mean_missed_per_ego": _mean(all_missed),
        "mean_recovered_per_ego": _mean(all_recovered),
        "recovery_rate_by_agent_count": by_agents,
    }
    return summary


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--tolerance",
        type=float,
        default=2.5,
        help="Cross-agent matching tolerance (m); use the sweep knee",
    )
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--min-agents", type=int, default=2)
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    summary = run(
        args.data_root, args.tolerance, args.frame_stride, args.min_agents, args.max_scenarios
    )

    print("=" * 66)
    print("OCCLUDED-OBJECT RECOVERY — cooperation's blind-spot value")
    print("=" * 66)
    print(f"Frames with blind spots:    {summary['frames_with_blind_spots']}")
    print(f"Matching tolerance:         {summary['tolerance_m']} m")
    print(f"Mean objects an ego MISSES alone:   {summary['mean_missed_per_ego']}")
    print(f"Mean of those RECOVERED by fleet:   {summary['mean_recovered_per_ego']}")
    print("-" * 66)
    print(
        f"HEADLINE — mean occluded-object recovery rate: "
        f"{summary['mean_recovery_rate']:.1%} "
        f"(median {summary['median_recovery_rate']:.1%})"
    )
    print("-" * 66)
    print("Recovery rate by agent count:")
    for k, v in summary["recovery_rate_by_agent_count"].items():
        print(f"  {k} agents: {v['mean_recovery_rate']:.1%}  ({v['frames']} frames)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()

"""Phase 4 (Task 4.4) — accuracy vs. bandwidth: does smart prioritization pay off?

Cooperation helps (Result 03), but sharing EVERY detection saturates a real V2X channel.
This measures the accuracy-bandwidth trade-off: under a per-agent transmission budget,
how much cooperative detection quality (mAP) does each policy retain?

Policies compared (each decides WHICH detections an agent transmits under the budget):
  - send_all   : transmit everything (ignores budget) — the upper-bound baseline.
  - random     : keep a random subset within budget — naive baseline.
  - prioritized: the hand-designed C++ Prioritizer (safety x confidence x novelty x
                 time-sensitivity) selects top-K within budget — the smart policy.

Only transmitted detections reach the C++ WorldModel for fusion; suppressed ones are
dropped (as they would be on a congested channel). We sweep the budget (as a fraction of
the detections an agent would send) and plot mAP vs. bandwidth.

MILESTONE (plan): "same perception accuracy at ~40% less bandwidth."

HONEST CAVEAT: simulated detector (GT + calibrated noise), not a CNN. The prioritizer
and budgeting are the REAL C++ components (via pybind), so this evaluates the actual
Phase-4 logic, not a Python re-implementation.

Usage:
    python scripts/analyze_communication_bandwidth.py \
        --data-root /kaggle/input/.../opv2v-2/test \
        --module-dir build/cpp --assoc-radius 2.5 --frame-stride 10 --min-agents 2 \
        --out /kaggle/working/results/communication_bandwidth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if _PY_ROOT.is_dir() and str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

BYTES_PER_OBS = 256  # matches the Prioritizer default serialized-observation size


def import_cpp(module_dir: str | None) -> Any:
    if module_dir:
        sys.path.insert(0, module_dir)
    import _omnicopilot_cpp as oc  # noqa: PLC0415
    if not hasattr(oc, "Prioritizer"):
        msg = "Built module has no Prioritizer binding — rebuild (cmake --build build)."
        raise RuntimeError(msg)
    return oc


def _obs(oc: Any, aid: str, oid: str, d: Any, t: float) -> Any:
    o = oc.Observation()
    o.observation_id = oid
    o.agent_id = aid
    o.object_class = oc.ObjectClass.VEHICLE
    o.position.x, o.position.y, o.position.z = (float(d.position[0]),
                                                float(d.position[1]),
                                                float(d.position[2]))
    o.confidence = float(d.confidence)
    o.velocity.vx, o.velocity.vy, o.velocity.vz = 0.0, 0.0, 0.0
    o.timestamp_s = t
    return o


def _dedupe(centers: np.ndarray, tol: float = 2.0) -> np.ndarray:
    kept: list[np.ndarray] = []
    for c in centers:
        if all(np.linalg.norm(c[:2] - k[:2]) > tol for k in kept):
            kept.append(c)
    return np.array(kept) if kept else np.zeros((0, 3))


def _select(oc: Any, policy: str, obs_list: list, budget_frac: float,
            prioritizer: Any, rng: np.random.Generator) -> list:
    """Return the subset of this agent's observations transmitted under the policy."""
    n = len(obs_list)
    if n == 0:
        return []
    if policy == "send_all":
        return obs_list
    budget_count = max(1, int(np.ceil(budget_frac * n)))
    if policy == "random":
        idx = rng.choice(n, size=min(budget_count, n), replace=False)
        return [obs_list[i] for i in idx]
    if policy == "prioritized":
        budget_bytes = budget_count * BYTES_PER_OBS
        res = prioritizer.prioritize(obs_list, budget_bytes, BYTES_PER_OBS)
        return [s.observation for s in res.selected]
    msg = f"unknown policy {policy}"
    raise ValueError(msg)


def run(oc: Any, data_root: Path, assoc_radius: float, frame_stride: int,
        min_agents: int, seed: int, budgets: list[float], max_scenarios: int | None) -> dict:
    from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415
    from omnicopilot.evaluation.metrics import compute_map  # noqa: PLC0415
    from omnicopilot.perception.simulated_detector import (  # noqa: PLC0415
        DetectorNoiseConfig, SimulatedDetector,
    )

    ds = OPV2VDataset(data_root)
    ds.load()
    sids = ds.scenario_ids()
    if max_scenarios is not None:
        sids = sids[:max_scenarios]

    detector = SimulatedDetector(DetectorNoiseConfig(), seed=seed)
    prioritizer = oc.Prioritizer(oc.PrioritizationWeights())
    rng = np.random.default_rng(seed)

    # Pre-compute per-frame detections once (same inputs for every policy/budget).
    frames_data = []  # list of (per_agent_obs: dict[aid->list[Detection3D]], gt7)
    for sid in sids:
        sc = ds.get_scenario(sid)
        if len(sc.agent_ids) < min_agents:
            continue
        for fi in sc.frame_indices[::frame_stride]:
            frames = ds.get_all_agent_frames(sid, fi, load_lidar=False)
            all_c = [fr.gt_locations_world() for fr in frames.values()]
            gt_union = _dedupe(np.vstack(all_c)) if all_c else np.zeros((0, 3))
            if len(gt_union) == 0:
                continue
            gt7 = np.hstack([gt_union, np.tile([4.5, 2.0, 1.5, 0.0], (len(gt_union), 1))])
            per_agent = {}
            for aid, fr in frames.items():
                centers = fr.gt_locations_world()
                dims = np.array([o.dimensions for o in fr.gt_objects]) if fr.gt_objects \
                    else np.zeros((0, 3))
                yaws = np.array([o.yaw_rad for o in fr.gt_objects]) if fr.gt_objects \
                    else np.zeros(0)
                sxyz = np.asarray(fr.pose)[:3, 3] if np.asarray(fr.pose).shape == (4, 4) \
                    else np.zeros(3)
                per_agent[aid] = detector.detect(centers, dims, yaws, sxyz)
            frames_data.append((per_agent, gt7))

    def score_policy(policy: str, budget_frac: float) -> dict:
        preds: list[np.ndarray] = []
        gts: list[np.ndarray] = []
        total_tx = 0
        total_avail = 0
        for per_agent, gt7 in frames_data:
            cfg = oc.WorldModelConfig()
            cfg.max_entities = 2000
            cfg.fusion.association_max_distance_m = assoc_radius
            wm = oc.WorldModel(cfg)
            step = 1.0
            for aid, dets in per_agent.items():
                obs_list = [_obs(oc, aid, f"{aid}_{j}", d, step) for j, d in enumerate(dets)]
                total_avail += len(obs_list)
                tx = _select(oc, policy, obs_list, budget_frac, prioritizer, rng)
                total_tx += len(tx)
                for o in tx:
                    wm.ingest_observations([o], 0.9)
            wm.tick(step + 0.1)
            rows = [[e.position.x, e.position.y, e.position.z, 4.5, 2.0, 1.5, 0.0,
                     float(e.confidence)] for e in wm.get_entities()]
            preds.append(np.array(rows) if rows else np.zeros((0, 8)))
            gts.append(gt7)
        m = compute_map(preds, gts, iou_thresholds=np.array([0.5, 0.7]))
        return {
            "mAP": round(m.mean_average_precision, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "bandwidth_frac": round(total_tx / max(total_avail, 1), 4),
        }

    results = {"send_all": {}, "random": {}, "prioritized": {}}
    # send_all is budget-independent (upper bound) — compute once.
    results["send_all"]["1.0"] = score_policy("send_all", 1.0)
    for b in budgets:
        results["random"][str(b)] = score_policy("random", b)
        results["prioritized"][str(b)] = score_policy("prioritized", b)

    return {
        "frames": len(frames_data),
        "assoc_radius_m": assoc_radius,
        "budgets": budgets,
        "results": results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--module-dir", type=str, default=None)
    p.add_argument("--assoc-radius", type=float, default=2.5)
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--min-agents", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budgets", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    oc = import_cpp(args.module_dir)
    r = run(oc, args.data_root, args.assoc_radius, args.frame_stride, args.min_agents,
            args.seed, args.budgets, args.max_scenarios)

    print("=" * 70)
    print("PHASE 4 — ACCURACY vs. BANDWIDTH (hand-designed prioritization)")
    print("=" * 70)
    print(f"Frames: {r['frames']}  | assoc radius {r['assoc_radius_m']} m")
    sa = r["results"]["send_all"]["1.0"]
    print(f"\nsend_all (100% bandwidth): mAP {sa['mAP']}  precision {sa['precision']}  "
          f"recall {sa['recall']}")
    print(f"\n{'budget':>8} {'policy':>12} {'actual_bw':>10} {'mAP':>8} "
          f"{'precision':>10} {'recall':>8}")
    for b in r["budgets"]:
        for pol in ("random", "prioritized"):
            d = r["results"][pol][str(b)]
            print(f"{b:>8} {pol:>12} {d['bandwidth_frac']:>10} {d['mAP']:>8} "
                  f"{d['precision']:>10} {d['recall']:>8}")
    print("\nRead: at each budget, 'prioritized' mAP should beat 'random' and retain "
          "most of send_all's mAP -> smart comms holds accuracy at lower bandwidth.")
    print("NOTE: simulated detector; prioritizer/budget are the real C++ components.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()

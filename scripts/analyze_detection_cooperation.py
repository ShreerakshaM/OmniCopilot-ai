"""Phase 2 (2b): cooperative gain under SIMULATED detection uncertainty.

Pipeline per frame:
  GT objects (world) --> SimulatedDetector (per agent, range/occlusion dropout +
  localization noise + false positives) --> per-agent detections
    (a) SINGLE-AGENT: evaluate the best single agent's detections vs. GT (AP).
    (b) COOPERATIVE: feed ALL agents' detections through the C++ WorldModel,
        take the fused entities as the cooperative "detections", evaluate vs. GT (AP).
  Report single-agent AP vs. cooperative AP -- the achievable gain, through OUR
  fusion, the same single-vs-cooperative comparison the OPV2V paper reports (as AP),
  but measured on our system.

HONEST CAVEAT (must appear in any result): detections are ground truth + a
literature-calibrated noise/dropout model (V2X-ViT noise levels), NOT a neural
network. A real detector (2a) is a drop-in replacement, blocked here only by the
Kaggle Python-3.12 vs. spconv/mmcv incompatibility.

Usage:
    python scripts/analyze_detection_cooperation.py \
        --data-root /kaggle/input/.../opv2v-2/test \
        --module-dir build/cpp \
        --assoc-radius 2.5 --frame-stride 10 --min-agents 2 --seed 0 \
        --out /kaggle/working/results/detection_cooperation.json
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


def import_cpp(module_dir: str | None) -> Any:
    """Import the compiled C++ world-model module."""
    if module_dir:
        sys.path.insert(0, module_dir)
    import _omnicopilot_cpp as oc

    return oc


def gt_boxes_for_frame(frames: dict) -> np.ndarray:
    """Union of all agents' GT as (K,7) x,y,z,l,w,h,yaw, deduplicated by proximity."""
    from omnicopilot.data.opv2v import OPV2VDataset

    groups = OPV2VDataset.match_objects_across_agents(frames, tolerance_m=2.0)
    # For each matched group, take one representative box (from the first agent in it).
    boxes = []
    # Build a lookup: (agent, object_key) -> (center, dims, yaw)
    obj_lookup = {}
    for aid, fr in frames.items():
        world = fr.gt_locations_world()
        for o, w in zip(fr.gt_objects, world):
            obj_lookup[(aid, o.object_key)] = (w, o.dimensions, o.yaw_rad)
    for g in groups:
        aid = next(iter(g))
        w, dims, yaw = obj_lookup[(aid, g[aid])]
        boxes.append([w[0], w[1], w[2], dims[0], dims[1], dims[2], yaw])
    return np.array(boxes, dtype=np.float64) if boxes else np.zeros((0, 7))


def detections_to_pred_array(dets: list) -> np.ndarray:
    """(N,8) x,y,z,l,w,h,heading,score from Detection3D list."""
    from omnicopilot.perception.detector import OpenCOODDetector

    return OpenCOODDetector.detections_to_boxes(dets)


def run(
    oc: Any,
    data_root: Path,
    assoc_radius: float,
    frame_stride: int,
    min_agents: int,
    seed: int,
    max_scenarios: int | None,
) -> dict:
    """Compute single-agent vs cooperative AP across sampled frames."""
    from omnicopilot.data.opv2v import OPV2VDataset
    from omnicopilot.evaluation.metrics import compute_map
    from omnicopilot.perception.simulated_detector import (
        DetectorNoiseConfig,
        SimulatedDetector,
    )

    ds = OPV2VDataset(data_root)
    ds.load()
    sids = ds.scenario_ids()
    if max_scenarios is not None:
        sids = sids[:max_scenarios]

    detector = SimulatedDetector(DetectorNoiseConfig(), seed=seed)

    single_preds: list[np.ndarray] = []
    coop_preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    by_agent_count: dict[int, dict[str, list[np.ndarray]]] = {}
    n_frames = 0

    for sid in sids:
        sc = ds.get_scenario(sid)
        if len(sc.agent_ids) < min_agents:
            continue
        for fi in sc.frame_indices[::frame_stride]:
            frames = ds.get_all_agent_frames(sid, fi, load_lidar=False)
            gt = gt_boxes_for_frame(frames)
            if gt.shape[0] == 0:
                continue

            # Per-agent simulated detections (agent world pos from its pose translation).
            per_agent_dets = {}
            for aid, fr in frames.items():
                centers = fr.gt_locations_world()
                dims = (
                    np.array([o.dimensions for o in fr.gt_objects], dtype=np.float64)
                    if fr.gt_objects
                    else np.zeros((0, 3))
                )
                yaws = (
                    np.array([o.yaw_rad for o in fr.gt_objects], dtype=np.float64)
                    if fr.gt_objects
                    else np.zeros(0)
                )
                sensor_xyz = (
                    np.asarray(fr.pose, dtype=np.float64)[:3, 3]
                    if np.asarray(fr.pose).shape == (4, 4)
                    else np.zeros(3)
                )
                per_agent_dets[aid] = detector.detect(centers, dims, yaws, sensor_xyz)

            # (a) SINGLE-AGENT baseline: the agent that detected the most objects.
            best_aid = max(per_agent_dets, key=lambda a: len(per_agent_dets[a]))
            single_preds.append(detections_to_pred_array(per_agent_dets[best_aid]))

            # (b) COOPERATIVE: feed all agents' detections into the C++ world model.
            cfg = oc.WorldModelConfig()
            cfg.max_entities = 2000
            cfg.fusion.association_max_distance_m = assoc_radius
            wm = oc.WorldModel(cfg)
            step_t = 1.0
            for aid, dets in per_agent_dets.items():
                for j, d in enumerate(dets):
                    obs = oc.Observation()
                    obs.observation_id = f"{aid}_{j}"
                    obs.agent_id = aid
                    obs.object_class = oc.ObjectClass.VEHICLE
                    obs.position.x = float(d.position[0])
                    obs.position.y = float(d.position[1])
                    obs.position.z = float(d.position[2])
                    obs.confidence = float(d.confidence)
                    obs.timestamp_s = step_t
                    wm.ingest_observations([obs], 0.9)
            wm.tick(step_t + 0.1)

            # Fused entities -> prediction array. Use default car dims + entity conf.
            fused = wm.get_entities()
            rows = []
            for e in fused:
                rows.append(
                    [
                        e.position.x,
                        e.position.y,
                        e.position.z,
                        4.5,
                        2.0,
                        1.5,
                        0.0,
                        float(e.confidence),
                    ]
                )
            coop_preds.append(np.array(rows, dtype=np.float64) if rows else np.zeros((0, 8)))
            gts.append(gt)
            bucket = by_agent_count.setdefault(
                len(frames), {"single": [], "cooperative": [], "ground_truth": []}
            )
            bucket["single"].append(single_preds[-1])
            bucket["cooperative"].append(coop_preds[-1])
            bucket["ground_truth"].append(gt)
            n_frames += 1

    thr = np.array([0.5, 0.7])
    single_m = compute_map(single_preds, gts, iou_thresholds=thr)
    coop_m = compute_map(coop_preds, gts, iou_thresholds=thr)

    breakdown = {}
    for agent_count, bucket in sorted(by_agent_count.items()):
        single_count = compute_map(bucket["single"], bucket["ground_truth"], iou_thresholds=thr)
        coop_count = compute_map(bucket["cooperative"], bucket["ground_truth"], iou_thresholds=thr)
        breakdown[str(agent_count)] = {
            "frames": len(bucket["ground_truth"]),
            "single_agent_map": round(single_count.mean_average_precision, 4),
            "cooperative_map": round(coop_count.mean_average_precision, 4),
            "absolute_map_gain": round(
                coop_count.mean_average_precision - single_count.mean_average_precision, 4
            ),
            "single_agent_recall": round(single_count.recall, 4),
            "cooperative_recall": round(coop_count.recall, 4),
        }

    return {
        "data_root": str(data_root),
        "assoc_radius_m": assoc_radius,
        "seed": seed,
        "frames": n_frames,
        "detector": "SimulatedDetector (GT + V2X-ViT-calibrated noise/dropout) "
        "-- NOT a neural network",
        "single_agent": {
            "mAP@[0.5,0.7]": round(single_m.mean_average_precision, 4),
            "precision": round(single_m.precision, 4),
            "recall": round(single_m.recall, 4),
        },
        "cooperative": {
            "mAP@[0.5,0.7]": round(coop_m.mean_average_precision, 4),
            "precision": round(coop_m.precision, 4),
            "recall": round(coop_m.recall, 4),
        },
        "absolute_map_gain": round(
            coop_m.mean_average_precision - single_m.mean_average_precision, 4
        ),
        "agent_count_breakdown": breakdown,
    }


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--module-dir", type=str, default=None)
    p.add_argument("--assoc-radius", type=float, default=2.5)
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--min-agents", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    oc = import_cpp(args.module_dir)
    summary = run(
        oc,
        args.data_root,
        args.assoc_radius,
        args.frame_stride,
        args.min_agents,
        args.seed,
        args.max_scenarios,
    )

    print("=" * 68)
    print("PHASE 2 (2b) — COOPERATIVE GAIN UNDER SIMULATED DETECTION")
    print("=" * 68)
    print(f"Detector: {summary['detector']}")
    print(f"Frames: {summary['frames']}  | assoc radius: {summary['assoc_radius_m']} m")
    print("-" * 68)
    s, c = summary["single_agent"], summary["cooperative"]
    print(f"{'':22} {'mAP':>8} {'precision':>10} {'recall':>8}")
    print(
        f"{'Single agent (best)':22} {s['mAP@[0.5,0.7]']:>8} {s['precision']:>10} {s['recall']:>8}"
    )
    print(
        f"{'Cooperative (fused)':22} {c['mAP@[0.5,0.7]']:>8} {c['precision']:>10} {c['recall']:>8}"
    )
    print("-" * 68)
    print(f"Absolute mAP gain from cooperation: {summary['absolute_map_gain']:+.4f}")
    print("\nPer-agent-count saturation:")
    print(
        f"{'Agents':>8} {'Frames':>8} {'Single mAP':>12} "
        f"{'Coop mAP':>10} {'Gain':>10} {'Coop recall':>12}"
    )
    for count, row in summary["agent_count_breakdown"].items():
        print(
            f"{count:>8} {row['frames']:>8} {row['single_agent_map']:>12.4f} "
            f"{row['cooperative_map']:>10.4f} {row['absolute_map_gain']:>+10.4f} "
            f"{row['cooperative_recall']:>12.4f}"
        )
    print("NOTE: simulated detector (GT + calibrated noise), not a CNN. See script docstring.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()

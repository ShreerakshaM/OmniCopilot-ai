"""Adversarial-agent demo — does the trust layer defend cooperative fusion?

Scenario: a fleet of HONEST agents plus one ADVERSARIAL/faulty agent that fabricates
phantom detections (spoofing) and mislocalizes real objects. We feed many frames through
the trust-enabled C++ WorldModel and show:

  1. The adversary's LEARNED TRUST decays over time (it never corroborates consensus and
     its phantoms get purged as uncorroborated singletons -> RecordIncorrect).
  2. Cooperative detection quality (mAP/precision vs. ground truth) stays clean WITH the
     trust layer, versus degraded WITHOUT it (adversary weighted equally).

This makes the Phase-3 trust layer's value concrete: cooperation is only useful if it is
robust to a bad participant. It runs entirely on the simulated detector + the C++ fusion
engine -- no GPU, no detector install.

HONEST CAVEAT: detections are simulated (GT + calibrated noise); the adversary is a
scripted spoofer, not a learned attacker. This demonstrates the fusion+trust mechanism's
robustness, not adversarial ML.

Usage:
    python scripts/adversarial_agent_demo.py \
        --data-root /kaggle/input/.../opv2v-2/test \
        --module-dir build/cpp --min-agents 3 --seed 0 \
        --out /kaggle/working/results/adversarial_demo.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if _PY_ROOT.is_dir() and str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))


def import_cpp(module_dir: str | None) -> Any:
    if module_dir:
        sys.path.insert(0, module_dir)
    import _omnicopilot_cpp as oc  # noqa: PLC0415
    if not hasattr(oc.WorldModel(oc.WorldModelConfig()), "get_agent_trust"):
        msg = ("The built module has no get_agent_trust -- rebuild after the trust-layer "
               "integration (cmake --build build).")
        raise RuntimeError(msg)
    return oc


def make_obs(oc: Any, agent_id: str, oid: str, xyz, conf: float, t: float) -> Any:
    o = oc.Observation()
    o.observation_id = oid
    o.agent_id = agent_id
    o.object_class = oc.ObjectClass.VEHICLE
    o.position.x, o.position.y, o.position.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    o.confidence = float(conf)
    o.timestamp_s = float(t)
    return o


def adversarial_detections(rng: np.random.Generator, gt_centers: np.ndarray,
                           fixed_phantoms: np.ndarray,
                           sensor_xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Adversary output: FIXED phantom objects at consistent fake locations (a
    persistent spoof), plus a few real objects shifted enough not to corroborate.

    Using FIXED phantom locations (not fresh random scatter each frame) is important:
    a consistent lie forms a stable single-source entity that never gets corroborated
    by honest agents, so it is penalized cleanly -- rather than random scatter that
    creates a churn of transient entities and pollutes the fused set indefinitely.
    """
    fabricated = []
    if len(gt_centers) > 0:
        pick = rng.choice(len(gt_centers), size=min(3, len(gt_centers)), replace=False)
        for i in pick:
            shift = rng.uniform(10.0, 20.0, size=3)
            shift[2] = 0.0
            fabricated.append(gt_centers[i] + shift)
    fabricated = np.array(fabricated, dtype=np.float64) if fabricated else np.zeros((0, 3))
    return np.vstack([fixed_phantoms, fabricated]) if len(fabricated) else fixed_phantoms


def run(oc: Any, data_root: Path, min_agents: int, seed: int,
        assoc_radius: float, max_scenarios: int | None) -> dict:
    from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415
    from omnicopilot.evaluation.metrics import compute_map  # noqa: PLC0415
    from omnicopilot.perception.detector import OpenCOODDetector  # noqa: PLC0415
    from omnicopilot.perception.simulated_detector import (  # noqa: PLC0415
        DetectorNoiseConfig, SimulatedDetector,
    )

    ds = OPV2VDataset(data_root)
    ds.load()

    honest = SimulatedDetector(DetectorNoiseConfig(), seed=seed)
    rng = np.random.default_rng(seed + 999)

    def build_wm(with_trust: bool) -> Any:
        cfg = oc.WorldModelConfig()
        cfg.max_entities = 4000
        cfg.fusion.association_max_distance_m = assoc_radius
        # Faster trust adaptation so decay is visible within one scenario's frames.
        cfg.reliability.trust_ema_alpha = 0.15
        cfg.reliability.calibration_period = 5
        if not with_trust:
            # Freeze trust at its initial value (adversary weighted like everyone).
            cfg.reliability.trust_ema_alpha = 0.0
            cfg.reliability.min_trust = cfg.reliability.initial_trust
            cfg.reliability.max_trust = cfg.reliability.initial_trust
        return oc.WorldModel(cfg)

    # Pick ONE scenario with enough agents and the most frames -- a single coherent
    # scene. Accumulating across DIFFERENT scenarios would be wrong (different world
    # coords, different agent IDs), and the "adversary" must be ONE fixed agent whose
    # trust accumulates over time.
    chosen = None
    for sid in ds.scenario_ids():
        sc = ds.get_scenario(sid)
        if len(sc.agent_ids) >= min_agents:
            if chosen is None or len(sc.frame_indices) > len(
                    ds.get_scenario(chosen).frame_indices):
                chosen = sid
    if chosen is None:
        msg = f"No scenario with >= {min_agents} agents."
        raise RuntimeError(msg)

    sc = ds.get_scenario(chosen)
    adversary_id = sc.agent_ids[-1]          # ONE fixed adversary for the whole run
    honest_ids = [a for a in sc.agent_ids if a != adversary_id]

    # The adversary's FIXED phantom objects: consistent fake locations near the
    # adversary's starting position, established once so the spoof is persistent.
    adv_start = np.zeros(3)
    first_frames = ds.get_all_agent_frames(chosen, sc.frame_indices[0], load_lidar=False)
    if adversary_id in first_frames:
        p = np.asarray(first_frames[adversary_id].pose)
        if p.shape == (4, 4):
            adv_start = p[:3, 3]
    fixed_phantoms = adv_start + rng.uniform(-60.0, 60.0, size=(8, 3))
    fixed_phantoms[:, 2] = 0.0

    wm_trust = build_wm(with_trust=True)
    wm_notrust = build_wm(with_trust=False)

    gts: list[np.ndarray] = []
    coop_trust: list[np.ndarray] = []
    coop_notrust: list[np.ndarray] = []
    adversary_trust_trace: list[float] = []
    honest_trust_trace: list[float] = []

    step_t = 1.0
    for frame_no, fi in enumerate(sc.frame_indices):
        frames = ds.get_all_agent_frames(chosen, fi, load_lidar=False)

        # This frame's GT (union across agents, deduped) for scoring.
        all_centers = [fr.gt_locations_world() for fr in frames.values()]
        gt_union = _dedupe(np.vstack(all_centers)) if all_centers else np.zeros((0, 3))
        gt7 = np.hstack([gt_union, np.tile([4.5, 2.0, 1.5, 0.0], (len(gt_union), 1))]) \
            if len(gt_union) else np.zeros((0, 7))

        # Honest agents' simulated detections; remember sensor positions.
        per_agent = {}
        sensor_positions = {}
        for aid, fr in frames.items():
            sxyz = np.asarray(fr.pose)[:3, 3] if np.asarray(fr.pose).shape == (4, 4) \
                else np.zeros(3)
            sensor_positions[aid] = sxyz
            if aid == adversary_id:
                continue
            centers = fr.gt_locations_world()
            dims = np.array([o.dimensions for o in fr.gt_objects]) if fr.gt_objects \
                else np.zeros((0, 3))
            yaws = np.array([o.yaw_rad for o in fr.gt_objects]) if fr.gt_objects \
                else np.zeros(0)
            per_agent[aid] = honest.detect(centers, dims, yaws, sxyz)

        adv_centers = adversarial_detections(
            rng, gt_union, fixed_phantoms,
            sensor_xyz=sensor_positions.get(adversary_id, (0, 0, 0)))

        step_t += 0.1  # ~10 Hz, realistic frame cadence
        for wm, store in ((wm_trust, coop_trust), (wm_notrust, coop_notrust)):
            for aid, dets in per_agent.items():
                for j, d in enumerate(dets):
                    wm.ingest_observations(
                        [make_obs(oc, aid, f"{aid}_{frame_no}_{j}", d.position,
                                  d.confidence, step_t)], 0.9)
            for j, c in enumerate(adv_centers):
                wm.ingest_observations(
                    [make_obs(oc, adversary_id, f"{adversary_id}_{frame_no}_{j}",
                              c, 0.9, step_t)], 0.9)
            wm.tick(step_t)
            rows = [[e.position.x, e.position.y, e.position.z, 4.5, 2.0, 1.5, 0.0,
                     float(e.confidence)] for e in wm.get_entities()]
            store.append(np.array(rows) if rows else np.zeros((0, 8)))

        gts.append(gt7)
        adversary_trust_trace.append(wm_trust.get_agent_trust(adversary_id))
        honest_trust_trace.append(wm_trust.get_agent_trust(honest_ids[0]))

    thr = np.array([0.5, 0.7])
    m_trust = compute_map(coop_trust, gts, iou_thresholds=thr)
    m_notrust = compute_map(coop_notrust, gts, iou_thresholds=thr)
    honest_final = honest_trust_trace[-1] if honest_trust_trace else None

    return {
        "scenario": chosen,
        "adversary_id": adversary_id,
        "frames": len(gts),
        "adversary_trust_start": round(adversary_trust_trace[0], 4) if adversary_trust_trace else None,
        "adversary_trust_end": round(adversary_trust_trace[-1], 4) if adversary_trust_trace else None,
        "adversary_trust_min": round(min(adversary_trust_trace), 4) if adversary_trust_trace else None,
        "honest_agent_final_trust": round(honest_final, 4) if honest_final is not None else None,
        "with_trust": {"mAP": round(m_trust.mean_average_precision, 4),
                       "precision": round(m_trust.precision, 4),
                       "recall": round(m_trust.recall, 4)},
        "without_trust": {"mAP": round(m_notrust.mean_average_precision, 4),
                          "precision": round(m_notrust.precision, 4),
                          "recall": round(m_notrust.recall, 4)},
        "precision_defended": round(m_trust.precision - m_notrust.precision, 4),
    }


def _dedupe(centers: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """Merge near-duplicate world centers (same physical object across agents)."""
    kept: list[np.ndarray] = []
    for c in centers:
        if all(np.linalg.norm(c[:2] - k[:2]) > tol for k in kept):
            kept.append(c)
    return np.array(kept) if kept else np.zeros((0, 3))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--module-dir", type=str, default=None)
    p.add_argument("--min-agents", type=int, default=3)
    p.add_argument("--assoc-radius", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    oc = import_cpp(args.module_dir)
    r = run(oc, args.data_root, args.min_agents, args.seed, args.assoc_radius,
            args.max_scenarios)

    print("=" * 68)
    print("ADVERSARIAL-AGENT DEMO — does the trust layer defend fusion?")
    print("=" * 68)
    print(f"Frames: {r['frames']}  (1 adversary + honest agents per frame)")
    print("-" * 68)
    print(f"Adversary trust:  start {r['adversary_trust_start']}  ->  "
          f"end {r['adversary_trust_end']}  (min {r['adversary_trust_min']})")
    print(f"Honest agent final trust: {r['honest_agent_final_trust']}")
    print("-" * 68)
    print(f"{'':16} {'mAP':>8} {'precision':>10} {'recall':>8}")
    w, wo = r["with_trust"], r["without_trust"]
    print(f"{'WITH trust':16} {w['mAP']:>8} {w['precision']:>10} {w['recall']:>8}")
    print(f"{'WITHOUT trust':16} {wo['mAP']:>8} {wo['precision']:>10} {wo['recall']:>8}")
    print("-" * 68)
    print(f"Precision defended by trust layer: {r['precision_defended']:+.4f}")
    print("NOTE: simulated detector + scripted spoofer, not adversarial ML.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()

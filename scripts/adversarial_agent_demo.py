"""Adversarial-agent demo — does the trust layer defend cooperative fusion?

Scenario: a fleet of HONEST agents plus one ADVERSARIAL/faulty agent that mislocalizes
objects and fabricates phantoms. Two-pass design (the two concerns must be separated):

  PASS 1 — LEARN TRUST: one PERSISTENT world model over all frames of a single scenario.
           The adversary associates to real entities but reports them OFF (disagrees with
           consensus) and fabricates uncorroborated phantoms, so its learned trust decays,
           while honest agents (who agree with consensus) keep high trust.

  PASS 2 — SCORE AP: a FRESH world model PER FRAME (like Result 03, so scoring is clean),
           seeding each agent's trust from Pass 1. WITH-trust uses the learned trust
           (adversary down-weighted); WITHOUT-trust uses a flat prior (adversary equal).
           Compare cooperative mAP/precision WITH vs. WITHOUT the trust layer.

A persistent model is needed to LEARN trust; a fresh-per-frame model is needed to SCORE
cleanly (a persistent model accumulates stale entities and destroys precision). Doing both
in one pass — the earlier mistake — cannot work.

HONEST CAVEAT: detections are simulated (GT + calibrated noise); the adversary is a
scripted spoofer, not a learned attacker. This shows the fusion+trust mechanism's
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
    if not hasattr(oc.WorldModelConfig().reliability, "agreement_radius_m"):
        msg = ("The built module has no reliability.agreement_radius_m -- rebuild after "
               "the consensus-gated trust fix.")
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


def _dedupe(centers: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """Merge near-duplicate world centers (same physical object across agents)."""
    kept: list[np.ndarray] = []
    for c in centers:
        if all(np.linalg.norm(c[:2] - k[:2]) > tol for k in kept):
            kept.append(c)
    return np.array(kept) if kept else np.zeros((0, 3))


def adversary_detections(gt_centers: np.ndarray, offset_m: float = 5.0) -> np.ndarray:
    """Adversary: reports every real object, but consistently OFFSET by offset_m.

    A consistent offset makes the adversary ASSOCIATE (passes the loose distance gate)
    while DISAGREEING with consensus (beyond the tight agreement radius) -- the exact
    case the trust layer must catch. Deterministic (no RNG) so trust decay is clean.
    """
    if len(gt_centers) == 0:
        return np.zeros((0, 3))
    out = gt_centers.copy()
    out[:, 0] += offset_m  # shift east
    return out


def honest_boxes(fr: Any, detector: Any) -> Any:
    """Simulated honest detections for one agent frame -> list[Detection3D]."""
    centers = fr.gt_locations_world()
    dims = np.array([o.dimensions for o in fr.gt_objects]) if fr.gt_objects \
        else np.zeros((0, 3))
    yaws = np.array([o.yaw_rad for o in fr.gt_objects]) if fr.gt_objects else np.zeros(0)
    sxyz = np.asarray(fr.pose)[:3, 3] if np.asarray(fr.pose).shape == (4, 4) \
        else np.zeros(3)
    return detector.detect(centers, dims, yaws, sxyz)


def choose_scenario(ds: Any, min_agents: int) -> str:
    """Scenario with >= min_agents and the most frames (one coherent scene)."""
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
    return chosen


def run(oc: Any, data_root: Path, min_agents: int, seed: int,
        assoc_radius: float, adv_offset_m: float) -> dict:
    from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415
    from omnicopilot.evaluation.metrics import compute_map  # noqa: PLC0415
    from omnicopilot.perception.simulated_detector import (  # noqa: PLC0415
        DetectorNoiseConfig, SimulatedDetector,
    )

    ds = OPV2VDataset(data_root)
    ds.load()
    chosen = choose_scenario(ds, min_agents)
    sc = ds.get_scenario(chosen)
    adversary_id = sc.agent_ids[-1]
    honest = SimulatedDetector(DetectorNoiseConfig(), seed=seed)

    def base_cfg() -> Any:
        cfg = oc.WorldModelConfig()
        cfg.max_entities = 4000
        cfg.fusion.association_max_distance_m = max(assoc_radius, adv_offset_m + 1.0)
        cfg.reliability.trust_ema_alpha = 0.15
        cfg.reliability.calibration_period = 5
        cfg.reliability.agreement_radius_m = 1.0
        return cfg

    # ── PASS 1: learn per-agent trust on ONE persistent model over all frames ──
    wm_learn = oc.WorldModel(base_cfg())
    adv_trust_trace: list[float] = []
    t = 1.0
    for frame_no, fi in enumerate(sc.frame_indices):
        frames = ds.get_all_agent_frames(chosen, fi, load_lidar=False)
        t += 0.1
        for aid, fr in frames.items():
            if aid == adversary_id:
                continue
            for j, d in enumerate(honest_boxes(fr, honest)):
                wm_learn.ingest_observations(
                    [make_obs(oc, aid, f"{aid}_{frame_no}_{j}", d.position,
                              d.confidence, t)], 0.9)
        adv_frame = frames.get(adversary_id)
        gt_c = adv_frame.gt_locations_world() if adv_frame else np.zeros((0, 3))
        for j, c in enumerate(adversary_detections(gt_c, adv_offset_m)):
            wm_learn.ingest_observations(
                [make_obs(oc, adversary_id, f"{adversary_id}_{frame_no}_{j}", c, 0.9, t)],
                0.9)
        wm_learn.tick(t)
        adv_trust_trace.append(wm_learn.get_agent_trust(adversary_id))

    learned_trust = {aid: wm_learn.get_agent_trust(aid) for aid in sc.agent_ids}
    honest_ids = [a for a in sc.agent_ids if a != adversary_id]
    honest_final = float(np.mean([learned_trust[a] for a in honest_ids])) if honest_ids else 0.0

    # ── PASS 2: score AP with a FRESH model per frame; WITH vs WITHOUT trust ──
    def score(use_learned: bool) -> Any:
        preds: list[np.ndarray] = []
        gts: list[np.ndarray] = []
        for fi in sc.frame_indices:
            frames = ds.get_all_agent_frames(chosen, fi, load_lidar=False)
            all_c = [fr.gt_locations_world() for fr in frames.values()]
            gt_union = _dedupe(np.vstack(all_c)) if all_c else np.zeros((0, 3))
            gt7 = np.hstack([gt_union, np.tile([4.5, 2.0, 1.5, 0.0], (len(gt_union), 1))]) \
                if len(gt_union) else np.zeros((0, 7))

            wm = oc.WorldModel(base_cfg())
            step = 1.0
            for aid, fr in frames.items():
                trust = learned_trust[aid] if use_learned else 0.9
                if aid == adversary_id:
                    for j, c in enumerate(adversary_detections(
                            fr.gt_locations_world(), adv_offset_m)):
                        wm.ingest_observations(
                            [make_obs(oc, aid, f"{aid}_{j}", c, 0.9, step)], trust)
                else:
                    for j, d in enumerate(honest_boxes(fr, honest)):
                        wm.ingest_observations(
                            [make_obs(oc, aid, f"{aid}_{j}", d.position, d.confidence,
                                      step)], trust)
            wm.tick(step + 0.1)
            rows = [[e.position.x, e.position.y, e.position.z, 4.5, 2.0, 1.5, 0.0,
                     float(e.confidence)] for e in wm.get_entities()]
            preds.append(np.array(rows) if rows else np.zeros((0, 8)))
            gts.append(gt7)
        return compute_map(preds, gts, iou_thresholds=np.array([0.5, 0.7]))

    m_with = score(use_learned=True)
    m_without = score(use_learned=False)

    return {
        "scenario": chosen,
        "adversary_id": adversary_id,
        "adversary_offset_m": adv_offset_m,
        "frames": len(sc.frame_indices),
        "adversary_trust_start": round(adv_trust_trace[0], 4) if adv_trust_trace else None,
        "adversary_trust_end": round(adv_trust_trace[-1], 4) if adv_trust_trace else None,
        "honest_agent_final_trust": round(honest_final, 4),
        "with_trust": {"mAP": round(m_with.mean_average_precision, 4),
                       "precision": round(m_with.precision, 4),
                       "recall": round(m_with.recall, 4)},
        "without_trust": {"mAP": round(m_without.mean_average_precision, 4),
                          "precision": round(m_without.precision, 4),
                          "recall": round(m_without.recall, 4)},
        "precision_defended": round(m_with.precision - m_without.precision, 4),
        "map_defended": round(m_with.mean_average_precision - m_without.mean_average_precision, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--module-dir", type=str, default=None)
    p.add_argument("--min-agents", type=int, default=3)
    p.add_argument("--assoc-radius", type=float, default=2.5)
    p.add_argument("--adv-offset-m", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    oc = import_cpp(args.module_dir)
    r = run(oc, args.data_root, args.min_agents, args.seed, args.assoc_radius,
            args.adv_offset_m)

    print("=" * 68)
    print("ADVERSARIAL-AGENT DEMO — does the trust layer defend fusion?")
    print("=" * 68)
    print(f"Scenario: {r['scenario']}  | adversary: {r['adversary_id']}  | "
          f"offset {r['adversary_offset_m']} m  | frames {r['frames']}")
    print("-" * 68)
    print(f"Adversary trust:  start {r['adversary_trust_start']}  ->  "
          f"end {r['adversary_trust_end']}")
    print(f"Honest agents' final trust: {r['honest_agent_final_trust']}")
    print("-" * 68)
    print(f"{'':16} {'mAP':>8} {'precision':>10} {'recall':>8}")
    w, wo = r["with_trust"], r["without_trust"]
    print(f"{'WITH trust':16} {w['mAP']:>8} {w['precision']:>10} {w['recall']:>8}")
    print(f"{'WITHOUT trust':16} {wo['mAP']:>8} {wo['precision']:>10} {wo['recall']:>8}")
    print("-" * 68)
    print(f"mAP defended by trust layer:       {r['map_defended']:+.4f}")
    print(f"Precision defended by trust layer: {r['precision_defended']:+.4f}")
    print("NOTE: simulated detector + scripted spoofer, not adversarial ML.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()

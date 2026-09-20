"""Sweep the fusion association radius against real OPV2V data.

Feeds real ground-truth objects (world frame) through the C++ WorldModel at a
range of ``association_max_distance_m`` values and reports how the fused entity
count and confirmed count change. The goal is to pick the radius that best
reproduces the true distinct-object count without over-merging distinct vehicles
or under-merging the same vehicle seen by several agents.

Interpretation of the output curve:
- Too small  -> the SAME car seen by N agents (with ~1-2 m pose noise) splits into
  several near-duplicate entities: entity count inflated, ``confirmed`` collapses.
- Too large  -> genuinely DISTINCT nearby cars merge into one entity: entity count
  deflated below the true object count.
- The knee (flat region) is the defensible radius.

Prerequisites:
- Built pybind11 module ``_omnicopilot_cpp`` with the FusionConfig binding
  (needs the "Wire FusionConfig through WorldModel" change; rebuild after pulling).
- OPV2V data available.

Usage:
    python scripts/sweep_association_radius.py \
        --data-root /kaggle/input/.../opv2v-2/test \
        --module-dir /kaggle/working/OmniCopilot-ai/build/cpp \
        --frame-stride 20 --min-agents 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Self-locating bootstrap: make `omnicopilot` importable no matter how invoked ──
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if _PY_ROOT.is_dir() and str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))


DEFAULT_RADII = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def import_cpp_module(module_dir: str | None) -> Any:
    """Import the compiled C++ pybind11 module."""
    if module_dir:
        sys.path.insert(0, module_dir)
    try:
        import _omnicopilot_cpp as oc  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "Could not import _omnicopilot_cpp. Build it first:\n"
            "  cmake -B build -DOMNICOPILOT_BUILD_BINDINGS=ON\n"
            "  cmake --build build\n"
            "then pass --module-dir <repo>/build/cpp"
        )
        raise ImportError(msg) from e
    # Fail loudly if the module predates the FusionConfig binding.
    if not hasattr(oc, "FusionConfig"):
        msg = (
            "The built module has no FusionConfig binding. Rebuild after pulling the "
            "'Wire FusionConfig through WorldModel' change:\n"
            "  cmake --build build -j"
        )
        raise RuntimeError(msg)
    return oc


def _obj_class(oc: Any, obj_type: str) -> Any:
    t = obj_type.lower()
    if "ped" in t:
        return oc.ObjectClass.PEDESTRIAN
    if "cycl" in t:
        return oc.ObjectClass.CYCLIST
    if "truck" in t:
        return oc.ObjectClass.TRUCK
    return oc.ObjectClass.VEHICLE


def fuse_frame(oc: Any, frames: dict, radius: float, agent_trust: float) -> tuple[int, int, int]:
    """Fuse one frame (all agents) at a given radius. Returns (entities, confirmed, best_single)."""
    cfg = oc.WorldModelConfig()
    cfg.max_entities = 2000
    cfg.fusion.association_max_distance_m = radius
    wm = oc.WorldModel(cfg)

    best_single = 0
    # Use a single, consistent step time for all observations in this frame, so the
    # lifecycle sees one clean simulation step (like the demo). Decoupled from the raw
    # dataset timestamp to avoid a huge decay dt that could suppress confirmation.
    step_t = 1.0
    for aid, frame in frames.items():
        world = frame.gt_locations_world()
        best_single = max(best_single, len(frame.gt_objects))
        for i, (obj, wpos) in enumerate(zip(frame.gt_objects, world)):
            obs = oc.Observation()
            obs.observation_id = f"{aid}_{i}"
            obs.agent_id = aid
            obs.object_class = _obj_class(oc, obj.obj_type)
            obs.position.x = float(wpos[0])
            obs.position.y = float(wpos[1])
            obs.position.z = float(wpos[2])
            obs.confidence = 0.9
            obs.timestamp_s = step_t
            wm.ingest_observations([obs], agent_trust)

    # Advance by a positive dt so the lifecycle update runs. WorldModel::Tick
    # early-returns if dt <= 0, which skips the Tentative->Confirmed transition
    # entirely (the bug that produced ~0 confirmed in the first sweep). Tick to a
    # time just past the observation time: one clean step, minimal decay.
    wm.tick(step_t + 0.1)
    stats = wm.get_stats()
    return stats.total_entities, stats.confirmed, best_single


def debug_one_frame(oc: Any, ds: Any, sid: str, fi: int, radius: float) -> None:
    """Dump diagnostics for a single real frame to explain association behavior.

    Prints, for the same physical objects seen by multiple agents, the cross-agent
    world-frame nearest-neighbor distances (via the loader's OWN matcher), and the
    C++ entity/confirmed counts. This tells us empirically whether objects land close
    enough to associate, or whether association/greedy assignment is the problem —
    rather than guessing.
    """
    import numpy as np  # noqa: PLC0415
    from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415

    frames = ds.get_all_agent_frames(sid, fi, load_lidar=False)
    per_agent = {aid: len(f.gt_objects) for aid, f in frames.items()}
    print(f"\n[DEBUG] scenario={sid} frame={fi} agents={list(frames)}")
    print(f"[DEBUG] per-agent object counts: {per_agent}")

    # 1) Ground-truth cross-agent matches via the loader's proximity matcher.
    for tol in (2.0, 3.0, 5.0):
        groups = OPV2VDataset.match_objects_across_agents(frames, tolerance_m=tol)
        multi = [g for g in groups if len(g) >= 2]
        print(f"[DEBUG] loader matcher tol={tol}m: {len(groups)} distinct objects, "
              f"{len(multi)} seen by >=2 agents")

    # 2) Feed through C++ exactly as the sweep does, report entities/confirmed.
    e, c, b = fuse_frame(oc, frames, radius, 0.9)
    print(f"[DEBUG] C++ fuse @ radius={radius}m: entities={e} confirmed={c} best_single={b}")

    # 3) Raw world-position spread: for the first agent's first few objects, show the
    #    nearest object in every OTHER agent (the distance association must beat).
    agents = list(frames)
    if len(agents) >= 2:
        a0 = agents[0]
        w0 = frames[a0].gt_locations_world()
        others = {aid: frames[aid].gt_locations_world() for aid in agents[1:]}
        print(f"[DEBUG] nearest cross-agent neighbor for first 5 objects of {a0}:")
        for i in range(min(5, len(w0))):
            dists = []
            for aid, w in others.items():
                if len(w):
                    d = float(np.min(np.linalg.norm(w - w0[i], axis=1)))
                    dists.append(f"{aid}:{d:.2f}m")
            print(f"    obj{i} @ ({w0[i][0]:.1f},{w0[i][1]:.1f}) -> {', '.join(dists)}")

    # 4) FRAME HYPOTHESIS TEST — decide what frame `location` is actually in on THIS
    #    dataset upload. Compare two candidate transforms by how many objects match
    #    across agents (matches => the transform is correct):
    #      H1: world = pose @ location   (location is EGO frame; current loader assumption)
    #      H2: world = location          (location is already WORLD frame)
    #    Also H3: world = inv(pose) @ location (in case location is world and we must
    #    map into a common ego frame the other way). The winner is whichever yields
    #    many multi-agent matches at ~2-3 m.
    def _match_count(centers_by_agent: dict, tol: float) -> tuple[int, int]:
        entries = []
        for aid, cs in centers_by_agent.items():
            for c in cs:
                entries.append((aid, c))
        used = [False] * len(entries)
        distinct = multi = 0
        for i in range(len(entries)):
            if used[i]:
                continue
            group_agents = {entries[i][0]}
            used[i] = True
            for j in range(i + 1, len(entries)):
                if used[j] or entries[j][0] in group_agents:
                    continue
                if float(np.linalg.norm(entries[i][1] - entries[j][1])) <= tol:
                    group_agents.add(entries[j][0])
                    used[j] = True
            distinct += 1
            if len(group_agents) >= 2:
                multi += 1
        return distinct, multi

    print("[DEBUG] --- transform hypothesis test (matches at tol=3.0m) ---")
    h1 = {aid: f.gt_locations_world() for aid, f in frames.items()}  # pose @ location
    h2 = {aid: np.array([o.location_ego for o in f.gt_objects], dtype=np.float64)
          for aid, f in frames.items()}                              # location as-is
    h3 = {}
    for aid, f in frames.items():
        inv = np.linalg.inv(f.pose)
        locs = np.array([o.location_ego for o in f.gt_objects], dtype=np.float64)
        h3[aid] = transform_points(inv, locs) if len(locs) else locs  # inv(pose) @ location
    for name, cand in (("H1 pose@loc", h1), ("H2 loc as-is", h2), ("H3 inv(pose)@loc", h3)):
        d, m = _match_count(cand, 3.0)
        print(f"    {name:>18}: {d} distinct, {m} seen by >=2 agents")
    print("[DEBUG] The correct transform is the one with MANY '>=2 agents' matches.")


def run_sweep(
    oc: Any,
    data_root: Path,
    radii: list[float],
    frame_stride: int,
    min_agents: int,
    agent_trust: float,
) -> dict:
    """Run the radius sweep across sampled frames."""
    from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415

    ds = OPV2VDataset(data_root)
    ds.load()
    if not ds.scenario_ids():
        msg = f"No scenarios under {data_root}"
        raise RuntimeError(msg)

    # Collect a stable sample of multi-agent frames once, reuse for every radius.
    sampled: list[tuple[str, int]] = []
    for sid in ds.scenario_ids():
        sc = ds.get_scenario(sid)
        if len(sc.agent_ids) < min_agents:
            continue
        for fi in sc.frame_indices[::frame_stride]:
            sampled.append((sid, fi))

    if not sampled:
        msg = f"No frames with >= {min_agents} agents found."
        raise RuntimeError(msg)

    print(f"Sampled {len(sampled)} frames (>= {min_agents} agents, stride {frame_stride}).")

    results = []
    for radius in radii:
        tot_entities = tot_confirmed = tot_best = n = 0
        for sid, fi in sampled:
            frames = ds.get_all_agent_frames(sid, fi, load_lidar=False)
            e, c, b = fuse_frame(oc, frames, radius, agent_trust)
            tot_entities += e
            tot_confirmed += c
            tot_best += b
            n += 1
        gain = (tot_entities / tot_best) if tot_best else 0.0
        row = {
            "radius_m": radius,
            "mean_entities": round(tot_entities / n, 2),
            "mean_confirmed": round(tot_confirmed / n, 2),
            "mean_best_single": round(tot_best / n, 2),
            "gain_vs_best_single": round(gain, 3),
        }
        results.append(row)
        print(
            f"radius={radius:>4.1f}m  mean_entities={row['mean_entities']:>7}  "
            f"mean_confirmed={row['mean_confirmed']:>7}  gain={row['gain_vs_best_single']}x"
        )

    return {"num_frames": len(sampled), "min_agents": min_agents, "sweep": results}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--module-dir", type=str, default=None,
                        help="Dir containing built _omnicopilot_cpp (build/cpp)")
    parser.add_argument("--radii", type=float, nargs="+", default=DEFAULT_RADII)
    parser.add_argument("--frame-stride", type=int, default=20,
                        help="Subsample every Nth common frame per scenario")
    parser.add_argument("--min-agents", type=int, default=3)
    parser.add_argument("--agent-trust", type=float, default=0.9)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON output path for the sweep table")
    parser.add_argument("--debug-frame", action="store_true",
                        help="Dump diagnostics for ONE real frame (why association "
                             "behaves as it does) instead of running the full sweep")
    args = parser.parse_args()

    oc = import_cpp_module(args.module_dir)

    if args.debug_frame:
        from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415
        ds = OPV2VDataset(args.data_root)
        ds.load()
        # Pick the first scenario with >= min_agents agents.
        target = None
        for sid in ds.scenario_ids():
            sc = ds.get_scenario(sid)
            if len(sc.agent_ids) >= args.min_agents:
                target = (sid, sc.frame_indices[0])
                break
        if target is None:
            print(f"No scenario with >= {args.min_agents} agents found.")
            return
        debug_one_frame(oc, ds, target[0], target[1], radius=5.0)
        return

    summary = run_sweep(
        oc, args.data_root, args.radii, args.frame_stride, args.min_agents, args.agent_trust
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote sweep table -> {args.out}")

    print("\nPick the radius at the knee: where mean_entities flattens near the true "
          "distinct-object count while mean_confirmed stays high.")


if __name__ == "__main__":
    main()

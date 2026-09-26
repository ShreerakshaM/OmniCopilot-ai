"""Feed OPV2V observations through the C++ world model — the real end-to-end pipeline.

    OPV2V data  →  Python loader  →  Observations  →  [pybind11]  →  C++ WorldModel
                                                                        → fused entities

This connects the two halves of the system that were previously isolated:
- Python data pipeline (loader) — produces per-agent ground-truth objects
- C++ core (world model / fusion / trust) — fuses them into a shared belief

It measures cooperative fusion THROUGH THE REAL C++ CORE (not the pure-Python
ground-truth analysis). Compared to the ground-truth ceiling (result 01), this shows
what the actual fusion engine produces: association, confidence combination, lifecycle.

Prerequisites:
- The pybind11 module `_omnicopilot_cpp` must be built and importable
  (build with: cmake -DOMNICOPILOT_BUILD_BINDINGS=ON, then add build/cpp to sys.path).
- OPV2V data available (real) OR use --synthetic for a no-data smoke test.

Usage:
    # Real data (e.g. on Kaggle after building the module):
    python scripts/feed_observations_demo.py \
        --data-root /kaggle/input/.../opv2v-2/test \
        --module-dir /kaggle/working/OmniCopilot-ai/build/cpp \
        --scenario <id> --frame <n>

    # No-data smoke test (synthetic observations):
    python scripts/feed_observations_demo.py --synthetic \
        --module-dir <path-to-build/cpp>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


def import_cpp_module(module_dir: str | None) -> Any:
    """Import the compiled C++ pybind11 module, adding its build dir to the path."""
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
    return oc


# Map OPV2V object type strings -> C++ ObjectClass enum.
def _obj_class(oc: Any, obj_type: str) -> Any:
    t = obj_type.lower()
    if "ped" in t:
        return oc.ObjectClass.PEDESTRIAN
    if "cycl" in t:
        return oc.ObjectClass.CYCLIST
    if "truck" in t:
        return oc.ObjectClass.TRUCK
    # OPV2V is predominantly "Car".
    return oc.ObjectClass.VEHICLE


def build_observation(oc: Any, agent_id: str, obs_id: str,
                      world_xyz: tuple[float, float, float],
                      obj_type: str, confidence: float, timestamp_s: float) -> Any:
    """Construct a C++ Observation in WORLD coordinates.

    NOTE: the C++ world model fuses in a single common frame. We pass world-frame
    positions (the loader transforms ego -> world via the validated pose @ location).
    """
    obs = oc.Observation()
    obs.observation_id = obs_id
    obs.agent_id = agent_id
    obs.object_class = _obj_class(oc, obj_type)
    obs.position.x = float(world_xyz[0])
    obs.position.y = float(world_xyz[1])
    obs.position.z = float(world_xyz[2])
    obs.confidence = float(confidence)
    obs.timestamp_s = float(timestamp_s)
    return obs


def run_real(oc: Any, data_root: Path, scenario_id: str | None,
             frame_idx: int | None, agent_trust: float) -> None:
    """Feed one real OPV2V frame (all agents) through the C++ world model."""
    from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415

    ds = OPV2VDataset(data_root)
    ds.load()
    if not ds.scenario_ids():
        msg = f"No scenarios under {data_root}"
        raise RuntimeError(msg)

    sid = scenario_id or ds.scenario_ids()[0]
    scenario = ds.get_scenario(sid)
    fi = frame_idx if frame_idx is not None else scenario.frame_indices[0]

    print(f"Scenario: {sid}")
    print(f"Frame:    {fi}")
    print(f"Agents:   {scenario.agent_ids}")

    frames = ds.get_all_agent_frames(sid, fi, load_lidar=False)

    # Build a fresh C++ world model.
    cfg = oc.WorldModelConfig()
    cfg.max_entities = 500
    wm = oc.WorldModel(cfg)

    # Feed each agent's ground-truth objects (in WORLD frame) into the C++ core.
    total_obs = 0
    single_agent_counts = {}
    for aid, frame in frames.items():
        world = frame.gt_locations_world()  # (M, 3) world coords via validated transform
        single_agent_counts[aid] = len(frame.gt_objects)
        for i, (obj, wpos) in enumerate(zip(frame.gt_objects, world)):
            obs = build_observation(
                oc, agent_id=aid, obs_id=f"{aid}_{fi}_{i}",
                world_xyz=(wpos[0], wpos[1], wpos[2]),
                obj_type=obj.obj_type, confidence=0.9, timestamp_s=fi / 10.0,
            )
            wm.ingest_observations([obs], agent_trust)
            total_obs += 1

    wm.tick(fi / 10.0)  # advance the world model one tick

    stats = wm.get_stats()
    entities = wm.get_entities()

    best_single = max(single_agent_counts.values()) if single_agent_counts else 0

    print("\n" + "=" * 60)
    print("REAL DATA -> C++ WORLD MODEL")
    print("=" * 60)
    print(f"Per-agent GT object counts: {single_agent_counts}")
    print(f"Total observations fed:     {total_obs}")
    print(f"Best single agent saw:      {best_single} objects")
    print(f"C++ fused entities:         {stats.total_entities} "
          f"(confirmed={stats.confirmed}, tentative={stats.tentative})")
    print(f"Mean fused confidence:      {stats.mean_confidence:.3f}")
    # NOTE: we deliberately do NOT print a "coverage gain vs best single agent" ratio
    # here. That ratio (fused_entities / best_single_raw_count) divides a DEDUPLICATED
    # count by a NON-deduplicated one -- apples-to-oranges, and misleading (it can fall
    # below 1.0 even when fusion is working correctly). For the correct cooperative
    # metric (single-agent AP vs. cooperative-fused AP, like-with-like), see
    # scripts/analyze_detection_cooperation.py and docs/results/03_detection_cooperation.md.
    print("\nInterpretation: this demo shows the end-to-end plumbing -- real per-agent "
          "objects\nflow through the pybind11 bridge into the C++ WorldModel, which "
          "associates the\nsame physical object seen by multiple agents into single "
          "confirmed entities and\nkeeps agent-exclusive objects separate. It is a "
          "PIPELINE demo, not a benchmark;\nfor the quantitative cooperative result use "
          "analyze_detection_cooperation.py.")


def run_synthetic(oc: Any, agent_trust: float) -> None:
    """No-data smoke test: two agents see one shared + some exclusive objects."""
    cfg = oc.WorldModelConfig()
    cfg.max_entities = 100
    wm = oc.WorldModel(cfg)

    # Agent A: shared pedestrian + 1 exclusive car.
    a_shared = build_observation(oc, "A", "A_ped", (10.0, 20.0, 0.0), "Pedestrian", 0.85, 1.0)
    a_excl = build_observation(oc, "A", "A_car", (30.0, 5.0, 0.0), "Car", 0.9, 1.0)
    wm.ingest_observations([a_shared, a_excl], agent_trust)

    # Agent B: SAME pedestrian (slightly offset) + 2 exclusive cars.
    b_shared = build_observation(oc, "B", "B_ped", (10.4, 20.2, 0.0), "Pedestrian", 0.80, 1.1)
    b_excl1 = build_observation(oc, "B", "B_car1", (50.0, -10.0, 0.0), "Car", 0.88, 1.1)
    b_excl2 = build_observation(oc, "B", "B_car2", (55.0, -12.0, 0.0), "Car", 0.82, 1.1)
    wm.ingest_observations([b_shared, b_excl1, b_excl2], agent_trust)

    wm.tick(1.2)

    stats = wm.get_stats()
    entities = wm.get_entities()

    print("=" * 60)
    print("SYNTHETIC SMOKE TEST -> C++ WORLD MODEL")
    print("=" * 60)
    print("Agent A saw: 2 objects (1 shared pedestrian + 1 car)")
    print("Agent B saw: 3 objects (1 shared pedestrian + 2 cars)")
    print("Best single agent: 3 objects")
    print(f"C++ fused entities: {stats.total_entities} "
          f"(expected 4 = 1 shared + 3 exclusive)")
    print(f"Confirmed (>=2 sources): {stats.confirmed} (expected 1 = the shared pedestrian)")
    print()
    for e in entities:
        print(f"  {e.entity_id}: pos=({e.position.x:.1f},{e.position.y:.1f}) "
              f"conf={e.confidence:.3f} sources={e.unique_source_count}")
    ok = stats.total_entities == 4 and stats.confirmed == 1
    print(f"\n{'PASS' if ok else 'CHECK'}: fusion merged the shared object, kept "
          f"exclusives separate.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=str, default=None,
                        help="Dir containing the built _omnicopilot_cpp module (build/cpp)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run the no-data synthetic smoke test")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="OPV2V split dir (for real run)")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--agent-trust", type=float, default=0.9)
    args = parser.parse_args()

    oc = import_cpp_module(args.module_dir)

    if args.synthetic or args.data_root is None:
        run_synthetic(oc, args.agent_trust)
    else:
        run_real(oc, args.data_root, args.scenario, args.frame, args.agent_trust)


if __name__ == "__main__":
    main()

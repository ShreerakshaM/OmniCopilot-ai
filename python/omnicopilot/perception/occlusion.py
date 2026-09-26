"""Occluded-object recovery metric — the differentiating cooperative-perception result.

OPV2V has no explicit occlusion label. We DERIVE what a single agent misses from the
multi-agent ground truth we already load: an object that some agent in the fleet sees,
but a given ego agent does NOT see, is (by construction) beyond that ego's own
perception — occluded, out of range, or out of field of view. Cooperation's value is
exactly how many of those an ego recovers by fusing its neighbors' observations.

Two quantities per frame, per candidate ego agent:
- ``missed``   = objects in the fleet union that this ego does not see on its own.
- ``recovered``= of those missed objects, how many appear in the fused set the ego would
  receive from cooperation (upper bound: the full fleet union; with a real detector and
  comms policy this becomes the achievable subset).

Objects are matched across agents by world-frame proximity (per-agent dict keys are NOT
stable across agents — a validated fact about this dataset), so we cluster world-frame
centers with a distance tolerance.

This module works on ground-truth objects now (the ceiling). In Phase 3 the same logic
runs on detector output for the achievable recovery rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class RecoveryStats:
    """Occluded-object recovery for one frame (aggregated over candidate egos)."""

    num_agents: int
    fleet_union_count: int  # distinct objects the whole fleet sees.
    mean_ego_visible: float  # mean objects a single ego sees alone.
    mean_missed: float  # mean objects an ego misses alone.
    mean_recovered: float  # mean missed objects recovered via cooperation.
    recovery_rate: float  # recovered / missed (0..1).


def cluster_world_points(points: npt.NDArray[np.float64], tolerance_m: float) -> list[list[int]]:
    """Greedy single-linkage-ish clustering of world-frame centers by distance.

    Args:
        points: (K, 3) or (K, 2) world coordinates (rows may come from many agents).
        tolerance_m: two centers within this distance are the same physical object.

    Returns:
        List of clusters; each cluster is a list of row indices into ``points``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return []
    xy = points[:, :2]
    n = xy.shape[0]
    assigned = -np.ones(n, dtype=int)
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i] != -1:
            continue
        cid = len(clusters)
        members = [i]
        assigned[i] = cid
        # Absorb any not-yet-assigned point within tolerance of the seed.
        d = np.linalg.norm(xy - xy[i], axis=1)
        for j in np.where(d <= tolerance_m)[0]:
            if assigned[j] == -1:
                assigned[j] = cid
                members.append(int(j))
        clusters.append(members)
    return clusters


def frame_recovery(
    per_agent_world_centers: dict[str, npt.NDArray[np.float64]],
    tolerance_m: float = 2.0,
) -> RecoveryStats:
    """Compute occluded-object recovery for one frame.

    Args:
        per_agent_world_centers: agent_id -> (M_a, 3) world-frame object centers
            (use ``OPV2VFrame.gt_locations_world()`` per agent, or detector output).
        tolerance_m: cross-agent matching tolerance (same physical object).

    Returns:
        RecoveryStats for this frame.
    """
    agents = list(per_agent_world_centers.keys())
    num_agents = len(agents)
    if num_agents == 0:
        return RecoveryStats(0, 0, 0.0, 0.0, 0.0, 0.0)

    # Stack all agents' centers, remembering which agent each row came from.
    rows: list[np.ndarray] = []
    owner: list[str] = []
    for aid, centers in per_agent_world_centers.items():
        c = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        for r in c:
            rows.append(r)
            owner.append(aid)
    if not rows:
        return RecoveryStats(num_agents, 0, 0.0, 0.0, 0.0, 0.0)

    pts = np.vstack(rows)
    clusters = cluster_world_points(pts, tolerance_m)
    fleet_union = len(clusters)

    # For each cluster (physical object), which agents see it?
    seen_by: list[set[str]] = [{owner[idx] for idx in cl} for cl in clusters]

    visibles, misseds, recovereds = [], [], []
    for ego in agents:
        # Objects this ego sees alone = clusters that include ego.
        ego_visible = sum(1 for s in seen_by if ego in s)
        # Missed = clusters the ego does NOT see but the fleet does.
        missed = sum(1 for s in seen_by if ego not in s)
        # Recovered via cooperation = all missed clusters seen by >=1 other agent.
        # (Ceiling: every missed object is recoverable since some agent saw it.)
        recovered = sum(1 for s in seen_by if ego not in s and len(s) >= 1)
        visibles.append(ego_visible)
        misseds.append(missed)
        recovereds.append(recovered)

    mean_missed = float(np.mean(misseds))
    mean_recovered = float(np.mean(recovereds))
    rate = (mean_recovered / mean_missed) if mean_missed > 0 else 0.0

    return RecoveryStats(
        num_agents=num_agents,
        fleet_union_count=fleet_union,
        mean_ego_visible=float(np.mean(visibles)),
        mean_missed=mean_missed,
        mean_recovered=mean_recovered,
        recovery_rate=rate,
    )

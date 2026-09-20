"""Unit tests for the occluded-object recovery metric."""

from __future__ import annotations

import numpy as np

from omnicopilot.perception.occlusion import cluster_world_points, frame_recovery


def test_cluster_merges_near_points():
    pts = np.array([[0, 0, 0], [0.5, 0.0, 0], [50, 50, 0]])
    clusters = cluster_world_points(pts, tolerance_m=2.0)
    assert len(clusters) == 2  # the two near points merge, the far one is separate


def test_cluster_empty():
    assert cluster_world_points(np.zeros((0, 3)), 2.0) == []


def test_recovery_two_agents_one_shared_one_exclusive_each():
    # Agent A sees objects at (0,0) [shared] and (10,0) [exclusive to A].
    # Agent B sees objects at (0.3,0) [same shared] and (20,0) [exclusive to B].
    a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    b = np.array([[0.3, 0.0, 0.0], [20.0, 0.0, 0.0]])
    stats = frame_recovery({"A": a, "B": b}, tolerance_m=2.0)

    assert stats.num_agents == 2
    # Fleet union: shared + A-excl + B-excl = 3 distinct objects.
    assert stats.fleet_union_count == 3
    # A sees 2 (shared + its exclusive), misses B's exclusive -> 1 missed, 1 recovered.
    # B symmetric. So mean missed = 1, mean recovered = 1, rate = 1.0.
    assert abs(stats.mean_missed - 1.0) < 1e-9
    assert abs(stats.mean_recovered - 1.0) < 1e-9
    assert abs(stats.recovery_rate - 1.0) < 1e-9


def test_recovery_single_agent_misses_nothing():
    # One agent alone: fleet union == what it sees, nothing missed.
    a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    stats = frame_recovery({"A": a}, tolerance_m=2.0)
    assert stats.fleet_union_count == 2
    assert stats.mean_missed == 0.0
    assert stats.recovery_rate == 0.0  # nothing to recover


def test_recovery_empty_frame():
    stats = frame_recovery({}, tolerance_m=2.0)
    assert stats.num_agents == 0
    assert stats.fleet_union_count == 0

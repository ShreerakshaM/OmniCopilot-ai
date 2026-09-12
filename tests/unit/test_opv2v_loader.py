"""Unit tests for the OPV2V loader's transform and parsing logic.

These tests do NOT require the actual dataset — they validate the coordinate
math and data structures that were empirically confirmed against real OPV2V data
(shared cars matched within ~2m using pose @ location).
"""

from __future__ import annotations

import math

import numpy as np

from omnicopilot.data.opv2v import (
    GroundTruthObject,
    OPV2VDataset,
    OPV2VFrame,
    pose_to_matrix,
    transform_points,
    x_to_world,
)


class TestXToWorld:
    """Tests for the 6-DoF -> 4x4 transform (mirrors OpenCOOD)."""

    def test_identity_pose(self) -> None:
        """Zero pose yields (near) identity."""
        m = x_to_world([0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(m, np.identity(4), atol=1e-9)

    def test_pure_translation(self) -> None:
        """Translation appears in the last column."""
        m = x_to_world([10.0, -5.0, 2.0, 0, 0, 0])
        np.testing.assert_allclose(m[:3, 3], [10.0, -5.0, 2.0], atol=1e-9)

    def test_yaw_90_degrees(self) -> None:
        """90-degree yaw rotates +x into +y."""
        m = x_to_world([0, 0, 0, 0, 90.0, 0])
        # A point at (1,0,0) local should map to ~(0,1,0) world.
        p = m @ np.array([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(p[:3], [0.0, 1.0, 0.0], atol=1e-9)


class TestPoseToMatrix:
    """Tests for handling both pose variants (4x4 matrix vs 6-vector)."""

    def test_accepts_4x4_matrix(self) -> None:
        """A pre-built 4x4 matrix is returned as-is."""
        mat = np.identity(4)
        mat[0, 3] = 7.0
        out = pose_to_matrix(mat)
        np.testing.assert_allclose(out, mat)

    def test_accepts_6_vector(self) -> None:
        """A 6-element pose is converted to a 4x4 matrix."""
        out = pose_to_matrix([1, 2, 3, 0, 0, 0])
        np.testing.assert_allclose(out[:3, 3], [1, 2, 3], atol=1e-9)

    def test_rejects_bad_shape(self) -> None:
        """An unexpected shape raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unexpected lidar_pose"):
            pose_to_matrix([1, 2, 3])  # 3 elements — invalid


class TestTransformPoints:
    """Tests for batch point transformation."""

    def test_translation(self) -> None:
        """Translating points shifts them correctly."""
        m = np.identity(4)
        m[:3, 3] = [10, 20, 30]
        pts = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
        out = transform_points(m, pts)
        np.testing.assert_allclose(out, [[11, 21, 31], [12, 22, 32]])


class TestCrossAgentMatching:
    """Tests for matching the same physical object across agents.

    Mirrors the validated real-data scenario: two agents observe the same cars,
    which land within a few meters of each other in the world frame.
    """

    def _make_frame(
        self, agent_id: str, pose: np.ndarray, objects: list[GroundTruthObject]
    ) -> OPV2VFrame:
        return OPV2VFrame(
            scenario_id="s",
            agent_id=agent_id,
            frame_idx=0,
            timestamp_s=0.0,
            pose=pose,
            gt_objects=objects,
        )

    def test_shared_object_matched(self) -> None:
        """A car seen by two agents (from different ego frames) is matched."""
        # Agent 0 at origin; agent 1 translated by (10, 0, 0).
        pose0 = np.identity(4)
        pose1 = np.identity(4)
        pose1[:3, 3] = [10.0, 0.0, 0.0]

        # A physical car at world (5, 0, 0):
        #  - in agent 0's ego frame: (5, 0, 0)
        #  - in agent 1's ego frame: (-5, 0, 0)
        obj0 = GroundTruthObject("k0", "Car", np.array([5.0, 0.0, 0.0]),
                                 np.array([4.5, 2.0, 1.5]), 0.0)
        obj1 = GroundTruthObject("k9", "Car", np.array([-5.0, 0.0, 0.0]),
                                 np.array([4.5, 2.0, 1.5]), 0.0)

        frames = {
            "0": self._make_frame("0", pose0, [obj0]),
            "1": self._make_frame("1", pose1, [obj1]),
        }
        groups = OPV2VDataset.match_objects_across_agents(frames, tolerance_m=3.0)

        # Exactly one group containing both agents' keys for the same car.
        assert len(groups) == 1
        assert groups[0] == {"0": "k0", "1": "k9"}

    def test_exclusive_objects_not_matched(self) -> None:
        """Objects far apart in world frame form separate groups (blind spots)."""
        pose0 = np.identity(4)
        pose1 = np.identity(4)
        pose1[:3, 3] = [100.0, 0.0, 0.0]  # agents far apart

        obj0 = GroundTruthObject("a", "Car", np.array([1.0, 0.0, 0.0]),
                                 np.array([4.5, 2.0, 1.5]), 0.0)
        obj1 = GroundTruthObject("b", "Car", np.array([1.0, 0.0, 0.0]),
                                 np.array([4.5, 2.0, 1.5]), 0.0)
        # world positions: obj0 -> (1,0,0); obj1 -> (101,0,0) — far apart.

        frames = {
            "0": self._make_frame("0", pose0, [obj0]),
            "1": self._make_frame("1", pose1, [obj1]),
        }
        groups = OPV2VDataset.match_objects_across_agents(frames, tolerance_m=3.0)
        assert len(groups) == 2  # not matched


class TestGroundTruthWorld:
    """Tests that GT locations transform to world frame correctly."""

    def test_gt_locations_world(self) -> None:
        """Ego-frame GT is transformed to world via the pose."""
        pose = np.identity(4)
        pose[:3, 3] = [100.0, 200.0, 0.0]
        obj = GroundTruthObject("k", "Car", np.array([5.0, 0.0, 0.0]),
                                np.array([4.5, 2.0, 1.5]), 0.0)
        frame = OPV2VFrame("s", "0", 0, 0.0, pose, gt_objects=[obj])
        world = frame.gt_locations_world()
        np.testing.assert_allclose(world[0], [105.0, 200.0, 0.0], atol=1e-9)

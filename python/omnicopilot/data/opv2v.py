"""OPV2V dataset loader.

Loads the OPV2V cooperative perception dataset and provides
per-agent, per-frame access to sensor data and ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class OPV2VFrame:
    """A single frame of data for one agent."""

    agent_id: str
    frame_idx: int
    timestamp_s: float
    # LiDAR point cloud (N, 4) — x, y, z, intensity.
    lidar_points: npt.NDArray[np.float32] | None = None
    # Camera images (list of H×W×3 arrays).
    camera_images: list[npt.NDArray[np.uint8]] = field(default_factory=list)
    # Agent pose (4x4 transformation matrix, world frame).
    pose: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    # Ground truth 3D bounding boxes for this agent's view.
    gt_boxes: npt.NDArray[np.float64] | None = None  # (M, 7) — x,y,z,l,w,h,heading.
    gt_labels: npt.NDArray[np.int64] | None = None  # (M,) class labels.


@dataclass
class OPV2VScene:
    """A full scene with multiple agents and frames."""

    scene_id: str
    num_frames: int
    agent_ids: list[str]
    # Per-agent, per-frame data.
    frames: dict[str, list[OPV2VFrame]] = field(default_factory=dict)


class OPV2VDataset:
    """OPV2V dataset interface.

    Provides access to multi-vehicle cooperative perception data.
    """

    def __init__(self, root_path: Path, split: str = "test") -> None:
        """Initialize OPV2V dataset loader.

        Args:
            root_path: Path to OPV2V dataset root.
            split: Dataset split ("train", "validate", "test").
        """
        self._root = root_path
        self._split = split
        self._scenes: list[OPV2VScene] = []

    def load(self) -> None:
        """Load dataset metadata (scene list, agent info). Lazy-loads frames."""
        # TODO: Implement — scan directory structure, build scene index.
        raise NotImplementedError

    def get_scene_ids(self) -> list[str]:
        """Return list of available scene IDs."""
        return [s.scene_id for s in self._scenes]

    def get_scene(self, scene_id: str) -> OPV2VScene:
        """Get a specific scene by ID."""
        for scene in self._scenes:
            if scene.scene_id == scene_id:
                return scene
        msg = f"Scene {scene_id} not found"
        raise KeyError(msg)

    def get_frame(self, scene_id: str, agent_id: str, frame_idx: int) -> OPV2VFrame:
        """Get a specific frame for an agent.

        Args:
            scene_id: Scene identifier.
            agent_id: Agent identifier.
            frame_idx: Frame index.

        Returns:
            Frame data for the specified agent and time.
        """
        # TODO: Implement — load from disk lazily.
        raise NotImplementedError

    def __len__(self) -> int:
        """Return total number of scenes."""
        return len(self._scenes)

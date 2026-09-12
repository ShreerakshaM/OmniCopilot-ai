"""OPV2V dataset loader.

Loads the OPV2V cooperative-perception dataset and provides per-agent, per-frame
access to LiDAR, poses, and ground-truth boxes — plus cross-agent object matching
in a common world frame.

Design notes (validated empirically against the real dataset):
- Each scenario is a directory; each numeric subdirectory is one CAV (agent).
- Per frame, each agent has ``<frame>.pcd`` (LiDAR) and ``<frame>.yaml`` (metadata).
- The YAML embeds numpy objects → must use ``yaml.unsafe_load`` (NOT ``safe_load``).
- ``lidar_pose`` is a 4x4 homogeneous matrix (this data variant) mapping the agent's
  LiDAR/ego frame to the world frame. (Official OPV2V uses [x,y,z,roll,yaw,pitch] in
  degrees — both forms are handled by :func:`pose_to_matrix`.)
- Ground-truth ``vehicles[*].location`` is in the agent's EGO frame; transform to world
  via ``world = pose @ [x, y, z, 1]`` (validated: shared cars match within ~2m).
- ``vehicles[*].extent`` is HALF-dimensions → full L/W/H = 2 x extent.
- ``vehicles[*].angle`` is [roll, yaw, pitch] in DEGREES; yaw = angle[1].
- Object dict keys are PER-AGENT — they do NOT identify the same physical car across
  agents. Match objects across agents by world-frame proximity (see
  :meth:`OPV2VDataset.match_objects_across_agents`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml


# ─── Coordinate transform helpers ───────────────────────────────────────────


def x_to_world(pose6: list[float]) -> npt.NDArray[np.float64]:
    """Build a 4x4 local->world transform from a 6-DoF pose.

    Mirrors OpenCOOD's ``x_to_world``. Input is ``[x, y, z, roll, yaw, pitch]`` with
    angles in DEGREES (CARLA convention).

    Args:
        pose6: ``[x, y, z, roll, yaw, pitch]`` (meters, degrees).

    Returns:
        4x4 homogeneous transform mapping local (ego/LiDAR) coords to world coords.
    """
    x, y, z, roll, yaw, pitch = pose6
    c_y, s_y = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    c_r, s_r = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    c_p, s_p = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))

    m = np.identity(4, dtype=np.float64)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    m[0, 0] = c_p * c_y
    m[0, 1] = c_y * s_p * s_r - s_y * c_r
    m[0, 2] = -c_y * s_p * c_r - s_y * s_r
    m[1, 0] = s_y * c_p
    m[1, 1] = s_y * s_p * s_r + c_y * c_r
    m[1, 2] = -s_y * s_p * c_r + c_y * s_r
    m[2, 0] = s_p
    m[2, 1] = -c_p * s_r
    m[2, 2] = c_p * c_r
    return m


def pose_to_matrix(lidar_pose: Any) -> npt.NDArray[np.float64]:
    """Return a 4x4 local->world matrix, accepting either data variant.

    Handles both:
    - A pre-built 4x4 matrix (this dataset variant).
    - A 6-element ``[x, y, z, roll, yaw, pitch]`` (official OPV2V; degrees).

    Args:
        lidar_pose: 4x4 matrix (list/array) or 6-element pose.

    Returns:
        4x4 homogeneous local->world transform.
    """
    arr = np.array(lidar_pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (6,) or arr.size == 6:
        return x_to_world(list(arr.reshape(-1)))
    msg = f"Unexpected lidar_pose shape {arr.shape}; expected (4,4) or (6,)"
    raise ValueError(msg)


def transform_points(
    matrix: npt.NDArray[np.float64], points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Apply a 4x4 transform to an (N,3) array of points.

    Args:
        matrix: 4x4 homogeneous transform.
        points: (N, 3) points.

    Returns:
        (N, 3) transformed points.
    """
    n = points.shape[0]
    homo = np.concatenate([points, np.ones((n, 1), dtype=points.dtype)], axis=1)  # (N,4)
    out = (matrix @ homo.T).T  # (N,4)
    return out[:, :3]


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class GroundTruthObject:
    """A single ground-truth object (from one agent's YAML)."""

    object_key: str  # Per-agent dict key (NOT stable across agents).
    obj_type: str  # e.g. "Car".
    location_ego: npt.NDArray[np.float64]  # (3,) center in this agent's ego frame.
    dimensions: npt.NDArray[np.float64]  # (3,) full length, width, height (meters).
    yaw_rad: float  # Heading in radians (ego frame).
    ass_id: int = -1  # Cross-agent association id (-1 if not populated).


@dataclass
class OPV2VFrame:
    """A single frame of data for one agent."""

    scenario_id: str
    agent_id: str
    frame_idx: int
    timestamp_s: float
    pose: npt.NDArray[np.float64]  # 4x4 ego->world.
    ego_speed_kmh: float = 0.0
    lidar_points: npt.NDArray[np.float32] | None = None  # (N,4) x,y,z,intensity (ego).
    gt_objects: list[GroundTruthObject] = field(default_factory=list)

    def gt_locations_world(self) -> npt.NDArray[np.float64]:
        """Ground-truth object centers transformed to world frame, shape (M, 3)."""
        if not self.gt_objects:
            return np.zeros((0, 3), dtype=np.float64)
        locs = np.array([o.location_ego for o in self.gt_objects], dtype=np.float64)
        return transform_points(self.pose, locs)


@dataclass
class OPV2VScenario:
    """Metadata for one scenario (multiple agents, shared frame indices)."""

    scenario_id: str
    path: Path
    agent_ids: list[str]
    frame_indices: list[int]  # Frame indices common across agents.


# ─── Dataset ───────────────────────────────────────────────────────────────


class OPV2VDataset:
    """OPV2V dataset interface (self-contained; no OpenCOOD install required).

    Uses the validated ``pose @ location`` ego->world transform. For the
    feature-fusion detection backbone (see plan correction C1), OpenCOOD's dataset
    classes are used separately; this loader provides object-level access for the
    world model, trust, and communication layers.
    """

    LIDAR_HZ: float = 10.0  # OPV2V is recorded at 10 Hz.

    def __init__(self, root_path: Path | str, split: str = "test") -> None:
        """Initialize the loader.

        Args:
            root_path: Path to the split directory (e.g. ``.../opv2v/test``).
            split: Split name (informational).
        """
        self._root = Path(root_path)
        self._split = split
        self._scenarios: dict[str, OPV2VScenario] = {}

    # ── Indexing ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Scan the directory tree and build the scenario index (metadata only).

        Does NOT load point clouds — those are loaded lazily in :meth:`get_frame`.
        """
        if not self._root.exists():
            msg = f"OPV2V root not found: {self._root}"
            raise FileNotFoundError(msg)

        self._scenarios.clear()
        for scenario_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            agent_dirs = sorted(a for a in scenario_dir.iterdir() if a.is_dir())
            if not agent_dirs:
                continue

            agent_ids = [a.name for a in agent_dirs]

            # Frame indices per agent, from .yaml filenames.
            per_agent_frames: list[set[int]] = []
            for a in agent_dirs:
                frames = {
                    int(f.stem)
                    for f in a.glob("*.yaml")
                    if f.stem.isdigit()
                }
                per_agent_frames.append(frames)

            # Common frames across all agents (so cooperation is possible).
            common = set.intersection(*per_agent_frames) if per_agent_frames else set()

            self._scenarios[scenario_dir.name] = OPV2VScenario(
                scenario_id=scenario_dir.name,
                path=scenario_dir,
                agent_ids=agent_ids,
                frame_indices=sorted(common),
            )

    def scenario_ids(self) -> list[str]:
        """Return all scenario IDs."""
        return list(self._scenarios.keys())

    def get_scenario(self, scenario_id: str) -> OPV2VScenario:
        """Return scenario metadata by ID."""
        if scenario_id not in self._scenarios:
            msg = f"Scenario {scenario_id} not found (loaded {len(self._scenarios)})"
            raise KeyError(msg)
        return self._scenarios[scenario_id]

    def __len__(self) -> int:
        """Number of scenarios."""
        return len(self._scenarios)

    # ── Frame loading ─────────────────────────────────────────────────────

    def get_frame(
        self,
        scenario_id: str,
        agent_id: str,
        frame_idx: int,
        load_lidar: bool = True,
    ) -> OPV2VFrame:
        """Load one frame for one agent.

        Args:
            scenario_id: Scenario identifier.
            agent_id: Agent (CAV) identifier.
            frame_idx: Frame index (integer; formatted as 6-digit filename).
            load_lidar: Whether to load the (large) point cloud.

        Returns:
            Fully populated :class:`OPV2VFrame`.
        """
        scenario = self.get_scenario(scenario_id)
        agent_dir = scenario.path / agent_id
        stem = f"{frame_idx:06d}"
        yaml_path = agent_dir / f"{stem}.yaml"
        if not yaml_path.exists():
            msg = f"Missing YAML: {yaml_path}"
            raise FileNotFoundError(msg)

        meta = self._load_yaml(yaml_path)
        pose = pose_to_matrix(meta["lidar_pose"])
        gt_objects = self._parse_gt(meta)

        lidar = None
        if load_lidar:
            pcd_path = agent_dir / f"{stem}.pcd"
            if pcd_path.exists():
                lidar = self._load_pcd(pcd_path)

        return OPV2VFrame(
            scenario_id=scenario_id,
            agent_id=agent_id,
            frame_idx=frame_idx,
            timestamp_s=frame_idx / self.LIDAR_HZ,
            pose=pose,
            ego_speed_kmh=float(meta.get("ego_speed", 0.0)),
            lidar_points=lidar,
            gt_objects=gt_objects,
        )

    def get_all_agent_frames(
        self, scenario_id: str, frame_idx: int, load_lidar: bool = False
    ) -> dict[str, OPV2VFrame]:
        """Load the same frame for ALL agents in a scenario.

        Args:
            scenario_id: Scenario identifier.
            frame_idx: Frame index.
            load_lidar: Whether to load point clouds (default False — metadata only).

        Returns:
            Mapping agent_id -> frame.
        """
        scenario = self.get_scenario(scenario_id)
        return {
            aid: self.get_frame(scenario_id, aid, frame_idx, load_lidar=load_lidar)
            for aid in scenario.agent_ids
        }

    # ── Cross-agent object matching ────────────────────────────────────────

    @staticmethod
    def match_objects_across_agents(
        frames: dict[str, OPV2VFrame], tolerance_m: float = 3.0
    ) -> list[dict[str, str]]:
        """Match ground-truth objects seen by multiple agents, in world frame.

        Object dict keys are per-agent, so identity is established by world-frame
        proximity (validated: shared cars match within ~2m).

        Args:
            frames: agent_id -> frame (all same scenario+frame_idx).
            tolerance_m: Max world-frame distance to consider the same object.

        Returns:
            List of match groups; each is a dict {agent_id: object_key}.
        """
        # Collect (agent_id, object_key, world_pos) for every object.
        entries: list[tuple[str, str, npt.NDArray[np.float64]]] = []
        for aid, frame in frames.items():
            world = frame.gt_locations_world()
            for obj, wpos in zip(frame.gt_objects, world, strict=True):
                entries.append((aid, obj.object_key, wpos))

        # Greedy clustering by world-frame proximity.
        used = [False] * len(entries)
        groups: list[dict[str, str]] = []
        for i in range(len(entries)):
            if used[i]:
                continue
            aid_i, key_i, pos_i = entries[i]
            group = {aid_i: key_i}
            used[i] = True
            for j in range(i + 1, len(entries)):
                if used[j]:
                    continue
                aid_j, key_j, pos_j = entries[j]
                if aid_j in group:
                    continue  # One object per agent per group.
                if float(np.linalg.norm(pos_i - pos_j)) <= tolerance_m:
                    group[aid_j] = key_j
                    used[j] = True
            groups.append(group)
        return groups

    # ── Internal parsing ──────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load an OPV2V YAML (contains embedded numpy objects → unsafe_load)."""
        with open(path) as f:
            # OPV2V YAMLs embed numpy objects; safe_load cannot reconstruct them.
            # These are trusted research data files.
            return yaml.unsafe_load(f)  # noqa: S506

    @staticmethod
    def _parse_gt(meta: dict[str, Any]) -> list[GroundTruthObject]:
        """Parse the ``vehicles`` block into ground-truth objects (ego frame)."""
        objects: list[GroundTruthObject] = []
        for key, v in meta.get("vehicles", {}).items():
            ext = v["extent"]  # HALF dimensions.
            objects.append(
                GroundTruthObject(
                    object_key=str(key),
                    obj_type=str(v.get("obj_type", "Car")),
                    location_ego=np.array(v["location"], dtype=np.float64),
                    dimensions=np.array(
                        [2.0 * ext[0], 2.0 * ext[1], 2.0 * ext[2]], dtype=np.float64
                    ),
                    yaw_rad=math.radians(v["angle"][1]),  # angle = [roll, yaw, pitch] deg.
                    ass_id=int(v.get("ass_id", -1)),
                )
            )
        return objects

    @staticmethod
    def _load_pcd(path: Path) -> npt.NDArray[np.float32]:
        """Load a LiDAR point cloud as (N, 4) — x, y, z, intensity.

        Uses open3d if available; falls back to a minimal ASCII/binary PCD reader.
        """
        try:
            import open3d as o3d  # noqa: PLC0415

            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)
            # open3d drops intensity; pad a zero intensity column for a stable (N,4).
            intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
            return np.concatenate([pts, intensity], axis=1)
        except ImportError:
            # Minimal fallback: parse PCD header + ASCII data.
            return OPV2VDataset._read_pcd_ascii(path)

    @staticmethod
    def _read_pcd_ascii(path: Path) -> npt.NDArray[np.float32]:
        """Very small ASCII PCD reader (fallback when open3d is unavailable)."""
        with open(path, "rb") as f:
            data_start = False
            rows: list[list[float]] = []
            for raw in f:
                line = raw.decode("latin-1").strip()
                if not data_start:
                    if line.startswith("DATA"):
                        if "ascii" not in line:
                            msg = "Binary PCD requires open3d; please install it."
                            raise RuntimeError(msg)
                        data_start = True
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    vals = [float(p) for p in parts[:4]]
                    while len(vals) < 4:
                        vals.append(0.0)
                    rows.append(vals)
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), np.float32)

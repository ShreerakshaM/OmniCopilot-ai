"""3D Object Detection wrapper.

Wraps pre-trained detection models (OpenPCDet / OpenCOOD) behind a common interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Detection3D:
    """A single 3D detection from the perception model."""

    # Bounding box center (x, y, z) in world coordinates.
    position: npt.NDArray[np.float64]  # shape (3,)
    # Bounding box dimensions (length, width, height).
    dimensions: npt.NDArray[np.float64]  # shape (3,)
    # Heading angle in radians.
    heading: float
    # Velocity estimate (vx, vy, vz).
    velocity: npt.NDArray[np.float64]  # shape (3,)
    # Object class index.
    class_id: int
    # Per-class confidence scores.
    class_scores: npt.NDArray[np.float64]  # shape (num_classes,)
    # Overall detection confidence.
    confidence: float
    # Track ID (stable across frames, -1 if no tracking).
    track_id: int = -1


@dataclass
class DetectionResult:
    """Batch of detections from one frame."""

    detections: list[Detection3D] = field(default_factory=list)
    frame_id: int = 0
    timestamp_s: float = 0.0
    # Inference metadata.
    inference_time_ms: float = 0.0


class BaseDetector(ABC):
    """Abstract base class for 3D object detectors."""

    @abstractmethod
    def detect(self, sensor_data: dict[str, Any]) -> DetectionResult:
        """Run detection on sensor data.

        Args:
            sensor_data: Dictionary containing sensor inputs.
                Expected keys depend on the model (e.g., "lidar_points", "camera_images").

        Returns:
            Detection result with all detections for this frame.
        """

    @abstractmethod
    def get_num_classes(self) -> int:
        """Return the number of object classes this detector supports."""

    @abstractmethod
    def get_class_names(self) -> list[str]:
        """Return ordered list of class names."""


class PointPillarsDetector(BaseDetector):
    """PointPillars-based 3D object detector.

    Wraps OpenPCDet's PointPillars implementation for LiDAR-based detection.
    """

    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        """Initialize PointPillars detector.

        Args:
            config_path: Path to OpenPCDet model config YAML.
            checkpoint_path: Path to trained model checkpoint.
        """
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._model: Any = None  # Lazy-loaded.

    def detect(self, sensor_data: dict[str, Any]) -> DetectionResult:
        """Run PointPillars detection on LiDAR point cloud."""
        # TODO: Implement — load model lazily, run inference.
        raise NotImplementedError

    def get_num_classes(self) -> int:
        """Return number of classes (vehicle, pedestrian, cyclist)."""
        return 3

    def get_class_names(self) -> list[str]:
        """Return class names."""
        return ["vehicle", "pedestrian", "cyclist"]



# ─── OpenCOOD single-agent detector (Phase 2 — inference only, no training) ──────
#
# This is the chosen Phase 2 detector (see docs/task_2.1_detector_integration_spec.md).
# It wraps an OpenCOOD *pretrained* PointPillar model on OPV2V and runs the
# NO-FUSION (single-agent) config first to establish the single-agent baseline mAP.
#
# CRITICAL — how we load OpenCOOD WITHOUT its full install:
#   `python setup.py install` / `pip install -e .` on OpenCOOD pins numba==0.49.0,
#   which fails on modern Python (same class of failure we hit before). So we DO NOT
#   install OpenCOOD as a package. Instead we add the cloned repo to sys.path and
#   import the specific model + utility modules we need. The CUDA rotated-NMS/IoU ops
#   OpenCOOD ships may need compilation; if they are unavailable we fall back to the
#   pure-Python BEV IoU + a simple score-threshold + greedy NMS (see _fallback_nms).
#
# Everything below the model-load boundary is plain tensor/array plumbing and is
# testable without a GPU. Only `_lazy_load_model` and the forward pass need Kaggle GPU.


class OpenCOODDetector(BaseDetector):
    """Single-agent 3D detector backed by an OpenCOOD pretrained PointPillar model.

    Usage (on Kaggle, GPU session):
        det = OpenCOODDetector(
            opencood_repo="/kaggle/working/OpenCOOD",
            config_path=".../point_pillar_single.yaml",   # NO-fusion config
            checkpoint_path=".../net_epochXX.pth",
        )
        result = det.detect({"lidar_points": pts_Nx4})    # ego-frame LiDAR (N,4)
    """

    # OPV2V is predominantly vehicles; OpenCOOD single-class models detect "vehicle".
    _CLASS_NAMES = ["vehicle"]

    def __init__(
        self,
        opencood_repo: str,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        score_threshold: float = 0.3,
    ) -> None:
        """Initialize the OpenCOOD detector wrapper (does NOT load the model yet).

        Args:
            opencood_repo: Path to the cloned OpenCOOD repo (added to sys.path; NOT
                pip-installed — avoids the numba==0.49.0 pin).
            config_path: Path to the OpenCOOD model config YAML (single-agent / no-fusion).
            checkpoint_path: Path to the pretrained .pth checkpoint.
            device: "cuda" or "cpu".
            score_threshold: Minimum detection score to keep.
        """
        self._repo = opencood_repo
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._score_threshold = score_threshold
        self._model: Any = None       # lazy-loaded torch model
        self._hypes: Any = None        # OpenCOOD parsed config dict

    # ── Model loading (GPU-only; fill in on Kaggle) ──────────────────────────
    def _lazy_load_model(self) -> None:
        """Load the OpenCOOD model + weights on first use.

        TODO (Kaggle GPU): the concrete import paths below are OpenCOOD's public API.
        Verify against the pinned OpenCOOD commit and adjust if the repo layout differs.
        """
        if self._model is not None:
            return

        import sys  # noqa: PLC0415

        if self._repo not in sys.path:
            sys.path.insert(0, self._repo)

        # NOTE: import OpenCOOD modules directly — do NOT `pip install` the repo.
        import torch  # noqa: PLC0415
        from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: PLC0415
        from opencood.tools import train_utils  # noqa: PLC0415

        self._hypes = load_yaml(self._config_path)
        # build_model / load_saved_model are OpenCOOD's standard entrypoints.
        model = train_utils.create_model(self._hypes)
        model, _ = train_utils.load_saved_model_from_path(  # TODO: confirm exact fn name
            self._checkpoint_path, model
        )
        model.to(self._device)
        model.eval()
        self._model = model

    # ── Inference ────────────────────────────────────────────────────────────
    def detect(self, sensor_data: dict[str, Any]) -> DetectionResult:
        """Run single-agent detection on one agent's LiDAR point cloud.

        Args:
            sensor_data: must contain "lidar_points" -> (N, 4) x,y,z,intensity in the
                agent's EGO frame. Optionally "frame_id", "timestamp_s".

        Returns:
            DetectionResult with Detection3D objects. Positions are in the EGO frame;
            the caller transforms to world frame via the agent pose (loader's
            validated pose @ location), exactly as feed_observations_demo.py does.
        """
        import time  # noqa: PLC0415

        self._lazy_load_model()
        points = np.asarray(sensor_data["lidar_points"], dtype=np.float32).reshape(-1, 4)

        t0 = time.perf_counter()
        # TODO (Kaggle GPU): preprocess `points` into the model's expected batch dict
        # using OpenCOOD's preprocessor from self._hypes, run forward under no_grad,
        # and post-process to (boxes_7, scores). Pseudocode:
        #
        #   batch = self._preprocess(points)                 # -> dict of tensors on device
        #   with torch.no_grad():
        #       output = self._model(batch)
        #   boxes_7, scores = self._postprocess(output)      # (K,7), (K,)
        #
        # Until implemented on GPU, raise clearly.
        raise NotImplementedError(
            "OpenCOOD forward pass not yet wired — implement _preprocess/_postprocess "
            "on the Kaggle GPU session using the pretrained checkpoint."
        )

        # boxes_7: (K,7) x,y,z,l,w,h,heading ; scores: (K,)
        # detections = self._to_detections(boxes_7, scores)
        # infer_ms = (time.perf_counter() - t0) * 1000.0
        # return DetectionResult(
        #     detections=detections,
        #     frame_id=int(sensor_data.get("frame_id", 0)),
        #     timestamp_s=float(sensor_data.get("timestamp_s", 0.0)),
        #     inference_time_ms=infer_ms,
        # )

    # ── Plumbing (GPU-free; safe to unit-test) ───────────────────────────────
    def _to_detections(
        self,
        boxes_7: npt.NDArray[np.float64],
        scores: npt.NDArray[np.float64],
    ) -> list[Detection3D]:
        """Convert model output (K,7 boxes + K scores) into Detection3D objects.

        Box format (matches evaluation/metrics.py): x, y, z, length, width, height, heading.
        This is the single source of truth for box layout across the project.
        """
        boxes_7 = np.asarray(boxes_7, dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        out: list[Detection3D] = []
        for box, score in zip(boxes_7, scores):
            if score < self._score_threshold:
                continue
            out.append(
                Detection3D(
                    position=box[:3].copy(),
                    dimensions=box[3:6].copy(),
                    heading=float(box[6]),
                    velocity=np.zeros(3, dtype=np.float64),
                    class_id=0,
                    class_scores=np.array([score], dtype=np.float64),
                    confidence=float(score),
                )
            )
        return out

    @staticmethod
    def detections_to_boxes(dets: list[Detection3D]) -> npt.NDArray[np.float64]:
        """Pack detections into an (N, 8) array x,y,z,l,w,h,heading,score for compute_map."""
        if not dets:
            return np.zeros((0, 8), dtype=np.float64)
        rows = [
            [*d.position.tolist(), *d.dimensions.tolist(), d.heading, d.confidence]
            for d in dets
        ]
        return np.asarray(rows, dtype=np.float64)

    def get_num_classes(self) -> int:
        """Single-class (vehicle) for the OPV2V PointPillar baseline."""
        return len(self._CLASS_NAMES)

    def get_class_names(self) -> list[str]:
        """Return class names."""
        return list(self._CLASS_NAMES)

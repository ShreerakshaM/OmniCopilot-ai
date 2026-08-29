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

"""Evaluation metrics for perception, communication, and safety."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class PerceptionMetrics:
    """Perception quality metrics."""

    mean_average_precision: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    occluded_detection_rate: float = 0.0
    mean_position_error_m: float = 0.0


@dataclass
class CommunicationMetrics:
    """Communication efficiency metrics."""

    bandwidth_used_bps: float = 0.0
    bandwidth_utilization: float = 0.0
    messages_per_second: float = 0.0
    information_efficiency: float = 0.0  # Value delivered per byte.
    redundant_message_rate: float = 0.0
    novelty_hit_rate: float = 0.0


@dataclass
class SafetyMetrics:
    """Safety-related metrics."""

    hazard_detection_rate: float = 0.0
    mean_time_to_detection_s: float = 0.0
    false_alarm_rate: float = 0.0
    missed_hazard_rate: float = 0.0


def compute_3d_iou(
    boxes_a: npt.NDArray[np.float64],
    boxes_b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute 3D IoU between two sets of bounding boxes.

    Args:
        boxes_a: Boxes (N, 7) — x, y, z, length, width, height, heading.
        boxes_b: Boxes (M, 7) — same format.

    Returns:
        IoU matrix, shape (N, M).
    """
    # TODO: Implement rotated 3D IoU computation.
    raise NotImplementedError


def compute_map(
    predictions: list[npt.NDArray[np.float64]],
    ground_truths: list[npt.NDArray[np.float64]],
    iou_thresholds: npt.NDArray[np.float64] | None = None,
) -> PerceptionMetrics:
    """Compute mean Average Precision (mAP) for 3D detection.

    Args:
        predictions: Per-frame predictions, each (N, 8) — x,y,z,l,w,h,heading,score.
        ground_truths: Per-frame GT boxes, each (M, 7).
        iou_thresholds: IoU thresholds for TP matching.

    Returns:
        Full perception metrics.
    """
    # TODO: Implement proper mAP computation with per-class breakdown.
    raise NotImplementedError

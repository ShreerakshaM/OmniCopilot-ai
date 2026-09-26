"""Unit tests for OpenCOODDetector GPU-free plumbing (box conversion).

Locks the box-layout contract shared with evaluation/metrics.py:
x, y, z, length, width, height, heading  (+ score for predictions).
Does NOT touch the model / GPU path.
"""

from __future__ import annotations

import numpy as np

from omnicopilot.evaluation.metrics import compute_map
from omnicopilot.perception.detector import Detection3D, OpenCOODDetector


def _make_detector(score_threshold=0.3):
    # Constructor does not load the model, so this is safe without GPU/OpenCOOD.
    return OpenCOODDetector(
        opencood_repo="/nonexistent",
        config_path="/nonexistent.yaml",
        checkpoint_path="/nonexistent.pth",
        device="cpu",
        score_threshold=score_threshold,
    )


def test_to_detections_filters_by_score():
    det = _make_detector(score_threshold=0.5)
    boxes = np.array(
        [
            [0, 0, 0, 4, 2, 1.5, 0.0],
            [10, 0, 0, 4, 2, 1.5, 0.0],
        ]
    )
    scores = np.array([0.9, 0.2])  # second below threshold
    out = det._to_detections(boxes, scores)
    assert len(out) == 1
    assert abs(out[0].confidence - 0.9) < 1e-9
    assert np.allclose(out[0].position, [0, 0, 0])
    assert np.allclose(out[0].dimensions, [4, 2, 1.5])


def test_detections_to_boxes_roundtrip_shape():
    dets = [
        Detection3D(
            position=np.array([1.0, 2.0, 3.0]),
            dimensions=np.array([4.0, 2.0, 1.5]),
            heading=0.5,
            velocity=np.zeros(3),
            class_id=0,
            class_scores=np.array([0.8]),
            confidence=0.8,
        )
    ]
    arr = OpenCOODDetector.detections_to_boxes(dets)
    assert arr.shape == (1, 8)
    assert np.allclose(arr[0], [1, 2, 3, 4, 2, 1.5, 0.5, 0.8])


def test_detections_to_boxes_empty():
    assert OpenCOODDetector.detections_to_boxes([]).shape == (0, 8)


def test_detector_output_feeds_compute_map():
    """End-to-end contract: detector boxes -> compute_map with GT of same layout."""
    det = _make_detector(score_threshold=0.0)
    boxes = np.array([[0, 0, 0, 4, 2, 1.5, 0.0]])
    scores = np.array([0.9])
    dets = det._to_detections(boxes, scores)
    preds = [OpenCOODDetector.detections_to_boxes(dets)]  # (1,8)
    gt = [np.array([[0, 0, 0, 4, 2, 1.5, 0.0]])]  # (1,7) same object
    m = compute_map(preds, gt, iou_thresholds=np.array([0.7]))
    assert abs(m.mean_average_precision - 1.0) < 1e-6

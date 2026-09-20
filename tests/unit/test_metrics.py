"""Unit tests for perception evaluation metrics (BEV IoU + AP)."""

from __future__ import annotations

import numpy as np

from omnicopilot.evaluation.metrics import compute_3d_iou, compute_map


def _box(x, y, z=0.0, length=4.0, width=2.0, height=1.5, heading=0.0):
    return [x, y, z, length, width, height, heading]


def test_iou_identical_boxes_is_one():
    a = np.array([_box(0, 0)])
    iou = compute_3d_iou(a, a)
    assert iou.shape == (1, 1)
    assert abs(iou[0, 0] - 1.0) < 1e-9


def test_iou_disjoint_boxes_is_zero():
    a = np.array([_box(0, 0)])
    b = np.array([_box(100, 100)])
    assert compute_3d_iou(a, b)[0, 0] == 0.0


def test_iou_partial_overlap_between_zero_and_one():
    a = np.array([_box(0, 0, length=4, width=4, height=4)])
    b = np.array([_box(2, 0, length=4, width=4, height=4)])  # shifted half-length in x
    val = compute_3d_iou(a, b)[0, 0]
    assert 0.0 < val < 1.0


def test_iou_empty_inputs():
    empty = np.zeros((0, 7))
    assert compute_3d_iou(empty, np.array([_box(0, 0)])).shape == (0, 1)
    assert compute_3d_iou(np.array([_box(0, 0)]), empty).shape == (1, 0)


def test_map_perfect_detection_is_one():
    gt = [np.array([_box(0, 0), _box(20, 0)])]
    # predictions == GT with score 0.9, appended as (…,7 -> 8) with score column.
    preds = [np.array([_box(0, 0) + [0.9], _box(20, 0) + [0.8]])]
    m = compute_map(preds, gt, iou_thresholds=np.array([0.7]))
    assert abs(m.mean_average_precision - 1.0) < 1e-6
    assert abs(m.recall - 1.0) < 1e-9
    assert abs(m.precision - 1.0) < 1e-9


def test_map_all_false_positives_is_zero():
    gt = [np.array([_box(0, 0)])]
    preds = [np.array([_box(500, 500) + [0.9]])]  # nowhere near GT
    m = compute_map(preds, gt, iou_thresholds=np.array([0.7]))
    assert m.mean_average_precision == 0.0
    assert m.recall == 0.0


def test_map_half_recall():
    # Two GT objects, only one detected -> recall 0.5, precision 1.0.
    gt = [np.array([_box(0, 0), _box(20, 0)])]
    preds = [np.array([_box(0, 0) + [0.9]])]
    m = compute_map(preds, gt, iou_thresholds=np.array([0.7]))
    assert abs(m.recall - 0.5) < 1e-9
    assert abs(m.precision - 1.0) < 1e-9


def test_map_position_error_reported():
    gt = [np.array([_box(0, 0)])]
    # Slightly offset but still high-IoU prediction.
    preds = [np.array([_box(0.3, 0.0) + [0.9]])]
    m = compute_map(preds, gt, iou_thresholds=np.array([0.3]))
    assert m.recall == 1.0
    assert 0.0 < m.mean_position_error_m < 1.0

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
    """Compute Bird's-Eye-View IoU between two sets of 3D boxes.

    We use axis-aligned BEV IoU (footprint overlap in the x-y plane, then the
    z-overlap ratio) as a pragmatic, dependency-free approximation of rotated 3D
    IoU. This matches how OPV2V AP is reported (BEV / 3D AP@IoU=0.7) closely enough
    for baseline evaluation, and avoids pulling in a rotated-IoU CUDA op. Heading is
    ignored for the overlap (a known, documented approximation); it can be tightened
    later with a rotated-polygon IoU if needed.

    Args:
        boxes_a: Boxes (N, 7) — x, y, z, length, width, height, heading.
        boxes_b: Boxes (M, 7) — same format.

    Returns:
        IoU matrix, shape (N, M), values in [0, 1].
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 7)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 7)
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    iou = np.zeros((n, m), dtype=np.float64)
    if n == 0 or m == 0:
        return iou

    def _extents(
        boxes: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
        length, width, height = boxes[:, 3], boxes[:, 4], boxes[:, 5]
        return (
            cx - length / 2.0,
            cx + length / 2.0,
            cy - width / 2.0,
            cy + width / 2.0,
            cz - height / 2.0,
            cz + height / 2.0,
            length * width * height,
        )

    ax0, ax1, ay0, ay1, az0, az1, avol = _extents(boxes_a)
    bx0, bx1, by0, by1, bz0, bz1, bvol = _extents(boxes_b)

    for i in range(n):
        ox = np.clip(np.minimum(ax1[i], bx1) - np.maximum(ax0[i], bx0), 0.0, None)
        oy = np.clip(np.minimum(ay1[i], by1) - np.maximum(ay0[i], by0), 0.0, None)
        oz = np.clip(np.minimum(az1[i], bz1) - np.maximum(az0[i], bz0), 0.0, None)
        inter = ox * oy * oz
        union = avol[i] + bvol - inter
        iou[i] = np.where(union > 0.0, inter / union, 0.0)
    return iou


def _average_precision(
    recall: npt.NDArray[np.float64], precision: npt.NDArray[np.float64]
) -> float:
    """Area under the precision-recall curve (VOC-style, all-points interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing.
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(
    predictions: list[npt.NDArray[np.float64]],
    ground_truths: list[npt.NDArray[np.float64]],
    iou_thresholds: npt.NDArray[np.float64] | None = None,
) -> PerceptionMetrics:
    """Compute Average Precision for 3D detection at given IoU threshold(s).

    Greedy per-frame TP/FP assignment (highest-score prediction claims the
    highest-IoU unmatched GT above threshold), then a global PR curve over all
    frames sorted by score. Averages AP over ``iou_thresholds`` (default [0.7],
    the OPV2V headline threshold).

    Args:
        predictions: Per-frame predictions, each (N, 8) — x,y,z,l,w,h,heading,score.
        ground_truths: Per-frame GT boxes, each (M, 7).
        iou_thresholds: IoU thresholds for TP matching. Default [0.7].

    Returns:
        PerceptionMetrics with mAP (mean over thresholds), and precision/recall/F1
        and mean position error reported at the FIRST threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = np.array([0.7], dtype=np.float64)
    iou_thresholds = np.atleast_1d(np.asarray(iou_thresholds, dtype=np.float64))

    total_gt = sum(int(np.asarray(g).reshape(-1, 7).shape[0]) for g in ground_truths)

    aps: list[float] = []
    first_precision = first_recall = first_pos_err = 0.0

    for ti, thr in enumerate(iou_thresholds):
        # Collect (score, is_tp, pos_error) across all frames.
        scored: list[tuple[float, int, float]] = []
        for preds, gts in zip(predictions, ground_truths):
            preds = np.asarray(preds, dtype=np.float64).reshape(-1, 8)
            gts = np.asarray(gts, dtype=np.float64).reshape(-1, 7)
            if preds.shape[0] == 0:
                continue
            order = np.argsort(-preds[:, 7])  # highest score first
            matched = np.zeros(gts.shape[0], dtype=bool)
            if gts.shape[0] > 0:
                iou = compute_3d_iou(preds[:, :7], gts)
            for pi in order:
                is_tp, pos_err = 0, 0.0
                if gts.shape[0] > 0:
                    ious_row = iou[pi].copy()
                    ious_row[matched] = -1.0
                    best = int(np.argmax(ious_row))
                    if ious_row[best] >= thr:
                        matched[best] = True
                        is_tp = 1
                        pos_err = float(np.linalg.norm(preds[pi, :3] - gts[best, :3]))
                scored.append((float(preds[pi, 7]), is_tp, pos_err))

        if not scored:
            aps.append(0.0)
            continue

        scored.sort(key=lambda x: -x[0])
        tp = np.array([s[1] for s in scored], dtype=np.float64)
        fp = 1.0 - tp
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / total_gt if total_gt > 0 else np.zeros_like(tp_cum)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        aps.append(_average_precision(recall, precision))

        if ti == 0:
            n_tp = int(tp.sum())
            n_pred = len(scored)
            first_precision = n_tp / n_pred if n_pred else 0.0
            first_recall = n_tp / total_gt if total_gt else 0.0
            errs = [s[2] for s in scored if s[1] == 1]
            first_pos_err = float(np.mean(errs)) if errs else 0.0

    f1 = (
        2 * first_precision * first_recall / (first_precision + first_recall)
        if (first_precision + first_recall) > 0
        else 0.0
    )

    return PerceptionMetrics(
        mean_average_precision=float(np.mean(aps)) if aps else 0.0,
        precision=first_precision,
        recall=first_recall,
        f1_score=f1,
        mean_position_error_m=first_pos_err,
    )

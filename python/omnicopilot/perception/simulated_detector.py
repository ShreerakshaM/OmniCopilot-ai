"""Simulated 3D detector — a controlled, literature-calibrated stand-in for a CNN.

WHY THIS EXISTS (honest framing):
Running a real pretrained 3D detector (OpenCOOD / MMDetection3D) was blocked by a
platform mismatch: those 2021-era stacks (spconv, mmcv) are incompatible with the
Kaggle runtime's Python 3.12. Rather than stall Phase 2, we model the detector's
IMPERFECTIONS directly on top of ground truth, so we can still measure how
cooperation behaves under realistic detection uncertainty. This is a SIMULATED
detector, not a neural network: every reported result must state that clearly.

The imperfections are calibrated to published V2X numbers so they are defensible,
not arbitrary:
- Localization noise: sigma_xyz in [0, 0.5] m, sigma_heading in [0, 1] deg
  (V2X-ViT "Noisy Setting"; see docs/prior_art.md).
- Detection dropout: probability of MISSING an object rises with range and with
  occlusion. This is the crux -- distant / occluded objects are exactly what a
  single agent loses and what cooperation should recover.
- False positives: a small per-frame rate, so precision is not trivially perfect.

Determinism: pass a seed for reproducible detections (needed for tests and for
stable single-vs-cooperative comparison).

Upgrade path: replacing this with a real detector (2a) is a drop-in swap -- it
produces the same Detection3D objects the downstream pipeline consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from omnicopilot.perception.detector import Detection3D


@dataclass
class DetectorNoiseConfig:
    """Calibrated detection-degradation parameters (see module docstring)."""

    # Localization noise (Gaussian), std dev.
    pos_noise_std_m: float = 0.2  # within V2X-ViT [0, 0.5] m
    heading_noise_std_deg: float = 0.5  # within V2X-ViT [0, 1] deg

    # Detection dropout: base miss rate + range/occlusion-driven increase.
    base_miss_rate: float = 0.05  # even near, clear objects occasionally missed
    range_full_miss_m: float = 100.0  # miss prob approaches high near this range
    max_range_miss_rate: float = 0.85  # miss prob at/after range_full_miss_m
    # Occlusion proxy: an object with many nearer neighbors between it and the
    # sensor is more likely occluded. We approximate with local crowding.
    occlusion_miss_boost: float = 0.30  # extra miss prob for highly-crowded objects

    # False positives.
    false_positive_rate: float = 0.03  # expected FPs per true object
    fp_spread_m: float = 60.0  # FPs scattered within this radius of sensor

    # Confidence model: score decreases with range + noise.
    conf_base: float = 0.95
    conf_range_falloff: float = 0.4  # score reduction at range_full_miss_m


class SimulatedDetector:
    """Turns ground-truth objects (world frame) into noisy per-agent detections."""

    def __init__(self, config: DetectorNoiseConfig | None = None, seed: int | None = None) -> None:
        """Initialize.

        Args:
            config: noise/dropout parameters.
            seed: RNG seed for reproducible detections.
        """
        self.cfg = config or DetectorNoiseConfig()
        self._rng = np.random.default_rng(seed)

    def _miss_probability(
        self, ranges: npt.NDArray[np.float64], crowding: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Per-object probability of being MISSED, from range + occlusion proxy."""
        c = self.cfg
        # Range term: linear ramp from base_miss_rate to max_range_miss_rate.
        frac = np.clip(ranges / c.range_full_miss_m, 0.0, 1.0)
        range_term = c.base_miss_rate + frac * (c.max_range_miss_rate - c.base_miss_rate)
        # Occlusion term: normalized crowding (0..1) scaled by boost.
        occ_term = c.occlusion_miss_boost * crowding
        return np.clip(range_term + occ_term, 0.0, 0.99)

    @staticmethod
    def _crowding(
        centers: npt.NDArray[np.float64], radius_m: float = 8.0
    ) -> npt.NDArray[np.float64]:
        """Normalized local crowding per object (occlusion proxy), in [0, 1]."""
        n = centers.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = np.linalg.norm(centers[:, :2] - centers[i, :2], axis=1)
            counts[i] = np.sum(d <= radius_m) - 1  # exclude self
        if counts.max() > 0:
            return counts / counts.max()
        return counts

    def detect(
        self,
        gt_centers_world: npt.NDArray[np.float64],  # (M, 3)
        gt_dims: npt.NDArray[np.float64],  # (M, 3) l, w, h
        gt_yaw: npt.NDArray[np.float64],  # (M,)
        sensor_xyz: npt.NDArray[np.float64],  # (3,) agent world position
    ) -> list[Detection3D]:
        """Produce noisy detections for one agent viewing the scene from sensor_xyz."""
        gt_centers_world = np.asarray(gt_centers_world, dtype=np.float64).reshape(-1, 3)
        gt_dims = np.asarray(gt_dims, dtype=np.float64).reshape(-1, 3)
        gt_yaw = np.asarray(gt_yaw, dtype=np.float64).reshape(-1)
        sensor_xyz = np.asarray(sensor_xyz, dtype=np.float64).reshape(3)
        m = gt_centers_world.shape[0]
        c = self.cfg
        dets: list[Detection3D] = []
        if m == 0:
            return self._add_false_positives(dets, sensor_xyz, 0)

        ranges = np.linalg.norm(gt_centers_world - sensor_xyz, axis=1)
        crowding = self._crowding(gt_centers_world)
        miss_p = self._miss_probability(ranges, crowding)
        keep = self._rng.random(m) >= miss_p  # True = detected

        for i in np.where(keep)[0]:
            # Localization noise.
            pos = gt_centers_world[i] + self._rng.normal(0.0, c.pos_noise_std_m, size=3)
            yaw = gt_yaw[i] + np.deg2rad(self._rng.normal(0.0, c.heading_noise_std_deg))
            # Confidence falls off with range.
            frac = min(ranges[i] / c.range_full_miss_m, 1.0)
            conf = float(np.clip(c.conf_base - frac * c.conf_range_falloff, 0.05, 1.0))
            dets.append(
                Detection3D(
                    position=pos,
                    dimensions=gt_dims[i].copy(),
                    heading=float(yaw),
                    velocity=np.zeros(3, dtype=np.float64),
                    class_id=0,
                    class_scores=np.array([conf], dtype=np.float64),
                    confidence=conf,
                )
            )
        return self._add_false_positives(dets, sensor_xyz, m)

    def _add_false_positives(
        self, dets: list[Detection3D], sensor_xyz: npt.NDArray[np.float64], n_true: int
    ) -> list[Detection3D]:
        """Sprinkle a few low-confidence phantom detections near the sensor."""
        c = self.cfg
        n_fp = self._rng.poisson(c.false_positive_rate * max(n_true, 1))
        for _ in range(int(n_fp)):
            offset = self._rng.uniform(-c.fp_spread_m, c.fp_spread_m, size=3)
            offset[2] = 0.0
            pos = sensor_xyz + offset
            conf = float(self._rng.uniform(0.1, 0.4))  # FPs tend to be low-confidence
            dets.append(
                Detection3D(
                    position=pos,
                    dimensions=np.array([4.5, 2.0, 1.5], dtype=np.float64),
                    heading=0.0,
                    velocity=np.zeros(3, dtype=np.float64),
                    class_id=0,
                    class_scores=np.array([conf], dtype=np.float64),
                    confidence=conf,
                )
            )
        return dets

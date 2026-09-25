"""Unit tests for the SimulatedDetector (deterministic via seed)."""

from __future__ import annotations

import numpy as np

from omnicopilot.perception.simulated_detector import (
    DetectorNoiseConfig,
    SimulatedDetector,
)


def _grid(n, spacing=6.0, base=(10.0, 0.0, 0.0)):
    """n objects in a line starting at base, spaced along x."""
    return np.array([[base[0] + i * spacing, base[1], base[2]] for i in range(n)],
                    dtype=np.float64)


def _dims(n):
    return np.tile(np.array([4.5, 2.0, 1.5]), (n, 1))


def test_near_objects_mostly_detected():
    # No dropout except the range ramp; objects close to sensor -> high detection.
    det = SimulatedDetector(DetectorNoiseConfig(base_miss_rate=0.0,
                                                occlusion_miss_boost=0.0,
                                                false_positive_rate=0.0),
                            seed=0)
    centers = _grid(10, spacing=1.0, base=(5.0, 0.0, 0.0))  # all within ~15 m
    out = det.detect(centers, _dims(10), np.zeros(10), sensor_xyz=np.zeros(3))
    # Near objects have only a small range-driven miss prob (~4-12%), so most detected.
    assert len(out) >= 7


def test_far_objects_mostly_missed():
    det = SimulatedDetector(DetectorNoiseConfig(false_positive_rate=0.0), seed=0)
    # Objects far beyond range_full_miss_m (100 m) -> high miss prob.
    centers = _grid(20, spacing=1.0, base=(150.0, 0.0, 0.0))
    out = det.detect(centers, _dims(20), np.zeros(20), sensor_xyz=np.zeros(3))
    # Most should be dropped at this range.
    assert len(out) < 8


def test_localization_noise_is_bounded():
    cfg = DetectorNoiseConfig(base_miss_rate=0.0, occlusion_miss_boost=0.0,
                              false_positive_rate=0.0, pos_noise_std_m=0.2)
    det = SimulatedDetector(cfg, seed=42)
    centers = _grid(50, spacing=2.0, base=(5.0, 0.0, 0.0))
    out = det.detect(centers, _dims(50), np.zeros(50), sensor_xyz=np.zeros(3))
    # Each detection should be near some GT center (within a few sigma).
    for d in out:
        nearest = np.min(np.linalg.norm(centers - d.position, axis=1))
        assert nearest < 1.5  # ~7 sigma bound on 0.2 m noise


def test_determinism_same_seed_same_output():
    cfg = DetectorNoiseConfig()
    a = SimulatedDetector(cfg, seed=7).detect(_grid(15), _dims(15), np.zeros(15), np.zeros(3))
    b = SimulatedDetector(cfg, seed=7).detect(_grid(15), _dims(15), np.zeros(15), np.zeros(3))
    assert len(a) == len(b)
    for da, db in zip(a, b):
        assert np.allclose(da.position, db.position)
        assert da.confidence == db.confidence


def test_confidence_falls_with_range():
    # Disable dropout entirely so both near and far objects are always detected,
    # isolating the confidence-vs-range behavior.
    cfg = DetectorNoiseConfig(base_miss_rate=0.0, occlusion_miss_boost=0.0,
                              max_range_miss_rate=0.0, false_positive_rate=0.0,
                              pos_noise_std_m=0.0)
    det = SimulatedDetector(cfg, seed=1)
    near = det.detect(np.array([[10.0, 0, 0]]), _dims(1), np.zeros(1), np.zeros(3))
    far = det.detect(np.array([[90.0, 0, 0]]), _dims(1), np.zeros(1), np.zeros(3))
    assert near and far  # dropout disabled -> both detected
    assert near[0].confidence > far[0].confidence


def test_empty_scene():
    det = SimulatedDetector(DetectorNoiseConfig(false_positive_rate=0.0), seed=0)
    out = det.detect(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(3))
    assert out == []

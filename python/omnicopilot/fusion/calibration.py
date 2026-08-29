"""Confidence calibration — ensure predicted confidences match true accuracy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class TemperatureScaling:
    """Temperature scaling for confidence calibration.

    Learns a single temperature parameter T such that softmax(logits/T)
    produces well-calibrated probabilities.
    """

    def __init__(self) -> None:
        """Initialize with temperature = 1.0 (uncalibrated)."""
        self._temperature: float = 1.0
        self._fitted: bool = False

    def fit(
        self,
        logits: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
    ) -> None:
        """Fit temperature on a validation set.

        Args:
            logits: Model logits, shape (N, num_classes).
            labels: True class indices, shape (N,).
        """
        # TODO: Implement — optimize temperature using NLL on validation set.
        raise NotImplementedError

    def calibrate(self, confidences: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply temperature scaling to raw confidences.

        Args:
            confidences: Raw confidence scores, shape (N,).

        Returns:
            Calibrated confidences, shape (N,).
        """
        if not self._fitted:
            return confidences
        # TODO: Apply learned temperature.
        raise NotImplementedError

    @property
    def temperature(self) -> float:
        """Return the learned temperature parameter."""
        return self._temperature


def expected_calibration_error(
    confidences: npt.NDArray[np.float64],
    accuracies: npt.NDArray[np.float64],
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: Predicted confidences, shape (N,).
        accuracies: Binary correctness labels, shape (N,).
        num_bins: Number of confidence bins.

    Returns:
        ECE score (lower is better).
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return float(ece)

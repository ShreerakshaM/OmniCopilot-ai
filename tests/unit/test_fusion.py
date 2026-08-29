"""Unit tests for fusion algorithms."""

from __future__ import annotations

import numpy as np

from omnicopilot.fusion.algorithms import FusedEstimate, WeightedAverageFusion


class TestWeightedAverageFusion:
    """Tests for weighted average fusion."""

    def test_single_observation(self) -> None:
        """Single observation returns itself."""
        fusion = WeightedAverageFusion()
        result = fusion.fuse(
            positions=np.array([[10.0, 20.0, 0.0]]),
            confidences=np.array([0.9]),
            trusts=np.array([1.0]),
        )
        np.testing.assert_allclose(result.position, [10.0, 20.0, 0.0])
        assert result.source_count == 1

    def test_two_equal_observations(self) -> None:
        """Two equal-weight observations average to midpoint."""
        fusion = WeightedAverageFusion()
        result = fusion.fuse(
            positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            confidences=np.array([0.9, 0.9]),
            trusts=np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(result.position, [5.0, 0.0, 0.0])
        assert result.source_count == 2

    def test_trust_affects_weight(self) -> None:
        """Higher trust pulls the estimate toward that observation."""
        fusion = WeightedAverageFusion()
        result = fusion.fuse(
            positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            confidences=np.array([0.9, 0.9]),
            trusts=np.array([0.1, 0.9]),  # Second agent much more trusted.
        )
        # Should be closer to [10, 0, 0].
        assert result.position[0] > 5.0

    def test_zero_trust_excluded(self) -> None:
        """Agent with zero trust doesn't contribute."""
        fusion = WeightedAverageFusion()
        result = fusion.fuse(
            positions=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            confidences=np.array([0.9, 0.9]),
            trusts=np.array([1.0, 0.0]),  # Second agent has zero trust.
        )
        np.testing.assert_allclose(result.position, [0.0, 0.0, 0.0], atol=1e-6)

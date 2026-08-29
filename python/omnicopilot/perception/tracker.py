"""Multi-Object Tracking (MOT) for stable object identities across frames."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from omnicopilot.perception.detector import Detection3D


@dataclass
class Track:
    """A tracked object with stable identity."""

    track_id: int
    class_id: int
    position: npt.NDArray[np.float64]  # shape (3,)
    velocity: npt.NDArray[np.float64]  # shape (3,)
    heading: float
    confidence: float
    age: int = 0  # Frames since first seen.
    hits: int = 0  # Frames with a matched detection.
    misses: int = 0  # Consecutive frames without a match.


class MultiObjectTracker:
    """Simple IoU-based multi-object tracker.

    Associates detections to tracks using 3D IoU or center-distance matching.
    Uses Hungarian algorithm for optimal assignment.
    """

    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 3,
        association_threshold: float = 3.0,
    ) -> None:
        """Initialize tracker.

        Args:
            max_age: Maximum frames without detection before track is removed.
            min_hits: Minimum detections before track is confirmed.
            association_threshold: Maximum distance for detection-to-track association.
        """
        self._max_age = max_age
        self._min_hits = min_hits
        self._association_threshold = association_threshold
        self._tracks: list[Track] = []
        self._next_id: int = 0

    def update(self, detections: list[Detection3D]) -> list[Track]:
        """Update tracks with new detections.

        Args:
            detections: Detections from the current frame.

        Returns:
            List of confirmed tracks (age >= min_hits).
        """
        # TODO: Implement Hungarian association + track management.
        raise NotImplementedError

    def get_active_tracks(self) -> list[Track]:
        """Get all currently active (not deleted) tracks."""
        return [t for t in self._tracks if t.misses <= self._max_age]

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0

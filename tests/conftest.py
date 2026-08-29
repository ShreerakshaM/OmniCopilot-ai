"""Shared test fixtures for OmniCopilot."""

from __future__ import annotations

import numpy as np
import pytest

from omnicopilot.data.schemas import ObservationSchema, Position3DSchema, Velocity3DSchema


@pytest.fixture
def sample_observation() -> ObservationSchema:
    """A sample observation for testing."""
    return ObservationSchema(
        observation_id="obs_001",
        agent_id="agent_1",
        track_id="track_1",
        object_class=2,  # Pedestrian
        position=Position3DSchema(x=10.0, y=20.0, z=0.0),
        velocity=Velocity3DSchema(vx=1.2, vy=0.3, vz=0.0),
        confidence=0.92,
        timestamp_s=1.0,
        sensor_type=1,  # LiDAR
    )


@pytest.fixture
def sample_observations_batch() -> list[ObservationSchema]:
    """Batch of observations from multiple agents."""
    rng = np.random.default_rng(42)
    observations = []
    for i in range(5):
        observations.append(
            ObservationSchema(
                observation_id=f"obs_{i:03d}",
                agent_id=f"agent_{i % 3}",
                track_id=f"track_{i // 2}",
                object_class=1,  # Vehicle
                position=Position3DSchema(
                    x=float(rng.uniform(0, 100)),
                    y=float(rng.uniform(0, 100)),
                    z=0.0,
                ),
                confidence=float(rng.uniform(0.5, 1.0)),
                timestamp_s=float(i * 0.1),
            )
        )
    return observations

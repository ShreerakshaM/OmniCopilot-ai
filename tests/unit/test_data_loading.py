"""Unit tests for data loading and schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnicopilot.data.schemas import (
    ExperimentConfigSchema,
    ObservationSchema,
    Position3DSchema,
)


class TestObservationSchema:
    """Tests for ObservationSchema validation."""

    def test_valid_observation(self, sample_observation: ObservationSchema) -> None:
        """Valid observation passes validation."""
        assert sample_observation.confidence == 0.92
        assert sample_observation.agent_id == "agent_1"

    def test_confidence_out_of_range(self) -> None:
        """Confidence > 1.0 raises validation error."""
        with pytest.raises(ValidationError):
            ObservationSchema(
                observation_id="test",
                agent_id="agent_1",
                position=Position3DSchema(x=0, y=0, z=0),
                confidence=1.5,  # Invalid!
                timestamp_s=0.0,
            )

    def test_negative_timestamp_rejected(self) -> None:
        """Negative timestamp raises validation error."""
        with pytest.raises(ValidationError):
            ObservationSchema(
                observation_id="test",
                agent_id="agent_1",
                position=Position3DSchema(x=0, y=0, z=0),
                confidence=0.5,
                timestamp_s=-1.0,  # Invalid!
            )


class TestExperimentConfigSchema:
    """Tests for experiment config validation."""

    def test_valid_config(self) -> None:
        """Valid experiment config passes."""
        config = ExperimentConfigSchema(
            experiment_name="test_run",
            num_agents=5,
            dataset="opv2v",
        )
        assert config.num_agents == 5

    def test_too_many_agents(self) -> None:
        """num_agents > 20 raises validation error."""
        with pytest.raises(ValidationError):
            ExperimentConfigSchema(
                experiment_name="test",
                num_agents=100,  # Invalid!
                dataset="opv2v",
            )

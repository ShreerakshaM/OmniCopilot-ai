"""Pydantic schemas for data validation.

These schemas validate data flowing through the system, ensuring
type safety at runtime boundaries (file loading, API responses, etc.).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class ObjectClassEnum(IntEnum):
    """Object classification."""

    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    MOTORCYCLE = 4
    TRUCK = 5
    BUS = 6
    STATIC_OBSTACLE = 7


class Position3DSchema(BaseModel):
    """3D position in world coordinates."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Velocity3DSchema(BaseModel):
    """3D velocity vector."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


class ObservationSchema(BaseModel):
    """Validated observation from an agent's perception."""

    observation_id: str
    agent_id: str
    track_id: str = ""
    object_class: ObjectClassEnum = ObjectClassEnum.UNKNOWN
    position: Position3DSchema
    velocity: Velocity3DSchema = Velocity3DSchema()
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp_s: float = Field(ge=0.0)
    sensor_type: int = 0


class EntitySchema(BaseModel):
    """Validated entity in the world model."""

    entity_id: str
    object_class: ObjectClassEnum = ObjectClassEnum.UNKNOWN
    position: Position3DSchema
    velocity: Velocity3DSchema = Velocity3DSchema()
    confidence: float = Field(ge=0.0, le=1.0)
    source_count: int = Field(ge=0)
    is_safety_critical: bool = False
    needs_corroboration: bool = True


class ExperimentConfigSchema(BaseModel):
    """Validated experiment configuration."""

    experiment_name: str
    num_agents: int = Field(ge=1, le=20)
    dataset: str
    communication_policy: str = "hand_designed"
    network_profile: str = "unlimited"
    seed: int = 42

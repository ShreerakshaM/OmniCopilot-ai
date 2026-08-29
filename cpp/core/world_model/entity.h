// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Entity — A tracked object in the world model.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace omnicopilot {

/// 3D position in world coordinates (meters).
struct Position3D {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  double DistanceTo(const Position3D& other) const;
};

/// 3D velocity vector (meters/second).
struct Velocity3D {
  double vx = 0.0;
  double vy = 0.0;
  double vz = 0.0;

  double Magnitude() const;
};

/// Object classification.
enum class ObjectClass : int {
  kUnknown = 0,
  kVehicle = 1,
  kPedestrian = 2,
  kCyclist = 3,
  kMotorcycle = 4,
  kTruck = 5,
  kBus = 6,
  kStaticObstacle = 7,
  kTrafficSign = 8,
  kTrafficLight = 9,
};

/// Entity lifecycle state.
enum class EntityState : int {
  kTentative = 0,   // Seen by one source, not yet confirmed.
  kConfirmed = 1,   // Corroborated by multiple sources.
  kPredicted = 2,   // No recent observation, position predicted.
  kStale = 3,       // Not observed recently, confidence decaying.
  kLost = 4,        // Marked for removal.
};

/// Threat level classification.
enum class ThreatLevel : int {
  kNone = 0,
  kLow = 1,
  kMedium = 2,
  kHigh = 3,
  kCritical = 4,
};

/// A single raw observation from an agent's perception system.
struct Observation {
  std::string observation_id;
  std::string agent_id;
  std::string track_id;

  ObjectClass object_class = ObjectClass::kUnknown;
  Position3D position;
  Velocity3D velocity;
  double confidence = 0.0;
  double timestamp_s = 0.0;
  int sensor_type = 0;

  /// Bounding box dimensions.
  double length = 0.0;
  double width = 0.0;
  double height = 0.0;
  double heading = 0.0;
};

/// Evidence record — provenance for one supporting observation.
struct Evidence {
  std::string agent_id;
  std::string observation_id;
  double confidence = 0.0;
  double timestamp_s = 0.0;
  int sensor_type = 0;
};

/// Predicted future state at a specific time horizon.
struct PredictedState {
  double time_horizon_s = 0.0;
  Position3D position;
  Velocity3D velocity;
  double confidence = 0.0;
};

/// A fused, tracked entity in the world model.
struct TrackedEntity {
  std::string entity_id;

  // Classification (fused).
  ObjectClass object_class = ObjectClass::kUnknown;
  double class_confidence = 0.0;

  // Kinematic state (best estimate).
  Position3D position;
  Velocity3D velocity;
  double heading = 0.0;
  double heading_rate = 0.0;

  // Bounding box (fused).
  double length = 0.0;
  double width = 0.0;
  double height = 0.0;

  // Uncertainty.
  double confidence = 0.0;

  // Lifecycle.
  EntityState state = EntityState::kTentative;
  double first_seen_s = 0.0;
  double last_updated_s = 0.0;
  uint32_t observation_count = 0;
  uint32_t unique_source_count = 0;

  // Evidence provenance.
  std::vector<Evidence> supporting_evidence;

  // Predictions.
  std::vector<PredictedState> predictions;

  // Safety assessment.
  ThreatLevel threat_level = ThreatLevel::kNone;
  double time_to_collision_s = -1.0;

  // Flags.
  bool needs_corroboration = true;
  bool is_safety_critical = false;
  bool active_acquisition_pending = false;
};

}  // namespace omnicopilot

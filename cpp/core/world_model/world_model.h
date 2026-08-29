// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// WorldModel — The central intelligence artifact.
// Maintains a fused, uncertainty-aware, temporal representation of all entities
// observed by the collective agent system.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"
#include "cpp/core/world_model/fusion.h"
#include "cpp/core/world_model/spatial_index.h"
#include "cpp/core/world_model/temporal_tracker.h"
#include "cpp/core/world_model/uncertainty.h"

namespace omnicopilot {

/// Configuration for the world model.
struct WorldModelConfig {
  /// Maximum number of entities to track simultaneously.
  uint32_t max_entities = 1000;

  /// Confidence threshold below which entities are removed.
  double stale_confidence_threshold = 0.1;

  /// Time (seconds) after which an unobserved entity becomes STALE.
  double stale_timeout_s = 2.0;

  /// Time (seconds) after which a stale entity is removed.
  double removal_timeout_s = 5.0;

  /// Minimum confidence for an entity to be CONFIRMED.
  double confirmation_threshold = 0.6;

  /// Minimum unique sources for an entity to be CONFIRMED.
  uint32_t confirmation_source_count = 2;

  /// Spatial index cell size (meters).
  double spatial_cell_size_m = 5.0;
};

/// Snapshot statistics of the world model.
struct WorldModelStats {
  uint32_t total_entities = 0;
  uint32_t confirmed = 0;
  uint32_t tentative = 0;
  uint32_t predicted = 0;
  uint32_t stale = 0;
  double mean_confidence = 0.0;
  uint64_t tick = 0;
};

/// The shared world model — core intelligence layer.
///
/// Thread-safety: NOT thread-safe. Callers must synchronize access externally
/// if used from multiple threads. This is intentional — the runtime loop is
/// single-threaded per agent.
class WorldModel {
 public:
  explicit WorldModel(WorldModelConfig config);
  ~WorldModel();

  // Non-copyable, movable.
  WorldModel(const WorldModel&) = delete;
  WorldModel& operator=(const WorldModel&) = delete;
  WorldModel(WorldModel&&) noexcept;
  WorldModel& operator=(WorldModel&&) noexcept;

  /// Advance the world model by one tick. Updates predictions, decays confidence,
  /// removes stale entities.
  /// @param current_time_s Current simulation time in seconds.
  void Tick(double current_time_s);

  /// Ingest a batch of observations from a single agent.
  /// Performs association, fusion, and entity creation/update.
  /// @param observations Observations from one agent at one timestep.
  /// @param agent_trust Trust score of the reporting agent [0, 1].
  void IngestObservations(const std::vector<Observation>& observations,
                          double agent_trust = 1.0);

  /// Query entities within a radius of a position.
  /// @param center Query center point.
  /// @param radius_m Search radius in meters.
  /// @return Entities within the radius, sorted by distance.
  std::vector<const TrackedEntity*> QueryRadius(Position3D center,
                                                double radius_m) const;

  /// Get a specific entity by ID.
  /// @return Pointer to entity, or nullptr if not found.
  const TrackedEntity* GetEntity(const std::string& entity_id) const;

  /// Get all entities matching a filter.
  /// @param min_confidence Minimum confidence threshold.
  /// @param object_class Optional class filter (-1 for any).
  std::vector<const TrackedEntity*> GetEntities(
      double min_confidence = 0.0, int object_class = -1) const;

  /// Get entities that need corroboration (single-source, uncertain).
  std::vector<const TrackedEntity*> GetUncertainEntities() const;

  /// Get entities flagged as safety-critical.
  std::vector<const TrackedEntity*> GetSafetyCriticalEntities() const;

  /// Get current world model statistics.
  WorldModelStats GetStats() const;

  /// Get the current simulation tick.
  uint64_t GetTick() const;

  /// Get the current simulation time.
  double GetCurrentTime() const;

  /// Reset the world model (clear all entities).
  void Reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

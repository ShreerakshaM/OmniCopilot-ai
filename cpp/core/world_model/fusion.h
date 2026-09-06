// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fusion — Multi-source observation fusion algorithms.
// Combines observations from multiple agents into a coherent entity state.

#pragma once

#include <memory>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Configuration for fusion algorithms.
struct FusionConfig {
  /// IoU threshold for associating observations to existing entities.
  double association_iou_threshold = 0.3;

  /// Maximum distance (meters) for observation-to-entity association.
  double association_max_distance_m = 5.0;

  /// Weight given to agent trust in fusion.
  double trust_weight = 0.5;

  /// Weight given to observation recency.
  double recency_weight = 0.3;

  /// Weight given to sensor reliability per type.
  double sensor_weight = 0.2;

  /// Minimum agreement ratio to resolve conflicts.
  double conflict_resolution_threshold = 0.6;
};

/// Result of observation-to-entity association.
struct AssociationResult {
  /// Index into the observations vector.
  size_t observation_idx;

  /// Entity ID this observation is associated with (empty if new entity).
  std::string entity_id;

  /// Association score (higher = more confident match).
  double score;
};

/// Multi-source fusion engine.
///
/// Responsibilities:
/// - Associate incoming observations to existing entities (Hungarian algorithm).
/// - Fuse matched observations into entity state updates.
/// - Handle conflicting observations (disagreement resolution).
/// - Create new entities from unmatched observations.
class FusionEngine {
 public:
  explicit FusionEngine(FusionConfig config);
  ~FusionEngine();

  FusionEngine(const FusionEngine&) = delete;
  FusionEngine& operator=(const FusionEngine&) = delete;
  FusionEngine(FusionEngine&&) noexcept;
  FusionEngine& operator=(FusionEngine&&) noexcept;

  /// Associate observations to existing entities.
  /// @param observations New observations from an agent.
  /// @param existing_entities Current entities in the world model.
  /// @return Association results mapping observations to entities.
  std::vector<AssociationResult> Associate(
      const std::vector<Observation>& observations,
      const std::vector<TrackedEntity>& existing_entities) const;

  /// Fuse an observation into an existing entity, updating its state.
  /// @param entity The entity to update (modified in place).
  /// @param obs The new observation.
  /// @param agent_trust Trust score of the reporting agent.
  void FuseObservation(TrackedEntity& entity, const Observation& obs,
                       double agent_trust) const;

  /// Resolve conflicting observations about the same entity.
  /// @param entity The entity with conflicting evidence.
  /// @param conflicting_obs Observations that disagree.
  /// @param agent_trusts Per-agent trust scores for the conflicting observers.
  void ResolveConflict(TrackedEntity& entity,
                       const std::vector<Observation>& conflicting_obs,
                       const std::vector<double>& agent_trusts) const;

  /// Create a new entity from an unmatched observation.
  /// @param obs The observation that didn't match any existing entity.
  /// @param agent_trust Trust score of the reporting agent.
  /// @return A new TrackedEntity in TENTATIVE state.
  TrackedEntity CreateEntity(const Observation& obs, double agent_trust) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

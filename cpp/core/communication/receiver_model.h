// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// ReceiverModel — Maintains a belief model of what each receiver agent knows.
// Enables receiver-aware communication (suppress redundant information).

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Configuration for receiver belief modelling.
struct ReceiverModelConfig {
  /// How quickly our belief about what the receiver knows decays.
  double belief_decay_halflife_s = 2.0;

  /// Minimum novelty score to justify transmission.
  double novelty_threshold = 0.3;
};

/// Estimated belief state of a single receiver agent.
struct ReceiverBelief {
  std::string agent_id;

  /// Last known position of this agent.
  Position3D position;

  /// Estimated set of entity IDs this agent likely knows about.
  /// With associated confidence in their knowledge.
  struct KnownEntity {
    std::string entity_id;
    double estimated_confidence = 0.0;
    double last_informed_time_s = 0.0;
  };
  std::vector<KnownEntity> known_entities;
};

/// Receiver belief model — theory-of-mind for communication.
///
/// Maintains an estimate of what each receiver agent already knows,
/// so we can compute the novelty/value of sending specific information.
class ReceiverModel {
 public:
  explicit ReceiverModel(ReceiverModelConfig config);
  ~ReceiverModel();

  ReceiverModel(const ReceiverModel&) = delete;
  ReceiverModel& operator=(const ReceiverModel&) = delete;
  ReceiverModel(ReceiverModel&&) noexcept;
  ReceiverModel& operator=(ReceiverModel&&) noexcept;

  /// Register a receiver agent to track.
  void RegisterAgent(const std::string& agent_id, Position3D initial_position);

  /// Update our belief about what a receiver knows (e.g., after we sent them info).
  void RecordTransmission(const std::string& receiver_id,
                          const std::string& entity_id,
                          double confidence, double time_s);

  /// Estimate the novelty of sending an observation to a specific receiver.
  /// @return Novelty score [0, 1]. 0 = they already know, 1 = completely new.
  double EstimateNovelty(const std::string& receiver_id,
                         const Observation& obs, double current_time_s) const;

  /// Estimate the information gain of sending an observation to a receiver.
  /// Higher than novelty — considers how much it would CHANGE their world model.
  double EstimateInformationGain(const std::string& receiver_id,
                                 const Observation& obs,
                                 double current_time_s) const;

  /// Get the current belief state for a receiver.
  const ReceiverBelief* GetBelief(const std::string& agent_id) const;

  /// Advance beliefs (decay confidence in what receivers know).
  void Tick(double current_time_s);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

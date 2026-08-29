// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Prioritizer — Hand-designed message prioritization baseline.
// Scores observations by safety relevance, novelty, confidence, and receiver benefit.

#pragma once

#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Prioritization weights (configurable per experiment).
struct PrioritizationWeights {
  double safety_relevance = 0.35;
  double confidence = 0.15;
  double novelty = 0.20;
  double time_sensitivity = 0.15;
  double receiver_benefit = 0.15;
};

/// A scored observation ready for transmission decision.
struct ScoredObservation {
  Observation observation;
  double total_score = 0.0;

  // Score breakdown (for interpretability / logging).
  double safety_score = 0.0;
  double confidence_score = 0.0;
  double novelty_score = 0.0;
  double time_sensitivity_score = 0.0;
  double receiver_benefit_score = 0.0;

  // Target receiver (empty = broadcast).
  std::string target_receiver;
};

/// Selection result from the prioritizer.
struct PrioritizationResult {
  std::vector<ScoredObservation> selected;     // Observations to transmit.
  std::vector<ScoredObservation> suppressed;   // Observations NOT transmitted.
  double total_selected_value = 0.0;
  double total_suppressed_value = 0.0;
};

/// Hand-designed observation prioritizer (baseline).
///
/// Scores each observation using a weighted combination of factors,
/// then selects the top-K within the available bandwidth budget.
class Prioritizer {
 public:
  explicit Prioritizer(PrioritizationWeights weights);
  ~Prioritizer();

  Prioritizer(const Prioritizer&) = delete;
  Prioritizer& operator=(const Prioritizer&) = delete;
  Prioritizer(Prioritizer&&) noexcept;
  Prioritizer& operator=(Prioritizer&&) noexcept;

  /// Score and select observations for transmission.
  /// @param observations All observations available for this timestep.
  /// @param budget_bytes Maximum bytes available for transmission.
  /// @param bytes_per_observation Estimated bytes per serialized observation.
  /// @return Selected and suppressed observations with scores.
  PrioritizationResult Prioritize(const std::vector<Observation>& observations,
                                  uint32_t budget_bytes,
                                  uint32_t bytes_per_observation = 256) const;

  /// Score a single observation (without selection).
  ScoredObservation Score(const Observation& obs) const;

  /// Update weights dynamically (e.g., based on scenario context).
  void SetWeights(PrioritizationWeights weights);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

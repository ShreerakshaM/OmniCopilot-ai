// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Uncertainty — Confidence representation, decay, and calibration.

#pragma once

#include <cstdint>

namespace omnicopilot {

/// Configuration for uncertainty management.
struct UncertaintyConfig {
  /// Confidence half-life in seconds (confidence halves after this time without observation).
  double confidence_halflife_s = 3.0;

  /// Minimum confidence (never decay below this).
  double min_confidence = 0.01;

  /// Confidence threshold below which an entity is considered stale.
  double stale_threshold = 0.15;

  /// Boost factor when multiple independent sources agree.
  double corroboration_boost = 0.15;

  /// Maximum confidence achievable.
  double max_confidence = 0.99;
};

/// Manages confidence decay, boosting, and calibration for entities.
class UncertaintyManager {
 public:
  explicit UncertaintyManager(UncertaintyConfig config);
  ~UncertaintyManager();

  /// Apply time-based confidence decay.
  /// @param current_confidence Current confidence value.
  /// @param elapsed_s Time since last observation (seconds).
  /// @return Decayed confidence value.
  double Decay(double current_confidence, double elapsed_s) const;

  /// Boost confidence when a new corroborating observation arrives.
  /// @param current_confidence Current confidence value.
  /// @param observation_confidence Confidence of the new observation.
  /// @param agent_trust Trust score of the observing agent.
  /// @return Updated confidence value.
  double Boost(double current_confidence, double observation_confidence,
               double agent_trust) const;

  /// Combine confidence from multiple independent sources (Bayesian update).
  /// @param confidences Individual confidence values from different sources.
  /// @param trusts Corresponding trust scores for each source.
  /// @return Combined confidence.
  double CombineIndependent(const double* confidences, const double* trusts,
                            uint32_t count) const;

  /// Check if confidence is below the stale threshold.
  bool IsStale(double confidence) const;

  /// Check if confidence warrants safety-critical classification.
  bool IsSafetyCritical(double confidence, double time_to_collision_s) const;

 private:
  UncertaintyConfig config_;
};

}  // namespace omnicopilot

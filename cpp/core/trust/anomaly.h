// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Anomaly — Detect anomalous/adversarial observations.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Type of detected anomaly.
enum class AnomalyType : int {
  kPhysicsViolation = 0,      // Impossible kinematics.
  kConsensusDeviation = 1,    // Disagrees with majority.
  kTemporalInconsistency = 2, // Contradicts own previous observations.
  kStatisticalOutlier = 3,    // Sudden accuracy change.
};

/// A detected anomaly report.
struct AnomalyReport {
  std::string agent_id;
  AnomalyType type;
  double severity = 0.0;  // [0, 1].
  std::string description;
  double timestamp_s = 0.0;
};

/// Anomaly detector for incoming observations.
class AnomalyDetector {
 public:
  AnomalyDetector();
  ~AnomalyDetector();

  /// Check an observation for physical plausibility.
  /// @return Anomaly report if violation detected, empty otherwise.
  std::vector<AnomalyReport> Check(const Observation& obs,
                                   const std::vector<TrackedEntity>& world_state) const;

  /// Check if an observation deviates from consensus.
  /// @param obs The observation to check.
  /// @param others Other observations of the same entity from different agents.
  std::vector<AnomalyReport> CheckConsensus(
      const Observation& obs,
      const std::vector<Observation>& others) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// TemporalTracker — State estimation and trajectory prediction.
// Implements Kalman filtering for smooth state estimation and linear/polynomial
// prediction of future entity positions.

#pragma once

#include <memory>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Configuration for temporal tracking.
struct TemporalTrackerConfig {
  /// Process noise (how much we expect the true state to change per second).
  double process_noise_position = 1.0;   // m²/s
  double process_noise_velocity = 2.0;   // (m/s)²/s

  /// Maximum prediction horizon (seconds).
  double max_prediction_horizon_s = 5.0;

  /// Prediction step size (seconds).
  double prediction_step_s = 0.5;
};

/// Temporal state estimator and predictor for a single entity.
///
/// Uses a constant-velocity Kalman filter for state estimation.
/// Produces trajectory predictions for collision risk assessment.
class TemporalTracker {
 public:
  explicit TemporalTracker(TemporalTrackerConfig config);
  ~TemporalTracker();

  TemporalTracker(const TemporalTracker&) = delete;
  TemporalTracker& operator=(const TemporalTracker&) = delete;
  TemporalTracker(TemporalTracker&&) noexcept;
  TemporalTracker& operator=(TemporalTracker&&) noexcept;

  /// Initialize a new track from an observation.
  void Initialize(const Observation& obs);

  /// Predict the state forward to the given time (no observation).
  void Predict(double target_time_s);

  /// Update the state with a new observation (Kalman update step).
  void Update(const Observation& obs);

  /// Get the current estimated position.
  Position3D GetPosition() const;

  /// Get the current estimated velocity.
  Velocity3D GetVelocity() const;

  /// Get the current position uncertainty (standard deviation in meters).
  double GetPositionUncertainty() const;

  /// Generate predictions for future time steps.
  /// @return Predicted states at intervals of prediction_step_s up to max horizon.
  std::vector<PredictedState> GetPredictions() const;

  /// Get the time of the last update.
  double GetLastUpdateTime() const;

  /// Check if this track has been initialized.
  bool IsInitialized() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

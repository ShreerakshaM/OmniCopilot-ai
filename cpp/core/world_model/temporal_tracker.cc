// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Constant-velocity Kalman filter for 2D state estimation (x, y, vx, vy).
// Z-axis tracked separately with simple exponential smoothing since most
// driving scenarios are approximately planar.

#include "cpp/core/world_model/temporal_tracker.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace omnicopilot {

// State vector: [x, y, vx, vy]  (4-dimensional)
// Measurement vector: [x, y] (2-dimensional, from observation position)
//
// State transition (constant velocity):
//   x'  = x + vx * dt
//   y'  = y + vy * dt
//   vx' = vx
//   vy' = vy
//
// We store the 4x4 covariance matrix P as a flat array (row-major).

static constexpr int kStateDim = 4;
static constexpr int kMeasDim = 2;

// Index helpers for the flat 4x4 matrix.
static inline double& Mat4(std::array<double, 16>& m, int r, int c) {
  return m[r * kStateDim + c];
}
static inline double Mat4(const std::array<double, 16>& m, int r, int c) {
  return m[r * kStateDim + c];
}

struct TemporalTracker::Impl {
  TemporalTrackerConfig config;
  bool initialized = false;

  // State: [x, y, vx, vy]
  std::array<double, kStateDim> state = {};

  // Z-axis tracked separately (simple smoothing).
  double z = 0.0;
  double vz = 0.0;

  // 4x4 covariance matrix P (row-major).
  std::array<double, 16> P = {};

  // Timestamps.
  double last_update_time_s = 0.0;
  double last_predict_time_s = 0.0;

  // Initialize P to large diagonal (high initial uncertainty).
  void InitCovariance() {
    P.fill(0.0);
    Mat4(P, 0, 0) = 1.0;    // x variance
    Mat4(P, 1, 1) = 1.0;    // y variance
    Mat4(P, 2, 2) = 10.0;   // vx variance (high — we don't know velocity)
    Mat4(P, 3, 3) = 10.0;   // vy variance
  }

  // Build process noise Q for timestep dt.
  std::array<double, 16> BuildQ(double dt) const {
    std::array<double, 16> Q = {};
    double qp = config.process_noise_position * dt;
    double qv = config.process_noise_velocity * dt;

    // Discrete noise model: position gets dt^2 * noise, velocity gets dt * noise,
    // plus cross terms.
    double dt2 = dt * dt;
    double dt3 = dt2 * dt / 2.0;
    double dt4 = dt2 * dt2 / 4.0;

    // Position-position block.
    Mat4(Q, 0, 0) = dt4 * qv + qp;
    Mat4(Q, 1, 1) = dt4 * qv + qp;

    // Position-velocity cross block.
    Mat4(Q, 0, 2) = dt3 * qv;
    Mat4(Q, 1, 3) = dt3 * qv;
    Mat4(Q, 2, 0) = dt3 * qv;
    Mat4(Q, 3, 1) = dt3 * qv;

    // Velocity-velocity block.
    Mat4(Q, 2, 2) = dt2 * qv;
    Mat4(Q, 3, 3) = dt2 * qv;

    return Q;
  }

  // Predict state and covariance forward by dt.
  void PredictStep(double dt) {
    if (dt <= 0.0) return;

    // State prediction: constant velocity.
    state[0] += state[2] * dt;  // x += vx * dt
    state[1] += state[3] * dt;  // y += vy * dt
    // vx, vy unchanged.

    // Z-axis: simple constant velocity.
    z += vz * dt;

    // Covariance prediction: P = F * P * F^T + Q
    // F = [[1, 0, dt, 0],
    //      [0, 1, 0, dt],
    //      [0, 0, 1,  0],
    //      [0, 0, 0,  1]]
    //
    // Rather than full matrix multiply, apply F analytically:
    // P' = F P F^T + Q
    //
    // Row/col operations for F P F^T:
    // New P[i][j] = P[i][j] + dt*P[i+2][j] + dt*P[i][j+2] + dt^2*P[i+2][j+2]
    // where index+2 applies only to rows/cols 0,1.

    std::array<double, 16> Pnew = {};
    for (int i = 0; i < kStateDim; ++i) {
      for (int j = 0; j < kStateDim; ++j) {
        double val = Mat4(P, i, j);
        // Add F contributions.
        if (i < kMeasDim) {
          val += dt * Mat4(P, i + kMeasDim, j);
        }
        if (j < kMeasDim) {
          val += dt * Mat4(P, i, j + kMeasDim);
        }
        if (i < kMeasDim && j < kMeasDim) {
          val += dt * dt * Mat4(P, i + kMeasDim, j + kMeasDim);
        }
        Mat4(Pnew, i, j) = val;
      }
    }

    // Add process noise.
    auto Q = BuildQ(dt);
    for (int i = 0; i < 16; ++i) {
      Pnew[i] += Q[i];
    }

    P = Pnew;
  }
};

TemporalTracker::TemporalTracker(TemporalTrackerConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

TemporalTracker::~TemporalTracker() = default;
TemporalTracker::TemporalTracker(TemporalTracker&&) noexcept = default;
TemporalTracker& TemporalTracker::operator=(TemporalTracker&&) noexcept =
    default;

void TemporalTracker::Initialize(const Observation& obs) {
  impl_->state[0] = obs.position.x;
  impl_->state[1] = obs.position.y;
  impl_->state[2] = obs.velocity.vx;
  impl_->state[3] = obs.velocity.vy;
  impl_->z = obs.position.z;
  impl_->vz = obs.velocity.vz;
  impl_->last_update_time_s = obs.timestamp_s;
  impl_->last_predict_time_s = obs.timestamp_s;
  impl_->InitCovariance();
  impl_->initialized = true;
}

void TemporalTracker::Predict(double target_time_s) {
  if (!impl_->initialized) return;

  double dt = target_time_s - impl_->last_predict_time_s;
  if (dt <= 0.0) return;

  impl_->PredictStep(dt);
  impl_->last_predict_time_s = target_time_s;
}

void TemporalTracker::Update(const Observation& obs) {
  if (!impl_->initialized) {
    Initialize(obs);
    return;
  }

  // First predict to observation time.
  double dt = obs.timestamp_s - impl_->last_predict_time_s;
  if (dt > 0.0) {
    impl_->PredictStep(dt);
    impl_->last_predict_time_s = obs.timestamp_s;
  }

  // Kalman update with measurement z = [obs.x, obs.y].
  // H = [[1, 0, 0, 0],
  //      [0, 1, 0, 0]]
  //
  // Measurement noise R (diagonal, based on observation confidence).
  // Higher confidence → lower noise.
  double base_noise = 2.0;  // meters^2
  double confidence = std::clamp(obs.confidence, 0.1, 1.0);
  double r = base_noise / confidence;  // Lower confidence → higher noise.

  // Innovation: y = z - H * x
  double y0 = obs.position.x - impl_->state[0];
  double y1 = obs.position.y - impl_->state[1];

  // Innovation covariance: S = H * P * H^T + R
  // Since H selects first 2 rows/cols:
  // S = P[0:2, 0:2] + R*I
  double s00 = Mat4(impl_->P, 0, 0) + r;
  double s01 = Mat4(impl_->P, 0, 1);
  double s10 = Mat4(impl_->P, 1, 0);
  double s11 = Mat4(impl_->P, 1, 1) + r;

  // Invert 2x2 S.
  double det = s00 * s11 - s01 * s10;
  if (std::abs(det) < 1e-12) {
    // Singular — skip update.
    return;
  }
  double inv_det = 1.0 / det;
  double si00 = s11 * inv_det;
  double si01 = -s01 * inv_det;
  double si10 = -s10 * inv_det;
  double si11 = s00 * inv_det;

  // Kalman gain: K = P * H^T * S^-1  (4x2 matrix)
  // K[i][j] = P[i][0]*Si[0][j] + P[i][1]*Si[1][j]
  std::array<double, 8> K = {};  // 4x2 row-major
  for (int i = 0; i < kStateDim; ++i) {
    double p0 = Mat4(impl_->P, i, 0);
    double p1 = Mat4(impl_->P, i, 1);
    K[i * 2 + 0] = p0 * si00 + p1 * si10;
    K[i * 2 + 1] = p0 * si01 + p1 * si11;
  }

  // State update: x = x + K * y
  for (int i = 0; i < kStateDim; ++i) {
    impl_->state[i] += K[i * 2 + 0] * y0 + K[i * 2 + 1] * y1;
  }

  // Covariance update: P = (I - K * H) * P
  // Since H selects cols 0,1: (I - K*H)[i][j] = I[i][j] - K[i][0]*(j==0) - K[i][1]*(j==1)
  std::array<double, 16> Pnew = {};
  for (int i = 0; i < kStateDim; ++i) {
    for (int j = 0; j < kStateDim; ++j) {
      double val = Mat4(impl_->P, i, j);
      val -= K[i * 2 + 0] * Mat4(impl_->P, 0, j);
      val -= K[i * 2 + 1] * Mat4(impl_->P, 1, j);
      Mat4(Pnew, i, j) = val;
    }
  }
  impl_->P = Pnew;

  // Z-axis: simple exponential smoothing.
  double alpha_z = std::clamp(confidence * 0.5, 0.1, 0.5);
  impl_->z += alpha_z * (obs.position.z - impl_->z);
  impl_->vz += alpha_z * (obs.velocity.vz - impl_->vz);

  impl_->last_update_time_s = obs.timestamp_s;
}

Position3D TemporalTracker::GetPosition() const {
  return {impl_->state[0], impl_->state[1], impl_->z};
}

Velocity3D TemporalTracker::GetVelocity() const {
  return {impl_->state[2], impl_->state[3], impl_->vz};
}

double TemporalTracker::GetPositionUncertainty() const {
  // Return average position standard deviation (sqrt of diagonal P entries).
  double var_x = Mat4(impl_->P, 0, 0);
  double var_y = Mat4(impl_->P, 1, 1);
  return std::sqrt((var_x + var_y) / 2.0);
}

std::vector<PredictedState> TemporalTracker::GetPredictions() const {
  std::vector<PredictedState> predictions;
  if (!impl_->initialized) return predictions;

  double step = impl_->config.prediction_step_s;
  double max_horizon = impl_->config.max_prediction_horizon_s;
  int num_steps = static_cast<int>(max_horizon / step);
  predictions.reserve(num_steps);

  // Current state as starting point.
  double px = impl_->state[0];
  double py = impl_->state[1];
  double vx = impl_->state[2];
  double vy = impl_->state[3];
  double pz = impl_->z;
  double v_z = impl_->vz;

  // Current uncertainty grows with prediction horizon.
  double base_uncertainty = GetPositionUncertainty();

  for (int i = 1; i <= num_steps; ++i) {
    double t = step * i;

    PredictedState ps;
    ps.time_horizon_s = t;
    ps.position = {px + vx * t, py + vy * t, pz + v_z * t};
    ps.velocity = {vx, vy, v_z};

    // Confidence decays with prediction horizon.
    // Simple model: confidence halves every max_horizon seconds.
    double decay = std::pow(0.5, t / max_horizon);
    ps.confidence = std::clamp(decay * (1.0 / (1.0 + base_uncertainty)), 0.0, 1.0);

    predictions.push_back(ps);
  }

  return predictions;
}

double TemporalTracker::GetLastUpdateTime() const {
  return impl_->last_update_time_s;
}

bool TemporalTracker::IsInitialized() const { return impl_->initialized; }

}  // namespace omnicopilot

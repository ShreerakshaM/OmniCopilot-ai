// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/trust/anomaly.h"

namespace omnicopilot {

struct AnomalyDetector::Impl {};

AnomalyDetector::AnomalyDetector() : impl_(std::make_unique<Impl>()) {}
AnomalyDetector::~AnomalyDetector() = default;

std::vector<AnomalyReport> AnomalyDetector::Check(
    const Observation& obs,
    const std::vector<TrackedEntity>& world_state) const {
  // TODO: Implement physical plausibility checks.
  return {};
}

std::vector<AnomalyReport> AnomalyDetector::CheckConsensus(
    const Observation& obs, const std::vector<Observation>& others) const {
  // TODO: Implement consensus deviation checks.
  return {};
}

}  // namespace omnicopilot

// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/active_acquisition/info_value.h"

namespace omnicopilot {

struct InfoValueEstimator::Impl {};

InfoValueEstimator::InfoValueEstimator() : impl_(std::make_unique<Impl>()) {}
InfoValueEstimator::~InfoValueEstimator() = default;

std::vector<InfoValueEstimate> InfoValueEstimator::Estimate(
    const TrackedEntity& target,
    const std::vector<std::string>& available_agents,
    const std::vector<Position3D>& agent_positions,
    const std::vector<double>& agent_trusts) const {
  // TODO: Implement information value estimation.
  return {};
}

}  // namespace omnicopilot

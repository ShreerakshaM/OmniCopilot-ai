// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// InfoValue — Estimate the information value of querying a specific agent.

#pragma once

#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Estimated information value from querying an agent.
struct InfoValueEstimate {
  std::string agent_id;
  double expected_information_gain = 0.0;  // [0, 1].
  double expected_uncertainty_reduction = 0.0;
  double cost = 0.0;                       // Communication cost.
  double net_value = 0.0;                  // gain - cost.
};

/// Estimates the information value of querying specific agents about uncertain entities.
class InfoValueEstimator {
 public:
  InfoValueEstimator();
  ~InfoValueEstimator();

  /// Estimate the information value of querying each available agent about a target entity.
  /// @param target The entity we want more information about.
  /// @param available_agents Agents that could potentially provide information.
  /// @param agent_positions Current positions of available agents.
  /// @param agent_trusts Trust scores of available agents.
  /// @return Ranked list of agents by expected value (highest first).
  std::vector<InfoValueEstimate> Estimate(
      const TrackedEntity& target,
      const std::vector<std::string>& available_agents,
      const std::vector<Position3D>& agent_positions,
      const std::vector<double>& agent_trusts) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

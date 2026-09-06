// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Byzantine — Byzantine fault tolerance for observation consensus.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Result of Byzantine consensus.
struct ConsensusResult {
  bool consensus_reached = false;
  Observation consensus_observation;  // Best-estimate from agreeing agents.
  double agreement_ratio = 0.0;       // Fraction of agents that agree.
  std::vector<std::string> agreeing_agents;
  std::vector<std::string> disagreeing_agents;
};

/// Byzantine fault-tolerant consensus for critical observations.
///
/// Requires agreement from a supermajority (>2/3) of agents to confirm
/// safety-critical observations. Identifies agents in the minority as
/// potentially compromised.
class ByzantineConsensus {
 public:
  ByzantineConsensus();
  ~ByzantineConsensus();

  /// Run consensus on multiple observations of the same entity.
  /// @param observations Observations from different agents about the same object.
  /// @param agent_trusts Trust scores for each observing agent.
  /// @param position_tolerance_m Maximum position difference to consider "agreeing".
  ConsensusResult RunConsensus(const std::vector<Observation>& observations,
                               const std::vector<double>& agent_trusts,
                               double position_tolerance_m = 3.0) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

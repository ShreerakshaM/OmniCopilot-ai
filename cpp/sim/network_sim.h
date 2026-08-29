// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// NetworkSim — Multi-agent network simulation.
// Manages channels between all agent pairs.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/communication/channel.h"

namespace omnicopilot {

/// Network simulation managing all inter-agent channels.
class NetworkSimulator {
 public:
  explicit NetworkSimulator(ChannelConfig default_config);
  ~NetworkSimulator();

  /// Register an agent in the network.
  void RegisterAgent(const std::string& agent_id);

  /// Send a message from one agent to another (or broadcast).
  bool Send(const ChannelMessage& message);

  /// Advance time — deliver messages whose time has come.
  /// @return All messages delivered to each agent.
  std::vector<std::pair<std::string, std::vector<ChannelMessage>>> Tick(
      double current_time_s);

  /// Set global network conditions (affects all channels).
  void SetGlobalConditions(double loss_rate, double latency_ms, double bandwidth_bps);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

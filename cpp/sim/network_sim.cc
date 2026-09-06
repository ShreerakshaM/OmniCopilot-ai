// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/sim/network_sim.h"

namespace omnicopilot {

struct NetworkSimulator::Impl {
  ChannelConfig default_config;
};

NetworkSimulator::NetworkSimulator(ChannelConfig default_config)
    : impl_(std::make_unique<Impl>()) {
  impl_->default_config = std::move(default_config);
}

NetworkSimulator::~NetworkSimulator() = default;

void NetworkSimulator::RegisterAgent(const std::string& agent_id) {
  // TODO: Create per-pair channels.
}

bool NetworkSimulator::Send(const ChannelMessage& message) {
  // TODO: Route through appropriate channel.
  return false;
}

std::vector<std::pair<std::string, std::vector<ChannelMessage>>>
NetworkSimulator::Tick(double current_time_s) {
  // TODO: Tick all channels, collect deliveries.
  return {};
}

void NetworkSimulator::SetGlobalConditions(double loss_rate, double latency_ms,
                                           double bandwidth_bps) {
  // TODO: Apply to all channels.
}

}  // namespace omnicopilot

// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/agent/agent.h"

namespace omnicopilot {

struct Agent::Impl {
  AgentConfig config;
  WorldModel world_model;
  Position3D position;

  explicit Impl(AgentConfig cfg)
      : config(std::move(cfg)),
        world_model(cfg.world_model_config),
        position(cfg.initial_position) {}
};

Agent::Agent(AgentConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Agent::~Agent() = default;
Agent::Agent(Agent&&) noexcept = default;
Agent& Agent::operator=(Agent&&) noexcept = default;

void Agent::Step(const std::vector<Observation>& local_observations,
                 const std::vector<ChannelMessage>& incoming_messages,
                 double current_time_s) {
  // 1. Ingest local observations.
  impl_->world_model.IngestObservations(local_observations, 1.0);

  // 2. TODO: Deserialize and ingest incoming cooperative observations.

  // 3. Tick world model.
  impl_->world_model.Tick(current_time_s);

  // 4. TODO: Run communication policy to decide what to share.
}

std::vector<ChannelMessage> Agent::GetOutgoingMessages() const {
  // TODO: Implement — return messages decided by communication policy.
  return {};
}

const WorldModel& Agent::GetWorldModel() const {
  return impl_->world_model;
}

Position3D Agent::GetPosition() const { return impl_->position; }

void Agent::SetPosition(Position3D position) { impl_->position = position; }

const std::string& Agent::GetId() const { return impl_->config.agent_id; }

}  // namespace omnicopilot

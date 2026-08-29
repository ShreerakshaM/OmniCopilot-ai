// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Agent — The main agent runtime that ties all components together.
// One Agent instance per simulated vehicle/robot.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/communication/channel.h"
#include "cpp/core/world_model/entity.h"
#include "cpp/core/world_model/world_model.h"

namespace omnicopilot {

/// Agent configuration.
struct AgentConfig {
  std::string agent_id;
  Position3D initial_position;
  WorldModelConfig world_model_config;
  ChannelConfig channel_config;
};

/// The main agent runtime — observe, fuse, decide, communicate loop.
///
/// Each agent:
/// 1. Receives local observations (from perception model).
/// 2. Receives cooperative messages (from other agents via channel).
/// 3. Fuses all observations into its world model.
/// 4. Decides what to communicate (prioritizer or learned policy).
/// 5. Transmits selected observations via channel.
/// 6. Optionally requests information (active acquisition).
class Agent {
 public:
  explicit Agent(AgentConfig config);
  ~Agent();

  Agent(const Agent&) = delete;
  Agent& operator=(const Agent&) = delete;
  Agent(Agent&&) noexcept;
  Agent& operator=(Agent&&) noexcept;

  /// Process one simulation step.
  /// @param local_observations Observations from this agent's perception.
  /// @param incoming_messages Messages received from other agents.
  /// @param current_time_s Current simulation time.
  void Step(const std::vector<Observation>& local_observations,
            const std::vector<ChannelMessage>& incoming_messages,
            double current_time_s);

  /// Get messages this agent wants to transmit (output of communication policy).
  std::vector<ChannelMessage> GetOutgoingMessages() const;

  /// Get the agent's current world model (read-only).
  const WorldModel& GetWorldModel() const;

  /// Get the agent's current position.
  Position3D GetPosition() const;

  /// Update the agent's position (called by simulation).
  void SetPosition(Position3D position);

  /// Get the agent's ID.
  const std::string& GetId() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

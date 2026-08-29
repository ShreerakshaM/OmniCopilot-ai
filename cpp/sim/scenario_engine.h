// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// ScenarioEngine — Orchestrates multi-agent simulation scenarios.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/agent/agent.h"
#include "cpp/sim/network_sim.h"

namespace omnicopilot {

/// Scenario configuration.
struct ScenarioConfig {
  std::string scenario_id;
  std::string dataset;           // "opv2v", "dair_v2x", "carla"
  uint32_t num_agents = 3;
  double duration_s = 30.0;
  double tick_rate_hz = 10.0;    // Simulation steps per second.
  ChannelConfig network_config;
};

/// Manages the simulation loop for a multi-agent scenario.
class ScenarioEngine {
 public:
  explicit ScenarioEngine(ScenarioConfig config);
  ~ScenarioEngine();

  /// Initialize the scenario (load data, create agents).
  void Initialize();

  /// Run one simulation tick.
  /// @return false if scenario has ended.
  bool Tick();

  /// Get current simulation time.
  double GetCurrentTime() const;

  /// Get current tick number.
  uint64_t GetTickCount() const;

  /// Check if scenario has completed.
  bool IsFinished() const;

  /// Get all agents.
  const std::vector<Agent>& GetAgents() const;

  /// Run the full scenario from start to finish.
  void Run();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

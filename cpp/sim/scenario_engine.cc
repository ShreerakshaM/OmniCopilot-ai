// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/sim/scenario_engine.h"

namespace omnicopilot {

struct ScenarioEngine::Impl {
  ScenarioConfig config;
  uint64_t tick_count = 0;
  double current_time_s = 0.0;
  std::vector<Agent> agents;
};

ScenarioEngine::ScenarioEngine(ScenarioConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

ScenarioEngine::~ScenarioEngine() = default;

void ScenarioEngine::Initialize() {
  // TODO: Load data, create agents.
}

bool ScenarioEngine::Tick() {
  // TODO: Implement simulation step.
  return false;
}

double ScenarioEngine::GetCurrentTime() const {
  return impl_->current_time_s;
}

uint64_t ScenarioEngine::GetTickCount() const { return impl_->tick_count; }

bool ScenarioEngine::IsFinished() const {
  return impl_->current_time_s >= impl_->config.duration_s;
}

const std::vector<Agent>& ScenarioEngine::GetAgents() const {
  return impl_->agents;
}

void ScenarioEngine::Run() {
  Initialize();
  while (!IsFinished()) {
    if (!Tick()) break;
  }
}

}  // namespace omnicopilot

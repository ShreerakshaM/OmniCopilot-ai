// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/active_acquisition/request_manager.h"

#include <vector>

namespace omnicopilot {

struct RequestManager::Impl {
  RequestManagerConfig config;
};

RequestManager::RequestManager(RequestManagerConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

RequestManager::~RequestManager() = default;
RequestManager::RequestManager(RequestManager&&) noexcept = default;
RequestManager& RequestManager::operator=(RequestManager&&) noexcept = default;

std::optional<std::string> RequestManager::SubmitRequest(
    const std::string& target_agent_id, const std::string& target_entity_id,
    Position3D location, double current_time_s, double expected_gain) {
  // TODO: Implement request submission.
  return std::nullopt;
}

void RequestManager::RecordResponse(
    const std::string& request_id,
    const std::vector<Observation>& response_observations) {
  // TODO: Implement response recording.
}

void RequestManager::Tick(double current_time_s) {
  // TODO: Implement timeout handling.
}

std::vector<InfoRequest> RequestManager::GetPendingRequests() const {
  return {};
}

RequestManager::Stats RequestManager::GetStats() const { return {}; }

}  // namespace omnicopilot

// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// RequestManager — Track outstanding information requests and responses.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// An outstanding information request.
struct InfoRequest {
  std::string request_id;
  std::string requester_id;
  std::string target_agent_id;
  std::string target_entity_id;
  Position3D target_location;
  double sent_time_s = 0.0;
  double deadline_s = 0.0;
  double expected_gain = 0.0;
};

/// Configuration for request management.
struct RequestManagerConfig {
  /// Request timeout (seconds).
  double request_timeout_s = 1.0;

  /// Maximum requests per tick.
  uint32_t max_requests_per_tick = 3;

  /// Budget: requests consume communication bandwidth.
  uint32_t request_size_bytes = 64;
};

/// Manages the lifecycle of active information requests.
class RequestManager {
 public:
  explicit RequestManager(RequestManagerConfig config);
  ~RequestManager();

  RequestManager(const RequestManager&) = delete;
  RequestManager& operator=(const RequestManager&) = delete;
  RequestManager(RequestManager&&) noexcept;
  RequestManager& operator=(RequestManager&&) noexcept;

  /// Submit a new information request.
  /// @return Request ID if accepted, empty if budget/limit exceeded.
  std::optional<std::string> SubmitRequest(const std::string& target_agent_id,
                                           const std::string& target_entity_id,
                                           Position3D location,
                                           double current_time_s,
                                           double expected_gain);

  /// Record that a response was received for a request.
  void RecordResponse(const std::string& request_id,
                      const std::vector<Observation>& response_observations);

  /// Advance time — expire timed-out requests.
  void Tick(double current_time_s);

  /// Get all currently outstanding (pending) requests.
  std::vector<InfoRequest> GetPendingRequests() const;

  /// Get statistics.
  struct Stats {
    uint32_t total_sent = 0;
    uint32_t total_fulfilled = 0;
    uint32_t total_timed_out = 0;
    double mean_response_time_s = 0.0;
    double mean_information_gain = 0.0;
  };
  Stats GetStats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

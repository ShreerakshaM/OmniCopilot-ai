// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// SourceSelector — Select the best agent to query for missing information.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/active_acquisition/info_value.h"
#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Configuration for source selection.
struct SourceSelectionConfig {
  /// Maximum simultaneous outstanding requests per agent.
  uint32_t max_requests_per_agent = 2;

  /// Maximum total outstanding requests.
  uint32_t max_total_requests = 5;

  /// Minimum net value to justify sending a request.
  double min_net_value = 0.2;
};

/// Selects which agent to query for active information acquisition.
class SourceSelector {
 public:
  explicit SourceSelector(SourceSelectionConfig config);
  ~SourceSelector();

  /// Select the best agent(s) to query about an uncertain entity.
  /// @param estimates Information value estimates from InfoValueEstimator.
  /// @return Agent IDs selected for querying (may be empty if none worthwhile).
  std::vector<std::string> Select(
      const std::vector<InfoValueEstimate>& estimates) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

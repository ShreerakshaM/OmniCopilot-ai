// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Reliability — Per-agent trust scoring and management.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace omnicopilot {

/// Configuration for agent reliability tracking.
struct ReliabilityConfig {
  /// Initial trust score for new agents.
  double initial_trust = 0.7;

  /// Exponential moving average alpha for trust updates.
  double trust_ema_alpha = 0.05;

  /// Trust floor (never go below this — allows recovery).
  double min_trust = 0.05;

  /// Trust ceiling.
  double max_trust = 0.99;

  /// Number of observations before trust is considered calibrated.
  uint32_t calibration_period = 50;
};

/// Trust record for a single agent.
struct AgentTrust {
  std::string agent_id;
  double trust_score = 0.7;
  double accuracy_recent = 0.0;    // Accuracy over recent window.
  double accuracy_lifetime = 0.0;  // Accuracy over all time.
  uint64_t total_observations = 0;
  uint64_t verified_correct = 0;
  uint64_t verified_incorrect = 0;
  bool is_calibrated = false;      // True once calibration_period observations received.
};

/// Agent reliability tracker.
///
/// Maintains per-agent trust scores based on historical accuracy.
/// Trust is updated when observations can be verified (against consensus or ground truth).
class ReliabilityTracker {
 public:
  explicit ReliabilityTracker(ReliabilityConfig config);
  ~ReliabilityTracker();

  ReliabilityTracker(const ReliabilityTracker&) = delete;
  ReliabilityTracker& operator=(const ReliabilityTracker&) = delete;
  ReliabilityTracker(ReliabilityTracker&&) noexcept;
  ReliabilityTracker& operator=(ReliabilityTracker&&) noexcept;

  /// Register a new agent.
  void RegisterAgent(const std::string& agent_id);

  /// Record a verified-correct observation from an agent.
  void RecordCorrect(const std::string& agent_id);

  /// Record a verified-incorrect observation from an agent.
  void RecordIncorrect(const std::string& agent_id);

  /// Get the current trust score for an agent.
  double GetTrust(const std::string& agent_id) const;

  /// Get full trust record for an agent.
  const AgentTrust* GetRecord(const std::string& agent_id) const;

  /// Get all agent trust records.
  std::vector<AgentTrust> GetAllRecords() const;

  /// Check if an agent's trust is below a threshold (potentially adversarial).
  bool IsSuspicious(const std::string& agent_id, double threshold = 0.3) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

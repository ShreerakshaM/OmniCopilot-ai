// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/trust/reliability.h"

#include <algorithm>
#include <unordered_map>

namespace omnicopilot {

struct ReliabilityTracker::Impl {
  ReliabilityConfig config;
  std::unordered_map<std::string, AgentTrust> agents;
};

ReliabilityTracker::ReliabilityTracker(ReliabilityConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

ReliabilityTracker::~ReliabilityTracker() = default;
ReliabilityTracker::ReliabilityTracker(ReliabilityTracker&&) noexcept = default;
ReliabilityTracker& ReliabilityTracker::operator=(
    ReliabilityTracker&&) noexcept = default;

void ReliabilityTracker::RegisterAgent(const std::string& agent_id) {
  if (impl_->agents.count(agent_id) > 0) return;

  AgentTrust record;
  record.agent_id = agent_id;
  record.trust_score = impl_->config.initial_trust;
  record.accuracy_recent = impl_->config.initial_trust;
  record.accuracy_lifetime = 0.0;
  record.total_observations = 0;
  record.verified_correct = 0;
  record.verified_incorrect = 0;
  record.is_calibrated = false;

  impl_->agents.emplace(agent_id, std::move(record));
}

void ReliabilityTracker::RecordCorrect(const std::string& agent_id) {
  auto it = impl_->agents.find(agent_id);
  if (it == impl_->agents.end()) {
    RegisterAgent(agent_id);
    it = impl_->agents.find(agent_id);
  }

  auto& record = it->second;
  record.total_observations++;
  record.verified_correct++;

  // Exponential moving average of accuracy.
  double alpha = impl_->config.trust_ema_alpha;
  record.accuracy_recent = record.accuracy_recent * (1.0 - alpha) + alpha * 1.0;

  // Lifetime accuracy.
  uint64_t verified = record.verified_correct + record.verified_incorrect;
  record.accuracy_lifetime =
      (verified > 0)
          ? static_cast<double>(record.verified_correct) / verified
          : 0.0;

  // Update trust: blend recent accuracy with lifetime.
  record.trust_score = 0.7 * record.accuracy_recent +
                       0.3 * record.accuracy_lifetime;
  record.trust_score =
      std::clamp(record.trust_score, impl_->config.min_trust,
                 impl_->config.max_trust);

  // Check calibration.
  if (record.total_observations >= impl_->config.calibration_period) {
    record.is_calibrated = true;
  }
}

void ReliabilityTracker::RecordIncorrect(const std::string& agent_id) {
  auto it = impl_->agents.find(agent_id);
  if (it == impl_->agents.end()) {
    RegisterAgent(agent_id);
    it = impl_->agents.find(agent_id);
  }

  auto& record = it->second;
  record.total_observations++;
  record.verified_incorrect++;

  // EMA update with incorrect observation (value = 0).
  double alpha = impl_->config.trust_ema_alpha;
  record.accuracy_recent = record.accuracy_recent * (1.0 - alpha);

  // Lifetime accuracy.
  uint64_t verified = record.verified_correct + record.verified_incorrect;
  record.accuracy_lifetime =
      (verified > 0)
          ? static_cast<double>(record.verified_correct) / verified
          : 0.0;

  // Update trust.
  record.trust_score = 0.7 * record.accuracy_recent +
                       0.3 * record.accuracy_lifetime;
  record.trust_score =
      std::clamp(record.trust_score, impl_->config.min_trust,
                 impl_->config.max_trust);

  if (record.total_observations >= impl_->config.calibration_period) {
    record.is_calibrated = true;
  }
}

double ReliabilityTracker::GetTrust(const std::string& agent_id) const {
  auto it = impl_->agents.find(agent_id);
  if (it == impl_->agents.end()) {
    return impl_->config.initial_trust;  // Unknown agent → default trust.
  }
  return it->second.trust_score;
}

const AgentTrust* ReliabilityTracker::GetRecord(
    const std::string& agent_id) const {
  auto it = impl_->agents.find(agent_id);
  if (it != impl_->agents.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<AgentTrust> ReliabilityTracker::GetAllRecords() const {
  std::vector<AgentTrust> result;
  result.reserve(impl_->agents.size());
  for (const auto& [id, record] : impl_->agents) {
    result.push_back(record);
  }
  return result;
}

bool ReliabilityTracker::IsSuspicious(const std::string& agent_id,
                                      double threshold) const {
  auto it = impl_->agents.find(agent_id);
  if (it == impl_->agents.end()) {
    return false;  // Unknown agent — can't judge yet.
  }
  // Only flag as suspicious if calibrated (enough observations).
  return it->second.is_calibrated && it->second.trust_score < threshold;
}

}  // namespace omnicopilot

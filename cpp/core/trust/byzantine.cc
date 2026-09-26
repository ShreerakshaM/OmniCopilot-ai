// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/trust/byzantine.h"

#include <algorithm>

namespace omnicopilot {

struct ByzantineConsensus::Impl {};

ByzantineConsensus::ByzantineConsensus() : impl_(std::make_unique<Impl>()) {}
ByzantineConsensus::~ByzantineConsensus() = default;

ConsensusResult ByzantineConsensus::RunConsensus(
    const std::vector<Observation>& observations,
    const std::vector<double>& agent_trusts,
    double position_tolerance_m) const {
  ConsensusResult result;
  if (observations.empty()) return result;

  struct Candidate {
    size_t index;
    double trust;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(observations.size());
  for (size_t i = 0; i < observations.size(); ++i) {
    candidates.push_back(
        {i, i < agent_trusts.size() ? std::clamp(agent_trusts[i], 0.0, 1.0)
                                    : 0.5});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.trust > b.trust;
            });

  const Observation& reference = observations[candidates.front().index];
  double agreeing_trust = 0.0;
  double total_trust = 0.0;
  std::vector<size_t> agreeing_indices;
  for (const auto& candidate : candidates) {
    total_trust += candidate.trust;
    const auto& obs = observations[candidate.index];
    bool same_class = obs.object_class == reference.object_class ||
                      obs.object_class == ObjectClass::kUnknown ||
                      reference.object_class == ObjectClass::kUnknown;
    bool same_position =
        obs.position.DistanceTo(reference.position) <= position_tolerance_m;
    if (same_class && same_position) {
      agreeing_trust += candidate.trust;
      agreeing_indices.push_back(candidate.index);
      result.agreeing_agents.push_back(obs.agent_id);
    } else {
      result.disagreeing_agents.push_back(obs.agent_id);
    }
  }

  result.agreement_ratio =
      total_trust > 0.0 ? agreeing_trust / total_trust : 0.0;
  result.consensus_reached = result.agreement_ratio > (2.0 / 3.0);
  if (!result.consensus_reached) return result;

  double weight_sum = 0.0;
  for (size_t index : agreeing_indices) {
    double trust = index < agent_trusts.size()
                       ? std::clamp(agent_trusts[index], 0.0, 1.0)
                       : 0.5;
    double weight = trust * std::max(observations[index].confidence, 0.0);
    result.consensus_observation.position.x +=
        weight * observations[index].position.x;
    result.consensus_observation.position.y +=
        weight * observations[index].position.y;
    result.consensus_observation.position.z +=
        weight * observations[index].position.z;
    result.consensus_observation.confidence +=
        weight * observations[index].confidence;
    weight_sum += weight;
  }
  if (weight_sum > 0.0) {
    result.consensus_observation = observations[candidates.front().index];
    result.consensus_observation.position.x = 0.0;
    result.consensus_observation.position.y = 0.0;
    result.consensus_observation.position.z = 0.0;
    result.consensus_observation.confidence = 0.0;
    for (size_t index : agreeing_indices) {
      double trust = index < agent_trusts.size()
                         ? std::clamp(agent_trusts[index], 0.0, 1.0)
                         : 0.5;
      double weight = trust * std::max(observations[index].confidence, 0.0);
      result.consensus_observation.position.x +=
          weight * observations[index].position.x / weight_sum;
      result.consensus_observation.position.y +=
          weight * observations[index].position.y / weight_sum;
      result.consensus_observation.position.z +=
          weight * observations[index].position.z / weight_sum;
      result.consensus_observation.confidence +=
          weight * observations[index].confidence / weight_sum;
    }
  }
  return result;
}

}  // namespace omnicopilot

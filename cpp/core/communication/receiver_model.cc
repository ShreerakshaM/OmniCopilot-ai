// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/communication/receiver_model.h"

#include <cmath>
#include <unordered_map>

namespace omnicopilot {

struct ReceiverModel::Impl {
  ReceiverModelConfig config;
  std::unordered_map<std::string, ReceiverBelief> beliefs;
};

ReceiverModel::ReceiverModel(ReceiverModelConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

ReceiverModel::~ReceiverModel() = default;
ReceiverModel::ReceiverModel(ReceiverModel&&) noexcept = default;
ReceiverModel& ReceiverModel::operator=(ReceiverModel&&) noexcept = default;

void ReceiverModel::RegisterAgent(const std::string& agent_id,
                                  Position3D initial_position) {
  ReceiverBelief belief;
  belief.agent_id = agent_id;
  belief.position = initial_position;
  impl_->beliefs.emplace(agent_id, std::move(belief));
}

void ReceiverModel::RecordTransmission(const std::string& receiver_id,
                                       const std::string& entity_id,
                                       double confidence, double time_s) {
  auto it = impl_->beliefs.find(receiver_id);
  if (it == impl_->beliefs.end()) return;

  // Update or add known entity.
  for (auto& known : it->second.known_entities) {
    if (known.entity_id == entity_id) {
      known.estimated_confidence = confidence;
      known.last_informed_time_s = time_s;
      return;
    }
  }
  it->second.known_entities.push_back({entity_id, confidence, time_s});
}

double ReceiverModel::EstimateNovelty(const std::string& receiver_id,
                                      const Observation& obs,
                                      double current_time_s) const {
  // TODO: Full implementation with belief decay.
  // For now: if receiver has no record of this entity's track, novelty = 1.0.
  auto it = impl_->beliefs.find(receiver_id);
  if (it == impl_->beliefs.end()) return 1.0;

  for (const auto& known : it->second.known_entities) {
    if (known.entity_id == obs.track_id) {
      // Known — novelty depends on how stale their knowledge is.
      double age = current_time_s - known.last_informed_time_s;
      double decay = 1.0 - std::exp(-age / impl_->config.belief_decay_halflife_s);
      return decay;  // Older knowledge → higher novelty of an update.
    }
  }
  return 1.0;  // Completely new information.
}

double ReceiverModel::EstimateInformationGain(const std::string& receiver_id,
                                              const Observation& obs,
                                              double current_time_s) const {
  double novelty = EstimateNovelty(receiver_id, obs, current_time_s);
  return novelty * obs.confidence;
}

const ReceiverBelief* ReceiverModel::GetBelief(
    const std::string& agent_id) const {
  auto it = impl_->beliefs.find(agent_id);
  if (it != impl_->beliefs.end()) return &it->second;
  return nullptr;
}

void ReceiverModel::Tick(double current_time_s) {
  // TODO: Decay beliefs about what receivers know over time.
}

}  // namespace omnicopilot

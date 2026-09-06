// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/communication/prioritizer.h"

#include <algorithm>
#include <cmath>

namespace omnicopilot {

struct Prioritizer::Impl {
  PrioritizationWeights weights;
};

Prioritizer::Prioritizer(PrioritizationWeights weights)
    : impl_(std::make_unique<Impl>()) {
  impl_->weights = std::move(weights);
}

Prioritizer::~Prioritizer() = default;
Prioritizer::Prioritizer(Prioritizer&&) noexcept = default;
Prioritizer& Prioritizer::operator=(Prioritizer&&) noexcept = default;

ScoredObservation Prioritizer::Score(const Observation& obs) const {
  ScoredObservation scored;
  scored.observation = obs;

  // Safety relevance: pedestrians and cyclists are highest priority,
  // then vehicles, then static objects.
  switch (obs.object_class) {
    case ObjectClass::kPedestrian:
      scored.safety_score = 1.0;
      break;
    case ObjectClass::kCyclist:
      scored.safety_score = 0.95;
      break;
    case ObjectClass::kMotorcycle:
      scored.safety_score = 0.85;
      break;
    case ObjectClass::kVehicle:
      scored.safety_score = 0.7;
      break;
    case ObjectClass::kTruck:
    case ObjectClass::kBus:
      scored.safety_score = 0.65;
      break;
    case ObjectClass::kStaticObstacle:
      scored.safety_score = 0.4;
      break;
    case ObjectClass::kTrafficSign:
    case ObjectClass::kTrafficLight:
      scored.safety_score = 0.3;
      break;
    default:
      scored.safety_score = 0.2;
      break;
  }

  // Confidence score: moderate confidence is most valuable.
  // Very high confidence = receiver likely already knows.
  // Very low confidence = probably noise.
  // Peak value around 0.5-0.7.
  scored.confidence_score =
      1.0 - std::pow(2.0 * obs.confidence - 1.0, 2.0);
  scored.confidence_score = std::clamp(scored.confidence_score, 0.0, 1.0);

  // Novelty: approximated by inverse confidence (low confidence = likely new).
  // True novelty requires receiver model — this is the hand-designed fallback.
  scored.novelty_score = 1.0 - obs.confidence;

  // Time sensitivity: faster objects are more time-sensitive.
  double speed = obs.velocity.Magnitude();
  scored.time_sensitivity_score = std::clamp(speed / 20.0, 0.0, 1.0);

  // Receiver benefit: placeholder — the full receiver model handles this.
  // Default: proportional to safety × (1 - confidence).
  scored.receiver_benefit_score =
      scored.safety_score * (1.0 - obs.confidence);

  // Total weighted score.
  scored.total_score =
      impl_->weights.safety_relevance * scored.safety_score +
      impl_->weights.confidence * scored.confidence_score +
      impl_->weights.novelty * scored.novelty_score +
      impl_->weights.time_sensitivity * scored.time_sensitivity_score +
      impl_->weights.receiver_benefit * scored.receiver_benefit_score;

  return scored;
}

PrioritizationResult Prioritizer::Prioritize(
    const std::vector<Observation>& observations, uint32_t budget_bytes,
    uint32_t bytes_per_observation) const {
  PrioritizationResult result;

  if (observations.empty() || budget_bytes == 0) {
    return result;
  }

  // Score all observations.
  std::vector<ScoredObservation> scored;
  scored.reserve(observations.size());
  for (const auto& obs : observations) {
    scored.push_back(Score(obs));
  }

  // Sort by total score (descending).
  std::sort(scored.begin(), scored.end(),
            [](const ScoredObservation& a, const ScoredObservation& b) {
              return a.total_score > b.total_score;
            });

  // Select top-K within budget.
  uint32_t remaining_budget = budget_bytes;

  for (auto& s : scored) {
    if (remaining_budget >= bytes_per_observation) {
      result.selected.push_back(std::move(s));
      result.total_selected_value += s.total_score;
      remaining_budget -= bytes_per_observation;
    } else {
      result.suppressed.push_back(std::move(s));
      result.total_suppressed_value += s.total_score;
    }
  }

  return result;
}

void Prioritizer::SetWeights(PrioritizationWeights weights) {
  impl_->weights = std::move(weights);
}

}  // namespace omnicopilot

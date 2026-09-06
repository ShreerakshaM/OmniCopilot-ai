// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/world_model/uncertainty.h"

#include <algorithm>
#include <cmath>

namespace omnicopilot {

UncertaintyManager::UncertaintyManager(UncertaintyConfig config)
    : config_(std::move(config)) {}

UncertaintyManager::~UncertaintyManager() = default;

double UncertaintyManager::Decay(double current_confidence,
                                 double elapsed_s) const {
  if (elapsed_s <= 0.0) {
    return current_confidence;
  }

  // Exponential decay: c(t) = c0 * 2^(-t / halflife)
  // This means confidence halves every halflife seconds.
  double decay_factor = std::pow(2.0, -elapsed_s / config_.confidence_halflife_s);
  double decayed = current_confidence * decay_factor;

  return std::max(decayed, config_.min_confidence);
}

double UncertaintyManager::Boost(double current_confidence,
                                 double observation_confidence,
                                 double agent_trust) const {
  // Effective observation strength = observation confidence weighted by trust.
  double effective_obs = observation_confidence * agent_trust;

  // Bayesian-style update: treat current and new as independent evidence.
  // P(A|B) = 1 - (1 - P(A)) * (1 - P(B))
  // This gives diminishing returns as confidence approaches 1.0.
  double combined =
      1.0 - (1.0 - current_confidence) * (1.0 - effective_obs);

  // Additional corroboration boost when a new independent source agrees.
  combined += config_.corroboration_boost * agent_trust;

  return std::clamp(combined, config_.min_confidence, config_.max_confidence);
}

double UncertaintyManager::CombineIndependent(const double* confidences,
                                              const double* trusts,
                                              uint32_t count) const {
  if (count == 0) {
    return 0.0;
  }
  if (count == 1) {
    return std::clamp(confidences[0] * trusts[0], config_.min_confidence,
                      config_.max_confidence);
  }

  // Combine independent evidence: P = 1 - ∏(1 - p_i * t_i)
  // Each source independently contributes evidence weighted by trust.
  double product_of_complements = 1.0;
  for (uint32_t i = 0; i < count; ++i) {
    double effective = confidences[i] * trusts[i];
    effective = std::clamp(effective, 0.0, 1.0);
    product_of_complements *= (1.0 - effective);
  }

  double combined = 1.0 - product_of_complements;
  return std::clamp(combined, config_.min_confidence, config_.max_confidence);
}

bool UncertaintyManager::IsStale(double confidence) const {
  return confidence <= config_.stale_threshold;
}

bool UncertaintyManager::IsSafetyCritical(double confidence,
                                          double time_to_collision_s) const {
  // An entity is safety-critical if:
  // 1. It's reasonably confident (not noise), AND
  // 2. Time-to-collision is short.
  constexpr double kMinConfidenceForSafety = 0.3;
  constexpr double kCriticalTTC_s = 5.0;

  return confidence >= kMinConfidenceForSafety && time_to_collision_s >= 0.0 &&
         time_to_collision_s <= kCriticalTTC_s;
}

}  // namespace omnicopilot

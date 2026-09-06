// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/world_model/fusion.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace omnicopilot {

// Monotonically increasing entity ID counter.
static uint64_t g_next_entity_id = 1;

static std::string GenerateEntityId() {
  return "entity_" + std::to_string(g_next_entity_id++);
}

struct FusionEngine::Impl {
  FusionConfig config;
};

FusionEngine::FusionEngine(FusionConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

FusionEngine::~FusionEngine() = default;
FusionEngine::FusionEngine(FusionEngine&&) noexcept = default;
FusionEngine& FusionEngine::operator=(FusionEngine&&) noexcept = default;

std::vector<AssociationResult> FusionEngine::Associate(
    const std::vector<Observation>& observations,
    const std::vector<TrackedEntity>& existing_entities) const {
  std::vector<AssociationResult> results;
  results.reserve(observations.size());

  if (existing_entities.empty()) {
    // No existing entities — all observations are new.
    for (size_t i = 0; i < observations.size(); ++i) {
      results.push_back({i, /*entity_id=*/"", /*score=*/0.0});
    }
    return results;
  }

  // Greedy nearest-neighbor association:
  // For each observation, find the closest existing entity within the distance
  // threshold. Use a greedy approach — assign closest pairs first to avoid
  // multiple observations claiming the same entity.
  //
  // A full Hungarian algorithm would be optimal, but greedy is sufficient for
  // typical agent observation counts (<100) and much simpler.

  // Track which entities have already been matched.
  std::set<std::string> matched_entities;

  // Score all (observation, entity) pairs.
  struct Candidate {
    size_t obs_idx;
    size_t entity_idx;
    double distance;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(observations.size() * existing_entities.size());

  for (size_t oi = 0; oi < observations.size(); ++oi) {
    for (size_t ei = 0; ei < existing_entities.size(); ++ei) {
      double dist =
          observations[oi].position.DistanceTo(existing_entities[ei].position);
      if (dist <= impl_->config.association_max_distance_m) {
        // Also check class compatibility — don't match a pedestrian to a truck.
        if (observations[oi].object_class != ObjectClass::kUnknown &&
            existing_entities[ei].object_class != ObjectClass::kUnknown &&
            observations[oi].object_class !=
                existing_entities[ei].object_class) {
          continue;  // Class mismatch — skip.
        }
        candidates.push_back({oi, ei, dist});
      }
    }
  }

  // Sort by distance (closest first).
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.distance < b.distance;
            });

  // Greedy assignment.
  std::set<size_t> matched_obs;

  for (const auto& c : candidates) {
    if (matched_obs.count(c.obs_idx) > 0) {
      continue;  // Observation already assigned.
    }
    const auto& eid = existing_entities[c.entity_idx].entity_id;
    if (matched_entities.count(eid) > 0) {
      continue;  // Entity already claimed.
    }

    // Score: inverse distance, normalized by max distance.
    double score =
        1.0 - (c.distance / impl_->config.association_max_distance_m);
    results.push_back({c.obs_idx, eid, score});
    matched_obs.insert(c.obs_idx);
    matched_entities.insert(eid);
  }

  // Remaining unmatched observations → new entities.
  for (size_t i = 0; i < observations.size(); ++i) {
    if (matched_obs.count(i) == 0) {
      results.push_back({i, /*entity_id=*/"", /*score=*/0.0});
    }
  }

  return results;
}

void FusionEngine::FuseObservation(TrackedEntity& entity,
                                   const Observation& obs,
                                   double agent_trust) const {
  // Effective weight of this observation.
  double weight = obs.confidence * agent_trust;

  // Weighted running average for position.
  // New position = (old_weight * old_pos + new_weight * new_pos) /
  //                (old_weight + new_weight)
  double old_weight = entity.confidence;
  double total_weight = old_weight + weight;

  if (total_weight > 1e-10) {
    double alpha = weight / total_weight;
    entity.position.x += alpha * (obs.position.x - entity.position.x);
    entity.position.y += alpha * (obs.position.y - entity.position.y);
    entity.position.z += alpha * (obs.position.z - entity.position.z);

    entity.velocity.vx += alpha * (obs.velocity.vx - entity.velocity.vx);
    entity.velocity.vy += alpha * (obs.velocity.vy - entity.velocity.vy);
    entity.velocity.vz += alpha * (obs.velocity.vz - entity.velocity.vz);
  }

  // Update heading (circular mean — handle wrap-around).
  if (std::abs(obs.heading) > 1e-10 || std::abs(entity.heading) > 1e-10) {
    double alpha = weight / std::max(total_weight, 1e-10);
    // Simple weighted interpolation (works when angles are close).
    double diff = obs.heading - entity.heading;
    // Normalize to [-pi, pi].
    while (diff > M_PI) diff -= 2.0 * M_PI;
    while (diff < -M_PI) diff += 2.0 * M_PI;
    entity.heading += alpha * diff;
  }

  // Update bounding box (weighted average).
  if (obs.length > 0.0) {
    double alpha = weight / std::max(total_weight, 1e-10);
    entity.length += alpha * (obs.length - entity.length);
    entity.width += alpha * (obs.width - entity.width);
    entity.height += alpha * (obs.height - entity.height);
  }

  // Update confidence: Bayesian combination of independent evidence.
  // P(A|B) = 1 - (1 - P(A)) * (1 - P(B))
  entity.confidence = 1.0 - (1.0 - entity.confidence) * (1.0 - weight);
  entity.confidence = std::clamp(entity.confidence, 0.0, 0.99);

  // Update class if new observation has higher class confidence.
  if (obs.confidence * agent_trust > entity.class_confidence) {
    entity.object_class = obs.object_class;
    entity.class_confidence = obs.confidence * agent_trust;
  }

  // Update metadata.
  entity.last_updated_s = obs.timestamp_s;
  entity.observation_count++;

  // Track unique sources.
  bool new_source = true;
  for (const auto& ev : entity.supporting_evidence) {
    if (ev.agent_id == obs.agent_id) {
      new_source = false;
      break;
    }
  }
  if (new_source) {
    entity.unique_source_count++;
  }

  // Add evidence record.
  Evidence ev;
  ev.agent_id = obs.agent_id;
  ev.observation_id = obs.observation_id;
  ev.confidence = obs.confidence;
  ev.timestamp_s = obs.timestamp_s;
  ev.sensor_type = obs.sensor_type;
  entity.supporting_evidence.push_back(std::move(ev));

  // Cap evidence history to prevent unbounded growth.
  constexpr size_t kMaxEvidence = 50;
  if (entity.supporting_evidence.size() > kMaxEvidence) {
    entity.supporting_evidence.erase(entity.supporting_evidence.begin());
  }

  // Update flags.
  entity.needs_corroboration = (entity.unique_source_count < 2);
}

void FusionEngine::ResolveConflict(
    TrackedEntity& entity, const std::vector<Observation>& conflicting_obs,
    const std::vector<double>& agent_trusts) const {
  if (conflicting_obs.empty()) {
    return;
  }

  // Weighted majority vote on class.
  std::unordered_map<int, double> class_votes;
  double total_vote_weight = 0.0;

  for (size_t i = 0; i < conflicting_obs.size(); ++i) {
    double trust = (i < agent_trusts.size()) ? agent_trusts[i] : 0.5;
    double vote_weight = conflicting_obs[i].confidence * trust;
    int cls = static_cast<int>(conflicting_obs[i].object_class);
    class_votes[cls] += vote_weight;
    total_vote_weight += vote_weight;
  }

  // Find winning class.
  int best_class = static_cast<int>(ObjectClass::kUnknown);
  double best_weight = 0.0;
  for (const auto& [cls, w] : class_votes) {
    if (w > best_weight) {
      best_weight = w;
      best_class = cls;
    }
  }

  // Only accept if winning class has sufficient agreement.
  double agreement_ratio =
      (total_vote_weight > 0.0) ? (best_weight / total_vote_weight) : 0.0;

  if (agreement_ratio >= impl_->config.conflict_resolution_threshold) {
    entity.object_class = static_cast<ObjectClass>(best_class);
    entity.class_confidence = agreement_ratio;
  } else {
    // Insufficient agreement — mark as uncertain.
    entity.class_confidence *= 0.5;
    entity.needs_corroboration = true;
  }

  // Weighted average of positions from agreeing observations.
  double pos_weight_sum = 0.0;
  double px = 0.0, py = 0.0, pz = 0.0;

  for (size_t i = 0; i < conflicting_obs.size(); ++i) {
    double trust = (i < agent_trusts.size()) ? agent_trusts[i] : 0.5;
    int cls = static_cast<int>(conflicting_obs[i].object_class);
    if (cls == best_class || agreement_ratio < impl_->config.conflict_resolution_threshold) {
      double w = conflicting_obs[i].confidence * trust;
      px += w * conflicting_obs[i].position.x;
      py += w * conflicting_obs[i].position.y;
      pz += w * conflicting_obs[i].position.z;
      pos_weight_sum += w;
    }
  }

  if (pos_weight_sum > 1e-10) {
    entity.position.x = px / pos_weight_sum;
    entity.position.y = py / pos_weight_sum;
    entity.position.z = pz / pos_weight_sum;
  }

  // Confidence reduced due to conflict.
  entity.confidence *= 0.8;
}

TrackedEntity FusionEngine::CreateEntity(const Observation& obs,
                                         double agent_trust) const {
  TrackedEntity entity;
  entity.entity_id = GenerateEntityId();

  // Copy kinematic state from observation.
  entity.object_class = obs.object_class;
  entity.class_confidence = obs.confidence * agent_trust;
  entity.position = obs.position;
  entity.velocity = obs.velocity;
  entity.heading = obs.heading;
  entity.length = obs.length;
  entity.width = obs.width;
  entity.height = obs.height;

  // Initial confidence = observation confidence weighted by agent trust.
  entity.confidence = obs.confidence * agent_trust;

  // Lifecycle.
  entity.state = EntityState::kTentative;
  entity.first_seen_s = obs.timestamp_s;
  entity.last_updated_s = obs.timestamp_s;
  entity.observation_count = 1;
  entity.unique_source_count = 1;

  // Initial evidence.
  Evidence ev;
  ev.agent_id = obs.agent_id;
  ev.observation_id = obs.observation_id;
  ev.confidence = obs.confidence;
  ev.timestamp_s = obs.timestamp_s;
  ev.sensor_type = obs.sensor_type;
  entity.supporting_evidence.push_back(std::move(ev));

  // Flags.
  entity.needs_corroboration = true;  // Only one source so far.
  entity.is_safety_critical = false;
  entity.active_acquisition_pending = false;

  return entity;
}

}  // namespace omnicopilot

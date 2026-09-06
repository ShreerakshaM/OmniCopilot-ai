// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/world_model/world_model.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

namespace omnicopilot {

struct WorldModel::Impl {
  WorldModelConfig config;

  // Sub-components.
  FusionEngine fusion;
  UncertaintyManager uncertainty;
  SpatialIndex spatial_index;

  // Per-entity temporal trackers.
  std::unordered_map<std::string, TemporalTracker> trackers;

  // All tracked entities, keyed by entity_id.
  std::unordered_map<std::string, TrackedEntity> entities;

  // Simulation state.
  uint64_t tick = 0;
  double current_time_s = 0.0;

  Impl(WorldModelConfig cfg)
      : config(std::move(cfg)),
        fusion(FusionConfig{}),
        uncertainty(UncertaintyConfig{}),
        spatial_index(SpatialIndexConfig{cfg.spatial_cell_size_m}) {}

  // Update entity lifecycle state based on confidence and timing.
  void UpdateEntityState(TrackedEntity& entity) {
    double elapsed = current_time_s - entity.last_updated_s;

    // Tentative → Confirmed: enough sources and confidence.
    if (entity.state == EntityState::kTentative &&
        entity.unique_source_count >= config.confirmation_source_count &&
        entity.confidence >= config.confirmation_threshold) {
      entity.state = EntityState::kConfirmed;
      spdlog::debug("Entity {} confirmed (sources={}, conf={:.2f})",
                    entity.entity_id, entity.unique_source_count,
                    entity.confidence);
    }

    // Any → Stale: not observed for too long.
    if (entity.state != EntityState::kLost &&
        entity.state != EntityState::kStale && entity.state != EntityState::kPredicted &&
        elapsed > config.stale_timeout_s) {
      entity.state = EntityState::kPredicted;
    }

    // Predicted → Stale: confidence dropped.
    if (entity.state == EntityState::kPredicted &&
        entity.confidence < config.stale_confidence_threshold) {
      entity.state = EntityState::kStale;
    }

    // Stale → Lost: too long without observation.
    if (entity.state == EntityState::kStale &&
        elapsed > config.removal_timeout_s) {
      entity.state = EntityState::kLost;
    }
  }

  // Remove entities marked as Lost.
  void PurgeEntities() {
    std::vector<std::string> to_remove;
    for (const auto& [eid, entity] : entities) {
      if (entity.state == EntityState::kLost) {
        to_remove.push_back(eid);
      }
    }
    for (const auto& eid : to_remove) {
      spatial_index.Remove(eid);
      trackers.erase(eid);
      entities.erase(eid);
    }
    if (!to_remove.empty()) {
      spdlog::debug("Purged {} lost entities", to_remove.size());
    }
  }
};

WorldModel::WorldModel(WorldModelConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

WorldModel::~WorldModel() = default;
WorldModel::WorldModel(WorldModel&&) noexcept = default;
WorldModel& WorldModel::operator=(WorldModel&&) noexcept = default;

void WorldModel::Tick(double current_time_s) {
  impl_->tick++;
  double prev_time = impl_->current_time_s;
  impl_->current_time_s = current_time_s;
  double dt = current_time_s - prev_time;

  if (dt <= 0.0) return;

  for (auto& [eid, entity] : impl_->entities) {
    // 1. Decay confidence over time.
    entity.confidence =
        impl_->uncertainty.Decay(entity.confidence, dt);

    // 2. Predict position forward via temporal tracker.
    auto tracker_it = impl_->trackers.find(eid);
    if (tracker_it != impl_->trackers.end() &&
        tracker_it->second.IsInitialized()) {
      tracker_it->second.Predict(current_time_s);

      // Update entity position from tracker prediction.
      entity.position = tracker_it->second.GetPosition();
      entity.velocity = tracker_it->second.GetVelocity();

      // Regenerate predictions.
      entity.predictions = tracker_it->second.GetPredictions();
    }

    // 3. Update spatial index with new predicted position.
    impl_->spatial_index.Upsert(eid, entity.position);

    // 4. Update lifecycle state.
    impl_->UpdateEntityState(entity);

    // 5. Update safety-critical flag.
    entity.is_safety_critical = impl_->uncertainty.IsSafetyCritical(
        entity.confidence, entity.time_to_collision_s);
  }

  // 6. Purge lost entities.
  impl_->PurgeEntities();
}

void WorldModel::IngestObservations(
    const std::vector<Observation>& observations, double agent_trust) {
  if (observations.empty()) return;

  // Collect current entities as a vector for association.
  std::vector<TrackedEntity> entity_vec;
  entity_vec.reserve(impl_->entities.size());
  for (const auto& [eid, entity] : impl_->entities) {
    entity_vec.push_back(entity);
  }

  // Associate observations to existing entities.
  auto associations = impl_->fusion.Associate(observations, entity_vec);

  for (const auto& assoc : associations) {
    const auto& obs = observations[assoc.observation_idx];

    if (assoc.entity_id.empty()) {
      // New entity — no match found.
      if (impl_->entities.size() >= impl_->config.max_entities) {
        spdlog::warn("Max entities ({}) reached, dropping new observation",
                     impl_->config.max_entities);
        continue;
      }

      TrackedEntity new_entity =
          impl_->fusion.CreateEntity(obs, agent_trust);
      std::string eid = new_entity.entity_id;

      // Initialize temporal tracker.
      TemporalTracker tracker(TemporalTrackerConfig{});
      tracker.Initialize(obs);
      impl_->trackers.emplace(eid, std::move(tracker));

      // Add to spatial index.
      impl_->spatial_index.Upsert(eid, new_entity.position);

      // Store entity.
      impl_->entities.emplace(eid, std::move(new_entity));

      spdlog::debug("Created entity {} from agent {} (conf={:.2f})", eid,
                    obs.agent_id, obs.confidence);
    } else {
      // Existing entity — fuse observation.
      auto it = impl_->entities.find(assoc.entity_id);
      if (it == impl_->entities.end()) continue;

      TrackedEntity& entity = it->second;
      impl_->fusion.FuseObservation(entity, obs, agent_trust);

      // Update temporal tracker.
      auto tracker_it = impl_->trackers.find(assoc.entity_id);
      if (tracker_it != impl_->trackers.end()) {
        tracker_it->second.Update(obs);
        entity.position = tracker_it->second.GetPosition();
        entity.velocity = tracker_it->second.GetVelocity();
        entity.predictions = tracker_it->second.GetPredictions();
      }

      // Update spatial index.
      impl_->spatial_index.Upsert(assoc.entity_id, entity.position);

      // Re-evaluate lifecycle state.
      impl_->UpdateEntityState(entity);

      // If entity was predicted/stale and got a new observation, revive it.
      if (entity.state == EntityState::kPredicted ||
          entity.state == EntityState::kStale) {
        if (entity.unique_source_count >= impl_->config.confirmation_source_count &&
            entity.confidence >= impl_->config.confirmation_threshold) {
          entity.state = EntityState::kConfirmed;
        } else {
          entity.state = EntityState::kTentative;
        }
      }
    }
  }

  // Update current time from observations.
  for (const auto& obs : observations) {
    if (obs.timestamp_s > impl_->current_time_s) {
      impl_->current_time_s = obs.timestamp_s;
    }
  }
}

std::vector<const TrackedEntity*> WorldModel::QueryRadius(
    Position3D center, double radius_m) const {
  auto entity_ids = impl_->spatial_index.QueryRadius(center, radius_m);

  std::vector<const TrackedEntity*> result;
  result.reserve(entity_ids.size());
  for (const auto& eid : entity_ids) {
    auto it = impl_->entities.find(eid);
    if (it != impl_->entities.end()) {
      result.push_back(&it->second);
    }
  }
  return result;
}

const TrackedEntity* WorldModel::GetEntity(
    const std::string& entity_id) const {
  auto it = impl_->entities.find(entity_id);
  if (it != impl_->entities.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<const TrackedEntity*> WorldModel::GetEntities(
    double min_confidence, int object_class) const {
  std::vector<const TrackedEntity*> result;
  for (const auto& [eid, entity] : impl_->entities) {
    if (entity.confidence < min_confidence) continue;
    if (object_class >= 0 &&
        static_cast<int>(entity.object_class) != object_class) {
      continue;
    }
    if (entity.state == EntityState::kLost) continue;
    result.push_back(&entity);
  }
  return result;
}

std::vector<const TrackedEntity*> WorldModel::GetUncertainEntities() const {
  std::vector<const TrackedEntity*> result;
  for (const auto& [eid, entity] : impl_->entities) {
    if (entity.state == EntityState::kLost) continue;
    if (entity.needs_corroboration) {
      result.push_back(&entity);
    }
  }
  return result;
}

std::vector<const TrackedEntity*> WorldModel::GetSafetyCriticalEntities()
    const {
  std::vector<const TrackedEntity*> result;
  for (const auto& [eid, entity] : impl_->entities) {
    if (entity.state == EntityState::kLost) continue;
    if (entity.is_safety_critical) {
      result.push_back(&entity);
    }
  }
  return result;
}

WorldModelStats WorldModel::GetStats() const {
  WorldModelStats stats;
  stats.tick = impl_->tick;

  double total_conf = 0.0;
  for (const auto& [eid, entity] : impl_->entities) {
    stats.total_entities++;
    total_conf += entity.confidence;

    switch (entity.state) {
      case EntityState::kTentative:
        stats.tentative++;
        break;
      case EntityState::kConfirmed:
        stats.confirmed++;
        break;
      case EntityState::kPredicted:
        stats.predicted++;
        break;
      case EntityState::kStale:
        stats.stale++;
        break;
      case EntityState::kLost:
        // Lost entities should be purged, but count just in case.
        break;
    }
  }

  stats.mean_confidence =
      (stats.total_entities > 0) ? (total_conf / stats.total_entities) : 0.0;

  return stats;
}

uint64_t WorldModel::GetTick() const { return impl_->tick; }

double WorldModel::GetCurrentTime() const { return impl_->current_time_s; }

void WorldModel::Reset() {
  impl_->entities.clear();
  impl_->trackers.clear();
  impl_->spatial_index.Clear();
  impl_->tick = 0;
  impl_->current_time_s = 0.0;
  spdlog::info("World model reset");
}

}  // namespace omnicopilot

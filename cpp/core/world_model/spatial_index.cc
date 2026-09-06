// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/world_model/spatial_index.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace omnicopilot {

// Hash for grid cell coordinates (ix, iy).
struct CellHash {
  size_t operator()(const std::pair<int, int>& cell) const {
    // Combine two ints using a simple hash mix.
    auto h1 = std::hash<int>{}(cell.first);
    auto h2 = std::hash<int>{}(cell.second);
    return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) +
                  (h1 >> 2));
  }
};

using Cell = std::pair<int, int>;

struct SpatialIndex::Impl {
  SpatialIndexConfig config;

  // Grid: cell → set of entity IDs in that cell.
  std::unordered_map<Cell, std::unordered_set<std::string>, CellHash> grid;

  // Reverse lookup: entity_id → (cell, position).
  struct EntityEntry {
    Cell cell;
    Position3D position;
  };
  std::unordered_map<std::string, EntityEntry> entities;

  // Convert world position to grid cell.
  Cell ToCell(const Position3D& pos) const {
    int ix = static_cast<int>(std::floor(pos.x / config.cell_size_m));
    int iy = static_cast<int>(std::floor(pos.y / config.cell_size_m));
    return {ix, iy};
  }
};

SpatialIndex::SpatialIndex(SpatialIndexConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

SpatialIndex::~SpatialIndex() = default;
SpatialIndex::SpatialIndex(SpatialIndex&&) noexcept = default;
SpatialIndex& SpatialIndex::operator=(SpatialIndex&&) noexcept = default;

void SpatialIndex::Upsert(const std::string& entity_id, Position3D position) {
  Cell new_cell = impl_->ToCell(position);

  auto it = impl_->entities.find(entity_id);
  if (it != impl_->entities.end()) {
    // Entity exists — remove from old cell if it changed.
    Cell old_cell = it->second.cell;
    if (old_cell != new_cell) {
      auto& old_bucket = impl_->grid[old_cell];
      old_bucket.erase(entity_id);
      if (old_bucket.empty()) {
        impl_->grid.erase(old_cell);
      }
      impl_->grid[new_cell].insert(entity_id);
    }
    it->second.cell = new_cell;
    it->second.position = position;
  } else {
    // New entity.
    impl_->grid[new_cell].insert(entity_id);
    impl_->entities[entity_id] = {new_cell, position};
  }
}

void SpatialIndex::Remove(const std::string& entity_id) {
  auto it = impl_->entities.find(entity_id);
  if (it == impl_->entities.end()) {
    return;
  }

  Cell cell = it->second.cell;
  auto& bucket = impl_->grid[cell];
  bucket.erase(entity_id);
  if (bucket.empty()) {
    impl_->grid.erase(cell);
  }
  impl_->entities.erase(it);
}

std::vector<std::string> SpatialIndex::QueryRadius(Position3D center,
                                                   double radius_m) const {
  // Determine which cells could contain entities within the radius.
  double cell_size = impl_->config.cell_size_m;
  int min_ix = static_cast<int>(std::floor((center.x - radius_m) / cell_size));
  int max_ix = static_cast<int>(std::floor((center.x + radius_m) / cell_size));
  int min_iy = static_cast<int>(std::floor((center.y - radius_m) / cell_size));
  int max_iy = static_cast<int>(std::floor((center.y + radius_m) / cell_size));

  double radius_sq = radius_m * radius_m;
  std::vector<std::pair<double, std::string>> candidates;

  for (int ix = min_ix; ix <= max_ix; ++ix) {
    for (int iy = min_iy; iy <= max_iy; ++iy) {
      auto grid_it = impl_->grid.find({ix, iy});
      if (grid_it == impl_->grid.end()) {
        continue;
      }
      for (const auto& eid : grid_it->second) {
        auto ent_it = impl_->entities.find(eid);
        if (ent_it == impl_->entities.end()) {
          continue;
        }
        const Position3D& pos = ent_it->second.position;
        double dx = pos.x - center.x;
        double dy = pos.y - center.y;
        double dz = pos.z - center.z;
        double dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq <= radius_sq) {
          candidates.emplace_back(dist_sq, eid);
        }
      }
    }
  }

  // Sort by distance (ascending).
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  std::vector<std::string> result;
  result.reserve(candidates.size());
  for (auto& [dist, eid] : candidates) {
    result.push_back(std::move(eid));
  }
  return result;
}

std::vector<std::string> SpatialIndex::QueryKNearest(Position3D center,
                                                     uint32_t k) const {
  if (k == 0 || impl_->entities.empty()) {
    return {};
  }

  // Brute-force over all entities — acceptable for typical entity counts (<1000).
  // For larger counts, switch to expanding-ring search over grid cells.
  std::vector<std::pair<double, std::string>> all;
  all.reserve(impl_->entities.size());

  for (const auto& [eid, entry] : impl_->entities) {
    double dx = entry.position.x - center.x;
    double dy = entry.position.y - center.y;
    double dz = entry.position.z - center.z;
    double dist_sq = dx * dx + dy * dy + dz * dz;
    all.emplace_back(dist_sq, eid);
  }

  // Partial sort to get top-K.
  uint32_t n = std::min(k, static_cast<uint32_t>(all.size()));
  std::partial_sort(all.begin(), all.begin() + n, all.end(),
                    [](const auto& a, const auto& b) {
                      return a.first < b.first;
                    });

  std::vector<std::string> result;
  result.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    result.push_back(std::move(all[i].second));
  }
  return result;
}

void SpatialIndex::Clear() {
  impl_->grid.clear();
  impl_->entities.clear();
}

size_t SpatialIndex::Size() const { return impl_->entities.size(); }

}  // namespace omnicopilot

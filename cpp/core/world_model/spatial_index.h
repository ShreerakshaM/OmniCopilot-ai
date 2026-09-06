// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// SpatialIndex — Efficient spatial queries over tracked entities.
// Grid-based spatial hashing for O(1) neighbor lookup.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Configuration for spatial indexing.
struct SpatialIndexConfig {
  double cell_size_m = 5.0;      // Grid cell size in meters.
  double world_min_x = -500.0;   // World bounds.
  double world_max_x = 500.0;
  double world_min_y = -500.0;
  double world_max_y = 500.0;
};

/// Grid-based spatial index for fast radius queries.
///
/// Supports:
/// - Insert/update entity positions.
/// - Radius queries (all entities within R meters of a point).
/// - K-nearest-neighbor queries.
/// - Remove entity from index.
class SpatialIndex {
 public:
  explicit SpatialIndex(SpatialIndexConfig config);
  ~SpatialIndex();

  SpatialIndex(const SpatialIndex&) = delete;
  SpatialIndex& operator=(const SpatialIndex&) = delete;
  SpatialIndex(SpatialIndex&&) noexcept;
  SpatialIndex& operator=(SpatialIndex&&) noexcept;

  /// Insert or update an entity's position in the index.
  void Upsert(const std::string& entity_id, Position3D position);

  /// Remove an entity from the index.
  void Remove(const std::string& entity_id);

  /// Find all entity IDs within radius_m of the given center.
  std::vector<std::string> QueryRadius(Position3D center, double radius_m) const;

  /// Find the K nearest entity IDs to the given center.
  std::vector<std::string> QueryKNearest(Position3D center, uint32_t k) const;

  /// Clear all entries from the index.
  void Clear();

  /// Get the number of entities currently indexed.
  size_t Size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot

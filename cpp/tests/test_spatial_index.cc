// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "cpp/core/world_model/spatial_index.h"

namespace omnicopilot {
namespace {

class SpatialIndexTest : public ::testing::Test {
 protected:
  SpatialIndexConfig config;
  void SetUp() override {
    config.cell_size_m = 10.0;
    config.world_min_x = -500.0;
    config.world_max_x = 500.0;
    config.world_min_y = -500.0;
    config.world_max_y = 500.0;
  }
};

TEST_F(SpatialIndexTest, EmptyIndex) {
  SpatialIndex idx(config);
  EXPECT_EQ(idx.Size(), 0);
  EXPECT_TRUE(idx.QueryRadius({0, 0, 0}, 100.0).empty());
}

TEST_F(SpatialIndexTest, InsertAndQuery) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {10.0, 20.0, 0.0});

  auto results = idx.QueryRadius({10.0, 20.0, 0.0}, 1.0);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0], "e1");
}

TEST_F(SpatialIndexTest, QueryMiss) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {10.0, 20.0, 0.0});

  auto results = idx.QueryRadius({200.0, 200.0, 0.0}, 1.0);
  EXPECT_TRUE(results.empty());
}

TEST_F(SpatialIndexTest, MultipleEntities) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {0.0, 0.0, 0.0});
  idx.Upsert("e2", {3.0, 4.0, 0.0});   // 5m away from origin.
  idx.Upsert("e3", {100.0, 0.0, 0.0});  // Far away.

  auto results = idx.QueryRadius({0.0, 0.0, 0.0}, 6.0);
  EXPECT_EQ(results.size(), 2);  // e1 and e2.

  // Results should be sorted by distance (e1 closer than e2).
  EXPECT_EQ(results[0], "e1");
  EXPECT_EQ(results[1], "e2");
}

TEST_F(SpatialIndexTest, UpdatePosition) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {0.0, 0.0, 0.0});

  // Move far away.
  idx.Upsert("e1", {200.0, 200.0, 0.0});

  // Should NOT be at old position.
  EXPECT_TRUE(idx.QueryRadius({0.0, 0.0, 0.0}, 5.0).empty());
  // Should be at new position.
  EXPECT_EQ(idx.QueryRadius({200.0, 200.0, 0.0}, 5.0).size(), 1);
  // Size unchanged.
  EXPECT_EQ(idx.Size(), 1);
}

TEST_F(SpatialIndexTest, Remove) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {10.0, 10.0, 0.0});
  EXPECT_EQ(idx.Size(), 1);

  idx.Remove("e1");
  EXPECT_EQ(idx.Size(), 0);
  EXPECT_TRUE(idx.QueryRadius({10.0, 10.0, 0.0}, 5.0).empty());
}

TEST_F(SpatialIndexTest, RemoveNonexistentDoesNothing) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {10.0, 10.0, 0.0});
  idx.Remove("nonexistent");
  EXPECT_EQ(idx.Size(), 1);
}

TEST_F(SpatialIndexTest, Clear) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {10.0, 10.0, 0.0});
  idx.Upsert("e2", {20.0, 20.0, 0.0});
  idx.Clear();
  EXPECT_EQ(idx.Size(), 0);
}

TEST_F(SpatialIndexTest, QueryKNearest) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {0.0, 0.0, 0.0});
  idx.Upsert("e2", {3.0, 0.0, 0.0});
  idx.Upsert("e3", {10.0, 0.0, 0.0});
  idx.Upsert("e4", {50.0, 0.0, 0.0});

  auto results = idx.QueryKNearest({0.0, 0.0, 0.0}, 2);
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0], "e1");
  EXPECT_EQ(results[1], "e2");
}

TEST_F(SpatialIndexTest, QueryKNearestMoreThanAvailable) {
  SpatialIndex idx(config);
  idx.Upsert("e1", {0.0, 0.0, 0.0});

  auto results = idx.QueryKNearest({0.0, 0.0, 0.0}, 10);
  EXPECT_EQ(results.size(), 1);
}

TEST_F(SpatialIndexTest, CrossCellBoundaryQuery) {
  // Entity at cell boundary — query from neighboring cell should find it.
  SpatialIndex idx(config);  // cell_size = 10
  idx.Upsert("e1", {9.5, 0.0, 0.0});  // Near boundary of cell (0,0).

  // Query from next cell, but within radius.
  auto results = idx.QueryRadius({11.0, 0.0, 0.0}, 3.0);
  EXPECT_EQ(results.size(), 1);
}

}  // namespace
}  // namespace omnicopilot

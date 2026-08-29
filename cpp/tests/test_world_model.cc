// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "cpp/core/world_model/world_model.h"

namespace omnicopilot {
namespace {

class WorldModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_.max_entities = 100;
    world_model_ = std::make_unique<WorldModel>(config_);
  }

  WorldModelConfig config_;
  std::unique_ptr<WorldModel> world_model_;
};

TEST_F(WorldModelTest, InitialStateIsEmpty) {
  auto stats = world_model_->GetStats();
  EXPECT_EQ(stats.total_entities, 0);
  EXPECT_EQ(stats.confirmed, 0);
}

TEST_F(WorldModelTest, IngestCreatesEntity) {
  Observation obs;
  obs.observation_id = "obs_1";
  obs.agent_id = "agent_1";
  obs.track_id = "track_1";
  obs.object_class = ObjectClass::kPedestrian;
  obs.position = {10.0, 20.0, 0.0};
  obs.confidence = 0.9;
  obs.timestamp_s = 1.0;

  world_model_->IngestObservations({obs}, 0.95);

  auto stats = world_model_->GetStats();
  EXPECT_EQ(stats.total_entities, 1);
  EXPECT_EQ(stats.tentative, 1);
}

TEST_F(WorldModelTest, RadiusQueryFindsNearbyEntities) {
  Observation obs;
  obs.observation_id = "obs_1";
  obs.agent_id = "agent_1";
  obs.position = {10.0, 10.0, 0.0};
  obs.confidence = 0.9;
  obs.timestamp_s = 1.0;

  world_model_->IngestObservations({obs}, 1.0);

  auto results = world_model_->QueryRadius({10.0, 10.0, 0.0}, 5.0);
  EXPECT_EQ(results.size(), 1);

  auto far_results = world_model_->QueryRadius({100.0, 100.0, 0.0}, 5.0);
  EXPECT_EQ(far_results.size(), 0);
}

TEST_F(WorldModelTest, StaleEntitiesDecay) {
  Observation obs;
  obs.observation_id = "obs_1";
  obs.agent_id = "agent_1";
  obs.position = {0.0, 0.0, 0.0};
  obs.confidence = 0.9;
  obs.timestamp_s = 0.0;

  world_model_->IngestObservations({obs}, 1.0);

  // Advance time well past stale timeout.
  world_model_->Tick(10.0);

  auto stats = world_model_->GetStats();
  // Entity should be stale or removed.
  EXPECT_EQ(stats.confirmed, 0);
}

TEST_F(WorldModelTest, ResetClearsAllEntities) {
  Observation obs;
  obs.observation_id = "obs_1";
  obs.agent_id = "agent_1";
  obs.position = {0.0, 0.0, 0.0};
  obs.confidence = 0.9;
  obs.timestamp_s = 0.0;

  world_model_->IngestObservations({obs}, 1.0);
  world_model_->Reset();

  auto stats = world_model_->GetStats();
  EXPECT_EQ(stats.total_entities, 0);
}

}  // namespace
}  // namespace omnicopilot

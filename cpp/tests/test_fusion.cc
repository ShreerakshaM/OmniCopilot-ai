// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "cpp/core/world_model/entity.h"
#include "cpp/core/world_model/fusion.h"

namespace omnicopilot {
namespace {

Observation MakeObs(const std::string& agent, double x, double y,
                    double conf, ObjectClass cls = ObjectClass::kPedestrian) {
  Observation obs;
  obs.observation_id = "obs_" + agent;
  obs.agent_id = agent;
  obs.track_id = "t1";
  obs.object_class = cls;
  obs.position = {x, y, 0.0};
  obs.velocity = {1.0, 0.0, 0.0};
  obs.confidence = conf;
  obs.timestamp_s = 1.0;
  obs.length = 0.5;
  obs.width = 0.5;
  obs.height = 1.7;
  return obs;
}

class FusionTest : public ::testing::Test {
 protected:
  FusionConfig config;
  void SetUp() override {
    config.association_max_distance_m = 5.0;
    config.conflict_resolution_threshold = 0.6;
  }
};

TEST_F(FusionTest, CreateEntityFromObservation) {
  FusionEngine engine(config);
  auto obs = MakeObs("a1", 10.0, 20.0, 0.9);

  auto entity = engine.CreateEntity(obs, 1.0);

  EXPECT_FALSE(entity.entity_id.empty());
  EXPECT_NEAR(entity.position.x, 10.0, 1e-6);
  EXPECT_NEAR(entity.position.y, 20.0, 1e-6);
  EXPECT_EQ(entity.object_class, ObjectClass::kPedestrian);
  EXPECT_NEAR(entity.confidence, 0.9, 0.01);
  EXPECT_EQ(entity.state, EntityState::kTentative);
  EXPECT_EQ(entity.unique_source_count, 1);
  EXPECT_TRUE(entity.needs_corroboration);
  EXPECT_EQ(entity.supporting_evidence.size(), 1);
}

TEST_F(FusionTest, AssociateMatchesCloseObservation) {
  FusionEngine engine(config);
  auto obs = MakeObs("a2", 10.5, 20.2, 0.85);

  TrackedEntity existing = engine.CreateEntity(
      MakeObs("a1", 10.0, 20.0, 0.9), 1.0);
  std::vector<TrackedEntity> entities = {existing};

  auto results = engine.Associate({obs}, entities);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].entity_id, existing.entity_id);  // Matched!
  EXPECT_GT(results[0].score, 0.0);
}

TEST_F(FusionTest, AssociateCreatesNewForFarObservation) {
  FusionEngine engine(config);
  auto obs = MakeObs("a2", 100.0, 200.0, 0.85);

  TrackedEntity existing = engine.CreateEntity(
      MakeObs("a1", 10.0, 20.0, 0.9), 1.0);
  std::vector<TrackedEntity> entities = {existing};

  auto results = engine.Associate({obs}, entities);
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results[0].entity_id.empty());  // No match → new entity.
}

TEST_F(FusionTest, AssociateRejectsClassMismatch) {
  FusionEngine engine(config);
  // Close but different class.
  auto obs = MakeObs("a2", 10.1, 20.0, 0.85, ObjectClass::kTruck);

  TrackedEntity existing = engine.CreateEntity(
      MakeObs("a1", 10.0, 20.0, 0.9, ObjectClass::kPedestrian), 1.0);
  std::vector<TrackedEntity> entities = {existing};

  auto results = engine.Associate({obs}, entities);
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results[0].entity_id.empty());  // Class mismatch → new.
}

TEST_F(FusionTest, FuseObservationUpdatesPosition) {
  FusionEngine engine(config);
  auto entity = engine.CreateEntity(MakeObs("a1", 10.0, 20.0, 0.9), 1.0);

  auto obs2 = MakeObs("a2", 12.0, 22.0, 0.9);
  engine.FuseObservation(entity, obs2, 1.0);

  // Position should move toward the new observation.
  EXPECT_GT(entity.position.x, 10.0);
  EXPECT_LT(entity.position.x, 12.0);
  EXPECT_EQ(entity.observation_count, 2);
  EXPECT_EQ(entity.unique_source_count, 2);
  EXPECT_FALSE(entity.needs_corroboration);
}

TEST_F(FusionTest, FuseObservationIncreasesConfidence) {
  FusionEngine engine(config);
  auto entity = engine.CreateEntity(MakeObs("a1", 10.0, 20.0, 0.7), 0.8);
  double conf_before = entity.confidence;

  engine.FuseObservation(entity, MakeObs("a2", 10.1, 20.0, 0.8), 0.9);
  EXPECT_GT(entity.confidence, conf_before);
}

TEST_F(FusionTest, AssociateNoExistingEntities) {
  FusionEngine engine(config);
  auto obs = MakeObs("a1", 10.0, 20.0, 0.9);

  auto results = engine.Associate({obs}, {});
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results[0].entity_id.empty());
}

TEST_F(FusionTest, ResolveConflictMajorityWins) {
  FusionEngine engine(config);
  auto entity = engine.CreateEntity(MakeObs("a1", 10.0, 20.0, 0.9), 1.0);

  // Three observations: 2 say Vehicle, 1 says Pedestrian.
  std::vector<Observation> conflicting = {
      MakeObs("a1", 10.0, 20.0, 0.8, ObjectClass::kVehicle),
      MakeObs("a2", 10.5, 20.0, 0.85, ObjectClass::kVehicle),
      MakeObs("a3", 10.2, 20.0, 0.7, ObjectClass::kPedestrian),
  };
  std::vector<double> trusts = {0.9, 0.9, 0.9};

  engine.ResolveConflict(entity, conflicting, trusts);
  EXPECT_EQ(entity.object_class, ObjectClass::kVehicle);
}

}  // namespace
}  // namespace omnicopilot

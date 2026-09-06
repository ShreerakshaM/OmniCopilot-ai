// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>

#include "cpp/core/world_model/entity.h"
#include "cpp/core/world_model/uncertainty.h"
#include "cpp/core/world_model/world_model.h"

namespace omnicopilot {
namespace {

// ─── Helper: create a test observation ──────────────────────────────────────

Observation MakeObs(const std::string& obs_id, const std::string& agent_id,
                    double x, double y, double confidence,
                    double timestamp_s = 1.0,
                    ObjectClass cls = ObjectClass::kPedestrian) {
  Observation obs;
  obs.observation_id = obs_id;
  obs.agent_id = agent_id;
  obs.track_id = "track_" + obs_id;
  obs.object_class = cls;
  obs.position = {x, y, 0.0};
  obs.velocity = {1.0, 0.0, 0.0};
  obs.confidence = confidence;
  obs.timestamp_s = timestamp_s;
  obs.length = 0.5;
  obs.width = 0.5;
  obs.height = 1.7;
  obs.heading = 0.0;
  return obs;
}

// ─── Entity Tests ───────────────────────────────────────────────────────────

TEST(Position3DTest, DistanceToSelf) {
  Position3D p{1.0, 2.0, 3.0};
  EXPECT_DOUBLE_EQ(p.DistanceTo(p), 0.0);
}

TEST(Position3DTest, DistanceToOther) {
  Position3D a{0.0, 0.0, 0.0};
  Position3D b{3.0, 4.0, 0.0};
  EXPECT_DOUBLE_EQ(a.DistanceTo(b), 5.0);
}

TEST(Position3DTest, DistanceTo3D) {
  Position3D a{1.0, 2.0, 3.0};
  Position3D b{4.0, 6.0, 3.0};
  EXPECT_DOUBLE_EQ(a.DistanceTo(b), 5.0);
}

TEST(Velocity3DTest, Magnitude) {
  Velocity3D v{3.0, 4.0, 0.0};
  EXPECT_DOUBLE_EQ(v.Magnitude(), 5.0);
}

TEST(Velocity3DTest, MagnitudeZero) {
  Velocity3D v{0.0, 0.0, 0.0};
  EXPECT_DOUBLE_EQ(v.Magnitude(), 0.0);
}

// ─── Uncertainty Tests ──────────────────────────────────────────────────────

class UncertaintyTest : public ::testing::Test {
 protected:
  UncertaintyConfig config;
  void SetUp() override {
    config.confidence_halflife_s = 3.0;
    config.min_confidence = 0.01;
    config.stale_threshold = 0.15;
    config.corroboration_boost = 0.15;
    config.max_confidence = 0.99;
  }
};

TEST_F(UncertaintyTest, DecayHalvesAtHalflife) {
  UncertaintyManager mgr(config);
  double decayed = mgr.Decay(0.8, 3.0);  // One halflife.
  EXPECT_NEAR(decayed, 0.4, 0.01);
}

TEST_F(UncertaintyTest, DecayZeroTimeNoChange) {
  UncertaintyManager mgr(config);
  EXPECT_DOUBLE_EQ(mgr.Decay(0.8, 0.0), 0.8);
}

TEST_F(UncertaintyTest, DecayNeverBelowMin) {
  UncertaintyManager mgr(config);
  double decayed = mgr.Decay(0.1, 100.0);  // Very long time.
  EXPECT_GE(decayed, config.min_confidence);
}

TEST_F(UncertaintyTest, BoostIncreasesConfidence) {
  UncertaintyManager mgr(config);
  double boosted = mgr.Boost(0.5, 0.9, 1.0);
  EXPECT_GT(boosted, 0.5);
}

TEST_F(UncertaintyTest, BoostCappedAtMax) {
  UncertaintyManager mgr(config);
  double boosted = mgr.Boost(0.95, 0.99, 1.0);
  EXPECT_LE(boosted, config.max_confidence);
}

TEST_F(UncertaintyTest, CombineIndependentSingleSource) {
  UncertaintyManager mgr(config);
  double conf = 0.8;
  double trust = 0.9;
  double result = mgr.CombineIndependent(&conf, &trust, 1);
  EXPECT_NEAR(result, 0.72, 0.01);
}

TEST_F(UncertaintyTest, CombineIndependentMultipleSources) {
  UncertaintyManager mgr(config);
  double confs[] = {0.7, 0.8, 0.6};
  double trusts[] = {1.0, 1.0, 1.0};
  double result = mgr.CombineIndependent(confs, trusts, 3);
  // 1 - (1-0.7)*(1-0.8)*(1-0.6) = 1 - 0.3*0.2*0.4 = 1 - 0.024 = 0.976
  EXPECT_NEAR(result, 0.976, 0.01);
}

TEST_F(UncertaintyTest, CombineIndependentEmpty) {
  UncertaintyManager mgr(config);
  EXPECT_DOUBLE_EQ(mgr.CombineIndependent(nullptr, nullptr, 0), 0.0);
}

TEST_F(UncertaintyTest, IsStale) {
  UncertaintyManager mgr(config);
  EXPECT_TRUE(mgr.IsStale(0.1));
  EXPECT_FALSE(mgr.IsStale(0.5));
  EXPECT_TRUE(mgr.IsStale(0.15));  // At threshold.
}

TEST_F(UncertaintyTest, IsSafetyCritical) {
  UncertaintyManager mgr(config);
  EXPECT_TRUE(mgr.IsSafetyCritical(0.8, 2.0));   // High conf, close TTC.
  EXPECT_FALSE(mgr.IsSafetyCritical(0.1, 2.0));   // Low conf.
  EXPECT_FALSE(mgr.IsSafetyCritical(0.8, 10.0));  // Far TTC.
  EXPECT_FALSE(mgr.IsSafetyCritical(0.8, -1.0));  // No TTC.
}

// ─── WorldModel Integration Tests ───────────────────────────────────────────

class WorldModelTest : public ::testing::Test {
 protected:
  WorldModelConfig config;
  void SetUp() override {
    config.max_entities = 100;
    config.stale_timeout_s = 2.0;
    config.removal_timeout_s = 5.0;
    config.confirmation_threshold = 0.6;
    config.confirmation_source_count = 2;
    config.stale_confidence_threshold = 0.1;
  }
};

TEST_F(WorldModelTest, InitialStateIsEmpty) {
  WorldModel wm(config);
  auto stats = wm.GetStats();
  EXPECT_EQ(stats.total_entities, 0);
  EXPECT_EQ(stats.confirmed, 0);
  EXPECT_EQ(stats.tentative, 0);
  EXPECT_EQ(stats.tick, 0);
}

TEST_F(WorldModelTest, IngestCreatesEntity) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 20.0, 0.9)}, 0.95);

  auto stats = wm.GetStats();
  EXPECT_EQ(stats.total_entities, 1);
  EXPECT_EQ(stats.tentative, 1);  // Single source → tentative.
}

TEST_F(WorldModelTest, SecondSourceConfirmsEntity) {
  WorldModel wm(config);
  // First agent sees pedestrian.
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 20.0, 0.9)}, 0.95);
  // Second agent sees same pedestrian (close position).
  wm.IngestObservations({MakeObs("o2", "a2", 10.5, 20.2, 0.85)}, 0.90);

  auto stats = wm.GetStats();
  EXPECT_EQ(stats.total_entities, 1);
  EXPECT_EQ(stats.confirmed, 1);  // Two sources → confirmed.
}

TEST_F(WorldModelTest, FarObservationsCreateSeparateEntities) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 20.0, 0.9)}, 0.95);
  // Far away — should NOT match existing entity.
  wm.IngestObservations({MakeObs("o2", "a2", 100.0, 200.0, 0.85)}, 0.90);

  auto stats = wm.GetStats();
  EXPECT_EQ(stats.total_entities, 2);
}

TEST_F(WorldModelTest, RadiusQueryFindsNearby) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 10.0, 0.9)}, 1.0);
  wm.IngestObservations({MakeObs("o2", "a1", 100.0, 100.0, 0.9, 1.0,
                                  ObjectClass::kVehicle)},
                        1.0);

  auto nearby = wm.QueryRadius({10.0, 10.0, 0.0}, 5.0);
  EXPECT_EQ(nearby.size(), 1);

  auto far = wm.QueryRadius({100.0, 100.0, 0.0}, 5.0);
  EXPECT_EQ(far.size(), 1);

  auto none = wm.QueryRadius({500.0, 500.0, 0.0}, 5.0);
  EXPECT_EQ(none.size(), 0);
}

TEST_F(WorldModelTest, TickDecaysConfidence) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 10.0, 0.9, 0.0)}, 1.0);

  auto before = wm.GetEntities(0.0);
  ASSERT_EQ(before.size(), 1);
  double conf_before = before[0]->confidence;

  // Advance 5 seconds — confidence should decay significantly.
  wm.Tick(5.0);

  auto after = wm.GetEntities(0.0);
  ASSERT_EQ(after.size(), 1);
  EXPECT_LT(after[0]->confidence, conf_before);
}

TEST_F(WorldModelTest, StaleEntitiesGetPurged) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 10.0, 0.5, 0.0)}, 0.5);

  // Advance well past removal timeout with many ticks.
  for (double t = 1.0; t <= 20.0; t += 1.0) {
    wm.Tick(t);
  }

  auto stats = wm.GetStats();
  EXPECT_EQ(stats.total_entities, 0);  // Should be purged.
}

TEST_F(WorldModelTest, ResetClearsEverything) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 10.0, 0.9)}, 1.0);
  EXPECT_EQ(wm.GetStats().total_entities, 1);

  wm.Reset();
  EXPECT_EQ(wm.GetStats().total_entities, 0);
  EXPECT_EQ(wm.GetTick(), 0);
}

TEST_F(WorldModelTest, GetUncertainEntities) {
  WorldModel wm(config);
  // Single-source entity → needs corroboration.
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 10.0, 0.9)}, 1.0);

  auto uncertain = wm.GetUncertainEntities();
  EXPECT_EQ(uncertain.size(), 1);
  EXPECT_TRUE(uncertain[0]->needs_corroboration);

  // Add second source → no longer uncertain.
  wm.IngestObservations({MakeObs("o2", "a2", 10.2, 10.1, 0.85)}, 0.9);

  uncertain = wm.GetUncertainEntities();
  EXPECT_EQ(uncertain.size(), 0);
}

TEST_F(WorldModelTest, MaxEntitiesEnforced) {
  config.max_entities = 3;
  WorldModel wm(config);

  for (int i = 0; i < 5; ++i) {
    double x = i * 100.0;  // Far apart — each creates a new entity.
    wm.IngestObservations(
        {MakeObs("o" + std::to_string(i), "a1", x, 0.0, 0.9)}, 1.0);
  }

  EXPECT_LE(wm.GetStats().total_entities, 3);
}

TEST_F(WorldModelTest, GetEntityById) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 20.0, 0.9)}, 1.0);

  auto all = wm.GetEntities(0.0);
  ASSERT_EQ(all.size(), 1);

  auto* entity = wm.GetEntity(all[0]->entity_id);
  ASSERT_NE(entity, nullptr);
  EXPECT_NEAR(entity->position.x, 10.0, 0.5);

  // Non-existent ID.
  EXPECT_EQ(wm.GetEntity("nonexistent"), nullptr);
}

TEST_F(WorldModelTest, MultipleObservationsFromSameAgentDontDoubleCount) {
  WorldModel wm(config);
  wm.IngestObservations({MakeObs("o1", "a1", 10.0, 20.0, 0.9, 1.0)}, 1.0);
  wm.IngestObservations({MakeObs("o2", "a1", 10.1, 20.0, 0.88, 2.0)}, 1.0);

  auto all = wm.GetEntities(0.0);
  ASSERT_EQ(all.size(), 1);
  // Still only one unique source.
  EXPECT_EQ(all[0]->unique_source_count, 1);
  EXPECT_EQ(all[0]->state, EntityState::kTentative);
}

}  // namespace
}  // namespace omnicopilot

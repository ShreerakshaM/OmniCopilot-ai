// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "cpp/core/communication/budget_manager.h"
#include "cpp/core/communication/channel.h"
#include "cpp/core/communication/prioritizer.h"
#include "cpp/core/communication/serializer.h"
#include "cpp/core/world_model/entity.h"

namespace omnicopilot {
namespace {

// ─── Budget Manager Tests ───────────────────────────────────────────────────

TEST(BudgetManagerTest, NewTickResetsBudget) {
  BudgetConfig config;
  config.budget_per_tick_bytes = 4096;
  config.critical_reserve_fraction = 0.2;
  BudgetManager bm(config);

  uint32_t available = bm.NewTick();
  EXPECT_EQ(available, 4096);
  EXPECT_EQ(bm.GetRemaining(), 4096);
  EXPECT_NEAR(bm.GetUtilization(), 0.0, 0.01);
}

TEST(BudgetManagerTest, AllocateReducesBudget) {
  BudgetConfig config;
  config.budget_per_tick_bytes = 1000;
  config.critical_reserve_fraction = 0.2;  // 200 bytes reserved.
  BudgetManager bm(config);
  bm.NewTick();

  EXPECT_TRUE(bm.Allocate(500, false));
  EXPECT_EQ(bm.GetRemaining(), 500);
  EXPECT_NEAR(bm.GetUtilization(), 0.5, 0.01);
}

TEST(BudgetManagerTest, NonCriticalCannotUseCriticalReserve) {
  BudgetConfig config;
  config.budget_per_tick_bytes = 1000;
  config.critical_reserve_fraction = 0.2;  // 200 reserved.
  BudgetManager bm(config);
  bm.NewTick();

  // Use 800 of the 800 non-critical budget.
  EXPECT_TRUE(bm.Allocate(800, false));
  // Next 100 would dip into critical reserve — should fail.
  EXPECT_FALSE(bm.Allocate(100, false));
}

TEST(BudgetManagerTest, CriticalCanUseReserve) {
  BudgetConfig config;
  config.budget_per_tick_bytes = 1000;
  config.critical_reserve_fraction = 0.2;
  BudgetManager bm(config);
  bm.NewTick();

  EXPECT_TRUE(bm.Allocate(800, false));
  // Critical message can use the reserve.
  EXPECT_TRUE(bm.Allocate(100, true));
}

// ─── Serializer Tests ───────────────────────────────────────────────────────

TEST(SerializerTest, RoundTripObservation) {
  Serializer ser;

  Observation obs;
  obs.observation_id = "obs_42";
  obs.agent_id = "agent_7";
  obs.track_id = "track_3";
  obs.object_class = ObjectClass::kPedestrian;
  obs.sensor_type = 1;
  obs.position = {12.5, -3.7, 0.1};
  obs.velocity = {1.2, -0.5, 0.0};
  obs.confidence = 0.87;
  obs.timestamp_s = 42.123;
  obs.length = 0.5;
  obs.width = 0.4;
  obs.height = 1.7;
  obs.heading = 1.57;

  auto bytes = ser.SerializeObservation(obs);
  EXPECT_GT(bytes.size(), 0);

  auto decoded = ser.DeserializeObservation(bytes.data(), bytes.size());

  EXPECT_EQ(decoded.observation_id, "obs_42");
  EXPECT_EQ(decoded.agent_id, "agent_7");
  EXPECT_EQ(decoded.track_id, "track_3");
  EXPECT_EQ(decoded.object_class, ObjectClass::kPedestrian);
  EXPECT_EQ(decoded.sensor_type, 1);
  EXPECT_DOUBLE_EQ(decoded.position.x, 12.5);
  EXPECT_DOUBLE_EQ(decoded.position.y, -3.7);
  EXPECT_DOUBLE_EQ(decoded.position.z, 0.1);
  EXPECT_DOUBLE_EQ(decoded.velocity.vx, 1.2);
  EXPECT_DOUBLE_EQ(decoded.velocity.vy, -0.5);
  EXPECT_DOUBLE_EQ(decoded.confidence, 0.87);
  EXPECT_DOUBLE_EQ(decoded.timestamp_s, 42.123);
  EXPECT_DOUBLE_EQ(decoded.heading, 1.57);
}

TEST(SerializerTest, RoundTripBatch) {
  Serializer ser;

  std::vector<Observation> batch;
  for (int i = 0; i < 5; ++i) {
    Observation obs;
    obs.observation_id = "obs_" + std::to_string(i);
    obs.agent_id = "agent_0";
    obs.position = {static_cast<double>(i) * 10.0, 0.0, 0.0};
    obs.confidence = 0.5 + i * 0.1;
    obs.timestamp_s = 1.0;
    batch.push_back(obs);
  }

  auto bytes = ser.SerializeBatch(batch);
  auto decoded = ser.DeserializeBatch(bytes.data(), bytes.size());

  ASSERT_EQ(decoded.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(decoded[i].observation_id, "obs_" + std::to_string(i));
    EXPECT_DOUBLE_EQ(decoded[i].position.x, i * 10.0);
  }
}

TEST(SerializerTest, EstimateSize) {
  Serializer ser;
  Observation obs;
  obs.observation_id = "obs_1";
  obs.agent_id = "agent_1";
  obs.track_id = "track_1";

  uint32_t estimated = ser.EstimateSize(obs);
  auto actual_bytes = ser.SerializeObservation(obs);

  // Estimate should be in the right ballpark (within 2x).
  EXPECT_GT(estimated, 0);
  EXPECT_LT(estimated, actual_bytes.size() * 2 + 50);
}

// ─── Channel Tests ──────────────────────────────────────────────────────────

TEST(ChannelTest, BasicEnqueueAndDeliver) {
  ChannelConfig config;
  config.max_bandwidth_bps = 1e9;  // Effectively unlimited.
  config.base_latency_ms = 10.0;
  config.latency_jitter_ms = 0.0;  // No jitter for deterministic test.
  config.packet_loss_rate = 0.0;   // No loss.
  config.max_queue_depth = 100;

  Channel ch(config);

  ChannelMessage msg;
  msg.message_id = "m1";
  msg.sender_id = "a1";
  msg.receiver_id = "a2";
  msg.size_bytes = 256;
  msg.enqueue_time_s = 1.0;

  EXPECT_TRUE(ch.Enqueue(msg));

  // Too early — not delivered yet.
  auto delivered = ch.Tick(1.005);  // 5ms — latency is 10ms.
  EXPECT_TRUE(delivered.empty());

  // Now past latency.
  delivered = ch.Tick(1.02);  // 20ms.
  EXPECT_EQ(delivered.size(), 1);
  EXPECT_EQ(delivered[0].message_id, "m1");
}

TEST(ChannelTest, PacketLossDropsMessages) {
  ChannelConfig config;
  config.max_bandwidth_bps = 1e9;
  config.base_latency_ms = 0.0;
  config.packet_loss_rate = 1.0;  // 100% loss!
  config.max_queue_depth = 100;

  Channel ch(config);

  ChannelMessage msg;
  msg.message_id = "m1";
  msg.size_bytes = 100;
  msg.enqueue_time_s = 0.0;

  // Should be dropped (100% loss rate).
  EXPECT_FALSE(ch.Enqueue(msg));

  auto stats = ch.GetStats();
  EXPECT_EQ(stats.total_dropped_loss, 1);
}

TEST(ChannelTest, QueueOverflowDrops) {
  ChannelConfig config;
  config.max_bandwidth_bps = 1e9;
  config.base_latency_ms = 1000.0;  // 1 second latency — messages stay in queue.
  config.packet_loss_rate = 0.0;
  config.max_queue_depth = 2;

  Channel ch(config);

  for (int i = 0; i < 3; ++i) {
    ChannelMessage msg;
    msg.message_id = "m" + std::to_string(i);
    msg.size_bytes = 100;
    msg.enqueue_time_s = 0.0;
    ch.Enqueue(msg);
  }

  auto stats = ch.GetStats();
  EXPECT_EQ(stats.total_dropped_queue, 1);  // Third message dropped.
}

TEST(ChannelTest, ResetClearsState) {
  ChannelConfig config;
  config.max_bandwidth_bps = 1e9;
  config.base_latency_ms = 100.0;
  config.packet_loss_rate = 0.0;
  config.max_queue_depth = 100;

  Channel ch(config);

  ChannelMessage msg;
  msg.message_id = "m1";
  msg.size_bytes = 100;
  msg.enqueue_time_s = 0.0;
  ch.Enqueue(msg);

  ch.Reset();
  auto stats = ch.GetStats();
  EXPECT_EQ(stats.total_enqueued, 0);
  EXPECT_EQ(stats.total_delivered, 0);
}

TEST(ChannelTest, SetConditionsOverrides) {
  ChannelConfig config;
  config.max_bandwidth_bps = 1e9;
  config.base_latency_ms = 10.0;
  config.packet_loss_rate = 0.0;

  Channel ch(config);

  // Override to high latency.
  ch.SetConditions(0.0, 500.0, 1e9);

  auto state = ch.GetState();
  EXPECT_NEAR(state.current_latency_ms, 500.0, 0.01);
}

// ─── Prioritizer Tests ──────────────────────────────────────────────────────

TEST(PrioritizerTest, PedestrianScoredHigherThanTrafficSign) {
  PrioritizationWeights weights;
  Prioritizer pri(weights);

  Observation ped;
  ped.object_class = ObjectClass::kPedestrian;
  ped.confidence = 0.8;
  ped.velocity = {1.0, 0.0, 0.0};

  Observation sign;
  sign.object_class = ObjectClass::kTrafficSign;
  sign.confidence = 0.8;
  sign.velocity = {0.0, 0.0, 0.0};

  auto ped_scored = pri.Score(ped);
  auto sign_scored = pri.Score(sign);

  EXPECT_GT(ped_scored.total_score, sign_scored.total_score);
}

TEST(PrioritizerTest, BudgetLimitsSelection) {
  PrioritizationWeights weights;
  Prioritizer pri(weights);

  std::vector<Observation> observations;
  for (int i = 0; i < 10; ++i) {
    Observation obs;
    obs.observation_id = "obs_" + std::to_string(i);
    obs.object_class = ObjectClass::kVehicle;
    obs.confidence = 0.5 + i * 0.05;
    obs.velocity = {10.0, 0.0, 0.0};
    observations.push_back(obs);
  }

  // Budget for only 3 observations (3 × 256 = 768 bytes).
  auto result = pri.Prioritize(observations, 768, 256);

  EXPECT_EQ(result.selected.size(), 3);
  EXPECT_EQ(result.suppressed.size(), 7);
  EXPECT_GT(result.total_selected_value, 0.0);
  EXPECT_GT(result.total_suppressed_value, 0.0);
}

TEST(PrioritizerTest, EmptyObservations) {
  PrioritizationWeights weights;
  Prioritizer pri(weights);

  auto result = pri.Prioritize({}, 4096, 256);
  EXPECT_TRUE(result.selected.empty());
  EXPECT_TRUE(result.suppressed.empty());
}

}  // namespace
}  // namespace omnicopilot

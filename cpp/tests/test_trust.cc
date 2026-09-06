// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "cpp/core/trust/reliability.h"

namespace omnicopilot {
namespace {

class ReliabilityTest : public ::testing::Test {
 protected:
  ReliabilityConfig config;
  void SetUp() override {
    config.initial_trust = 0.7;
    config.trust_ema_alpha = 0.1;
    config.min_trust = 0.05;
    config.max_trust = 0.99;
    config.calibration_period = 10;
  }
};

TEST_F(ReliabilityTest, UnknownAgentGetsDefaultTrust) {
  ReliabilityTracker tracker(config);
  EXPECT_DOUBLE_EQ(tracker.GetTrust("unknown_agent"), config.initial_trust);
}

TEST_F(ReliabilityTest, RegisterAndGetTrust) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");
  EXPECT_DOUBLE_EQ(tracker.GetTrust("a1"), config.initial_trust);
}

TEST_F(ReliabilityTest, CorrectObservationsIncreaseTrust) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");

  double initial = tracker.GetTrust("a1");
  for (int i = 0; i < 20; ++i) {
    tracker.RecordCorrect("a1");
  }
  EXPECT_GT(tracker.GetTrust("a1"), initial);
}

TEST_F(ReliabilityTest, IncorrectObservationsDecreaseTrust) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");

  double initial = tracker.GetTrust("a1");
  for (int i = 0; i < 20; ++i) {
    tracker.RecordIncorrect("a1");
  }
  EXPECT_LT(tracker.GetTrust("a1"), initial);
}

TEST_F(ReliabilityTest, TrustNeverBelowMin) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");

  for (int i = 0; i < 100; ++i) {
    tracker.RecordIncorrect("a1");
  }
  EXPECT_GE(tracker.GetTrust("a1"), config.min_trust);
}

TEST_F(ReliabilityTest, TrustNeverAboveMax) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");

  for (int i = 0; i < 100; ++i) {
    tracker.RecordCorrect("a1");
  }
  EXPECT_LE(tracker.GetTrust("a1"), config.max_trust);
}

TEST_F(ReliabilityTest, IsSuspiciousOnlyAfterCalibration) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");

  // Before calibration period — should NOT be flagged.
  for (int i = 0; i < 5; ++i) {
    tracker.RecordIncorrect("a1");
  }
  EXPECT_FALSE(tracker.IsSuspicious("a1", 0.3));

  // After calibration period — should be flagged.
  for (int i = 0; i < 10; ++i) {
    tracker.RecordIncorrect("a1");
  }
  EXPECT_TRUE(tracker.IsSuspicious("a1", 0.3));
}

TEST_F(ReliabilityTest, GoodAgentNotSuspicious) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");

  for (int i = 0; i < 50; ++i) {
    tracker.RecordCorrect("a1");
  }
  EXPECT_FALSE(tracker.IsSuspicious("a1", 0.3));
}

TEST_F(ReliabilityTest, GetRecord) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");
  tracker.RecordCorrect("a1");
  tracker.RecordCorrect("a1");
  tracker.RecordIncorrect("a1");

  auto* record = tracker.GetRecord("a1");
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record->agent_id, "a1");
  EXPECT_EQ(record->total_observations, 3);
  EXPECT_EQ(record->verified_correct, 2);
  EXPECT_EQ(record->verified_incorrect, 1);
}

TEST_F(ReliabilityTest, GetRecordNonexistent) {
  ReliabilityTracker tracker(config);
  EXPECT_EQ(tracker.GetRecord("nonexistent"), nullptr);
}

TEST_F(ReliabilityTest, GetAllRecords) {
  ReliabilityTracker tracker(config);
  tracker.RegisterAgent("a1");
  tracker.RegisterAgent("a2");
  tracker.RegisterAgent("a3");

  auto records = tracker.GetAllRecords();
  EXPECT_EQ(records.size(), 3);
}

TEST_F(ReliabilityTest, AutoRegisterOnRecord) {
  ReliabilityTracker tracker(config);
  // Don't register — RecordCorrect should auto-register.
  tracker.RecordCorrect("a1");

  auto* record = tracker.GetRecord("a1");
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record->verified_correct, 1);
}

}  // namespace
}  // namespace omnicopilot
